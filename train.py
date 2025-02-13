"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python train.py --policy_update_batch_size=32

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""

import gc
import os
import random
import re
import sys
import time
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import torch
import torch._dynamo
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoModelForCausalLM, AutoTokenizer

import logicpuzzles

print(f"Python: {sys.version}, PyTorch: {torch.__version__}")

# Reinforcement Learning Terminology Overview:
# Episode: A single interaction sequence consisting of a prompt, model response, and resulting reward
# Rollout: A batch of episodes collected using the current policy
# Policy: The model's strategy for generating responses, represented by the neural network parameters and generation config
# Group: In Group-Relative Policy Optimization (GRPO), a set of episodes sharing the same prompt
# Reference Model: A frozen copy of the policy model used to prevent the policy from deviating too far from its previous behavior

# -----------------------------------------------------------------------------
# I/O
out_dir = 'out'
save_interval = 128  # save a checkpoint every N rollouts
# wandb logging
wandb_log = False  # disabled by default
wandb_project = 'r1'
wandb_run_name = 'r1-' + str(int(time.time()))
# dataset
tasks = "all"  # eg "countdown", or "countdown,nim"
# training
checkpoint_path = 'Qwen/Qwen2.5-1.5B-Instruct'
total_rollouts = 1024
reference_model_sync_interval = -1  # reset the reference model every N rollouts (diabled by default)
## rollout stage
episodes_per_rollout = 512  # total number of episodes to collect per rollout
device_rollout_batch_size = 256  # number of completions to generate, in parallel, per device (trades off throughput for memory)
group_size = 32  # number of completions per prompt (increasing leads to more fine-grained advantages)
temperature = 0.7  # sampling temperature for response generation
max_new_tokens = 512  # maximum length of generated responses
## learning stage
policy_epochs = 1  # number of training epochs to run using the current rollout batch
policy_update_batch_size = 16  # training batch size
clip_epsilon = 0.2  # clipping parameter for the policy ratio
adaptive_kl_penalty = False  # whether or not to adaptively adjust beta_kld
beta_kld = 0.04  # initial KL penalty coefficient (provides regularization)
# optimizer (adam)
learning_rate = 5e-6
weight_decay = 0.0
beta1 = 0.9
beta2 = 0.95
grad_clip = 0.5
# learning rate decay
decay_lr = True
warmup_iters = 1000
lr_decay_iters = 30000
min_lr = learning_rate / 1000
# DDP settings
backend = 'nccl'  # 'nccl', 'gloo', etc.
# system
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = False  # use PyTorch 2.0 compilation
# -----------------------------------------------------------------------------
# config overrides
config_keys = [k for k, v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read())  # overrides from command line or config file
config = {k: globals()[k] for k in config_keys}  # will be useful for logging

# -----------------------------------------------------------------------------
# various inits, derived attributes, I/O setup
ddp = int(os.environ.get('RANK', -1)) != -1  # is this a ddp run?
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0  # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank  # each process gets a different seed

    # world_size number of processes will be training simultaneously, so we
    # can scale down episodes_per_rollout by splitting it across processes
    assert episodes_per_rollout % ddp_world_size == 0
    device_episodes_per_rollout = episodes_per_rollout // ddp_world_size
    assert policy_update_batch_size % ddp_world_size == 0
    device_policy_update_batch_size = policy_update_batch_size // ddp_world_size
    assert device_episodes_per_rollout % device_rollout_batch_size == 0
    assert device_rollout_batch_size % group_size == 0
    device_prompts_per_batch = device_rollout_batch_size // group_size
else:
    # if not ddp, we are running on a single gpu in one process
    master_process = True
    seed_offset = 0
    ddp_world_size = 1
    device_episodes_per_rollout = episodes_per_rollout
    device_policy_update_batch_size = policy_update_batch_size
    assert device_rollout_batch_size % group_size == 0
    device_prompts_per_batch = device_rollout_batch_size // group_size

assert device_rollout_batch_size > 0
assert group_size > 1
assert temperature > 0
assert device_episodes_per_rollout % device_rollout_batch_size == 0
# assert device_episodes_per_rollout % device_policy_update_batch_size == 0

def print0(*args, **kwargs):
    if master_process:
        print(*args, **kwargs)

# calculate derived values
episode_generation_iters = device_episodes_per_rollout // device_rollout_batch_size
updates_per_epoch = device_episodes_per_rollout // device_policy_update_batch_size
total_device_mem = torch.cuda.get_device_properties(device).total_memory / 1024**3

print0(f'{device_episodes_per_rollout=}, {device_rollout_batch_size=}, {device_prompts_per_batch=}, {group_size=}, {episode_generation_iters=}')
print0(f'{policy_update_batch_size=}, {device_policy_update_batch_size=}, {updates_per_epoch=}')

if master_process:
    os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(5492 + seed_offset)
random.seed(5492 + seed_offset)
np.random.seed(5492 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# -----------------------------------------------------------------------------
# model, optimizer, dataset init

# model init
# unlike in nanoGPT, we always want to start from a pretrained model
print0("initializing model")
tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, padding_side='left')
model = AutoModelForCausalLM.from_pretrained(checkpoint_path, torch_dtype=torch.bfloat16, attn_implementation='sdpa', device_map=None)
model.to(device)
model.train()
model.gradient_checkpointing_enable()

# reference model init
def update_reference_model(model):
    ref_model = deepcopy(model)
    for param in ref_model.parameters():  # ref model is fully frozen
        param.requires_grad = False
    ref_model.eval()
    ref_model.cpu()  # keep out of GPU memory when not in use
    return ref_model

ref_model = update_reference_model(model)

# compile the model
if compile:
    print0('compiling the model... (takes a ~minute)')
    torch._dynamo.config.optimize_ddp = False  # DDPOptimizer does not support higher order op in graph
    model.generation_config.cache_implementation = 'static'  # generation-time kv cache compatible with torch.compile
    model.forward = torch.compile(model.forward, dynamic=True)  # requires PyTorch 2.0

# wrap model into DDP container
if ddp:
    # note: reference model not wrapped b/c we never need to sync it accross devices - its static
    model = DDP(model, device_ids=[ddp_local_rank], static_graph=True)

# dataset
dataset = logicpuzzles.gen_dataset(tasks)

# initialize a GradScaler. If enabled=False scaler is a no-op
grad_scaling_enabled = (dtype == 'float16')
if grad_scaling_enabled:
    print0("Grad scaling is enabled for float16 training")
scaler = torch.amp.GradScaler(device=device_type, enabled=grad_scaling_enabled)

# optimizer
optimizer = torch.optim.AdamW(model.parameters(), learning_rate, (beta1, beta2), weight_decay=weight_decay, fused=True)

# learning rate decay scheduler using LambdaLR
def get_lr(it):
    # linear warmup for warmup_iters steps
    if it < warmup_iters:
        return (it + 1) / (warmup_iters + 1)
    # if it > lr_decay_iters, return min learning rate ratio
    if it > lr_decay_iters:
        return min_lr / learning_rate
    # in between, use linear decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    return 1.0 - decay_ratio * (1.0 - min_lr/learning_rate)

lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, get_lr if decay_lr else (lambda _: 1))

# logging
if wandb_log:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config, mode=None if master_process else "disabled")

# -----------------------------------------------------------------------------
# helper fns and classes

@torch.no_grad()
def generate_completions(model, tokenizer, prompts, group_size, max_new_tokens, temperature):
    if isinstance(prompts, str):
        prompts = [prompts]
    prompt_inputs = tokenizer(prompts, return_tensors='pt', padding=True).to(model.device)
    prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]

    # disable gradient checkpointing and set model to eval mode for generation
    model.gradient_checkpointing_disable()
    was_training = model.training
    model.eval()
    with ctx:
        full_ids = model.generate(
            prompt_ids,
            attention_mask=prompt_mask,
            num_return_sequences=group_size,
            max_new_tokens=max_new_tokens,
            use_cache=True,  # use kv cache to greatly speed up generation
            do_sample=True,
            temperature=temperature,
        )
    model.train(was_training)
    model.gradient_checkpointing_enable()

    prompt_length = prompt_ids.size(1)
    prompt_mask = prompt_mask.repeat_interleave(group_size, dim=0)
    return full_ids, prompt_length, prompt_mask

@torch.compile(dynamic=True)
def selective_log_softmax(logits, index):
    logprobs = logits.log_softmax(dim=-1)
    return torch.gather(logprobs, dim=-1, index=index.unsqueeze(-1)).squeeze(-1)

def get_per_token_logprobs(model, input_ids, attention_mask, logits_to_keep, batch_size=512):
    per_token_logprobs = []
    with ctx:
        for batch_input_ids, batch_attention_mask in zip(input_ids.split(batch_size), attention_mask.split(batch_size)):
            logits = model(  # num_logits_to_keep changes to logits_to_keep in transformers 4.49
                input_ids=batch_input_ids, attention_mask=batch_attention_mask, num_logits_to_keep=logits_to_keep + 1
            ).logits
            logits = logits[:, :-1, :]  # exclude logits for the next token prediction
            logits = logits[:, -logits_to_keep:]
            batch_input_ids = batch_input_ids[:, -logits_to_keep:]
            per_token_logprobs.append(selective_log_softmax(logits, batch_input_ids))
        return torch.cat(per_token_logprobs, dim=0)

def pad_and_stack(tensors: list, pad_value: int):
    max_len = max(x.size(1) for x in tensors)
    padded = [torch.nn.functional.pad(x, (0, max_len - x.size(1)), value=pad_value) for x in tensors]
    return torch.cat(padded)

def masked_mean(tensor, mask):
    masked_sum = (tensor * mask).sum()
    return masked_sum / (mask.sum() + 1e-8)

def free_up_memory():
    gc.collect()
    torch.cuda.empty_cache()

@dataclass
class Episode:
    prompt_ids: torch.LongTensor
    prompt_mask: torch.IntTensor
    completion_ids: torch.LongTensor
    completion_mask: torch.IntTensor
    advantage: torch.FloatTensor
    ref_logprobs: torch.FloatTensor
    old_logprobs: torch.FloatTensor

class AdaptiveKLController:
    # https://github.com/openai/lm-human-preferences/blob/cbfd210bb8b08f6bc5c26878c10984b90f516c66/lm_human_preferences/train_policy.py#L115
    def __init__(self, init_kl_coef, target_kl=0.5, horizon=100):
        self.value = init_kl_coef
        self.target_kl = target_kl
        self.horizon = horizon

    def update(self, current_kl, n_steps):
        # reduce kl penalty if kl is below target, increase if kl is above target
        proportional_error = np.clip(current_kl / self.target_kl - 1, -0.2, 0.2)
        mult = 1 + proportional_error * n_steps / self.horizon
        self.value *= mult

class ConstantKLController:
    def __init__(self, init_kl_coef):
        self.value = init_kl_coef

    def update(self, *args, **kwargs):
        pass

kl_ctl = AdaptiveKLController(beta_kld, horizon=policy_update_batch_size * 5) if adaptive_kl_penalty else ConstantKLController(beta_kld)

# -----------------------------------------------------------------------------
# reward function definitions
def accuracy_reward_fn(texts, eval_fn):
    accuracy_rewards = []
    for text in texts:
        answers = re.findall(r'<answer>\s*(.*?)\s*</answer>', text)
        reward = 1.0 if answers and eval_fn(answers[-1]) else 0.0  # only check the final answer if any exist
        accuracy_rewards.append(reward)
    return torch.tensor(accuracy_rewards, device=device)

def format_reward_fn(texts):
    format_rewards = []
    for text in texts:
        has_correct_pattern = bool(re.search(r'<think>.*?</think>\s*<answer>.*?</answer>', text, re.DOTALL))
        reward = 1.0 if has_correct_pattern else 0.0

        think_pairs = len(re.findall(r'<think>.*?</think>', text, re.DOTALL))
        answer_pairs = len(re.findall(r'<answer>.*?</answer>', text, re.DOTALL))
        if think_pairs > 1:
            reward -= 0.2
        if answer_pairs > 1:
            reward -= 0.2
        format_rewards.append(reward)

    return torch.tensor(format_rewards, device=device)

# -----------------------------------------------------------------------------
# main training loop
raw_model = model.module if ddp else model  # Unwrap DDP container if needed
iter_num = 0
print0("training start!")
for rollout in range(total_rollouts):
    # save a checkpoint
    if rollout % save_interval == 0 and rollout > 0 and master_process:
        torch.cuda.synchronize()
        checkpoint = {
            'model': raw_model.state_dict(),
            'optimizer': optimizer.state_dict(),
            # todo: save kl_div manager
            'rollout': rollout,
            'config': config,
        }
        print0(f'saving checkpoint to {out_dir}')
        torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))

    # reset reference model periodically
    if rollout % reference_model_sync_interval == 0 and reference_model_sync_interval > 0 and rollout > 0:
        print0('updating reference model')
        del ref_model
        free_up_memory()
        ref_model = update_reference_model(raw_model)

    # --------------------------------------------------------------------------------
    # Rollout Phase: using the current "policy" (model) gather a batch of "episodes"
    # (prompt, completion, reward) from the "environment" for training.
    # --------------------------------------------------------------------------------
    torch.cuda.reset_peak_memory_stats(device)
    with torch.no_grad(), ctx:
        t0 = time.time()
        episodes: list[Episode] = []
        rewards, format_rewards, acc_rewards, query_lengths, response_lengths = [], [], [], [], []
        for _ in range(episode_generation_iters):
            messages_batch, eval_fn_batch = zip(*[next(dataset) for _ in range(device_prompts_per_batch)])
            prompts = [tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True) for msgs in messages_batch]

            # generation is fairly expensive compared to everything else. in an ideal setup, generation
            # would be performed on an inference-optimized server (eg vllm, sglang), and, after each "learning"
            # phase, weights would be communicated to that server and updated. that server could then
            # continue generating while the learning phase begins. here we're optimizing for simplicity,
            # so we just compute them in the same process. every generation can be used for
            # multiple updates if policy_epochs>1, in order to improve sample efficiency.
            full_ids, prompt_length, prompt_mask = generate_completions(raw_model, tokenizer, prompts, group_size, max_new_tokens, temperature)
            prompt_ids = full_ids[:, :prompt_length]
            completion_ids = full_ids[:, prompt_length:]
            full_text = tokenizer.batch_decode(full_ids, skip_special_tokens=True)
            completion_text = tokenizer.batch_decode(completion_ids, skip_special_tokens=True)

            # mask everything after the first EOS token
            is_eos = completion_ids == tokenizer.eos_token_id
            eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
            eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
            sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
            completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

            query_lengths.extend(prompt_mask.sum(dim=1).tolist())
            response_lengths.extend(completion_mask.sum(dim=1).tolist())
            attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
            logits_to_keep = completion_mask.size(1)

            # compute log probabilities - static during learning phase
            ref_model.to(device)
            ref_logprob = get_per_token_logprobs(ref_model, full_ids, attention_mask, logits_to_keep=logits_to_keep, batch_size=8)  # process in small batches to avoid OOM
            ref_model.cpu()
            old_logprob = get_per_token_logprobs(model, full_ids, attention_mask, logits_to_keep=logits_to_keep, batch_size=8)

            # compute rewards and group-relative advantages
            for k, eval_fn in enumerate(eval_fn_batch):
                start = k * group_size
                end = start + group_size
                group_text = completion_text[start:end]

                group_acc_rewards = accuracy_reward_fn(group_text, eval_fn)
                group_fmt_rewards = format_reward_fn(group_text)
                group_rewards = torch.stack([group_acc_rewards, group_fmt_rewards]).sum(dim=0)
                r_std, r_mean = torch.std_mean(group_rewards)
                group_advantages = (group_rewards - r_mean) / (r_std + 1e-4)

                episodes.extend([
                    Episode(
                        prompt_ids=prompt_ids[i : i + 1],
                        prompt_mask=prompt_mask[i : i + 1],
                        completion_ids=completion_ids[i : i + 1],
                        completion_mask=completion_mask[i : i + 1],
                        advantage=group_advantages[i - start],
                        ref_logprobs=ref_logprob[i : i + 1],
                        old_logprobs=old_logprob[i : i + 1],
                    )
                    for i in range(start, end)
                ])
                rewards.extend(group_rewards.tolist())
                acc_rewards.extend(group_acc_rewards.tolist())
                format_rewards.extend(group_fmt_rewards.tolist())

        # logging
        reward_mean, acc_reward_mean, format_reward_mean = np.mean(rewards), np.mean(acc_rewards), np.mean(format_rewards)
        reward_std, std_acc_reward, std_format_reward = np.std(rewards), np.std(acc_rewards), np.std(format_rewards)
        queries_len_mean, queries_len_std, queries_len_max = np.mean(query_lengths), np.std(query_lengths), np.max(query_lengths)
        responses_len_mean, responses_len_std, responses_len_max = np.mean(response_lengths), np.std(response_lengths), np.max(response_lengths)
        dt = time.time() - t0
        max_mem = torch.cuda.max_memory_allocated(device) / 1024**3
        print0(
            f'rollout:{rollout}/{total_rollouts} reward_mean:{reward_mean:.3f} acc_reward_mean:{acc_reward_mean:.3f} '
            f'format_reward_mean:{format_reward_mean:.3f} generation_time:{dt:.2f}s gpu_mem:{max_mem:.1f}/{total_device_mem:.1f}GB'
        )
        if wandb_log:
            wandb.log(
                {
                    'rollout': rollout,
                    'env/reward_mean': reward_mean,
                    'env/acc_reward_mean': acc_reward_mean,
                    'env/format_reward_mean': format_reward_mean,
                    'env/reward_std': reward_std,
                    'tokens/queries_len_mean': queries_len_mean,
                    'tokens/queries_len_std': queries_len_std,
                    'tokens/queries_len_max': queries_len_max,
                    'tokens/responses_len_mean': responses_len_mean,
                    'tokens/responses_len_std': responses_len_std,
                    'tokens/responses_len_max': responses_len_max,
                }
            )

        # display example
        if rollout % 5 == 0:
            print0("=== Example Generation ===")
            print0(random.choice(full_text).strip())
            print0("========================")

    # clean up temporary variables that are using gpu memory
    del full_ids, ref_logprob, old_logprob, attention_mask, prompt_ids, prompt_mask  # type: ignore
    del completion_ids, completion_mask, group_advantages, is_eos, eos_idx, sequence_indices   # type: ignore
    free_up_memory()

    # -----------------------------------------------------------------------------------
    # Learning Phase: run multiple GRPO epochs on the data gathered in the latest rollout
    # -----------------------------------------------------------------------------------
    for epoch in range(policy_epochs):
        random.shuffle(episodes)
        for batch_start in range(0, len(episodes), device_policy_update_batch_size):
            torch.cuda.reset_peak_memory_stats(device)
            t0 = time.time()
            batch_end = min(batch_start + device_policy_update_batch_size, len(episodes))
            batch = episodes[batch_start:batch_end]

            # form batch tensors from individual episodes
            batch_prompt_ids = pad_and_stack([t.prompt_ids for t in batch], pad_value=tokenizer.pad_token_id)
            batch_prompt_mask = pad_and_stack([t.prompt_mask for t in batch], pad_value=0)
            batch_completion_ids = pad_and_stack([t.completion_ids for t in batch], pad_value=tokenizer.pad_token_id)
            batch_completion_mask = pad_and_stack([t.completion_mask for t in batch], pad_value=0)

            batch_ref_logprobs = pad_and_stack([t.ref_logprobs for t in batch], pad_value=1)  # log(1) = 0
            batch_old_logprobs = pad_and_stack([t.old_logprobs for t in batch], pad_value=1)
            batch_advantages = torch.stack([t.advantage for t in batch]).unsqueeze(1)

            batch_ids = torch.cat([batch_prompt_ids, batch_completion_ids], dim=1)
            batch_attention_mask = torch.cat([batch_prompt_mask, batch_completion_mask], dim=1)
            logits_to_keep = batch_completion_ids.size(1)

            # compute loss
            with ctx:
                # compute new log probs and policy ratios
                new_logprobs = get_per_token_logprobs(model, batch_ids, attention_mask=batch_attention_mask, logits_to_keep=logits_to_keep, batch_size=2)

                # compute ppo-style surrogate losses using group-relative advantages
                policy_ratio = torch.exp(new_logprobs - batch_old_logprobs)
                surrogate1 = policy_ratio * batch_advantages
                surrogate2 = torch.clamp(policy_ratio, 1 - clip_epsilon, 1 + clip_epsilon) * batch_advantages
                grpo_loss = -torch.min(surrogate1, surrogate2)

                # apply per-token KL divergence penalties; http://joschu.net/blog/kl-approx.html
                kld_ref = torch.exp(batch_ref_logprobs - new_logprobs) - (batch_ref_logprobs - new_logprobs) - 1
                grpo_loss = grpo_loss + kl_ctl.value * kld_ref

                # mask and mean the loss, account for completion length so completions are weighted equally
                grpo_loss = (grpo_loss * batch_completion_mask).sum(dim=1) / batch_completion_mask.sum(dim=1)
                grpo_loss = grpo_loss.mean()

            # backward pass, with gradient scaling if training in fp16
            scaler.scale(grpo_loss).backward()

            # clip gradients
            scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            # step the optimizer (and scaler if training w/ fp16)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)  # flush gradients to free memory

            with torch.no_grad():
                # step schedulers
                lr = lr_scheduler.get_lr()[0]
                lr_scheduler.step()
                kl_coef = kl_ctl.value
                kl_ctl.update(masked_mean(kld_ref, batch_completion_mask).item(), policy_update_batch_size)  # adaptively update kl penalty

                # compute diagnostics
                kld_old = torch.exp(batch_old_logprobs - new_logprobs) - (batch_old_logprobs - new_logprobs) - 1
                # TODO: double check these masked means
                kl_old = masked_mean(kld_old, batch_completion_mask).item()
                kl_ref = masked_mean(kld_ref, batch_completion_mask).item()

                valid_policy_ratio = policy_ratio[batch_completion_mask.bool()]
                clipfrac = (((valid_policy_ratio < 1 - clip_epsilon) | (valid_policy_ratio > 1 + clip_epsilon)).float().mean())

                loss = grpo_loss.item()
                norm = norm.item()
                clipfrac = clipfrac.item()
                ratio_mean = valid_policy_ratio.mean().item()
                ratio_min = valid_policy_ratio.min().item()
                ratio_max = valid_policy_ratio.max().item()

                # logging
                iter_num += 1
                dt = time.time() - t0

                max_mem = torch.cuda.max_memory_allocated(device) / 1024**3
                print0(
                    f'  iter_num:{iter_num} epoch:{epoch}/{policy_epochs} episodes:{batch_start}/{len(episodes)} loss:{loss:.8f} '
                    f'norm:{norm:.4f} kl_ref:{kl_ref:.4f} kl_old:{kl_old:.4f} clipfrac:{clipfrac:.3f} lr:{lr:.2e} step_time:{dt * 1000:.2f}ms '
                    f'gpu_mem:{max_mem:.1f}/{total_device_mem:.1f}GB'
                )
                if wandb_log:
                    wandb.log(
                        {
                            'iter': iter_num,
                            'learning/learning_rate': lr,
                            'learning/step_time_ms': dt * 1000,
                            'learning/loss': loss,
                            'learning/grad_norm': norm,
                            'learning/kl_coef': kl_coef,
                            'learning/policy/kl_ref': kl_ref,
                            'learning/policy/kl_old': kl_old,
                            'learning/policy/clipfrac': clipfrac,
                            'learning/policy/policy_ratio_mean': ratio_mean,
                            'learning/policy/policy_ratio_min': ratio_min,
                            'learning/policy/policy_ratio_max': ratio_max,
                        }
                    )

    # clean up temporary variables that are using gpu memory
    del episodes, batch_old_logprobs, new_logprobs, batch_completion_mask, batch_prompt_mask
    del policy_ratio, valid_policy_ratio, kld_old, kld_ref
    del batch_prompt_ids, batch_completion_ids, batch_ids, batch_attention_mask, batch_ref_logprobs, batch_advantages
    del surrogate1, surrogate2, grpo_loss
    free_up_memory()

if ddp:
    destroy_process_group()
