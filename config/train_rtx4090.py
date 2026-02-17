# A configuration sized for training Qwen2.5-1.5B-Instruct on an RTX 4090 (24Gb mem).
# Tested on a 2x4090 node.
#
# launch as the following (e.g. in a screen session) and wait ~1 days:
# $ torchrun --standalone --nproc_per_node=2 train.py config/train_rtx4090.py

wandb_log = True
wandb_run_name='r1-qwen2p5-1p5B-instruct'

checkpoint_path = 'Qwen/Qwen2.5-1.5B-Instruct'

device_rollout_batch_size = 256
group_size = 16
episodes_per_rollout = device_rollout_batch_size * 2

policy_epochs = 1

max_new_tokens = 512

policy_update_batch_size = 8
