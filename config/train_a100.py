# A configuration sized for training Qwen2.5-3B-Instruct on a 40GB A100.
# Tested on an 8xA100 (40GB) node.
#
# launch as the following (e.g. in a screen session) and wait ~1 days:
# $ torchrun --standalone --nproc_per_node=8 train.py config/train_a100.py

wandb_log = True
wandb_run_name='r1-qwen2p5-3B-instruct'

checkpoint_path = 'Qwen/Qwen2.5-3B-Instruct'

_num_devices = 8

device_rollout_batch_size = 128
episodes_per_rollout = device_rollout_batch_size * _num_devices
group_size = 16
policy_epochs = 1
policy_update_batch_size = 4 * _num_devices

max_new_tokens = 512 + 256
