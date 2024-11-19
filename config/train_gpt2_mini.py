# config for training GPT-2 (124M) down to very nice loss of ~2.85 on 1 node of 8X A100 40GB
# launch as the following (e.g. in a screen session) and wait ~5 days:
# $ torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py

wandb_log = True
wandb_project = 'owt'
wandb_run_name='gpt2-mini-6l-5h-emb768'

# these make the total batch size be ~0.5M
# 8 batch size * 512 block size * 5 gradaccum * 4 GPUs = 491,520
batch_size = 8
block_size = 512
gradient_accumulation_steps = 25 * 4

# this makes total number of tokens be 300B
max_iters = 10000
lr_decay_iters = 10000

n_layer = 6
n_head = 6
n_embd = 768
dropout = 0.2

# eval stuff
eval_interval = 1000
eval_iters = 200
log_interval = 10

learning_rate = 1e-4 # with baby networks can afford to go a bit higher
max_iters = 5000
lr_decay_iters = 5000 # make equal to max_iters usually
min_lr = 1e-4 # learning_rate / 10 usually

# weight decay
weight_decay = 1e-1
