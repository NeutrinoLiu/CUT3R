# export CUDA_VISIBLE_DEVICES=0
# python train.py \
#     --config-name dpt_air_headonly epochs=8 \
#     pgs_stride=2 pgs_num_heads=12 pgs_depth=4 \
#     lr=1e-6 min_lr=1e-7 exp_name="air_2124_1e-6"

# export CUDA_VISIBLE_DEVICES=1
# python train.py \
#     --config-name dpt_air_headonly epochs=8 \
#     pgs_stride=4 pgs_num_heads=12 pgs_depth=4 \
#     lr=1e-6 min_lr=1e-7 exp_name="air_4124_1e-6"

# export CUDA_VISIBLE_DEVICES=2
# python train.py \
#     --config-name dpt_air_vanilla epochs=8 \
#     lr=1e-6 min_lr=1e-7 exp_name="air_vanilla_1e-6"

# run each of the above commands in seperate tmux, with naming the same as the cuda device


# --- Start Training on GPU 0 ---
tmux new -s cuda0 -d
tmux send-keys -t cuda0 'export CUDA_VISIBLE_DEVICES=0' C-m
tmux send-keys -t cuda0 'python train.py --config-name dpt_air_headonly epochs=8 pgs_stride=2 pgs_num_heads=12 pgs_depth=4 lr=2e-6 min_lr=2e-7 exp_name="air_2124_2e-6"' C-m

# --- Start Training on GPU 1 ---
tmux new -s cuda1 -d
tmux send-keys -t cuda1 'export CUDA_VISIBLE_DEVICES=1' C-m
tmux send-keys -t cuda1 'python train.py --config-name dpt_air_headonly epochs=8 pgs_stride=4 pgs_num_heads=12 pgs_depth=4 lr=2e-6 min_lr=2e-7 exp_name="air_4124_2e-6"' C-m

# --- Start Training on GPU 2 ---
tmux new -s cuda2 -d
tmux send-keys -t cuda2 'export CUDA_VISIBLE_DEVICES=2' C-m
tmux send-keys -t cuda2 'python train.py --config-name dpt_air_vanilla epochs=8 lr=2e-6 min_lr=2e-7 exp_name="air_vanilla_2e-6"' C-m

echo "Launched training jobs in tmux sessions: cuda0, cuda1, cuda2"
echo "Use 'tmux ls' to list sessions."
echo "Use 'tmux attach -t <session_name>' (e.g., 'tmux attach -t cuda0') to view a specific job."