cd ../src
export CUDA_VISIBLE_DEVICES=0,1,2

# stride  heads   depth   lr      tag

# 2       8       2       1e-6    2826
# 2       8       4       1e-6    2846
# 2       12      2       1e-6    21226
# 2       12       4       1e-6    21246

# 2       8       2       1e-7    2827
# 2       8       4       1e-7    2847
# 2       12      2       1e-7    21227
# 2       12       4       1e-7    21247

# NCCL_DEBUG=TRACE TORCH_DISTRIBUTED_DEBUG=DETAIL HYDRA_FULL_ERROR=1 accelerate launch --multi_gpu train.py \
#     --config-name dpt_mf_headonly epochs=1 \
#     pgs_stride=2 pgs_num_heads=8 pgs_depth=2 \
#     lr=1e-6 min_lr=1e-7 exp_name=2826

# NCCL_DEBUG=TRACE TORCH_DISTRIBUTED_DEBUG=DETAIL HYDRA_FULL_ERROR=1 accelerate launch --multi_gpu train.py \
#     --config-name dpt_mf_headonly epochs=1 \
#     pgs_stride=2 pgs_num_heads=8 pgs_depth=4 \
#     lr=1e-6 min_lr=1e-7 exp_name=2846

NCCL_DEBUG=TRACE TORCH_DISTRIBUTED_DEBUG=DETAIL HYDRA_FULL_ERROR=1 accelerate launch --multi_gpu train.py \
    --config-name dpt_mf_headonly epochs=1 \
    pgs_stride=2 pgs_num_heads=8 pgs_depth=2 \
    lr=1e-7 min_lr=1e-8 exp_name=2827

NCCL_DEBUG=TRACE TORCH_DISTRIBUTED_DEBUG=DETAIL HYDRA_FULL_ERROR=1 accelerate launch --multi_gpu train.py \
    --config-name dpt_mf_headonly epochs=1 \
    pgs_stride=2 pgs_num_heads=8 pgs_depth=4 \
    lr=1e-7 min_lr=1e-8 exp_name=2847

NCCL_DEBUG=TRACE TORCH_DISTRIBUTED_DEBUG=DETAIL HYDRA_FULL_ERROR=1 accelerate launch --multi_gpu train.py \
    --config-name dpt_mf_headonly epochs=1 \
    pgs_stride=2 pgs_num_heads=12 pgs_depth=2 \
    lr=1e-6 min_lr=1e-7 exp_name=21226

NCCL_DEBUG=TRACE TORCH_DISTRIBUTED_DEBUG=DETAIL HYDRA_FULL_ERROR=1 accelerate launch --multi_gpu train.py \
    --config-name dpt_mf_headonly epochs=1 \
    pgs_stride=2 pgs_num_heads=12 pgs_depth=4 \
    lr=1e-6 min_lr=1e-7 exp_name=21246

NCCL_DEBUG=TRACE TORCH_DISTRIBUTED_DEBUG=DETAIL HYDRA_FULL_ERROR=1 accelerate launch --multi_gpu train.py \
    --config-name dpt_mf_headonly epochs=1 \
    pgs_stride=2 pgs_num_heads=12 pgs_depth=2 \
    lr=1e-7 min_lr=1e-8 exp_name=21227

NCCL_DEBUG=TRACE TORCH_DISTRIBUTED_DEBUG=DETAIL HYDRA_FULL_ERROR=1 accelerate launch --multi_gpu train.py \
    --config-name dpt_mf_headonly epochs=1 \
    pgs_stride=2 pgs_num_heads=12 pgs_depth=4 \
    lr=1e-7 min_lr=1e-8 exp_name=21247