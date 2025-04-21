cd ../src
export CUDA_VISIBLE_DEVICES=0,1,2

# stride  heads   depth   lr      tag

# 2       12       4       1e-5    21245_8
# 2       12       4       1e-6    21246_8


NCCL_DEBUG=TRACE TORCH_DISTRIBUTED_DEBUG=DETAIL HYDRA_FULL_ERROR=1 accelerate launch --multi_gpu train.py \
    --config-name dpt_mf_headonly epochs=8 \
    pgs_stride=2 pgs_num_heads=12 pgs_depth=4 \
    lr=1e-5 min_lr=1e-6 exp_name=21245_8

NCCL_DEBUG=TRACE TORCH_DISTRIBUTED_DEBUG=DETAIL HYDRA_FULL_ERROR=1 accelerate launch --multi_gpu train.py \
    --config-name dpt_mf_headonly epochs=8 \
    pgs_stride=2 pgs_num_heads=12 pgs_depth=4 \
    lr=1e-6 min_lr=1e-7 exp_name=21246_8


