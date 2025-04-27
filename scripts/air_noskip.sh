export CUDA_VISIBLE_DEVICES=0,1,2
NCCL_DEBUG=TRACE TORCH_DISTRIBUTED_DEBUG=DETAIL HYDRA_FULL_ERROR=1 accelerate launch --multi_gpu train.py \
    --config-name dpt_air_headonly epochs=8 \
    pgs_stride=4 pgs_num_heads=12 pgs_depth=4 \
    lr=1e-6 min_lr=1e-7 exp_name="air_4124_1e-6_noskip" \
    long_context=False num_views=16 num_test_views=16 batch_size=6

NCCL_DEBUG=TRACE TORCH_DISTRIBUTED_DEBUG=DETAIL HYDRA_FULL_ERROR=1 accelerate launch --multi_gpu train.py \
    --config-name dpt_air_headonly epochs=8 \
    pgs_stride=2 pgs_num_heads=12 pgs_depth=4 \
    lr=1e-6 min_lr=1e-7 exp_name="air_2124_1e-6_noskip" \
    long_context=False num_views=16 num_test_views=16 batch_size=6

NCCL_DEBUG=TRACE TORCH_DISTRIBUTED_DEBUG=DETAIL HYDRA_FULL_ERROR=1 accelerate launch --multi_gpu train.py \
    --config-name dpt_air_vanilla epochs=8 \
    lr=1e-6 min_lr=1e-7 exp_name="air_vanilla_1e-6_noskip" \
    long_context=False num_views=16 num_test_views=16 batch_size=2 pre_eval=True

NCCL_DEBUG=TRACE TORCH_DISTRIBUTED_DEBUG=DETAIL HYDRA_FULL_ERROR=1 accelerate launch --multi_gpu train.py \
    --config-name dpt_air_headonly epochs=8 \
    pgs_stride=1 pgs_revisit_gap=2 pgs_num_heads=12 pgs_depth=4 \
    lr=1e-6 min_lr=1e-7 exp_name="air_2r124_1e-6_noskip" \
    long_context=False num_views=16 num_test_views=16 batch_size=6

NCCL_DEBUG=TRACE TORCH_DISTRIBUTED_DEBUG=DETAIL HYDRA_FULL_ERROR=1 accelerate launch --multi_gpu train.py \
    --config-name dpt_air_headonly epochs=8 \
    pgs_stride=1 pgs_revisit_gap=2 pgs_num_heads=12 pgs_depth=4 \
    lr=1e-6 min_lr=1e-7 exp_name="air_2r124_1e-6_continue" \
    long_context=False num_views=16 num_test_views=16 batch_size=1 freeze=encoder pretrained=./2r124.pth pre_eval=True

NCCL_DEBUG=TRACE TORCH_DISTRIBUTED_DEBUG=DETAIL HYDRA_FULL_ERROR=1 accelerate launch --multi_gpu train.py \
    --config-name dpt_multi_headonly epochs=8 \
    pgs_stride=1 pgs_revisit_gap=2 pgs_num_heads=12 pgs_depth=4 \
    lr=1e-6 min_lr=1e-7 exp_name="multi_2r124_1e-6_noskip" \
    long_context=False num_views=16 num_test_views=16 batch_size=6