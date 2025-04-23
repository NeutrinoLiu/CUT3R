#!/bin/bash

set -e

workdir='.'
model_name='ours'
if [ -z "$CKPT" ]; then
    echo "Error: CKPT environment variable is not set."
    exit 1
fi
ckpt_name="$CKPT"
model_weights="${workdir}/src/${ckpt_name}.pth"
datasets=('sintel' 'bonn' 'kitti')
time_stamp="$(date +%Y%m%d_%H%M%S)"

for data in "${datasets[@]}"; do
    output_dir="${workdir}/eval_results/video_depth/${data}_${model_name}_${ckpt_name}_${time_stamp}"
    echo "$output_dir"
    accelerate launch --num_processes 3  eval/video_depth/launch.py \
        --weights "$model_weights" \
        --output_dir "$output_dir" \
        --eval_dataset "$data" \
        --size 512
    python eval/video_depth/eval_depth.py \
    --output_dir "$output_dir" \
    --eval_dataset "$data" \
    --align "scale"
done
