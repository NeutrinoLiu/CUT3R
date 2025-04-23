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
# datasets=('scannet' 'tum' 'sintel')
datasets=('tum' 'sintel')
time_stamp="$(date +%Y%m%d_%H%M%S)"

for data in "${datasets[@]}"; do
    output_dir="${workdir}/eval_results/relpose/${data}_${model_name}_${ckpt_name}_${time_stamp}"
    echo "$output_dir"
    accelerate launch --num_processes 3 --main_process_port 29558 eval/relpose/launch.py \
        --weights "$model_weights" \
        --output_dir "$output_dir" \
        --eval_dataset "$data" \
        --size 512
done


