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
time_stamp="$(date +%Y%m%d_%H%M%S)"

output_dir="${workdir}/eval_results/mv_recon/${model_name}_${ckpt_name}_${time_stamp}"
echo "$output_dir"
accelerate launch --num_processes 3 --main_process_port 29501 eval/mv_recon/launch.py \
    --weights "$model_weights" \
    --output_dir "$output_dir" \
    --model_name "$model_name"