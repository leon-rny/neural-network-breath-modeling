#!/bin/bash
start_time=$(date +%s)
for region in mouth nose; do
  for seed in 0 1 7 42 123; do
    python ablation_architectures.py --variant conv_baseline --region $region --seed $seed
    python ablation_architectures.py --variant conv_slim      --region $region --seed $seed
    python ablation_architectures.py --variant mlp            --region $region --seed $seed
    python ablation_architectures.py --variant mlp_small      --region $region --seed $seed
    python ablation_architectures.py --variant conv_asym      --region $region --seed $seed
  done
done
end_time=$(date +%s)
elapsed=$((end_time - start_time))
echo "Total elapsed time: $elapsed seconds"
