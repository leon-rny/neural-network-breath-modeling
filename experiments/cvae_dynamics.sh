#!/bin/bash
start_time=$(date +%s)
for region in mouth nose; do
  for seed in 0 1 7 42 123; do
    python ablation_training_dynamics.py --config beta_cap_0.001      --region $region --seed $seed
    python ablation_training_dynamics.py --config beta_cap_0.01       --region $region --seed $seed
    python ablation_training_dynamics.py --config beta_cap_0.1        --region $region --seed $seed
    python ablation_training_dynamics.py --config beta_cap_0.5        --region $region --seed $seed
    python ablation_training_dynamics.py --config beta_cap_1.0        --region $region --seed $seed
    python ablation_training_dynamics.py --config lag_5_100           --region $region --seed $seed
    python ablation_training_dynamics.py --config lag_5_250           --region $region --seed $seed
    python ablation_training_dynamics.py --config lag_10_100          --region $region --seed $seed
    python ablation_training_dynamics.py --config lag_10_250          --region $region --seed $seed
    python ablation_training_dynamics.py --config lag_5_250_beta_0.1  --region $region --seed $seed
    python ablation_training_dynamics.py --config lag_5_250_beta_0.01 --region $region --seed $seed
  done
done
end_time=$(date +%s)
elapsed=$((end_time - start_time))
echo "Total elapsed time: $elapsed seconds"
