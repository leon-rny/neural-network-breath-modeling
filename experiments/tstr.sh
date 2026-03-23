for region in mouth nose; do
  for seed in 0 1 7 42 123; do
    python -m core.tstr --model trtr --region $region --seed $seed --force_rebuild
  done
done