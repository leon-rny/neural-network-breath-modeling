for region in mouth nose; do
  for seed in 0 1 7 42 123; do
    python -m core.train --model cvae --region $region --seed $seed
    python -m core.tstr --model cvae --region $region --seed $seed
  done
done