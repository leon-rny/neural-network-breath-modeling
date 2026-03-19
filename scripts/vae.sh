for region in mouth nose; do
  for seed in 0 1 7 42 123; do
    # vanilla
    python -m core.train --model vae --region $region --seed $seed --free_bits 0.0
    python -m core.tstr --model vae --region $region --seed $seed --free_bits 0.0

    # somewhat healthy
    python -m core.train --model vae --region $region --seed $seed --free_bits 0.1
    python -m core.tstr --model vae --region $region --seed $seed --free_bits 0.1

    # over-constrained
    python -m core.train --model vae --region $region --seed $seed --free_bits 2.0
    python -m core.tstr --model vae --region $region --seed $seed --free_bits 2.0
  done
done