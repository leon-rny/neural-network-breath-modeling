for region in mouth nose; do
  for fb in 0.0 0.1 2.0; do
    for seed in 0 1 7 42 123; do
      python -m core.train --model vae --region $region --seed $seed --free_bits $fb
      python -m core.tstr --model vae --region $region --seed $seed --free_bits $fb
    done
  done
done