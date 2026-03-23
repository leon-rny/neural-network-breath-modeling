for region in mouth nose; do
  for augmentation_ratio in 0.1 0.25 0.5 2.0 4.0 10.0; do
    for seed in 0 1 7 42 123; do
        python -m core.tstr --model cvae --region $region --seed $seed --mode tstr_plus --augmentation_ratio $augmentation_ratio
    done
  done
done