# for region in mouth nose; do
#   for ld in 8 16 32; do
#     for seed in 0 1 7 42 123; do
#       python -m core.train --model cvae --region $region --seed $seed --latent_dim $ld
#       python -m core.tstr --model cvae --region $region --seed $seed --latent_dim $ld
#     done
#   done
# done

start_time=$(date +%s)
for region in mouth; do
  for seed in 0 1 7 42 123; do
    python -m core.train --model cvae --region $region --seed $seed
    python -m core.tstr --model cvae --region $region --seed $seed
  done
done
end_time=$(date +%s)
elapsed=$((end_time - start_time))
echo "Total elapsed time: $elapsed seconds"