start_time=$(date +%s)
for region in mouth; do
  for seed in 0 1 7 42 123; do
    python -m core.train --model cvae_part --region $region --seed $seed --part_embed_dim 16
    python -m core.tstr  --model cvae_part --region $region --seed $seed --part_embed_dim 16
  done
done
end_time=$(date +%s)
elapsed=$((end_time - start_time))
echo "Total elapsed time: $elapsed seconds"