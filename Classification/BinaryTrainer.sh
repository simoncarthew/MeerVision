#!/bin/bash

# Define parameters for batch sizes and image sizes
batch_sizes=(6 4 2)
image_sizes=(64)

# Function to execute the BinaryTrainer script with retry logic
run_binary_trainer() {
  local batch=$1
  local img_size=$2
  local max_retries=5
  local attempt=1

  # Loop to retry the command up to max_retries times
  while [ $attempt -le $max_retries ]; do
    echo "Running BinaryTrainer with batch: $batch, image_size: $img_size (Attempt $attempt)"
    python Classification/BinaryTrainer.py --batch $batch --image_size $img_size --dont_ask

    # Check if the command succeeded
    if [ $? -eq 0 ]; then
      echo "Execution successful for batch: $batch, image_size: $img_size"
      return 0
    else
      echo "Execution failed for batch: $batch, image_size: $img_size (Attempt $attempt)"
      ((attempt++))
    fi
  done

  # If we reach here, all attempts have failed
  echo "Giving up on batch: $batch, image_size: $img_size after $max_retries attempts."
  return 1
}

# Loop through batch sizes and image sizes
for img_size in "${image_sizes[@]}"; do
  for batch in "${batch_sizes[@]}"; do
    run_binary_trainer $batch $img_size
  done
done

# After training, merge the results
echo "Merging results..."
python Classification/MergeResults.py --parent_dir Classification/Results/Binary --merged_dir Classification/Results/Binary --delete_originals
