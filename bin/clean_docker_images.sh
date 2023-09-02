#!/bin/bash

# Path to your docker-compose file
compose_file="docker-compose.yml"

# Define image names array
image_names=("feyntune-inference-gpu" "feyntune-train-gpu" "feyntune-inference-cpu" "feyntune-train-cpu")

# Loop over image names
for image_name in "${image_names[@]}"; do
  if [ "$image_name" != "null" ]; then
    # Get container IDs that are using the image
    container_ids=$(docker ps -a -q --filter ancestor=$image_name)

    # If there are any containers using the image, stop and remove them
    if [ ! -z "$container_ids" ]; then
      docker stop $container_ids
      docker rm $container_ids
    fi

    # Delete all but the most recent version of the image
    docker image ls -q --filter "before=$(docker image ls -q $image_name | head -n 1)" $image_name | xargs -r docker image rm
  fi
done
