#!/bin/bash

# Configuration
URL="http://localhost:8000/RoadSegmentationQ2"
IMAGE_PATH="samples/deep_globe.jpg"
NUM_REQUESTS=5
SLEEP_SECONDS=5

# Function to make a single request
make_request() {
    curl -X POST \
        -F "image=@${IMAGE_PATH}" \
        "${URL}" \
        --output "output/output_${1}.tif" \
        &  # Run in background
}

while true; do
    echo -e "\n🚀 Sending ${NUM_REQUESTS} concurrent requests..."

    # Launch multiple requests in parallel
    for i in $(seq 1 ${NUM_REQUESTS}); do
        make_request $i
        echo "Request $i sent"
    done

    # Wait for all background processes to complete
    wait
    echo "All requests completed"

    echo -e "💤 Sleeping for ${SLEEP_SECONDS} seconds...\n"
    sleep ${SLEEP_SECONDS}
done
