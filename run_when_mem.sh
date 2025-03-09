#!/bin/bash

# Set the memory threshold in MB
MEMORY_THRESHOLD_MB=5000

# Python script to execute
PYTHON_SCRIPT="adapt_drones/train.py"

# Function to get available memory in MB
get_available_memory() {
    free -m | awk '/^Mem:/{print $7}'
}

echo "Monitoring available memory. Threshold: ${MEMORY_THRESHOLD_MB} MB"

# Loop until the memory threshold is met
while true; do
    available_memory=$(get_available_memory)
    echo "Available memory: ${available_memory} MB"

    if (( available_memory > MEMORY_THRESHOLD_MB )); then
        echo "Threshold met. Running ${PYTHON_SCRIPT} with arguments: $*"
        python3 "$PYTHON_SCRIPT" "$@"
        break
    fi

    # Wait for a few seconds before checking again
    sleep 5
done
