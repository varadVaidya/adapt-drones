import numpy as np

# Load the file
data = np.load("/home/shyam/Desktop/IISc/adapt-drones/data/snowy-lake-170_dataset_aligned.npz")

# Show all keys
print("Keys in file:", list(data.keys()))
print()

# Print first 20 rows of each array
for key in data.files:
    arr = data[key]
    print(f"Key: {key}")
    print(f"Shape: {arr.shape}, Dtype: {arr.dtype}")
    
    # Print first 20 rows, or full array if smaller
    if arr.ndim == 0:
        # Scalar value
        print(arr)
    elif arr.ndim == 1:
        print(arr[:5])
    elif arr.ndim >= 2:
        print(arr[:5])  # Will print first 5 rows
    print()
