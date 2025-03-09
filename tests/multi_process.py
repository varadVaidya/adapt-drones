import concurrent.futures
import numpy as np


def compute_with_seed(seed):
    np.random.seed(seed)
    result1 = np.random.random()
    result2 = np.random.random()
    return seed, result1, result2


def main():
    seeds = range(100)
    num_cores = 4

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_cores) as executor:
        results = list(executor.map(compute_with_seed, seeds))

    # Convert results to a NumPy array
    results_array = np.array(results)

    print("Final Results Array:")
    print(results_array)


if __name__ == "__main__":
    main()
