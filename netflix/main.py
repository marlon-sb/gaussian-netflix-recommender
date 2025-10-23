import numpy as np
import kmeans
import common
import naive_em
import em


def run_kmeans_experiment():
    """
    Runs the K-means experiment for K=[1, 2, 3, 4] and seeds=[0, 1, 2, 3, 4],
    then plots the best run for each K and reports the lowest cost.
    """

    try:
        X = np.loadtxt('toy_data.txt')
        print(f"Loaded 'toy_data.txt' with shape {X.shape}")
    except FileNotFoundError:
        print("Error: 'toy_data.txt' not found.")
        return

    K_values = [1, 2, 3, 4]
    seeds = [0, 1, 2, 3, 4]

    best_results = {}

    for K in K_values:
        print(f"\nTesting K={K}")

        best_cost_for_K = float('inf')
        best_mixture_for_K = None
        best_post_for_K = None
        best_seed_for_K = -1

        for seed in seeds:
            # Initialize and run K-means
            mixture, post = common.init(X, K, seed)
            mixture, post, cost = kmeans.run(X, mixture, post)

            print(f"  Seed {seed}: Cost = {cost:.4f}")

            # Check for best run
            if cost < best_cost_for_K:
                best_cost_for_K = cost
                best_mixture_for_K = mixture
                best_post_for_K = post
                best_seed_for_K = seed

        print(f"  Best cost for K={K} is {best_cost_for_K:.4f} (from seed {best_seed_for_K})")

        best_results[K] = {
            'cost': best_cost_for_K,
            'mixture': best_mixture_for_K,
            'post': best_post_for_K,
            'seed': best_seed_for_K
        }

    # Plotting and Reporting
    print("Plotting the best run for each K...")

    for K, result in best_results.items():
        title = f"K-Means (K={K}) | Seed={result['seed']} | Cost={result['cost']:.2f}"
        common.plot(X, result['mixture'], result['post'], title)
        print(f"Cost|K={K} = {result['cost']:.4f}")



if __name__ == "__main__":
    run_kmeans_experiment()
