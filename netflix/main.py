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


def run_em_experiment(X):
    """
    Runs the Naive EM algorithm on the toy dataset for K=1, 2, 3, 4,
    and calculates the BIC for each.
    """
    print("\n--- Starting Naive EM Experiment (BIC) ---")

    K_values = [1, 2, 3, 4]
    seeds = [0, 1, 2, 3, 4]

    best_bic_score = -float('inf')
    best_K = -1
    bic_scores = []

    for K in K_values:
        print(f"\n--- Testing K={K} ---")

        best_ll_for_K = -float('inf')
        best_mixture_for_K = None

        for seed in seeds:
            mixture, post = common.init(X, K, seed)
            mixture, post, ll = naive_em.run(X, mixture, post)

            if ll > best_ll_for_K:
                best_ll_for_K = ll
                best_mixture_for_K = mixture

        bic_score = common.bic(X, best_mixture_for_K, best_ll_for_K)
        bic_scores.append((K, bic_score, best_ll_for_K))

        print(f"  Best Log-Likelihood: {best_ll_for_K:.4f}")
        print(f"  BIC Score: {bic_score:.4f}")

        if bic_score > best_bic_score:
            best_bic_score = bic_score
            best_K = K

    print("\n--- BIC Results ---")
    for K, bic_val, ll_val in bic_scores:
        print(f"K={K}: LL = {ll_val:.4f}, BIC = {bic_val:.4f}")

    print(f"\nBest model: K={best_K} (Highest BIC)")


if __name__ == "__main__":
    X_toy = np.loadtxt('toy_data.txt')
    run_em_experiment(X_toy)
