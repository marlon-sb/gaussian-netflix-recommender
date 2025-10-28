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


def run_em_test(X_incomplete, X_gold):
    """
    Runs the EM algorithm on the test dataset and compares
    the final log-likelihood to the gold standard.
    """
    print("\n--- Starting EM Test ---")

    K = 4
    seed = 0
    gold_ll = None

    # Parse test_solutions.txt to find the final LL
    try:
        with open('test_solutions.txt', 'r') as f:
            for line in f:
                # Find lines that start with "LL:"
                if line.strip().startswith("LL:"):
                    # Overwrite gold_ll, so we get the *last* one
                    gold_ll = float(line.split(':')[1].strip())

        if gold_ll is None:
            print("  ERROR: Could not find 'LL:' in test_solutions.txt")
            return
    except FileNotFoundError:
        print("  ERROR: 'test_solutions.txt' not found.")
        return
    except Exception as e:
        print(f"  ERROR: Failed to parse test_solutions.txt: {e}")
        return

    # Run our EM algorithm
    mixture, post = common.init(X_incomplete, K, seed)
    mixture, post, ll = em.run(X_incomplete, mixture, post)

    print(f"  [EM Test] K={K}, Seed={seed}")
    print(f"  Your Log-Likelihood: {ll:.12f}")
    print(f"  Gold Log-Likelihood: {gold_ll:.12f}")

    if np.isclose(ll, gold_ll):
        print("\n  SUCCESS: Log-likelihood matches test solution.")
    else:
        print("\n  ERROR: Log-likelihood does not match.")


def run_netflix_experiment(X_incomplete, X_gold):
    """
    Runs the EM algorithm on the Netflix dataset for K=1 and K=12.
    Reports the best log-likelihoods and the final RMSE.
    """
    print("\n--- Starting Final Netflix Experiment ---")

    K_values = [1, 12]
    seeds = [0, 1, 2, 3, 4]
    best_results = {}

    for K in K_values:
        print(f"\n--- Testing K={K} ---")
        best_ll_for_K = -float('inf')
        best_mixture_for_K = None

        for seed in seeds:
            print(f"  Running seed {seed}...")
            mixture, post = common.init(X_incomplete, K, seed)
            mixture, post, ll = em.run(X_incomplete, mixture, post)
            print(f"    Log-Likelihood: {ll:.4f}")

            if ll > best_ll_for_K:
                best_ll_for_K = ll
                best_mixture_for_K = mixture

        print(f"  Best Log-Likelihood for K={K}: {best_ll_for_K:.4f}")
        best_results[K] = (best_ll_for_K, best_mixture_for_K)

    print("\nFinal Results")
    ll_k1 = best_results[1][0]
    ll_k12 = best_results[12][0]

    print(f"Log-likelihood|K=1 = {ll_k1:.4f}")
    print(f"Log-likelihood|K=12 = {ll_k12:.4f}")

    print("\nCalculating RMSE")
    best_mixture_k12 = best_results[12][1]

    X_pred = em.fill_matrix(X_incomplete, best_mixture_k12)

    mask = (X_gold != 0)
    rmse_val = common.rmse(X_gold[mask], X_pred[mask])

    print(f"RMSE = {rmse_val:.4f}")


if __name__ == "__main__":

    try:
        X_incomplete = np.loadtxt('netflix_incomplete.txt')
        X_gold = np.loadtxt('netflix_complete.txt')
        print("Loaded Netflix data files.")
    except FileNotFoundError:
        print("Error: Netflix data file not found.")
        exit()


    run_netflix_experiment(X_incomplete, X_gold)
