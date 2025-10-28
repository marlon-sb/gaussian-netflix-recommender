# Collaborative Filtering via Gaussian Mixtures

## Table of Contents
- [Project Overview](#project-overview)
- [Algorithms & Models](#algorithms--models)
  - [1. K-Means (Baseline)](#1-k-means-baseline)
  - [2. Naive EM Algorithm](#2-naive-em-algorithm)
  - [3. Bayesian Information Criterion (BIC)](#3-bayesian-information-criterion-bic)
  - [4. EM for Matrix Completion](#4-em-for-matrix-completion)
  - [5. Matrix Filling (Prediction)](#5-matrix-filling-prediction)
- [Dataset](#dataset)
  - [Toy Dataset](#toy-dataset)
  - [Test Dataset](#test-dataset)
  - [Netflix Dataset](#netflix-dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Final Results](#final-results)
  - [Log-Likelihood](#log-likelihood)
  - [Root Mean Squared Error (RMSE)](#root-mean-squared-error-rmse)

***

## Project Overview

This project implements a collaborative filtering system to predict missing movie ratings from a subset of the Netflix dataset.

The core assumption is that users' rating profiles (their vector of movie ratings) are samples from a **Gaussian Mixture Model (GMM)**. The model learns to cluster users into $K$ different "types" and then predicts missing ratings based on the average ratings of the cluster a user most likely belongs to.

The project is structured in several parts:

- **K-Means Baseline**: A baseline clustering algorithm is run on a simple 2D dataset.
- **Naive EM Algorithm**: A full GMM is implemented and trained with the Expectation-Maximization (EM) algorithm on the complete 2D dataset.
- **Bayesian Information Criterion (BIC)**: The BIC is implemented to find the optimal number of clusters ($K$).
- **EM for Matrix Completion**: The EM algorithm is extended to handle the real Netflix dataset, which is mostly composed of missing (unobserved) ratings.
- **Matrix Filling & Evaluation**: The final trained GMM is used to predict all missing ratings, and the model's accuracy is measured using Root Mean Squared Error (RMSE).

***

## Algorithms & Models

### 1. K-Means (Baseline)

Found in `kmeans.py`, this is a standard K-Means implementation used as a baseline to find optimal clusters on the `toy_data.txt`.

### 2. Naive EM Algorithm

Found in `naive_em.py`, this is a complete implementation of the **Expectation-Maximization (EM)** algorithm for a **Gaussian Mixture Model**. It is "naive" because it assumes it has all the data and is only used on the 2D `toy_data.txt`.

### 3. Bayesian Information Criterion (BIC)

Implemented in `common.py`, the `bic` function is used for model selection. It calculates a score for a trained GMM that rewards high log-likelihood but penalizes model complexity (a high number of parameters), helping to select the best $K$.

### 4. EM for Matrix Completion

Found in `em.py`, this is the core of the project. It's an advanced implementation of the EM algorithm that is specifically designed to work with **incomplete data** (i.e., the Netflix matrix where 0s represent missing ratings).

**Key features include:**

- **E-Step**: Correctly calculates the log-likelihood and posteriors by marginalizing over (ignoring) the missing ratings.
- **M-Step**: Updates the model parameters (means, variances, and priors) using the posterior probabilities, while correctly handling missing data. A minimum variance is enforced to prevent cluster collapse.
- **Log-space operations**: All critical calculations are performed in the log-domain (using `logsumexp`) to prevent numerical underflow.

### 5. Matrix Filling (Prediction)

The `fill_matrix` function in `em.py` uses the final, trained GMM to predict all missing ratings. For each user:

1. Calculate the posterior probability $P(j \mid x_{\text{obs}})$.
2. Compute the predicted rating for a missing movie as the expected value over all cluster means:

$$
\mathbb{E}[x_{\text{miss}}] = \sum_{j=1}^K P(j \mid x_{\text{obs}}) \cdot \mu_{j, \text{miss}}
$$

***
## Dataset

### Toy Dataset

- **`toy_data.txt`**: A simple 2D dataset used to develop and test the `kmeans.py` and `naive_em.py` implementations.

### Test Dataset

- **`test_incomplete.txt` / `test_solutions.txt`**: A small, complex dataset used to verify the correctness of the `em.py` implementation against known log-likelihoods.

### Netflix Dataset

- **`netflix_incomplete.txt`**: The primary dataset for matrix completion. This is a large, sparse user-movie matrix $(n \times d)$ where most entries are `0`, indicating missing values.
- **`netflix_complete.txt`**: The ground truth matrix. This file contains the actual ratings and is used only to compute the final RMSE.

***

## Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/marlon-sb/gaussian-netflix-recommender.git
cd gaussian-netflix-recommender
```
### Step 2: Create and Activate a Virtual Environment

```bash
# Create the virtual environment
python -m venv venv

# Activate (on macOS/Linux)
source venv/bin/activate

# Activate (on Windows)
venv\Scripts\activate
```
### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

## Usage

The `main.py` script can run various components of the pipeline depending on configuration.

To run a specific component, edit the bottom of `main.py`:

```python
if __name__ == "__main__":
    # Choose your experiment here
```

By default, the script runs the final Netflix matrix completion experiment, which may take a few minutes.

## Final Results

The final experiment was run using the `netflix_incomplete.txt` dataset.

The model was trained using 5 different seeds:

```python
seeds = [0, 1, 2, 3, 4]
```

with the following values of `K`:
* `K = 1`
* `K = 12`

### Log-Likelihood

| K | Best Log-Likelihood |
|---|---|
| 1 | -1,521,060.9540 |
| 12 | -1,390,234.4223 |

### Root Mean Squared Error (RMSE)

The best model at `K = 12` was evaluated by comparing predicted ratings against the ground truth in `netflix_complete.txt`.

```text
Final RMSE: 0.4589
```
