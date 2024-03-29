import numpy as np
import pandas as pd

from tqdm import tqdm
from scipy.special import loggamma
from scipy.stats import norm as normal_dist
from scipy.stats import gamma as gamma_dist
from scipy.stats import poisson as poi_dist
from scipy.stats import uniform as uni_dist

from argparse import ArgumentParser

p = ArgumentParser()
p.add_argument('--print_summary', action='store_true', help='Toggle to print summary')
p.add_argument('--iterations', type=int, default=15000, help='Number of iterations for MCMC')
p.add_argument('--burn_in', type=int, default=5000, help='Number of iterations for burn in')
args = p.parse_args()

P = 10

# Get the dataset
# Format:
# 10 rows of <T_{i}, X_{i}> pairs
dataset = np.genfromtxt('dataset.txt', delimiter=' ')

def log_posterior_for_alpha(alpha, beta, thetas):
    """
    Un-normalized log-posterior for alpha
    """
    if alpha <= 0 or beta <= 0 or any(thetas <= 0):
        return -np.inf
    first_term = P * (alpha * np.log(beta) - loggamma(alpha))
    second_term = alpha * sum(np.log(thetas))
    third_term = -alpha
    return first_term + second_term + third_term

def log_posterior_for_beta(beta, alpha, thetas):
    """
    Un-normalized log-posterior for beta
    """
    if alpha <= 0 or beta <= 0 or any(thetas <= 0):
        return -np.inf
    first_term = P * alpha * np.log(beta)
    second_term = -beta * sum(thetas)
    third_term = -0.9 * np.log(beta) + -beta
    return first_term + second_term + third_term

def log_posterior_for_theta(thetas, idx, dataset, alpha, beta):
    """
    Un-normalized log-posterior for thetas
    """
    if alpha <= 0 or beta <= 0 or thetas[idx] <= 0:
        return -np.inf
    t_i, x_i = dataset[idx]
    first_term = (x_i + alpha - 1) * np.log(thetas[idx])
    second_term = - thetas[idx] * (t_i + beta)
    return first_term + second_term


# Proposal dist and Acceptance probability dist
proposal_dist = normal_dist(scale=np.ones((2 + P,)))
uniform_dist = uni_dist(loc=np.zeros(2 + P), scale=np.ones(2 + P))

def mcmc_loop(alpha_init, beta_init, thetas_init, burn_in, iterations):
    """
    MCMC loop function
    """
    # MCMC sampling stats
    samples_alpha = [alpha_init]
    samples_beta = [beta_init]
    samples_thetas = [[theta_init] for theta_init in thetas_init]

    alpha = alpha_init
    beta = beta_init
    thetas = thetas_init

    for i in tqdm(range(burn_in + iterations)):
        # Sample deltas
        deltas = proposal_dist.rvs()

        # Sample acceptance values
        acceptance_vals = uniform_dist.rvs()

        # Get new alphas, betas and thetas
        new_alpha = alpha + deltas[0]
        new_beta = beta + deltas[1]
        new_thetas = thetas + deltas[2:]

        # Accept or reject alpha
        accept_alpha = False
        if np.log(acceptance_vals[0]) < log_posterior_for_alpha(new_alpha, beta, thetas) - \
                                        log_posterior_for_alpha(alpha, beta, thetas):
            accept_alpha = True
            if i >= args.burn_in:
                samples_alpha.append(new_alpha)

        # Accept or reject beta
        accept_beta = False
        if np.log(acceptance_vals[1]) < log_posterior_for_beta(new_beta, alpha, thetas) - \
                                        log_posterior_for_beta(beta, alpha, thetas):
            accept_beta = True
            if i >= args.burn_in:
                samples_beta.append(new_beta)

        # Accept or reject thetas
        accept_thetas = [False] * P
        for idx in range(P):
            if np.log(acceptance_vals[2 + idx]) < log_posterior_for_theta(new_thetas, idx, dataset, alpha, beta) - \
                                                  log_posterior_for_theta(thetas, idx, dataset, alpha, beta):
                accept_thetas[idx] = True
                if i >= args.burn_in:
                    samples_thetas[idx].append(new_thetas[idx])

        # Now perform swapping
        if accept_alpha:
            alpha = new_alpha
        if accept_beta:
            beta = new_beta
        for idx in range(P):
            if accept_thetas[idx]:
                thetas[idx] = new_thetas[idx]

    return (samples_alpha, samples_beta, samples_thetas)

samples = mcmc_loop(0.1, 0.2, np.array([0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75]),
                    burn_in=args.burn_in, iterations=args.iterations)

def format_sample_info(samples):
    info_dict = {}
    # Alpha info
    info_dict['alpha'] = {}
    info_dict['alpha']['mean'] = np.mean(samples[0])
    info_dict['alpha']['std'] = np.std(samples[0])

    # Beta info
    info_dict['beta'] = {}
    info_dict['beta']['mean'] = np.mean(samples[1])
    info_dict['beta']['std'] = np.std(samples[1])

    # Theta info
    for idx in range(P):
        info_dict['theta{}'.format(idx)] = {}
        info_dict['theta{}'.format(idx)]['mean'] = np.mean(samples[2][idx])
        info_dict['theta{}'.format(idx)]['std'] = np.std(samples[2][idx])

    return info_dict

full_dict = format_sample_info(samples)
print(pd.DataFrame.from_dict(full_dict, orient='index', columns=['mean', 'std']))

# Prediction
predictions = []
for idx in range(P):
    # Get theta (mean that is)
    pred_i = 0.0
    for sample in samples[2][idx]:
        lambda_idx = dataset[idx, 0] * sample
        pred_i += round(poi_dist(lambda_idx).rvs(1000).mean())
    predictions.append(round(pred_i / len(samples[2][idx])))

print(predictions)
