import pymc3
import numpy as np

from argparse import ArgumentParser

p = ArgumentParser()
p.add_argument('--proposal_dist', type=str, default='PoissonProposal',
                                  help='Specify a different proposal distribution, one of \
                                        UniformProposal, NormalProposal \
                                        or PoissonProposal (default)')
p.add_argument('--print_summary', action='store_true', help='Toggle to print summary of trace')
args = p.parse_args()

# Get the dataset
# Format:
# 10 rows of <T_{i}, X_{i}> pairs
dataset = np.genfromtxt('dataset.txt', delimiter=' ')

# Create the model
pumps_mcmc_model = pymc3.Model()
with pumps_mcmc_model:
    alpha = pymc3.Exponential('alpha', 1.0)
    beta = pymc3.Gamma('beta', 0.1, 1.0)
    for i in range(dataset.shape[0]):
        theta = pymc3.Gamma('theta{}'.format(i), alpha, beta)
        lambd = pymc3.Deterministic('lambda{}'.format(i), theta * dataset[i, 0])
        x = pymc3.Poisson('x{}'.format(i), lambd, observed=dataset[i, 1])

# Perform Metropolis-Hastings algorithm step
# and print the trace of variables
with pumps_mcmc_model:
    step = pymc3.Metropolis(proposal_dist=getattr(pymc3.step_methods.metropolis,
                                                  args.proposal_dist))
    trace = pymc3.sample(10000, step=step)
    if args.print_summary:
        print(pymc3.summary(trace))

# Perform prediction
with pumps_mcmc_model:
    xs = pymc3.sample_posterior_predictive(trace, samples=10000)
    predictions = [round(np.mean(xs['x{}'.format(i)])) for i in range(dataset.shape[0])]
    print(predictions)
