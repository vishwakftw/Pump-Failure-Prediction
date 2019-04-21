import pymc3
import numpy as np

from argparse import ArgumentParser

p = ArgumentParser()
p.add_argument('--iterations', type=int, default=20000,
                               help='Number of iterations for VI')
args = p.parse_args()

# Get the dataset
# Format:
# 10 rows of <T_{i}, X_{i}> pairs
dataset = np.genfromtxt('dataset.txt', delimiter=' ')

# Create the model
pumps_vi_model = pymc3.Model()
with pumps_vi_model:
    alpha = pymc3.Exponential('alpha', 1.0)
    beta = pymc3.Gamma('beta', 0.1, 1.0)
    for i in range(dataset.shape[0]):
        theta = pymc3.Gamma('theta{}'.format(i), alpha, beta)
        lambd = pymc3.Deterministic('lambda{}'.format(i), theta * dataset[i, 0])
        x = pymc3.Poisson('x{}'.format(i), lambd, observed=dataset[i, 1])

# Run variational inference to obtain the approximate posterior
with pumps_vi_model:
    approx_post = pymc3.fit(n=args.iterations)

# Perform prediction
    variables = approx_post.sample()
    lambdas = [round(np.mean(variables['lambda{}'.format(i)])) for i in range(dataset.shape[0])]
    # Each prediction is distributed in a Poisson manner, hence the mean of the observations
    # is equal to the Poisson parameter in the limiting case, which happens to be lambdas
    predictions = lambdas.copy()
    print(predictions)
