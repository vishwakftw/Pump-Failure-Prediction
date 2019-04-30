# Pump Failure Prediction

#### What is the goal of the problem?

- You have 10 power plants pumps. For these 10 pumps, you are given the number of failures (x<sub>i</sub>) and the number of hours in thousands that the pumps have run for (t<sub>i</sub>).

- We seek to model the relationship between x<sub>i</sub> and t<sub>i</sub> using a Gamma-Poisson Hierarchical model as done by George _et al_ in their paper Conjugate Likelihood Distributions in 1993.

#### What is the goal of the assignment?

- We will be solving this problem using Metropolis-Hastings algorithm - a popular variant of the Monte-Carlo Markov Chain (MCMC) algorithm, which will be implemented natively using "standard" libraries.

- Our implementation will then be compared against an implementation in PyMC3, a popular probabilistic programming framework.

- The sampling algorithm will be compared against an variational inference algorithm which is optimization based. This has been implemented using PyMC3 as well.

#### Further details

- Further details of the problem and comparison of implementations are presented in the [report](report.pdf).

#### Running the code

- Please install the requirements specified in [requirements.txt](requirements.txt). To use the PyMC3 implementations, you will need to install PyMC3.
- `python <script> --help` will print a set of configurable options while running the code where `<script>` is one of `[numpy_mcmc.py, pymc3_mcmc.py, pymc3_vi.py]`.
- The dataset for the assignment is [here](dataset.txt)

#### References

- OpenBUGS description: <http://www.openbugs.net/Examples/Pumps.html>
- George E.I., Makov U.E., Smith A.F.M., _Conjugate Likelihood Distributions_, Scandinavian Journal of Statistics, 1993.

-------

###### Done as part of coursework pertaining to CS5350 : Bayesian Data Analysis offered in Spring 2019
