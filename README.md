# CS5350-BDA
Coursework pertaining to CS5350 : Bayesian Data Analysis offered in Spring 2019

#### What is the goal of the problem?

- You have 10 power plants pumps. For these 10 pumps, you are given the number of failures (x<sub>i</sub>) and the number of hours in thousands that the pumps have run for (t<sub>i</sub>).

- We seek to model the relationship between xi and ti using a Gamma-Poisson Hierarchical model as done by George et al in their paper Conjugate Likelihood Distributions in 1993.

#### What is the goal of the assignment?

- We will be solving this problem using Metropolis-Hastings algorithm - a popular variant of the Monte-Carlo Markov Chain (MCMC) algorithm, which will be implemented natively using "standard" libraries.

- Our implementation will then be compared against an implementation in PyMC3, a popular probabilistic programming framework.

- The sampling algorithm will be compared against an variational inference algorithm which is optimization based. This has been implemented using PyMC3 as well.

#### Further details

- Further details of the problem and comparison of implementations are presented in the report.
