# juila implementations of Bayesian Vector Autoregressive (BVAR) framework
 Toolkit julia source codes and example outputs for Bayesian (Structural) Vector Autoregressive (VAR) models. There are identification strategies in multivariabe time series analysis that requires bayesian framework, such as, 
 * Dynamic probabilistic forecasting estimations
 * Stochastic volatility
 * Time-varying parameters
 * 'Big-data' or large dimension models
 * Structural identification (partial-equilibrium)

 A lot of the research and source codes are mainly written in MATLAB. The purpose of this repository is to direct-transalte those source codes from academic research and codes publicly available into open-sourced [julia programming languge](https://julialang.org/).

## Contents of source code
 | Source code | Model framework        | Source
 --- | --- | --- 
 | bvar      | Bayesian VAR model with Gibbs sampling (Minnesota-prior)   | Gary Koop and [Dimitris Korobilis replication](https://sites.google.com/site/dimitriskorobilis/matlab) of [Christiano et al. (2016)](https://onlinelibrary.wiley.com/doi/pdf/10.3982/ECTA11776)
 | tvp_var      | Time-varying parameter VAR with stochastic volatility (Code directly available; application example of macroeconomic consumer sentiment) | [Harron Mumtaz replication code](https://sites.google.com/site/hmumtaz77/research-papers) 
 | bh_bsvar      | Bayesian Structural Vector Autoregressive Model with Sign Restriction  | [Baumeister and Hamilton (2015) replication code](https://onlinelibrary.wiley.com/doi/abs/10.3982/ECTA12356)
 | mfvar      | Mixed-frequency VAR model  | [Harron Mumtaz replication code](https://sites.google.com/site/hmumtaz77/research-papers) 
