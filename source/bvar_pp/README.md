# Code for Pandemic Prior

This is a repository copy of code authored and written by Danilo Cascaldi-Garcia/

This repository contains the author's code and make edits for my application and do NOT claim authorship of the original work. To reference the original paper and code, reference the [author's website](https://sites.google.com/site/cascaldigarcia/pandemic-priors-bvar).

______________________________________

Files and replication JULIA codes of Cascaldi-Garcia, D., "Pandemic Priors"

Use of code for research purposes is permitted as long as proper reference to source is given.
______________________________________

The main program BVAR_Pandemic_Priors.jl performs the Pandemic Priors Bayesian VAR estimation with the time dummies on March to August 2020, and identifies an EBP shock with a recursive Cholesky structure, where EBP is ordered first.

"covid_periods" defines how many monthly dummies to include (starting from and including March 2020).  Set to zero to run a conventional Minnesota Prior as in Banbura, Giannone, and Reichlin (2010).

"\phi" defines how much signal the econometrician would like to take from the pandemic period. With \phi =999, the value for \phi will be the optimal from a marginal likelihood standpoint. With \phi close to zero the time dummies are "active," soaking all the pandemics variance; with phi close to infinity the time dummies are "inactive," and the model boils down to a conventional Minnesota Prior.

Auxiliary functions, including the creation of the dummy observations are stored in functions_Pandemic_Priors.jl.

Codes written in Julia v1.8.5

This version: February 2023

Danilo Cascaldi-Garcia

Thanks to William Gatt for suggestions of adjustments.