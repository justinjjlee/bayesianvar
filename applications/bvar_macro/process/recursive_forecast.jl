using Pkg
#using Distributed#, GPUArrays
#addprocs(5)

using Base: @kwdef
using Statistics, LinearAlgebra, Distributions, Random
using BenchmarkTools, ProgressMeter
using DelimitedFiles, DataFrames
#using Plots, LaTeXStrings
using CSV, Dates
using JLD
Random.seed!(1234)
using Plots
cd(@__DIR__)
# Data location in application
str_dir_git = splitdir(splitdir(splitdir(pwd())[1])[1])[1]
# Pull bvar functions 
include(str_dir_git*"/source/bvar/functions.jl")

# ================================================================================================
# Fitness 
# ................................................................................................
 
# Data import
str_dataloc = str_dir_git*"/applications/bvar_macro/data/df_wk.csv"
dfoj = CSV.read(str_dataloc, DataFrame)
# log-differencing interval
global δ_interval = 7 # weekly of seven days
global δ_diff = 52
global boot_t = 500 # requirement time period for bootstrap sampling
global boot_n = 230 # Boostrap MC sampling estimation sample count

df = dfoj[completecases(dfoj),:]

# Data manipulation
dfmod = log.(df[δ_diff:end,2:end]) .- log.(df[1:end-δ_diff+1,2:end])
# Use the complete case of rows 
y = dfmod[completecases(dfmod), :]

# ................................................................................................
# Fitness 
# ................................................................................................
 
# Define parameters
iterparam = bvar_params(
    constant = true,     # true: if you desire intercepts, false: otherwise 
    p = 7,               # Number of lags on dependent variables
    differenced = true,  # Is the data differenced?
    forecasting = true,  # true: Compute h-step ahead predictions, false: no prediction
    quantile_ci = 0.05,  # confidence interval quantile range
    forecast_method = 1, # 0: Direct forecasts/1: Iterated forecasts
    repfor = 10,         # Number of times to obtain a draw from the predictive 
                         #  density, for each generated draw of the parameters
    h = 1,               #  h-period ahead forecast: medium term average
    impulses = false,    # true: compute impulse responses, false: no impulse responses
    ihor = 21,           # Horizon to compute impulse responses
    # Set prior for BVAR model:
    prior = 2,           # prior = 1 --> Indepependent Normal-Whishart Prior
                         # prior = 2 --> Indepependent Minnesota-Whishart Prior
    # Gibbs-related preliminaries
    nsave = 1000,         # Final number of draws to saves
    nburn = 200           # Draws to discard (burn-in)
)

# Each iteration sets to start with specific dataframe
iter_start = 400 #th data points, ending and last period to forecast
iter_lengn = 300 # look-back period, Period to fit and calirate the mode rolling
iter_end,k = size(y) #th data points #th data points

# Periods to go through the iteration of recursive forecast
sim_iteration = iter_start:iter_end
# Saving results in matrix
ypred_tensor = zeros(length(sim_iteration), iterparam.nsave, 10, k)

for (idx, sim_iter) in enumerate(sim_iteration)
    # Select data period to be trained
    sim_period = (sim_iter-iter_lengn):sim_iter
    y_iter = y[sim_period, :]

    # Fit the model, only output projection
    ~, ~, ypred_draws, ~, ~, ~ = bvar_base(Matrix{Float64}(y_iter), iterparam)

    # save the forecast results
    ypred_tensor[idx, :, :, :] = ypred_draws
    print("$(sim_iter) of $(iter_end) completed")
end

# In case pause and data save is required
#using JLD
#@save "simdata.jld" ypred_tensor

# =============================================================================================================
# Projections 
# .............................................................................................................
# Process projections
# Projections and confidence band calculation, quantile confidence band
yhat_fit_ub = zeros(length(sim_iteration), k)
yhat_fit_mi = zeros(length(sim_iteration), k)
yhat_fit_lb = zeros(length(sim_iteration), k)
[yhat_fit_ub[irow, icol] = quantile(ypred_tensor[irow, :, end, icol], 1-iterparam.quantile_ci) for irow in 1:length(sim_iteration) for icol in 1:k]
[yhat_fit_mi[irow, icol] = quantile(ypred_tensor[irow, :, end, icol], 0.5) for irow in 1:length(sim_iteration) for icol in 1:k]
[yhat_fit_lb[irow, icol] = quantile(ypred_tensor[irow, :, end, icol], iterparam.quantile_ci) for irow in 1:length(sim_iteration) for icol in 1:k]

# Building process for projections

# Time period set
t = df[end-length(sim_iteration)+1:end,:date_wk] 
# one-week ahead forecast, so need to add a week for each
t += Day(7)

plot_yhat_median = DataFrame(yhat_fit_mi, names(y))
plot_yhat_median[:, "time"] = t

# Pull vintage projection
str_dataloc_vint = str_dir_git*"/applications/bvar_macro/results/projections/df_wk_proj_vint.csv"

# Save vintage projection by appending the new (if not exists)
plot_yhat_median |> CSV.write(str_dataloc_vint)

# .............................................................................................................
# Plotting
# Get the date of the vintage run date
t_lookback = 52 * 4
plot_t = t[end-t_lookback-1:end-1]
# Most recent last couple weeks of data points from: y
plot_y = y[end-t_lookback:end, :]

plot_hat = plot_yhat_median[end-t_lookback-1:end-1, :]
plot_hat_lb = plot_yhat_median[end-t_lookback-1:end-1, 1:k] .- yhat_fit_lb[end-t_lookback-1:end-1,1:k]
plot_hat_ub = yhat_fit_ub[end-t_lookback-1:end-1, 1:k] .- plot_yhat_median[end-t_lookback-1:end-1, 1:k]

str_titles =[
    "S&P 500",
    "Unemployment Claims - New",
    "Economic Policy Uncertainty",
    "Short-term Treasury Yields"
]

# Plot the projections
for iter_col in 1:k
    iter_name = names(plot_y)[iter_col]
    plot(plot_t, plot_y[:,iter_col], label="Actual", dpi=300)
    plot!(plot_t, plot_hat[:,iter_col], 
        label="Projection - Vintage", 
        color=:orange, fc=:orange, fa=0.3,
        ribbon=(plot_hat_lb[:,iter_col], plot_hat_ub[:,iter_col]),
        linewidth = 3,
        )
    hline!([0], color=:black, label=false, linewidth=1, linestyle=:dash)
    title!("Recursive Forecast: "*str_titles[iter_col])
    savefig(str_dir_git*"/applications/bvar_macro/results/projections/proj_$(iter_name)_recursive.png")
end