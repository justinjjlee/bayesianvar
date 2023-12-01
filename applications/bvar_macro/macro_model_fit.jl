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

# Data location in application
str_dir_git = splitdir(splitdir(splitdir(pwd())[1])[1])[1]
# Pull bvar functions 
include(str_dir_git*"/source/bvar/functions.jl")
# ================================================================================================
# Data import from FRED to be updated soon
#   Dataframe and format:
#       Christiano et al.: 
#       Phillips and Okun's Law (job opening should follow my prior package)
#       Stock market and uncertainty
# ================================================================================================
# Data update and save
cd(@__DIR__)
include("./data/proc_data_macro.jl")
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
# For real-time measures of incomplete data
ỹ = dfmod[map(!,completecases(dfmod)),:]

# Paramter defined
T, k = size(y) # Total calibration data size
# ................................................................................................
# Fitness 
# ................................................................................................
 
# Define parameters
testparam = bvar_params(
    constant = true,     # true: if you desire intercepts, false: otherwise 
    p = 7,               # Number of lags on dependent variables
    differenced = true, # Is the data differenced?
    forecasting = true,  # true: Compute h-step ahead predictions, false: no prediction
    quantile_ci = 0.05,  # confidence interval quantile range
    forecast_method = 1, # 0: Direct forecasts/1: Iterated forecasts
    repfor = 10,         # Number of times to obtain a draw from the predictive 
                        #  density, for each generated draw of the parameters
    h = 5,              # Number of forecast periods
    impulses = true,     # true: compute impulse responses, false: no impulse responses
    ihor = 21,           # Horizon to compute impulse responses
    # Set prior for BVAR model:
    prior = 2,           # prior = 1 --> Indepependent Normal-Whishart Prior
                        # prior = 2 --> Indepependent Minnesota-Whishart Prior
    # Gibbs-related preliminaries
    nsave = 1000,         # Final number of draws to saves
    nburn = 200           # Draws to discard (burn-in)
)
# Time fixes
t = df[δ_diff:end-size(ỹ)[1],:date_wk] # Total time period (real-time measurements for benchmarking)
t_tilde = df[end-size(ỹ)[1]+1:end,:date_wk] # Real-time period to be 
# Generate horizon ahead forecast
t_hat = [t[end] + Day(iter * δ_interval) for iter in 1:testparam.h]

# Full-sample model train and projection ----------------------------------------------------------------------
ALPHA_draws, SIGMA_draws, ypred_draws, irf_draws,
    X, Y = bvar_base(Matrix{Float64}(y), testparam)

# Posterior mean of parameters:
ALPHA_mean = dropdims(mean(ALPHA_draws, dims = 1); dims=1); #posterior mean of ALPHA
SIGMA_mean = dropdims(mean(SIGMA_draws, dims = 1); dims=1); #posterior mean of SIGMA

# Evaluate fitness 
yhat_fit = zeros(testparam.nsave, size(X)[1], size(Y)[2])
yhat_fit_ub = zeros(size(X)[1], size(Y)[2])
yhat_fit_lb = zeros(size(X)[1], size(Y)[2])
for iGibbs = 1:testparam.nsave
    yhat_fit[iGibbs,:,:] = X * ALPHA_draws[iGibbs,:,:]
end

# quantile calculations
yhat_fit_mean = median(yhat_fit, dims = 1)[1,:,:]

# =============================================================================================================
# Projections 
# .............................................................................................................
# Process projections
# Projections and confidence band calculation, quantile confidence band
yhat_fit_ub = zeros(length(t_hat), k)
yhat_fit_mi = zeros(length(t_hat), k)
yhat_fit_lb = zeros(length(t_hat), k)
[yhat_fit_ub[irow, icol] = quantile(ypred_draws[:,irow, icol], 1-testparam.quantile_ci) for irow in 1:size(yhat_fit_lb,1) for icol in 1:size(yhat_fit_lb,2)]
[yhat_fit_mi[irow, icol] = quantile(ypred_draws[:,irow, icol], 0.5) for irow in 1:size(yhat_fit_lb,1) for icol in 1:size(yhat_fit_lb,2)]
[yhat_fit_lb[irow, icol] = quantile(ypred_draws[:,irow, icol], testparam.quantile_ci) for irow in 1:size(yhat_fit_lb,1) for icol in 1:size(yhat_fit_lb,2)]

# Building process for projections
plot_yhat_median = DataFrame(yhat_fit_mi, names(y))
plot_yhat_median[:, "time"] = t_hat

# Pull vintage projection
str_dataloc_vint = str_dir_git*"/applications/bvar_macro/results/projections/df_wk_proj.csv"
df_proj = CSV.read(str_dataloc_vint, DataFrame)

# Save the 1-month horizon forecast only if they don't exist
# Avoid redundancy - save and add only when there is a new projection
#   Second optional (commented out) condition would only be false if the real-time projection matches exactly with the date code run and data releases
if df_proj[end,"time"] ∉ plot_yhat_median[:,"time"] #today() ∉ plot_yhat_median[:,"time"]
    # If weekly projections are made deligently, save the 1-week horizon
    if df_proj[end,"time"]+Day(7) == plot_yhat_median[1,"time"]
        df_proj = vcat(df_proj, plot_yhat_median[1:1,:])
    else  
        # If the projections were not made in the past (no vintage) and not on record, then, save the results 
        #   This meas saving all forecasts past of the vintage data points given the datetime of the code run
        #   The projections should not go beyond today's date (no real-time backfills)
        tempdf = plot_yhat_median[
            (plot_yhat_median[:,"time"] .< today()) .& 
            (plot_yhat_median[:,"time"] .!= df_proj[end,"time"])
            , :]
        df_proj = vcat(df_proj, tempdf)
    end
end
# Save vintage projection by appending the new (if not exists)
df_proj |> CSV.write(str_dataloc_vint)

# .............................................................................................................
# Plotting
# Get the date of the vintage run date
t_lookback = 52 * 2
# Most recent last couple weeks of data points from: y
plot_y = y[end-t_lookback:end, :]
plot_y[:, "time"] = t[end-t_lookback:end]

# Plot the projections
for iter_col in 1:k
    iter_name = names(plot_y)[iter_col]
    plot(plot_y[:,"time"], plot_y[:,iter_col], label=iter_name, dpi=300)
    plot!(df_proj[:,"time"], df_proj[:,iter_col], 
        label="Projection - Vintage", color=:orange,
        linewidth = 3
        )
    plot!(plot_yhat_median[:,"time"], plot_yhat_median[:,iter_col],
        color=:orange, linestyle =:dot,
        ribbon=(
            plot_yhat_median[:,iter_col] .- yhat_fit_lb[:,iter_col], 
            yhat_fit_ub[:,iter_col] .- plot_yhat_median[:,iter_col]),
        label = "Projection - Latest",fc=:orange, fa=0.3, linewidth = 2
    )
    savefig(str_dir_git*"/applications/bvar_macro/results/projections/proj_$(iter_name).png")
end
