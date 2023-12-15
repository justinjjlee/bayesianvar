# Bayesian VAR model with Minnesota prior
#   Data sample and code transaltion from source 
# NOTE on transaltion
#   Cholesky depcomposition, by default on MATLAB, is upper-triangle
#       the simpler-hard-coded cholesky function (with hermitian depednece)
#       is coded in lower-triangle format:: transpose needed
#   Cholesky factorization needs computation of Hermitian form
#       this includes any distribtuion approach (Multinomial distribtuion)
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

# Set working directory
cd(@__DIR__)

# Data location in application
str_dir_git = splitdir(splitdir(pwd())[1])[1]

include("functions.jl")
# VAR using the Gibbs sampler, based on independent Normal Wishar prior

#------------------------------LOAD DATA-----------------------------------
# Load Quarterly US data on inflation, unemployment and interest rate, 
# 1953:Q1 - 2006:Q3
# Data: inflation, unemployment and interest rate
#Yraw = DataFrame(readdlm("Yraw.dat"), [:inf, :unemp, :ffr])
#Yraw = Matrix(Yraw)
#Yraw = Yraw[2:end, :] - Yraw[1:end-1, :]
#str_varname = ["PCE Inflation", "Unemployment", "Federal Funds Rate"]

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


# =============================================================================================================
# Fitness 
# .............................................................................................................
# Posterior mean of parameters:
ALPHA_mean = dropdims(mean(ALPHA_draws, dims = 1); dims=1); #posterior mean of ALPHA
SIGMA_mean = dropdims(mean(SIGMA_draws, dims = 1); dims=1); #posterior mean of SIGMA

# Evaluate fitness 
yhat_fit = zeros(testparam.nsave, size(X)[1], size(Y)[2])
yhat_fit_ub = zeros(size(X)[1], size(Y)[2])
yhat_fit_lb = zeros(size(X)[1], size(Y)[2])
for iGibbs = 1:testparam.nsave
    yhat_fit[iGibbs,:,:] = X * ALPHA_draws[iGibbs,:,:]
    # quantile calculations
end
yhat_fit_mean = median(yhat_fit, dims = 1)[1,:,:]

# Recover the level 


# plotting
for iter ∈ 1:length(str_varname)
    iter_varname = str_varname[iter]

    plot(yhat_fit_ub[:,iter]
        , fillrange = yhat_fit_lb[:,iter]
        , fillstyle = :/, label = "Estimated confidence band"
        ,legend=:topleft
    )
    # Point estimates
    plot!(Y[:,iter], color = :black, linestyle = :dash, label = "Actual")
    # Centre estimate
    plot!(yhat_fit_mean[:,iter], linecolor = :red, label = "Median Estimate")
    # Add title
    title!("Model fitness of variable: $(iter_varname)")
    # Print chart
    png("Fitness_$(iter_varname).png")
end


# Recover the level information



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
    plot(plot_y[:,"time"], plot_y[:,iter_col], label=iter_name)
    plot!(plot_yhat_median[:,"time"], plot_yhat_median[:,iter_col],
        ribbon=(
            plot_yhat_median[:,iter_col] .- yhat_fit_lb[:,iter_col], 
            yhat_fit_ub[:,iter_col] .- plot_yhat_median[:,iter_col]),
        label = "Projection",fc=:orange, fa=0.3, linewidth = 4
    )
    savefig(str_dir_git*"/applications/bvar_macro/results/projections/proj_$(iter_name).png")
end


recorddf = vcat(plot_y, plot_yhat_median)
recorddf |> CSV.write("record.csv")


# =============================================================================================================
# Sample Evaluation 
# .............................................................................................................
# Bootstrap draws ---------------------------------------------------------------------------------------------
global boot_A = zeros(testparam.nsave, testparam.constant+(testparam.p*k), k, boot_n)
global boot_Σ = zeros(testparam.nsave, k, k, boot_n)
global boot_ŷ = zeros(testparam.nsave, testparam.h, k, boot_n)
global boot_ϕ = zeros(testparam.nsave, k, k, testparam.ihor, boot_n)
# Below should be distributed computing 

# Bootstrap iteration
#@sync @distributed 
print("Bootstrap Sampling Calibration")
for iter_boot in 1:boot_n
    # Randomly select the sample period on bootstratp
    # Starting with the sample period
    #   Preserving the least required bootstrap
    boot_t_start = rand(1:(size(y)[1] - boot_t + 1))
    boot_t_end   = boot_t_start + boot_t - 1
    # Select the period sought
    y_boot = y[boot_t_start:boot_t_end, :]

    # Model build
    ALPHA_boot, SIGMA_boot, ypred_boot,# irf_boot,
        X_boot, Y_boot = bvar_base(Matrix{Float64}(y_boot), testparam)

    # Save the result - iteration 
    global boot_A[:,:,:,iter_boot] = ALPHA_boot
    global boot_Σ[:,:,:,iter_boot] = SIGMA_boot
    global boot_ŷ[:,:,:,iter_boot] = ypred_boot
    #global boot_ϕ[:,:,:,:,iter_boot] = irf_boot
end

# Save to file ------------------------------------------------------------------------------------------------
save(str_dir_git*"/applications/bvar_macro/results/simulation.jld",
    "ALPHA_draws", ALPHA_draws,
    "SIGMA_draws", SIGMA_draws, 
    "ypred_draws", ypred_draws,
    "irf_draws", irf_draws,
    "boot_A", boot_A,
    "boot_Σ", boot_Σ,
    "boot_ŷ", boot_ŷ,
    "boot_ϕ", boot_ϕ
)


# Development 

params = testparam
function postmodel_process(, params)
    # Post process

    # Coefficient draw plots
    #   Distribution for full sample
    #   Distribution for the lastest sample 
    #   Distribution for the bootstrap sample

    # Forecast projection 

end

# ====================== End Sampling Posteriors ===========================



# Forecasting
# (1) All-time usage 
# (2) Bootstrap trained parameter
# (3) Most recent last couple years trained parameter forecast

if forecasting
    yhat_mean = dropdims(mean(yhat_draws; dims=1); dims=1)
    logp_mean = mean(log.(logp_draws))

    # at T+h
    if forecast_method == 0
        global true_value = Y1[T+1,:];
    elseif forecast_method == 1
        global true_value = Y1[T+h,:];
    end
end
plot!([Yraw[p+1:end,2], yhat_fit_mean[:,2]])


# Scenario planning?