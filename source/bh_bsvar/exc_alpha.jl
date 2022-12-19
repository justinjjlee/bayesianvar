# core algorithm................................................................
# ==============================================================================
#=
# Change working directory, if necessary
# run following
include("exc_alpha.jl");
=#
tic()
print("▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦\n");
print("Computation of Bayesian Stuructural Vector Autoregression \n");
print("Original code in MATLAB by Baumeister and Hamilton (2015)\n");
print("julia version (this code) by Justin J. Lee\n");
print("▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦\n");
print("▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦\n");
print("▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦\n");
# This version excludes long-run impact.;
# include packages needed.;
using ProgressMeter, Distributions, DataFrames, Optim, PyPlot, GPUArrays;
# Package uses:
print("▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦\n");
print("Include all functions used in computations \n");
include("functions.jl");
print("▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦\n");
include("sub_precompile.jl");
print("▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦\n");
## Translate code ######################################## below ##############
#of exc_alpha MATLAB....
print("Computation for drawing sturcutural contemporaneous impact matrix\n");
if nlags > 0
  XX = yall[((tstart-1):(tend-1)), :]; # frist lag order
  for ilags = 2:nlags # starting with lag order of 2.;
    XX = [XX yall[((tstart-ilags):(tend-ilags)), :]];
  end
  XX = [XX ones(Float64, T, 1)];
else
  XX = ones(Float64, T, 1);
end

omegahat = (YY'*YY - YY'*XX*inv(XX'*XX)*XX'*YY)/T;

# Calculate variance-covariance matrix of univariate AR residuals
e = zeros(T, n);
for i_var = 1:n
  if nlags > 0
    ylags = yall[((tstart-1):(tend-1)), i_var];
    for i_lags = 2:nlags
      ylags = [ylags yall[((tstart-i_lags):(tend-i_lags)), i_var]];
    end
    ylags = [ylags ones(T, 1)];
  else
    ylags = ones(T, 1);
  end
  e[:, i_var] = yall[(tstart:tend), i_var]-(ylags*inv(ylags'*ylags)*
                                            ylags'*yall[(tstart:tend), i_var]);
end
# residuals based on AR estimate.;
Sstar = e'*e./T;
# ..............................................................................
# Calculate inverse of Mtilde for standard Minnesota prior
v1 = 1:1:nlags; v1 = v1[:, :]'; # to change into multi-Array
v1 = v1' .^ (-2.0*lambda1); # need to be in Float, not integer for type-stable.;
v2 = 1 ./ diag(Sstar);
v3 = kron(v1, v2);
v3 = (lambda0^2.0)*[v3; (lambda3^2.0)];
v3 = 1 ./ v3; v3 = vec(v3); # vectories for for type-stable.;
Mtildeinv = diagm(v3);
# ..............................................................................
# Calculate parameters that characterize posteriors.;
kappastar = kappa + (T/2);
ytilde = ((YY')*YY) + eta*Mtildeinv*eta';
yxtilde = ((YY')*XX) + eta*Mtildeinv;
xtildei = zeros(k, k, n);
for i_var = 1:n
  xtildei[:, :, i_var] = inv(XX'*XX + Mtildeinv);
end
# ..............................................................................
# Random-Walk Metropolis-Hastings algorithm - sample iterations: structural A.;
print("▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦\n");
print("Random-Walk Metropolis-Hastings algorithm computation\n");
print("Sub section: calculating posterior modes and Hessian for optimality\n");
# create functional form - function handle
func_nLLpost(estA) = func_nLLposterior(estA, signA, kappa, T, omegahat, Sstar,
                                     ytilde, xtildei, yxtilde);
estA_max = optimize(func_nLLpost, cA, BFGS(),
      Optim.Options(iterations = 10000, store_trace=true, extended_trace=true));
#if not require hessian, then do autodiff = true in Optim.Options
#fieldnames(estA_max)
estA_median = estA_max.minimizer;
# and recover Hessian,
invM_hessian = estA_max.trace[end].metadata["~inv(H)"];
print("⬕⬕⬕⬕⬕⬕⬕⬕⬕⬕⬕⬕⬕⬕⬕⬕⬕⬕⬕⬕⬕⬕⬕⬕⬕⬕⬕⬕⬕⬕⬕⬕⬕⬕⬕⬕⬕⬕⬕⬕⬕⬕⬕⬕⬕⬕⬕⬕⬕⬕⬕⬕⬕⬕⬕⬕⬕⬕⬕⬕⬕⬕⬕⬕⬕⬕⬕⬕\n");
print("Posterior median\n");
display(estA_median);
print("Inverse Hessian Matrix\n");
display(invM_hessian);
# calculate optimal step size for MH algorithm,
if minimum(eig(invM_hessian)[1]) > 0
  scale_optimal = chol(Hermitian(invM_hessian))';
else
  scale_optimal = eye(nA);
end
estA_old = estA_median

print("Initialize Metropolis-Hastings (MH) algorithm \n");
#estA_old = testingAnew;
LLposterior_old, zeta_old = func_LLposterior(estA_old, signA, kappa, T, omegahat, Sstar, ytilde, xtildei, yxtilde)
mstar_old = func_setA(estA_old) * eta;

# Save results from iterations,
A_posterior = zeros(nA, ndraws);
zeta_posterior = zeros(n, ndraws);
mstar_posterior = zeros(n, k, ndraws);

###############################################################################
# Metropolis-Hastings core algorithm,
progress_MH = Progress(ndraws, 1, "MH.iter:", 30)

n_accept = 0;
rate_accept = 0;
# print progress bar to initiate MH-core algorithm
rate_accept = "$(rate_accept)%"; name_rate = "real-time acceptance rate";
ProgressMeter.next!(progress_MH; showvalues = [(name_rate, rate_accept)]);
for i_iter = 1:ndraws
    # propose new structural contemporaneous matrix,
    estA_new = estA_old +
               (chsi*scale_optimal*randn(nA,1)) /
                sqrt(0.5*((M_randn[i_iter, 1]^2) + (M_randn[i_iter, 2]^2)));
    if minimum(sign(estA_new).*signA) >= 0.0
      LLposterior_new, zeta_new, mstar_new = func_LLposterior(estA_new, signA, kappa, T, omegahat, Sstar, ytilde, xtildei, yxtilde);
      prob_rand = rand(1)[1,1];
      if prob_rand <= minimum([exp(LLposterior_new - LLposterior_old); 1])
        estA_old = estA_new;
        zeta_old = zeta_new;
        mstar_old = mstar_new;
        LLposterior_old = LLposterior_new;
        n_accept = n_accept + 1;
      end
    end
    A_posterior[:, i_iter] = estA_old;
    zeta_posterior[:, i_iter] = zeta_old;
    mstar_posterior[:, :, i_iter] = mstar_old;

    rate_accept = round((n_accept/i_iter*100), 2);
    rate_accept = "$(rate_accept)%"; name_rate = "real-time acceptance rate";
    ProgressMeter.next!(progress_MH; showvalues = [(name_rate, rate_accept)]);
end
A_posterior = A_posterior[:, ((nburn+1):end)];
zeta_posterior = zeta_posterior[:, ((nburn+1):end)];
mstar_posterior = mstar_posterior[:, :, ((nburn+1):end)];
print("▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦\n");

#check behaviour
print("▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦\n");
print("Plotting prior and posterior density functions \n");
include("sub_plot_posteriors.jl");
print("▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦\n");
print("Computing impulse response functions \n");
# run impulse responses.
include("sub_compute_irfBSVAR.jl");
print("▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦\n");
print("Computing historical decompositions \n");
include("sub_compute_hdBSVAR.jl")




# print square for completions.;
print("\n▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦\n");
# end time,
toc()
print("\n⬔⬔⬔⬔⬔⬔⬔⬔⬔⬔⬔⬔⬔⬔⬔⬔⬔⬔⬔⬔⬔⬔⬔⬔⬔⬔⬔⬔⬔⬔⬔⬔⬔⬔⬔⬔⬔⬔⬔⬔⬔⬔⬔⬔⬔⬔⬔⬔⬔⬔⬔⬔⬔⬔⬔⬔⬔⬔⬔⬔⬔⬔⬔⬔⬔⬔\n");
