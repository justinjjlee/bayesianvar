# ==============================================================================
## function to be updated .....................................................
#Set vector to matrix of A
function func_setA(vecA::Array)
  A = [ 1         0         0         -vecA[1];
        -vecA[2]  1         -vecA[3]  -vecA[4];
        -vecA[5]  -vecA[6]  1         -vecA[7];
        -vecA[8]  -vecA[9]  -vecA[10] 1
      ];
  return A
end
# ==============================================================================
# functions of prior densities,
function func_pdfPriors(x::Any)
  # producer parameter used for plotting purpose, only.;
  param_prior = zeros(size(x)[1], 2);
  pdf_prior = zeros(size(x)[1], size(x[:,:])[2]);
  # α_sp : oil price shock on oil supply
  param_prior[1, :] = [2 3];
  pdf_prior[1, :] = pdf(Beta(param_prior[1, 1], param_prior[1, 2]), x[1,:]);
  # α_ys : oil supply shock on aggregate output
  param_prior[2, :] = [3 1/5];
  pdf_prior[2, :] = pdf(Gamma(param_prior[2, 1], param_prior[2, 2]), x[2, :]);
  # α_yc : crack spread on aggregate output
  param_prior[3, :] = [3 1/5];
  pdf_prior[3, :] = pdf(Gamma(param_prior[3, 1], param_prior[3, 2]), -x[3, :]);
  # α_yp : oil price shock on aggregate output
  param_prior[4, :] = [3 1/5];
  pdf_prior[4, :] = pdf(Gamma(param_prior[4, 1], param_prior[4, 2]), -x[4, :]);
  # α_cs : oil supply shock on crack spread
  param_prior[5, :] = [5 1/2];
  pdf_prior[5, :] = pdf(Gamma(param_prior[5, 1], param_prior[5, 2]), x[5, :]);
  # α_cy : aggregate output shock on crack spread
  param_prior[6, :] = [5 2];
  pdf_prior[6, :] =  pdf(Gamma(param_prior[6, 1], param_prior[6, 2]), x[6, :]);
  # α_cp : oil price shock on crack spread
  param_prior[7, :] = [1 NaN];
  pdf_prior[7, :] = pdf(TDist(param_prior[7, 1]), x[7, :]);
  # α_ps : oil supply shock on crack spread
  param_prior[8, :] = [5 1/2];
  pdf_prior[8, :] = pdf(Gamma(param_prior[8, 1], param_prior[8, 2]), -x[8, :]);
  # α_py : aggregate output shock on oil price
  param_prior[9, :] = [10 1/3];
  pdf_prior[9, :] =  pdf(Gamma(param_prior[9, 1], param_prior[9, 2]), x[9, :]);
  # α_pc : crac spread shock on oil prce
  param_prior[10, :] = [3 1/7];
  pdf_prior[10, :] = pdf(Gamma(param_prior[10, 1], param_prior[10, 2]), -x[10, :]);

  return pdf_prior
end
# ==============================================================================
# ==============================================================================
# Function not need be updated :::::::::::::::::::::::::::::::::::::::::::::::::
# ==============================================================================
# ==============================================================================
# Function to detrend data in matrix operation of linear least squre methods,
function func_detrend(Y::Array)
  T = size(Y)[1];
  M_fit = zeros(T, 2); # simple linear frist
  M_fit[:, 1] = (1:1:T)./T; # capturing linear trend,
  M_fit[:, 2] = ones(T, 1); # capturing constant term,
  y_trend = Y - M_fit*(M_fit\Y);
  return y_trend
end
# ==============================================================================
# function of log-likelihood of prior densities
function func_LLprior(vecA::Array, signA::Array)
  # save log-likelihood.;
  LL = sum(log(func_pdfPriors(vecA)));
  return LL;
end
# ==============================================================================
# Function to calculate posterior negative log-likelihood,;
function func_nLLposterior(estA, signA, kappa, T, omegahat, Sstar, ytilde,
                          xtildei, yxtilde)
  n = size(omegahat)[1];
  A = func_setA(estA);
  Q = A * omegahat * A';
  tau = kappa .* diag(A*Sstar*A');
  kappastar = kappa + (T/2);
  zetastar = zeros(n,1); zetastar = zetastar[:, :];
  for i_var = 1:n
    A_i = A[i_var, :]; A_i = A_i[:, :]';
    ytildei = A_i * ytilde * A_i';
    yxtildei = A_i * yxtilde;
    zetastar_i = ytildei - (yxtildei*xtildei[:,:,i_var]*yxtildei');
    zetastar[i_var, 1] = zetastar_i[1, 1];
  end
  LL = func_LLprior(estA, signA) + ((T/2)*log(det(Q))) -
       (kappastar'*log((2*tau/T) + (zetastar/T))) + (kappa'*log(tau));
  negLL = -LL; negLL = negLL[1,1]; #reshape from singleton Array to a scaler.;
  return negLL;
end
# ==============================================================================
# Function to calculate posterior negative log-likelihood,;
function func_LLposterior(estA, signA, kappa, T, omegahat, Sstar, ytilde,
                          xtildei, yxtilde)
  n = size(omegahat)[1]; k = size(xtildei)[1];
  A = func_setA(estA);
  Q = A * omegahat * A';
  tau = kappa .* diag(A*Sstar*A');
  kappastar = kappa + (T/2);
  zeta_new = zeros(n,1); zeta_new = zeta_new[:, :];
  mstar_new = zeros(n, k);
  for i_var = 1:n
    A_i = A[i_var, :]; A_i = A_i[:, :]';
    ytildei = A_i * ytilde * A_i';
    yxtildei = A_i * yxtilde;
    zeta_new_i = ytildei - (yxtildei*xtildei[:,:,i_var]*yxtildei');
    zeta_new[i_var, 1] = zeta_new_i[1, 1];
    mstar_new[i_var, :] = yxtildei * xtildei[:, :, i_var];
  end
  LL = func_LLprior(estA, signA) + ((T/2)*log(det(Q))) -
       (kappastar'*log((2*tau/T) + (zeta_new/T))) + (kappa'*log(tau));
  LL = LL[1,1]; #reshape from singleton Array to a scaler.;

  return LL, zeta_new, mstar_new;
end
# ==============================================================================
# function of VAR(p)
function func_VectorAR(y::Array, p::Int64)
  t,k = size(y);
  y = y';
  Y = y[:, (p:t)];
  for i_lag = 1:(p-1)
    Y = [Y; y[:, ((p-i_lag):(t-i_lag))]];
  end
  Z = [ones(1, (t-p)); Y[:, (1:(t-p))]];
  Y = y[:, ((p+1):t)];
  # least square estimate
  #B = (Y*Z')*inv(Z*Z');
  B = (Y*Z')/(Z*Z');
  U = Y - (B*Z);
  SIGMA = (U*U')/(t-p-(p*k)-1);
  V = B[:, 1]; V = V[:, :]; # convert to array format
  A = B[:, (2:end)];
  return A, SIGMA, U, V;
end
