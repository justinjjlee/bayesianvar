
################################################################################
# variables & parameters to be used, and explanation

mutable struct  PrecompileParameter
        # kcum: indicator variable for cumulation of row response of column shocks.
        kcum::Array{Int64}
        # knorm_scale: Indicator variable for normalizing scale (signs), -1 or 1.;
        knorm_scale::Array{Int64}
        yall::Array{Float64} # data matrix to be used.
        varnames::Vector{String} # variable names
        shocknames::Vector{String} # shock names
        nlags::Int64; # lag order
        tstart::Int64; # start estimation of vector indicator
        tend::Int64; # end estimation of vector indicator

        date_start::Float64; # start date in numeric form
        date_end::Float64; # end date in numeric form
        time::FloatRange{Float64}; # time range used

        T::Int64; # number of observations
        n::Int64; # number of variable used
        k::Int64; # number of regressors.;

        cA::Vector{Float64} # Set prior density parameters
        nA::Int64 # number of unknown parameters for structural matrix A
        signA::Vector{Float64} # vector of sign restrictions
        names_α::Vector{String} # names for elements of matrix A

        kappa::Matrix{Float64} # weights on priors
        eta::Matrix{Float64} # prior weight used for B
        lambda0::Float64 # weight on piors, larger value puts lower weight on priors
        lambda1::Float64 # weight on higher lag order, 1 makes essentially zero.
        lambda3::Float64 #

        ndraws::Int64 # number of iterations for MH algorithm
        nburn::Int64 # number of 'burn-ins' to discard from computational results
        nuse::Int64 # number of iteration samples to be used
        # step-for Hessian decomposed matrix to make acceptance rate to be around 30%
        chsi::Float64
        #M_randn::Array{Float64,2} # pre-generate random numbers of N(0,1) for computation

        hmax::Int64 # maximum horizon of impulse response to be computed
        prob_tail::Float64; # percentile of tail-end distribution of confidence band
        index0::Int64 # percentile - median indicator
        index1::Int64 # percentile - lower half of prob_tail
        index2::Int64 # percentile - upper half of prob_tail
        horizon_impact::Matrix{Int64} # sequence vector for ploting impuse responses

        x_prior::FloatRange{Float64} # x-axis used to plot prior density functions
        x_prior_sign::Array{Float64} # signs for number entries for parameters
        y_prior::Array{Float64} # Matrix of prior density functions to plot
end

################################################################################
##read in data.
print("Reading in data and initializing parameters \n");
print("Read in data \n");
kcum = [1; 1; 1; 1;
        1; 1; 1; 1;
        1; 1; 1; 1;
        1; 1; 1; 1;
       ];
knorm_scale = [1; 1; 1; 1;
               1; 1; 1; 1;
               1; 1; 1; 1;
               1; 1; 1; 1;
              ];

# ..............................................................................
# save directory if necessary
#cd("/home/justinjlee/Dropbox/wMike_BSVAR/computation/julia");
savedir = pwd() * "//results";
# ..............................................................................
#read in data using package DataFrames
data_all = readtable("Lee_2017_wcs_2var.csv"); #in data frame
data_all = convert(Array, data_all);
# data range horizon: Jan-1983 ~ Dec. 2016
# transform all to differences in log-level data.
yall = log(data_all[(2:end), :]) - log(data_all[(1:(end-1)), :]);
# ..............................................................................
# Define variables
varnames = [" production"; " output"; " c.spread"; " price-crude"];
shocknames = [" supply shock";
              " aggregate demand shock";
              " ref.supply shock";
              " resid.demand shock"
             ];
# ..............................................................................
# Set lag order and time frame.
nlags = 12;
tstart = nlags + 1;
tend = size(yall)[1];

date_start = 1970 + 1/12;
date_end = 2016 + 12/12;
time = date_start : 1/12 : date_end;

YY = yall[(tstart:tend), :];
T = size(YY)[1];
n = size(YY)[2];
time = time[(end-T+1):end];
k = (n*nlags) + 1;
# ..............................................................................
# End of data compile ##########################################################
print("Parameter initialization..... \n");
srand(599158629);
cA = [0.5; 0.02; -0.1; -0.5; 0.3; 0.2; 0; -0.8; 0.5; -0.02]; nA = length(cA);
signA = [0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0];
names_α = [" α_(sp)"; " α_(ys)"; " α_(yc)"; " α_(yp)"; " α_(cs)";
          " α_(cy)"; " α_(cp)"; " α_(ps)"; " α_(py)"; " α_(pc)"
         ];
# ..............................................................................
# Define parameters for Metropolis-Hastings (MH) core algorithm
kappa = 2 * ones(n,1);
eta = [eye(n) zeros(n, k-n)];
lambda0 = 1.0e9;
lambda1 = 1.0;
lambda3 = 100.0;
chsi = 0.5; # or 0.5 for earlier version

ndraws = convert(Int64, 3e5);
nburn = convert(Int64, 1e5);
nuse = ndraws - nburn;
M_randn = randn(ndraws, 2);
# ..............................................................................
# Specifications for the impulse response and historical decompositions.
hmax = 15;
prob_tail = 0.025;
index0 = convert(Int64, round(0.5*nuse));
index1 = convert(Int64, round(prob_tail*nuse));
index2 = convert(Int64, round((1-prob_tail)*nuse));
horizon_impact = (0:1:hmax);

est_irf = zeros(n, n, length(horizon_impact), nuse);
hd_posterior = zeros(n, n, T, nuse);
#=
est_irf = GPUArray(zeros(n, n, length(horizon_impact), nuse));
hd_posterior = GPUArray(zeros(n, n, T, nuse));
=#
# ..............................................................................
# End of parameters initialization #############################################
print("Assign prior density..... \n");
names_α = [" α_(sp)"; " α_(ys)"; " α_(yc)"; " α_(yp)"; " α_(cs)";
          " α_(cy)"; " α_(cp)"; " α_(ps)"; " α_(py)"; " α_(pc)"
         ];
x_prior = -10:0.01:10;
# get prior density all at once, in order by row:
y_prior = func_pdfPriors(kron(x_prior', ones(nA, 1)));
################################################################################
# End of script for precompilation,
print("Pre-compilation completed.....\n");
