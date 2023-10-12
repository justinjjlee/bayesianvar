# Code to test mixed-frequency VAR framework
using Pkg
using ProgressMeter
using DataFrames
using Statistics, LinearAlgebra, Distributions, PDMats
using Random
using PGFPlots
rng = MersenneTwister(1234);
# Some useful note on fitting distribution for julia
# https://stackoverflow.com/questions/57559589/distributions-jl-do-not-accept-my-non-positive-definite-covariance-function-w
# https://stackoverflow.com/questions/68009451/cannot-fit-a-mvnormal-distribution-with-distributions-jl-error-posdefexception
# https://docs.julialang.org/en/v1/stdlib/LinearAlgebra/

cd("--/GitHub/julia-VectorAR.jl/proc")
# Following the mumtaz code to generate simulated data
include("../test/sim_data_mixFrequency.jl")
include("../src/func_VectorAR.jl")
# Generate data
data, dataid, dataid0, mid, dataM = data_gen();

## NEED TO TEST WITH 3 VARIABLE wiht additional X col
#       TO SEE IF IT WORKS WELL


# computation settings
REPS=11000;
BURN=10500;
L = 3; # lag count

# Data ettings
Y = deepcopy(data);
N=size(Y)[2];





# Create data matrix with lag and constant
X = prepare( Y,L )
# Adjust data based on lag term
Y = Y[L+1:end, :];
X = X[L+1:end, :];
dataid0=dataid0[L+1:end, :];
dataM=dataM[L+1:end, :];

# Data ettings
T, N=size(Y);

# ................. Initial values for VAR coefficients
b0=X\Y;  #ols
e0=Y-X*b0;

# var/covar for error term
sigma=eye(N);

#priors for VAR coefficients (Banbura et.al)
lamdaP  = 1;
tauP    = 10*lamdaP;
epsilonP= 1;
muP=mean(Y)';
sigmaP=Array{Float64}(undef, N, 1);
deltaP=Array{Float64}(undef, N, 1);
e0=Array{Float64}(undef, T-1, N);

for i=1:N
    ytemp=Y[:,i];
    xtemp=hcat(lagspit(ytemp,1), ones(size(ytemp)[1],1));
    ytemp=ytemp[2:end,:];
    xtemp=xtemp[2:end,:];
    btemp=xtemp\ytemp;
    etemp=ytemp-xtemp*btemp;
    stemp=etemp'*etemp/(T - 1);
    if abs(btemp[1])>1
        btemp[1]=1;
    end
    deltaP[i] = btemp[1];
    sigmaP[i] = stemp[1];
    e0[:,i] = etemp;
end

#dummy data to implement priors see http://ideas.repec.org/p/ecb/ecbwps/20080966.html
yd,xd = create_dummies(lamdaP,tauP,deltaP,epsilonP,L,muP,sigmaP,N);

#Initial values for the Kalman filter B0/0
beta0 = Y[L,:]';
for j=1:L-1
    beta0=hcat(beta0, Y[L-j,:]');
end
P00=eye(size(beta0)[2])*0.1;  #P[0/0]





# -- step 1 Draw VAR coefficients  
X0=vcat(X, xd); #add dummy obs
Y0=vcat(Y, yd);
mstar=vec(X0\Y0);
vstar=kron(sigma,invpd(X0'*X0));
chck=-1;
while chck<0
    global varcoef=mstar+(randn(1,N*(N*L+1))*chol(vstar))'; #draw but keep stable
    ee=stability(varcoef,N,L);
    if ee==0;
        chck=1;
    end
end

time_weight = 1/4 # for monthly - quarterly frequency
global H_vec_empty = zeros(N, N*L)
# Given the first variable is infrequent, 
idx_col_freq = 2
# kron vec for X_values
temp_mat = eye(N-2+1)
H_vec_empty[idx_col_freq:N, idx_col_freq:N] = temp_mat

# For those with average observation
global H_vec = deepcopy(H_vec_empty)
for i in 1:N:N*L
    H_vec[1, i] = time_weight
end









num_var_param_coeff = size(X)[2] * size(Y)[2]
global bmat = Array{Float64}(undef, (REPS-BURN + 1), num_var_param_coeff)
global smat = Array{Float64}(undef, (REPS-BURN + 1), N, N)
global dmat = Array{Float64}(undef, (REPS-BURN + 1), size(Y)[1])







# Gibbs sampler
gibbs1=1;
for gibbs=1:REPS
    # -- step 1 Draw VAR coefficients  
    X0=vcat(X, xd); #add dummy obs
    Y0=vcat(Y, yd);
    mstar=vec(X0\Y0);
    vstar=kron(sigma,invpd(X0'*X0));
    chck=-1;
    while chck<0
        global varcoef=mstar+(randn(1,N*(N*L+1))*chol(vstar))'; #draw but keep stable
        ee=stability(varcoef,N,L);
        if ee==0;
            chck=1;
        end
        #println("keep trying")
    end

    # -- step 2 Draw VAR covariance
    # sample residual
    resids=Y0-X0*reshape(varcoef,N*L+1,N);
    scaleS=(resids'*resids);
    sigma=iwpQ(T,invpd(scaleS)); #draw for inverse Wishart

    # -- step 3 Carter Kohn algorithm  to draw monthly data
    # Initial settings
    ns=size(P00)[2];
    F, MUx = comp(varcoef,N,L,1); #companion form for coefficients
    Q=zeros(ns, ns);
    Q[1:N,1:N]=sigma; #companion form for covariance

    # Carter and Kohn algorithm to draw the factor
    beta_tt = zeros(T,ns); # will hold the filtered state variable
    ptt = zeros(T,ns,ns);    # will hold its variance
    # -- -- Step 6a run Kalman Filter
    beta11 = beta0;
    p11 = P00;

    for i=1:T
        nanid=mid[i,1]; #checks if the vector is missing
        if nanid==1 #missing
            H=H_vec_empty
    
            rr=zeros(1,N);
            rr[1]=1e10;  #big variance so missing data ignored
            R=diagx(rr);
        else  #valid  observation for first variable every 3rd month
            H=H_vec        
            rr=zeros(1,N);
            R=diagx(rr);
        end
        
        x=H;
    
        #Prediction
        beta10=MUx+beta11*F';
        p10=F*p11*F'+Q;
        yhat=(x*(beta10)')';
    
        eta = dataid0[i,:]' .- yhat;
        feta=(x*p10*x')+R;
    
        #updating
        K=(p10*x')*invpd(feta);
        beta11=(beta10'.+ K*eta')';
        p11=p10-K*(x*p10);
        ptt[i,:,:]=p11;
        beta_tt[i,:]=beta11;
    end

    # Backward recursion to calculate the mean and variance of the distribution of the state
    #vector
    beta2 = zeros(T,ns);   #this will hold the draw of the state variable
    bm2=beta2;
    jv=1:2; #index of non singular block
    jv1=[1 3 5]; #state variables to draw, 3, 5 are lagged states

    jv = 1:1:N
    jv1 = 1:N:N*L

    wa = randn(T,ns);

    i = T;  #period t
    p00 = ptt[i,jv1,jv1]; 
    beta2[i,:] = beta_tt[i,:];
    # Define distribution for the coefficients
    #   Make sure that the matrix is hermitian for cholesky factorization
    σ_var = Matrix(Hermitian(p00.+ eye(size(p00)[1]) .* 1e-10))

    dist_beta2 = MvNormal(vec(beta_tt[i:i,jv1]), σ_var)
    beta2[i,jv1]=rand(dist_beta2,1);
    #beta_tt(i:i,jv1)+(wa(i:i,jv1)*cholx(p00));   #draw for beta in period t from N(beta_tt,ptt)
    q=Q[jv,jv];
    mu=MUx[jv];
    f=F[jv,:];
    #periods t-1..to .1
    for i=T-1:-1:1
        # Select iteration
        pt_temp = ptt[i,:,:];

        bm = beta_tt[i,:]' + (pt_temp*f'*invpd(f*pt_temp*f'.+q)*(beta2[i+1,jv]'-(mu' .- beta_tt[i,:]'*f'))')';  
        pm = pt_temp - pt_temp*f'*invpd(f*pt_temp*f'+q)*f*pt_temp;  
        beta2[i,:] = bm;
        # Distribution draw
        #   following make sure that the matrix is computationally positive definite:
        #σ_var = pm[jv1,jv1]
        σ_var = Matrix(Hermitian(pm[jv1,jv1] .+ eye(size(pm[jv1,jv1])[1]) .* 1e-10)) 
        #σ_var = PDMat(pm[jv1,jv1])
        dist_beta2 = MvNormal(bm[jv1], σ_var)
        beta2[i,jv1]=rand(dist_beta2, 1);
        #bm(jv1)+(wa(i:i,jv1)*cholx(pm(jv1,jv1)));  
        bm2[i,:]=bm;
    end

    out = beta2[:,1]; #draw of monthly data

    # WHY IS THIS STEP EXIST?
        # This is causing a problem of fitting of weird values,
        # proposal steps are not stable or stationary -  covariance matrix of Σ not hermitian
    datax=hcat(out, dataM);
    Y=datax;
    X=prepare(Y,L);
    Y=Y[L+1:end,:];
    X=X[L+1:end,:];

    #disp(sprintf('Iteration Number= %s ', num2str(gibbs)));
    # After the burnout
    if gibbs >= BURN
        global dmat[gibbs1, :]=out;
        global bmat[gibbs1,:]=varcoef;
        global smat[gibbs1,:,:]=sigma;
        global gibbs1 = gibbs1 + 1;  
    end
    println(gibbs)
end


yfit = DataFrame(y = Y[:,1], yhat = vec(median(dmat, dims = 1)), time = 1:1:T)
# Gadfly plotting
#plot(stack(yfit, [:y, :yhat]), x = :time, y = :value, color= :variable, Geom.line)


g = Axis([
        Plots.Linear(yfit[:, :time], yfit[:, :y], style="no marks, black, thick", legendentry = "True value"),
        Plots.Linear(yfit[:, :time], yfit[:, :yhat], style="no marks, orange", legendentry = "Fitted - Median"),
    ], 
    title = "Fitness of Mixed Frequency VAR Model",
    legendPos="north west"
)
save("fig_mfvar_fitness.svg", g);

# Posterior density plot