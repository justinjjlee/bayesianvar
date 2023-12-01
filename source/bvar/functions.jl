# Common functions to be used for VAR model

eye(n) = Matrix{Float64}(I, n, n)
chol(mat) = convert(Array{Float64}, cholesky(Hermitian(mat)).U');
# For creating diagonal matrix;
# Nuance in julia calculation 
# https://web.eecs.umich.edu/~fessler/course/551/julia/tutor/03-diag.html
diagx(x) = diagm(vec(x))

function prior_Minn(nlags, Sstar; λ₀ = 1.0e9, λ₁ = 1.0, λ₃ = 100.0)
    # Prior matrix build for var-covar in error term
    #   Invers-wishart distribution with Minnesota prior shirnkage
    
    v1 = 1:1:nlags
    v1 = v1' # to change into multi-Array
    v1 = v1' .^ (-2.0*λ₁) # need to be in Float, not integer for type-stable.;
    v2 = 1 ./ diagx(Sstar)
    v3 = kron(v1, v2);
    #v3 = (λ₀^2.0)*[v3; (λ₃^2.0)];
    #v3 = 1 ./ v3; 
    Mtildeinv = diagx(v3); # vectorize for type-stable.;
    # Calculate inverse of Mtilde for standard Minnesota prior
    return Mtildeinv
end

function lagAR(X,p)
    # Autoregressive lag 
    #   This function creates lags of the matrix X(t), in the form:
    #            Xlag = [X(t-1),...,X(t-p)]
    #   Written by Dimitris Korobilis, March 2007
    Traw, N = size(X);
    Xlag = zeros(Traw, N*p);
    for ii=1:p
        Xlag[p+1:Traw,(N*(ii-1)+1):N*ii] = X[p+1-ii:Traw-ii, 1:N];
    end
    return Xlag
end

function mlag2(X,p)
    # Creation of lag folder
    try
        global Traw, N = size(X)
    catch # Vector
        global Traw = length(X)
        global N = 1
    end
    Xlag = zeros(Traw, N*p)
    for ii = 1:p
        Xlag[p+1:Traw,(N*(ii-1)+1):N*ii] .= X[p+1-ii:Traw-ii,1:N];
    end
    return Xlag
end

function wish(h, n)
    # Command:  s=wish(h,n)
    # Purpose:  Draws an m x m matrix from a wishart distribution
    #           with scale matrix h and degrees of freedom nu = n.
    #           This procedure uses Bartlett's decomposition.
    # Inputs:   h     -- m x m scale matrix.
    #           n     -- scalar degrees of freedom.
    # Outputs:  s     -- m x m matrix draw from the wishart
    #                    distribution.
    # Note: Parameterized so that mean is n*h

    A = chol(h)'*randn(size(h,1),n);
    return A*A'
end

@kwdef mutable struct bvar_params
    constant::Bool # true: if you desire intercepts, false: otherwise 
    p::Int64               # Number of lags on dependent variables
    differenced::Bool # Is the data differenced?

    forecasting::Bool     # true: Compute h-step ahead predictions, false: no prediction
    quantile_ci::Float64 # confidence interval quantile range

    forecast_method::Int64 # 0: Direct forecasts/1: Iterated forecasts

    repfor::Int64         # Number of times to obtain a draw from the predictive 
                        # density, for each generated draw of the parameters                     

    h::Int64               # Number of forecast periods

    impulses::Bool       # true: compute impulse responses, false: no impulse responses
    ihor::Int64           # Horizon to compute impulse responses

    # Set prior for BVAR model:
    prior::Int64  # prior = 1 --> Indepependent Normal-Whishart Prior
                # prior = 2 --> Indepependent Minnesota-Whishart Prior

    # Gibbs-related preliminaries
    nsave::Int64         # Final number of draws to save
    nburn::Int64         # Draws to discard (burn-in)
end

function bvar_base(Yraw, params)
    # unpack parameters
    constant = params.constant     # true: if you desire intercepts, false: otherwise 
    p = params.p               # Number of lags on dependent variables
    differenced = params.differenced # Is the data differenced?
    forecasting = params.forecasting # true: Compute h-step ahead predictions, false: no prediction
    quantile_ci = params.quantile_ci # confidence interval quantile range
    forecast_method = params.forecast_method # 0: Direct forecasts/1: Iterated forecasts
    repfor = params.repfor        # Number of times to obtain a draw from the predictive 
                        #  density, for each generated draw of the parameters
    h = params.h              # Number of forecast periods
    impulses = params.impulses     # true: compute impulse responses, false: no impulse responses
    ihor = params.ihor           # Horizon to compute impulse responses
    # Set prior for BVAR model:
    prior = params.prior           # prior = 1 --> Indepependent Normal-Whishart Prior
                        # prior = 2 --> Indepependent Minnesota-Whishart Prior
    # Gibbs-related preliminaries
    nsave = params.nsave         # Final number of draws to save
    nburn = params.nburn        # Draws to discard (burn-in)
    # Code translation from MATLAB to julia
    # VAR using the Gibbs sampler, based on independent Normal Wishar prior
    #
    #   Translator - Justin Lee (GITHUB)
    #--------------------------------------------------------------------------
    # Bayesian estimation, prediction and impulse response analysis in VAR
    # models using the Gibbs sampler. Dependent on your choice of forecasting,
    # the VAR model is:
    #
    # Iterated forecasts:
    #     Y(t) = A0 + Y(t-1) x A1 + ... + Y(t-p) x Ap + e(t)
    #
    # so that in this case there are p lags of Y (from 1 to p).
    #
    # Direct h-step ahead foreacsts:
    #     Y(t+h) = A0 + Y(t) x A1 + ... + Y(t-p+1) x Ap + e(t+h)
    #
    # so that in this case there are also p lags of Y (from 0 to p-1).
    #
    # In any of the two cases, the model is written as:
    #
    #                   Y(t) = X(t) x A + e(t)
    #
    # where e(t) ~ N(0,SIGMA), and A summarizes all parameters. Note that we
    # also use the vector a which is defined as a=vec(A).
    #--------------------------------------------------------------------------
    # NOTES: The code sacrifices efficiency for clarity. It follows the
    #        theoretical equations in the monograph and the manual.
    #
    # AUTHORS: Gary Koop and Dimitris Korobilis
    # CONTACT: dikorombilis@yahoo.gr
    #--------------------------------------------------------------------------

    
    #--------------------------DATA HANDLING-----------------------------------
    # Get initial dimensions of dependent variable
    Traw, M = size(Yraw);
    # Other set ups
    ntot = nsave + nburn;     # Total number of draws
    it_print = floor(ntot/10) # Print on the screen every "it_print"-th iteration

    # The model specification is different when implementing direct forecasts,
    # compared to the specification when computing iterated forecasts.
    if forecasting
        if h<=0
            error("You have set forecasting, but the forecast horizon choice is wrong")
        end    

        # Now create VAR specification according to forecast method
        if forecast_method==0       # Direct forecasts
            Y1 = Yraw[h+1:end,:];
            Y2 = Yraw[2:end-h,:];
            Traw = Traw - h - 1;
        elseif forecast_method == 1   # Iterated forecasts
            Y1 = Yraw;
            Y2 = Yraw;
        else
            error("Wrong choice of forecast_method")
        end
    else
        Y1 = Yraw;
        Y2 = Yraw;
    end

    # Generate lagged Y matrix. This will be part of the X matrix
    Ylag = mlag2(Y2,p); # Y is [T x M]. ylag is [T x (Mp)]

    # Now define matrix X which has all the R.H.S. variables 
    # (constant, lags of the dependent variable and exogenous regressors/dummies)
    if constant
        X1 = [ones(Traw-p,1) Ylag[p+1:Traw,:]];
    else
        X1 = Ylag[p+1:Traw,:];  #ok<UNRCH>
    end

    # Get size of final matrix X
    Traw3, K = size(X1);

    # Create the block diagonal matrix Z
    Z1 = kron(eye(M),X1);

    # Form Y matrix accordingly
    # Delete first "LAGS" rows to match the dimensions of X matrix
    Y1 = Matrix(Y1[p+1:Traw,:]); # This is the final Y matrix used for the VAR

    # Traw was the dimesnion of the initial data. T is the number of actual 
    # time series observations of Y and X (we lose the p-lags)
    T = Traw - p;

    # ========= FORECASTING SET-UP:
    # Now keep also the last "h" or 1 observations to evaluate (pseudo-)forecasts
    if forecasting
        Y_pred = zeros(nsave*repfor,M); # Matrix to save prediction draws
        PL =zeros(nsave,1);             # Matrix to save Predictive Likelihood
        
        if forecast_method==0  # Direct forecasts, we only need to keep the 
            Y = Y1[1:end-1,:]; #    last observation
            X = X1[1:end-1,:];
            Z = kron(eye(M),X);
            T = T - 1;
        else # Iterated forecasts, we keep the last h observations
            #Y = Y1[1:end-h,:];
            #X = X1[1:end-h,:];
            #Z = kron(eye(M),X);
            #T = T - h;
            # Forecast to be steps ahead
            Y = Y1;
            X = X1;
            Z = Z1;
        end
    else
        Y = Y1;
        X = X1;
        Z = Z1;
    end

    # ========= IMPULSE RESPONSES SET-UP:
    # Create matrices to store forecasts
    if impulses
        # Impulse response horizon
        imp = zeros(nsave,M,ihor);
        bigj = zeros(M,M*p);
        bigj[1:M,1:M] = eye(M);
    end

    #-----------------------------PRELIMINARIES--------------------------------
    # First get ML estimators
    A_OLS = inv(X'*X)*(X'*Y); # This is the matrix of regression coefficients
    a_OLS = A_OLS[:];         # This is the vector of parameters, i.e. it holds
                            # that a_OLS = vec(A_OLS)
    SSE = (Y - X*A_OLS)'*(Y - X*A_OLS);   # Sum of squared errors
    SIGMA_OLS = SSE./(T-K+1);

    # Initialize Bayesian posterior parameters using OLS values
    alpha = a_OLS;     # This is the single draw from the posterior of alpha
    ALPHA = A_OLS;     # This is the single draw from the posterior of ALPHA
    SSE_Gibbs = SSE;   # This is the single draw from the posterior of SSE
    SIGMA = SIGMA_OLS; # This is the single draw from the posterior of SIGMA

    # Storage space for posterior draws
    alpha_draws = zeros(nsave,K*M);
    ALPHA_draws = zeros(nsave,K,M);
    SIGMA_draws = zeros(nsave,M,M);
    # drawing of results
    irf_draws = zeros(nsave,M,M,ihor);
    # predictor (This is redundant of above, but for simplicity of QC-ing)
    ypred_draws = zeros(nsave, repfor, M);
    logp_draws = zeros(nsave, 1);

    sigma_sq = zeros(M,1); # vector to store residual variances
    for i = 1:M
        # Create lags of dependent variable in i-th equation
        Ylag_i = mlag2(Yraw[:,i],p);
        Ylag_i = Ylag_i[p+1:Traw,:];
        # Dependent variable in i-th equation
        Y_i = Yraw[p+1:Traw,i];
        # OLS estimates of i-th equation
        alpha_i = inv(Ylag_i'*Ylag_i)*(Ylag_i'*Y_i);
        sigma_sq[i,1] = (1 ./(Traw-p+1))*(Y_i - Ylag_i*alpha_i)'*(Y_i - Ylag_i*alpha_i);
    end
    #-----------------Prior hyperparameters for bvar model
    n = K*M; # Total number of parameters (size of vector alpha)
    # Define hyperparameters
    if prior == 1 # Normal-Wishart
        a_prior = 0*ones(n,1);   #<---- prior mean of alpha (parameter vector)
        V_prior = 10*eye(n);     #<---- prior variance of alpha
        
        # Hyperparameters on inv(SIGMA) ~ W(v_prior,inv(S_prior))
        v_prior = M;             #<---- prior Degrees of Freedom (DoF) of SIGMA
        S_prior = eye(M);            #<---- prior scale of SIGMA
        inv_S_prior = inv(S_prior);
    elseif prior == 2 # Minnesota-Whishart
        # Prior mean on VAR regression coefficients
        A_prior = [zeros(1,M); 0.9*eye(M); zeros((p-1)*M,M)];  #<---- prior mean of ALPHA (parameter matrix) 
        a_prior = A_prior[:];               #<---- prior mean of alpha (parameter vector)
        
        # Minnesota Variance on VAR regression coefficients
        # First define the hyperparameters 'a_bar_i'
        a_bar_1 = 0.5;
        a_bar_2 = 0.5;
        a_bar_3 = 10^2;
        
        # Now get residual variances of univariate p-lag autoregressions. Here
        # we just run the AR(p) model on each equation, ignoring the constant
        # and exogenous variables (if they have been specified for the original
        # VAR model)
        sigma_sq = zeros(M,1); # vector to store residual variances
        for i = 1:M
            # Create lags of dependent variable in i-th equation
            Ylag_i = mlag2(Yraw[:,i],p);
            Ylag_i = Ylag_i[p+1:Traw,:];
            # Dependent variable in i-th equation
            Y_i = Yraw[p+1:Traw,i];
            # OLS estimates of i-th equation
            alpha_i = inv(Ylag_i'*Ylag_i)*(Ylag_i'*Y_i);
            sigma_sq[i,1] = (1 ./(Traw-p+1))*(Y_i - Ylag_i*alpha_i)'*(Y_i - Ylag_i*alpha_i);
        end
        
        # Now define prior hyperparameters.
        # Create an array of dimensions K x M, which will contain the K diagonal
        # elements of the covariance matrix, in each of the M equations.
        V_i = zeros(K,M);
        
        # index in each equation which are the own lags
        ind = zeros(M,p);
        for i=1:M
            ind[i,:] = constant+i:M:K;
        end
        for i = 1:M  # for each i-th equation
            ll = 1 # Save for global local variable
            for j = 1:K   # for each j-th RHS variable
                if constant # if there is constant in the model use this code:
                    if j==1
                        V_i[j,i] = a_bar_3*sigma_sq[i,1]; # variance on constant                
                    elseif length(findall(==(j), ind[i,:])) > 0
                        V_i[j,i] = a_bar_1./(p^2); # variance on own lags           
                    else
                        for kj=1:M
                            if length(findall(==(j), ind[kj,:])) > 0
                                ll = kj;            
                            end
                        end                 # variance on other lags   
                        V_i[j,i] = (a_bar_2*sigma_sq[i,1])./((p^2)*sigma_sq[ll,1]);      
                    end
                else  # if there is no constant in the model use this:
                    if length(findall(==(j), ind[i,:])) > 0
                        V_i[j,i] = a_bar_1./(p^2); # variance on own lags
                    else
                        for kj=1:M
                            if length(findall(==(j), ind[kj,:])) > 0
                                ll = kj;
                            end                        
                        end                 # variance on other lags  
                        V_i[j,i] = (a_bar_2*sigma_sq[i,1])./((p^2)*sigma_sq[ll,1]);            
                    end
                end
            end
        end
        
        # Now V is a diagonal matrix with diagonal elements the V_i
        #   Function altered for my own version
        V_prior = diagx(V_i[:]);  # this is the prior variance of the vector alpha
        
        # Hyperparameters on inv(SIGMA) ~ W(v_prior,inv(S_prior))
        v_prior = M;
        S_prior = eye(M);
        inv_S_prior = inv(S_prior);   
    end

    # Note this is a control flow everything needs to be defined within
    #@btime begin
    #    print("blab")
    #end

    # Use the commented out print function if the ProgressMeter.jl package is not available
    #print("Number of iterations")
    @showprogress for irep = 1:ntot  #Start the Gibbs "loop"
        #if mod(irep,it_print) == 0
        #    print(irep); # Print of number of iterations
        #end
        
        # Posterior density estimation, update the latest assumptions
        VARIANCE = kron(inv(SIGMA),eye(T));
        V_post = inv(inv(V_prior) + Z'*VARIANCE*Z);
        a_post = V_post*(inv(V_prior)*a_prior + Z'*VARIANCE*Y[:]);
        alpha = a_post + 4*chol(V_post)*randn(n,1); # Draw of alpha
        
        ALPHA = reshape(alpha,K,M); # Draw of ALPHA
        
        # Posterior of SIGMA|ALPHA,Data ~ iW(inv(S_post),v_post)
        v_post = T + v_prior;
        S_post = S_prior + (Y - X*ALPHA)'*(Y - X*ALPHA);
        SIGMA = inv(wish(inv(S_post),v_post));# Draw SIGMA

        # Store results  
        if irep > nburn               
            # =========FORECASTING:
            if forecasting
                if forecast_method == 0   # Direct forecasts
                    Y_temp = zeros(repfor,M);
                    # compute 'repfor' predictions for each draw of ALPHA and SIGMA
                    for ii = 1:repfor
                        X_fore = [1 Y[T,:]' X[T,2:M*(p-1)+1]'];
                        # Forecast of T+1 conditional on data at time T
                        # NOTE: confirm if this is supposed to be lower or upper bound
                        Y_temp[ii,:] = X_fore*ALPHA + randn(1,M)*chol(SIGMA)';
                    end
                    # Matrix of predictions
                    Y_pred[((irep-nburn)-1)*repfor+1:(irep-nburn)*repfor,:] = Y_temp;
                    ypred_draws[irep-nburn, :,:] = Y_temp;
                    # Predictive likelihood
                    #   julia has default vectorizing, so matrix multiplication
                    #       requres
                    p_mu_hat = vec(X[T,:]'*ALPHA)
                    logp_iter = pdf(MvNormal(p_mu_hat, Hermitian(SIGMA)), Y1[T+1,:])
                    PL[irep-nburn,:] .= logp_iter
                    logp_draws[irep-nburn,:] .= logp_iter
                    
                    if PL[irep-nburn,:] == 0
                        PL[irep-nburn,:] = 1;
                        logp_draws[irep-nburn,:] = 1;
                    end
                elseif forecast_method == 1   # Iterated forecasts
                    # The usual way is to write the VAR(p) model in companion
                    # form, i.e. as VAR(1) model in order to estimate the
                    # h-step ahead forecasts directly (this is similar to the 
                    # code we use below to obtain impulse responses). Here we 
                    # just iterate over h periods, obtaining the forecasts at 
                    # T+1, T+2, ..., T+h iteratively.
                    Y_temp2 = zeros(repfor,M);
                    # Save dummy to be used outside of for-loop
                    X_new_temp = 0
                    for ii = 1:repfor
                        # Forecast of T+1 conditional on data at time T
                        X_fore = [1 Y[T,:]' X[T,2:M*(p-1)+1]'];
                        Y_hat = X_fore*ALPHA + randn(1,M)*chol(SIGMA)';
                        Y_temp = Y_hat;
                        X_new_temp = X_fore;
                        for i = 1:h-1  # Predict T+2, T+3 until T+h                   
                            if i <= p
                                # Create matrix of dependent variables for                       
                                # predictions. Dependent on the horizon, use the previous                       
                                # forecasts to create the new right-hand side variables
                                # which is used to evaluate the next forecast.                       
                                X_new_temp = [1 Y_hat X_new_temp[:,2:M*(p-i)+1]]; #--------------------------------- This changed from X_fore to X_new_temp
                                # This gives the forecast T+i for i=1,..,p                       
                                Y_temp = X_new_temp*ALPHA + randn(1,M)*chol(SIGMA)';                       
                                Y_hat = [Y_hat Y_temp];
                            else
                                X_new_temp = vcat(1, Y_hat[:,1:M*p]')'
                                Y_temp = X_new_temp*ALPHA + randn(1,M)*chol(SIGMA)';
                                Y_hat = [Y_hat Y_temp];
                            end
                        end #  the last value of 'Y_temp' is the prediction T+h
                        Y_temp2[ii,:] = Y_temp;
                    end
                    # Matrix of predictions               
                    Y_pred[((irep-nburn)-1)*repfor+1:(irep-nburn)*repfor,:] = Y_temp2;
                    ypred_draws[irep-nburn, :,:] = Y_temp2;
                    # Predictive likelihood
                    p_mu_hat = vec(X_new_temp*ALPHA)
                    #logp_iter = pdf(MvNormal(p_mu_hat, Hermitian(SIGMA)), Y1[T+1,:])
                    logp_iter = pdf(MvNormal(p_mu_hat, Hermitian(SIGMA)), Y1[end,:])
                    PL[irep-nburn,:] .= logp_iter
                    logp_draws[irep-nburn,:] .= logp_iter

                    if PL[irep-nburn,:] == 0
                        PL[irep-nburn,:] = 1;
                        logp_draws[irep-nburn,:] = 1;
                    end
                end
            end # end forecasting
            # =========Forecasting ends here
            
            # =========IMPULSE RESPONSES:
            if impulses==1
                biga = zeros(M*p,M*p);
                for j = 1:p-1
                    biga[j*M+1:M*(j+1),M*(j-1)+1:j*M] = eye(M);
                end
                
                atemp = ALPHA[2:end,:];
                atemp = atemp[:];
                splace = 0;
                for ii = 1:p
                    for iii = 1:M
                        biga[iii,(ii-1)*M+1:ii*M] = atemp[splace+1:splace+M,1]';
                        splace = splace + M;
                    end                
                end
                
                # St dev matrix for structural VAR
                #   Its multiplication of identity matrix, so should be ok with 
                #   transpose or not.
                STCO = chol(SIGMA);
                
                # Now get impulse responses for 1 through nhor future periods
                impresp = zeros(M,M*ihor);
                impresp[1:M,1:M] = STCO;
                bigai = biga;
                # 3-D savings
                irf_3d_iter = zeros(M, M, ihor)
                irf_3d_iter[:,:,1] = STCO;
                for j = 1:ihor-1
                    # Calculate j-horizon response
                    irt_iter_hor = bigj*bigai*bigj'*STCO
                    # Save and stack variable (complicated)
                    impresp[:,j*M+1:(j+1)*M] = irt_iter_hor;
                    bigai = bigai*biga;
                    # Save 3-D (simple)
                    irf_3d_iter[:,:,j+1] = irt_iter_hor;
                end
                
                # Save irf in 3D
                irf_draws[irep-nburn,:,:,:] = irf_3d_iter

                # the original to save the result ............
                #=
                # Get the responses of all M variables to a shock imposed on
                # the 'equatN'- th equation:
                equatN = M; #this assumes that the interest rate is sorted last in Y
                impf_m = zeros(M,ihor);
                jj=0;
                for ij = 1:ihor
                    jj = jj + equatN;
                    impf_m[:,ij] = impresp[:,jj];
                end

                # Save (don't need to - index is complicated)
                #imp[irep-nburn,:,:] = impf_m;
                =#
            end
                
            #----- Save draws of the parameters
            alpha_draws[irep-nburn,:] = alpha;
            ALPHA_draws[irep-nburn,:,:] = ALPHA;
            SIGMA_draws[irep-nburn,:,:] = SIGMA;

        end # end saving results
    end #end the main Gibbs for loop



    # Return values
    return ALPHA_draws, SIGMA_draws, ypred_draws, irf_draws, X, Y
end