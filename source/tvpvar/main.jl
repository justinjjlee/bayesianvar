function main()
    # call by type following
    #=
    # for parallelizing
    @everywhere include("tvpvar.jl")
    throw = @spawn main();
    catch = fetch(throw);

    # or without one
    include('tvpvar.jl');
    main();
    =#
    #written by h.mumtaz. Errors/bug reports to h.mumtaz@qmul.ac.uk
    #see http://quant-econ.net/jl/getting_started.html to set up Julia environment and get started in Julia
    cd("--")
    #using CUDAnative, CUDAdrv; # GPU computation
    # Need to include the package separately, world age issue.
    #include("./function/functions.jl");
    #load data
    #data = readcsv("./data/usdata.csv")
    print("Initializing ... \n");
    N = size(data, 2); N = convert(Int64, N)
    NN = convert(Int64, N*(N-1)/2)

    REPS=50000; # number of replication
    KEEP=1000;  # number of iterations to keep
    BURN = REPS - KEEP; #burn-in's
    saveresults = true; #true to save results to results.jld
    maxdraws = 100;
    EX = 1;

    L = 2; # lag order
    KK = convert(Int64, N*(N*L+EX));
    Y, X = prepare(data, L);
    Y = Y[(L+1):end, :];
    X = X[(L+1):end, :];

    print("Setting priors ... \n");
    #priors
    T0 = 40; # training set size
    y0 = Y[(1:T0),:];
    x0 = X[(1:T0),:];
    b0 = x0 \ y0;
    e0 = y0 - (x0*b0);
    sigma0 = (e0'*e0)/T0;
    V0 = kron(sigma0, invpd(x0'*x0));

    #priors for Q
    Q0 = V0*T0*3.5e-04  #prior for the variance of the transition equation error
    P00 = V0; #variance of the intial state vector  variance of state variable p[t-1/t-1]
    beta0 = vec(b0)';       #intial state vector   %state variable  b[t-1/t-1]
    Q = Q0;
    #priors and starting values for aij
    C0 = chol(sigma0);

    C0 = C0 ./ repmat(diag(C0), 1, N);
    C0 = invpd(C0)';
    PC0 = 10.0;


    amatx = Array{Float64}(1,NN);
    D0 = Array{Any}(N-1);
    DD = Array{Any}(N-1);
    j=1;
    for i=1:(N-1)
        temp = C0[(i+1), (1:i)]
        temp_size = length(temp)
        #amatx[1,j:j+size(temp,2)-1]=temp
        amatx[1, (j:(j+temp_size-1))] = temp
        D0[i] = eye(temp_size) .* (0.001)
        DD[i] = eye(temp_size) .* (0.001)
        j = j + temp_size
    end

    print("Initiate particle Gibbs sampling ... \n");
    #remove initial sample
    Y = Y[(T0+1):end,:];
    X = X[(T0+1):end,:];
    T = size(X, 1);

    #priors for the initial condition of the stochastic volatility
    MU0 = log.(diag(sigma0)); #prior mean

    SS0 = 10.0;
    g0 = 0.1^2; #prior scale parameter for inverse gamma
    Tg0 = 1;    #prior degrees of freedom

    #starting values for svol
    hlast = diff(Y).^2 + 1e-4;
    hlast = [hlast[1:2, :]; hlast];
    b0 = X \ Y;
    e0 = Y - (X*b0);
    epsilon = e0 * C0; #starting values for orthogonolised resids
    errorx = e0;  #starting values for VAR residuals
    beta00 = repmat(beta0, T, 1);
    #50 iterations to smooth initial estimates of stochastic volatility
    for m = 1:50
      hnew = zeros((T+1), N)
      for i = 1:N
          htemp = getsvol(hlast[:,i], g0, MU0[i,1], SS0, epsilon[:,i]);
          hnew[:, i] = htemp;
      end
      hlast=hnew;
    end

    print("Gibbs sampling for partial equilibrium ... \n");
    #save
    bsave = Array{Float64}(KEEP,T,KK);
    asave = Array{Float64}(KEEP,T,NN);
    hsave = Array{Float64}(KEEP,T,N);
    esave = Array{Float64}(KEEP,T,N); # residual
    qsave = Array{Float64}(KEEP,KK,KK);
    gsave = Array{Float64}(KEEP,1,N);
    qasave = Array{Float64}(KEEP,NN,NN);

    p = ProgressMeter.Progress(REPS, 0.01);
    # for parallelizing purpose, variables that are getting editted:
    #hlast = convert(SharedArray{Float64}, hlast);
    #DD = convert(SharedArray{Any}, DD);

    # Start the Gibbs samplar
    for jgibbs = 1:REPS
        #println(tgibbs)
        ######Step 1 of the Gibbs Sampler: Sample g[i]

        g=getG(hlast,Tg0,g0,N)
        #####Step 2 of the Gibbs Sampler: Sample the SVOL
        hnew=hlast;

        hnew = SharedArray(hnew);
        #for j = 1:N
        @sync @parallel for j = 1:N
            temp_hlast = convert(Array{Float64}, hlast[:,j])

            # Convert to GPU array and spit back out
            #temp_hlast = CU
            htemp=getsvol(temp_hlast,g[1,j],MU0[j,1],SS0,epsilon[:,j])
            hnew[:,j]=htemp
        end
        hlast=convert(Array, hnew);

        #######Step 3 of the Gibbs sampler: Sample elements of the A matrix
        amat=getamat(NN,T,errorx,C0,PC0,DD,N,hlast)

        QA=zeros(NN,NN)
        j=1
        for i=2:N
            xtemp=-errorx[:,1:i-1]
            a1=amat[:,j:j+size(xtemp,2)-1];
            a1errors=diff(a1)
            scaleD=(a1errors'*a1errors)+D0[i-1]
            tmp=iwpq(T+size(a1,2)+1,invpd(scaleD))
            DD[i-1]=tmp
            QA[j:j+size(a1,2)-1,j:j+size(a1,2)-1]=tmp
            j=j+size(a1,2)
        end

        ###########Step 4 DRAW TVP coefficients


        beta_tt,ptt=kfilter(Y,X,Q,amat,hlast,beta0[:,:],P00,L)
        beta2,problem=carterkohn(beta_tt,ptt,maxdraws,N,L,Q)
        if problem==1
            beta2=beta00
        else
            beta00=beta2
        end


        #update residuals
        for j=1:T
            errorx[j,:]=Y[j,:]-reshape(beta2[j,:],N*L+1,N)'*X[j,:] #var residuals
            a=vec(amat[j,:])
            A=chofac(N,a)

            #epsilon[j,:]=errorx[j,:]*A'
            epsilon[j,:]=A'*errorx[j,:]
        end


        ###########Step 5: Draw Q

        errorq=diff(beta2)
        scaleQ=(errorq'*errorq)+Q0
        Q=iwpq(T+T0,invpd(scaleQ))

        #####################
        if jgibbs>BURN
            if problem == 0
                bsave[(jgibbs - BURN),:,:]=beta2
                asave[(jgibbs - BURN),:,:]=amat
                hsave[(jgibbs - BURN),:,:]=hlast[2:T+1,:]
                esave[(jgibbs - BURN),:,:]=errorx; # residual.
                qsave[(jgibbs - BURN),:,:]=Q
                gsave[(jgibbs - BURN),:,:]=g
                qasave[(jgibbs - BURN),:,:]=QA
            end

        end
        ProgressMeter.next!(p)
    end # end of Gibbs

    if saveresults
        save("results.jld", "bsave",bsave, "asave",asave, "hsave",hsave,
             "qsave",qsave,"gsave",gsave,"qasave",qasave, "esave", esave);
    end


end
