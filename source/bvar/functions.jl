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