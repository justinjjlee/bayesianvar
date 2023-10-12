using StatsBase, StatsFuns, DataFrames;
using ProgressMeter#, ParallelAccelerator;
using JLD; # PyPlot
############################################
#@acc begin # intel lab parallel accelerator
# https://julialang.org/blog/2016/03/parallelaccelerator
############################################
function lag0(x::Array{Float64,2},p::Int64)
    #copy arrays
  (R,C)=size(x)
  x1=x[1:(R-p),1:end]
  out=[zeros(p,C);x1]
  return out
end

function invpd(X::Array{Float64,2})
    N=size(X,2)
    out=X\eye(N)
    return out
end

function prepare(data::Array{Float64,2},p::Int64)
    T,N=size(data)
    Y=data
    X=zeros(T,N*p+1)
    i=1
    for j=1:p
        tmp=lag0(Y,j)
        X[:,i:i+N-1]=tmp
        i=i+N
    end
    X[:,end]=ones(T,1)
    return Y,X
end

function cholx(x::Array{Float64,2})
    if isposdef(x)
        out=chol(x)
    else
        out=sqrtm(x)
    end
    return out
end


function getsvol0(hlead::Float64,g::Float64,mubar::Float64,sigmabar::Float64)
    #time period 0
    ss = sigmabar*g/(g + sigmabar)  #variance
    mu = ss*(mubar/sigmabar + log(hlead)/g)  #mean
    #draw from lognormal  using mu and ss
    h = exp.(mu + sqrt(ss)*randn(1,1))
    return h
end


function getsvolt(hlead::Float64,hlag::Float64,g::Float64,yt::Float64,hlastx::Float64)
    mu = (log(hlead)+log(hlag))./2
    ss = g/2
    #candidate draw from lognormal
    htrial = exp(mu + sqrt(ss)*randn(1,1))
    #acceptance probability in logs
    lp1 = -0.5*log(htrial) - (yt^2)./(2*htrial)  #numerator
    lp0 = -0.5*log(hlastx) - (yt^2)./(2*hlastx)   #denominator
    accept = minimum([1;exp(lp1 - lp0)])  #ensure accept<=1
    u = rand(1)
    if u[1] <= accept[1]
        h = htrial
    else
        h = hlastx
    end
    hnew=h
    return hnew
end

 function getsvolT(hlag::Float64,g::Float64,yt::Float64,hlastx::Float64)
    mu = log(hlag)
    ss = g
    #candidate draw from lognormal
    htrial = exp(mu + sqrt(ss)*randn(1,1))
    #acceptance probability in logs
    lp1 = -0.5*log(htrial) - (yt^2)./(2*htrial)  #numerator
    lp0 = -0.5*log(hlastx) - (yt^2)./(2*hlastx)   #denominator
    accept = minimum([1;exp(lp1 - lp0)])  #ensure accept<=1
    u = rand(1)
    if u[1] <= accept[1]
        h = htrial
    else
        h = hlastx
    end
    hnew=h
    return hnew
end

function getsvolx(hlast::Array{Float64,1},g::Float64,mubar::Float64,sigmabar::Float64,errors::Array{Float64,1})
     T=size(errors,1)
     hnew=Array{Float64}(T+1)#zeros(T+1)

    i=1
    #time period 0
    hlead=hlast[i+1]
    h=getsvol0(hlead,g,mubar,sigmabar)

    hnew[i]=h[1,1]
    #time period 1:T
    for i=2:T
       hlead=hlast[i+1]
       hlag=hnew[i-1]
       yt=errors[i-1]  #note change
       h=getsvolt(hlead,hlag,g,yt,hlast[i])
       hnew[i]=h[1,1]
    end
    # time period t+1
    i=T+1
    hlag=hnew[i-1]
    yt=errors[i-1]  #note change
    h=getsvolT(hlag,g,yt,hlast[i])
    hnew[i]=h[1,1]

    return hnew
end







@everywhere function getsvol(hlast::Array{Float64,1},g::Float64,mubar::Float64,sigmabar::Float64,errors::Array{Float64,1})

    T=size(errors,1)
    hnew=Array{Float64}(T+1)#zeros(T+1)

    i=1
    #time period 0
    hlead=hlast[i+1,1]
    ss = sigmabar*g/(g + sigmabar)  #variance
    mu = ss*(mubar/sigmabar + log(hlead)/g)  #mean
    #draw from lognormal  using mu and ss
    h = exp.(mu + sqrt(ss)*randn(1,1))
    hnew[i]=h[1,1]
    #time period 1 to t-1
    for i=2:T
        hlead=hlast[i+1,1]
        hlag=hnew[i-1,1]
        yt=errors[i-1,1]  #note change
        #mean and variance of the proposal log normal density
        mu = (log(hlead[1,1])+log(hlag[1,1]))/2
        ss = g/2
        #candidate draw from lognormal
        htrial = exp.(mu + sqrt(ss)*randn(1,1))
        #acceptance probability in logs
        lp1 = -0.5*log(htrial[1,1]) - (yt^2)/(2*htrial[1,1])  #numerator
        lp0 = -0.5*log(hlast[i,1]) - (yt^2)/(2*hlast[i,1])   #denominator
        accept = minimum([1;exp(lp1 - lp0)])  #ensure accept<=1
        u = rand(1,1)
        if u[1,1] .<= accept[1,1]
            h = htrial
        else
            h = hlast[i,1]
        end
        hnew[i,1]=h[1,1]
    end


    #time period T
    i=T+1
    yt=errors[i-1,1]
    hlag=hnew[i-1,1]
    #mean and variance of the proposal density
    mu = log.(hlag)   # only have ht-1
    ss = g
    #candidate draw from lognormal
    htrial = exp.(mu + sqrt(ss)*randn(1,1))
    #acceptance probability
    lp1 = -0.5*log.(htrial[1,1]) - (yt^2)/(2*htrial[1,1])
    lp0 = -0.5*log.(hlast[i,1]) - (yt^2)/(2*hlast[i,1])
    accept = minimum([1;exp.(lp1 - lp0)])  #ensure accept<=1
    u = rand(1,1)
    if u[1,1] .<= accept[1,1]
        h = htrial
    else
        h = hlast[i,1]
    end
    hnew[i,1]=h[1,1]
    return hnew
end

###############
function ig( resids::Array{Float64,1},T0::Int64,D0::Float64 )
    #compute posterior df and scale matrix
    T=size(resids,1)
    T1=T0+T
    D1=D0+resids'*resids
    #draw from IG
    z0=randn(T1,1)
    z0z0=z0'*z0;
    sigma2=D1./z0z0
    return sigma2
end

######################################
function kf1(beta0::Array{Float64,2},p00x::Array{Float64,2},hlast::Array{Float64,1},Q::Array{Float64,2},Y::Array{Float64,1},X::Array{Float64,2})
    #Step 1 Set up matrices for the Kalman Filter
    ns=size(beta0,2)
    t=size(Y,1)
    beta_tt=Array{Float64}(t,ns)          #will hold the filtered state variable
    ptt=Array{Any}(t)   # will hold its variance
    #initialise the state variable
    beta11=beta0
    p11=p00x
    for i=1:t
        x=X[i,:]'; # change to 1, ~ dimension
        R=hlast[i+1,1]
        #Prediction
        beta10=beta11
        p10=p11+Q

        yhat=(x*(beta10)')' #yhat=(x*(beta10)')'
        eta=Y[i,1]-yhat
        feta=(x*p10*x')+R
        #updating
        K=(p10*x')*inv(feta)
        beta11=(beta10'+K*eta')'
        p11=p10-K*(x*p10)
        ptt[i]=p11
        beta_tt[i,:]=beta11
    end
    return beta_tt,ptt
end



function ckohn1(beta_tt::Array{Float64,2},ptt::Array{Any,1},Q::Array{Float64,2})
    t,ns=size(beta_tt)
    #Carter and Kohn Backward recursion to calculate the mean and variance of the distribution of the state vector
    beta2 = Array{Float64}(t,ns)#zeros(t,ns)   #this will hold the draw of the state variable
    i=t #
    p00=ptt[i]
    beta2[i,:]=beta_tt[i,:]'+(randn(1,ns)*chol(Hermitian(p00)))
    #periods t-1..to 1
    for i=t-1:-1:1
        pt=ptt[i]
        iFptF=inv(pt+Q)
        bm=beta_tt[i,:]'+(pt*iFptF*(beta2[i+1:i+1,:]-beta_tt[i,:]')')'
        pm=pt-pt*iFptF*pt
                #println(isposdef(pm))
        beta2[i,:]=bm+(randn(1,ns)*chol(Hermitian(pm)))
    end
    return beta2
end










####################
function  iwpq(v::Int64,ixpx::Array{Float64,2})
    k=size(ixpx,1)
    cx=chol(Hermitian(ixpx))'
    z=zeros(v,k)
    mu=zeros(k,1)
    for i=1:v
        z[i,:]=(cx*randn(k,1))';
    end
    out=invpd(z'*z)
    return out
end


#########################

function  chofac(N::Int64,chovec::Array{Float64,1});
#written by Tim Cogley
#CF = chofac(N,chovec);
#This transforms a vector of cholesky coefficients, chovec, into a lower
#triangular cholesky matrix, CF, of size N.
    CF = eye(N)
    i = 1
    for j = 2:N
       k = 1
       while k < j
                CF[j,k]= chovec[i]
          i = i+1
          k = k+1
       end
    end
    return CF
end

#######################

function stability(beta::Array{Float64,1},N::Int64,L::Int64,ex::Int64)

    FF=zeros(N*L,N*L)
    FF[N+1:N*L,1:N*(L-1)]=eye(N*(L-1))
    temp=reshape(beta,N*L+ex,N)
    FF[1:N,1:N*L]=temp[1:N*L,:]'
    E,V=eig(FF)
    ee=maximum(abs.(E))

    S=0
    if ee>1
        S=1
    end
        #println([S ee])
    return S
end










########################
function kfilter(Y::Array{Float64,2},X::Array{Float64,2},Q::Array{Float64,2},amat::Array{Float64,2},hlast::Array{Float64,2},beta0::Array{Float64,2},P00::Array{Float64,2},L::Int64)
    T=size(Y,1)
    N=size(Y,2)
    #Step 2a Set up matrices for the Kalman Filter
    ns=size(beta0,2)
    beta_tt=zeros(T,ns) #will hold the filtered state variable
    ptt=Array{Any}(T)   #will hold its variance
    beta11=beta0
    p11=P00
    #% %%%%%%%%%%%Step 2b run Kalman Filter

    for i=1:T
        x=kron(eye(N),X[i,:])';
        A=chofac(N,vec(amat[i,:]))
        iA=inv(A)
        H=diagm(vec(hlast[i+1,:]))
        R=iA*H*iA'
        #Prediction
        beta10=beta11
        p10=p11+Q
        yhat=(x*(beta10)')'
        eta=Y[i,:]-yhat'
        feta=(x*p10*x')+R
        #updating
        K=(p10*x')*inv(feta)
        beta11=(beta10'+K*eta)'
        p11=p10-K*(x*p10)
        ptt[i]=p11
        beta_tt[i,:]=beta11
    end
    return beta_tt,ptt
end

function carterkohn(beta_tt::Array{Float64,2},ptt::Array{Any,1},maxdraws::Int64,N::Int64,L::Int64,Q::Array{Float64,2})
T,ns=size(beta_tt)
beta2 = Array{Float64}(T,ns)#zeros(T,ns)  #this will hold the draw of the state variable
roots=zeros(T)
chck=-1
problem=0
trys=1
while chck<0 && trys<=maxdraws

    i=T  #period t
    p00=ptt[i]
    beta2[i,:]=beta_tt[i:i,:]+(randn(1,ns)*chol(Hermitian(p00)))   #draw for beta in period t from N(beta_tt,ptt)
    roots[i,1]=stability(vec(beta2[i,:]),N,L,1)

    #periods t-1..to .1

    for i=T-1:-1:1
        pt=ptt[i]
        iFptF=inv(pt+Q)
        bm=beta_tt[i:i,:]+(pt*iFptF*(beta2[i+1:i+1,:]-beta_tt[i,:]')')'  #update the filtered beta for information contained in beta[t+1]                                                                               %i.e. beta2(i+1:i+1,:) eq 8.16 pp193 in Kim Nelson
        pm=pt-pt*iFptF*pt #  %update covariance of beta
        beta2[i:i,:]=bm+(randn(1,ns)*chol(Hermitian(pm)))  #draw for beta in period t from N(bm,pm)eq 8.17 pp193 in Kim Nelson
        roots[i]=stability(vec(beta2[i,:]),N,L,1)
    end

    if sum(roots)==0
        chck=1
    else
        trys=trys+1
    end

end

if chck<0
problem=1
end


return beta2,problem
end


############################
function getG(hlast::Array{Float64,2},Tg0::Int64,g0::Float64,N::Int64)
g=zeros(1,N)
for j=1:N
    resids=diff(log.(hlast[:,j]))
    gg=ig( resids,Tg0,g0)
    g[1,j]=gg[1,1]
end
    return g
end



##############################
function getamat(NN::Int64,T::Int64,errorx::Array{Float64,2},C0::Array{Float64,2},PC0::Float64,DD::Array{Any,1},N::Int64,hlast::Array{Float64,2})

    amat=zeros(T,NN)
    j=1
    for i=2:N
        ytemp=errorx[:,i]
        xtemp=-errorx[:,1:i-1]
        a0=vec(C0[i,1:i-1])
        pa=PC0.*diagm(abs.(a0))
        Qa=DD[i-1]
        att,ppt=kf1(a0[:,:]',pa,hlast[:,i],Qa,ytemp,xtemp)
        a1=ckohn1(att,ppt,Qa)
        amat[:,j:j+size(a1,2)-1]=a1
        j=j+size(a1,2)
    end
    return amat
end



############################################
#end # intel lab parallel accelerator
############################################
