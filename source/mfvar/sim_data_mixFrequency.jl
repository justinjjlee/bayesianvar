# Following the Mumtaz code - translating MATLAB
function data_gen()
    ## generate artificial data 
    nobs=996; #996 months 332 quarters
    # Y / X (single vectors), with exogenous factor attached
    btrue=[0.95 0.1;
        0.1 0.95;
        -0.1 0;
        0 -0.1;
        -0.05  0;
        0 -0.05;
        0 0];
    
    sigmatrue=[2  1;
            1 2];
    Σchol = Matrix(cholesky(sigmatrue).U)

    datatrue=zeros(nobs,2);
    for j=4:nobs
        # Adding a lag term with coefficient multiplier, with randomness
        datatrue[j,:]= vcat(datatrue[j-1,:], 
                            datatrue[j-2,:], 
                            datatrue[j-3,:], 1)'*btrue .+ randn(1,2)*Σchol;
    end

    #assume first variable is subject to temporal aggregation
    #   What would be lower-frequency data
    dataQ=zeros(Int(nobs/3),1); # quarterly data Y
    jj=1;
    for j=1:3:nobs
        tmp=datatrue[j:j+2,1];
        dataQ[jj,1]=mean(tmp);
        jj=jj+1;
    end

    dataM = datatrue[:,2]; #monthly data X

    #arrange data
    #put missing observations
    nrows = size(dataQ)[1]

    dataN=hcat(Array{Union{Float64, Missing}}(missing,nrows,2), dataQ);  #puts NANs for missing obs
    # NOTE, for those users of MATLAB, the reshaping for MATLAB matrix is row-major,
    #   for julia, it is column major - thus, for julia, the original matrix requires transpose
    #       for vector reshaping.
    dataN=reshape(dataN',:,1)

    #same as above but zeros for missing
    data0=hcat(zeros(nrows,2), dataQ);
    data0=reshape(data0',:,1)

    #initial value of data just repeated observations
    dataX=repeat(dataQ, outer = (1, 3));
    dataX=reshape(dataX',:,1)


    data=hcat(dataX, dataM);
    dataid=hcat(dataN, dataM);
    dataid0=hcat(data0, dataM);

    mid=ismissing.(dataid);  #id for missing obs

    return data, dataid, dataid0, mid, dataM
end