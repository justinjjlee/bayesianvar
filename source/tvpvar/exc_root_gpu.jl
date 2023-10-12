using CUDAdrv, CUDAnative

function kernel_vadd(a, b, c)
    i = threadIdx().x
    c[i] = a[i] + b[i]
    return
end

# generate some data
len = 512
a = rand(Int, len)
b = rand(Int, len)

# allocate & upload to the GPU
d_a = CuArray(a)
d_b = CuArray(b)
d_c = similar(d_a)

# execute and fetch results
@cuda (1,len) kernel_vadd(d_a, d_b, d_c)
c = Array(d_c)
