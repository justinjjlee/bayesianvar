import Pkg;
using Statistics, DataFrame,
        CSV, Plots, FredData, MultivariateStats;
using ExcelReaders

cd(@__DIR__)
# NOTE: Need to check data transformation of sort
# PUll data
include("sub_pullFRED.jl");
# Compile functions
include("sub_functions.jl");
time = unc_epu.data[:, :date];
X = hcat(func_std(unc_epu.data[:, :value]),
        func_std(inx_hpi.data[:, :value]),
        func_std(inx_con.data[:, :value]),
        func_std(num_per.data[:, :value]),
        func_std(num_str.data[:, :value]),
        func_std(spnd_co.data[:, :value]),
        func_std(inx_nsq.data[:, :value]),
        func_std(inx_ted.data[:, :value]),
        func_std(num_une.data[:, :value]),
        func_std(num_com.data[:, :value]),
        func_std(num_clm.data[:, :value]),
        func_std(num_ret.data[:, :value])
    )

### Factor estimate
# Import the data - KPI
y = Matrix{Float64}(y)

# Go through the factor rotation process
f_null, f_fin, expl = func_dfm(X, y, 1)
plot(time, hcat(f_null, f_fin)) # CHeck

index_fin = -func_std(f_fin);

# Print out file
df_fin = DataFrame(date = time, index = index_fin[:])