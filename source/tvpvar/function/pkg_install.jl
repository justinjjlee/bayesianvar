Pkg.add.([
    "Cairo";
    "CSV";
    "CUDAnative";
    "CUDAdrv";
    "DataFrames";
    "Fontconfig";
    "FredData";
    "Gadfly";
    "JLD";
    "ProgressMeter";
    "StatsBase";
    "StatsFuns";
    "Quandl"
    ]);
Pkg.update();

Pkg.checkout("Knet");
Pkg.rm(["CUDAdrv"; "CUDAapi"]);

#=
Other notes
https://github.com/denizyuret/Knet.jl/issues/230
Need NVIDIA toolkit
=#
