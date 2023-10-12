using FredData, Quandl;
using CSV;
#=
# For more information regarding FRED API pull
# https://github.com/micahjsmith/FredData.jl
=#
time_latest = "2018-03-01";
#set ID for FRED-St.Louis Fed, use your own
id_fred = Fred("--");
# University of Michigan aggregate sentiment.
umich_index = get_data(id_fred, "UMCSENT";
                        observation_start = "1978-01-01",
                        observation_end = time_latest,
                        );
umich_index = umich_index.data;
umich_index = umich_index[:,[:date, :value]];
# Personal Consumption Expenditures - excl Food and Energy
# eventually want inflation.
pce_growth = get_data(id_fred, "PCEPILFE";
                        observation_start = "1978-01-01",
                        observation_end = time_latest,
                        units = "pc1");
pce_growth = pce_growth.data;
pce_growth = pce_growth[:,[:date, :value]];
##############
# Quandl data
##############
# This probably would not work anymore
set_auth_token("--");
sap_real = quandl("MULTPL/SP500_REAL_PRICE_MONTH", format = "DataFrame",
                from = "1978-01-01",
                to = time_latest
                );
sap_real[2:end, :Value] = 100 * (log.(sap_real[2:end, :Value]) -
                                 log.(sap_real[1:(end - 1), :Value]));

###############################################################################
# Aggregate data
data = hcat(
    convert(Array{Float64}, sap_real[:Value]),
    convert(Array{Float64}, pce_growth[:value]),
    convert(Array{Float64}, umich_index[:value])
    );
data = data[2:end, :];
# Standardize data
data = (data .- mean(data,1))./std(data,1);
###############################################################################
data_out = hcat(umich_index, pce_growth[:value], sap_real[:Value]);
data_out = data_out[2:end, :];
names!(data_out, [:time, :umich_index, :inf_pce, :sap_del_real]);
# write into file
CSV.write("./data/data_consumer.csv", data_out);
