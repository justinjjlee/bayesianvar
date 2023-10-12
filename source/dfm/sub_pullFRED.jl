# Need FREDData.jl and API connection
# NOTE: FRED API CONNECTION NOTE
# https://research.stlouisfed.org/docs/api/fred/series_observations.html#units
using FredData
# Set working directory
cd(@__DIR__)

# Data location in application
str_dir_git = splitdir(splitdir(pwd())[1])[1]

# Import credential locally for FRED & BLS API
apikeys = JSON.parsefile(splitdir(pwd())[1]*"/credential.json")
str_api = apikeys["api_key_fred"]

date_start = "2005-03-01";
date_end = "2019-12-31";

### Pull data from FRED - FED ST.Louis
# Economic policy uncertainty
unc_epu = get_data(str_api, "USEPUINDXD",
                    observation_start = date_start,
                    observation_end = date_end,
                    frequency = "m",
                    units = "lin",
                    aggregation_method = "avg"
                    );
# Indix, Case-Shiller home price index
inx_hpi = get_data(str_api, "CSUSHPISA",
                    observation_start = date_start,
                    observation_end = date_end,
                    frequency = "m",
                    units = "pc1",
                    aggregation_method = "avg"
                    );
# COnsumer confidence - aggregate
inx_con = get_data(str_api, "UMCSENT",
                    observation_start = date_start,
                    observation_end = date_end,
                    frequency = "m",
                    units = "lin",
                    aggregation_method = "avg"
                    );
# Construction - building permit
num_per = get_data(str_api, "PERMIT",
                    observation_start = date_start,
                    observation_end = date_end,
                    frequency = "m",
                    units = "ch1",
                    aggregation_method = "avg"
                    );
# Construction - construction starts
num_str = get_data(str_api, "HOUST",
                    observation_start = date_start,
                    observation_end = date_end,
                    frequency = "m",
                    units = "ch1",
                    aggregation_method = "avg"
                    );
# Monthly house supply
num_com = get_data(str_api, "MSACSR",
                    observation_start = date_start,
                    observation_end = date_end,
                    frequency = "m",
                    units = "lin",
                    aggregation_method = "avg"
                    );
# Construction spend - residential
spnd_co = get_data(str_api, "PRRESCONS",
                    observation_start = date_start,
                    observation_end = date_end,
                    frequency = "m",
                    units = "ch1",
                    aggregation_method = "avg"
                    );
# Stock marekt index
inx_nsq = get_data(str_api, "NASDAQCOM",
                    observation_start = date_start,
                    observation_end = date_end,
                    frequency = "m",
                    units = "ch1",
                    aggregation_method = "avg"
                    );
# TED spread
inx_ted = get_data(str_api, "TEDRATE",
                    observation_start = date_start,
                    observation_end = date_end,
                    frequency = "m",
                    units = "ch1",
                    aggregation_method = "avg"
                    );
# Unemployment data
num_une = get_data(str_api, "UNRATE",
                    observation_start = date_start,
                    observation_end = date_end,
                    frequency = "m",
                    units = "pc1",
                    aggregation_method = "avg"
                    );
# Unemployment data
num_clm = get_data(str_api, "CCSA",
                    observation_start = date_start,
                    observation_end = date_end,
                    frequency = "m",
                    units = "pc1",
                    aggregation_method = "avg"
                    );
# Unemployment data
num_ret = get_data(str_api, "RSEAS",
                observation_start = date_start,
                observation_end = date_end,
                frequency = "m",
                units = "pc1",
                aggregation_method = "avg"
                );
