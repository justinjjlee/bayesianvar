# Data pull process macro
using Pkg
using FredData, BlsData, JSON
using Dates, Statistics, Distributions
using CSV, DataFrames
# Set working directory
cd(@__DIR__)

# Data location in application
str_dir_git = splitdir(splitdir(pwd())[1])[1]

# Import credential locally for FRED & BLS API
apikeys = JSON.parsefile(splitdir(pwd())[1]*"/credential.json")
api_key_fred = apikeys["api_key_fred"]
api_key_bls = apikeys["api_key_bls"]

api_fred = Fred(api_key_fred);
# Specific date for mapping
date_start = "1985-01-01"
year_start = parse(Int64, date_start[1:4])

#date_last = "2019-12-01"; date_last_txt = "2019 December";
#year_end = parse(Int64, date_last[1:4])

#using FredApi
#set_api_key(api_key_fred)
#sap = get_symbols("SP500", "2020-11-05", "2023-10-09")

using HTTP.IOExtras
# ...........................................................................
# Weekly data ---------------------------------------------------------------
# NOTE: adjust dates to be week start Monday
# S&P 500 index
sap = FredData.get_data(api_fred, "SP500"; observation_start = "2020-11-05")
sap = sap.data[:,["date", "value"]]
rename!(sap, [:date, :sap])
# Drop missing dates (holidays)
sap = sap[.~isnan.(sap.sap), :]
# Need to attach historic data
sap_hist = CSV.read(str_dir_git*"/applications/bvar_macro/SPX.csv", DataFrame)
sap_hist = sap_hist[:, ["Date", "Close"]]
rename!(sap_hist, [:date, :sap])
sap_hist = sap_hist[sap_hist.date .>= Date("1984-12-31"), :] # Adust for dates available
append!(sap_hist, sap, cols = :orderequal)
# Get start of date MOnday week date
sap_hist[:, "date_wk"] = Dates.firstdayofweek.(sap_hist.date)
# Aggregate to weekly frequency
sap_wk = combine(groupby(sap_hist, :date_wk), :sap => mean => :sap)

# Remove missing dates
# Weekly jobless claim
claims = FredData.get_data(api_fred, "ICSA"; observation_start = date_start)
claims = claims.data[:, ["date", "value"]]
rename!(claims, [:date, :claims])
# make the date week starting Monday, the data is week ending Friday
claims.date = claims.date .- Day(5)
claims[:, "date_wk"] = Dates.firstdayofweek.(claims.date)
claims = claims[:,["date_wk", "claims"]]

# Economic policy uncertainty
epu = FredData.get_data(api_fred, "USEPUINDXD"; observation_start = date_start)
epu = epu.data[:,["date", "value"]]
rename!(epu, [:date, :epu])
# Drop missing dates (holidays)
epu = epu[.~isnan.(epu.epu), :]
# Get start of date MOnday week date
epu[:,"date_wk"] = Dates.firstdayofweek.(epu.date)
# Aggregate to weekly frequency
epu_wk = combine(groupby(epu, :date_wk), :epu => mean => :epu)

# Market yield, either FFR or bond yield (co-move)
yield = FredData.get_data(api_fred, "DGS10"; observation_start = date_start)
yield = yield.data[:, ["date", "value"]]
rename!(yield, [:date, :yield])
# Drop missing dates (holidays)
yield = yield[.~isnan.(yield.yield), :]
# Get start of date MOnday week date
yield[:,"date_wk"] = Dates.firstdayofweek.(yield.date)
# Aggregate to weekly frequency
yield_wk = combine(groupby(yield, :date_wk), :yield => mean => :yield)

# join the weekly data as dataframe and save in the application
dfwk = leftjoin(sap_wk, claims, on=:date_wk)
dfwk = leftjoin(dfwk, epu_wk, on=:date_wk)
dfwk = leftjoin(dfwk, yield_wk, on=:date_wk)
# Write to the dataframe
CSV.write(str_dir_git*"/applications/bvar_macro/df_wk.csv", dfwk)
# ...........................................................................
# Monthly data --------------------------------------------------------------

# Consumer confidence
belief = FredData.get_data(api_fred, "UMCSENT"; observation_start = date_start)
belief = belief.data[:, ["date", "value"]]
rename!(belief, [:date, :belief])
# PCE inflaiton
pceall = FredData.get_data(api_fred, "PCE"; observation_start = date_start, units="pc1")
pceall = pceall.data[:, ["date", "value"]]
rename!(pceall, [:date, :pce])

# unemployment rate
unemp = FredData.get_data(api_fred, "UNRATE"; observation_start = date_start)
unemp = unemp.data[:, ["date", "value"]]
rename!(unemp, [:date, :unrate])

# Short-term unemployment rate
suemp = FredData.get_data(api_fred, "LNS13008397"; observation_start = date_start)
suemp = suemp.data[:, ["date", "value"]]
rename!(suemp, [:date, :strate])

# Market yield, either FFR or bond yield (co-move)
yield = FredData.get_data(api_fred, "DGS10"; observation_start = date_start, frequency="m", aggregation_method="avg")
yield = yield.data[:, ["date", "value"]]
rename!(yield, [:date, :yield])

# join the monthly data as dataframe and save in the application
dfmo = leftjoin(yield, belief, on=:date)
dfmo = leftjoin(dfmo, pceall, on=:date)
dfmo = leftjoin(dfmo, unemp, on=:date)
dfmo = leftjoin(dfmo, suemp, on=:date)
# Write to the dataframe
CSV.write(str_dir_git*"/applications/bvar_macro/df_mo.csv", dfmo)

# Excit the application