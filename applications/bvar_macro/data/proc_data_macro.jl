# Data pull process macro
using Pkg
using FredData, JSON
using Dates, Statistics, Distributions
using CSV, DataFrames
# Warning, HTTP version has to be the latest
# ] add HTTP@1.10.1
# Set working directory
#cd(@__DIR__)

# Data location in application
str_dir = splitdir(splitdir(splitdir(splitdir(pwd())[1])[1])[1])[1]

# Import credential locally for FRED & BLS API
apikeys = JSON.parsefile(str_dir*"/credential.json")
api_key_fred = apikeys["credentials"]["api_key_fred"]

api_fred = Fred(api_key_fred);
# Specific date for mapping
date_start = "1985-01-01"
year_start = parse(Int64, date_start[1:4])

#date_last = "2019-12-01"; date_last_txt = "2019 December";
#year_end = parse(Int64, date_last[1:4])

#using HTTP.IOExtras
# ...........................................................................
# Weekly data ---------------------------------------------------------------
# NOTE: adjust dates to be week start Monday
# S&P 500 index
sap = FredData.get_data(api_fred, "SP500"; observation_start = "2020-11-05")
sap = sap.data[:,["date", "value"]]
rename!(sap, [:date, :sap])
# Drop missing dates (holidays)
sap = filter(row -> all(x -> !(x isa Number && isnan(x)), row), sap)
# Need to attach historic data
sap_hist = CSV.read(str_dir_git*"/applications/bvar_macro/data/SPX.csv", DataFrame)
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
epu = filter(row -> all(x -> !(x isa Number && isnan(x)), row), epu)
# Get start of date MOnday week date
epu[:,"date_wk"] = Dates.firstdayofweek.(epu.date)
# Aggregate to weekly frequency
epu_wk = combine(groupby(epu, :date_wk), :epu => mean => :epu)

# Market yield, either FFR or bond yield (co-move)
yield = FredData.get_data(api_fred, "DGS10"; observation_start = date_start)
yield = yield.data[:, ["date", "value"]]
rename!(yield, [:date, :yield])
# Drop missing dates (holidays)
yield = filter(row -> all(x -> !(x isa Number && isnan(x)), row), yield)
# Get start of date MOnday week date
yield[:,"date_wk"] = Dates.firstdayofweek.(yield.date)
# Aggregate to weekly frequency
yield_wk = combine(groupby(yield, :date_wk), :yield => mean => :yield)

# join the weekly data as dataframe and save in the application
dfwk = leftjoin(sap_wk, claims, on=:date_wk)
dfwk = leftjoin(dfwk, epu_wk, on=:date_wk)
dfwk = leftjoin(dfwk, yield_wk, on=:date_wk)
# Write to the dataframe
CSV.write(str_dir_git*"/applications/bvar_macro/data/df_wk.csv", dfwk)