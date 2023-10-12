using DataFrames, JLD, ProgressMeter;
using Gadfly, Cairo, Fontconfig;
dark_panel = Theme(panel_fill="white", background_color = "white");
Gadfly.push_theme(dark_panel);

################################################################################
# Run the program
cd("--")
include("./function/functions.jl");
include("./exc_pulldata.jl");
include("main.jl");
#addprocs(7);
main();
################################################################################








res = load("results.jld");
#time = 1960+1/12:1/12:2017+12/12;
# or
ndraw,T,n = size(res["hsave"])[1:3];
L = 2; # lag order used.
α = 0.1; # confidence band parameter;
time = 1900+1/12:1/12:2018+3/12;
time = time[end-T+1:end];
#time = linspace(1960, 2017, T);
name_var = ["SAP"; "Inflation"; "Sentiment"];
##################
# Impulse response - cholesky
##################
# Organize contemporaneous matrix.
Amat = zeros(n,n,T,ndraw);
for t = 1:T
    for idraw = 1:ndraw
        temp_draw = res["asave"][idraw,t,:];
        Amat[:,:,t,idraw] = eye(n);
        temp_lvl = 1;
        temp_cng = 1;
        while temp_lvl-1 != length(temp_draw)
            temp_x = 1+temp_cng;
            temp_y = 1:temp_cng;
            temp_ind = temp_lvl:(temp_lvl+temp_cng-1);
            Amat[temp_x,temp_y,t,idraw] = temp_draw[temp_ind];
            temp_lvl = temp_lvl + temp_cng;
            temp_cng += 1;
        end
    end
end
# Organize coefficients
# NOTE: constant on last order, to calculate impulse response function,
#   we do not need it.
Bmat = zeros(n,(n*L),T,ndraw);
for t = 1:T
    for idraw = 1:ndraw
        temp_draw = res["bsave"][idraw,t,:];
        temp_draw = reshape(temp_draw, ((n*L) + 1), n);# including constant;
        temp_draw = temp_draw[1:(end-1), :]';
        Bmat[:,:,t,idraw] = temp_draw;
    end
end
# compute responses
kron(eye(n), zeros(n,L))
# only look at one horizon,
irf_horizon = 6;
irf_i = Array{Float64}(n,n,length(0:1:irf_horizon),T,ndraw);

p = ProgressMeter.Progress(ndraw, 0.01);
for idraw = 1:ndraw
    for t = 1:T
        temp_A = Amat[:, :, t, idraw];
        # always compute a unit shock.
        COMP = [Bmat[:, :, t, idraw]; eye(n*(L-1)) zeros(n*(L-1),n)];
        J = [eye(n*(L-1)) zeros(n*(L-1),n)];
        # create companion matrix,

        for irf_h = 0:1:(irf_horizon - 1)
            temp_irf = COMP^irf_h;

            temp_irf = J*(temp_irf)*J'; #/Amat[:, :, t, idraw]
            irf_i[:,:,(irf_h + 1),t,idraw] = temp_irf;
        end
    end
    ProgressMeter.next!(p); # track progress - will take some time.
end












# First, plot standard erros, hsave.
res_se = res["hsave"];
y_ub = Array{Float64}(T,n);
y_lb = Array{Float64}(T,n);

for i = 1:T
    for j = 1:n
        y_ub[i,j] = quantile(res_se[:,i,j], (1-α));
        y_lb[i,j] = quantile(res_se[:,i,j], α);
    end
end

res_se1 = DataFrame(
    x = time, # time
    y1 = vec(median(res_se[:,:,1],1)), # median response
    y1max = y_ub[:,1],
    y1min = y_lb[:,1],
    Variable = "SAP"
    );
res_se2 = DataFrame(
    x = time, # time
    y1 = vec(median(res_se[:,:,2],1)), # median response
    y1max = y_ub[:,2],
    y1min = y_lb[:,2],
    Variable = "Inflation"
    );
res_se3 = DataFrame(
    x = time, # time
    y1 = vec(median(res_se[:,:,3],1)), # median response
    y1max = y_ub[:,3],
    y1min = y_lb[:,3],
    Variable = "Sentiment"
    );
#res_se1 = vcat(res_se1, res_se2, res_se3);
plot_i = plot(res_se1, x=:x, y=:y1, ymin=:y1min, ymax=:y1max, color=:Variable,
         Geom.line, Geom.ribbon,
         Guide.ylabel("Underlying uncertainty (Index)"),
         Guide.xlabel("Time (Month)")
         )
draw(PNG("fig_StochasticVolatility_1.png", 8inch, 5inch, dpi =  400), plot_i);
#
plot_i = plot(res_se2, x=:x, y=:y1, ymin=:y1min, ymax=:y1max, color=:Variable,
         Geom.line, Geom.ribbon,
         Guide.ylabel("Underlying uncertainty (Index)"),
         Guide.xlabel("Time (Month)")
         )
draw(PNG("fig_StochasticVolatility_2.png", 8inch, 5inch, dpi =  400), plot_i);
#
plot_i = plot(res_se3, x=:x, y=:y1, ymin=:y1min, ymax=:y1max, color=:Variable,
         Geom.line, Geom.ribbon,
         Guide.ylabel("Underlying uncertainty (Index)"),
         Guide.xlabel("Time (Month)")
         )
draw(PNG("fig_StochasticVolatility_3.png", 8inch, 5inch, dpi =  400), plot_i);


for i_response = 1:n
    for i_shock = 1:n
        temp_irf = irf_i[i_response, i_shock, :, :, :];
        temp_num_irf = 2:2:4; temp_irfcount = length(temp_num_irf);

        temp_ub = Array{Float64}(T, temp_irfcount);
        temp_lb = Array{Float64}(T, temp_irfcount);

        for i_time = 1:T
            for i_hor = 2:2:4
                ind_temp = convert(Int64, i_hor/2);
                temp_ub[i_time, ind_temp] = quantile(temp_irf[i_hor, i_time, :], (1-α));
                temp_lb[i_time, ind_temp] = quantile(temp_irf[i_hor, i_time, :], α);
            end
        end

        # organize impulse response functions
        temp_num = 1;
        count_temp = temp_num_irf[temp_num];
        temp_plot = DataFrame(
            x = time, # time
            y = vec(median(temp_irf[count_temp,:,:],2)), # median response
            ymax = temp_ub[:, temp_num],
            ymin = temp_lb[:, temp_num],
            Variable = "Horizon $(count_temp)"
            );
        while temp_num < length(temp_num_irf)
            temp_num += 1; count_temp = temp_num_irf[temp_num];
            temp_plotadd = DataFrame(
                x = time, # time
                y = vec(median(temp_irf[count_temp,:,:],2)), # median response
                ymax = temp_ub[:, temp_num],
                ymin = temp_lb[:, temp_num],
                Variable = "Horizon $(count_temp)"
                );
            temp_plot = vcat(temp_plot, temp_plotadd);
        end

        # start plotting

        plot_i = plot(temp_plot, x=:x, y=:y, ymin=:ymin, ymax=:ymax, color=:Variable,
                 Geom.line, Geom.ribbon,
                 Guide.ylabel("Response"),
                 Guide.xlabel("Time (Month)")
                 ); plot_i
        draw(PNG("fig_irf_$(i_response)_on_$(i_shock)_wCI.png", 8inch, 5inch, dpi =  400), plot_i);

        plot_i = plot(temp_plot, x=:x, y=:y, color=:Variable,
                 Geom.line,
                 Guide.ylabel("Response"),
                 Guide.xlabel("Time (Month)")
                 ); plot_i
        draw(PNG("fig_irf_$(i_response)_on_$(i_shock).png", 8inch, 5inch, dpi =  400), plot_i);
    end
end
