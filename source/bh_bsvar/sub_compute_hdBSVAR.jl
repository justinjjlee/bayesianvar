# historical decomposition after running irfBSVAR
# for this paper, all responses needs to be cumulated,
irf_used = est_irf_cumsum;
# compute HD
# other option
# @showprogress 1 "HD-calculation-sample: " for i_iter = 1:nuse
progress_MH_time = Progress(nuse, 1, "HD.iter:", 30)

for i_iter = 1:nuse
 A = func_setA(A_posterior[:, i_iter]);
 B = Bdraw[:, :, i_iter];
 U = YY*A' - XX*B';
 for i_shocks = 1:n
  ulast = U[:, i_shocks];
  uj = zeros(T, n);
  for i_horizon = 1:hmax
   uj = uj + (ulast * irf_used[:, i_shocks, i_horizon, i_iter]');
   ulast = [0; ulast[1:(end-1)]];
  end
  hd_posterior[:, i_shocks, :, i_iter] = uj';
 end
 i_iter_pct = "$(round(i_iter/nuse*100, 2))%";
 ProgressMeter.next!(progress_MH_time; showvalues = [("Iter:", i_iter_pct)]);
end

# PUll quantiles,
hd_posterior_lb  = zeros(n, n, T);
hd_posterior_median = zeros(n, n, T);
hd_posterior_ub = zeros(n, n, T);

progress_MH_hor = Progress(n*n*T, 1, "HD.hor:", 30)
for i_responses = 1:n
 for i_shocks = 1:n
  for i_time = 1:T
    local hd_post = hd_posterior[i_responses, i_shocks, i_time, :]; #no squeeze J
    hd_post = sort(hd_post);
    hd_posterior_lb[i_responses, i_shocks, i_time]  = hd_post[index1];
    hd_posterior_median[i_responses, i_shocks, i_time] = hd_post[index0];
    hd_posterior_ub[i_responses, i_shocks, i_time] = hd_post[index2];

    i_iter_pct = "$(round((i_responses*i_shocks*i_time)/(n*n*T)*100, 2))%";
    ProgressMeter.next!(progress_MH_hor; showvalues = [("Iter:", i_iter_pct)]);
  end
 end
end

# Plot historical decomposition,
for i_responses = 1:n
 #fig_hd= figure("pyplot_subplot_column", figsize=(20, 20))
 figure("pyplot_subplot_column", figsize=(15, 25))
 subplots_adjust(hspace = 0.3, wspace = 0.3)
 # data compared must be stationary and detrended.
 local yy = cumsum(func_detrend(YY[:, i_responses]));
 for i_shocks = 1:n
  subplot(n, 1, i_shocks)
  local hd_i_med = hd_posterior_median[i_responses, i_shocks, :];
  local hd_i_ub = hd_posterior_ub[i_responses, i_shocks, :];
  local hd_i_lb = hd_posterior_lb[i_responses, i_shocks, :];
  plt[:plot](time, hd_i_med, color = :black);
  plt[:plot](time, hd_i_ub, color = :black, linestyle = :dashed);
  plt[:plot](time, hd_i_lb, color = :black, linestyle = :dashed);
  plt[:plot](time, yy, color = :red);
  grid("on");
  title("... on $(shocknames[i_shocks])")
  ax = gca()
  # set y limit individually,
  if i_responses == 1
   ax[:set_ylim]([-0.08, 0.08]);
  elseif i_responses == 2
   ax[:set_ylim]([-0.2, 0.2]);
  elseif i_responses == 3
   ax[:set_ylim]([-0.7, 0.7]);
  else
   ax[:set_ylim]([-1.2, 1.2]);
  end
  if i_shocks == n
   xlabel("Time (month)", fontsize = 14);
  end
  # set x-axis limit
  ax[:set_xlim](time[1], time[end]);
  tic_minor_interval = matplotlib[:ticker][:MultipleLocator](1);
  ax[:xaxis][:set_minor_locator](tic_minor_interval);
  tic_minor_interval = matplotlib[:ticker][:MultipleLocator](3);
  ax[:xaxis][:set_major_locator](tic_minor_interval);
 end
 suptitle("Cumulative responses of $(varnames[i_responses])");
 #fig_hd[:canvas][:draw](); #update figure,;
 savefig("fig_hd_$(varnames[i_responses]).pdf", dpi = 800)
 PyPlot.close();
end
