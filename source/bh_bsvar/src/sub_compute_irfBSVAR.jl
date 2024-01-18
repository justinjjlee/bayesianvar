#impulse response
cholstar_i = zeros(k, k, n);
for i_var = 1:n
  cholstar_i[:, :, i_var] = chol(Hermitian(xtildei[:, :, i_var]))';
end

invdDraw = zeros(n, nuse);
Bdraw = zeros(n, k, nuse);
est_irf = zeros(n, n, length(horizon_impact), nuse);

# to use in one-standard deviation shock, following notation of LutKepohl (2005)

progress_irf = Progress(nuse, 1, "IRF.iter:", 30)

for i_iter = 1:nuse
  local A = func_setA(A_posterior[:, i_iter]);
  local taustar = kappa.*diag(A*Sstar*A') + zeta_posterior[:, i_iter]./2;
  invdDraw[:, i_iter] = [rand(Gamma(kappastar[i_var], 1./taustar[i_var]), 1)[1] for i_var = 1:n]

  for i_var = 1:n
    Bdraw[i_var, :, i_iter] = mstar_posterior[i_var, :, i_iter]' +
                              (1/sqrt(invdDraw[i_var, i_iter])) *
                              (cholstar_i[:, :, i_var]*randn(k,1))';
  end
  # compute response,
  invA = inv(A);
  #=
  # for one standard deviation shock, add following steps.
  Ddraw = 1./invdDraw[:, i_iter];
  D_tst = diagm(diag(Ddraw));
  invA = A\D_tst .^ 0.5
  =#

  # reduced-form coefficient
  B_reduced = A\Bdraw[:, :, i_iter];
  B_reduced = B_reduced[:, (1:(end-1))];
  COMP_matrix = [B_reduced; kron(eye(n), eye(nlags-1)) zeros((n*(nlags-1)), n)];
  J = [eye(n) zeros(n, (n*(nlags-1)))];
  for i_irf = 1:length(horizon_impact)
    h_irf = horizon_impact[i_irf];
    local phi_irf_matrix = J*(COMP_matrix^(h_irf))*J';
    est_irf[:, :, i_irf, i_iter] = phi_irf_matrix * invA;
  end
  ProgressMeter.next!(progress_irf);
end
# cumulate response for future use,
est_irf_cumsum = cumsum(est_irf, 3);

# plot impulse response.
# plot prior and posterior
fig_irf = figure("pyplot_subplot_column", figsize=(20, 20))
subplots_adjust(hspace = 0.3, wspace = 0.3)

for i_shocks = 1:n
  for i_responses = 1:n
    num_subplot = i_responses + (i_shocks - 1)*n;
    subplot(n, n, num_subplot)
    responses_i = est_irf_cumsum[i_responses, i_shocks, :, :];
    irf_posterior_i_index = sort(responses_i, 2);
    irf_lb = irf_posterior_i_index[:, index1];
    irf_median = irf_posterior_i_index[:, index0];
    irf_ub = irf_posterior_i_index[:, index2];
    plt[:plot](horizon_impact, irf_median, color = :black);
    plt[:plot](horizon_impact, irf_lb, linestyle = :dashed, color = :black);
    plt[:plot](horizon_impact, irf_ub, linestyle = :dashed, color = :black);
    # add zero line,
    plt[:plot](horizon_impact, squeeze(zeros(1,(hmax+1)), 1),
               linewidth = 1.1, linestyle = :dotted, color = :black)
    # change axis
    ax = gca();
    ax[:set_xlim]([0, 15]);
    ax[:set_xticks]([0, 3, 6, 9, 12, 15]);
    # name responses on y axis,
    if any(num_subplot .== [13 14 15 16])
      local ind_response = convert(Int64, num_subplot - 12);
      #ax[:set_ylabel] = "Response of $(varnames[ind_response])";
      xlabel("Response of $(varnames[ind_response])", fontsize = 8)
    end
    # name shocks on x axis,
    if any(num_subplot .== [1 5 9 13])
      local ind_shock = convert(Int64, floor(num_subplot/4) + 1);
      ylabel("$(shocknames[ind_shock])", fontsize = 8);
    end
  end
end
suptitle("Impulse responses and predictive 95-percentile confidence band");
fig_irf[:canvas][:draw](); #update figure,;
savefig("fig_irf.pdf", dpi = 800)
PyPlot.close();
