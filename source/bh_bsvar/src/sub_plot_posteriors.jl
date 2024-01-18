# plot prior and posterior
fig_posterior = figure("pyplot_subplot_column", figsize=(20, 20))
nbins = 200;
subplots_adjust(hspace = 0.5, wspace = 0.4)
for i_subplot = 1:nA
  subplot(2,5,i_subplot)
  # calculate median and percentile bound for legend,
  local A_posterior_i_index = sort(A_posterior[i_subplot, :]);
  local post_i_lower = round(A_posterior_i_index[index1], 3);
  local post_i_median = round(A_posterior_i_index[index0], 3);
  local post_i_upper = round(A_posterior_i_index[index2], 3);
  local label_prior = "prior draw of $(names_α[i_subplot])";
  local label_posterior = "Posterior draw: \n {$(post_i_lower), $(post_i_median), $(post_i_upper)}"
  plt[:hist](A_posterior[i_subplot,:], nbins, normed = true, label = label_posterior, color = :grey) # Histogram
  plt[:plot](x_prior, y_prior[i_subplot, :], linewidth = 2, label = label_prior, color = :black);

  # change axis
  ax = gca();
  # Change axis limit as following,
  if i_subplot <= 5
    ax[:set_xlim]([minimum([0 minimum(A_posterior[i_subplot, :])]),
                   maximum([0 maximum(A_posterior[i_subplot, :])])
                  ]);
  else
    i_subplot <= 5
    ax[:set_xlim]([minimum([0 minimum([-1; A_posterior[i_subplot, :]])]),
                   maximum([0 maximum(A_posterior[i_subplot, :])])
                  ]);
  end
  legend(bbox_to_anchor = (1, -0.1), borderaxespad=0, edgecolor = "none",
         facecolor = "none", fontsize = 8)
  #grid("on")
  #xlabel("X")
  #ylabel("Y")
  #title("Histogram")
end
suptitle("Prior and posterior density of identified elements of A {lower, median, upper}");
fig_posterior[:canvas][:draw](); #update figure,;
savefig("fig_posterior.pdf", dpi = 800)
PyPlot.close();
# end of file
################################################################################
