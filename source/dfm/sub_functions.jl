# Functions

function func_std(x)
    # Able to use in matrix
    μ = kron(mean(x, dims = 1), ones(size(x)[1],1));
    σ = kron(std(x, dims = 1), ones(size(x)[1],1));
    return((x .- μ)./σ)
end

function func_dfm(x, y, num_f)
    # Dynamic factor model
    t, m = size(y);
    t, k = size(x);

    # Estimate initial factor estimation
    𝐏₀ = fit(PCA, x; maxoutdim = num_f);
    𝐅₀ = projection(𝐏₀)        # Factor estimated
    𝐕 = principalratio(𝐏₀)    # Percent of Factor explained
    𝚪₀ = (𝐅₀'*𝐅₀)\𝐅₀'*x;       # Calculate factor loading

    # Regression facotr change
    q = x';
    z = hcat(y, 𝐅₀)';
    γ₀ = (q*z')/(z*z');
    γ₀_y = γ₀[:, 1:m];
    x̃₀ = x .- (γ₀_y * y')'; # variations not explained
    ssr₀ = sum((q - γ₀*z).^2, dims = 2);

    # Establish proposed step
    # Estimate initial factor estimation
    𝐏₁ = fit(PCA, x̃₀; maxoutdim = num_f);
    𝐅₁ = projection(𝐏₁)        # Factor estimated
    𝐕 = vcat(𝐕, principalratio(𝐏₁))    # Percent of Factor explained
    𝚪₁ = (𝐅₁'*𝐅₁)\𝐅₁'*x;       # Calculate factor loading

    # Regression facotr change
    z = hcat(y, 𝐅₁)';
    γ₁ = (q*z')/(z*z');
    γ₁_y = γ₁[:, 1:m];
    x̃₁ = x .- (γ₁_y * y')'; # variations not explained
    ssr₁ = sum((q - γ₁*z).^2, dims = 2);

    # Parameter to save
    𝐅₀₀ = 𝐅₀; # very first factor extracted
    # Converge to equilibrium
    iter = 1;
    while sum(abs.(ssr₁ .- ssr₀)) >= 10^(-6)
        # Repeat above step

        ssr₀ = ssr₁
        𝐅₀ = 𝐅₁

        # Estimate initial factor estimation
        𝐏₁ = fit(PCA, x̃₀; maxoutdim = num_f);
        𝐅₁ = projection(𝐏₁)        # Factor estimated
        𝐕 = vcat(𝐕, principalratio(𝐏₁))    # Percent of Factor explained
        𝚪₁ = (𝐅₁'*𝐅₁)\𝐅₁'*x;       # Calculate factor loading

        # Regression facotr change
        z = hcat(y, 𝐅₁)';
        γ₁ = (q*z')/(z*z');
        γ₁_y = γ₁[:, 1:m];
        x̃₁ = x .- (γ₁_y * y')'; # variations not explained
        ssr₁ = sum((q - γ₁*z).^2, dims = 2);


        iter += 1;
        println("Iteration: $(iter)")
    end
    return(𝐅₀₀, 𝐅₁, 𝐕)
end
