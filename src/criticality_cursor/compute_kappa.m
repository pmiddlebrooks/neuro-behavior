function kappa = compute_kappa(avalancheSizes)
    % computeKappa calculates κ based on Shew et al., 2009
    % INPUT:
    %   avalanche_sizes - vector of avalanche sizes
    % OUTPUT:
    %   kappa - criticality metric (κ = 1 at criticality; k < 1
    %   subcritical; k > 1 supercritical)

    m = 10;  % Number of beta points
    avalancheSizes = avalancheSizes(avalancheSizes > 0);

    % Get size bounds
    l = min(avalancheSizes);
    L = max(avalancheSizes);

    % Define beta points (log-spaced)
    % beta_k = round(logspace(log10(l), log10(L), m));
beta_k = round(logspace(log10(l + 1), log10(L), m));

    % Empirical CDF
    sorted_sizes = sort(avalancheSizes);
    F_empirical = @(x) sum(sorted_sizes <= x) / length(sorted_sizes);

    % Theoretical CDF from -3/2 PDF (=> -1/2 CDF)
    F_theoretical = @(beta) (1 - sqrt(l / beta)) / (1 - sqrt(l / L));

    % Sum of ratios (empirical / theoretical)
    kappa_sum = 0;
    for k = 1:m
        beta = beta_k(k);
        F_beta = F_empirical(beta);
        F_na = F_theoretical(beta);
        kappa_sum = kappa_sum + (F_beta / F_na);
    end

    % Final κ
    kappa = kappa_sum / m;
end
