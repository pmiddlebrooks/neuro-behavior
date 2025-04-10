function result = branching_ratio_mr_estimation(data, maxSlopes)
% MR_estimation - Multistep Regression estimate of the branching ratio
% Adapted from Wilting and Preismann 2018
% Github python repo: https://github.com/jwilting/WiltingPriesemann2018/tree/master
%
% INPUT:
%   data       - time series vector (1D array)
%   maxSlopes  - maximum lag k to compute regression slopes (default: 40)
% OUTPUT:
%   result     - struct containing estimated branching ratio, etc.

    if nargin < 2
        maxSlopes = 40;
    end

    % Ensure column vector
    data = data(:);
    
    % Compute regression slopes r_k
    k = (1:maxSlopes-1)';
    r_k = zeros(size(k));
    intercepts = zeros(size(k));
    std_errs = zeros(size(k));
    
    for i = 1:length(k)
        lag = k(i);
        x = data(1:end - lag);
        y = data(1 + lag:end);

        % Perform linear regression: y = r_k * x + intercept
        Xmat = [x, ones(size(x))];
        coeffs = Xmat \ y;
        y_fit = Xmat * coeffs;
        residuals = y - y_fit;

        r_k(i) = coeffs(1);
        intercepts(i) = coeffs(2);
        std_errs(i) = std(residuals);
    end

    % Exponential model: r_k = a * b^k
    fitfunc = @(p, k) abs(p(1)) .* abs(p(2)) .^ k;

    % Initial guess: [a, b]
    p0 = [r_k(1), 1.0];

    % Fit using non-linear least squares
    options = optimset('MaxFunEvals', 1e5, 'Display', 'off');
    weights = std_errs .* linspace(1, 10, length(std_errs))';
    fitObj = @(p) (fitfunc(p, k) - r_k) ./ weights;

    p_opt = lsqnonlin(fitObj, p0, [], [], options);

    % Output results
    result.branching_ratio = p_opt(2);
    result.autocorrelation_time = -1 / log(p_opt(2));
    result.naive_branching_ratio = r_k(1);
    result.k = k;
    result.r_k = r_k;
    result.intercepts = intercepts;
    result.std_errs = std_errs;
    result.p_opt = p_opt;
end
