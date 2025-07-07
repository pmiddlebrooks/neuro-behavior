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
    
    % Debug: Check input data
    if any(isnan(data)) || any(isinf(data))
        error('Input data contains NaN or Inf values');
    end
    
    if all(data == 0)
        error('Input data contains all zeros');
    end
    
    if length(data) < maxSlopes + 10
        error('Input data too short for maxSlopes = %d (need at least %d samples)', maxSlopes, maxSlopes + 10);
    end
    
    % Compute regression slopes r_k
    k = (1:maxSlopes-1)';
    r_k = zeros(size(k));
    intercepts = zeros(size(k));
    std_errs = zeros(size(k));
    
    for i = 1:length(k)
        lag = k(i);
        x = data(1:end - lag);
        y = data(1 + lag:end);

        % Debug: Check for valid regression data
        if all(x == 0) || all(y == 0)
            fprintf('Warning: Lag %d has all-zero x or y data\n', lag);
            r_k(i) = 0;
            intercepts(i) = 0;
            std_errs(i) = 0;
            continue;
        end

        % Perform linear regression: y = r_k * x + intercept
        Xmat = [x, ones(size(x))];
        
        % Check for singular matrix
        if cond(Xmat) > 1e12
            fprintf('Warning: Singular matrix for lag %d\n', lag);
            r_k(i) = 0;
            intercepts(i) = 0;
            std_errs(i) = 0;
            continue;
        end
        
        coeffs = Xmat \ y;
        y_fit = Xmat * coeffs;
        residuals = y - y_fit;

        r_k(i) = coeffs(1);
        intercepts(i) = coeffs(2);
        std_errs(i) = std(residuals);
        
        % Debug: Check for extreme values
        if abs(r_k(i)) > 10 || abs(intercepts(i)) > 1000
            fprintf('Warning: Extreme values at lag %d: r_k=%.3f, intercept=%.3f\n', lag, r_k(i), intercepts(i));
        end
    end

    % Debug: Check r_k values
    if any(isnan(r_k)) || any(isinf(r_k))
        error('Regression slopes r_k contain NaN or Inf values');
    end
    
    % Remove any zero std_errs to avoid division by zero
    valid_idx = std_errs > 0;
    if sum(valid_idx) < 3
        error('Too few valid regression slopes for fitting');
    end
    
    k_valid = k(valid_idx);
    r_k_valid = r_k(valid_idx);
    std_errs_valid = std_errs(valid_idx);

    % Exponential model: r_k = a * b^k
    fitfunc = @(p, k) abs(p(1)) .* abs(p(2)) .^ k;

    % Initial guess: [a, b]
    p0 = [r_k_valid(1), 0.9]; % Start with slightly conservative guess

    % Fit using non-linear least squares
    options = optimset('MaxFunEvals', 1e5, 'Display', 'off');
    weights = std_errs_valid .* linspace(1, 10, length(std_errs_valid))';
    
    % Debug: Check weights
    if any(weights == 0) || any(isnan(weights)) || any(isinf(weights))
        error('Invalid weights in objective function');
    end
    
    fitObj = @(p) (fitfunc(p, k_valid) - r_k_valid) ./ weights;
    
    % Debug: Check initial objective function value
    initial_obj = fitObj(p0);
    if any(isnan(initial_obj)) || any(isinf(initial_obj))
        error('Initial objective function returns NaN or Inf values');
    end

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
