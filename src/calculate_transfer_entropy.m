% Function to calculate transfer entropy from y to x with a given delay
function te = calculate_transfer_entropy(x, y, delay, nBins)
% calculate_transfer_entropy: Transfer entropy from y to x with specified delay
%   TE_{y->x}(delay) = I(x_t; y_{t-delay} | x_{t-delay})
%   Inputs:
%     x     - target time series (vector)
%     y     - source time series (vector)
%     delay - lag (integer, in samples)
%     nBins - number of bins for discretization (optional, default: 8)
%   Output:
%     te    - transfer entropy value (in bits)

if nargin < 4 || isempty(nBins)
    nBins = 8;
end

% Ensure column vectors
x = x(:);
y = y(:);

if length(x) <= delay || length(y) <= delay
    te = nan;
    return;
end

% Align time series for delay
x_t = x(1+delay:end);
x_past = x(1:end-delay);
y_past = y(1:end-delay);

% Remove rows with NaNs in any variable
validIdx = ~isnan(x_t) & ~isnan(x_past) & ~isnan(y_past);
x_t = x_t(validIdx);
x_past = x_past(validIdx);
y_past = y_past(validIdx);

if length(x_t) < 10
    te = nan;
    return;
end

% Discretize variables using same bin edges as in calculate_mutual_information
[~, edgesX] = histcounts([x_t; x_past], nBins);
[~, edgesY] = histcounts([y_past], nBins);

x_t_disc = discretize(x_t, edgesX);
x_past_disc = discretize(x_past, edgesX);
y_past_disc = discretize(y_past, edgesY);

% Remove any NaNs from discretization
validIdx = ~isnan(x_t_disc) & ~isnan(x_past_disc) & ~isnan(y_past_disc);
x_t_disc = x_t_disc(validIdx);
x_past_disc = x_past_disc(validIdx);
y_past_disc = y_past_disc(validIdx);

if length(x_t_disc) < 10
    te = nan;
    return;
end

% Joint histogram: p(x_t, x_past, y_past)
jointHist = accumarray([x_t_disc, x_past_disc, y_past_disc], 1, [nBins, nBins, nBins]);
jointProb = jointHist / sum(jointHist(:));

% Marginals for conditional probabilities
% p(x_t, x_past)
jointHist_xx = accumarray([x_t_disc, x_past_disc], 1, [nBins, nBins]);
jointProb_xx = jointHist_xx / sum(jointHist_xx(:));
% p(x_past, y_past)
jointHist_xy = accumarray([x_past_disc, y_past_disc], 1, [nBins, nBins]);
jointProb_xy = jointHist_xy / sum(jointHist_xy(:));
% p(x_past)
marginal_x_past = accumarray(x_past_disc, 1, [nBins, 1]);
marginal_x_past = marginal_x_past / sum(marginal_x_past);

% Calculate transfer entropy
te = 0;
for i = 1:nBins
    for j = 1:nBins
        for k = 1:nBins
            p_xyz = jointProb(i, j, k);
            p_xx = jointProb_xx(i, j);
            p_xy = jointProb_xy(j, k);
            p_x = marginal_x_past(j);
            if p_xyz > 0 && p_xx > 0 && p_xy > 0 && p_x > 0
                te = te + p_xyz * log2((p_xyz * p_x) / (p_xx * p_xy));
            end
        end
    end
end
end 