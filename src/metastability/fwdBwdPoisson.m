function [alpha, beta, gamma, xi, logP] = fwdBwdPoisson(data, pi0, A, lambda, dt)
% FWDBWDPOISSON Run forward-backward algorithm for Poisson HMM.
% Returns posterior probabilities and log-likelihood.

[T, N] = size(data);
M = size(lambda,1);
data = double(data);

% Compute log-emission probabilities using full Poisson model
logB = zeros(T, M);
for m = 1:M
    lam = lambda(m, :) * dt;        % expected count per bin
    lam(lam <= 0) = 1e-6;           % avoid log(0)
    for t = 1:T
        k = data(t, :);
        logB(t, m) = sum(k .* log(lam) - lam - gammaln(k + 1));
    end
end

% Forward pass
alpha = zeros(T, M);
scale = zeros(T,1);
alpha(1,:) = pi0 .* exp(logB(1,:));
scale(1) = sum(alpha(1,:));
if scale(1) <= 1e-12
    scale(1) = 1e-12;
end
alpha(1,:) = alpha(1,:) / scale(1);

for t = 2:T
    alpha_prev = reshape(alpha(t - 1, :), 1, []);
    alpha(t,:) = (alpha_prev * A) .* exp(logB(t,:));
    scale(t) = sum(alpha(t,:));
    if scale(t) <= 1e-12
        scale(t) = 1e-12;
    end
    alpha(t,:) = alpha(t,:) / scale(t);
end

% Backward pass
beta = zeros(T, M);
beta(T,:) = ones(1,M) / scale(T);
for t = T-1:-1:1
    beta(t,:) = (beta(t+1,:) .* exp(logB(t+1,:))) * A';
    beta(t,:) = beta(t,:) / max(scale(t), 1e-12);
end

% Compute posteriors
gamma = alpha .* beta;
gamma = normalize_to_1(gamma);


% Compute pairwise state transition probabilities (xi)
xi = zeros(T-1, M, M);
for t = 1:T-1
    prob = (alpha(t,:)' .* A) .* (exp(logB(t+1,:)) .* beta(t+1,:));
    probSum = sum(prob(:));
    if ~isfinite(probSum) || probSum == 0
        xi(t,:,:) = ones(M) / M^2;
    else
        xi(t,:,:) = prob / probSum;
    end
end

% Total log-likelihood
logP = sum(log(scale));

% Enforce state index bounds in case decodeHMM calls this
if nargout > 4 && exist('decodeHMM', 'file')
    if isfield(gamma, 'stateSeq')
        gamma.stateSeq = min(max(gamma.stateSeq, 1), M);
    end
end
end
