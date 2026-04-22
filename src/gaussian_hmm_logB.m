function logB = gaussian_hmm_logB(observationMatrix, mu, varDiag)
% GAUSSIAN_HMM_LOGB Log-probability of each observation under diagonal Gaussian states.
%
% Variables:
%   - observationMatrix: [T x D]
%   - mu: [K x D] state means
%   - varDiag: [K x D] state variances (positive)
% Goal:
%   Vectorized log N(x_t; mu_k, diag(varDiag_k)) for all t,k.
% Returns:
%   - logB: [T x K]

X = double(observationMatrix);
[T, D] = size(X);
K = size(mu, 1);
logConst = D * log(2 * pi);
logB = zeros(T, K);
for k = 1:K
    diffMat = bsxfun(@minus, X, mu(k, :));
    v = varDiag(k, :);
    quad = sum((diffMat .^ 2) ./ v + log(v), 2);
    logB(:, k) = -0.5 * (logConst + quad);
end
end
