function logLik = gaussian_hmm_sequence_loglik(observationMatrix, model)
% GAUSSIAN_HMM_SEQUENCE_LOGLIK Total log p(observationMatrix | HMM parameters).
%
% Variables:
%   - observationMatrix: [T x D]
%   - model: struct with pi0, A, mu, varDiag, numStates
% Goal:
%   Forward-algorithm log-likelihood (single Gaussian-emission HMM, diagonal cov).
% Returns:
%   - logLik: scalar log probability of the full sequence

X = double(observationMatrix);
T = size(X, 1);
K = model.numStates;
logB = gaussian_hmm_logB(X, model.mu, model.varDiag);
logA = log(model.A + realmin);
logPi = log(model.pi0(:) + realmin);

logAlpha = zeros(T, K);
logAlpha(1, :) = logPi' + logB(1, :);
for t = 2:T
    for j = 1:K
        logAlpha(t, j) = logsumexp_vec(logAlpha(t - 1, :)' + logA(:, j)) + logB(t, j);
    end
end
logLik = logsumexp_vec(logAlpha(end, :)');
end

function s = logsumexp_vec(v)
m = max(v);
s = m + log(sum(exp(v - m)));
end
