function [model, diagnostics] = gaussian_hmm_fit(observationMatrix, numStates, opts)
% GAUSSIAN_HMM_FIT Multivariate Gaussian-emission HMM via Baum-Welch (EM).
%
% Variables:
%   - observationMatrix: [nBins x nFeatures], no NaN/Inf
%   - numStates: number of hidden states K (>=2)
%   - opts: struct with optional fields:
%       .maxIter (default 200), .tol (default 1e-4)
%       .numRestarts (default 5) independent initializations; best train log-lik kept
%       .regVar (default 1e-3) floor on emission variance (per dim, per state)
%       .decode (default true) run Viterbi after fit and store model.stateSeq
% Goal:
%   Fit diagonal-covariance Gaussian HMM (full state transitions) for continuous LFP features.
% Returns:
%   - model: struct with fields pi0 [Kx1], A [KxK], mu [KxD], varDiag [KxD],
%            stateSeq [nBinsx1], statePosterior [nBinsxK], maxPosterior [nBinsx1]
%   - diagnostics: struct with trainLogLik, numIterations, restartIdx

if nargin < 3
    opts = struct();
end
maxIter = get_opt(opts, 'maxIter', 200);
tol = get_opt(opts, 'tol', 1e-4);
numRestarts = get_opt(opts, 'numRestarts', 5);
regVar = get_opt(opts, 'regVar', 1e-3);
doDecode = get_opt(opts, 'decode', true);

observationMatrix = double(observationMatrix);
nonFiniteMask = ~isfinite(observationMatrix(:));
if any(nonFiniteMask)
    warning('gaussian_hmm_fit:NonFiniteInput', ...
        'Replacing %d non-finite feature entries with 0 (e.g. zero-variance columns after zscore).', ...
        nnz(nonFiniteMask));
    observationMatrix(nonFiniteMask) = 0;
end
nBins = size(observationMatrix, 1);
assert(numStates >= 2 && numStates < nBins, 'numStates must be in [2, nBins).');

bestTrainLl = -inf;
bestModel = [];
bestDiag = struct('trainLogLik', -inf, 'numIterations', 0, 'restartIdx', 0);

savedRng = rng;
baseSeed = 0;
if isfield(savedRng, 'Seed')
    baseSeed = savedRng.Seed;
end
for restartIdx = 1:numRestarts
    rng(mod(baseSeed + restartIdx * 7919, 2^31 - 2));
    [thisModel, thisDiag] = em_single_run(observationMatrix, numStates, maxIter, tol, regVar);
    if thisDiag.trainLogLik > bestTrainLl
        bestTrainLl = thisDiag.trainLogLik;
        bestModel = thisModel;
        bestDiag = thisDiag;
        bestDiag.restartIdx = restartIdx;
    end
end
rng(savedRng);

model = bestModel;
diagnostics = bestDiag;

if doDecode && ~isempty(model)
    model.stateSeq = gaussian_hmm_viterbi(observationMatrix, model);
    logB = gaussian_hmm_logB(observationMatrix, model.mu, model.varDiag);
    [statePosterior, ~, ~] = e_step(model.pi0, model.A, logB);
    model.statePosterior = statePosterior;
    model.maxPosterior = max(statePosterior, [], 2);
end
end

function v = get_opt(opts, name, defaultVal)
if isfield(opts, name)
    v = opts.(name);
else
    v = defaultVal;
end
end

function [model, diagOut] = em_single_run(X, K, maxIter, tol, regVar)
[T, D] = size(X);
rowIdx = randperm(T, K);
mu = X(rowIdx, :);
varDiag = ones(K, D) .* (var(X, [], 1) + regVar);
pi0 = ones(K, 1) / K;
A = ones(K, K) / K;
diagA = 0.12;
A = (1 - diagA) * A + diagA * eye(K);
A = bsxfun(@rdivide, A, sum(A, 2));

prevLl = -inf;
logLik = -inf;
for iterIdx = 1:maxIter
    logB = gaussian_hmm_logB(X, mu, varDiag);
    [gamma, xiSum, logLik] = e_step(pi0, A, logB);
    [pi0, A, mu, varDiag] = m_step(X, gamma, xiSum, regVar);
    if abs(logLik - prevLl) < tol && iterIdx > 2
        break;
    end
    prevLl = logLik;
end

model = struct('pi0', pi0, 'A', A, 'mu', mu, 'varDiag', varDiag, 'numStates', K, ...
    'stateSeq', [], 'statePosterior', [], 'maxPosterior', []);
diagOut = struct('trainLogLik', logLik, 'numIterations', iterIdx);
end

function [gamma, xiSum, logLik] = e_step(pi0, A, logB)
[T, K] = size(logB);
logA = log(A + realmin);

logAlpha = zeros(T, K);
logAlpha(1, :) = log(pi0(:)' + realmin) + logB(1, :);
for t = 2:T
    for j = 1:K
        logAlpha(t, j) = logsumexp_vec(logAlpha(t - 1, :)' + logA(:, j)) + logB(t, j);
    end
end
logLik = logsumexp_vec(logAlpha(end, :)');

logBeta = zeros(T, K);
logBeta(end, :) = 0;
for t = (T - 1):-1:1
    for i = 1:K
        logBeta(t, i) = logsumexp_vec(logA(i, :)' + logB(t + 1, :)' + logBeta(t + 1, :)');
    end
end

logGamma = logAlpha + logBeta;
logRowMax = max(logGamma, [], 2);
logGamma = logGamma - (logRowMax + log(sum(exp(logGamma - logRowMax), 2)));
gamma = exp(logGamma);

xiSum = zeros(K, K);
for t = 1:(T - 1)
    logXi = bsxfun(@plus, logAlpha(t, :)', logA);
    logXi = bsxfun(@plus, logXi, logB(t + 1, :));
    logXi = bsxfun(@plus, logXi, logBeta(t + 1, :));
    xiT = exp(logXi - (max(logXi(:)) + log(sum(exp(logXi(:) - max(logXi(:)))))));
    xiSum = xiSum + xiT;
end
end

function s = logsumexp_vec(v)
m = max(v);
s = m + log(sum(exp(v - m)));
end

function [pi0, A, mu, varDiag] = m_step(X, gamma, xiSum, regVar)
[~, D] = size(X);
K = size(gamma, 2);
gammaSum = sum(gamma, 1)' + realmin;
pi0 = gamma(1, :)' / sum(gamma(1, :));

A = xiSum + 1e-6;
A = bsxfun(@rdivide, A, sum(A, 2));

mu = (gamma' * X) ./ gammaSum;

varDiag = zeros(K, D);
for k = 1:K
    diffMat = bsxfun(@minus, X, mu(k, :));
    varDiag(k, :) = (gamma(:, k)' * (diffMat .^ 2)) / gammaSum(k);
end
varDiag = max(varDiag, regVar);
end
