function model = fit_poisson_HMM(dataMat, numStates, numReps)
[T, N] = size(dataMat);
dt = 0.001;
logL = zeros(numReps, 1);
models = cell(numReps, 1);

for rep = 1:numReps
    pi0 = normalize(rand(1, numStates), 2);
    A = normalize(rand(numStates, numStates), 2);
    lambda = rand(numStates, N) * max(mean(dataMat)) / dt;

    [pi0, A, lambda, ll] = emPoissonHMM(dataMat, pi0, A, lambda, dt);
    models{rep} = struct('pi0', pi0, 'A', A, 'lambda', lambda, 'logL', ll);
    logL(rep) = ll(end);
end

[~, bestIdx] = max(logL);
model = models{bestIdx};
[gamma, stateSeq] = decodeHMM(dataMat, model.pi0, model.A, model.lambda, dt);
model.gamma = gamma;
model.stateSeq = stateSeq;
end


% ========== EM Algorithm ==========
function [pi0, A, lambda, logL] = emPoissonHMM(data, pi0, A, lambda, dt)
maxIter = 100;
tol = 1e-4;
[T, N] = size(data);
M = size(A, 1);
logL = zeros(maxIter, 1);

for iter = 1:maxIter
    [alpha, beta, gamma, xi, logP] = fwdBwdPoisson(data, pi0, A, lambda, dt);
    logL(iter) = logP;

    pi0 = gamma(1, :);
    A = normalize(sum(xi, 1), 2);

    for m = 1:M
        weightedSum = sum(data .* gamma(:, m), 1);
        totalWeight = sum(gamma(:, m));
        lambda(m, :) = -log(1 - (weightedSum / totalWeight)) / dt;
    end

    if iter > 1 && abs(logL(iter) - logL(iter - 1)) < tol
        logL = logL(1:iter);
        break;
    end
end
end


% ========== Forward-Backward ==========
function [alpha, beta, gamma, xi, logP] = fwdBwdPoisson(data, pi0, A, lambda, dt)
[T, N] = size(data);
M = size(lambda, 1);
logLambda = log(1 - exp(-lambda * dt));

logB = zeros(T, M);
for m = 1:M
    logB(:, m) = sum(data .* logLambda(m, :), 2);
end

alpha = zeros(T, M);
scale = zeros(T, 1);
alpha(1, :) = pi0 .* exp(logB(1, :));
scale(1) = sum(alpha(1, :));
alpha(1, :) = alpha(1, :) / scale(1);

for t = 2:T
% assert(ndims(A) == 2, 'A must be 2D');
if ~ismatrix(A)
    disp('asdfa')
end
alpha_t_minus_1 = reshape(alpha(t - 1, :), 1, []);
alpha(t, :) = (alpha_t_minus_1 * A) .* exp(logB(t, :));
scale(t) = sum(alpha(t, :));
    alpha(t, :) = alpha(t, :) / scale(t);
end

beta = zeros(T, M);
beta(T, :) = ones(1, M) / scale(T);
for t = T - 1:-1:1
    beta(t, :) = (beta(t + 1, :) .* exp(logB(t + 1, :))) * A';
    beta(t, :) = beta(t, :) / scale(t);
end

gamma = alpha .* beta;
gamma = normalize(gamma, 2);

xi = zeros(T - 1, M, M);
for t = 1:T - 1
    prob = (alpha(t, :)' .* A) .* (exp(logB(t + 1, :)) .* beta(t + 1, :));
    xi(t, :, :) = prob / sum(prob(:));
end

logP = sum(log(scale));
end


% ========== Decode Sequence ==========
function [gamma, states] = decodeHMM(data, pi0, A, lambda, dt)
[T, ~] = size(data);
M = size(lambda, 1);
logLambda = log(1 - exp(-lambda * dt));
logB = zeros(T, M);
for m = 1:M
    logB(:, m) = sum(data .* logLambda(m, :), 2);
end

delta = zeros(T, M);
psi = zeros(T, M);
delta(1, :) = log(pi0 + eps) + logB(1, :);
for t = 2:T
    for m = 1:M
        [delta(t, m), psi(t, m)] = max(delta(t - 1, :) + log(A(:, m)' + eps));
    end
    delta(t, :) = delta(t, :) + logB(t, :);
end

states = zeros(T, 1);
[~, states(T)] = max(delta(T, :));
for t = T - 1:-1:1
    states(t) = psi(t + 1, states(t + 1));
end

[~, ~, gamma, ~, ~] = fwdBwdPoisson(data, pi0, A, lambda, dt);
end


% ========== Normalize ==========
function out = normalize(mat, dim)
if nargin < 2
    dim = 1;
end
sums = sum(mat, dim);
out = bsxfun(@rdivide, mat, sums);
out(isnan(out)) = 1 / size(mat, dim);
end
