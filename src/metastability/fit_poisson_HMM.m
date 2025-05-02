function model = fit_poisson_HMM(dataMat, binSize, numStates, numReps)
% FIT_POISSON_HMM Fit a Poisson Hidden Markov Model (HMM) to spike train data.
% This function performs multiple EM initializations and returns the best-fit model.
% Inputs:
%   dataMat: [T x N] binary matrix of spikes (1 ms bins)
%   numStates: number of HMM hidden states to fit
%   numReps: number of random initializations of EM to run
% Output:
%   model: struct containing estimated HMM parameters and decoded states

[T, N] = size(dataMat);
% dt = 0.001;
dt = binSize;
% dataMat(dataMat > 1) = 1;  % Ensure binary spike counts
dataMat = double(dataMat);  % Convert to double to avoid integer multiplication issues

bestLL = -inf;
model = struct();  % Ensure model is initialized
modelFound = false;

for rep = 1:numReps
    % Random initialization of model parameters
    pi0 = normalize(rand(1, numStates), 2);
    A = normalize(rand(numStates, numStates), 2);
    lambda = rand(numStates, N) * max(mean(dataMat)) / dt;

    try
        % Run EM algorithm for this initialization
        [pi0_new, A_new, lambda_new, ll] = emPoissonHMM(dataMat, pi0, A, lambda, dt);

        % Check and store best model
        if all(isfinite(ll)) && ll(end) > bestLL
            bestLL = ll(end);
            model.pi0 = pi0_new;
            model.A = A_new;
            model.lambda = lambda_new;
            model.logL = ll;
            modelFound = true;
        end
    catch
        warning('EM failed on rep %d — skipping.', rep);
        continue;
    end
end

if ~modelFound
    warning('All EM initializations failed to produce a valid model. Returning empty model.');
    model = struct();
    return;
end

% Decode the state sequence using Viterbi and posterior probabilities
[gamma, states] = decodeHMM(dataMat, model.pi0, model.A, model.lambda, dt);
model.gamma = gamma;
model.stateSeq = states;
end

function [pi0, A, lambda, logL] = emPoissonHMM(data, pi0, A, lambda, dt)
% EMPOISSONHMM Run the Expectation-Maximization (EM) algorithm for Poisson HMM.
% Inputs: initialized HMM parameters
% Outputs: updated parameters and log-likelihood per iteration

maxIter = 100;
tol = 1e-4;
[T, N] = size(data);
M = size(A, 1);
logL = zeros(maxIter,1);

for iter = 1:maxIter
    [alpha, beta, gamma, xi, logP] = fwdBwdPoisson(data, pi0, A, lambda, dt);
    logL(iter) = logP;

    % M-step updates
    pi0 = gamma(1,:);
    A = normalize(squeeze(sum(xi,1)), 2);  % transition matrix

    % Update emission rates for each state
    for m = 1:M
        weightedSum = sum(data .* gamma(:,m), 1);
        totalWeight = sum(gamma(:,m));
        % lambda(m,:) = -log(1 - (weightedSum / totalWeight)) / dt;
ratio = weightedSum / totalWeight;
ratio(ratio >= 1) = 1 - 1e-6;      % clamp maximum
ratio(ratio <= 0) = 1e-6;          % clamp minimum
lambda(m,:) = -log(1 - ratio) / dt;
    end

    % Check for convergence
    if iter > 1 && abs(logL(iter) - logL(iter-1)) < tol
        logL = logL(1:iter);
        break;
    end
end
end

% function [alpha, beta, gamma, xi, logP] = fwdBwdPoisson(data, pi0, A, lambda, dt)
% % FWDBWDPOISSON Run forward-backward algorithm for Poisson HMM.
% % Returns posterior probabilities and log-likelihood.
% 
% [T, N] = size(data);
% M = size(lambda,1);
% data = double(data);
% logLambda = log(1 - exp(-lambda * dt));
% 
% % Compute log-emission probabilities
% logB = zeros(T, M);
% for m = 1:M
%     logB(:,m) = sum(data .* logLambda(m,:), 2);
% end
% 
% % Forward pass
% alpha = zeros(T, M);
% scale = zeros(T,1);
% alpha(1,:) = pi0 .* exp(logB(1,:));
% scale(1) = sum(alpha(1,:));
% alpha(1,:) = alpha(1,:) / scale(1);
% 
% for t = 2:T
%     alpha_prev = reshape(alpha(t - 1, :), 1, []);
%     alpha(t,:) = (alpha_prev * A) .* exp(logB(t,:));
%     scale(t) = sum(alpha(t,:));
%     alpha(t,:) = alpha(t,:) / scale(t);
% end
% 
% % Backward pass
% beta = zeros(T, M);
% beta(T,:) = ones(1,M) / scale(T);
% for t = T-1:-1:1
%     beta(t,:) = (beta(t+1,:) .* exp(logB(t+1,:))) * A';
%     beta(t,:) = beta(t,:) / scale(t);
% end
% 
% % Compute posteriors
% gamma = alpha .* beta;
% gamma = normalize(gamma, 2);
% 
% % Compute pairwise state transition probabilities (xi)
% xi = zeros(T-1, M, M);
% for t = 1:T-1
%     prob = (alpha(t,:)' .* A) .* (exp(logB(t+1,:)) .* beta(t+1,:));
%     xi(t,:,:) = prob / sum(prob(:));
% end
% 
% % Total log-likelihood
% logP = sum(log(scale));
% end

function [gamma, states] = decodeHMM(data, pi0, A, lambda, dt)
% DECODEHMM Decode most likely state sequence and posterior state probabilities.

[T, N] = size(data);
M = size(lambda,1);
logLambda = log(1 - exp(-lambda * dt));

% Compute log-emission probabilities
logB = zeros(T, M);
for m = 1:M
    logB(:,m) = sum(data .* logLambda(m,:), 2);
end

% Viterbi algorithm for most likely state sequence
delta = zeros(T, M);
psi = zeros(T, M);
delta(1,:) = log(pi0 + eps) + logB(1,:);
for t = 2:T
    for m = 1:M
        [delta(t,m), psi(t,m)] = max(delta(t-1,:) + log(A(:,m)' + eps));
    end
    delta(t,:) = delta(t,:) + logB(t,:);
end

% Backtrace
states = zeros(T,1);
[~, states(T)] = max(delta(T,:));
for t = T-1:-1:1
    states(t) = psi(t+1, states(t+1));
end

% Posterior probabilities via forward-backward
[~, ~, gamma, ~, ~] = fwdBwdPoisson(data, pi0, A, lambda, dt);
end

function out = normalize(mat, dim)
% NORMALIZE Normalize matrix along given dimension
if nargin < 2
    dim = 1;
end
sums = sum(mat, dim);
out = bsxfun(@rdivide, mat, sums);
out(isnan(out)) = 1 / size(mat,dim);
end
