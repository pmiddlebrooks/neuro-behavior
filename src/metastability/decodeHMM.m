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
