function [alpha, beta, gamma, xi, logP] = fwdBwdPoisson(data, pi0, A, lambda, dt)
[T, N] = size(data);
M = size(lambda,1);
logLambda = log(1 - exp(-lambda * dt));

logB = zeros(T, M);
for m = 1:M
    logB(:,m) = sum(data .* logLambda(m,:), 2);
end

alpha = zeros(T, M);
scale = zeros(T,1);
alpha(1,:) = pi0 .* exp(logB(1,:));
scale(1) = sum(alpha(1,:));
alpha(1,:) = alpha(1,:) / scale(1);

for t = 2:T
    alpha_prev = reshape(alpha(t - 1, :), 1, []);
    alpha(t,:) = (alpha_prev * A) .* exp(logB(t,:));
    scale(t) = sum(alpha(t,:));
    alpha(t,:) = alpha(t,:) / scale(t);
end

beta = zeros(T, M);
beta(T,:) = ones(1,M) / scale(T);
for t = T-1:-1:1
    beta(t,:) = (beta(t+1,:) .* exp(logB(t+1,:))) * A';
    beta(t,:) = beta(t,:) / scale(t);
end

gamma = alpha .* beta;
gamma = normalize(gamma, 2);

xi = zeros(T-1, M, M);
for t = 1:T-1
    prob = (alpha(t,:)' .* A) .* (exp(logB(t+1,:)) .* beta(t+1,:));
    xi(t,:,:) = prob / sum(prob(:));
end

logP = sum(log(scale));
end
