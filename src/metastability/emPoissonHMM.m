function [pi0, A, lambda, logL] = emPoissonHMM(data, pi0, A, lambda, dt)
% EMPOISSONHMM Run the Expectation-Maximization (EM) algorithm for Poisson HMM.
% Inputs: initialized HMM parameters
% Outputs: updated parameters and log-likelihood per iteration

maxIter = 100;
tol = 1e-4;
[T, N] = size(data);
M = size(A, 1);
logL = zeros(maxIter,1);

data = double(data);

for iter = 1:maxIter
    [alpha, beta, gamma, xi, logP] = fwdBwdPoisson(data, pi0, A, lambda, dt);
    logL(iter) = logP;

    % M-step updates
    pi0 = normalize_to_1(gamma(1,:), 2);
    if any(~isfinite(xi(:)))
        warning('Non-finite xi detected in EM');
        xi(~isfinite(xi)) = 1e-6;
    end
    A = normalize_to_1(squeeze(sum(xi,1)), 2);  % transition matrix

    if any(pi0<0) || any(A(:)<0)
        disp('whoooasaa... normalize it not z-socre it')
    end

    % Update emission rates for each state with occupancy safeguard
    for m = 1:M
        weight = gamma(:,m)';
        if sum(weight) < 1e-3  % state never occupied
            lambda(m,:) = mean(data, 1) / dt;  % fallback
            warning('State %d has low occupancy; fallback lambda used.', m);
        else
            weightedSum = sum(weight .* data)';
            lambda(m,:) = weightedSum / (sum(weight) * dt);
        end
    end

    % Clean up any numeric artifacts
    lambda(~isfinite(lambda)) = 1;
    lambda(lambda < 0 | ~isreal(lambda)) = 1;

    % Check for convergence
    if iter > 1 && abs(logL(iter) - logL(iter-1)) < tol
        logL = logL(1:iter);
        break;
    end
end
end
