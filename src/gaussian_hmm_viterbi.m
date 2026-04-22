function stateSeq = gaussian_hmm_viterbi(observationMatrix, model)
% GAUSSIAN_HMM_VITERBI MAP hidden-state path for a Gaussian-emission HMM.
%
% Variables:
%   - observationMatrix: [T x D]
%   - model: struct from gaussian_hmm_fit with pi0, A, mu, varDiag, numStates
% Goal:
%   Decode the most likely state sequence given observations and fixed parameters.
% Returns:
%   - stateSeq: [T x 1] state indices in 1..K

X = double(observationMatrix);
K = model.numStates;
T = size(X, 1);
logB = gaussian_hmm_logB(X, model.mu, model.varDiag);
logA = log(model.A + realmin);
logPi = log(model.pi0(:) + realmin);

delta = zeros(T, K);
psi = zeros(T, K);
delta(1, :) = logPi' + logB(1, :);
for t = 2:T
    for j = 1:K
        scores = delta(t - 1, :)' + logA(:, j);
        [delta(t, j), psi(t, j)] = max(scores);
    end
    delta(t, :) = delta(t, :) + logB(t, :);
end
[~, pathEnd] = max(delta(end, :));
stateSeq = zeros(T, 1);
stateSeq(T) = pathEnd;
for t = (T - 1):-1:1
    stateSeq(t) = psi(t + 1, stateSeq(t + 1));
end
end
