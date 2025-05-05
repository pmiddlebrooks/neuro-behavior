function model = fit_poisson_HMM(dataMat, binSize, numStates, numReps)
% FIT_POISSON_HMM Fit a Poisson Hidden Markov Model (HMM) to spike train data.
% 
% Inputs:
%   dataMat   : [T x N] spike count matrix; rows = time bins, columns = neurons
%   binSize   : bin size in seconds (e.g., 0.05 for 50 ms bins)
%   numStates : number of HMM hidden states to fit
%   numReps   : number of EM initializations (repetitions)
%
% Output:
%   model: struct containing estimated HMM parameters and decoded states
%       .pi0     : 1 x numStates vector, initial state probabilities
%       .A       : numStates x numStates transition matrix
%       .lambda  : numStates x N matrix of firing rates (Hz) per state and neuron
%       .logL    : EM log-likelihood trajectory
%       .gamma   : posterior probabilities of each state per time bin
%       .stateSeq: most likely sequence of states (Viterbi path)

[T, N] = size(dataMat);
dt = binSize;  % bin size in seconds

bestLL = -inf;
model = struct();  % Ensure model is initialized
modelFound = false;
validLL = -Inf(numReps, 1);
validModels = cell(numReps, 1);

for rep = 1:numReps
    % Random initialization of model parameters
    pi0 = normalize_to_1(rand(1, numStates), 2);
    A = normalize_to_1(rand(numStates, numStates), 2);

    % Manual forced distinct lambda initialization for testing
    lambda = zeros(numStates, N);
    % lambda(1,:) = 10 * [1 0.5 0.2 0.1 1];
    % lambda(2,:) = 10 * [0.2 2.5 0.2 1.5 0.4];
    % lambda(3,:) = 10 * [0.8 0.9 4.0 0.3 0.1];
    lambda = lambda(1:numStates, :);  % Clamp in case numStates < 3

    try
        % Run EM algorithm for this initialization
        [pi0_new, A_new, lambda_new, ll] = emPoissonHMM(dataMat, pi0, A, lambda, dt);

        % Ensure model is valid before accepting
        if any(~isfinite(A_new(:))) || any(~isfinite(lambda_new(:))) || ...
           any(lambda_new(:) < 0) || any(isnan(pi0_new)) || ...
           any(~isfinite(ll)) || any(imag(ll) ~= 0)

            warning('Rep %d produced invalid model parameters. Skipping.', rep);
            continue;
        end

        % Check and store best model
        if all(isfinite(ll)) && isreal(ll(end))
            validModels{rep}.pi0 = pi0_new;
            validModels{rep}.A = A_new;
            validModels{rep}.lambda = lambda_new;
            validModels{rep}.logL = ll;
            validLL(rep) = ll(end);
        else
            warning('Rep %d failed: logL not finite or complex.', rep);
            validLL(rep) = -Inf;
        end
    catch
        warning('EM failed on rep %d — skipping.', rep);
        continue;
    end
end

[~, bestIdx] = max(validLL);
if isfinite(validLL(bestIdx))
    model = validModels{bestIdx};
    modelFound = true;
else
    warning('All EM initializations failed. Returning empty model.');
    model = struct();
end

% Decode the state sequence using Viterbi and posterior probabilities
[gamma, states] = decodeHMM(dataMat, model.pi0, model.A, model.lambda, dt);
model.gamma = gamma;
model.stateSeq = states;
end
