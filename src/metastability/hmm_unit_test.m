% TEST_FIT_POISSON_HMM Unit test for the fit_poisson_HMM function.
% This script generates synthetic Poisson-distributed spike data from
% a known HMM and verifies that fit_poisson_HMM recovers reasonable
% model parameters and hidden state sequences.

rng(42);  % For reproducibility

% Parameters for simulated HMM
T = 30000;         % number of time bins
N = 5;            % number of neurons
M_true = 3;       % true number of states

% Define true transition matrix and initial distribution
A_true = [0.90 0.05 0.05;
          0.05 0.90 0.05;
          0.05 0.05 0.90];
pi0_true = [0.7, 0.2, 0.1];

% Define firing rates (Hz) for each neuron in each state
lambda_true = [5 2 1 1 3;    % state 1
               1 4 2 2 5;    % state 2
               2 1 5 3 1];   % state 3
lambda_true = [20 15 10 10 30;
               10 25 15 15 35;
               15 10 35 20 10];
% Simulate hidden state sequence
z = zeros(T,1);
z(1) = find(mnrnd(1, pi0_true));
for t = 2:T
    z(t) = find(mnrnd(1, A_true(z(t-1),:)));
end

% Generate spike matrix (binary spikes in 1 ms bins)
dataMat = zeros(T, N);
for t = 1:T
    for n = 1:N
        rate = lambda_true(z(t), n) / 1000;  % convert to per-ms rate
        dataMat(t, n) = rand < rate;
    end
end

% Fit the model using the code under test
numStates = 3;
% for numStates = 4:8
numReps = 10;
model = fit_poisson_HMM(dataMat, binSize, numStates, numReps);

% Validate results
if isempty(fieldnames(model))
    disp(numStates)
    warning('Model fitting failed: returned empty model.');
end
% end
fprintf('Log-likelihood: %.2f\n', model.logL(end));
figure;
imagesc(model.gamma');
xlabel('Time'); ylabel('Inferred State');
title('Posterior State Probabilities');

figure;
plot(z, 'k-', 'DisplayName', 'True state');
hold on;
plot(model.stateSeq, 'r--', 'DisplayName', 'Inferred state');
xlabel('Time'); ylabel('State');
title('True vs Inferred State Sequence');
legend;

% Basic assertion (could expand with actual comparisons)
assert(numel(unique(model.stateSeq)) == numStates, 'Number of inferred states does not match input.');





%%
% TEST_POISSON_HMM_CLEAN A clean and stable unit test for fit_poisson_HMM.
% This script tests HMM inference on simulated neural spike data with clearly
% separable activity patterns across states.
rng(1);  % For reproducibility

% Settings
numStates = 3;
T_per_state = 100;
numReps = 10;
N = 5;  % number of neurons
binSize = 0.05;

% Define state-specific firing rates (Hz)
lambdaTrue = [
    20  5  2  1  8;
     1 25  2 10  4;
     5  5 20  2  1
];

% Build hidden state sequence z
z = repmat([1;2;3], T_per_state * numReps, 1);  % cycles through all 3
T = length(z);

% Generate spike count matrix
dataMat = poissrnd(lambdaTrue(z,:) * binSize);


%%
% Parameters
T = 4500;           % Number of time bins
N = 5;              % Number of neurons
numStates = 3;
numReps = 5;
binSize = 0.02;     % 20 ms bins (in seconds)

% Define distinct rate profiles per state (in Hz)
lambda_true = [20 2 18 1 10;     % State 1
               2 18 4 15 10;     % State 2
               8 8 8 8 30];      % State 3

% Construct ground-truth state sequence: 500 bins per state in order
z = [ones(50,1); 2*ones(50,1); 3*ones(50,1)];
z = repmat(z, 30, 1);
% z = [];
% for i = 1:100
%     z = [z; repmat(1,50,1); repmat(2,30,1); repmat(3,20,1)];
% end

% Simulate spike counts using Poisson draws per bin
dataMat = zeros(T, N);
for t = 1:T
    s = z(t);
    ratePerBin = lambda_true(s,:) * binSize;
    dataMat(t,:) = poissrnd(ratePerBin);
end

%% Fit Poisson HMM
model = fit_poisson_HMM(dataMat, binSize, numStates, numReps);

% Validate model
if isempty(fieldnames(model))
    error('Model fitting failed: returned empty model.');
end

assert(isreal(model.logL(end)) && isfinite(model.logL(end)), 'Invalid log-likelihood');
assert(all(isreal(model.lambda(:))) && all(model.lambda(:) >= 0), 'Invalid lambda values');

fprintf('✅ Model fitting succeeded. Final log-likelihood: %.2f\n', model.logL(end));

% Plot diagnostics
figure;
plot(model.logL, 'LineWidth', 2); xlabel('EM Iteration'); ylabel('Log-Likelihood');
title('Log-Likelihood over EM Iterations');

figure;
imagesc(model.gamma'); xlabel('Time'); ylabel('State');
title('Posterior State Probabilities (gamma)');

figure;
plot(z, 'k-', 'DisplayName', 'True'); hold on;
plot(model.stateSeq, 'r--', 'DisplayName', 'Inferred');
xlabel('Time'); ylabel('State'); legend;
title('True vs. Inferred State Sequence');












%%
% Toy parameters
binSize = 0.02;  % 20 ms
T = 10;
N = 3;
M = 2;

lambda = [10 5 15;   % state 1
          3 8 7];    % state 2
dt = binSize;

% Simulate easy data
z = [1 1 1 2 2 2 1 1 2 2]';
dataMat = poissrnd(lambda(z,:) * dt);

% Initial pi0 and A
pi0 = normalize(rand(1,M), 2);
A = normalize(rand(M,M), 2);

% Run fwd-bwd directly
[alpha, beta, gamma, xi, logP] = fwdBwdPoisson(dataMat, pi0, A, lambda, dt);

% Check outputs
disp(logP);
assert(isfinite(logP) && isreal(logP), 'logP invalid');
assert(all(isfinite(alpha(:))) && all(isreal(alpha(:))), 'alpha invalid');
assert(all(isfinite(gamma(:))) && all(isreal(gamma(:))), 'gamma invalid');





%%    TEST REAL DAtA
opts = neuro_behavior_options;
opts.frameSize = .05;
opts.minFiringRate = 1;
getDataType = 'spikes';
opts.collectFor = 5 * 60;
opts.firingRateCheckTime = 5 * 60;
get_standard_data

areas = {'M23', 'M56', 'DS', 'VS'};
idList = {idM23, idM56, idDS, idVS};

%%
hmm = fit_poisson_HMM_model_selection(dataMat(:,idM56), opts.frameSize, 3:7, 3);

%
fprintf('✅ Model fitting succeeded. Final log-likelihood: %.2f\n', hmm.logL(end));

% Plot diagnostics
figure;
plot(hmm.logL, 'LineWidth', 2); xlabel('EM Iteration'); ylabel('Log-Likelihood');
title('Log-Likelihood over EM Iterations');

figure;
imagesc(hmm.gamma'); xlabel('Time'); ylabel('State');
title('Posterior State Probabilities (gamma)');

figure;
plot(z, 'k-', 'DisplayName', 'True'); hold on;
plot(hmm.stateSeq, 'r--', 'DisplayName', 'Inferred');
xlabel('Time'); ylabel('State'); legend;
title('True vs. Inferred State Sequence');



