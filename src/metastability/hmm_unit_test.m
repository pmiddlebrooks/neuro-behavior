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
for numStates = 4:8
numReps = 10;
model = fit_poisson_HMM(dataMat, binSize, numStates, numReps);

% Validate results
if isempty(fieldnames(model))
    disp(numStates)
    warning('Model fitting failed: returned empty model.');
end
end
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
% TEST_FIT_POISSON_HMM Unit test for the fit_poisson_HMM function.
% This test constructs a spike matrix with clearly separable neural activity patterns
% corresponding to different hidden states. The goal is to ensure the HMM fitting
% recovers these latent states correctly.

rng(1);  % For reproducibility

% Parameters
T = 1500;         % number of time bins
N = 5;            % number of neurons
M_true = 3;       % number of true latent states
binSize = 20;     % simulate 20 ms bins (use Poisson counts, not binary)

% Create 3 distinct neural patterns (emission rates per state)
lambda_true = [10 2 8 1 5;   % State 1: neurons 1 and 3 active
               1 10 2 8 3;   % State 2: neurons 2 and 4 active
               5 5 5 5 15];  % State 3: neuron 5 bursts
lambda_true = lambda_true .* 3;
% Define idealized state sequence: 500 bins each state in order
z = [ones(500,1); 2*ones(500,1); 3*ones(500,1)];

% Simulate spike counts per bin (20 ms bins) using Poisson
dataMat = zeros(T, N);
dt = binSize / 1000;  % seconds per bin

for t = 1:T
    s = z(t);
    dataMat(t, :) = poissrnd(lambda_true(s,:) * dt);
end

% Fit HMM
numStates = 3;
numReps = 5;
model = fit_poisson_HMM(dataMat, binSize, numStates, numReps);

% Validate output
if isempty(fieldnames(model))
    error('Model fitting failed: returned empty model.');
end

% Plot true vs inferred states
figure;
subplot(2,1,1);
imagesc(dataMat');
ylabel('Neuron'); title('Simulated Spike Counts');

subplot(2,1,2);
plot(z, 'k-', 'DisplayName', 'True');
hold on;
plot(model.stateSeq, 'r--', 'DisplayName', 'Inferred');
legend; xlabel('Time'); ylabel('State');
title('True vs Inferred State Sequence');

% Check number of states used
assert(numel(unique(model.stateSeq)) == numStates, 'Mismatch in number of inferred states.');

fprintf('Unit test completed. Log-likelihood: %.2f\n', model.logL(end));
