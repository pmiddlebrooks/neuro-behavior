%% Get LFP band powers
opts = neuro_behavior_options;
opts.minActTime = .16;
opts.collectStart = 0 * 60 * 60; % seconds
opts.collectFor = 60 * 60; % seconds
opts.frameSize = .1;

getDataType = 'lfp';
get_standard_data



%%
% Define frequency bands
freqBands = struct( ...
    'alpha', [8 13], ...
    'beta', [13 30], ...
    'lowGamma', [30 50], ...
    'highGamma', [50 80] ...
    );
freqIdx = repmat([1 2 3 4], 1, 4);
binnedBandPowers = [];
for iArea = 1 : 4
    iBinnedBandPowers = bin_bandpower_by_frames(lfpPerArea(:,iArea), freqBands, opts.fsLfp, opts.frameSize);
    binnedBandPowers = [binnedBandPowers, iBinnedBandPowers];
end




%% HMM model for state estimation
% Example inputs
maxStates = 20; % Maximum number of HMM states to evaluate
numFolds = 5;   % Number of folds for cross-validation
lambda = 10;

% Use previously computed binnedBandPowers
[bestNumStates, stateEstimates, hmmModels] = fit_hmm_crossval_cov_penalty(binnedBandPowers, maxStates, numFolds, lambda);

% Access optimal HMM properties
disp('Optimal Number of States:');
disp(size(optimalHMM.mu, 1));

%% HMM model for 3 states
    % Train the best model on the full dataset
    hmm = fitgmdist(binnedBandPowers, 3, 'Replicates', 10, 'CovarianceType', 'diagonal');

    % State estimations
    stateEstimates = cluster(hmm, binnedBandPowers);
%%
fun = @sRGB_to_OKLab;
colors = maxdistcolor(3, fun);
nSample = 1000;
x = 1:nSample;

% Create the bar plot
figure;
hold on;

for i = 1:nSample
    % Draw a segment for each state with its corresponding color
    patch([x(i)-0.5, x(i)+0.5, x(i)+0.5, x(i)-0.5], ...
          [0, 0, 1, 1], colors(stateEstimates(i),:), 'EdgeColor', 'none');
end
















function binnedBandPowers = bin_bandpower_by_frames(lfpData, freqBands, samplingFreq, frameSize)
% BIN_BANDPOWER_BY_FRAMES Computes band power and downsamples into bins.
%
% Inputs:
%   lfpData - Vector of LFP data (1D array).
%   samplingFreq - Sampling frequency of the LFP data in Hz.
%   frameSize - Bin size in seconds for downsampling the band powers.
%
% Outputs:
%   binnedBandPowers - Struct containing downsampled band powers for:
%                      - Alpha (8-13 Hz)
%                      - Beta (13-30 Hz)
%                      - Low Gamma (30-50 Hz)
%                      - High Gamma (50-80 Hz)


% % Define frequencies for wavelet transform
% freqRange = linspace(8, 100, 200); % Adjust range as needed
% numFreqs = length(freqRange);

% Perform wavelet transform
[waveletCoefs, frequencies] = cwt(lfpData, 'amor', samplingFreq, 'FrequencyLimits', [8 100]);

% Compute power at each frequency
waveletPower = abs(waveletCoefs).^2;
% Normalize (or z-score) the power
waveletPower = zscore(waveletPower);

% Preallocate band powers
bandPowers = [];

% Extract power for each frequency band
for bandName = fieldnames(freqBands)'
    band = freqBands.(bandName{1});
    bandIdx = frequencies > band(1) & frequencies < band(2);
    bandPowers = [bandPowers, mean(waveletPower(bandIdx, :), 1)'];
end

% Downsample into bins
frameSamples = round(frameSize * samplingFreq); % Samples per frame
numFrames = floor(length(lfpData) / frameSamples);
binnedBandPowers = [];

for iBand = 1 : size(bandPowers, 2)
    bandData = bandPowers(:,iBand);
    binnedBandPowers = [binnedBandPowers, arrayfun(@(i) ...
        mean(bandData((i-1)*frameSamples+1:i*frameSamples)), 1:numFrames)'];
end
end



function [bestNumStates, stateEstimates, hmmModels] = fit_hmm_crossval_cov_penalty(binnedBandPowers, maxStates, numFolds, lambda)
% FIT_HMM_CROSSVAL_COV_PENALTY Fits HMM and determines the optimal number of states using penalized log-likelihood.
%
% Inputs:
%   binnedBandPowers - Struct with binned band power data (alpha, beta, lowGamma, highGamma).
%   maxStates - Maximum number of states to evaluate.
%
% Outputs:
%   bestNumStates - Optimal number of states based on penalized likelihood.
%   stateEstimates - State assignments for the best model.
%   hmmModels - Cell array of trained HMMs for each number of states.

    numBins = size(binnedBandPowers, 1);
    foldSize = floor(numBins / numFolds);

    % Initialize storage
    penalizedLikelihoods = nan(maxStates, 1);
    hmmModels = cell(maxStates, 1);

    for numStates = 2:maxStates
        foldLikelihoods = zeros(numFolds, 1);
        for fold = 1:numFolds
            % Split data into training and test sets
            testIdx = (1:foldSize) + (fold-1)*foldSize;
            trainIdx = setdiff(1:numBins, testIdx);

            trainData = binnedBandPowers(trainIdx, :);
            testData = binnedBandPowers(testIdx, :);

            % Train HMM on training data
            hmm = fitgmdist(trainData, numStates, 'Replicates', 5, 'CovarianceType', 'diagonal');
            hmmModels{numStates} = hmm;

            % Evaluate log-likelihood on test data
            testLogLikelihood = sum(log(pdf(hmm, testData)));

            % Compute penalty based on covariance similarity
            covarianceMatrices = hmm.Sigma; % Covariance matrices for each state
            numCovariances = size(covarianceMatrices, 3); % Number of states
            similarityPenalty = 0;
            for i = 1:numCovariances
                for j = i+1:numCovariances
                    % Frobenius norm of the difference between covariance matrices
                    similarityPenalty = similarityPenalty + norm(covarianceMatrices(:,:,i) - covarianceMatrices(:,:,j), 'fro');
                end
            end
            similarityPenalty = similarityPenalty / (numCovariances * (numCovariances - 1)); % Average penalty

            % Penalize the log-likelihood
            foldLikelihoods(fold) = testLogLikelihood - lambda * similarityPenalty;
        end

        % Average cross-validated penalized likelihood
        penalizedLikelihoods(numStates) = mean(foldLikelihoods);
    end

    % Find the best number of states
    [~, bestNumStates] = max(penalizedLikelihoods);

    % Train the best model on the full dataset
    bestHMM = fitgmdist(featureMatrix, bestNumStates, 'Replicates', 10, 'CovarianceType', 'diagonal');

    % State estimations
    stateEstimates = cluster(bestHMM, featureMatrix);
    hmmModels{bestNumStates} = bestHMM;
end

