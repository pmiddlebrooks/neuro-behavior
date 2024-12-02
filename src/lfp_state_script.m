% This script more or less follows: 
% Akella et al 2024 bioRxiv: Deciphering neuronal variability across states reveals dynamic sensory encoding
% https://www.biorxiv.org/content/10.1101/2024.04.03.587408v2


%% Get LFP band powers
opts = neuro_behavior_options;
opts.minActTime = .16;
opts.collectStart = 0 * 60 * 60; % seconds
opts.collectFor = 60 * 60; % seconds
opts.frameSize = .1;

getDataType = 'lfp';
get_standard_data
getDataType = 'behavior';
get_standard_data
[dataBhv, bhvIDMat] = curate_behavior_labels(dataBhv, opts);


% for plotting consistency
monitorPositions = get(0, 'MonitorPositions');
if exist('/Users/paulmiddlebrooks/Projects/', 'dir')
    monitorPositions = flipud(monitorPositions);
end
monitorOne = monitorPositions(1, :); % Just use single monitor if you don't have second one
monitorTwo = monitorPositions(size(monitorPositions, 1), :); % Just use single monitor if you don't have second one

%%
% - get one lfp from each brain area (4 lfps)
% - Compute lfp power via wavelet
% - For each band, average the lfp powers within that band's range (4 bands X 4 areas = 16 features
% - zscore each band
% - bin lfp bands into frames by averaging lfp band power within each band
% - fit hmm on frame-binned lfp power

%% Definitions
% Define frequency bands
freqBands = struct( ...
    'alpha', [8 13], ...
    'beta', [13 30], ...
    'lowGamma', [30 50], ...
    'highGamma', [50 80] ...
    );
%%
bands = {'alpha', [8 13]; ...
         'beta', [13 30]; ...
         'lowGamma', [30 50]; ...
         'highGamma', [50 80]};
numBands = size(bands, 1);

%% Plot some lfp powers as Q/A
% Input Parameters
signal = lfpPerArea(1:10000,2); % Example LFP signal

% Compute the continuous wavelet transform
freqLimits = [min(cellfun(@(x) x(1), bands(:, 2))), max(cellfun(@(x) x(2), bands(:, 2)))];
[cfs, frequencies] = cwt(signal, 'amor', opts.fsLfp, 'FrequencyLimits', freqLimits);

% Preallocate band power matrix
time = linspace(0, length(signal)/opts.fsLfp, size(cfs, 2));
bandPowers = zeros(numBands, length(time));

% Compute power for each band
for i = 1:numBands
    % Extract the frequency range for the current band
    freqRange = bands{i, 2};
    
    % Identify the indices corresponding to the band frequencies
    freqIdx = frequencies >= freqRange(1) & frequencies <= freqRange(2);
    
    % Compute power by summing the squared magnitude of the wavelet coefficients
    bandPowers(i, :) = mean(abs(cfs(freqIdx, :)).^2, 1);
end

% Plot the heatmap of band powers
figure;
imagesc(time, 1:numBands, zscore(bandPowers, [], 2));
colormap('jet');
colorbar;
set(gca, 'YTick', 1:numBands, 'YTickLabel', bands(:, 1));
xlabel('Time (s)');
ylabel('Frequency Bands');
title('LFP Band Power Over Time');


%%

%% Get binned signals for fitting the hmm

freqIdx = repmat([1 2 3 4], 1, 4);
binnedBandPowers = [];
binnedEnvelopes = [];
for iArea = 1 : 4
    [iBinnedZPower, iBinnedEnvelopes, timeBins] = bin_bandpower_by_frames(lfpPerArea(:,iArea), opts.fsLfp, bands, opts.frameSize);
    binnedBandPowers = [binnedBandPowers, iBinnedZPower'];
    binnedEnvelopes = [binnedEnvelopes, iBinnedEnvelopes'];
end

%% plot results
plotRange = (1:400);
figure(1232);
subplot(2, 1, 1);
imagesc(binnedBandPowers(plotRange,5:8)');
colormap('jet');
colorbar;
xlabel('Time (s)');
ylabel('Frequency Bands');
title('Binned Z-Scored Power');

subplot(2, 1, 2);
plot(plotRange, binnedEnvelopes(plotRange,5:8)');
legend(bands(:, 1));
xlabel('Time (s)');
ylabel('Envelope Amplitude');
title('Binned Envelopes');

%%
figure(228);
hold on;
for i = 5:8
    plot(1:40, binnedBandPowers(1:40, i), 'DisplayName', sprintf('Column %d', i), 'lineWidth', 2);
end
hold off;
legend

%% are any of them alpha/beta low, gammas high? and vice versa?
idx = [1 2 3 4] + 12;
featureMatrix = binnedEnvelopes;
lowHigh = sum(featureMatrix(:,idx(1)) < 0 & featureMatrix(:,idx(2)) < 0 & featureMatrix(:,idx(3)) > 0 & featureMatrix(:,idx(4)) > 0) / size(featureMatrix, 1);
highLow = sum(featureMatrix(:,idx(1)) > 0 & featureMatrix(:,idx(2)) > 0 & featureMatrix(:,idx(3)) < 0 & featureMatrix(:,idx(4)) < 0) / size(featureMatrix, 1);
allLow = sum(featureMatrix(:,idx(1)) < 0 & featureMatrix(:,idx(2)) < 0 & featureMatrix(:,idx(3)) < 0 & featureMatrix(:,idx(4)) < 0) / size(featureMatrix, 1);
allHigh = sum(featureMatrix(:,idx(1)) > 0 & featureMatrix(:,idx(2)) > 0 & featureMatrix(:,idx(3)) > 0 & featureMatrix(:,idx(4)) > 0) / size(featureMatrix, 1);
fprintf('LowHigh: %.3f\tHighLow: %.3f\tAllLow: %.3f\tAllHigh: %.3f\tTotal: %.3f\t\n', lowHigh, highLow, allLow, allHigh, sum([lowHigh highLow allLow allHigh]))










%% HMM
                preInd = find(diff(bhvIDMat) ~= 0); % 1 frame prior to all behavior transitions
                transInd = sort([preInd; preInd+1]);
idxFit = transInd;

%% HMM model for state estimation
% Example inputs
maxStates = 8; % Maximum number of HMM states to evaluate
numFolds = 3;   % Number of folds for cross-validation
lambda = 1;

% Use previously computed binnedBandPowers
[bestNumStates, stateEstimates, hmmModels, likelihoods] = fit_hmm_crossval_cov_penalty(binnedEnvelopes(idxFit,:), maxStates, numFolds, lambda);

% Access optimal HMM properties
disp('Optimal Number of States:');
disp(bestNumStates);

%% HMM model for X states
    % Train the best model on the full dataset
    featureMatrix = zscore(binnedEnvelopes(idxFit,:));
    hmm = fitgmdist(featureMatrix, 5, 'Replicates', 10, 'CovarianceType', 'diagonal');

    % State estimations
    stateEstimates = cluster(hmm, featureMatrix);

    [uniqueIntegers, ~, indices] = unique(stateEstimates);
counts = accumarray(indices, 1)


%% Re-create fig 1.1 from poster
    % Create a maximized figure on the second monitor
    fig = figure(554); clf
    set(fig, 'Position', monitorTwo);
    nState = length(uniqueIntegers);
    [ax, pos] = tight_subplot(1, nState, [.08 .02], .1);
fun = @sRGB_to_OKLab;
colors = maxdistcolor(nState, fun);

    alphaInd = [1 5 9 13];
for i = 1 : nState
    meanPower = mean(featureMatrix(stateEstimates == i, :), 1);
    meanByBand = reshape(meanPower, 4, 4);
        axes(ax(i)); hold on;
    for j = 1:4
        plot(1:4, meanByBand(j,:), 'color', colors(i,:));
    end
plot(1:4, mean(meanByBand, 1), 'k', 'lineWidth', 3)
ylim([-1.1 1.1])
yline(0, '--', 'color', [.5 .5 .5], 'linewidth', 2)
end


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















function [binnedZPower, binnedEnvelopes, timeBins] = bin_bandpower_by_frames(signal, fs, bands, frameSize)
    % bin_bandpower_by_frames: Computes band powers and envelopes of LFP signals,
    % bins them by frames, and returns results.
    %
    % Inputs:
    %   - signal: LFP signal (1D array).
    %   - fs: Sampling rate (Hz).
    %   - bands: Cell array of frequency bands (e.g., {'alpha', [8 13]; 'beta', [13 30]}).
    %   - frameSize: Frame size in seconds for binning.
    %
    % Outputs:
    %   - binnedZPower: Z-scored band power binned by frames.
    %   - binnedEnvelopes: Band-specific envelopes binned by frames.
    %   - timeBins: Time vector for bin midpoints.

    % Step 1: Compute power using cwt
    freqLimits = [min(cellfun(@(x) x(1), bands(:, 2))), max(cellfun(@(x) x(2), bands(:, 2)))];
    [cfs, frequencies] = cwt(signal, 'amor', fs, 'FrequencyLimits', freqLimits);

    % Z-score the power at each frequency
    zScoredPower = zscore(abs(cfs).^2, 0, 2);

    % Step 2: Allocate matrices for band power and envelopes
    numBands = size(bands, 1);
    bandPowerSignals = zeros(numBands, length(signal));
    bandEnvelopes = zeros(numBands, length(signal));

    % Step 3: Compute power and envelopes for each band
    for i = 1:numBands
        % Get the frequency range for the current band
        freqRange = bands{i, 2};

        % Identify indices corresponding to the band frequencies
        freqIdx = frequencies >= freqRange(1) & frequencies <= freqRange(2);

        % Average z-scored power within the band
        bandPowerSignals(i, :) = mean(zScoredPower(freqIdx, :), 1);

        % Compute the envelope using the Hilbert transform
        bandEnvelopes(i, :) = abs(hilbert(bandPowerSignals(i, :)));
    end

    % Step 4: Bin power and envelopes by frames
    frameSamples = frameSize * fs;
    numFrames = floor(length(signal) / frameSamples);

    % Preallocate binned outputs
    binnedZPower = zeros(numBands, numFrames);
    binnedEnvelopes = zeros(numBands, numFrames);
    timeBins = zeros(1, numFrames);

    for frameIdx = 1:numFrames
        % Frame indices
        startIdx = (frameIdx - 1) * frameSamples + 1;
        endIdx = startIdx + frameSamples - 1;

        % Extract frame
        frameZPower = bandPowerSignals(:, startIdx:endIdx);
        frameEnvelope = bandEnvelopes(:, startIdx:endIdx);

        % Average across the frame
        binnedZPower(:, frameIdx) = mean(frameZPower, 2);
        binnedEnvelopes(:, frameIdx) = mean(frameEnvelope, 2);

        % Compute bin midpoint time
        timeBins(frameIdx) = (startIdx + endIdx) / (2 * fs);
    end
end


% function binnedBandPowers = bin_bandpower_by_frames(lfpData, freqBands, samplingFreq, frameSize)
% % BIN_BANDPOWER_BY_FRAMES Computes band power and downsamples into bins.
% %
% % Inputs:
% %   lfpData - Vector of LFP data (1D array).
% %   samplingFreq - Sampling frequency of the LFP data in Hz.
% %   frameSize - Bin size in seconds for downsampling the band powers.
% %
% % Outputs:
% %   binnedBandPowers - Struct containing downsampled band powers for:
% %                      - Alpha (8-13 Hz)
% %                      - Beta (13-30 Hz)
% %                      - Low Gamma (30-50 Hz)
% %                      - High Gamma (50-80 Hz)
% 
% 
% % % Define frequencies for wavelet transform
% % freqRange = linspace(8, 100, 200); % Adjust range as needed
% % numFreqs = length(freqRange);
% 
% % Perform wavelet transform
% [waveletCoefs, frequencies] = cwt(lfpData, 'amor', samplingFreq, 'FrequencyLimits', [8 100]);
% 
% % Compute power at each frequency
% waveletPower = abs(waveletCoefs).^2;
% % Normalize (or z-score) the power
% waveletPower = zscore(waveletPower);
% 
% % Preallocate band powers
% bandPowers = [];
% 
% % Extract power for each frequency band
% for bandName = fieldnames(freqBands)'
%     band = freqBands.(bandName{1});
%     bandIdx = frequencies > band(1) & frequencies < band(2);
%     bandPowers = [bandPowers, mean(waveletPower(bandIdx, :), 1)'];
% end
% 
% % Downsample into bins
% frameSamples = round(frameSize * samplingFreq); % Samples per frame
% numFrames = floor(length(lfpData) / frameSamples);
% binnedBandPowers = [];
% 
% for iBand = 1 : size(bandPowers, 2)
%     bandData = bandPowers(:,iBand);
%     binnedBandPowers = [binnedBandPowers, arrayfun(@(i) ...
%         mean(bandData((i-1)*frameSamples+1:i*frameSamples)), 1:numFrames)'];
% end
% end



function [bestNumStates, stateEstimates, hmmModels, penalizedLikelihoods] = fit_hmm_crossval_cov_penalty(featureMatrix, maxStates, numFolds, lambda)
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
featureMatrix = zscore(featureMatrix);
    numBins = size(featureMatrix, 1);
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

            trainData = featureMatrix(trainIdx, :);
            testData = featureMatrix(testIdx, :);

            % Train HMM on training data
options = statset('MaxIter', 500);

            hmm = fitgmdist(trainData, numStates, 'Replicates', 5, 'CovarianceType', 'full', 'Options', options);
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

