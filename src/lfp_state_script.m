%% Get LFP band powers
opts = neuro_behavior_options;
opts.minActTime = .16;
opts.collectStart = 0 * 60 * 60; % seconds
opts.collectFor = 60 * 60; % seconds
opts.frameSize = .1;

getDataType = 'lfp';
get_standard_data



%%

binnedBandPowers = bin_bandpower_by_frames(lfpPerArea(:,1), opts.fsLfp, opts.frameSize);



%% HMM model for state estimation
% Example inputs
maxStates = 10; % Maximum number of HMM states to evaluate
numFolds = 5;   % Number of folds for cross-validation

% Use previously computed binnedBandPowers
optimalHMM = estimate_hmm_states(binnedBandPowers, maxStates, numFolds);

% Access optimal HMM properties
disp('Optimal Number of States:');
disp(size(optimalHMM.mu, 1));






function binnedBandPowers = bin_bandpower_by_frames(lfpData, samplingFreq, frameSize)
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

    % Define frequency bands
    freqBands = struct( ...
        'alpha', [8 13], ...
        'beta', [13 30], ...
        'lowGamma', [30 50], ...
        'highGamma', [50 80] ...
    );

    % Define frequencies for wavelet transform
    freqRange = linspace(8, 80, 100); % Adjust range as needed
    numFreqs = length(freqRange);
    
    % Perform wavelet transform
    waveletCoefs = cwt(lfpData, 'amor', samplingFreq, 'FrequencyLimits', [2 100]);
    
    % Compute power at each frequency
    waveletPower = abs(waveletCoefs).^2;

    % Preallocate band powers
    bandPowers = struct();

    % Extract power for each frequency band
    for bandName = fieldnames(freqBands)'
        band = freqBands.(bandName{1});
        bandIdx = freqRange > band(1) & freqRange < band(2);
        bandPowers.(bandName{1}) = mean(waveletPower(bandIdx, :), 1);
    end

    % Downsample into bins
    frameSamples = round(frameSize * samplingFreq); % Samples per frame
    numFrames = floor(length(lfpData) / frameSamples);
    binnedBandPowers = struct();

    for bandName = fieldnames(bandPowers)'
        bandData = bandPowers.(bandName{1});
        binnedBandPowers.(bandName{1}) = arrayfun(@(i) ...
            mean(bandData((i-1)*frameSamples+1:i*frameSamples)), 1:numFrames);
    end
end



function optimalHMM = estimate_hmm_states(binnedBandPowers, maxStates, numFolds)

% Explanation:

% Input Preparation:
% Combine binned band power values into a feature matrix, where each row
% represents a time bin and columns represent the alpha, beta, low gamma,
% and high gamma bands.  

% HMM Training:
% Use fitgmdist (Gaussian Mixture Model) to approximate the HMM emission
% probabilities. States are treated as clusters with diagonal covariance.  

% Cross-Validation:
% Split data into training and testing folds. Train HMM on training data
% and evaluate likelihood on testing data. 
% Penalize the likelihood for state definitions with high similarity using
% the pairwise distance between state means. 

% State Estimation:
% After finding the optimal number of states, train the best HMM on the
% full dataset and output state assignments. 
% This approach evaluates the number of states while accounting for the
% distinctiveness of state definitions. Adjust maxStates and numFolds as
% needed based on your dataset size.  


% ESTIMATE_HMM_STATES Estimates states using HMM based on binned band power data.
%
% Inputs:
%   binnedBandPowers - Struct containing binned band powers for Alpha, Beta, Low Gamma, High Gamma.
%   maxStates - Maximum number of states to evaluate for the HMM.
%   numFolds - Number of folds for cross-validation.
%
% Outputs:
%   optimalHMM - HMM model with the optimal number of states based on penalized likelihood.

    % Combine binned band powers into a matrix
    bands = fieldnames(binnedBandPowers);
    dataMatrix = cell2mat(struct2cell(binnedBandPowers)');
    
    % Standardize data (mean = 0, std = 1)
    standardizedData = (dataMatrix - mean(dataMatrix, 2)) ./ std(dataMatrix, [], 2);
    
    % Prepare for cross-validation
    numSamples = size(standardizedData, 2);
    cvIndices = crossvalind('Kfold', numSamples, numFolds);
    logLikelihoods = zeros(maxStates, numFolds);
    penalties = zeros(maxStates, numFolds);
    
    for numStates = 1:maxStates
        for fold = 1:numFolds
            % Split data into training and testing sets
            trainData = standardizedData(:, cvIndices ~= fold);
            testData = standardizedData(:, cvIndices == fold);
            
            % Fit HMM to training data
            hmmModel = fitgmdist(trainData', numStates, 'CovarianceType', 'diagonal', ...
                'RegularizationValue', 1e-6, 'Options', statset('MaxIter', 500, 'TolFun', 1e-5));
            
            % Evaluate on test data
            testLikelihood = sum(log(pdf(hmmModel, testData')));
            logLikelihoods(numStates, fold) = testLikelihood;
            
            % Compute penalty for similar state definitions
            stateMeans = hmmModel.mu;
            stateDistances = pdist(stateMeans, 'euclidean');
            penalty = sum(1 ./ (stateDistances + eps)); % Avoid division by zero
            penalties(numStates, fold) = penalty;
        end
    end
    
    % Compute penalized likelihood
    meanLogLikelihood = mean(logLikelihoods, 2);
    meanPenalty = mean(penalties, 2);
    penalizedLikelihood = meanLogLikelihood - meanPenalty;
    
    % Determine optimal number of states
    [~, optimalNumStates] = max(penalizedLikelihood);
    
    % Fit the optimal HMM model
    optimalHMM = fitgmdist(standardizedData', optimalNumStates, 'CovarianceType', 'diagonal', ...
        'RegularizationValue', 1e-6, 'Options', statset('MaxIter', 500, 'TolFun', 1e-5));
end

