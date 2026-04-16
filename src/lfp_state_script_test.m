%% LFP channel-inclusion sensitivity test (spontaneous only)
% Variables:
%   - lfpPath: path to spontaneous session folder containing lfp.mat
%   - areaDefs: brain-area depth definitions [um]
%   - bands: frequency bands [Hz] used for power integration
% Goal:
%   Compare band-power similarity for each brain area when the area signal is:
%   (1) one middle-depth channel, (2) mean of two channels near 1/3 and 2/3
%   depth positions, or (3) mean of all channels in that area.
% Notes:
%   Includes HMM-style state modeling on time-resolved band envelopes.


%% User settings
lfpPath = 'E:\Dropbox\Data\spontaneous\data\ag\ag112321_1';
doCleanArtifacts = false;
welchWindowSec = 2;
welchOverlapFrac = 0.5;
frameSizeSec = 0.01;
binMethod = 'cwt'; % 'cwt' or 'stft' if supported by lfp_bin_bandpower
maxStates = 10;
numFolds = 3;
numReplicates = 10;
modelModeIdx = 2; % 1=singleMiddle, 2=twoThirdsAverage, 3=allChannelsAverage

% Depths corresponding to brain areas [um]
areaDefs = struct( ...
    'name', {'M23', 'M56', 'CC', 'DS', 'VS'}, ...
    'range', {[0 500], [501 1240], [1241 1540], [1541 2700], [2701 3840]} ...
    );

bands = { ...
    'alpha', [8 13]; ...
    'beta', [13 30]; ...
    'lowGamma', [30 50]; ...
    'highGamma', [50 80] ...
    };

%% Load spontaneous LFP struct
load(fullfile(lfpPath, 'lfp.mat'), 'lfp');
assert(exist('lfp', 'var') == 1, 'lfp variable not found in lfp.mat');
assert(isstruct(lfp), 'lfp must be a struct array');

fs = lfp(1).samplerate;
for channelIdx = 1:numel(lfp)
    assert(isfield(lfp(channelIdx), 'data'), 'lfp.data missing');
    assert(isfield(lfp(channelIdx), 'depth'), 'lfp.depth missing');
    assert(isfield(lfp(channelIdx), 'samplerate'), 'lfp.samplerate missing');
    assert(lfp(channelIdx).samplerate == fs, 'All channels must share sampling rate');
end

%% Build per-area representative signals for each channel-selection mode
modeNames = {'singleMiddle', 'twoThirdsAverage', 'allChannelsAverage'};
numModes = numel(modeNames);
numAreas = numel(areaDefs);
numBands = size(bands, 1);

areaSignals = cell(numAreas, numModes);
areaMeta = repmat(struct('depths', [], 'selectedDepths', [], 'selectedIdx', []), numAreas, numModes);

for areaIdx = 1:numAreas
    thisRange = areaDefs(areaIdx).range;
    [areaSignals(areaIdx, :), areaMeta(areaIdx, :)] = get_area_signals_by_mode(lfp, thisRange);
end

%% Optional artifact cleaning and low-pass
if doCleanArtifacts
    for areaIdx = 1:numAreas
        for modeIdx = 1:numModes
            if ~isempty(areaSignals{areaIdx, modeIdx})
                areaSignals{areaIdx, modeIdx} = double(areaSignals{areaIdx, modeIdx});
                areaSignals{areaIdx, modeIdx} = clean_lfp_artifacts(areaSignals{areaIdx, modeIdx}, fs, ...
                    'spikeThresh', 4, ...
                    'spikeWinSize', 50, ...
                    'notchFreqs', [60 120 180], ...
                    'lowpassFreq', 300, ...
                    'useHampel', true, ...
                    'hampelK', 5, ...
                    'hampelNsigma', 3, ...
                    'detrendOrder', 'linear');
            end
        end
    end
else
    for areaIdx = 1:numAreas
        for modeIdx = 1:numModes
            if ~isempty(areaSignals{areaIdx, modeIdx})
                areaSignals{areaIdx, modeIdx} = double(areaSignals{areaIdx, modeIdx});
                areaSignals{areaIdx, modeIdx} = lowpass(areaSignals{areaIdx, modeIdx}, 300, fs);
            end
        end
    end
end

%% Compute band powers from Welch PSD integration
bandPowerMat = nan(numAreas, numModes, numBands);
for areaIdx = 1:numAreas
    for modeIdx = 1:numModes
        signal = areaSignals{areaIdx, modeIdx};
        if isempty(signal)
            continue;
        end
        bandPowerMat(areaIdx, modeIdx, :) = compute_band_powers_welch( ...
            signal, fs, bands, welchWindowSec, welchOverlapFrac);
    end
end

% Normalize within area and mode for shape comparisons
normBandPowerMat = bandPowerMat ./ sum(bandPowerMat, 3, 'omitnan');

%% Similarity metrics across channel-selection modes
% Pairwise mode comparisons: [single vs twoThirds], [single vs all], [twoThirds vs all]
modePairs = [1 2; 1 3; 2 3];
pairNames = {'single_vs_twoThirds', 'single_vs_all', 'twoThirds_vs_all'};
numPairs = size(modePairs, 1);

simCorr = nan(numAreas, numPairs);
simCosine = nan(numAreas, numPairs);
simMeanAbsPctDiff = nan(numAreas, numPairs);

for areaIdx = 1:numAreas
    for pairIdx = 1:numPairs
        modeA = modePairs(pairIdx, 1);
        modeB = modePairs(pairIdx, 2);

        powerA = squeeze(normBandPowerMat(areaIdx, modeA, :));
        powerB = squeeze(normBandPowerMat(areaIdx, modeB, :));
        if any(isnan(powerA)) || any(isnan(powerB))
            continue;
        end

        simCorr(areaIdx, pairIdx) = corr(powerA, powerB);
        simCosine(areaIdx, pairIdx) = dot(powerA, powerB) / (norm(powerA) * norm(powerB));
        simMeanAbsPctDiff(areaIdx, pairIdx) = mean(abs(powerA - powerB) ./ max(powerB, eps)) * 100;
    end
end

%% Print channel-selection summary
fprintf('\n=== Channel selection summary by area ===\n');
for areaIdx = 1:numAreas
    fprintf('\n%s [%d %d] um\n', areaDefs(areaIdx).name, areaDefs(areaIdx).range(1), areaDefs(areaIdx).range(2));
    for modeIdx = 1:numModes
        selDepths = areaMeta(areaIdx, modeIdx).selectedDepths;
        if isempty(selDepths)
            fprintf('  %-17s : no channels in range\n', modeNames{modeIdx});
        else
            fprintf('  %-17s : n=%d, selectedDepths=%s\n', modeNames{modeIdx}, ...
                numel(areaMeta(areaIdx, modeIdx).selectedIdx), mat2str(selDepths));
        end
    end
end

%% Print similarity summary table
fprintf('\n=== Similarity of normalized band-power profiles by area ===\n');
for pairIdx = 1:numPairs
    fprintf('\n-- %s --\n', pairNames{pairIdx});
    for areaIdx = 1:numAreas
        fprintf('%-4s  corr=%6.3f  cosine=%6.3f  meanAbsPctDiff=%7.2f%%\n', ...
            areaDefs(areaIdx).name, simCorr(areaIdx, pairIdx), simCosine(areaIdx, pairIdx), simMeanAbsPctDiff(areaIdx, pairIdx));
    end
end

%% Plot: normalized band-power profiles per area
figure(1101); clf;
tiledlayout(numAreas, 1, 'TileSpacing', 'compact', 'Padding', 'compact');
for areaIdx = 1:numAreas
    nexttile; hold on;
    for modeIdx = 1:numModes
        y = squeeze(normBandPowerMat(areaIdx, modeIdx, :));
        plot(1:numBands, y, 'o-', 'LineWidth', 2, 'DisplayName', modeNames{modeIdx});
    end
    xticks(1:numBands);
    xticklabels(bands(:, 1));
    ylabel('Normalized power');
    title(sprintf('%s band-power profile', areaDefs(areaIdx).name));
    yline(0, '--', 'Color', [0.7 0.7 0.7]);
    if areaIdx == 1
        legend('Location', 'best');
    end
end
xlabel('Band');

%% Plot: similarity heatmaps
figure(1102); clf;
tiledlayout(1, 3, 'TileSpacing', 'compact', 'Padding', 'compact');

nexttile;
imagesc(simCorr);
colorbar;
title('Correlation');
xticks(1:numPairs); xticklabels(pairNames);
yticks(1:numAreas); yticklabels({areaDefs.name});

nexttile;
imagesc(simCosine);
colorbar;
title('Cosine similarity');
xticks(1:numPairs); xticklabels(pairNames);
yticks(1:numAreas); yticklabels({areaDefs.name});

nexttile;
imagesc(simMeanAbsPctDiff);
colorbar;
title('Mean abs pct diff');
xticks(1:numPairs); xticklabels(pairNames);
yticks(1:numAreas); yticklabels({areaDefs.name});

%% Build time-resolved features and fit HMM per mode
% Feature layout per mode:
%   [area1_band1 ... area1_bandN area2_band1 ... areaN_bandN]
hmmResults = repmat(struct( ...
    'modeName', '', ...
    'featureMatrix', [], ...
    'timeBins', [], ...
    'bestNumStates', nan, ...
    'stateEstimates', [], ...
    'hmm', [], ...
    'likelihoods', []), numModes, 1);

for modeIdx = modelModeIdx
    binnedBandPowersMode = [];
    binnedEnvelopesMode = [];
    timeBinsRef = [];

    for areaIdx = 1:numAreas
        signal = areaSignals{areaIdx, modeIdx};
        if isempty(signal)
            continue;
        end

        [iBinnedPower, iBinnedEnvelopes, iTimeBins] = lfp_bin_bandpower( ...
            signal, fs, bands, frameSizeSec, binMethod);

        % Keep orientation consistent with lfp_state_script style:
        % rows = time bins, cols = features
        binnedBandPowersMode = [binnedBandPowersMode, iBinnedPower'];
        binnedEnvelopesMode = [binnedEnvelopesMode, iBinnedEnvelopes'];

        if isempty(timeBinsRef)
            timeBinsRef = iTimeBins(:);
        end
    end

    if isempty(binnedEnvelopesMode)
        continue;
    end

    idxFit = 1:size(binnedEnvelopesMode, 1);
    featureMatrix = zscore(binnedEnvelopesMode(idxFit, :));

    % Use cross-validated state selection helper when available.
    if exist('fit_hmm_crossval_cov_penalty', 'file') == 2
        hmmFitOpts = struct();
        hmmFitOpts.stateRange = 2:maxStates;
        hmmFitOpts.numFolds = numFolds;
        hmmFitOpts.minState = frameSizeSec; % minimum state duration in seconds
        hmmFitOpts.plotFlag = false;
        hmmFitOpts.penaltyType = 'normalized';

        [bestNumStates, stateEstimates, ~, likelihoods] = fit_hmm_crossval_cov_penalty( ...
            featureMatrix, hmmFitOpts);
    else
        % Fallback: select state count by BIC over diagonal-covariance GMMs.
        [bestNumStates, likelihoods] = select_num_states_bic(featureMatrix, maxStates, numReplicates);
        options = statset('MaxIter', 500);
        hmmFallback = fitgmdist(featureMatrix, bestNumStates, ...
            'Replicates', numReplicates, ...
            'CovarianceType', 'diagonal', ...
            'Options', options);
        stateEstimates = cluster(hmmFallback, featureMatrix);
    end

    options = statset('MaxIter', 500);
    hmm = fitgmdist(featureMatrix, bestNumStates, ...
        'Replicates', numReplicates, ...
        'CovarianceType', 'diagonal', ...
        'Options', options);
    stateEstimates = cluster(hmm, featureMatrix);

    hmmResults(modeIdx).modeName = modeNames{modeIdx};
    hmmResults(modeIdx).featureMatrix = featureMatrix;
    hmmResults(modeIdx).timeBins = timeBinsRef;
    hmmResults(modeIdx).bestNumStates = bestNumStates;
    hmmResults(modeIdx).stateEstimates = stateEstimates;
    hmmResults(modeIdx).hmm = hmm;
    hmmResults(modeIdx).likelihoods = likelihoods;
end

%% Print HMM summary
fprintf('\n=== HMM summary by channel-inclusion mode ===\n');
for modeIdx = modelModeIdx
    if isempty(hmmResults(modeIdx).stateEstimates)
        fprintf('%-17s : no valid features\n', modeNames{modeIdx});
        continue;
    end
    uniqueStates = unique(hmmResults(modeIdx).stateEstimates);
    fprintf('%-17s : bestNumStates=%d, nFrames=%d, statesFound=%s\n', ...
        modeNames{modeIdx}, ...
        hmmResults(modeIdx).bestNumStates, ...
        numel(hmmResults(modeIdx).stateEstimates), ...
        mat2str(uniqueStates'));
end

%% Plot HMM state mean power profiles (per mode)
for modeIdx = modelModeIdx
    if isempty(hmmResults(modeIdx).stateEstimates)
        continue;
    end

    featureMatrix = hmmResults(modeIdx).featureMatrix;
    stateEstimates = hmmResults(modeIdx).stateEstimates;
    uniqueStates = unique(stateEstimates);
    nState = numel(uniqueStates);
    nFeatExpected = numAreas * numBands;
    if size(featureMatrix, 2) ~= nFeatExpected
        % Skip plotting if features are incomplete due to missing area signals.
        continue;
    end

    figure(1200 + modeIdx); clf;
    tiledlayout(1, nState, 'TileSpacing', 'compact', 'Padding', 'compact');
    for stateIdx = 1:nState
        currentState = uniqueStates(stateIdx);
        meanPower = mean(featureMatrix(stateEstimates == currentState, :), 1);
        meanByBand = reshape(meanPower, numBands, numAreas)';

        nexttile; hold on;
        for areaIdx = 1:numAreas
            plot(1:numBands, meanByBand(areaIdx, :), 'o-', 'LineWidth', 1.5, ...
                'DisplayName', areaDefs(areaIdx).name);
        end
        plot(1:numBands, mean(meanByBand, 1), 'k-o', 'LineWidth', 3, 'DisplayName', 'meanAreas');
        xticks(1:numBands);
        xticklabels(bands(:, 1));
        xlim([0.5 numBands + 0.5]);
        yline(0, '--', 'Color', [0.6 0.6 0.6]);
        xlabel('Power band');
        ylabel('Z envelope');
        title(sprintf('%s state %d (n=%d)', modeNames{modeIdx}, currentState, sum(stateEstimates == currentState)));
        if stateIdx == 1
            legend('Location', 'best');
        end
    end
end

%% Plot HMM state timelines (per mode)
for modeIdx = modelModeIdx
    if isempty(hmmResults(modeIdx).stateEstimates)
        continue;
    end

    stateEstimates = hmmResults(modeIdx).stateEstimates;
    uniqueStates = unique(stateEstimates);
    nState = numel(uniqueStates);
    xTime = (0:numel(stateEstimates)-1) * frameSizeSec;
    colorMap = lines(max(nState, 3));

    figure(1300 + modeIdx); clf; hold on;
    for frameIdx = 1:numel(xTime)
        stateVal = stateEstimates(frameIdx);
        patch([xTime(frameIdx), xTime(frameIdx) + frameSizeSec, xTime(frameIdx) + frameSizeSec, xTime(frameIdx)], ...
            [0, 0, 1, 1], colorMap(stateVal, :), 'EdgeColor', 'none');
    end
    xlim([0 max(xTime) + frameSizeSec]);
    ylim([0 1]);
    yticks([]);
    xlabel('Time (s)');
    title(sprintf('State timeline: %s (nStates=%d)', modeNames{modeIdx}, hmmResults(modeIdx).bestNumStates));
end

%% Local functions
function [signalsByMode, metaByMode] = get_area_signals_by_mode(lfpStruct, depthRange)
% Variables:
%   - lfpStruct: struct array with fields data, depth, samplerate
%   - depthRange: two-element vector [minDepth maxDepth]
% Goal:
%   Return representative area signals for 3 channel-selection modes:
%   middle channel, two-third channels average, and all-channel average.
% Returns:
%   - signalsByMode: 1x3 cell of row vectors
%   - metaByMode: 1x3 struct with source depth/index metadata

depthAll = [lfpStruct.depth];
inAreaIdx = find(depthAll >= depthRange(1) & depthAll <= depthRange(2));

signalsByMode = {[], [], []};
metaByMode = repmat(struct('depths', [], 'selectedDepths', [], 'selectedIdx', []), 1, 3);
if isempty(inAreaIdx)
    return;
end

areaDepths = depthAll(inAreaIdx);
[areaDepths, sortIdx] = sort(areaDepths);
inAreaIdx = inAreaIdx(sortIdx);

channelData = cell2mat(arrayfun(@(i) lfpStruct(i).data(:)', inAreaIdx, 'UniformOutput', false)');
channelData = channelData';
% channelData shape: [nSamples x nChannels]

% Mode 1: single middle-most channel
midTarget = median(areaDepths);
singleIdx = nearest_depth_index(areaDepths, midTarget);
signalsByMode{1} = channelData(:, singleIdx);
metaByMode(1).depths = areaDepths;
metaByMode(1).selectedDepths = areaDepths(singleIdx);
metaByMode(1).selectedIdx = inAreaIdx(singleIdx);

% Mode 2: average of channels nearest first and second thirds of depth range
thirdTargets = [areaDepths(1) + (areaDepths(end) - areaDepths(1)) / 3, ...
                areaDepths(1) + 2 * (areaDepths(end) - areaDepths(1)) / 3];
thirdIdx = zeros(1, 2);
for idx = 1:2
    thirdIdx(idx) = nearest_depth_index(areaDepths, thirdTargets(idx));
end
thirdIdx = unique(thirdIdx, 'stable');
signalsByMode{2} = mean(channelData(:, thirdIdx), 2);
metaByMode(2).depths = areaDepths;
metaByMode(2).selectedDepths = areaDepths(thirdIdx);
metaByMode(2).selectedIdx = inAreaIdx(thirdIdx);

% Mode 3: average of all channels
signalsByMode{3} = mean(channelData, 2);
metaByMode(3).depths = areaDepths;
metaByMode(3).selectedDepths = areaDepths;
metaByMode(3).selectedIdx = inAreaIdx;
end

function idx = nearest_depth_index(depths, targetDepth)
% Variables:
%   - depths: sorted vector of available channel depths
%   - targetDepth: target depth to match
% Goal:
%   Return index of channel depth nearest targetDepth.

[~, idx] = min(abs(depths - targetDepth));
end

function bandPowers = compute_band_powers_welch(signal, fs, bands, windowSec, overlapFrac)
% Variables:
%   - signal: 1D LFP signal vector
%   - fs: sampling rate [Hz]
%   - bands: Nx2 cell array with band labels and [low high] limits
%   - windowSec: Welch window length [seconds]
%   - overlapFrac: overlap fraction for Welch segments
% Goal:
%   Compute integrated PSD power per band using Welch's method.

signal = double(signal(:));
numBands = size(bands, 1);

winLen = max(16, round(windowSec * fs));
noverlap = min(winLen - 1, round(winLen * overlapFrac));
nfft = max(256, 2 ^ nextpow2(winLen));

[pxx, f] = pwelch(signal, winLen, noverlap, nfft, fs);

bandPowers = nan(1, numBands);
for bandIdx = 1:numBands
    freqRange = bands{bandIdx, 2};
    freqMask = f >= freqRange(1) & f <= freqRange(2);
    bandPowers(bandIdx) = trapz(f(freqMask), pxx(freqMask));
end

function [bestNumStates, criterionVals] = select_num_states_bic(featureMatrix, maxStates, numReplicates)
% Variables:
%   - featureMatrix: rows are time bins, columns are features
%   - maxStates: max number of mixture states to test
%   - numReplicates: number of random initializations per fit
% Goal:
%   Select an HMM-like state count using minimum BIC of diagonal GMMs.
% Returns:
%   - bestNumStates: selected state count
%   - criterionVals: BIC values per candidate state count

criterionVals = nan(maxStates, 1);
options = statset('MaxIter', 500);

for stateCount = 1:maxStates
    try
        model = fitgmdist(featureMatrix, stateCount, ...
            'Replicates', numReplicates, ...
            'CovarianceType', 'diagonal', ...
            'Options', options);
        criterionVals(stateCount) = model.BIC;
    catch
        criterionVals(stateCount) = nan;
    end
end

validIdx = find(~isnan(criterionVals));
assert(~isempty(validIdx), 'State-selection failed for all candidate state counts.');
[~, minPos] = min(criterionVals(validIdx));
bestNumStates = validIdx(minPos);
end
end
