%% LFP channel-inclusion sensitivity test (spontaneous only)
% Variables:
%   - lfpPath: path to spontaneous session folder containing lfp.mat
%   - areaDefs: brain-area depth definitions [um]
%   - bands: frequency bands [Hz] (Akella et al. 2024 bioRxiv for HMM; Welch summary uses same bands)
% Goal:
%   Compare band-power similarity for each brain area when the area signal is:
%   (1) one middle-depth channel, (2) mean of two channels near 1/3 and 2/3
%   depth positions, or (3) mean of all channels in that area.
% Notes:
%   HMM: one LFP per area (M23, M56, DS, VS), Butterworth bandpass + Hilbert envelope,
%   non-overlapping bins (hmmBinSizeSec), diagonal Gaussian-emission HMM (Baum-Welch).


%% User settings
paths = get_paths;
% lfpPath = fullfile(paths.dropPath, 'spontaneous\data\ag\ag112321_1');
lfpPath = fullfile(paths.dropPath, 'spontaneous/data/ag/ag112321_1');
lfpPath = fullfile(paths.dropPath, 'spontaneous/data/ey/ey042822');

doCleanArtifacts = false;
welchWindowSec = .5;
welchOverlapFrac = 0.5;
hmmBinSizeSec = 0.02; % non-overlapping bins for HMM features (20 ms)
bandpassOrder = 11; % Akella et al.: 11th-order Butterworth per band
maxStates = 10;
numFolds = 3;
hmmNumRestarts = 5;
hmmMaxIter = 200;
modelModeIdx = 2; % 1=singleMiddle, 2=twoThirdsAverage, 3=allChannelsAverage
nMinPlot = 5; % first minutes to visualize in spectrogram heatmaps
saveHmmResults = true;

% Depths corresponding to brain areas [um] — four sample areas (no CC)
areaDefs = struct( ...
    'name', {'M23', 'M56', 'DS', 'VS'}, ...
    'range', {[0 500], [501 1240], [1541 2700], [2701 3840]} ...
    );

% Akella et al. 2024 bioRxiv (LFP oscillation states)
bands = { ...
    'theta', [3 8]; ...
    'beta', [10 30]; ...
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

% Print channel-selection summary
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

% Print similarity summary table
fprintf('\n=== Similarity of normalized band-power profiles by area ===\n');
for pairIdx = 1:numPairs
    fprintf('\n-- %s --\n', pairNames{pairIdx});
    for areaIdx = 1:numAreas
        fprintf('%-4s  corr=%6.3f  cosine=%6.3f  meanAbsPctDiff=%7.2f%%\n', ...
            areaDefs(areaIdx).name, simCorr(areaIdx, pairIdx), simCosine(areaIdx, pairIdx), simMeanAbsPctDiff(areaIdx, pairIdx));
    end
end

%% Power spectra heatmaps (twoThirdsAverage mode)
% Variables:
%   - nMinPlot: number of minutes from session start to include
%   - twoThirdsModeIdx: channel-selection mode index for twoThirdsAverage
% Goal:
%   For each brain area, plot frequency-vs-time power (spectrogram) for
%   the first nMinPlot minutes without reducing into predefined bands.
twoThirdsModeIdx = 2;
samplesToPlot = round(nMinPlot * 60 * fs);
specWindowSamples = max(64, round(welchWindowSec * fs));
specOverlapSamples = min(specWindowSamples - 1, round(specWindowSamples * welchOverlapFrac));
specNfft = max(256, 2 ^ nextpow2(specWindowSamples));
plotFreqRangeHz = [1 120];

figure(1103); clf;
tiledlayout(numAreas, 1, 'TileSpacing', 'compact', 'Padding', 'compact');
for areaIdx = 1:numAreas
    nexttile;
    signal = areaSignals{areaIdx, twoThirdsModeIdx};
    if isempty(signal)
        axis off;
        title(sprintf('%s twoThirdsAverage: no signal', areaDefs(areaIdx).name));
        continue;
    end

    nUse = min(numel(signal), samplesToPlot);
    signal = double(signal(1:nUse));
    [specStft, specFreq, specTime] = spectrogram(signal, specWindowSamples, specOverlapSamples, specNfft, fs);
    specPower = abs(specStft) .^ 2;
    freqMask = specFreq >= plotFreqRangeHz(1) & specFreq <= plotFreqRangeHz(2);
    specPowerDb = 10 * log10(specPower(freqMask, :) + eps);
    imagesc(specTime / 60, specFreq(freqMask), specPowerDb);
    axis xy;
    colormap(gca, 'turbo');
    colorbar;
    ylabel('Frequency (Hz)');
    title(sprintf('%s twoThirdsAverage power spectrogram (first %.1f min)', areaDefs(areaIdx).name, nUse / fs / 60));
end
xlabel('Time (min)');

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

% Plot: similarity heatmaps
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

%% Build time-resolved features and fit Gaussian HMM per mode
min2Model = 60; % how many minutes to model
cutoffIdx = round(60 * min2Model * fs);
% Feature layout per mode:
%   [area1_band1 ... area1_bandN area2_band1 ... areaN_bandN]  (N = numBands)
hmmResults = repmat(struct( ...
    'modeName', '', ...
    'featureMatrix', [], ...
    'timeBins', [], ...
    'bestNumStates', nan, ...
    'stateEstimates', [], ...
    'hmm', [], ...
    'likelihoods', [], ...
    'cvScores', []), numModes, 1);

emOpts = struct('maxIter', hmmMaxIter, 'tol', 1e-4, ...
    'numRestarts', hmmNumRestarts, 'regVar', 1e-3, 'decode', true, 'verbose', true);

for modeIdx = modelModeIdx
    fprintf('\n[HMM progress] Starting mode %d/%d: %s\n', modeIdx, numModes, modeNames{modeIdx});
    binnedEnvelopesMode = cell(numAreas, 1);
    minBinsAcrossAreas = inf;

    for areaIdx = 1:numAreas
        fprintf('[HMM progress]   Building features for area %d/%d: %s\n', areaIdx, numAreas, areaDefs(areaIdx).name);
        signal = areaSignals{areaIdx, modeIdx};
        if isempty(signal)
            fprintf('[HMM progress]   -> %s skipped (empty signal)\n', areaDefs(areaIdx).name);
            continue;
        end
        nUse = min(numel(signal), cutoffIdx);
        signal = signal(1:nUse);

        [iBinnedEnvelopes, ~] = build_lfp_hilbert_bins( ...
            signal, fs, bands, hmmBinSizeSec, bandpassOrder);

        if isempty(iBinnedEnvelopes)
            fprintf('[HMM progress]   -> %s skipped (no binned envelopes)\n', areaDefs(areaIdx).name);
            continue;
        end

        minBinsAcrossAreas = min(minBinsAcrossAreas, size(iBinnedEnvelopes, 1));
        binnedEnvelopesMode{areaIdx} = iBinnedEnvelopes;
        fprintf('[HMM progress]   -> %s features ready (%d bins x %d bands)\n', ...
            areaDefs(areaIdx).name, size(iBinnedEnvelopes, 1), size(iBinnedEnvelopes, 2));
    end

    if ~isfinite(minBinsAcrossAreas) || minBinsAcrossAreas < 2
        fprintf('[HMM progress] Mode %s skipped (insufficient bins)\n', modeNames{modeIdx});
        continue;
    end

    binnedStack = [];
    for areaIdx = 1:numAreas
        if isempty(binnedEnvelopesMode{areaIdx})
            binnedStack = [];
            break;
        end
        binnedStack = [binnedStack, binnedEnvelopesMode{areaIdx}(1:minBinsAcrossAreas, :)]; %#ok<AGROW>
    end

    if isempty(binnedStack) || size(binnedStack, 2) ~= numAreas * numBands
        fprintf('[HMM progress] Mode %s skipped (incomplete feature stack)\n', modeNames{modeIdx});
        continue;
    end

    timeBinsRef = ((1:minBinsAcrossAreas)' - 0.5) * hmmBinSizeSec;
    featureMatrix = zscore_columns_finite(binnedStack);
    fprintf('[HMM progress] Running CV state selection for %s (T=%d, D=%d, K=%s, folds=%d)\n', ...
        modeNames{modeIdx}, size(featureMatrix, 1), size(featureMatrix, 2), mat2str(2:maxStates), numFolds);

    [bestNumStates, cvScores, hmmModel, ~] = gaussian_hmm_cv_select_num_states( ...
        featureMatrix, 2:maxStates, numFolds, emOpts);
    stateEstimates = hmmModel.stateSeq;
    fprintf('[HMM progress] Finished %s: bestNumStates=%d, nFrames=%d\n', ...
        modeNames{modeIdx}, bestNumStates, numel(stateEstimates));

    hmmResults(modeIdx).modeName = modeNames{modeIdx};
    hmmResults(modeIdx).featureMatrix = featureMatrix;
    hmmResults(modeIdx).timeBins = timeBinsRef;
    hmmResults(modeIdx).bestNumStates = bestNumStates;
    hmmResults(modeIdx).stateEstimates = stateEstimates;
    hmmResults(modeIdx).hmm = hmmModel;
    hmmResults(modeIdx).likelihoods = cvScores;
    hmmResults(modeIdx).cvScores = cvScores;
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
    tileLayout = tiledlayout(1, nState, 'TileSpacing', 'compact', 'Padding', 'compact');

    meanByBandPerState = cell(nState, 1);
    yMinAll = inf;
    yMaxAll = -inf;
    for stateIdx = 1:nState
        currentState = uniqueStates(stateIdx);
        meanPower = mean(featureMatrix(stateEstimates == currentState, :), 1);
        meanByBand = reshape(meanPower, numBands, numAreas)';
        meanByBandPerState{stateIdx} = meanByBand;
        meanAcrossAreas = mean(meanByBand, 1);
        yMinAll = min([yMinAll, meanByBand(:)', meanAcrossAreas]);
        yMaxAll = max([yMaxAll, meanByBand(:)', meanAcrossAreas]);
    end
    if ~isfinite(yMinAll) || ~isfinite(yMaxAll)
        yLo = -1;
        yHi = 1;
    elseif yMinAll == yMaxAll
        pad = 0.5;
        yLo = yMinAll - pad;
        yHi = yMaxAll + pad;
    else
        pad = 0.05 * (yMaxAll - yMinAll);
        yLo = yMinAll - pad;
        yHi = yMaxAll + pad;
    end

    for stateIdx = 1:nState
        currentState = uniqueStates(stateIdx);
        meanByBand = meanByBandPerState{stateIdx};
        ax = nexttile(tileLayout); hold(ax, 'on');
        for areaIdx = 1:numAreas
            plot(ax, 1:numBands, meanByBand(areaIdx, :), 'o-', 'LineWidth', 1.5, ...
                'DisplayName', areaDefs(areaIdx).name);
        end
        plot(ax, 1:numBands, mean(meanByBand, 1), 'k-o', 'LineWidth', 3, 'DisplayName', 'meanAreas');
        xticks(ax, 1:numBands);
        xticklabels(ax, bands(:, 1));
        xlim(ax, [0.5 numBands + 0.5]);
        ylim(ax, [yLo yHi]);
        yline(ax, 0, '--', 'Color', [0.6 0.6 0.6]);
        xlabel(ax, 'Power band');
        ylabel(ax, 'Z envelope');
        title(ax, sprintf('%s state %d (n=%d)', modeNames{modeIdx}, currentState, sum(stateEstimates == currentState)));
        if stateIdx == 1
            legend(ax, 'Location', 'best');
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
    xTime = (0:numel(stateEstimates) - 1) * hmmBinSizeSec;
    colorMap = lines(max(nState, 3));

    figure(1300 + modeIdx); clf; hold on;
    for frameIdx = 1:numel(xTime)
        stateVal = stateEstimates(frameIdx);
        patch([xTime(frameIdx), xTime(frameIdx) + hmmBinSizeSec, xTime(frameIdx) + hmmBinSizeSec, xTime(frameIdx)], ...
            [0, 0, 1, 1], colorMap(stateVal, :), 'EdgeColor', 'none');
    end
    xlim([0 max(xTime) + hmmBinSizeSec]);
    ylim([0 1]);
    yticks([]);
    xlabel('Time (s)');
    title(sprintf('State timeline: %s (nStates=%d)', modeNames{modeIdx}, hmmResults(modeIdx).bestNumStates));
end

%% Save HMM results (session-scoped, metastability-style pathing)
if saveHmmResults
    [~, sessionName] = fileparts(lfpPath);
    saveDir = fullfile(paths.spontaneousResultsPath, sessionName);
    if ~exist(saveDir, 'dir')
        mkdir(saveDir);
    end

    modeledModes = modeNames(modelModeIdx);
    modeledModes = modeledModes(~cellfun(@isempty, modeledModes));

    lfpHmmResults = struct();
    lfpHmmResults.sessionName = sessionName;
    lfpHmmResults.lfpPath = lfpPath;
    lfpHmmResults.areaDefs = areaDefs;
    lfpHmmResults.bands = bands;
    lfpHmmResults.modeNames = modeNames;
    lfpHmmResults.modeledModeIdx = modelModeIdx;
    lfpHmmResults.hmmResults = hmmResults;
    lfpHmmResults.settings = struct( ...
        'hmmBinSizeSec', hmmBinSizeSec, ...
        'bandpassOrder', bandpassOrder, ...
        'maxStates', maxStates, ...
        'numFolds', numFolds, ...
        'hmmNumRestarts', hmmNumRestarts, ...
        'hmmMaxIter', hmmMaxIter, ...
        'modelModeIdx', modelModeIdx, ...
        'min2Model', min2Model, ...
        'doCleanArtifacts', doCleanArtifacts, ...
        'welchWindowSec', welchWindowSec, ...
        'welchOverlapFrac', welchOverlapFrac);
    lfpHmmResults.createdAt = datestr(now, 'yyyy-mm-dd_HH-MM-SS');

    saveFilename = sprintf('hmm_lfp_states_bin%.3f_order%d_mode%d.mat', ...
        hmmBinSizeSec, bandpassOrder, modelModeIdx);
    saveFilepath = fullfile(saveDir, saveFilename);
    fprintf('\nSaving LFP HMM results to:\n%s\n', saveFilepath);
    save(saveFilepath, 'lfpHmmResults', '-v7.3');

    summaryPath = fullfile(saveDir, sprintf('HMM_summary_%s.txt', sessionName));
    fid = fopen(summaryPath, 'w');
    if fid ~= -1
        fprintf(fid, 'LFP HMM Analysis Summary\n');
        fprintf(fid, '========================\n\n');
        fprintf(fid, 'Created: %s\n', lfpHmmResults.createdAt);
        fprintf(fid, 'Session: %s\n', sessionName);
        fprintf(fid, 'LFP path: %s\n', lfpPath);
        fprintf(fid, 'Modeled modes: %s\n', strjoin(modeledModes, ', '));
        fprintf(fid, 'Bin size: %.4f s\n', hmmBinSizeSec);
        fprintf(fid, 'Bandpass order: %d\n', bandpassOrder);
        fprintf(fid, 'Bands:\n');
        for bandIdx = 1:numBands
            fprintf(fid, '  %s: [%g %g] Hz\n', bands{bandIdx, 1}, bands{bandIdx, 2}(1), bands{bandIdx, 2}(2));
        end
        fprintf(fid, '\nPer-mode results:\n');
        for modeIdx = modelModeIdx
            if isempty(hmmResults(modeIdx).stateEstimates)
                fprintf(fid, '  %s: no valid HMM result\n', modeNames{modeIdx});
            else
                fprintf(fid, '  %s: bestNumStates=%d, nFrames=%d\n', ...
                    modeNames{modeIdx}, hmmResults(modeIdx).bestNumStates, numel(hmmResults(modeIdx).stateEstimates));
            end
        end
        fclose(fid);
        fprintf('Summary saved to:\n%s\n', summaryPath);
    else
        warning('Could not write summary file: %s', summaryPath);
    end
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
