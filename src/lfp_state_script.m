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

% lowpass the LFP at 300hz
lfpPerArea = lowpass(lfpPerArea, 300, opts.fsLfp);

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
bands = {'alpha', [8 13]; ...
    'beta', [13 30]; ...
    'lowGamma', [30 50]; ...
    'highGamma', [50 80]};
bands = {'low alpha', [3 6]; ...
    'high alpha', [8 13]; ...
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






%% Get binned signals for fitting the hmm
% bands = {'low', [.1 12]};
numBands = size(bands, 1);

freqIdx = repmat(1:numBands, 1, 4);
% freqIdx = repmat([1 2 3 4], 1, 4);
binnedBandPowers = [];
binnedEnvelopes = [];
method = 'stft';
method = 'cwt';
for iArea = 1 : 4
    [iBinnedZPower, iBinnedEnvelopes, timeBins] = bin_bandpower_by_frames(lfpPerArea(:,iArea), opts.fsLfp, bands, opts.frameSize, method);
    binnedBandPowers = [binnedBandPowers, iBinnedZPower'];
    binnedEnvelopes = [binnedEnvelopes, iBinnedEnvelopes'];
end

%% plot results
bandIdx = 6:10;
plotRange = (1:400);
figure(1232);
subplot(2, 1, 1);

imagesc(binnedBandPowers(plotRange,plotBands)');
colormap('jet');
colorbar;
xlabel('Time (s)');
ylabel('Frequency Bands');
title('Binned Z-Scored Power');

subplot(2, 1, 2);
plot(plotRange, binnedEnvelopes(plotRange,plotBands)');
legend(bands(:, 1));
xlabel('Time (s)');
ylabel('Envelope Amplitude');
title('Binned Envelopes');

%%
figure(228);
hold on;
for i = bandIdx
    plot(1:40, binnedBandPowers(1:40, i), 'DisplayName', sprintf('Column %d', i), 'lineWidth', 2);
end
hold off;
legend

%%
bandIdx = 6:10; % m56
corr(binnedBandPowers(:,bandIdx))
corr(binnedEnvelopes(:,bandIdx))
%% are any of them alpha/beta low, gammas high? and vice versa?
idx = [1 2 3 4] + 4;
featureMatrix = binnedBandPowers;
lowHigh = sum(featureMatrix(:,idx(1)) < 0 & featureMatrix(:,idx(2)) < 0 & featureMatrix(:,idx(3)) > 0 & featureMatrix(:,idx(4)) > 0) / size(featureMatrix, 1);
highLow = sum(featureMatrix(:,idx(1)) > 0 & featureMatrix(:,idx(2)) > 0 & featureMatrix(:,idx(3)) < 0 & featureMatrix(:,idx(4)) < 0) / size(featureMatrix, 1);
allLow = sum(featureMatrix(:,idx(1)) < 0 & featureMatrix(:,idx(2)) < 0 & featureMatrix(:,idx(3)) < 0 & featureMatrix(:,idx(4)) < 0) / size(featureMatrix, 1);
allHigh = sum(featureMatrix(:,idx(1)) > 0 & featureMatrix(:,idx(2)) > 0 & featureMatrix(:,idx(3)) > 0 & featureMatrix(:,idx(4)) > 0) / size(featureMatrix, 1);
fprintf('LowHigh: %.3f\tHighLow: %.3f\tAllLow: %.3f\tAllHigh: %.3f\tTotal: %.3f\t\n', lowHigh, highLow, allLow, allHigh, sum([lowHigh highLow allLow allHigh]))











%% Test LFPs around behavior transitions
preInd = find(diff(bhvIDMat) ~= 0); % 1 frame prior to all behavior transitions
preIndLfp = preInd ./ opts.frameSize * opts.fsLfp;
fullTime = -1.2 : 1/opts.fsLfp : 1.2; % seconds around onset
fullWindow = round(fullTime(1:end-1) * opts.fsLfp); % frames around onset w.r.t. zWindow (remove last frame)
windowCenter = find(fullTime == 0);

areaBandIdx = 2;
numBands = size(bands, 1);


fun = @sRGB_to_OKLab;
colors = maxdistcolor(size(bands, 1), fun);

for i = 1 : length(preIndLfp)
    signal = lfpPerArea(preIndLfp(i) + fullWindow, areaBandIdx);

    % get the power spectra of the 4 bands
    [cfs, frequencies] = cwt(signal, 'amor', opts.fsLfp, ...
        'FrequencyLimits', freqLimits);
    powerSpectra = abs(cfs).^2;

    % Z-score the power at each frequency
    zScoredPower = zscore(abs(cfs).^2, 0, 2);

    % Step 2: Allocate matrices for band power and envelopes
    bandPowerSignals = zeros(numBands, length(signal));
    bandEnvelopes = zeros(numBands, length(signal));

    % Step 3: Compute power and envelopes for each band
    for iBand = 1:numBands
        % Get the frequency range for the current band
        freqRange = bands{iBand, 2};

        % Identify indices corresponding to the band frequencies
        freqIdx = frequencies >= freqRange(1) & frequencies <= freqRange(2);

        % Average z-scored power within the band
        bandPowerSignals(iBand, :) = mean(zScoredPower(freqIdx, :), 1);

        % Compute the envelope using the Hilbert transform
        bandEnvelopes(iBand, :) = abs(hilbert(bandPowerSignals(iBand, :)));
    end
    figure(1332);
    plotTime = -1 : 1/opts.fsLfp : 1; % seconds around onset
    plotWindow = round(plotTime(1:end-1) * opts.fsLfp); % frames around onset w.r.t. zWindow (remove last frame)
    subplot(2, 1, 1);
    % imagesc(fullWindow, 1:size(bandPowerSignals, 1), bandPowerSignals);
    imagesc(plotTime(1:end-1), 1:size(bandPowerSignals, 1), zscore(bandPowerSignals(:, windowCenter + plotWindow), [], 2));
    colormap('jet');
    colorbar;
    xlabel('Time (s)');
    ylabel('Frequency Bands');
    title('Z-Scored Power');

    subplot(2, 1, 2); cla; hold on;
    for iBand = 1 : numBands
        % plot(fullWindow, bandEnvelopes(iBand,:), 'lineWidth', 2, 'Color', colors(iBand,:));
        plot(plotTime(1:end-1), zscore(bandEnvelopes(iBand, windowCenter + plotWindow), [], 2), 'lineWidth', 2, 'Color', colors(iBand,:));
    end
    legend(bands(:, 1));
    xlabel('Time (s)');
    xlim([plotTime(1) plotTime(end-1)])
    yline(0)
    ylabel('Envelope Amplitude');
    title('Z-scored Envelopes');

    % Adjust subplot widths to align
    h1 = subplot(2, 1, 1);
    h2 = subplot(2, 1, 2);
    pos1 = get(h1, 'Position');
    pos2 = get(h2, 'Position');
    pos2(3) = pos1(3); % Make bottom subplot same width as top
    set(h2, 'Position', pos2);


end






%%
idxFit = 1:size(binnedEnvelopes,1);
%% HMM model for state estimation
% Example inputs
maxStates = 8; % Maximum number of HMM states to evaluate
numFolds = 3;   % Number of folds for cross-validation
lambda = 1;
areaBandIdx = 11:15;
featureMatrix = zscore(binnedEnvelopes(idxFit,[areaBandIdx]));
nArea = length(areaBandIdx)/numBands;

% Use previously computed binnedBandPowers
% [bestNumStates, stateEstimates, hmmModels, likelihoods] = fit_hmm_crossval_cov_penalty(binnedEnvelopes(idxFit,:), maxStates, numFolds, lambda);
[bestNumStates, stateEstimates, hmmModels, likelihoods] = fit_hmm_crossval_cov_penalty(featureMatrix, maxStates, numFolds);

% Access optimal HMM properties
disp('Optimal Number of States:');
disp(bestNumStates);

likelihoods

% HMM model for X states
% Train the best model on the full dataset
featureMatrix = zscore(binnedEnvelopes(idxFit,areaBandIdx));
options = statset('MaxIter', 500);

hmm = fitgmdist(featureMatrix, bestNumStates, 'Replicates', 10, 'CovarianceType', 'diagonal', 'Options', options);
% hmm = fitgmdist(featureMatrix, 3, 'Replicates', 10, 'CovarianceType', 'diagonal');

% State estimations
stateEstimates = cluster(hmm, featureMatrix);

[uniqueIntegers, ~, indices] = unique(stateEstimates);
counts = accumarray(indices, 1);


%% Re-create fig 1.1 from poster
% Create a maximized figure on the second monitor
fig = figure(554); clf
set(fig, 'Position', monitorTwo);
nState = length(uniqueIntegers);
[ax, pos] = tight_subplot(1, nState, [.08 .02], .1);
fun = @sRGB_to_OKLab;
colors = maxdistcolor(nState, fun);
for i = 1 : nState
    meanPower = mean(featureMatrix(stateEstimates == i, :), 1);
    meanByBand = reshape(meanPower, 1, numBands);
    % meanByBand = meanPower;
    axes(ax(i)); hold on;
    xticks(1:numBands)
    xticklabels({'low alpha', 'high alpha', 'beta', 'low gamma', 'high gamma'})
    for j = 1:nArea
        plot(1:numBands, meanByBand(j,:), 'color', colors(i,:));
    end
    plot(1:numBands, mean(meanByBand, 1), 'o-', 'color', colors(i,:), 'lineWidth', 3)
    ylim([-1 3])
    xlim([0.5 numBands+.5])
    yline(0, '--', 'color', [.5 .5 .5], 'linewidth', 2)
    xlabel('Power Band');
    ylabel('Normalized Power')
    title(['State ', num2str(i), '  n = ',num2str(sum(stateEstimates == i))])
end
sgtitle('Recreating fig 2E from Akella et al 2024 bioRxiv')
load handel
sound(y(1:round(2.2*Fs)),Fs)

% Plot LFP states into neural umap
plotStatesMap = 1;
fun = @sRGB_to_OKLab;
colors = maxdistcolor(nState, fun);

if plotStatesMap
    colorsForPlot = arrayfun(@(x) colors(x,:), stateEstimates, 'UniformOutput', false);
    colorsForPlot = vertcat(colorsForPlot{:}); % Convert cell array to a matrix
    figH = figHStates;
    plotPos = [monitorOne(1) + monitorOne(3)/2, 1, monitorOne(3)/2, monitorOne(4)];
    titleM = sprintf('LFP States- %s %s bin=%.2f min_dist=%.2f spread=%.1f nn=%d', selectFrom, fitType, opts.frameSize, min_dist, spread, n_neighbors);
    plotFrames = 1:length(bhvIDMat);
    plot_3d_scatter
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













%% YOu have a fit HMM. Project the state labels into UMAP space defined by spiking
getDataType = 'spikes';
get_standard_data
[dataBhv, bhvIDMat] = curate_behavior_labels(dataBhv, opts);

%%
nDim = 8;
colors = colors_for_behaviors(codes);

figHFull = 10;
figHStates = 11;

selectFrom = 'M56';
selectFrom = 'DS';
switch selectFrom
    case 'M56'
        idSelect = idM56;
    case 'DS'
        idSelect = idDS;
end
fitType = ['UMAP ', num2str(mDim), 'D'];

min_dist = (.02);
spread = 1.3;
n_neighbors = 10;


% Fit umap
[projSelect, ~, ~, ~] = run_umap(dataMat(:, idSelect), 'n_components', nDim, 'randomize', false, 'verbose', 'none', ...
    'min_dist', min_dist, 'spread', spread, 'n_neighbors', n_neighbors);
pause(4); close

% --------------------------------------------
%% Plot FULL TIME OF ALL BEHAVIORS
plotFullMap = 1;
if plotFullMap
    colorsForPlot = arrayfun(@(x) colors(x,:), bhvIDMat + 2, 'UniformOutput', false);
    colorsForPlot = vertcat(colorsForPlot{:}); % Convert cell array to a matrix
    figH = figHFull;
    plotPos = [monitorOne(1), 1, monitorOne(3)/2, monitorOne(4)];
    titleM = sprintf('%s %s bin=%.2f min_dist=%.2f spread=%.1f nn=%d', selectFrom, fitType, opts.frameSize, min_dist, spread, n_neighbors);
    plotFrames = 1:length(bhvIDMat);
    plot_3d_scatter
end
%% HMM model for X states
% Train the best model on the full dataset
numStates = 3;
featureMatrix = zscore(binnedEnvelopes(idxFit,:));
options = statset('MaxIter', 500);

hmm = fitgmdist(featureMatrix, numStates, 'Replicates', 10, 'CovarianceType', 'diagonal', 'Options', options);

% State estimations
stateEstimates = cluster(hmm, featureMatrix);

[uniqueIntegers, ~, indices] = unique(stateEstimates);
counts = accumarray(indices, 1)
%
%% Plot LFP states into neural umap
plotStatesMap = 1;
fun = @sRGB_to_OKLab;
colors = maxdistcolor(numStates, fun);

if plotStatesMap
    colorsForPlot = arrayfun(@(x) colors(x,:), stateEstimates, 'UniformOutput', false);
    colorsForPlot = vertcat(colorsForPlot{:}); % Convert cell array to a matrix
    figH = figHStates;
    plotPos = [monitorOne(1) + monitorOne(3)/2, 1, monitorOne(3)/2, monitorOne(4)];
    titleM = sprintf('LFP States- %s %s bin=%.2f min_dist=%.2f spread=%.1f nn=%d', selectFrom, fitType, opts.frameSize, min_dist, spread, n_neighbors);
    plotFrames = 1:length(bhvIDMat);
    plot_3d_scatter
end








%% Cluster the umap blobs, and plot LFP powers with blob labels

% minpts
% To select a value for minpts, consider a value greater than or equal to one plus the number of dimensions of the input data [1]. For example, for an n-by-p matrix X, set the value of 'minpts' greater than or equal to p+1.
% [1] Ester, M., H.-P. Kriegel, J. Sander, and X. Xiaowei. “A density-based algorithm for discovering clusters in large spatial databases with noise.” In Proceedings of the Second International Conference on Knowledge Discovery in Databases and Data Mining, 226-231. Portland, OR: AAAI Press, 1996.
minpts = size(projSelect, 2) + 1; % Default is 5
minpts = 9; % Default is 5

%
% epsilon
% kD = pdist2(projSelect,projSelect,'euc','Smallest',minpts);
% % Plot the k-distance graph.
% plot(sort(kD(end,:)));
% title('k-distance graph')
% xlabel('Points sorted with 50th nearest distances')
% ylabel('50th nearest distances')
% grid

epsilon = .2; % Default is 0.6
% epsilon = 1; % Default is 0.6

idx = dbscan(projSelect(:,1:2), epsilon, minpts);

clusters = unique(idx);
nCluster = length(clusters)


% Plot results
fig = figure(916); clf
set(fig, 'Position', monitorTwo);
[ax, pos] = tight_subplot(1,2, [.08 .02], .1);

dim1 = 1;
dim2 = 2;
dim3 = 3;
% plot dbscan assignments
axes(ax(1))
gscatter(projSelect(:,dim1),projSelect(:,dim2),idx, hsv(length(clusters)));
% compare with behavior labels
axes(ax(2))
scatter(projSelect(:,dim1),projSelect(:,dim2), 50, colorsForPlot, '.'); %, 'filled', '.');


%% Plot the LFP power spectra with the dbscan clusters
min2plot = .5;
nSampleLfp = floor(min2plot*60*opts.fsLfp);
nSampleSpike = floor(min2plot * 60 / opts.frameSize);

for p = 0 : 30
    lfpRange = floor(p * 60 * opts.fsLfp) + 1: floor(p * 60 * opts.fsLfp) + nSampleLfp;
    spikeRange = floor(p * 60 / opts.frameSize) + 1: floor(p * 60 / opts.frameSize) + nSampleSpike;

    bandPowersPerArea = zeros(16, nSampleLfp);
    areaBandIdx = 0;
    powerPerArea = [];
    for iArea = 1:4

        [cfs, frequencies] = cwt(lfpPerArea(lfpRange,iArea), 'amor', opts.fsLfp, 'FrequencyLimits', freqLimits);

        % Preallocate band power matrix
        time = linspace(0, nSampleLfp, size(cfs, 2));
        bandPowers = zeros(numBands, length(time));

        % % Compute power for each band
        % for j = 1:numBands
        %     % Extract the frequency range for the current band
        %     freqRange = bands{j, 2};
        %
        %     % Identify the indices corresponding to the band frequencies
        %     freqIdx = frequencies >= freqRange(1) & frequencies <= freqRange(2);
        %
        %     % Compute power by summing the squared magnitude of the wavelet coefficients
        %     bandPowers(j, :) = zscore(mean(abs(cfs(freqIdx, :)).^2, 1), 0, 2);
        % end
        % bandPowersPerArea((1:4) + areaIdx,:) = bandPowers;
        areaPower = zscore(abs(cfs).^2, 0, 2);
        powerPerArea = [powerPerArea; areaPower];
        areaBandIdx = areaBandIdx + 4;
    end

    %
    figure(832); clf;
    hold on;
    % Define the relative heights of the subplots
    heightRatio = [3 1]; % Upper plot is 3 times the height of the lower plot
    totalHeight = sum(heightRatio);


    % Top subplot (3/4 of the figure height)
    subplot('Position', [0.1, 0.4, 0.8, 0.55]); % [left, bottom, width, height]
    % imagesc(bandPowersPerArea);
    imagesc(powerPerArea);
    % title('Upper Plot');
    xlabel('lfp samples');
    ylabel('frequencies');
    ylim([.5 16.5])
    ylim([.5 size(powerPerArea, 1) + .5])
    % xlim([lfpRange(1) lfpRange(end)])
    xlim([1 size(powerPerArea,2)])


    % Bottom subplot (1/4 of the figure height)
    subplot('Position', [0.1, 0.1, 0.8, 0.2]); % [left, bottom, width, height]


    fun = @sRGB_to_OKLab;
    colors = maxdistcolor(nCluster, fun);
    x = spikeRange;
    idxPlot = idx(x);
    idxPlot(idxPlot == -1) = 0;
    % Create the bar plot
    for i = 1:nSampleSpike
        % Draw a segment for each state with its corresponding color
        patch([x(i)-0.5, x(i)+0.5, x(i)+0.5, x(i)-0.5], ...
            [0, 0, 1, 1], colors(idxPlot(i)+1,:), 'EdgeColor', 'none');
    end
    xlabel('spiking frames')
    % xlim([0 nSampleSpike])
    xlim([spikeRange(1) spikeRange(end)])

end

%% bin the lfp powers
powerPerAreaBinned = zeros(size(powerPerArea, 1), nSampleSpike);
% timeBins = zeros(1, nSampleSpike);
frameSamples = round(opts.frameSize * opts.fsLfp); % Samples per frame
for frameIdx = 1:nSampleSpike
    % Frame indices
    startIdx = (frameIdx - 1) * frameSamples + 1;
    endIdx = startIdx + frameSamples - 1;

    % Extract frame
    frameZPower = powerPerArea(:, startIdx:endIdx);
    % frameEnvelope = bandEnvelopes(:, startIdx:endIdx);

    % Average across the frame
    powerPerAreaBinned(:, frameIdx) = mean(frameZPower, 2);
    % binnedEnvelopes(:, frameIdx) = mean(frameEnvelope, 2);

    % Compute bin midpoint time
    % timeBins(frameIdx) = (startIdx + endIdx) / (2 * fs);
end

% writematrix([powerPerArea], [paths.saveDataPath, 'lfp_1250hz.csv'])
% writematrix([powerPerAreaBinned; idx(x)'], [paths.saveDataPath, 'lfp_blobs.csv'])
% writematrix(idx(x), [paths.saveDataPath, 'blobs_framesize_100ms.csv'])
% writematrix(frequencies, [paths.saveDataPath, 'frequencies.csv'])






%%
lfpl = lowpass(lfpPerArea(1:10^4,3), 300, 1250);
figure(); hold on;
plot(lfpPerArea(1:10^4,3))
plot(lfpl)
















%% Do umap on the binned LFP signals
nDim = 3;

figHFull = 10;
figHStates = 11;
fitType = ['UMAP ', num2str(mDim), 'D'];

min_dist = (.02);
spread = 1.3;
n_neighbors = 20;


% Fit umap
[projSelect, ~, ~, ~] = run_umap(binnedBandPowers, 'n_components', nDim, 'randomize', false, 'verbose', 'none', ...
    'min_dist', min_dist, 'spread', spread, 'n_neighbors', n_neighbors);
% [projSelect, ~, ~, ~] = run_umap(binnedBandPowers, 'n_components', nDim, 'randomize', false, 'verbose', 'none');
% pause(4); close

% --------------------------------------------
% Plot FULL TIME OF ALL BEHAVIORS
plotFullMap = 1;
if plotFullMap
    colors = colors_for_behaviors(codes);
    colorsForPlot = arrayfun(@(x) colors(x,:), bhvIDMat + 2, 'UniformOutput', false);
    colorsForPlot = vertcat(colorsForPlot{:}); % Convert cell array to a matrix
    figH = figHFull;
    plotPos = [monitorOne(1), 1, monitorOne(3)/2, monitorOne(4)];
    titleM = sprintf('%s %s bin=%.2f min_dist=%.2f spread=%.1f nn=%d', selectFrom, fitType, opts.frameSize, min_dist, spread, n_neighbors);
    plotFrames = 1:length(bhvIDMat);
    plot_3d_scatter
end
% Plot LFP states into neural umap
plotStatesMap = 1;
fun = @sRGB_to_OKLab;
colors = maxdistcolor(nState, fun);

if plotStatesMap
    colorsForPlot = arrayfun(@(x) colors(x,:), stateEstimates, 'UniformOutput', false);
    colorsForPlot = vertcat(colorsForPlot{:}); % Convert cell array to a matrix
    figH = figHStates;
    plotPos = [monitorOne(1) + monitorOne(3)/2, 1, monitorOne(3)/2, monitorOne(4)];
    titleM = sprintf('LFP States- %s %s bin=%.2f min_dist=%.2f spread=%.1f nn=%d', selectFrom, fitType, opts.frameSize, min_dist, spread, n_neighbors);
    plotFrames = 1:length(bhvIDMat);
    plot_3d_scatter
end









%%
bandPowers = [];
for iArea = 1:4

    [cfs, frequencies] = cwt(lfpPerArea(lfpRange,iArea), 'amor', opts.fsLfp, 'FrequencyLimits', [0 12]);
    areaPower = zscore(abs(cfs).^2, 0, 2);

    % Preallocate band power matrix
    time = linspace(0, nSampleLfp, size(cfs, 2));
    bandPowers = [bandPowers; areaPower];

end







function [binnedZPower, binnedEnvelopes, timeBins] = bin_bandpower_by_frames(signal, fs, bands, frameSize, method)
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

if nargin < 5
    method = 'cwt';
end
% Frame parameters
frameSamples = round(frameSize * fs); % Samples per frame
numFrames = floor(length(signal) / frameSamples);
numBands = size(bands, 1);

signal = zscore(signal, [], 1);
% Preallocate binned outputs
binnedZPower = zeros(numBands, numFrames);
binnedEnvelopes = zeros(numBands, numFrames);


% Step 1: Compute power
% Calculate band powers using the selected method
switch lower(method)
    case 'stft'
        % STFT parameters
        stftWindowSize = round(0.8 * fs); % ~800 ms window
        stftOverlap = round(0.7 * fs);    % ~700 ms overlap
        fftLength = 2^nextpow2(stftWindowSize);     % FFT length

        % Compute STFT
        [cfs, frequencies, t] = stft(signal, fs, ...
            'Window', hann(stftWindowSize, 'periodic'), ...
            'OverlapLength', stftOverlap, ...
            'FFTLength', fftLength);
        powerSpectra = abs(cfs).^2; % Power spectrum

        % Z-score the power at each frequency
        zScoredPower = zscore(abs(cfs).^2, 0, 2);

        % Interpolate STFT results to match 100ms bins
        timeBins = linspace(0, length(signal) / fs, numFrames);
        for i = 1:numBands
            freqRange = bands{i, 2};
            bandIdx = frequencies >= freqRange(1) & frequencies <= freqRange(2); % Frequency indices for the band
            bandPower = mean(zScoredPower(bandIdx, :), 1); % Average power across band

            % Interpolate power to align with 100ms bins
            binnedZPower(i, :) = interp1(t, bandPower, timeBins, 'linear', 0);

            % Compute the envelope using the Hilbert transform
            binnedEnvelopes(i, :) = abs(hilbert(binnedZPower(i, :)));
        end

    case 'cwt'
        freqLimits = [min(cellfun(@(x) x(1), bands(:, 2))), max(cellfun(@(x) x(2), bands(:, 2)))];
        % Use CWT
        [cfs, frequencies] = cwt(signal, 'amor', fs, ...
            'FrequencyLimits', freqLimits);
        powerSpectra = abs(cfs).^2;

        % Z-score the power at each frequency
        zScoredPower = zscore(abs(cfs).^2, 0, 2);

        % Step 2: Allocate matrices for band power and envelopes
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
        % Preallocate binned outputs
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

end




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
testLogLikelihood = nan(maxStates, 1);
topEigenvalue = nan(maxStates, 1);
hmmModels = cell(maxStates, 1);

for numStates = 2:maxStates
    foldLikelihoods = zeros(numFolds, 1);
    iTopEigenvalue = zeros(numFolds, 1);
    iTestLogLikelihood = zeros(numFolds, 1);
    for fold = 1:numFolds
        % Split data into training and test sets
        testIdx = (1:foldSize) + (fold-1)*foldSize;
        trainIdx = setdiff(1:numBins, testIdx);

        trainData = featureMatrix(trainIdx, :);
        testData = featureMatrix(testIdx, :);

        % Train HMM on training data
        options = statset('MaxIter', 500);

        hmm = fitgmdist(trainData, numStates, 'Replicates', 10, 'CovarianceType', 'full', 'Options', options);
        hmmModels{numStates} = hmm;

        % Evaluate log-likelihood on test data
        iTestLogLikelihood(fold) = sum(log(pdf(hmm, testData)));

        % Compute similarity as the top eigenvalue of the state definition
        % matrix
        % Extract state definition matrix
        stateDefinitionMatrix = hmm.mu; % Size: numStates x numFeatures
        % Compute the covariance matrix of the state definition matrix
        covMatrix = cov(stateDefinitionMatrix);

        % Compute the top eigenvalue
        iTopEigenvalue(fold) = max(eig(covMatrix));


        %     % Compute similarity penalty
        %
        %
        %     % Compute penalty based on covariance similarity
        %     covarianceMatrices = hmm.Sigma; % Covariance matrices for each state
        %     numCovariances = size(covarianceMatrices, 3); % Number of states
        %     similarityPenalty = 0;
        %     for i = 1:numCovariances
        %         for j = i+1:numCovariances
        %             % Frobenius norm of the difference between covariance matrices
        %             similarityPenalty = similarityPenalty + norm(covarianceMatrices(:,:,i) - covarianceMatrices(:,:,j), 'fro');
        %         end
        %     end
        %     similarityPenalty = similarityPenalty / (numCovariances * (numCovariances - 1)); % Average penalty
        %
        %     % Penalize the log-likelihood
        %     foldLikelihoods(fold) = testLogLikelihood - lambda * similarityPenalty;
    end

    % Average likelihoods and top eigenvalues
    % penalizedLikelihoods(numStates) = mean(foldLikelihoods);
    testLogLikelihood(numStates) = mean(iTestLogLikelihood);
    topEigenvalue(numStates) = mean(iTopEigenvalue);
end

normLogLike = normalizeMetric(testLogLikelihood);
normTopEig = normalizeMetric(topEigenvalue);

penalizedLikelihoods = normLogLike ./ normTopEig;
[maxVal, bestNumStates] = max(penalizedLikelihoods);


% % Find the best number of states
% [~, bestNumStates] = max(penalizedLikelihoods);

% Train the best model on the full dataset
bestHMM = fitgmdist(featureMatrix, bestNumStates, 'Replicates', 10, 'CovarianceType', 'diagonal');

% State estimations
stateEstimates = cluster(bestHMM, featureMatrix);
hmmModels{bestNumStates} = bestHMM;
end



% Helper function: Normalize a metric between minVal and maxVal
function normMetric = normalizeMetric(metric)
normMetric = 2 * (metric - min(metric)) / (max(metric) - min(metric)) - 1;
end

% Helper function: Normalize rows of a matrix to sum to 1
function normalizedMatrix = normalize(matrix, dim)
normalizedMatrix = bsxfun(@rdivide, matrix, sum(matrix, dim));
end


