%%
opts = neuro_behavior_options;
opts.minActTime = .16;
opts.collectStart = 0 * 60 * 60; % seconds
opts.collectFor = 4*60 * 60; % seconds

paths = get_paths;

tauRange = [1.2 2.5];
alphaRange = [1.5 2.2];
paramSDRange = [1.3 1.7];





monitorPositions = get(0, 'MonitorPositions');
monitorOne = monitorPositions(1, :); % Just use single monitor if you don't have second one
monitorTwo = monitorPositions(size(monitorPositions, 1), :); % Just use single monitor if you don't have second one














%%   ====================================       Naturalistic vs. Mark Task       ==============================================
opts = neuro_behavior_options;
opts.frameSize = .001;
opts.minFiringRate = .05;
getDataType = 'spikes';
opts.collectFor = 20 * 60;
opts.firingRateCheckTime = 5 * 60;
get_standard_data
%%
% Mark's reach data
dataR = load(fullfile(paths.dropPath, 'reach_data/Y4_100623_Spiketimes_idchan_BEH.mat'));
[dataMatR, idLabels, areaLabels, rmvNeurons] = neural_matrix_mark_data(dataR, opts);
% Get data until 1 sec after the last reach ending.
cutOff = (dataR.R(end,2) + 1000) / 1000 / opts.frameSize;
dataMatR = dataMatR(1:cutOff,:);

% Ensure dataMatR is same size as dataMat
dataMatR = dataMatR(1:size(dataMat, 1),:);
idM23R = find(strcmp(areaLabels, 'M23'));
idM56R = find(strcmp(areaLabels, 'M56'));
idDSR = find(strcmp(areaLabels, 'DS'));
idVSR = find(strcmp(areaLabels, 'VS'));


%%      test avalance durations
areas = {'M23', 'M56', 'DS', 'VS'};
idList = {idM23, idM56, idDS, idVS};
idListR = {idM23R, idM56R, idDSR, idVSR};

pcaFlag = 1;
pcaFirstFlag = 1;
thresholdFlag = 1;

thresholds = 1:12;
thresholds = .5:.05:.8;
binSizes = .004 : .002 : .05;


[numAvalanchesN, uniqueSizesN, numAvalanchesR, uniqueSizesR, thresholdN, thresholdR] = deal(zeros(length(areas), length(thresholds), length(binSizes)));
for a = 1 : length(areas)
    aID = idList{a};
    aIDR = idListR{a};
    for t = 1:length(thresholds)
        for b = 1:length(binSizes)
            dataMatNat = neural_matrix_ms_to_frames(dataMat(:, aID), binSizes(b));
            dataMatReach = neural_matrix_ms_to_frames(dataMatR(:, aIDR), binSizes(b));

            if pcaFlag
                [coeff, score, ~, ~, explained, mu] = pca(dataMatNat);
                forDim = find(cumsum(explained) > 30, 1);
                forDim = max(3, forDim);
                forDim = min(6, forDim);
                if pcaFirstFlag
                    fprintf('Natural Using PCA first %d\n', forDim)
                    nDim = 1:forDim;
                else
                    fprintf('Natural Using PCA Last many from %d to %d\n', forDim+1, size(score, 2))
                    nDim = forDim+1:size(score, 2);
                end
                dataMatNat = score(:,nDim) * coeff(:,nDim)' + mu;

                [coeff, score, ~, ~, explained, mu] = pca(dataMatReach);
                forDim = find(cumsum(explained) > 30, 1);
                forDim = max(3, forDim);
                forDim = min(6, forDim);
                if pcaFirstFlag
                    fprintf('Reach Using PCA first %d\n', forDim)
                    nDim = 1:forDim;
                else
                    fprintf('Reach Using PCA Last many from %d to %d\n', forDim+1, size(score, 2))
                    nDim = forDim+1:size(score, 2);
                end
                dataMatReach = score(:,nDim) * coeff(:,nDim)' + mu;
            end


            if thresholdFlag
                dataMatNat = round(sum(dataMatNat, 2));
                dataMatReach = round(sum(dataMatReach, 2));

                thresholdN = thresholds(t) * median(dataMatNat);
                thresholdR = thresholds(t) * median(dataMatReach);
                fprintf('Threshold Natural:\t%.3f\n', thresholdN)
                fprintf('Threshold Reach:\t%.3f\n', thresholdR)
            else
                thresholdN = thresholds(t);
                thresholdR = thresholds(t);

            end



            [numAvalanchesN(a,t,b), uniqueSizesN(a,t,b)] = avalanches(dataMatNat, thresholdN);
            [numAvalanchesR(a,t,b), uniqueSizesR(a,t,b)] = avalanches(dataMatReach, thresholdR);
        end
    end
end

%%     Find the minimum bin size that produces enough avalanches
% Minimum avalanche number and unique number of sizes
minAvN = 10000;
minSizeN = 50;

% plotMat = numAvalanchesR;
plotMat = numAvalanchesN;
nAvMat = {squeeze(plotMat(1,:,:)), squeeze(plotMat(2,:,:)), squeeze(plotMat(3,:,:)), squeeze(plotMat(4,:,:))};
% Find global color limits across all matrices
minC = min(cellfun(@(x) min(x(:)), nAvMat));
maxC = max(cellfun(@(x) max(x(:)), nAvMat));

% plotMat2 = uniqueSizesR;
plotMat2 = uniqueSizesN;
nSizeMat = {squeeze(plotMat2(1,:,:)), squeeze(plotMat2(2,:,:)), squeeze(plotMat2(3,:,:)), squeeze(plotMat2(4,:,:))};
% Find global color limits across all matrices
minC2 = min(cellfun(@(x) min(x(:)), nSizeMat));
maxC2 = max(cellfun(@(x) max(x(:)), nSizeMat));

figure(88); clf;
ha = tight_subplot(2, length(areas), [0.07 0.03], [0.15 0.1], [0.07 0.05]);  % [gap, lower margin, upper margin]
for a = 1 : length(areas)

    % subplot(2,4,a)
    axes(ha(a));
    hold on;

    imagesc('XData',binSizes,'YData',thresholds,'CData',nAvMat{a}, [minC maxC])
    colormap hot;
    colorbar;
    set(ha(a), 'XTickLabelMode', 'auto');  % Enable Y-axis labels
    set(ha(a), 'YTickLabelMode', 'auto');  % Enable Y-axis labels
    % Optional: remove background axes color
    set(gca, 'Color', 'none');
    % Set title
    title([areas{a}, ' nAvalances']);

    % subplot(2,4,a+length(areas)
    axes(ha(a+length(areas)));
    hold on;

    imagesc('XData',binSizes,'YData',thresholds,'CData',nSizeMat{a}, [minC2 maxC2])
    colormap hot;
    colorbar;
    ylabel('Thresholds')
    xlabel('Bin Sizes')
    set(ha(a+length(areas)), 'XTickLabelMode', 'auto');  % Enable Y-axis labels
    set(ha(a+length(areas)), 'YTickLabelMode', 'auto');  % Enable Y-axis labels
    set(gca, 'Color', 'none');
    % Set title
    title([areas{a}, ' nUnique Sizes']);


    highAvIdx = nAvMat{a} >= minAvN;
    highSizeIdx = nSizeMat{a} >= minSizeN;
    % Find common indices where both matrices have high values
    commonIdx = highAvIdx & highSizeIdx;
    % Get row and column indices of common elements
    [rowIdx, colIdx] = find(commonIdx);

    if isempty(rowIdx)
        fprintf('No common near-optimal element found.\n');
        return;
    end

    % Among the common indices, find the one with the smallest binSize (smallest column index)
    rowOptions = rowIdx(colIdx == min(colIdx));

    % Step 3: Among those rows, find the best mutual trade-off
    % Use a scoring function, e.g., sum of normalized values
    scores = zeros(size(rowOptions));
    for i = 1:length(rowOptions)
        r = rowOptions(i);
        v = nAvMat{a}(r, min(colIdx));
        u = nSizeMat{a}(r, min(colIdx));

        % Normalize by their respective maxima
        scores(i) = (v / max(nAvMat{a}(:))) + (u / max(nSizeMat{a}(:)));
    end

    % Pick the row with the highest combined score
    [~, bestIdx] = max(scores);
    bestRow = rowOptions(bestIdx);
    bestCol = min(colIdx);

    % Display result
    fprintf('---------------   Area %s -------------\n', areas{a})
    fprintf('Chosen Threshold: %.3f\t BinSize: %.3f \t(Row=%d, Col=%d)\n', thresholds(bestRow), binSizes(bestCol), bestRow, bestCol);
    fprintf('NumAvalanches: %.2f\n', nAvMat{a}(bestRow, bestCol));
    fprintf('UniqueSizes: %.2f\n', nSizeMat{a}(bestRow, bestCol));

end

%%               Run the data
pcaFlag = 1;
pcaFirstFlag = 1;
thresholdFlag = 1;
thresholdBinSize = .025;
% medianPct = .5; % set threshold at medianPct of the median of the summed signal
medianPct = .75; % set threshold at medianPct of the median of the summed signal


% Initialize variables
areas = {'M23', 'M56', 'DS', 'VS'};
[optBinSize, tau, tauC, alpha, paramSD, decades] = deal(nan(length(areas), 2));
idList = {idM23, idM56, idDS, idVS};
idListR = {idM23R, idM56R, idDSR, idVSR};
% Branching ratio histogram values
minBR = 0; maxBR = 3;
edges = minBR : .1 : maxBR;
centers = edges(1:end-1) + diff(edges) / 2;
brHist = nan(length(areas), length(centers), 2);

% tpct = [.7 .75 .8 .85];
% tpct = [.08 .15 .25 .5];
% threshods = 1:10;
% for t = 1:length(tpct)
% medianPct = tpct(t);
% minSpike = tpct(t);
tic
for a = 2 : length(areas)
    aID = idList{a};
    aIDR = idListR{a};

    % If using the threshold method
    if thresholdFlag
        optBinSize(a, 1) = thresholdBinSize;
        optBinSize(a, 2) = thresholdBinSize;
    else
        optBinSize(a, 1) = optimal_bin_size(dataMat(:,aID));
        optBinSize(a, 2) = optimal_bin_size(dataMatR(:,aIDR));
    end
    fprintf('-------------------------\tArea: %s\n', areas{a})

    dataMatNat = neural_matrix_ms_to_frames(dataMat(:, aID), optBinSize(a, 1));
    dataMatReach = neural_matrix_ms_to_frames(dataMatR(:, aIDR), optBinSize(a, 2));

    if pcaFlag
        [coeff, score, ~, ~, explained, mu] = pca(dataMatNat);
        forDim = find(cumsum(explained) > 30, 1);
        forDim = max(3, forDim);
        forDim = min(6, forDim);
        if pcaFirstFlag
            fprintf('Natural Using PCA first %d\n', forDim)
            nDim = 1:forDim;
        else
            fprintf('Natural Using PCA Last many from %d to %d\n', forDim+1, size(score, 2))
            nDim = forDim+1:size(score, 2);
        end
        dataMatNat = score(:,nDim) * coeff(:,nDim)' + mu;

        [coeff, score, ~, ~, explained, mu] = pca(dataMatReach);
        forDim = find(cumsum(explained) > 30, 1);
        forDim = max(3, forDim);
        forDim = min(6, forDim);
        if pcaFirstFlag
            fprintf('Reach Using PCA first %d\n', forDim)
            nDim = 1:forDim;
        else
            fprintf('Reach Using PCA Last many from %d to %d\n', forDim+1, size(score, 2))
            nDim = forDim+1:size(score, 2);
        end
        dataMatReach = score(:,nDim) * coeff(:,nDim)' + mu;
    end

    if thresholdFlag
        dataMatNat = round(sum(dataMatNat, 2));
        dataMatReach = round(sum(dataMatReach, 2));

        % threshPct = .08;
        % threshSpikes = threshPct * max(dataMatNat);
        threshSpikes = medianPct * median(dataMatNat);
        dataMatNat(dataMatNat < threshSpikes) = 0;
        fprintf('Natural: Using Threshold method \tBinsize: %.3f\tProp zeros: %.3f\n', optBinSize(a, 1), sum(dataMatNat == 0) / length(dataMatNat))

        % threshSpikes = threshPct * max(dataMatReach);
        threshSpikes = medianPct * median(dataMatReach);
        dataMatReach(dataMatReach < threshSpikes) = 0;
        fprintf('Reach: Using Threshold method \tBinsize: %.3f\tProp zeros: %.3f\n', optBinSize(a, 2), sum(dataMatReach == 0) / length(dataMatReach))
    end

    % % For testing threshold, etc
    % dataMatNat = round(sum(dataMatNat, 2));
    % dataMatReach = round(sum(dataMatReach, 2));
    % dataMatNat(dataMatNat <= minSpike) = 0;
    % dataMatReach(dataMatReach <= minSpike) = 0;
    %
    % sufficient = avalanches_bin_sufficiency(dataMatNat);
    % sufficient = avalanches_bin_sufficiency(dataMatReach);

    % Natural avalanches
    asdfMat = rastertoasdf2(dataMatNat', optBinSize(a, 1)*1000, 'CBModel', 'Spikes', 'DS');
    Av(a, 1) = avprops(asdfMat, 'ratio', 'fingerprint');
    br = Av(a, 1).branchingRatio;
    brHist(a,:, 1) = histcounts(br(br>0), edges, 'Normalization', 'pdf');
    [tau(a, 1), tauC(a, 1), alpha(a, 1), paramSD(a, 1), decades(a, 1)] = avalanche_log(Av(a, 1), 0);

    % Reach avalanches
    asdfMat = rastertoasdf2(dataMatReach', optBinSize(a, 2)*1000, 'CBModel', 'Spikes', 'DS');
    Av(a, 2) = avprops(asdfMat, 'ratio', 'fingerprint');
    br = Av(a, 2).branchingRatio;
    brHist(a,:, 2) = histcounts(br(br>0), edges, 'Normalization', 'pdf');
    [tau(a, 2), tauC(a, 2), alpha(a, 2), paramSD(a, 2), decades(a, 2)] = avalanche_log(Av(a, 2), 0);
    toc/60
end
% end

%%
fileName = fullfile(paths.dropPath, 'avalanches_criticality_naturalistic_vs_reaching_50ms_threshold.mat');
save(fileName, 'README', 'Av', 'brHist', 'tau', 'tauC', 'alpha', 'paramSD', 'optBinSize', 'areas')

%%
figure(36); clf;
ha = tight_subplot(1, length(areas), [0.05 0.05], [0.15 0.1], [0.07 0.05]);  % [gap, lower margin, upper margin]
for a = 1 : length(areas)
    axes(ha(a));
    hold on;
    linewidth = 2;
    % for a = 1 : length(areas)
    plot(centers(2:end), brHist(a,2:end,1), 'k', 'LineWidth',linewidth)
    plot(centers(2:end), brHist(a,2:end,2), 'r', 'LineWidth',linewidth)
    xlim([0 2])
    grid on;
    set(ha(a), 'XTickLabelMode', 'auto');  % Enable Y-axis labels
    title([areas{a}])
end
legend({'Natural', 'Reach'})
sgtitle('Branching Ratio PDFs')
copy_figure_to_clipboard
%
pause
figure(37); clf;
ha = tight_subplot(1, length(areas), [0.05 0.05], [0.15 0.1], [0.07 0.05]);  % [gap, lower margin, upper margin]
for a = 1 : length(areas)
    % Activate subplot
    axes(ha(a));
    hold on;
    plot(1, tau(a,1), 'ok', 'LineWidth', linewidth)
    plot(1, tau(a,2), 'or', 'LineWidth', linewidth)

    plot(2, tauC(a,1), 'ok', 'LineWidth', linewidth)
    plot(2, tauC(a,2), 'or', 'LineWidth', linewidth)

    plot(3, alpha(a,1), 'ok', 'LineWidth', linewidth)
    plot(3, alpha(a,2), 'or', 'LineWidth', linewidth)

    plot(4, paramSD(a,1), 'ok', 'LineWidth', linewidth)
    plot(4, paramSD(a,2), 'or', 'LineWidth', linewidth)

    plot([1 1], tauRange, 'b', 'LineWidth', 1)
    plot([2 2], tauRange, 'b', 'LineWidth', 1)
    plot([3 3], alphaRange, 'b', 'LineWidth', 1)
    plot([4 4], paramSDRange, 'b', 'LineWidth', 1)
    xticks(1:4)
    xticklabels({'tau', 'tauC', 'alpha', 'paramSD'})
    xlim([.5 4.5])
    ylim([.5 5])
    grid on;
    title(areas{a})
end
legend({'Natural', 'Reach'})
sgtitle('Avalanche Params Reaching vs ITI')
copy_figure_to_clipboard






































%%   =============    LFP VERSION: Test transition vs. within-bout criticality BETWEEN behaviors.   =============
opts = neuro_behavior_options;
opts.minFiringRate = .1;
opts.collectStart = 0;
opts.collectFor = 60 * 60;
getDataType = 'lfp';

opts.frameSize = .001;
animal = 'ag25290';
sessionNrn = '112321';
nrnDataPath = strcat(paths.nrnDataPath, 'animal_',animal,'/', sessionNrn, '/');
nrnDataPath = [nrnDataPath, 'recording1/'];
opts.dataPath = nrnDataPath;

data = readmatrix([opts.dataPath, 'lfp.txt']);
data = data(1 + (opts.collectStart * opts.fsLfp) : (opts.collectStart + opts.collectFor) * opts.fsLfp, :);
data = fliplr(data); % flip data so first column (channel) is brain surface

channelSpacing = 100;
channelDepth = 1 : channelSpacing : channelSpacing * size(data, 2);

%%
% lfpAnalyze = data(:,[1 3 5]);
% lfpAnalyze = data(:,[7 9 11]);
lfpAnalyze = data(:,[18 20 22 24]);
% lfpAnalyze = data(:,[29 33 37]);
areas = {'DS'};

nChannels = size(lfpAnalyze, 2);

% Keep only peaks below threshold
nSTD = 2;
binSize = .05;
samplesPerBin = round(binSize * opts.fsLfp);  % Samples per bin

% z-score the lfps
lfpZ = zscore(lfpAnalyze);

% Initialize binned counts
nBins = floor(size(lfpZ, 1) / samplesPerBin);
binnedCounts = zeros(nBins, 1);

for a = 1:nChannels %: length(areas)

    % Find negative peaks (invert the signal)
    [negPeaks, locs] = findpeaks(-lfpZ(:,a));  % Peaks in -lfp_z are troughs in lfp_z
    realPeaks = -negPeaks;                 % Convert back to true LFP peak values
    % Keep only peaks that exceed threshold
    validIdx = realPeaks < -nSTD;
    peakLocs = locs(validIdx);  % Indices of valid peaks


    % Bin the crossing events
    for i = 1:length(peakLocs)
        binIdx = floor(peakLocs(i) / samplesPerBin) + 1;
        if binIdx <= nBins
            binnedCounts(binIdx) = binnedCounts(binIdx) + 1;
        end
    end
end

%
threshold = 0;
sum(binnedCounts == 0) / length(binnedCounts)
[avalancheLengths, avalancheSizes] = avalanches(binnedCounts, threshold);
numAv = length(avalancheSizes)
uniqueAv = length(unique(avalancheLengths))
kappa = compute_kappa(avalancheSizes)
%%

% dataMatT = neural_matrix_ms_to_frames(transMat, optBinSize);
asdfMat = rastertoasdf2(binnedCounts', binSize*1000, 'CBModel', 'Spikes', 'trans');
Av = avprops(asdfMat, 'ratio', 'fingerprint');
br = Av.branchingRatio;
brHist = histcounts(br(br>0), edges, 'Normalization','pdf');

[tau, ~, tauC, ~, alpha, ~, paramSD, decades] = avalanche_log(Av, 1)



%%
figure(36); clf;
ha = tight_subplot(1, length(areas), [0.05 0.05], [0.15 0.1], [0.07 0.05]);  % [gap, lower margin, upper margin]
for a = 1 %: length(areas)
    axes(ha(a));
    hold on;
    linewidth = 2;
    % for a = 1 : length(areas)
    plot(centers(2:end), brHist(2:end), 'k', 'LineWidth',linewidth)
    % plot(centers(2:end), brHist(a,2:end,2), 'r', 'LineWidth',linewidth)
    xlim([0 2])
    grid on;
    set(ha(a), 'XTickLabelMode', 'auto');  % Enable Y-axis labels
    title([areas{a}])
end
% legend({'ITI', 'Reach'})
sgtitle('Branching Ratio PDFs')
copy_figure_to_clipboard
%
%%
figure(37); clf;
ha = tight_subplot(1, length(areas), [0.05 0.05], [0.15 0.1], [0.07 0.05]);  % [gap, lower margin, upper margin]
for a = 1 %: length(areas)
    % Activate subplot
    axes(ha(a));
    hold on;
    plot(1, tau, 'ok', 'LineWidth', linewidth)
    % plot(1, tau(a,2), 'or', 'LineWidth', linewidth)

    plot(2, tauC, 'ok', 'LineWidth', linewidth)
    % plot(2, tauC(a,2), 'or', 'LineWidth', linewidth)

    plot(3, alpha, 'ok', 'LineWidth', linewidth)
    % plot(3, alpha(a,2), 'or', 'LineWidth', linewidth)

    plot(4, paramSD, 'ok', 'LineWidth', linewidth)
    % plot(4, paramSD(a,2), 'or', 'LineWidth', linewidth)

    plot(5, decades, 'ok', 'LineWidth', linewidth)
    % plot(5, decades(a,2), 'or', 'LineWidth', linewidth)

    plot([1 1], tauRange, 'b', 'LineWidth', 1)
    plot([2 2], tauRange, 'b', 'LineWidth', 1)
    plot([3 3], alphaRange, 'b', 'LineWidth', 1)

    plot([4 4], paramSDRange, 'b', 'LineWidth', 1)
    xticks(1:5)
    xticklabels({'tau', 'tauC', 'alpha', 'paramSD', 'decades'})
    xlim([.5 5.5])
    ylim([.5 4])
    grid on;
    set(ha(a), 'YTickLabelMode', 'auto');  % Enable Y-axis labels
    title(areas{a})
end
% legend({'ITI', 'Reaches'})
sgtitle('Avalanche Params Reaching vs ITI')
copy_figure_to_clipboard




























%%   =============     Test transition vs. within-bout criticality BETWEEN behaviors.   =============
opts = neuro_behavior_options;
opts.minFiringRate = .1;
opts.collectStart = 0;
opts.collectFor = 60 * 60;
getDataType = 'spikes';

opts.frameSize = .001;
get_standard_data


%%
bhvIDAlt = zeros(length(bhvID), 1);
groomIdx = ismember(bhvID, 5:10);
locIdx = ismember(bhvID, [0 1 2 13 14 15]);
bhvIDAlt(locIdx) = 1;
bhvIDAlt(groomIdx) = 2;
analyzeCodes = 1:2;
analyzeBhv = {'locomotions', 'grooms'};
bhvID = bhvIDAlt;
%%
pcaFlag = 1;
pcaFirstFlag = 1;
thresholdFlag = 1;
thresholdBinSize = [.026 .008 .006 .008];
thresholds = [.5 .55 .7 .55];

preTime = .2;
postTime = .1;

minBR = 0; maxBR = 2.5;
edges = minBR : .1 : maxBR;
centers = edges(1:end-1) + diff(edges) / 2;


areas = {'M23', 'M56', 'DS', 'VS'};
[tau, tauC, alpha, paramSD] = deal(nan(length(areas), length(analyzeBhv), 2));
brHist = nan(length(areas), length(analyzeBhv), length(centers), 2);
optBinSize = nan(length(areas), 1);
idList = {idM23, idM56, idDS, idVS};

for a = 1 : length(areas)
    fprintf('Area %s\n', areas{a})
    tic


    aID = idList{a};

    % Find optimal bin size for avalannce analyses
    % If using the threshold method
    if thresholdFlag
        optBinSize(a) = thresholdBinSize(a);
        opts.frameSize = optBinSize(a);
        get_standard_data

        bhvIDAlt = zeros(length(bhvID), 1);
        groomIdx = ismember(bhvID, 5:10);
        locIdx = ismember(bhvID, [0 1 2 13 14 15]);
        bhvIDAlt(locIdx) = 1;
        bhvIDAlt(groomIdx) = 2;
        analyzeCodes = 1:2;
        analyzeBhv = {'locomotions', 'grooms'};
        bhvID = bhvIDAlt;

        if pcaFlag
            [coeff, score, ~, ~, explained, mu] = pca(dataMat(:,aID));
            forDim = find(cumsum(explained) > 30, 1);
            forDim = max(3, forDim);
            forDim = min(6, forDim);
            if pcaFirstFlag
                fprintf('Using PCA first %d\n', forDim)
                nDim = 1:forDim;
            else
                fprintf('Using PCA Last many from %d to %D\n', forDim+1, size(score, 2))
                nDim = forDim+1:size(score, 2);
            end
            pcaMat = score(:,nDim) * coeff(:,nDim)' + mu;
        end
    else
        optBinSize(a) = optimal_bin_size(dataMat(:, aID));
    end


    % Make the matrices of the transitions and within-bouts at 1 kHz
    transWindow = round((-preTime/opts.frameSize : postTime/opts.frameSize - 1));

    for b = 1 : length(analyzeCodes)
        fprintf('Behavior: %s\n', analyzeBhv{b})
        bID = analyzeCodes(b);



        % First row of zeros so first spike counts as avalanche
        if thresholdFlag
            transMat = 0;
        else
            transMat = zeros(1, length(aID));
        end
        withinMat = transMat;

        % Transition indices
        preIndAll = find(diff(bhvID) ~= 0); % 1 frame prior to all behavior transitions
        preInd = preIndAll(bhvID(preIndAll+1) == bID);

        for bout = 1 : length(preInd)-1
            % fprintf('bout %d\n', bout)


            % Collect transition avalanche data
            transInd = preInd(bout) + transWindow;
            % Only use transitions that are consitent behaviors going into and out
            % of the transition
            if transInd(1) > 0 && sum(diff(bhvID(transInd)) == 0) == length(transWindow) - 2

                % Convert the 1 kHz dataMats to the optimal bin size for avalanche analysis
                % tempMat = neural_matrix_ms_to_frames(dataMat(transInd, aID), optBinSize(a));
                tempMat = pcaMat(transInd,:);
                % if pcaFlag
                %     [coeff, score, ~, ~, explained, mu] = pca(tempMat);
                %     forDim = find(cumsum(explained) > 30, 1);
                %     forDim = max(3, forDim);
                %     forDim = min(6, forDim);
                %     if pcaFirstFlag
                %         fprintf('Using PCA first %d\n', forDim)
                %         nDim = 1:forDim;
                %     else
                %         fprintf('Using PCA Last many from %d to %D\n', forDim+1, size(score, 2))
                %         nDim = forDim+1:size(score, 2);
                %     end
                %     tempMat = score(:,nDim) * coeff(:,nDim)' + mu;
                % end

                if thresholdFlag
                    tempMat = round(sum(tempMat, 2));
                    % threshPct = .08;
                    % threshSpikes = threshPct * max(dataMatReach);
                    threshSpikes = thresholds(a) * median(tempMat);
                    tempMat(tempMat < threshSpikes) = 0;
                    % fprintf('Using Threshold method \tBinsize: %.3f\tProp zeros: %.3f\n', optBinSize(a), sum(dataMatNat == 0) / length(dataMatNat))
                end
                % Find avalances within the data
                zeroBins = find(sum(tempMat, 2) == 0);
                if length(zeroBins) > 1 && any(diff(zeroBins)>1)
                    transMat = [transMat; tempMat(zeroBins(1) : zeroBins(end)-1, :)];
                end
            end


            preIndNext = preIndAll(find(preIndAll > preInd(bout), 1));
            % Collect within-bout avalanche data
            withinInd = preInd(bout) + transWindow(end) + 1 : preIndNext + transWindow(1) - 1;
            % If there is any within-bout data...
            if ~isempty(withinInd)
                % Convert the 1 kHz dataMats to the optimal bin size for avalanche analysis
                % tempMat = neural_matrix_ms_to_frames(dataMat(withinInd, aID), optBinSize(a));
                tempMat = pcaMat(withinInd,:);
                if thresholdFlag
                    tempMat = round(sum(tempMat, 2));
                    % threshPct = .08;
                    % threshSpikes = threshPct * max(dataMatReach);
                    threshSpikes = thresholds(a) * median(tempMat);
                    tempMat(tempMat < threshSpikes) = 0;
                end
                zeroBins = find(sum(tempMat, 2) == 0);
                if length(zeroBins) > 1 && any(diff(zeroBins)>1)
                    withinMat = [withinMat; tempMat(zeroBins(1) : zeroBins(end)-1, :)];
                end
            end

        end
        transMat = [transMat; zeros(1, size(transMat, 2))]; % add a final row of zeros to close out last avalanche
        withinMat = [withinMat; zeros(1, size(withinMat, 2))]; % add a final row of zeros to close out last avalanche




        % dataMatT = neural_matrix_ms_to_frames(transMat, optBinSize);
        asdfMat = rastertoasdf2(transMat', optBinSize(a)*1000, 'CBModel', 'Spikes', 'trans');
        Av(a, b, 1) = avprops(asdfMat, 'ratio', 'fingerprint');
        br = Av(a, b, 1).branchingRatio;
        brHist(a,b,:,1) = histcounts(br(br>0), edges, 'Normalization','pdf');

        [tau(a, b, 1), tauC(a, b, 1), alpha(a, b, 1), paramSD(a, b, 1), decades(a, b, 1)] = avalanche_log(Av(a, b, 1), 0);


        % dataMatW = neural_matrix_ms_to_frames(withinMat, optBinSize);
        asdfMat = rastertoasdf2(withinMat', optBinSize(a)*1000, 'CBModel', 'Spikes', 'within');
        Av(a, b, 2) = avprops(asdfMat, 'ratio', 'fingerprint');
        br = Av(a, b, 2).branchingRatio;
        brHist(a,b,:,2) = histcounts(br(br>0), edges, 'Normalization','pdf');
        [tau(a, b, 2), tauC(a, b, 2), alpha(a, b, 2), paramSD(a, b, 2), decades(a, b, 2)] = avalanche_log(Av(a, b, 2), 0);
    end
end
% fileName = fullfile(paths.dropPath, 'avalanche_analyses.mat');
% save(fileName, 'Av', 'brPeak', 'tau', 'alpha', 'paramSD', 'optBinSize', 'areas', 'nSubsample', '-append')


%%
a = 2;
fig = figure(37); clf;
set(fig, 'Position', monitorTwo);
linewidth = 2;
ha = tight_subplot(2, ceil(length(analyzeCodes)/2), [0.05 0.02], [0.15 0.1], [0.07 0.05]);  % [gap, lower margin, upper margin]
for b = 1 : length(analyzeCodes)
    axes(ha(b));
    hold on; grid on;
    plot(centers(2:end), reshape(brHist(a,b,2:end,1), length(centers(2:end)), 1), 'k', 'LineWidth',linewidth)
    plot(centers(2:end), reshape(brHist(a,b,2:end,2), length(centers(2:end)), 1), 'r', 'LineWidth',linewidth)
    title(analyzeBhv{b}, 'Interpreter','none')
    set(ha(b), 'XTickLabelMode', 'auto');  % Enable Y-axis labels
    xlim([0 2])
    if b == 1
        legend({'Trans', 'Within'})
    end
end
sgtitle([areas{a}, ' Branching Ratios Transitions vs Within-Bout'])
copy_figure_to_clipboard
%%
% a = 2;
fig2 = figure(38); clf;
set(fig2, 'Position', monitorTwo);

ha = tight_subplot(2, ceil(length(analyzeCodes)/2), [0.05 0.02], [0.15 0.1], [0.07 0.05]);  % [gap, lower margin, upper margin]
for b = 1 : length(analyzeCodes)
    % Activate subplot
    axes(ha(b));
    % for a = 1 : length(areas)
    hold on;

    plot(1, tau(a,b,1), 'ok', 'LineWidth', linewidth)
    plot(1, tau(a,b,2), 'or', 'LineWidth', linewidth)

    plot(2, tauC(a,b,1), 'ok', 'LineWidth', linewidth)
    plot(2, tauC(a,b,2), 'or', 'LineWidth', linewidth)

    plot(3, alpha(a,b,1), 'ok', 'LineWidth', linewidth)
    plot(3, alpha(a,b,2), 'or', 'LineWidth', linewidth)

    plot(4, paramSD(a,b,1), 'ok', 'LineWidth', linewidth)
    plot(4, paramSD(a,b,2), 'or', 'LineWidth', linewidth)

    plot(5, decades(a,b,1), 'ok', 'LineWidth', linewidth)
    plot(5, decades(a,b,2), 'or', 'LineWidth', linewidth)

    plot([1 1], tauRange, 'b', 'LineWidth', 1)
    plot([2 2], tauRange, 'b', 'LineWidth', 1)
    plot([3 3], alphaRange, 'b', 'LineWidth', 1)
    plot([4 4], paramSDRange, 'b', 'LineWidth', 1)

    xticks(1:5)
    xticklabels({'tau', 'tauC', 'alpha', 'paramSD', 'decades'})
    xlim([.5 5.5])
    ylim([.5 5])
    set(ha(b), 'YTickLabelMode', 'auto');  % Enable Y-axis labels
    grid on;
    title(analyzeBhv{b}, 'interpreter', 'none')
    if b == 1
        legend({'Trans', 'Within'})
    end
end
sgtitle([areas{a}, ' Avalanche Params Transitions vs Within-Bout'])
copy_figure_to_clipboard








































%%   ================================            Sliding window of criticality parameters          ================================
% Slide a window from 1sec prior to 100ms after each time point
% Get criticality parameters at each time piont
% Assess metastability of criticality over time
opts = neuro_behavior_options;
opts.frameSize = .001;
opts.minFiringRate = .1;
getDataType = 'spikes';
opts.collectStart = 0*60*60;
opts.collectFor = 15 * 60;
opts.firingRateCheckTime = 5 * 60;
get_standard_data

%%
pcaFlag = 0;
pcaFirstFlag = 0;
thresholdFlag = 1;
thresholdBinSize = .015;

preTime = 3;
postTime = .2;
stepSize = 0.1;       % Step size in seconds
numSteps = floor((size(dataMat, 1) / 1000 - stepSize) / stepSize) - 1;

% Initialize variables
areas = {'M23', 'M56', 'DS', 'VS'};
[brPeak, tau, tauC, alpha, paramSD, decades] = deal(cell(length(areas), 1));
optBinSize = nan(length(areas), 1);
idList = {idM23, idM56, idDS, idVS};
% Branching ratio histogram values
minBR = 0; maxBR = 2.5;
edges = minBR : .1 : maxBR;
centers = edges(1:end-1) + diff(edges) / 2;

% poolID = parpool(4, 'IdleTimeout', Inf);
% parfor a = 1  : length(areas)
for a = 2 : length(areas)
    tic
    aID = idList{a};

    % Find optimal bin size for avalanche analyses
    % If using the threshold method
    if thresholdFlag
        optBinSize(a) = thresholdBinSize;
    else
        optBinSize(a) = optimal_bin_size(dataMat(:,aID));
    end
    fprintf('-------------\tArea: %s\n', areas{a})

    dataMatNat = neural_matrix_ms_to_frames(dataMat(:, aID), optBinSize(a));

    if pcaFlag
        [coeff, score, ~, ~, explained, mu] = pca(dataMatNat);
        forDim = find(cumsum(explained) > 30, 1);
        forDim = max(3, forDim);
        forDim = min(6, forDim);
        if pcaFirstFlag
            fprintf('Using PCA first %d\n', forDim)
            nDim = 1:forDim;
        else
            fprintf('Using PCA Last many from %d to %d\n', forDim+1, size(score, 2))
            nDim = forDim+1:size(score, 2);
        end
        dataMatNat = score(:,nDim) * coeff(:,nDim)' + mu;
    end

    if thresholdFlag
        dataMatNat = round(sum(dataMatNat, 2));
        threshPct = .08;
        % threshSpikes = threshPct * max(dataMatReach);
        threshSpikes = .8*median(dataMatNat);
        dataMatNat(dataMatNat < threshSpikes) = 0;
        fprintf('Using Threshold method \tBinsize: %.3f\tProp zeros: %.3f\n', optBinSize(a), sum(dataMatNat == 0) / length(dataMatNat))
    end


    transWindow = (floor(-preTime/optBinSize(a)) : ceil(postTime/optBinSize(a)) - 1);

    % % Calculate number of windows to preallocate
    % numWindows = floor((size(dataMatNat, 1) - 1) / stepRows) + 1;

    [iTau, iTauC, iAlpha, iparamSD, iDecades] = deal(nan(numSteps, 1));

    iBrHist = nan(numSteps, length(centers));

    % Find firsst valid index to start collecting windowed data
    firstIdx = ceil(preTime / stepSize);
    for i = firstIdx: numSteps %-transWindow(1): size(dataMatNat, 1) - transWindow(end)
        % Find the index in dataMatNat for this step
        iRealTime = i * stepSize;
        iIdx = round(iRealTime / optBinSize(a));
        % Find the index in dataMatNat for this step
        % iIdx = -transWindow(1) + round((i-1) * stepRows);
        fprintf('\t%s\t %d of %d: %.3f  finished \n', areas{a}, i, numSteps, i/numSteps)
        % Make sure there are avalanches in the window
        sufficient = avalanches_bin_sufficiency(dataMatNat(iIdx + transWindow + 1, :));
        zeroBins = find(sum(dataMatNat(iIdx + transWindow + 1, :), 2) == 0);
        if length(zeroBins) > 1 && any(diff(zeroBins)>1)

            asdfMat = rastertoasdf2(dataMatNat(iIdx + transWindow + 1, :)', optBinSize(a)*1000, 'CBModel', 'Spikes', 'DS');
            Av = avprops(asdfMat, 'ratio', 'fingerprint');
            iBrHist(i,:) = histcounts(Av.branchingRatio, edges, 'Normalization', 'pdf');
            [iTau(i), iTauC(i), iAlpha(i), iparamSD(i), iDecades(i)] = avalanche_log(Av, 0);
        end

    end
    fprintf('\nArea %s\t %.1f\n\n', areas{a}, toc/60)

    brPeak{a} = iBrHist;
    tau{a} = iTau;
    tauC{a} = iTauC;
    alpha{a} = iAlpha;
    paramSD{a} = iparamSD;
    decades{a} = iDecades;
end
% delete(poolID)
%%
a = 3
t = stepSize * (1 : numSteps);
figure(); clf;
plot(t, tau{a})
hold on;
plot(t, alpha{a})
plot(t, decades{a})

%%
fileName = fullfile(paths.dropPath, 'avalanche_data_last10min.mat');
save(fileName, 'brPeak', 'tau', 'tauC', 'alpha', 'paramSD', 'optBinSize', 'centers', 'stepSize', 'preTime', 'postTime', 'areas')






























%% ==============================================             Mark's reaching vs ITI             ==============================================
opts.minFiringRate = .1;
opts.frameSize = .001;

% Load Mark's reach data and make it a ms neural data matrix
dataR = load(fullfile(paths.dropPath, 'reach_data/Y4_100623_Spiketimes_idchan_BEH.mat'));
eventTimes = load(fullfile(paths.dropPath, 'Y4_06-Oct-2023 14_14_53_NIBEH.mat'));
[dataMatR, idLabels, areaLabels, rmvNeurons] = neural_matrix_mark_data(dataR, opts);
idM23R = find(strcmp(areaLabels, 'M23'));
idM56R = find(strcmp(areaLabels, 'M56'));
idDSR = find(strcmp(areaLabels, 'DS'));
idVSR = find(strcmp(areaLabels, 'VS'));

% timeEmJs = dataR.und_time_EM_JS;
% align arduino time with neural time
% timeEmJs(:,1) = timeEmJs(:,1) + dataR.ArduinoOffset;



%%
pcaFlag = 1;
pcaFirstFlag = 1;
thresholdFlag = 1;

preTime = .3; % ms before reach onset
postTime = 1.5;

areas = {'M23', 'M56', 'DS', 'VS'};


% Branching ratio histogram values
minBR = 0; maxBR = 4;
edges = minBR : .1 : maxBR;
centers = edges(1:end-1) + diff(edges) / 2;


[brWeighted, brStepwise, brMr, tau, pSz, tauC, pSzC, alpha, pDr, paramSD, decades, dcc, kappa] = ...
    deal(nan(length(areas), 2));
brHist = nan(length(areas), length(centers), 2);
optBinSize = nan(length(areas), 1);
idList = {idM23R, idM56R, idDSR, idVSR};

% Find all reach starts
block1Err = 1;
block1Corr = 2;
block2Err = 3;
block2Corr = 4;
% trialIdx = dataR.block(:, 3) == block2Err;
trialIdx = ismember(dataR.block(:, 3), 1:4);
trialIdx1 = ismember(dataR.block(:, 3), 1:2);
trialIdx2 = ismember(dataR.block(:, 3), 3:4);

for a = 1 : length(areas)
    fprintf('Area %s\n', areas{a})
    tic
    aID = idList{a};

    % If using the threshold method
    if thresholdFlag
        if pcaFlag
            thresholds = [.1 .55 .55 .8];  % These values are determined in code above and manually set for now.
            thresholdBinSize = [.018 .006 .012 .008];
        else
            thresholds = [.75 .75 .75 .75];
            thresholdBinSize = [.016 .002 .006 .004];
        end
        optBinSize(a) = thresholdBinSize(a);
    else
        optBinSize(a) = optimal_bin_size(dataMatR(:, aID));
    end

    reachWindow = floor(-preTime/optBinSize(a)) : ceil(postTime/optBinSize(a)) - 1;
    baseWindow = reachWindow - floor(3/optBinSize(a)) - 1;

    dataMatReach = neural_matrix_ms_to_frames(dataMatR(:, aID), optBinSize(a));

    if pcaFlag
        [coeff, score, ~, ~, explained, mu] = pca(dataMatReach);
        forDim = find(cumsum(explained) > 30, 1);
        forDim = max(3, forDim);
        forDim = min(6, forDim);
        if pcaFirstFlag
            fprintf('Using PCA first %d\n', forDim)
            nDim = 1:forDim;
        else
            fprintf('Using PCA Last many from %d to %D\n', forDim+1, size(score, 2))
            nDim = forDim+1:size(score, 2);
        end
        dataMatReach = score(:,nDim) * coeff(:,nDim)' + mu;
    end

    if thresholdFlag
        dataMatReach = round(sum(dataMatReach, 2));
        threshSpikes = thresholds(a)*median(dataMatReach);
        % threshSpikes = max(.1 * max(dataMatReach), min(dataMatReach));
        dataMatReach(dataMatReach < threshSpikes) = 0;
    end



    % % Reach vs ITI
    % rStarts = round(dataR.R(trialIdx,1)/optBinSize(a)/1000);  % in frame time
    % endEpochs = round(sort([eventTimes.ERtimes, eventTimes.SOLtimes])/optBinSize(a)/1000)';
    % % get rid of trials with negative epochs
    % goodTrials = find((endEpochs - rStarts) > 1);
    %
    % % initialize data mats: start with a row of zeros so first spike counts as
    % % avalanche
    % baseMat = zeros(1, size(dataMatReach, 2));
    % reachMat = zeros(1, size(dataMatReach, 2));
    % for i = 1 : length(goodTrials)
    %     iTrial = goodTrials(i);
    %
    %     reachWindow = rStarts(iTrial) - floor(preTime/optBinSize(a)) : endEpochs(iTrial);
    %     baseWindow = reachWindow - length(reachWindow);
    %
    %     % baseline matrix
    %     % iBaseMat = dataMatReach(rStarts(iTrial) + baseWindow, :);
    %     iBaseMat = dataMatReach(baseWindow, :);
    %     % Find avalances within the data
    %     zeroBins = find(sum(iBaseMat, 2) == 0);
    %     if length(zeroBins) > 1 && any(diff(zeroBins)>1)
    %         baseMat = [baseMat; iBaseMat(zeroBins(1) : zeroBins(end)-1, :)];
    %     end
    %
    %     % reach data matrix
    %     % iReachMat = dataMatReach(rStarts(iTrial) + reachWindow, :);
    %     iReachMat = dataMatReach(reachWindow, :);
    %     % Find avalances within the data
    %     zeroBins = find(sum(iReachMat, 2) == 0);
    %     if length(zeroBins) > 1 && any(diff(zeroBins)>1)
    %         reachMat = [reachMat; iReachMat(zeroBins(1) : zeroBins(end)-1, :)];
    %     end
    %
    % end


    % Block 1 vs Block 2
    % rStarts1 = round(dataR.R(trialIdx1,1)/optBinSize(a)/1000);  % in frame time
    % rStarts2 = round(dataR.R(trialIdx2,1)/optBinSize(a)/1000);  % in frame time
    % % initialize data mats: start with a row of zeros so first spike counts as
    % % avalanche
    % baseMat = zeros(1, size(dataMatReach, 2));
    % reachMat = zeros(1, size(dataMatReach, 2));
    % for i = 1 : length(rStarts1)
    %
    %     %     reachWindow = rStarts(iTrial) - floor(preTime/optBinSize(a)) : endEpochs(iTrial);
    %     window = rStarts1(i) - floor(preTime/optBinSize(a)) : rStarts1(i) + floor(postTime/optBinSize(a));
    %
    %     % baseline matrix
    %     iBaseMat = dataMatReach(window, :);
    %     % Find avalances within the data
    %     zeroBins = find(sum(iBaseMat, 2) == 0);
    %     if length(zeroBins) > 1 && any(diff(zeroBins)>1)
    %         baseMat = [baseMat; iBaseMat(zeroBins(1) : zeroBins(end)-1, :)];
    %     end
    % end
    % for i = 1 : length(rStarts2)
    %
    %     window = rStarts2(i) - floor(preTime/optBinSize(a)) : rStarts2(i) + floor(postTime/optBinSize(a));
    %     % reach data matrix
    %     iReachMat = dataMatReach(window, :);
    %     % Find avalances within the data
    %     zeroBins = find(sum(iReachMat, 2) == 0);
    %     if length(zeroBins) > 1 && any(diff(zeroBins)>1)
    %         reachMat = [reachMat; iReachMat(zeroBins(1) : zeroBins(end)-1, :)];
    %     end
    %
    % end




    % % Error versus correct reward epoch
    %    solTimes = round(eventTimes.SOLtimes/optBinSize(a)/1000)';
    %    errTimes = round(eventTimes.ERtimes/optBinSize(a)/1000)';
    %
    %    % initialize data mats: start with a row of zeros so first spike counts as
    %    % avalanche
    %    baseMat = zeros(1, size(dataMatReach, 2));
    %    reachMat = zeros(1, size(dataMatReach, 2));
    %    for i = 1 : length(solTimes)
    %
    %        window = solTimes(i) + floor((1:3/optBinSize(a)));
    %
    %        % baseline matrix
    %        % iBaseMat = dataMatReach(rStarts(iTrial) + baseWindow, :);
    %        iBaseMat = dataMatReach(window, :);
    %        % Find avalances within the data
    %        zeroBins = find(sum(iBaseMat, 2) == 0);
    %        if length(zeroBins) > 1 && any(diff(zeroBins)>1)
    %            baseMat = [baseMat; iBaseMat(zeroBins(1) : zeroBins(end)-1, :)];
    %        end
    %    end
    %     for i = 1 : length(errTimes)
    %
    %         window = errTimes(i) + floor((1:3/optBinSize(a)));
    %       % reach data matrix
    %        % iReachMat = dataMatReach(rStarts(iTrial) + reachWindow, :);
    %        iReachMat = dataMatReach(window, :);
    %        % Find avalances within the data
    %        zeroBins = find(sum(iReachMat, 2) == 0);
    %        if length(zeroBins) > 1 && any(diff(zeroBins)>1)
    %            reachMat = [reachMat; iReachMat(zeroBins(1) : zeroBins(end)-1, :)];
    %        end
    %
    %    end



    baseMat = [baseMat; zeros(1, size(baseMat, 2))]; % add a final row of zeros to close out last avalanche
    reachMat = [reachMat; zeros(1, size(reachMat, 2))]; % add a final row of zeros to close out last avalanche

    plotFlag = 0;

    asdfMat = rastertoasdf2(baseMat', optBinSize(a)*1000, 'CBModel', 'Spikes', 'ITI');
    Av(a, 1) = avprops(asdfMat, 'ratio', 'fingerprint');
    br = Av(a, 1).branchingRatio;
    brHist(a,:, 1) = histcounts(br(br>0), edges, 'Normalization','pdf');
    brWeighted(a,1) = weighted_branching_ratio(baseMat);
    brStepwise(a,1) = stepwise_branching_ratio(baseMat);
    [tau(a, 1), pSz(a, 1), tauC(a, 1), pSzC(a, 1), alpha(a, 1), pDr(a, 1), ...
        paramSD(a, 1), decades(a, 1)] = avalanche_log(Av(a, 1), plotFlag);
    kappa(a,1) = compute_kappa(Av(a,1).size);
    dcc(a,1) = distance_to_criticality(tau(a,1), alpha(a,1), paramSD(a,1));
    result = branching_ratio_mr_estimation(baseMat);
    brMr(a,1) = result.branching_ratio;

    %
    asdfMat = rastertoasdf2(reachMat', optBinSize(a)*1000, 'CBModel', 'Spikes', 'Reach');
    Av(a, 2) = avprops(asdfMat, 'ratio', 'fingerprint');
    br = Av(a, 2).branchingRatio;
    brHist(a,:, 2) = histcounts(br(br>0), edges, 'Normalization','pdf');
    brWeighted(a,2) = weighted_branching_ratio(reachMat);
    brStepwise(a,2) = stepwise_branching_ratio(reachMat);
    [tau(a, 2), pSz(a, 2), tauC(a, 2), pSzC(a, 2), alpha(a, 2), pDr(a, 2), ...
        paramSD(a, 2), decades(a, 2)] = avalanche_log(Av(a, 2), plotFlag);
    kappa(a,2) = compute_kappa(Av(a,2).size);
    dcc(a,2) = distance_to_criticality(tau(a,2), alpha(a,2), paramSD(a,2));
    result = branching_ratio_mr_estimation(reachMat);
    brMr(a,2) = result.branching_ratio;
end

%%
figure(36); clf;
ha = tight_subplot(1, length(areas), [0.05 0.05], [0.15 0.1], [0.07 0.05]);  % [gap, lower margin, upper margin]
for a = 1 : length(areas)
    axes(ha(a));
    hold on;
    linewidth = 2;
    % for a = 1 : length(areas)
    plot(centers(2:end), brHist(a,2:end,1), 'k', 'LineWidth',linewidth)
    plot(centers(2:end), brHist(a,2:end,2), 'r', 'LineWidth',linewidth)
    xlim([0 2])
    grid on;
    set(ha(a), 'XTickLabelMode', 'auto');  % Enable Y-axis labels
    title([areas{a}])
end
legend({'ITI', 'Reach'})
sgtitle('Branching Ratio PDFs')
copy_figure_to_clipboard
%
%%
figure(37); clf;
ha = tight_subplot(1, length(areas), [0.05 0.05], [0.15 0.1], [0.07 0.05]);  % [gap, lower margin, upper margin]
for a = 1 : length(areas)
    % Activate subplot
    axes(ha(a));
    hold on;
    plot(1, tau(a,1), 'ok', 'LineWidth', linewidth)
    plot(1, tau(a,2), 'or', 'LineWidth', linewidth)

    plot(2, tauC(a,1), 'ok', 'LineWidth', linewidth)
    plot(2, tauC(a,2), 'or', 'LineWidth', linewidth)

    plot(3, alpha(a,1), 'ok', 'LineWidth', linewidth)
    plot(3, alpha(a,2), 'or', 'LineWidth', linewidth)

    plot(4, paramSD(a,1), 'ok', 'LineWidth', linewidth)
    plot(4, paramSD(a,2), 'or', 'LineWidth', linewidth)

    plot(5, decades(a,1), 'ok', 'LineWidth', linewidth)
    plot(5, decades(a,2), 'or', 'LineWidth', linewidth)

    plot([1 1], tauRange, 'b', 'LineWidth', 1)
    plot([2 2], tauRange, 'b', 'LineWidth', 1)
    plot([3 3], alphaRange, 'b', 'LineWidth', 1)

    plot([4 4], paramSDRange, 'b', 'LineWidth', 1)
    xticks(1:5)
    xticklabels({'tau', 'tauC', 'alpha', 'paramSD', 'decades'})
    xlim([.5 5.5])
    ylim([.5 4])
    grid on;
    set(ha(a), 'YTickLabelMode', 'auto');  % Enable Y-axis labels
    title(areas{a})
end
legend({'ITI', 'Reaches'})
sgtitle('Avalanche Params Reaching vs ITI')
copy_figure_to_clipboard



%% Perform avalanche shape collapse for all shapes
a = 1;
c = 2;
% compute average temporal profiles
avgProfiles = avgshapes(Av(a,c).shape, Av(a,c).duration, 'cutoffs', 4, 20);

%& plot all profiles
figure;
for iProfile = 1:length(avgProfiles)
    hold on
    plot(1:length(avgProfiles{iProfile}), avgProfiles{iProfile});
end
hold off

xlabel('Time Bin, t', 'fontsize', 14)
ylabel('Neurons Active, s(t)', 'fontsize', 14)
title('Mean Temporal Profiles', 'fontsize', 14)

% compute shape collapse statistics (SC) and plot
[sigmaNuZInvSC, secondDrv, range, errors] = avshapecollapse(avgProfiles, 'plot');

sigmaSC = avshapecollapsestd(avgProfiles);

title(['Avalanche Shape Collapse', char(10), '1/(sigma nu z) = ',...
    num2str(sigmaNuZInvSC), ' +/- ', num2str(sigmaSC)], 'fontsize', 14)





















%% ==============================================             Mark's Sliding Window
opts.minFiringRate = .1;
opts.frameSize = .001;
opts.minFiringRate = .05;
% Load Mark's reach data and make it a ms neural data matrix
dataR = load(fullfile(paths.dropPath, 'reach_data/Y4_100623_Spiketimes_idchan_BEH.mat'));
[dataMatR, idLabels, areaLabels, rmvNeurons] = neural_matrix_mark_data(dataR, opts);
% Get data until 1 sec after the last reach ending.
cutOff = (dataR.R(end,2) + 1000) / 1000 / opts.frameSize;
dataMatR = dataMatR(1:cutOff,:);

idM23R = find(strcmp(areaLabels, 'M23'));
idM56R = find(strcmp(areaLabels, 'M56'));
idDSR = find(strcmp(areaLabels, 'DS'));
idVSR = find(strcmp(areaLabels, 'VS'));

%%

% howLong = 10 * 60;
% dataMatR = dataMatR(1:(howLong * 1000), :);


pcaFlag = 0;
pcaFirstFlag = 0;
thresholdFlag = 1;
thresholdBinSize = .05;

preTime = 3*60;  %3;
postTime = 3*60;%.2;
stepSize = 1.5*60;%0.1;       % Step size in seconds
numSteps = floor((size(dataMatR, 1) / 1000 - stepSize) / stepSize) - 1;
realTime = nan(numSteps, 1);
isiMult = 10; % Multiple of mean ISI to determine minimum bin size

% Initialize variables
areas = {'M23', 'M56', 'DS', 'VS'};
[brPeak] = deal(cell(length(areas), 1));
[tau, tauC, alpha, paramSD, decades, kappa, dcc, brMr] = deal(nan(numSteps, length(areas)));
optBinSize = nan(length(areas), 1);
idList = {idM23R, idM56R, idDSR, idVSR};
% Branching ratio histogram values
minBR = 0; maxBR = 2.5;
edges = minBR : .1 : maxBR;
centers = edges(1:end-1) + diff(edges) / 2;

% poolID = parpool(4, 'IdleTimeout', Inf);
% parfor a = 1  : length(areas)
for a = 1 : length(areas)
    tic
    aID = idList{a};

    % Remove crazy high multi-units
    aRmv = max(dataMatR(:,aID)) > 2;
    aID(aRmv) = [];

    % Find optimal bin size for avalanche analyses
    % If using the threshold method
    if thresholdFlag
        optBinSize(a) = thresholdBinSize;
    optBinSize(a) = isiMult * round(mean(diff(find(sum(dataMatR(:, aID), 2))))) / 1000;
    else
        optBinSize(a) = optimal_bin_size(dataMatR(:,aID));
    end
    fprintf('-------------\tArea: %s\n', areas{a})

    dataMatFrames = neural_matrix_ms_to_frames(dataMatR(:, aID), optBinSize(a));

    if pcaFlag
        [coeff, score, ~, ~, explained, mu] = pca(dataMatFrames);
        forDim = find(cumsum(explained) > 30, 1);
        forDim = max(3, forDim);
        forDim = min(6, forDim);
        if pcaFirstFlag
            fprintf('Using PCA first %d\n', forDim)
            nDim = 1:forDim;
        else
            fprintf('Using PCA Last many from %d to %D\n', forDim+1, size(score, 2))
            nDim = forDim+1:size(score, 2);
        end
        dataMatFrames = score(:,nDim) * coeff(:,nDim)' + mu;
    end

    if thresholdFlag
        dataMatFrames = round(sum(dataMatFrames, 2));
        % threshPct = .08;
        % threshSpikes = threshPct * max(dataMatReach);
        threshSpikes = .75*median(dataMatFrames);
        dataMatFrames(dataMatFrames < threshSpikes) = 0;
        fprintf('Using Threshold method \tBinsize: %.3f\tProp zeros: %.3f\n', optBinSize(a), sum(dataMatFrames == 0) / length(dataMatFrames))
    end


    transWindow = (floor(-preTime/optBinSize(a)) : ceil(postTime/optBinSize(a)) - 1);

    % % Calculate number of windows to preallocate
    % numWindows = floor((size(dataMatFrames, 1) - 1) / stepRows) + 1;

    [iTau, iTauC, iAlpha, iParamSD, iDecades, iDcc, iDccC, iKappa, iBrMr] = deal(nan(numSteps, 1));

    iBrHist = nan(numSteps, length(centers));

    % Find firsst valid index to start collecting windowed data
    firstIdx = ceil(preTime / stepSize);
    for i = firstIdx: numSteps %-transWindow(1): size(dataMatFrames, 1) - transWindow(end)
            fprintf('Working on \t%s\t %d of %d: %.3f \n', areas{a}, i, numSteps, i/numSteps)

            % Find the index in dataMatFrames for this step
        realTime(i) = i * stepSize;
        iIdx = round(iRealTime / optBinSize(a));
        % Find the index in dataMatFrames for this step
        % iIdx = -transWindow(1) + round((i-1) * stepRows);

        % Make sure there are avalanches in the window
        zeroBins = find(sum(dataMatFrames(iIdx + transWindow + 1, :), 2) == 0);
        if length(zeroBins) > 1 && any(diff(zeroBins)>1)

            asdfMat = rastertoasdf2(dataMatFrames(iIdx + transWindow + 1, :)', optBinSize(a)*1000, 'CBModel', 'Spikes', 'DS');
            Av = avprops(asdfMat, 'ratio', 'fingerprint');
            iBrHist(i,:) = histcounts(Av.branchingRatio, edges, 'Normalization', 'pdf');
            [iTau(i), ~, iTauC(i), ~, iAlpha(i), ~, iParamSD(i), iDecades(i)] = avalanche_log(Av, 0);

            iKappa(i) = compute_kappa(Av.size);
            iDcc(i) = distance_to_criticality(iTau(i), iAlpha(i), iParamSD(i));
            iDccC(i) = distance_to_criticality(iTauC(i), iAlpha(i), iParamSD(i));
            result = branching_ratio_mr_estimation(dataMatFrames(iIdx + transWindow + 1, :));
            iBrMr(i) = result.branching_ratio;


        end

    end
    fprintf('\nArea %s\t %.1f\n\n', areas{a}, toc/60)

    brPeak{a} = iBrHist;
    tau(:,a) = iTau;
    tauC(:,a) = iTauC;
    alpha(:,a) = iAlpha;
    paramSD(:,a) = iParamSD;
    decades(:,a) = iDecades;
    kappa(:,a) = iKappa;
    dcc(:,a) = iDcc;
    dccC(:,a) = iDccC;
    brMr(:,a) = iBrMr;
end
% delete(poolID)

%%
mins = realTime/60;
plotParam =  brMr;
figure(54); clf; hold on;
plot(mins, plotParam(:,1), '-ok', 'lineWidth', 2);
plot(mins, plotParam(:,2), '-ob', 'lineWidth', 2);
plot(mins, plotParam(:,3), '-or', 'lineWidth', 2);
plot(mins, plotParam(:,4), '-o', 'color', [0 .75 0], 'lineWidth', 2);
% plot(mins, plotParam(:,1), '-k', 'lineWidth', 2);
% plot(mins, plotParam(:,2), '-b', 'lineWidth', 2);
% plot(mins, plotParam(:,3), '-r', 'lineWidth', 2);
% plot(mins, plotParam(:,4), '-', 'color', [0 .75 0], 'lineWidth', 2);
% xline(block2Start/60, 'linewidth', 2)
legend({'M23', 'M56', 'DS', 'VS'}, 'Location','northwest')
xlabel('Minutes')
ylabel('Distance to criticality')

% h1 = plot(reachCorr/60, 0, '.', 'color', [0 .5 0], 'MarkerSize', 30);
% h2 = plot(reachErr/60, -.02, '.', 'color', [.3 .3 .3], 'MarkerSize', 30);
% set(h1, 'HandleVisibility', 'off');
% set(h2, 'HandleVisibility', 'off');
% title('Reach Data 5 min w 1 s steps')





























%% =======================    Behavior scale-free?     =======================
% Define behaviors to include (input as a vector)
bhvInclude = analyzeCodes           ; % Modify this as needed
someBhvs = 'groom';
bhvInclude = codes(contains(behaviors, someBhvs));

% Load behavior labels (bhvID) - Ensure bhvID is a column vector
bhvID = bhvID(:);

% Filter bhvID to only include specified behaviors
include_mask = ismember(bhvID, bhvInclude);
bhvFiltered = bhvID(include_mask);

% Compute behavior durations (time spent in each behavior state)
change_points = find(diff([NaN; bhvFiltered]) ~= 0); % Find transitions
durations = diff(change_points); % Compute durations

% Fit power-law distribution using Clauset's plfit (ensure you have plfit.m)
[alpha, xmin, L] = plfit(durations);

% Compare with exponential and lognormal distributions
% Exponential fit
lambda = 1/mean(durations);
L_exp = sum(log(lambda * exp(-lambda * durations)));

% Lognormal fit
mu = mean(log(durations));
sigma = std(log(durations));
L_lognorm = sum(log((1./(durations * sigma * sqrt(2*pi))) .* exp(-((log(durations)-mu).^2) / (2*sigma^2))));

% Compare likelihoods
R_exp = 2 * (L - L_exp);
R_lognorm = 2 * (L - L_lognorm);

% Compute p-values
p_exp = 1 - chi2cdf(R_exp, 1);
p_lognorm = 1 - chi2cdf(R_lognorm, 1);

% Display results
fprintf('Power-law exponent (alpha): %.3f\n', alpha);
fprintf('Comparison with exponential: R=%.3f, p=%.3f\n', R_exp, p_exp);
fprintf('Comparison with lognormal: R=%.3f, p=%.3f\n', R_lognorm, p_lognorm);

% Plot the distribution
figure;
% Define logarithmic bins
binEdges = logspace(log10(min(durations)), log10(max(durations)), 20); % Adjust number of bins
% Compute histogram
[counts, edges] = histcounts(durations, binEdges, 'Normalization', 'pdf');
% Compute bin centers
binCenters = sqrt(edges(1:end-1) .* edges(2:end));
bar(binCenters, counts, 'histc');
hold on;
plot(sort(durations), (xmin ./ sort(durations)).^alpha, 'r', 'LineWidth', 2);
set(gca, 'XScale', 'log', 'YScale', 'log');
xlabel('Duration'); ylabel('Probability Density');
title('Scale-Free Test: Behavior Durations');
legend('Empirical Data', 'Power-Law Fit');




%% =======================    Kinematics scale-free?     =======================
% Define behaviors to include (input as a vector)
opts = neuro_behavior_options;
opts.collectStart = 0 * 60 * 60; % seconds
opts.collectFor = 1 * 60 * 60; % seconds
animal = 'ag25290';
sessionBhv = '112321_1';
sessionNrn = '112321';
if strcmp(sessionBhv, '112321_1')
    sessionSave = '112321';
end

getDataType = 'kinematics';
opts.bhvDataPath = strcat(paths.bhvDataPath, 'animal_',animal,'/');
getDataType = 'kinematics';
% kinData = load_data(opts, getDataType);
kinData = load_data(opts, getDataType);

[coeff, score, ~, ~, explained, mu] = pca(kinData);
forDim = find(cumsum(explained) > 30, 1);
forDim = max(3, forDim);
forDim = min(6, forDim);
nDim = 1:forDim;
kinPCA = score(:,nDim) * coeff(:,nDim)' + mu;

for i = 1:size(kinData, 2)
    alpha(i) = compute_DFA(kinData(:, i), 0);
end

figure();
plot(alpha, 'o', 'linewidth', 2)
xline(15.5)
xline(30.5)
xlabel('Kinematic parameters')
ylabel('Alpha')
title('Long-range temporal correlations')



















%% =======================    Fake data     =======================
% run fakeTrialWiseData

% Make a 1000hz neural matrix
dataFake = zeros(round(max(nnAll(:)) * 1000), size(nnAll, 1));
for n = 1 : size(nnAll, 1)
    nSpikes = round(nnAll(n,:)* 1000);
    nSpikes(isnan(nSpikes)) = [];
    dataFake(nSpikes, n) = 1;
end

%%
pcaFlag = 0;
pcaFirstFlag = 1;
thresholdFlag = 1;
thresholdBinSize = .05;

preTime = .2; % ms before reach onset
postTime = 2;

areas = {'Fake'};


% Branching ratio histogram values
minBR = 0; maxBR = 4;
edges = minBR : .1 : maxBR;
centers = edges(1:end-1) + diff(edges) / 2;


[tau, tauC, alpha, paramSD, decades] = deal(nan(length(areas), 2));
brHist = nan(length(areas), length(centers), 2);
optBinSize = nan(length(areas), 1);
idList = {1:size(dataFake, 2)};

for a = 1 : length(areas)
    fprintf('Area %s\n', areas{a})
    tic
    aID = idList{a};

    % If using the threshold method
    if thresholdFlag
        optBinSize(a) = thresholdBinSize;
    else
        optBinSize(a) = optimal_bin_size(dataFake(:, aID));
    end

    reachWindow = floor(-preTime/optBinSize(a)) : ceil(postTime/optBinSize(a)) - 1;
    baseWindow = reachWindow - floor(3/optBinSize(a)) - 1;

    dataMatReach = neural_matrix_ms_to_frames(dataFake(:, aID), optBinSize(a));

    if pcaFlag
        [coeff, score, ~, ~, explained, mu] = pca(dataMatReach);
        forDim = find(cumsum(explained) > 30, 1);
        forDim = max(3, forDim);
        forDim = min(6, forDim);
        if pcaFirstFlag
            fprintf('Using PCA first %d\n', forDim)
            nDim = 1:forDim;
        else
            fprintf('Using PCA Last many from %d to %D\n', forDim+1, size(score, 2))
            nDim = forDim+1:size(score, 2);
        end
        dataMatReach = score(:,nDim) * coeff(:,nDim)' + mu;
    end

    if thresholdFlag
        dataMatReach = round(sum(dataMatReach, 2));
        threshPct = .08;
        % threshSpikes = threshPct * max(dataMatReach);
        threshSpikes = .9*median(dataMatReach);
        dataMatReach(dataMatReach < threshSpikes) = 0;
    end
    % rStarts = 1000 * (10:10:590);  % in frame time
    rStarts = round((10:10:590)/optBinSize(a));  % in frame time

    % initialize data mats: start with a row of zeros so first spike counts as
    % avalanche
    baseMat = zeros(1, size(dataMatReach, 2));
    reachMat = zeros(1, size(dataMatReach, 2));
    for i = 1 : length(rStarts) - 1


        % baseline matrix
        iBaseMat = dataMatReach(rStarts(i) + baseWindow, :);
        % Find avalances within the data
        zeroBins = find(sum(iBaseMat, 2) == 0);
        if length(zeroBins) > 1 && any(diff(zeroBins)>1)
            baseMat = [baseMat; iBaseMat(zeroBins(1) : zeroBins(end)-1, :)];
        end

        % reach data matrix
        iReachMat = dataMatReach(rStarts(i) + reachWindow, :);
        % Find avalances within the data
        zeroBins = find(sum(iReachMat, 2) == 0);
        if length(zeroBins) > 1 && any(diff(zeroBins)>1)
            reachMat = [reachMat; iReachMat(zeroBins(1) : zeroBins(end)-1, :)];
        end

    end
    baseMat = [baseMat; zeros(1, size(baseMat, 2))]; % add a final row of zeros to close out last avalanche
    reachMat = [reachMat; zeros(1, size(reachMat, 2))]; % add a final row of zeros to close out last avalanche

    plotFlag = 0;

    asdfMat = rastertoasdf2(baseMat', optBinSize(a)*1000, 'CBModel', 'Spikes', 'reach');
    Av(a, 1) = avprops(asdfMat, 'ratio', 'fingerprint');
    br = Av(a, 1).branchingRatio;
    brHist(a,:, 1) = histcounts(br(br>0), edges, 'Normalization','pdf');
    [tau(a, 1), tauC(a, 1), alpha(a, 1), paramSD(a, 1), decades(a, 1)] = avalanche_log(Av(a, 1), plotFlag);


    %
    asdfMat = rastertoasdf2(reachMat', optBinSize(a)*1000, 'CBModel', 'Spikes', 'ITI');
    Av(a, 2) = avprops(asdfMat, 'ratio', 'fingerprint');
    br = Av(a, 2).branchingRatio;
    brHist(a,:, 2) = histcounts(br(br>0), edges, 'Normalization','pdf');
    [tau(a, 2), tauC(a, 2), alpha(a, 2), paramSD(a, 2), decades(a, 2)] = avalanche_log(Av(a, 2), plotFlag);

    % [tauB, alphaB, paramSDB]
    % [tau, alpha, paramSD]
end

%%
figure(36); clf;
ha = tight_subplot(1, length(areas), [0.05 0.05], [0.15 0.1], [0.07 0.05]);  % [gap, lower margin, upper margin]
for a = 1 : length(areas)
    axes(ha(a));
    hold on;
    linewidth = 2;
    % for a = 1 : length(areas)
    plot(centers(2:end), brHist(a,2:end,1), 'k', 'LineWidth',linewidth)
    plot(centers(2:end), brHist(a,2:end,2), 'r', 'LineWidth',linewidth)
    xlim([0 2])
    grid on;
    set(ha(a), 'XTickLabelMode', 'auto');  % Enable Y-axis labels
    title([areas{a}])
end
legend({'ITI', 'Reach'})
sgtitle('Branching Ratio PDFs')
copy_figure_to_clipboard
%
pause
figure(37); clf;
ha = tight_subplot(1, length(areas), [0.05 0.05], [0.15 0.1], [0.07 0.05]);  % [gap, lower margin, upper margin]
for a = 1 : length(areas)
    % Activate subplot
    axes(ha(a));
    hold on;
    plot(1, tau(a,1), 'ok', 'LineWidth', linewidth)
    plot(1, tau(a,2), 'or', 'LineWidth', linewidth)

    plot(2, tauC(a,1), 'ok', 'LineWidth', linewidth)
    plot(2, tauC(a,2), 'or', 'LineWidth', linewidth)

    plot(3, alpha(a,1), 'ok', 'LineWidth', linewidth)
    plot(3, alpha(a,2), 'or', 'LineWidth', linewidth)

    plot(4, paramSD(a,1), 'ok', 'LineWidth', linewidth)
    plot(4, paramSD(a,2), 'or', 'LineWidth', linewidth)

    plot([1 1], tauRange, 'b', 'LineWidth', 1)
    plot([2 2], tauRange, 'b', 'LineWidth', 1)
    plot([3 3], alphaRange, 'b', 'LineWidth', 1)
    plot([4 4], paramSDRange, 'b', 'LineWidth', 1)
    xticks(1:4)
    xticklabels({'tau', 'tauC', 'alpha', 'paramSD'})
    xlim([.5 4.5])
    ylim([.5 5])
    grid on;
    title(areas{a})
end
legend({'ITI', 'Reaches'})
sgtitle('Avalanche Params Reaching vs ITI')
copy_figure_to_clipboard












%%

function [tau, pSz, tauC, pSzC, alpha, pDr, paramSD, decades] = avalanche_log(Av, plotFlag)


if plotFlag == 1
    plotFlag = 'plot';
else
    plotFlag = 'nothing';
end

% size distribution (SZ)
[tau, xminSZ, xmaxSZ, sigmaSZ, pSz, pCritSZ, ksDR, DataSZ] =...
    avpropvals(Av.size, 'size', plotFlag);
tau = cell2mat(tau);
pSz = cell2mat(pSz);

decades = log10(cell2mat(xmaxSZ)/cell2mat(xminSZ));

% size distribution (SZ) with cutoffs
tauC = nan;
pSzC = nan;
UniqSizes = unique(Av.size);
Occurances = hist(Av.size,UniqSizes);
% AllowedSizes = UniqSizes(Occurances >= 20);
AllowedSizes = UniqSizes(Occurances >= 4);
% AllowedSizes = UniqSizes(Occurances >= 2);
AllowedSizes(AllowedSizes < 4) = [];
% AllowedSizes(AllowedSizes < 3) = [];
if length(AllowedSizes) > 1
    LimSize = Av.size(ismember(Av.size,AllowedSizes));
    [tauC, xminSZ, xmaxSZ, sigmaSZ, pSzC, pCritSZ, DataSZ] =...
        avpropvals(LimSize, 'size', plotFlag);
    tauC = cell2mat(tauC);
    pSzC = cell2mat(pSzC);
end
% decades = log10(xmaxSZ/xminSZ);

% duration distribution (DR)
if length(unique(Av.duration)) > 1
    [alpha, xminDR, xmaxDR, sigmaDR, pDr, pCritDR, ksDR, DataDR] =...
        avpropvals(Av.duration, 'duration', plotFlag);
    alpha = cell2mat(alpha);
    pDr = cell2mat(pDr);
    % size given duration distribution (SD)
    [paramSD, waste, waste, sigmaSD] = avpropvals({Av.size, Av.duration},...
        'sizgivdur', 'durmin', xminDR{1}, 'durmax', xmaxDR{1}, plotFlag);
else
    alpha = nan;
    paramSD = nan;
end
end



function optBinSize = optimal_bin_size(dataMat)
timeSeries = sum(dataMat, 2);
positiveIndices = find(timeSeries > 0);
checkISI = repelem(positiveIndices, timeSeries(positiveIndices));
optBinSize = ceil(mean(diff(checkISI))) / 1000;
if optBinSize == 0; optBinSize = .001; end
% Old version:
%     optBinSize{a}(iter, 1) = round(mean(diff(find(sum(dataMat(:, idSelect), 2))))) / 1000;
% if optBinSize{a}(iter, 1) == 0; optBinSize{a}(iter, 1) = .001; end

end

function [avalancheLengths, avalancheSizes] = avalanches(dataMat, threshold)
% Detects avalanches and computes their sizes.
% INPUTS:
%   dataMat   - matrix [time x channels] or [time x neurons]
%   threshold - scalar threshold for defining avalanche activity
% OUTPUTS:
%   numAvalanches   - number of detected avalanches
%   uniqueSizes     - number of unique avalanche sizes
%   avalancheSizes  - vector of avalanche sizes (sum of activity per avalanche)

% Sum activity across columns to get a time series
timeSeries = sum(dataMat, 2);

% Initialize variables
inAvalanche = false;
avalancheLengths = [];     % For duration (optional)
avalancheSizes = [];       % Stores the sum of activity per avalanche
startIdx = 0;

for t = 1:length(timeSeries)
    if timeSeries(t) > threshold
        if ~inAvalanche
            startIdx = t;  % Mark the beginning of an avalanche
            inAvalanche = true;
        end
    elseif inAvalanche
        % End of avalanche
        endIdx = t - 1;
        duration = endIdx - startIdx + 1;
        sizeSum = sum(timeSeries(startIdx:endIdx, :)); % Total activity
        avalancheLengths = [avalancheLengths, duration];
        avalancheSizes = [avalancheSizes, sizeSum];
        inAvalanche = false;
    end
end

% Handle if signal ends during an avalanche
if inAvalanche
    endIdx = length(timeSeries);
    duration = endIdx - startIdx + 1;
    sizeSum = sum(sum(dataMat(startIdx:endIdx, :)));
    avalancheLengths = [avalancheLengths, duration];
    avalancheSizes = [avalancheSizes, sizeSum];
end

% Summary outputs
numAvalanches = length(avalancheSizes);
uniqueSizes = length(unique(avalancheSizes));
end



