%%
% Criticality Spontaneous Behavior vs d2
% Compares behavior switching dynamics with neural criticality (d2) using
% non-overlapping windows across the full spontaneous session.
%
% For each window:
%   1) Compute behavior switch rate from frame-wise behavior labels.
%   2) Compute population activity and d2 per area from spike data.
%   3) Scatter plots: d2 vs switch rate; d2 vs popActivity (sum of bins).
%   4) Controlled analysis: OLS d2 ~ popActivity + switchRate (z-scored predictors
%      per area), partial correlation (d2, switch | pop), delta R^2, nested F-test;
%      summary table, partial-residual plot, bar summary of incremental R^2 and p.

%% =============================    Data Loading    =============================
fprintf('\n=== Loading Spontaneous Data for Behavior-vs-d2 ===\n');

if ~exist('sessionName', 'var') || isempty(sessionName)
    error('sessionName must be defined (e.g. ''ag112321_1'').');
end

sessionType = 'spontaneous';
opts = neuro_behavior_options;
opts.frameSize = 0.001;
opts.firingRateCheckTime = 5 * 60;
opts.collectStart = 0;
opts.collectEnd = [];
opts.minFiringRate = 0.1;
opts.maxFiringRate = 100;
if ~isfield(opts, 'fsBhv') || isempty(opts.fsBhv)
    opts.fsBhv = 30;
end

dataStruct = load_sliding_window_data(sessionType, 'spikes', ...
    'sessionName', sessionName, 'opts', opts);

areas = dataStruct.areas;
idMatIdx = dataStruct.idMatIdx;
numAreas = length(areas);
if isfield(dataStruct, 'areasToTest') && ~isempty(dataStruct.areasToTest)
    areasToTest = dataStruct.areasToTest;
else
    areasToTest = 1:numAreas;
end

saveDir = dataStruct.saveDir;
if ~exist(saveDir, 'dir')
    mkdir(saveDir);
end


% Recording time range
if isfield(dataStruct, 'spikeData') && isfield(dataStruct.spikeData, 'collectStart')
    timeRange = [dataStruct.spikeData.collectStart, dataStruct.spikeData.collectEnd];
else
    timeRange = [0, max(dataStruct.spikeTimes)];
end
sessionStartSec = timeRange(1);
sessionEndSec = timeRange(2);
sessionDurationSec = sessionEndSec - sessionStartSec;

if sessionDurationSec <= 0
    error('Invalid session duration: [%.3f, %.3f].', sessionStartSec, sessionEndSec);
end

%% =============================    Behavior Labels    =============================
paths = get_paths;
pathParts = strsplit(sessionName, filesep);
subDir = pathParts{1}(1:min(2, numel(pathParts{1})));
sessionFolder = fullfile(paths.spontaneousDataPath, subDir, sessionName);

csvFiles = dir(fullfile(sessionFolder, 'behavior_labels*.csv'));
if isempty(csvFiles)
    error('No behavior_labels*.csv found in %s', sessionFolder);
elseif numel(csvFiles) > 1
    warning('Multiple behavior_labels*.csv files found. Using first: %s', csvFiles(1).name);
end

behaviorTable = readtable(fullfile(sessionFolder, csvFiles(1).name));
if ~ismember('Code', behaviorTable.Properties.VariableNames)
    error('Behavior CSV must contain a ''Code'' column.');
end
behaviorLabelsRaw = behaviorTable.Code(:);

% Optional smoothing (recommended for frame-wise label noise)
smoothBehaviorLabels = false;
if smoothBehaviorLabels
    smoothOpts = struct();
    smoothOpts.fsBhv = opts.fsBhv;
    smoothOpts.smoothingWindow = 0.25;
    smoothOpts.summarize = false;
    behaviorLabels = behavior_label_smoothing(behaviorLabelsRaw, smoothOpts);
else
    behaviorLabels = behaviorLabelsRaw;
end

% Optional behavior recoding to collapse labels into super-categories.
% Format per row: { [oldLabelList], newLabelScalar }
% Example:
% behaviorRecodingRules = {
%     [5 6 7], 1;
%     [8 9],   2
% };
useBehaviorRecoding = true;
behaviorRecodingRules = {...
    [0 1 2], 1;
    [3], 2;
    4, 3;
    5:12, 4;
    13:15, 5;
    };
if useBehaviorRecoding
    behaviorLabels = recode_behavior_labels(behaviorLabels, behaviorRecodingRules);
end

numBehaviorFrames = numel(behaviorLabels);
behaviorTimeAxisSec = sessionStartSec + (0:(numBehaviorFrames - 1))' ./ opts.fsBhv;

fprintf('Loaded behavior labels: %d frames (fs=%.2f Hz)\n', numBehaviorFrames, opts.fsBhv);

%% =============================    Configuration    =============================
% Window configuration
windowSizeSec = 10;          % Non-overlapping window size for both behavior and d2
windowBufferSec = 0;        % Use all data: include every fully-contained window

% d2 configuration
pOrder = 10;
critType = 2;
normalizeD2 = false;
nShuffles = 8;

% Optional neural subsampling configuration for windowed d2
useSubsampling = true;        % If true, subsample neurons within each area/window
nSubsamples = 30;             % Number of independent subsampling iterations
nNeuronsSubsample = 20;       % Number of neurons per subsample
minNeuronsMultiple = 1.2;     % Minimum neurons required = round(nNeuronsSubsample * minNeuronsMultiple)

% Bin configuration
useOptimalBinWindowFunction = false;
binSizeManual = 0.025;
minSpikesPerBin = 3;
minBinsPerWindow = 1000;

makePlots = true;

% Controlling for popActivity: multiple regression and partial relationships
runControlledAnalysis = true;
standardizePredictorsForRegression = true;  % z-score popActivity and switchRate within each area

%% =============================    Bin Size Per Area    =============================
binSize = zeros(1, numAreas);

if useOptimalBinWindowFunction
    for a = areasToTest
        neuronIds = dataStruct.idLabel{a};
        firingRateHz = calculate_firing_rate_from_spikes( ...
            dataStruct.spikeTimes, dataStruct.spikeClusters, neuronIds, timeRange);
        [binSize(a), ~] = find_optimal_bin_and_window(firingRateHz, minSpikesPerBin, minBinsPerWindow);
    end
else
    if isempty(binSizeManual) || ~isscalar(binSizeManual) || binSizeManual <= 0
        error('binSizeManual must be a positive scalar.');
    end
    binSize(:) = binSizeManual;
end

%% =============================    Non-overlapping Windows    =============================
% Build non-overlapping windows inside buffered session bounds.
windowStartMin = sessionStartSec + windowBufferSec;
windowEndMax = sessionEndSec - windowBufferSec;
if windowEndMax <= windowStartMin
    error('No valid time after applying windowBufferSec=%.2f.', windowBufferSec);
end

numWindows = floor((windowEndMax - windowStartMin) / windowSizeSec);
if numWindows < 1
    error('No valid non-overlapping windows. Reduce windowSizeSec or windowBufferSec.');
end

windowStartSec = windowStartMin + (0:(numWindows - 1))' * windowSizeSec;
windowEndSec = windowStartSec + windowSizeSec;
windowCenterSec = (windowStartSec + windowEndSec) / 2;

fprintf('Using %d non-overlapping windows (%.2f s each)\n', numWindows, windowSizeSec);

%% =============================    Behavior Metric Per Window    =============================
switchRatePerWindow = nan(numWindows, 1);
for w = 1:numWindows
    inWindow = behaviorTimeAxisSec >= windowStartSec(w) & behaviorTimeAxisSec < windowEndSec(w);
    labelsWindow = behaviorLabels(inWindow);
    switchRatePerWindow(w) = calculate_switch_rate(labelsWindow, windowSizeSec);
end

%% =============================    d2 Per Window (Per Area)    =============================
d2PerWindowByArea = nan(numWindows, numAreas);
popActivityPerWindowByArea = nan(numWindows, numAreas);  % Sum of all bins in each window
if normalizeD2
    d2NormalizedPerWindowByArea = nan(numWindows, numAreas);
else
    d2NormalizedPerWindowByArea = [];
end

for a = areasToTest
    fprintf('Computing per-window d2 for area %s...\n', areas{a});

    neuronIds = dataStruct.idLabel{a};
    aDataMat = bin_spikes(dataStruct.spikeTimes, dataStruct.spikeClusters, ...
        neuronIds, timeRange, binSize(a));
    numTimePoints = size(aDataMat, 1);

    for w = 1:numWindows
        windowCenter = windowCenterSec(w);
        [startIdx, endIdx] = calculate_window_indices_from_center( ...
            windowCenter, windowSizeSec, binSize(a), numTimePoints);

        if ~(startIdx >= 1 && endIdx <= numTimePoints && endIdx > startIdx)
            continue;
        end

        wDataMat = aDataMat(startIdx:endIdx, :);
        wPopActivity = [];

        % Optional subsampling to stabilize area-size effects (matches sequence script style)
        if useSubsampling
            nNeuronsTotal = size(wDataMat, 2);
            minNeuronsRequired = round(nNeuronsSubsample * minNeuronsMultiple);
            if nNeuronsTotal < minNeuronsRequired
                continue;
            end

            nThisSub = min(nNeuronsSubsample, nNeuronsTotal);
            wPopMatrix = nan(nSubsamples, size(wDataMat, 1));
            for ss = 1:nSubsamples
                if nThisSub == nNeuronsTotal
                    neuronIdx = 1:nNeuronsTotal;
                else
                    neuronIdx = randperm(nNeuronsTotal, nThisSub);
                end
                subMat = wDataMat(:, neuronIdx);
                wPopMatrix(ss, :) = sum(subMat, 2)';
            end
            wPopActivity = mean(wPopMatrix, 1)';  % average across subsamples
        else
            wPopActivity = sum(wDataMat, 2);
        end

        if isempty(wPopActivity)
            continue;
        end
        
        % Population activity summary for this window:
        % sum across all time bins in the window.
        popActivityPerWindowByArea(w, a) = sum(wPopActivity);

        try
            [varphi, ~] = myYuleWalker3(double(wPopActivity), pOrder);
            d2Val = getFixedPointDistance2(pOrder, critType, varphi);
            d2PerWindowByArea(w, a) = d2Val;
        catch
            d2PerWindowByArea(w, a) = nan;
        end

        if normalizeD2
            numTimeBins = size(wDataMat, 1);
            d2Shuffled = nan(1, nShuffles);
            for s = 1:nShuffles
                % Build a shuffled population activity trace using the same
                % subsampling setting as the real trace.
                if useSubsampling
                    nNeuronsTotal = size(wDataMat, 2);
                    nThisSub = min(nNeuronsSubsample, nNeuronsTotal);
                    permutedPopMatrix = nan(nSubsamples, numTimeBins);

                    for ss = 1:nSubsamples
                        if nThisSub == nNeuronsTotal
                            neuronIdx = 1:nNeuronsTotal;
                        else
                            neuronIdx = randperm(nNeuronsTotal, nThisSub);
                        end

                        subMat = wDataMat(:, neuronIdx);
                        permutedSubMat = zeros(size(subMat));
                        for n = 1:size(subMat, 2)
                            shiftAmount = randi(numTimeBins);
                            permutedSubMat(:, n) = circshift(subMat(:, n), shiftAmount);
                        end
                        permutedPopMatrix(ss, :) = sum(permutedSubMat, 2)';
                    end
                    permutedPop = mean(permutedPopMatrix, 1)';
                else
                    numNeurons = size(wDataMat, 2);
                    permutedDataMat = zeros(size(wDataMat));
                    for n = 1:numNeurons
                        shiftAmount = randi(numTimeBins);
                        permutedDataMat(:, n) = circshift(wDataMat(:, n), shiftAmount);
                    end
                    permutedPop = sum(permutedDataMat, 2);
                end

                try
                    [varphiPerm, ~] = myYuleWalker3(double(permutedPop), pOrder);
                    d2Shuffled(s) = getFixedPointDistance2(pOrder, critType, varphiPerm);
                catch
                    d2Shuffled(s) = nan;
                end
            end

            meanShuffled = nanmean(d2Shuffled);
            if ~isnan(meanShuffled) && meanShuffled > 0
                d2NormalizedPerWindowByArea(w, a) = d2Val / meanShuffled;
            end
        end
    end
end

%% =============================    Controlled analysis (d2 vs behavior | popActivity)    =============================
% Tests whether switch rate explains d2 beyond population firing (sum of bins
% in window). Uses OLS: d2 ~ 1 + popActivity + switchRate, plus nested models.
controlledAnalysis = struct();
controlledAnalysis.perArea = cell(1, numAreas);
controlledAnalysis.summaryTable = [];

if runControlledAnalysis
    fprintf('\n=== Controlled analysis: d2 ~ popActivity + switchRate (per area) ===\n');

    summaryRows = {};

    for idx = 1:numel(areasToTest)
        a = areasToTest(idx);
        if normalizeD2
            d2Vec = d2NormalizedPerWindowByArea(:, a);
        else
            d2Vec = d2PerWindowByArea(:, a);
        end
        popVec = popActivityPerWindowByArea(:, a);
        swVec = switchRatePerWindow(:);

        validMask = ~isnan(d2Vec) & ~isnan(popVec) & ~isnan(swVec);
        nObs = sum(validMask);
        if nObs < 10
            warning('Area %s: only %d valid windows; skipping controlled analysis.', areas{a}, nObs);
            controlledAnalysis.perArea{a} = struct('skipped', true, 'nObs', nObs);
            continue;
        end

        y = d2Vec(validMask);
        popRaw = popVec(validMask);
        swRaw = swVec(validMask);

        if standardizePredictorsForRegression
            popX = (popRaw - mean(popRaw)) / std(popRaw);
            swX = (swRaw - mean(swRaw)) / std(swRaw);
        else
            popX = popRaw;
            swX = swRaw;
        end

        statsStruct = run_d2_behavior_pop_regression(y, popX, swX, popRaw, swRaw);

        % Partial correlation: d2 with switchRate controlling for popActivity
        try
            [rhoPart, pPart] = partialcorr([y, swRaw, popRaw]);
            statsStruct.partialCorr_d2_switch_given_pop = rhoPart(1, 2);
            statsStruct.partialP_d2_switch_given_pop = pPart(1, 2);
        catch
            statsStruct.partialCorr_d2_switch_given_pop = nan;
            statsStruct.partialP_d2_switch_given_pop = nan;
        end

        statsStruct.areaName = areas{a};
        statsStruct.nObs = nObs;
        controlledAnalysis.perArea{a} = statsStruct;

        summaryRows(end+1, :) = { ...
            areas{a}, nObs, ...
            statsStruct.r2_full, statsStruct.r2_popOnly, statsStruct.deltaR2_switch, ...
            statsStruct.p_switch_slope, statsStruct.partialCorr_d2_switch_given_pop, ...
            statsStruct.partialP_d2_switch_given_pop, statsStruct.f_p_nested}; %#ok<AGROW>
    end

    if ~isempty(summaryRows)
        controlledAnalysis.summaryTable = cell2table(summaryRows, 'VariableNames', { ...
            'area', 'nObs', 'R2_full', 'R2_popOnly', 'deltaR2_switch', ...
            'p_switch_slope', 'partialR_d2_sw_given_pop', 'partialP_d2_sw_given_pop', 'p_nested_F_switch'});
        disp(controlledAnalysis.summaryTable);
        fprintf(['Interpretation: deltaR2_switch = extra variance in d2 explained by switch rate\n' ...
            'after popActivity; small p_switch_slope / p_nested_F_switch => switch rate still\n' ...
            'associates with d2 beyond firing rate (per area, linear model).\n']);
    end
end

%% =============================    Plotting    =============================
if makePlots
    fprintf('\n=== Plotting d2 vs behavior switch rate ===\n');

    numAreasToPlot = numel(areasToTest);
    numCols = ceil(sqrt(numAreasToPlot));
    numRows = ceil(numAreasToPlot / numCols);

    figure(3301); clf;
    t = tiledlayout(numRows, numCols, 'TileSpacing', 'compact', 'Padding', 'compact');

    for idx = 1:numAreasToPlot
        a = areasToTest(idx);
        nexttile;
        hold on;

        if normalizeD2
            d2Vals = d2NormalizedPerWindowByArea(:, a);
            yLabelText = 'd2 (normalized)';
        else
            d2Vals = d2PerWindowByArea(:, a);
            yLabelText = 'd2';
        end

        validIdx = ~isnan(switchRatePerWindow) & ~isnan(d2Vals);
        xVals = switchRatePerWindow(validIdx);
        yVals = d2Vals(validIdx);

        if ~isempty(xVals)
            % Open markers make density/overlap easier to inspect.
            scatter(xVals, yVals, 20, [0.2 0.2 0.8], 'o');
            if numel(xVals) > 2
                % Linear fit and p-value for slope/association.
                fitCoeffs = polyfit(xVals, yVals, 1);
                xFit = linspace(min(xVals), max(xVals), 100);
                yFit = polyval(fitCoeffs, xFit);
                plot(xFit, yFit, '-', 'Color', [0.1 0.1 0.1], 'LineWidth', 1.5);
                
                [rhoVal, pVal] = corr(xVals, yVals, 'Rows', 'complete', 'Type', 'Pearson');
                title(sprintf('%s (r=%.2f, p=%.3g)', areas{a}, rhoVal, pVal), 'Interpreter', 'none');
            else
                title(areas{a}, 'Interpreter', 'none');
            end
        else
            title(sprintf('%s (no valid windows)', areas{a}), 'Interpreter', 'none');
        end

        xlabel('Behavior switch rate (switches/s)');
        ylabel(yLabelText);
        grid on;
    end

    if exist('sessionName', 'var') && ~isempty(sessionName)
        titleText = sprintf('%s - d2 vs Behavior Switch Rate (non-overlapping %.1fs windows)', ...
            sessionName, windowSizeSec);
    else
        titleText = sprintf('d2 vs Behavior Switch Rate (non-overlapping %.1fs windows)', windowSizeSec);
    end
    title(t, titleText, 'Interpreter', 'none');

    plotFile = fullfile(saveDir, sprintf('criticality_spontaneous_behavior_vs_d2_scatter_win%.1f.png', windowSizeSec));
    exportgraphics(gcf, plotFile, 'Resolution', 300);
    fprintf('Saved plot to: %s\n', plotFile);
    
    fprintf('\n=== Plotting d2 vs popActivity ===\n');
    
    figure(3302); clf;
    tPop = tiledlayout(numRows, numCols, 'TileSpacing', 'compact', 'Padding', 'compact');
    
    for idx = 1:numAreasToPlot
        a = areasToTest(idx);
        nexttile;
        hold on;
        
        if normalizeD2
            d2Vals = d2NormalizedPerWindowByArea(:, a);
            yLabelText = 'd2 (normalized)';
        else
            d2Vals = d2PerWindowByArea(:, a);
            yLabelText = 'd2';
        end
        
        popVals = popActivityPerWindowByArea(:, a);
        validIdx = ~isnan(popVals) & ~isnan(d2Vals);
        xVals = popVals(validIdx);
        yVals = d2Vals(validIdx);
        
        if ~isempty(xVals)
            % Open markers make density/overlap easier to inspect.
            scatter(xVals, yVals, 20, [0.2 0.6 0.2], 'o');
            if numel(xVals) > 2
                % Linear fit and p-value for slope/association.
                fitCoeffs = polyfit(xVals, yVals, 1);
                xFit = linspace(min(xVals), max(xVals), 100);
                yFit = polyval(fitCoeffs, xFit);
                plot(xFit, yFit, '-', 'Color', [0.1 0.1 0.1], 'LineWidth', 1.5);
                
                [rhoVal, pVal] = corr(xVals, yVals, 'Rows', 'complete', 'Type', 'Pearson');
                title(sprintf('%s (r=%.2f, p=%.3g)', areas{a}, rhoVal, pVal), 'Interpreter', 'none');
            else
                title(areas{a}, 'Interpreter', 'none');
            end
        else
            title(sprintf('%s (no valid windows)', areas{a}), 'Interpreter', 'none');
        end
        
        xlabel('popActivity (sum of all bins in window)');
        ylabel(yLabelText);
        grid on;
    end
    
    if exist('sessionName', 'var') && ~isempty(sessionName)
        titleTextPop = sprintf('%s - d2 vs popActivity (non-overlapping %.1fs windows)', ...
            sessionName, windowSizeSec);
    else
        titleTextPop = sprintf('d2 vs popActivity (non-overlapping %.1fs windows)', windowSizeSec);
    end
    title(tPop, titleTextPop, 'Interpreter', 'none');
    
    plotFilePop = fullfile(saveDir, sprintf('criticality_spontaneous_behavior_vs_popActivity_scatter_win%.1f.png', windowSizeSec));
    exportgraphics(gcf, plotFilePop, 'Resolution', 300);
    fprintf('Saved plot to: %s\n', plotFilePop);

    % --- Controlled analysis plots (partial residuals + coefficient summary)
    if runControlledAnalysis && isfield(controlledAnalysis, 'summaryTable') && ...
            ~isempty(controlledAnalysis.summaryTable) && height(controlledAnalysis.summaryTable) > 0
        fprintf('\n=== Plotting controlled analysis (d2 | popActivity) ===\n');

        tblS = controlledAnalysis.summaryTable;
        nSum = height(tblS);

        % Figure: partial residual plot per area (d2 minus pop-only fit vs switch rate)
        figure(3303); clf;
        tPart = tiledlayout(numRows, numCols, 'TileSpacing', 'compact', 'Padding', 'compact');
        for idx = 1:numAreasToPlot
            a = areasToTest(idx);
            nexttile;
            hold on;
            st = controlledAnalysis.perArea{a};
            if isempty(st) || (isfield(st, 'skipped') && st.skipped)
                title(sprintf('%s (skipped)', areas{a}), 'Interpreter', 'none');
                xlabel('Switch rate (switches/s)');
                ylabel('Partial residual');
                grid on;
                continue;
            end
            if normalizeD2
                d2Vec = d2NormalizedPerWindowByArea(:, a);
            else
                d2Vec = d2PerWindowByArea(:, a);
            end
            popVec = popActivityPerWindowByArea(:, a);
            swVec = switchRatePerWindow(:);
            validMask = ~isnan(d2Vec) & ~isnan(popVec) & ~isnan(swVec);
            y = d2Vec(validMask);
            popRaw = popVec(validMask);
            swPlot = swVec(validMask);
            if standardizePredictorsForRegression
                popX = (popRaw - mean(popRaw)) / std(popRaw);
            else
                popX = popRaw;
            end
            Xpop = [ones(sum(validMask), 1), popX];
            betaPop = Xpop \ y;
            yHatPop = Xpop * betaPop;
            partResid = y - yHatPop;
            scatter(swPlot, partResid, 20, [0.45 0.2 0.55], 'o');
            if numel(swPlot) > 2
                pSw = polyfit(swPlot, partResid, 1);
                xx = linspace(min(swPlot), max(swPlot), 50);
                plot(xx, polyval(pSw, xx), 'k-', 'LineWidth', 1.2);
            end
            if isfield(st, 'partialCorr_d2_switch_given_pop') && ~isnan(st.partialCorr_d2_switch_given_pop)
                title(sprintf('%s (partial r=%.2f)', areas{a}, st.partialCorr_d2_switch_given_pop), 'Interpreter', 'none');
            else
                title(areas{a}, 'Interpreter', 'none');
            end
            xlabel('Switch rate (switches/s)');
            ylabel('d2 residual | popActivity');
            grid on;
        end
        if exist('sessionName', 'var') && ~isempty(sessionName)
            titlePart = sprintf('%s - Partial residuals: (d2 - fit(d2~pop)) vs switch rate', sessionName);
        else
            titlePart = 'Partial residuals: (d2 - fit(d2~popActivity)) vs switch rate';
        end
        title(tPart, titlePart, 'Interpreter', 'none');
        plotFilePart = fullfile(saveDir, sprintf('criticality_spontaneous_d2_partial_residuals_vs_switch_win%.1f.png', windowSizeSec));
        exportgraphics(gcf, plotFilePart, 'Resolution', 300);
        fprintf('Saved plot to: %s\n', plotFilePart);

        % Figure: bar summary of incremental R^2 and -log10(p) for switch in full model
        figure(3304); clf;
        tiledlayout(2, 1, 'TileSpacing', 'compact', 'Padding', 'compact');

        nexttile;
        bar(1:nSum, tblS.deltaR2_switch, 'FaceColor', [0.3 0.45 0.65]);
        set(gca, 'XTick', 1:nSum, 'XTickLabel', tblS.area, 'XTickLabelRotation', 45);
        ylabel('\Delta R^2 (full - pop-only)');
        title('Incremental variance in d2 explained by switch rate (after popActivity)');
        grid on;

        nexttile;
        epsP = 1e-300;
        negLogP = -log10(max(tblS.p_switch_slope, epsP));
        bar(1:nSum, negLogP, 'FaceColor', [0.55 0.35 0.25]);
        hold on;
        yline(-log10(0.05), 'r--', 'LineWidth', 1);
        set(gca, 'XTick', 1:nSum, 'XTickLabel', tblS.area, 'XTickLabelRotation', 45);
        ylabel('-log_{10}(p) switch slope');
        title('Significance of switchRate coefficient in d2 ~ popActivity + switchRate');
        grid on;

        if exist('sessionName', 'var') && ~isempty(sessionName)
            sgtitle(sprintf('%s - Controlled analysis summary', sessionName), 'Interpreter', 'none');
        else
            sgtitle('Controlled analysis summary', 'Interpreter', 'none');
        end

        plotFileBars = fullfile(saveDir, sprintf('criticality_spontaneous_d2_controlled_summary_bars_win%.1f.png', windowSizeSec));
        exportgraphics(gcf, plotFileBars, 'Resolution', 300);
        fprintf('Saved plot to: %s\n', plotFileBars);
    end
end

fprintf('\n=== Analysis Complete ===\n');

%% =============================    Save results (optional)    =============================
results = struct();
results.sessionName = sessionName;
results.areas = areas;
results.idMatIdx = idMatIdx;
results.areasToTest = areasToTest;
results.timeRange = timeRange;
results.windowSizeSec = windowSizeSec;
results.windowBufferSec = windowBufferSec;
results.windowStartSec = windowStartSec;
results.windowEndSec = windowEndSec;
results.windowCenterSec = windowCenterSec;
results.numWindows = numWindows;
results.switchRatePerWindow = switchRatePerWindow;
results.d2PerWindowByArea = d2PerWindowByArea;
results.popActivityPerWindowByArea = popActivityPerWindowByArea;
results.binSize = binSize;
results.params.pOrder = pOrder;
results.params.critType = critType;
results.params.normalizeD2 = normalizeD2;
results.params.useSubsampling = useSubsampling;
results.params.standardizePredictorsForRegression = standardizePredictorsForRegression;
if runControlledAnalysis
    results.controlledAnalysis = controlledAnalysis;
end

resultsFile = fullfile(saveDir, sprintf('criticality_spontaneous_behavior_vs_d2_win%.1f.mat', windowSizeSec));
save(resultsFile, 'results');
fprintf('Saved results to: %s\n', resultsFile);

%% =============================    Local Functions    =============================
function statsOut = run_d2_behavior_pop_regression(y, popX, swX, ~, ~)
% run_d2_behavior_pop_regression OLS for d2 ~ 1 + popActivity + switchRate.
%
% Variables:
%   y     - d2 per window (vector).
%   popX  - popActivity predictor (often z-scored within area).
%   swX   - switch rate predictor (often z-scored within area).
%
% Goal:
%   Quantify unique contribution of switch rate after controlling for
%   popActivity (nested F-test, delta R^2, p-value on switch slope).
%
% Returns:
%   statsOut - struct with R2, deltaR2, p-values, betas.

n = numel(y);
if n < 4
    statsOut = struct('r2_full', nan, 'r2_popOnly', nan, 'deltaR2_switch', nan, ...
        'p_switch_slope', nan, 'f_p_nested', nan, 'beta_full', []);
    return;
end

Xf = [ones(n, 1), popX(:), swX(:)];
Xp = [ones(n, 1), popX(:)];

betaFull = Xf \ y(:);
yHatFull = Xf * betaFull;
sseFull = sum((y(:) - yHatFull).^2);

betaPop = Xp \ y(:);
yHatPop = Xp * betaPop;
ssePop = sum((y(:) - yHatPop).^2);

sst = sum((y(:) - mean(y)).^2);
if sst <= 0
    r2Full = 0;
    r2Pop = 0;
else
    r2Full = 1 - sseFull / sst;
    r2Pop = 1 - ssePop / sst;
end

deltaR2 = r2Full - r2Pop;

m = 3;
mse = sseFull / max(n - m, 1);
try
    varBeta = mse * pinv(Xf' * Xf);
    seBeta = sqrt(max(diag(varBeta), 0));
    tStat = betaFull ./ max(seBeta, eps);
    pSwitch = 2 * (1 - tcdf(abs(tStat(3)), n - m));
catch
    pSwitch = nan;
end

df1 = 1;
df2 = n - m;
if sseFull > 0 && df2 > 0
    fStat = ((ssePop - sseFull) / df1) / (sseFull / df2);
    fP = 1 - fcdf(fStat, df1, df2);
else
    fP = nan;
end

statsOut = struct();
statsOut.r2_full = r2Full;
statsOut.r2_popOnly = r2Pop;
statsOut.deltaR2_switch = deltaR2;
statsOut.p_switch_slope = pSwitch;
statsOut.f_p_nested = fP;
statsOut.beta_full = betaFull;
statsOut.sse_full = sseFull;
statsOut.sse_popOnly = ssePop;
end

function switchRate = calculate_switch_rate(labelsWindow, windowDurationSec)
% calculate_switch_rate Compute behavior label switching rate in one window.
%
% Variables:
%   labelsWindow      - Column vector of behavior label IDs for one window.
%   windowDurationSec - Duration of the window in seconds.
%
% Goal:
%   Quantify how often behavior labels switch in time, as:
%       (# of adjacent label changes) / (window duration in seconds).
%
% Returns:
%   switchRate - Switch rate in switches/second (NaN if invalid input).

if isempty(labelsWindow) || numel(labelsWindow) < 2 || windowDurationSec <= 0
    switchRate = nan;
    return;
end

labelsWindow = labelsWindow(:);
validMask = ~isnan(labelsWindow);
labelsWindow = labelsWindow(validMask);
if numel(labelsWindow) < 2
    switchRate = nan;
    return;
end

numSwitches = sum(diff(labelsWindow) ~= 0);
switchRate = numSwitches / windowDurationSec;
end

function labelsOut = recode_behavior_labels(labelsIn, behaviorRecodingRules)
% recode_behavior_labels Collapse behavior labels into super-categories.
%
% Variables:
%   labelsIn              - Column vector of original behavior labels.
%   behaviorRecodingRules - Cell array of remapping rules where each row is:
%                           { [oldLabels], newLabel }
%
% Goal:
%   Reassign multiple original behavior labels to user-defined super-category
%   labels before downstream switch-rate analysis.
%
% Returns:
%   labelsOut - Reassigned behavior labels (same shape as labelsIn).

labelsOut = labelsIn;

if isempty(behaviorRecodingRules)
    return;
end

if ~iscell(behaviorRecodingRules) || size(behaviorRecodingRules, 2) ~= 2
    error('behaviorRecodingRules must be an Nx2 cell array: { [oldLabels], newLabel }.');
end

for iRule = 1:size(behaviorRecodingRules, 1)
    oldLabels = behaviorRecodingRules{iRule, 1};
    newLabel = behaviorRecodingRules{iRule, 2};
    
    if isempty(oldLabels) || ~isnumeric(oldLabels) || ~isscalar(newLabel) || ~isnumeric(newLabel)
        error('Invalid behavior recoding rule at row %d.', iRule);
    end
    
    mask = ismember(labelsOut, oldLabels);
    labelsOut(mask) = newLabel;
end
end

