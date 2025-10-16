%%
% Criticality Reach-only Script (d2 + mrBr)
% Analyzes reach data only; saves results to a folder named after the data file

paths = get_paths;


% Sliding window size (seconds)
slidingWindowSize = 3;

% Flags
loadExistingResults = false;
makePlots = true;

% Discover all reach data files and process each
reachDir = fullfile(paths.dropPath, 'reach_task/data');
matFiles = dir(fullfile(reachDir, '*.mat'));
reachDataFiles = cell(1, numel(matFiles));
for i = 1:numel(matFiles)
    reachDataFiles{i} = fullfile(reachDir, matFiles(i).name);
end
reachDataFiles = cell(1);
% reachDataFiles{1} = fullfile(paths.reachDataPath, 'Y4_06-Oct-2023 14_14_53_NeuroBeh.mat');
reachDataFiles{1} = fullfile(paths.reachDataPath, 'makeSpikes.mat');
for fileIdx = 1:numel(reachDataFiles)
    % try
    run_reach_analysis_for_file(reachDataFiles{fileIdx}, slidingWindowSize, makePlots, paths);
    % catch ME
    % fprintf('Skipping %s due to error: %s\n', reachDataFiles{fileIdx}, ME.message);
    % end
end

return












function run_reach_analysis_for_file(reachDataFile, slidingWindowSize, makePlots, paths)

% Flags
analyzeD2 = true;      % compute d2
analyzeMrBr = false;    % compute mrBr

minSegmentLength = 50;

minSpikesPerBin = 3;
maxSpikesPerBin = 50;
minBinsPerWindow = 1000;

areasToTest = 1:4;


[~, dataBaseName, ~] = fileparts(reachDataFile);
saveDir = fullfile(paths.dropPath, 'reach_task/results', dataBaseName);
if ~exist(saveDir, 'dir'); mkdir(saveDir); end

resultsPath = fullfile(saveDir, sprintf('criticality_reach_ar_win%d.mat', slidingWindowSize));
% if exist(resultsPath, 'file')
%     fprintf('Results already exist for %s, skipping.\n', dataBaseName);
%     return
% end

dataR = load(reachDataFile);

opts = neuro_behavior_options;
opts.frameSize = .001;
opts.firingRateCheckTime = 5 * 60;
opts.collectStart = 0;
opts.collectFor = round(min(dataR.R(end,1) + 5000, max(dataR.CSV(:,1)*1000)) / 1000);
opts.minFiringRate = .1;
opts.maxFiringRate = 70;

[dataMatR, idLabels, areaLabels] = neural_matrix_mark_data(dataR, opts);
areas = {'M23', 'M56', 'DS', 'VS'};
idM23R = find(strcmp(areaLabels, 'M23'));
idM56R = find(strcmp(areaLabels, 'M56'));
idDSR = find(strcmp(areaLabels, 'DS'));
idVSR = find(strcmp(areaLabels, 'VS'));
idListRea = {idM23R, idM56R, idDSR, idVSR};

pcaFlag = 0;
pcaFirstFlag = 1;
nDim = 4;
thresholdFlag = 1;
thresholdPct = 0.75;
candidateFrameSizes = [.02 .03 .04 0.05, .075, 0.1 .15 .2];
candidateWindowSizes = [30, 45, 60, 90, 120];
windowSizes = repmat(slidingWindowSize, 1, 4);
pOrder = 10;
critType = 2;
d2StepSize = .02;

reconstructedDataMatRea = cell(1, length(areas));
for a = areasToTest
    aID = idListRea{a}; thisDataMat = dataMatR(:, aID);
    if pcaFlag
        [coeff, score, ~, ~, explained, mu] = pca(thisDataMat);
        forDim = find(cumsum(explained) > 30, 1); forDim = max(3, min(6, forDim));
        nDim = 1:forDim; reconstructedDataMatRea{a} = score(:,nDim) * coeff(:,nDim)' + mu;
    else
        reconstructedDataMatRea{a} = thisDataMat;
    end
end

optimalBinSizeRea = zeros(1, length(areas));
optimalWindowSizeRea = zeros(1, length(areas));
for a = areasToTest
    thisDataMat = reconstructedDataMatRea{a};
    [optimalBinSizeRea(a), optimalWindowSizeRea(a)] = ...
        find_optimal_bin_and_window(thisDataMat, candidateFrameSizes, candidateWindowSizes, minSpikesPerBin, maxSpikesPerBin, minBinsPerWindow);
    fprintf('Area %s: optimal bin size = %.3f s, optimal window size = %.1f s\n', areas{a}, optimalBinSizeRea(a), optimalWindowSizeRea(a));
end

d2StepSizeRea = optimalBinSizeRea;
d2WindowSizeRea = windowSizes;
validMask = isfinite(optimalBinSizeRea) & (optimalBinSizeRea > 0);
areasToTest = areasToTest(validMask);

    [popActivity, mrBrRea, d2Rea, startSRea, popActivityWindows, popActivityFull] = ...
        deal(cell(1, length(areas)));
for a = areasToTest
    fprintf('\nProcessing area %s (Reach)...\n', areas{a}); tic;
    aID = idListRea{a};
    stepSamples = round(d2StepSizeRea(a) / optimalBinSizeRea(a));
    winSamples = round(d2WindowSizeRea(a) / optimalBinSizeRea(a));

    % Skip this area if there aren't enough samples
    if winSamples < minSegmentLength
        fprintf('\nSkipping: Not enough data in %s (Reach)...\n', areas{a});
        continue
    end

    aDataMatRea = neural_matrix_ms_to_frames(dataMatR(:, aID), optimalBinSizeRea(a));
    numTimePoints = size(aDataMatRea, 1);
    numWindows = floor((numTimePoints - winSamples) / stepSamples) + 1;
    if pcaFlag
        [coeff, score, ~, ~, ~, mu] = pca(aDataMatRea);
        forDim = find(cumsum(var(score)) > 30, 1); forDim = max(3, min(6, forDim));
        nDim = 1:forDim; aDataMatRea = score(:,nDim) * coeff(:,nDim)' + mu;
    end
    popActivity{a} = round(sum(aDataMatRea, 2));
    [startSRea{a}, mrBrRea{a}, d2Rea{a}, popActivityWindows{a}, popActivityFull{a}] = ...
        deal(nan(1, numWindows));
    for w = 1:numWindows
        startIdx = (w - 1) * stepSamples + 1; endIdx = startIdx + winSamples - 1;
        startSRea{a}(w) = (startIdx + round(winSamples/2)-1) * optimalBinSizeRea(a);
        wPopActivity = popActivity{a}(startIdx:endIdx);
        popActivityWindows{a}(w) = mean(wPopActivity); % Store mean population activity for this window
        popActivityFull{a}(w) = popActivity{a}((startIdx + round(winSamples/2)-1));
        if analyzeMrBr
            result = branching_ratio_mr_estimation(wPopActivity);
            mrBrRea{a}(w) = result.branching_ratio;
        else
            mrBrRea{a}(w) = nan;
        end
        if analyzeD2
            [varphi, ~] = myYuleWalker3(wPopActivity, pOrder);
            d2Rea{a}(w) = getFixedPointDistance2(pOrder, critType, varphi);
        else
            d2Rea{a}(w) = nan;
        end
    end
    fprintf('Area %s completed in %.1f minutes\n', areas{a}, toc/60);
end

% Compute correlations between population activity and criticality measures
fprintf('\n=== Population Activity Correlations (Windowed) ===\n');
for a = areasToTest
    if ~isempty(popActivityWindows{a}) && ~isempty(d2Rea{a}) && ~isempty(mrBrRea{a})
            
        % Remove NaN values for correlation
        validIdx = ~isnan(popActivityWindows{a}) & ~isnan(d2Rea{a});
        if sum(validIdx) > 10 % Need sufficient data points
            popAct = popActivity{a}(validIdx);
            d2Vals = d2Rea{a}(validIdx);
            % Correlate popActivity with d2
            [rPopD2, pPopD2] = corrcoef(popAct, d2Vals);
            fprintf('Area %s: PopActivity (windowed) vs d2: r=%.3f, p=%.3f (n=%d)\n', areas{a}, rPopD2(1,2), pPopD2(1,2), sum(validIdx));
        else
            fprintf('Area %s: Insufficient d2 valid data points for correlation (n=%d)\n', areas{a}, sum(validIdx));
        end   
        validIdx = ~isnan(popActivityWindows{a}) & ~isnan(mrBrRea{a});
            popAct = popActivity{a}(validIdx);
            mrBrVals = mrBrRea{a}(validIdx);
        if sum(validIdx) > 10 % Need sufficient data points
            [rPopMrBr, pPopMrBr] = corrcoef(popAct, mrBrVals);
            fprintf('Area %s: PopActivity (windowed) vs mrBr: r=%.3f, p=%.3f (n=%d)\n', areas{a}, rPopMrBr(1,2), pPopMrBr(1,2), sum(validIdx));
        else
            fprintf('Area %s: Insufficient mrBR valid data points for correlation (n=%d)\n', areas{a}, sum(validIdx));
        end
    end
end

% Compute correlations between full population activity and criticality measures
fprintf('\n=== Full Population Activity Correlations ===\n');
for a = areasToTest
    if ~isempty(popActivity{a}) && ~isempty(d2Rea{a}) && ~isempty(mrBrRea{a})
            
        % Remove NaN values for correlation (popActivity is already time-locked to d2/mrBr)
        validIdx = ~isnan(popActivityFull{a}) & ~isnan(d2Rea{a});
            popAct = popActivityFull{a}(validIdx);
            d2Vals = d2Rea{a}(validIdx);
        if sum(validIdx) > 10 % Need sufficient data points
            % Correlate full popActivity with d2
            [rPopD2, pPopD2] = corrcoef(popAct, d2Vals);
            fprintf('Area %s: PopActivity (full) vs d2: r=%.3f, p=%.3f (n=%d)\n', areas{a}, rPopD2(1,2), pPopD2(1,2), sum(validIdx));
         else
            fprintf('Area %s: Insufficient d2 valid data points for full correlation (n=%d)\n', areas{a}, sum(validIdx));
        end
           
        validIdx = ~isnan(popActivityFull{a}) & ~isnan(mrBrRea{a});
            popAct = popActivityFull{a}(validIdx);
            mrBrVals = mrBrRea{a}(validIdx);
        if sum(validIdx) > 10 % Need sufficient data points
            % Correlate full popActivity with mrBr
            [rPopMrBr, pPopMrBr] = corrcoef(popAct, mrBrVals);
            fprintf('Area %s: PopActivity (full) vs mrBr: r=%.3f, p=%.3f (n=%d)\n', areas{a}, rPopMrBr(1,2), pPopMrBr(1,2), sum(validIdx));
        else
            fprintf('Area %s: Insufficient mrBr valid data points for full correlation (n=%d)\n', areas{a}, sum(validIdx));
        end
    end
end

results = struct(); 
results.areas = areas; 
results.reach.mrBr = mrBrRea; 
results.reach.d2 = d2Rea; 
results.reach.startS = startSRea;
results.reach.popActivity = popActivityWindows;
results.reach.optimalBinSize = optimalBinSizeRea; 
results.reach.optimalWindowSize = optimalWindowSizeRea;
results.reach.d2StepSize = d2StepSizeRea; results.reach.d2WindowSize = d2WindowSizeRea;
save(resultsPath, 'results'); fprintf('Saved reach-only d2/mrBr to %s\n', resultsPath);

if makePlots
    % Detect monitors and size figure to full screen (prefer second monitor if present)
    monitorPositions = get(0, 'MonitorPositions');
    monitorOne = monitorPositions(1, :);
    monitorTwo = monitorPositions(size(monitorPositions, 1), :);
    if size(monitorPositions, 1) >= 2
        targetPos = monitorTwo;
    else
        targetPos = monitorOne;
    end
    figure(900); clf;
    set(gcf, 'Units', 'pixels');
    set(gcf, 'Position', targetPos);
    % drawnow; % let graphics settle before adding axes
    numRows = length(areasToTest);
    ha = tight_subplot(numRows, 1, [0.08 0.04], [0.15 0.1], [0.08 0.04]);
    for idx = 1:length(areasToTest)
        a = areasToTest(idx); 
        axes(ha(idx)); hold on;
        
        % Normalize population activity to 0-1 range for plotting
        if ~isempty(popActivityWindows{a}) && any(~isnan(popActivityWindows{a}))
            popActNorm = (popActivityWindows{a} - min(popActivityWindows{a}(~isnan(popActivityWindows{a})))) / ...
                        (max(popActivityWindows{a}(~isnan(popActivityWindows{a}))) - min(popActivityWindows{a}(~isnan(popActivityWindows{a}))));
            popActNorm(isnan(popActivityWindows{a})) = nan;
        else
            popActNorm = nan(size(popActivityWindows{a}));
        end
        
        if analyzeD2
            yyaxis left; 
            plot(startSRea{a}, d2Rea{a}, '-', 'Color', [0 0 1], 'LineWidth', 2); 
            ylabel('d2', 'Color', [0 0 1]); ylim('auto');
        end
        if analyzeMrBr
            yyaxis right; 
            plot(startSRea{a}, mrBrRea{a}, '-', 'Color', [0 0 0], 'LineWidth', 2); 
            yline(1, 'k:', 'LineWidth', 1.5); 
            ylabel('mrBr', 'Color', [0 0 0]); ylim('auto');
        end
        
        % Add normalized population activity as a third y-axis (using left axis with different color)
        if ~isempty(popActNorm) && any(~isnan(popActNorm))
            yyaxis left;
            plot(startSRea{a}, popActNorm, '-', 'Color', [1 0 0], 'LineWidth', 1, 'LineStyle', ':'); 
        end
        
        % Also add full population activity (normalized) for comparison
        if ~isempty(popActivity{a}) && any(~isnan(popActivity{a}))
            % Get time points for full population activity
            timeIndices = (1:length(popActivity{a})) * optimalBinSizeRea(a);
            % Normalize full population activity
            popActFullNorm = (popActivity{a} - min(popActivity{a}(~isnan(popActivity{a})))) / ...
                           (max(popActivity{a}(~isnan(popActivity{a}))) - min(popActivity{a}(~isnan(popActivity{a}))));
            popActFullNorm(isnan(popActivity{a})) = nan;
            
            yyaxis left;
            plot(timeIndices, popActFullNorm, '-', 'Color', [0.8 0.4 0.4], 'LineWidth', 0.5, 'LineStyle', '-.'); 
        end
        
        if ~isempty(startSRea{a})
        xlim([startSRea{a}(1) startSRea{a}(end)])
        end
        title(sprintf('%s - d2 (blue), mrBr (black), PopActWin (red dotted), PopActFull (brown dash-dot)', areas{a})); xlabel('Time (s)'); grid on; set(gca, 'XTickLabelMode', 'auto'); set(gca, 'YTickLabelMode', 'auto');
    end
    sgtitle(sprintf('Reach-only d2 (blue, left) and mrBr (black, right) - win=%gs', slidingWindowSize));
    exportgraphics(gcf, fullfile(saveDir, sprintf('criticality_reach_ar_win%d.png', slidingWindowSize)), 'Resolution', 300);
end

% Create scatter plots for correlations
if makePlots
    figure(901); clf;
    set(gcf, 'Units', 'pixels');
    set(gcf, 'Position', targetPos);
    
    % 4 rows x numAreas columns: d2 vs popActivityFull, d2 vs popActivityWindows, 
    % mrBr vs popActivityFull, mrBr vs popActivityWindows
    numRows = 4;
    numCols = length(areasToTest);
    ha = tight_subplot(numRows, numCols, [0.08 0.04], [0.08 0.1], [0.06 0.04]);
    
    for idx = 1:length(areasToTest)
        a = areasToTest(idx);
        
        % Row 1: d2 vs popActivityFull
        axes(ha(idx));
        if ~isempty(popActivityFull{a}) && ~isempty(d2Rea{a})
            validIdx = ~isnan(popActivityFull{a}) & ~isnan(d2Rea{a});
            if sum(validIdx) > 5
                scatter(popActivityFull{a}(validIdx), d2Rea{a}(validIdx), 20, 'filled', 'MarkerFaceAlpha', 0.6);
                [r, p] = corrcoef(popActivityFull{a}(validIdx), d2Rea{a}(validIdx));
                title(sprintf('%s: d2 vs PopActFull\nr=%.3f, p=%.3f', areas{a}, r(1,2), p(1,2)));
            else
                title(sprintf('%s: d2 vs PopActFull\nInsufficient data', areas{a}));
            end
        end
        xlabel('PopActivity Full'); ylabel('d2');
        grid on;
        
        % Row 2: d2 vs popActivityWindows
        axes(ha(numCols + idx));
        if ~isempty(popActivityWindows{a}) && ~isempty(d2Rea{a})
            validIdx = ~isnan(popActivityWindows{a}) & ~isnan(d2Rea{a});
            if sum(validIdx) > 5
                scatter(popActivityWindows{a}(validIdx), d2Rea{a}(validIdx), 20, 'filled', 'MarkerFaceAlpha', 0.6);
                [r, p] = corrcoef(popActivityWindows{a}(validIdx), d2Rea{a}(validIdx));
                title(sprintf('%s: d2 vs PopActWin\nr=%.3f, p=%.3f', areas{a}, r(1,2), p(1,2)));
            else
                title(sprintf('%s: d2 vs PopActWin\nInsufficient data', areas{a}));
            end
        end
        xlabel('PopActivity Windows'); ylabel('d2');
        grid on;
        
        % Row 3: mrBr vs popActivityFull
        axes(ha(2*numCols + idx));
        if ~isempty(popActivityFull{a}) && ~isempty(mrBrRea{a})
            validIdx = ~isnan(popActivityFull{a}) & ~isnan(mrBrRea{a});
            if sum(validIdx) > 5
                scatter(popActivityFull{a}(validIdx), mrBrRea{a}(validIdx), 20, 'filled', 'MarkerFaceAlpha', 0.6);
                [r, p] = corrcoef(popActivityFull{a}(validIdx), mrBrRea{a}(validIdx));
                title(sprintf('%s: mrBr vs PopActFull\nr=%.3f, p=%.3f', areas{a}, r(1,2), p(1,2)));
            else
                title(sprintf('%s: mrBr vs PopActFull\nInsufficient data', areas{a}));
            end
        end
        xlabel('PopActivity Full'); ylabel('mrBr');
        grid on;
        
        % Row 4: mrBr vs popActivityWindows
        axes(ha(3*numCols + idx));
        if ~isempty(popActivityWindows{a}) && ~isempty(mrBrRea{a})
            validIdx = ~isnan(popActivityWindows{a}) & ~isnan(mrBrRea{a});
            if sum(validIdx) > 5
                scatter(popActivityWindows{a}(validIdx), mrBrRea{a}(validIdx), 20, 'filled', 'MarkerFaceAlpha', 0.6);
                [r, p] = corrcoef(popActivityWindows{a}(validIdx), mrBrRea{a}(validIdx));
                title(sprintf('%s: mrBr vs PopActWin\nr=%.3f, p=%.3f', areas{a}, r(1,2), p(1,2)));
            else
                title(sprintf('%s: mrBr vs PopActWin\nInsufficient data', areas{a}));
            end
        end
        xlabel('PopActivity Windows'); ylabel('mrBr');
        grid on;
    end
    
    sgtitle(sprintf('Population Activity vs Criticality Correlations - win=%gs', slidingWindowSize));
    exportgraphics(gcf, fullfile(saveDir, sprintf('criticality_correlations_win%d.png', slidingWindowSize)), 'Resolution', 300);
    fprintf('Saved correlation scatter plots to: %s\n', fullfile(saveDir, sprintf('criticality_correlations_win%d.png', slidingWindowSize)));
end
end
