%%
% Criticality Sliding Window Analysis Script (d2 + mrBr)
% Unified script for analyzing both reach data and naturalistic data
% Analyzes data using sliding window approach; saves results to data-specific folders

paths = get_paths;

% =============================    Configuration    =============================
% Data type selection
dataType = 'reach';  % 'reach' or 'naturalistic'

% Sliding window size (seconds)
slidingWindowSize = 3;

% Flags
loadExistingResults = false;
makePlots = true;

% Analysis flags
analyzeD2 = true;      % compute d2
analyzeMrBr = false;   % compute mrBr

% Analysis parameters
minSegmentLength = 50;
minSpikesPerBin = 3;
maxSpikesPerBin = 50;
minBinsPerWindow = 1000;

% Areas to analyze
areasToTest = 1:4;
areasToPlot = areasToTest;

% PCA options
pcaFlag = 0;           % Set to 1 to use PCA
pcaFirstFlag = 1;      % Use first nDim dimensions if 1, last nDim if 0
nDim = 4;              % Number of PCA dimensions to use

% Threshold options
thresholdFlag = 1;     % Set to 1 to use threshold method
thresholdPct = 0.75;   % Threshold as percentage of median

% Optimal bin/window size search parameters
candidateFrameSizes = [.02 .03 .04 0.05, .075, 0.1 .15 .2];
candidateWindowSizes = [30, 45, 60, 90, 120];
windowSizes = repmat(slidingWindowSize, 1, 4);
pOrder = 10;
critType = 2;
d2StepSize = .02;

%% =============================    Data Loading    =============================
fprintf('\n=== Loading %s data ===\n', dataType);

if strcmp(dataType, 'reach')
    % Load reach data
    reachDataFile = fullfile(paths.reachDataPath, 'Y4_06-Oct-2023 14_14_53_NeuroBeh.mat');
    % reachDataFile = fullfile(paths.reachDataPath, 'makeSpikes.mat');
    
    [~, dataBaseName, ~] = fileparts(reachDataFile);
    saveDir = fullfile(paths.dropPath, 'reach_task/results', dataBaseName);
    if ~exist(saveDir, 'dir'); mkdir(saveDir); end
    
    resultsPath = fullfile(saveDir, sprintf('criticality_sliding_window_ar_win%d.mat', slidingWindowSize));
    
    dataR = load(reachDataFile);
    
    opts = neuro_behavior_options;
    opts.frameSize = .001;
    opts.firingRateCheckTime = 5 * 60;
    opts.collectStart = 0;
    opts.collectFor = round(min(dataR.R(end,1) + 5000, max(dataR.CSV(:,1)*1000)) / 1000);
    opts.minFiringRate = .1;
    opts.maxFiringRate = 70;
    
    [dataMat, idLabels, areaLabels] = neural_matrix_mark_data(dataR, opts);
    areas = {'M23', 'M56', 'DS', 'VS'};
    idM23 = find(strcmp(areaLabels, 'M23'));
    idM56 = find(strcmp(areaLabels, 'M56'));
    idDS = find(strcmp(areaLabels, 'DS'));
    idVS = find(strcmp(areaLabels, 'VS'));
    idList = {idM23, idM56, idDS, idVS};
    
elseif strcmp(dataType, 'naturalistic')
    % Load naturalistic data
    getDataType = 'spikes';
    opts.firingRateCheckTime = 5 * 60;
    opts.collectStart = 0 * 60; % seconds
    opts.collectFor = 45 * 60; % seconds
    opts.minFiringRate = .05;
    get_standard_data
    
    areas = {'M23', 'M56', 'DS', 'VS'};
    idList = {idM23, idM56, idDS, idVS};
    
    % Create save directory for naturalistic data
    saveDir = fullfile(paths.dropPath, 'criticality/results');
    if ~exist(saveDir, 'dir'); mkdir(saveDir); end
    resultsPath = fullfile(saveDir, sprintf('criticality_sliding_window_ar_win%d.mat', slidingWindowSize));
    
else
    error('Invalid dataType. Must be ''reach'' or ''naturalistic''');
end

% =============================    Analysis    =============================
fprintf('\n=== %s Data Analysis ===\n', dataType);

% Adjust areasToTest based on which areas have data
% areasToTest = areasToTest(~cellfun(@isempty, idList));

% Step 1-2: Apply PCA to original data if requested
fprintf('\n--- Step 1-2: PCA on original data if requested ---\n');
reconstructedDataMat = cell(1, length(areas));
for a = areasToTest
    aID = idList{a}; 
    thisDataMat = dataMat(:, aID);
    if pcaFlag
        [coeff, score, ~, ~, explained, mu] = pca(thisDataMat);
        forDim = find(cumsum(explained) > 30, 1); 
        forDim = max(3, min(6, forDim));
        nDim = 1:forDim; 
        reconstructedDataMat{a} = score(:,nDim) * coeff(:,nDim)' + mu;
    else
        reconstructedDataMat{a} = thisDataMat;
    end
end

% Step 3: Find optimal parameters using reconstructed data
fprintf('\n--- Step 3: Finding optimal parameters ---\n');
optimalBinSize = zeros(1, length(areas));
optimalWindowSize = zeros(1, length(areas));
for a = areasToTest
    thisDataMat = reconstructedDataMat{a};
    [optimalBinSize(a), optimalWindowSize(a)] = ...
        find_optimal_bin_and_window(thisDataMat, candidateFrameSizes, candidateWindowSizes, minSpikesPerBin, maxSpikesPerBin, minBinsPerWindow);
    fprintf('Area %s: optimal bin size = %.3f s, optimal window size = %.1f s\n', areas{a}, optimalBinSize(a), optimalWindowSize(a));
end

% Use optimal bin sizes for each area
d2StepSizeData = optimalBinSize;
d2WindowSizeData = windowSizes;
validMask = isfinite(optimalBinSize(areasToTest)) & (optimalBinSize(areasToTest) > 0);
areasToTest = areasToTest(validMask);

% Initialize results
[popActivity, mrBr, d2, startS, popActivityWindows, popActivityFull] = ...
    deal(cell(1, length(areas)));

for a = areasToTest
    fprintf('\nProcessing area %s (%s)...\n', areas{a}, dataType); 
    tic;
    aID = idList{a};
    stepSamples = round(d2StepSizeData(a) / optimalBinSize(a));
    winSamples = round(d2WindowSizeData(a) / optimalBinSize(a));

    % Skip this area if there aren't enough samples
    if winSamples < minSegmentLength
        fprintf('\nSkipping: Not enough data in %s (%s)...\n', areas{a}, dataType);
        continue
    end

    aDataMat = neural_matrix_ms_to_frames(dataMat(:, aID), optimalBinSize(a));
    numTimePoints = size(aDataMat, 1);
    numWindows = floor((numTimePoints - winSamples) / stepSamples) + 1;
    if pcaFlag
        [coeff, score, ~, ~, explained, mu] = pca(aDataMat);
        forDim = find(cumsum(explained) > 30, 1); 
        forDim = max(3, min(6, forDim));
        nDim = 1:forDim; 
        aDataMat = score(:,nDim) * coeff(:,nDim)' + mu;
    end
    popActivity{a} = round(sum(aDataMat, 2));
    [startS{a}, mrBr{a}, d2{a}, popActivityWindows{a}, popActivityFull{a}] = ...
        deal(nan(1, numWindows));
    for w = 1:numWindows
        startIdx = (w - 1) * stepSamples + 1; 
        endIdx = startIdx + winSamples - 1;
        startS{a}(w) = (startIdx + round(winSamples/2)-1) * optimalBinSize(a);
        wPopActivity = popActivity{a}(startIdx:endIdx);
        popActivityWindows{a}(w) = mean(wPopActivity); % Store mean population activity for this window
        popActivityFull{a}(w) = popActivity{a}((startIdx + round(winSamples/2)-1));
        if analyzeMrBr
            result = branching_ratio_mr_estimation(wPopActivity);
            mrBr{a}(w) = result.branching_ratio;
        else
            mrBr{a}(w) = nan;
        end
        if analyzeD2
            [varphi, ~] = myYuleWalker3(wPopActivity, pOrder);
            d2{a}(w) = getFixedPointDistance2(pOrder, critType, varphi);
        else
            d2{a}(w) = nan;
        end
    end
    fprintf('Area %s completed in %.1f minutes\n', areas{a}, toc/60);
end

% =============================    Correlations    =============================
% Compute correlations between population activity and criticality measures
fprintf('\n=== Population Activity Correlations (Windowed) ===\n');
for a = areasToTest
    if ~isempty(popActivityWindows{a}) && ~isempty(d2{a}) && ~isempty(mrBr{a})
            
        % Remove NaN values for correlation
        validIdx = ~isnan(popActivityWindows{a}) & ~isnan(d2{a});
        if sum(validIdx) > 10 % Need sufficient data points
            popAct = popActivity{a}(validIdx);
            d2Vals = d2{a}(validIdx);
            % Correlate popActivity with d2
            [rPopD2, pPopD2] = corrcoef(popAct, d2Vals);
            fprintf('Area %s: PopActivity (windowed) vs d2: r=%.3f, p=%.3f (n=%d)\n', areas{a}, rPopD2(1,2), pPopD2(1,2), sum(validIdx));
        else
            fprintf('Area %s: Insufficient d2 valid data points for correlation (n=%d)\n', areas{a}, sum(validIdx));
        end   
        validIdx = ~isnan(popActivityWindows{a}) & ~isnan(mrBr{a});
            popAct = popActivity{a}(validIdx);
            mrBrVals = mrBr{a}(validIdx);
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
    if ~isempty(popActivity{a}) && ~isempty(d2{a}) && ~isempty(mrBr{a})
            
        % Remove NaN values for correlation (popActivity is already time-locked to d2/mrBr)
        validIdx = ~isnan(popActivityFull{a}) & ~isnan(d2{a});
            popAct = popActivityFull{a}(validIdx);
            d2Vals = d2{a}(validIdx);
        if sum(validIdx) > 10 % Need sufficient data points
            % Correlate full popActivity with d2
            [rPopD2, pPopD2] = corrcoef(popAct, d2Vals);
            fprintf('Area %s: PopActivity (full) vs d2: r=%.3f, p=%.3f (n=%d)\n', areas{a}, rPopD2(1,2), pPopD2(1,2), sum(validIdx));
         else
            fprintf('Area %s: Insufficient d2 valid data points for full correlation (n=%d)\n', areas{a}, sum(validIdx));
        end
           
        validIdx = ~isnan(popActivityFull{a}) & ~isnan(mrBr{a});
            popAct = popActivityFull{a}(validIdx);
            mrBrVals = mrBr{a}(validIdx);
        if sum(validIdx) > 10 % Need sufficient data points
            % Correlate full popActivity with mrBr
            [rPopMrBr, pPopMrBr] = corrcoef(popAct, mrBrVals);
            fprintf('Area %s: PopActivity (full) vs mrBr: r=%.3f, p=%.3f (n=%d)\n', areas{a}, rPopMrBr(1,2), pPopMrBr(1,2), sum(validIdx));
        else
            fprintf('Area %s: Insufficient mrBr valid data points for full correlation (n=%d)\n', areas{a}, sum(validIdx));
        end
    end
end

% =============================    Save Results    =============================
results = struct(); 
results.dataType = dataType;
results.areas = areas; 
results.mrBr = mrBr; 
results.d2 = d2; 
results.startS = startS;
results.popActivity = popActivityWindows;
results.optimalBinSize = optimalBinSize; 
results.optimalWindowSize = optimalWindowSize;
results.d2StepSize = d2StepSizeData; 
results.d2WindowSize = d2WindowSizeData;
results.params.slidingWindowSize = slidingWindowSize;
results.params.analyzeD2 = analyzeD2;
results.params.analyzeMrBr = analyzeMrBr;
results.params.pcaFlag = pcaFlag;
results.params.pcaFirstFlag = pcaFirstFlag;
results.params.nDim = nDim;
results.params.pOrder = pOrder;
results.params.critType = critType;

save(resultsPath, 'results'); 
fprintf('Saved %s d2/mrBr to %s\n', dataType, resultsPath);

% =============================    Plotting    =============================
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
    
    % Time series plot
    figure(900); clf;
    set(gcf, 'Units', 'pixels');
    set(gcf, 'Position', targetPos);
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
            plot(startS{a}, d2{a}, '-', 'Color', [0 0 1], 'LineWidth', 2); 
            ylabel('d2', 'Color', [0 0 1]); ylim('auto');
        end
        if analyzeMrBr
            yyaxis right; 
            plot(startS{a}, mrBr{a}, '-', 'Color', [0 0 0], 'LineWidth', 2); 
            yline(1, 'k:', 'LineWidth', 1.5); 
            ylabel('mrBr', 'Color', [0 0 0]); ylim('auto');
        end
        
        % Add normalized population activity as a third y-axis (using left axis with different color)
        if ~isempty(popActNorm) && any(~isnan(popActNorm))
            yyaxis left;
            plot(startS{a}, popActNorm, '-', 'Color', [1 0 0], 'LineWidth', 1, 'LineStyle', ':'); 
        end
        
        % Also add full population activity (normalized) for comparison
        if ~isempty(popActivity{a}) && any(~isnan(popActivity{a}))
            % Get time points for full population activity
            timeIndices = (1:length(popActivity{a})) * optimalBinSize(a);
            % Normalize full population activity
            popActFullNorm = (popActivity{a} - min(popActivity{a}(~isnan(popActivity{a})))) / ...
                           (max(popActivity{a}(~isnan(popActivity{a}))) - min(popActivity{a}(~isnan(popActivity{a}))));
            popActFullNorm(isnan(popActivity{a})) = nan;
            
            yyaxis left;
            plot(timeIndices, popActFullNorm, '-', 'Color', [0.8 0.4 0.4], 'LineWidth', 0.5, 'LineStyle', '-.'); 
        end
        
        if ~isempty(startS{a})
        xlim([startS{a}(1) startS{a}(end)])
        end
        title(sprintf('%s - d2 (blue), mrBr (black), PopActWin (red dotted), PopActFull (brown dash-dot)', areas{a})); 
        xlabel('Time (s)'); grid on; set(gca, 'XTickLabelMode', 'auto'); set(gca, 'YTickLabelMode', 'auto');
    end
    sgtitle(sprintf('%s d2 (blue, left) and mrBr (black, right) - win=%gs', dataType, slidingWindowSize));
    exportgraphics(gcf, fullfile(saveDir, sprintf('criticality_%s_ar_win%d.png', dataType, slidingWindowSize)), 'Resolution', 300);
    
    % Scatter plots for correlations
    figure(901); clf;
    set(gcf, 'Units', 'pixels');
    set(gcf, 'Position', targetPos);
    
    % Use original areasToTest values to determine numAreas
    numAreas = length(areasToPlot);
    
    % 2 rows x 4 columns: d2 vs popActivityFull, d2 vs popActivityWindows
    numRows = 2;
    numCols = 4;
    ha = tight_subplot(numRows, numCols, [0.08 0.04], [0.08 0.1], [0.06 0.04]);
    
    for idx = 1:numAreas
        a = areasToPlot(idx);
        
        % Row 1: d2 vs popActivityFull
        axes(ha(idx));
        if ~isempty(popActivityFull{a}) && ~isempty(d2{a})
            validIdx = ~isnan(popActivityFull{a}) & ~isnan(d2{a});
            if sum(validIdx) > 5
                xData = popActivityFull{a}(validIdx);
                yData = d2{a}(validIdx);
                scatter(xData, yData, 20, 'filled', 'MarkerFaceAlpha', 0.6);
                [r, p] = corrcoef(xData, yData);
                title(sprintf('%s: d2 vs PopActFull\nr=%.3f, p=%.3f', areas{a}, r(1,2), p(1,2)));
                
                % Add best fitting regression line
                hold on;
                p_fit = polyfit(xData, yData, 1);
                x_fit = linspace(min(xData), max(xData), 100);
                y_fit = polyval(p_fit, x_fit);
                plot(x_fit, y_fit, 'r-', 'LineWidth', 2);
            else
                title(sprintf('%s: d2 vs PopActFull\nInsufficient data', areas{a}));
            end
        end
        xlabel('PopActivity Full'); ylabel('d2');
        grid on;
        
        % Row 2: d2 vs popActivityWindows
        axes(ha(numAreas + idx));
        if ~isempty(popActivityWindows{a}) && ~isempty(d2{a})
            validIdx = ~isnan(popActivityWindows{a}) & ~isnan(d2{a});
            if sum(validIdx) > 5
                xData = popActivityWindows{a}(validIdx);
                yData = d2{a}(validIdx);
                scatter(xData, yData, 20, 'filled', 'MarkerFaceAlpha', 0.6);
                [r, p] = corrcoef(xData, yData);
                title(sprintf('%s: d2 vs PopActWin\nr=%.3f, p=%.3f', areas{a}, r(1,2), p(1,2)));
                
                % Add best fitting regression line
                hold on;
                p_fit = polyfit(xData, yData, 1);
                x_fit = linspace(min(xData), max(xData), 100);
                y_fit = polyval(p_fit, x_fit);
                plot(x_fit, y_fit, 'r-', 'LineWidth', 2);
            else
                title(sprintf('%s: d2 vs PopActWin\nInsufficient data', areas{a}));
            end
        end
        xlabel('PopActivity Windows'); ylabel('d2');
        grid on;
    end
    
    sgtitle(sprintf('%s Population Activity vs Criticality Correlations - win=%gs', dataType, slidingWindowSize));
    exportgraphics(gcf, fullfile(saveDir, sprintf('criticality_%s_correlations_win%d.png', dataType, slidingWindowSize)), 'Resolution', 300);
    fprintf('Saved %s correlation scatter plots to: %s\n', dataType, fullfile(saveDir, sprintf('criticality_%s_correlations_win%d.png', dataType, slidingWindowSize)));
end

fprintf('\n=== %s Analysis Complete ===\n', dataType);
