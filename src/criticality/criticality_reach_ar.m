%%
% Criticality Reach-only Script (d2 + mrBr)
% Analyzes reach data only; saves results to a folder named after the data file

paths = get_paths;


% Sliding window size (seconds)
slidingWindowSize = 4;

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
% reachDataFiles{1} = fullfile(paths.dropPath, 'reach_task/data/Copy_of_Y4_100623_Spiketimes_idchan_BEH.mat');
reachDataFiles{1} = fullfile(paths.dropPath, 'reach_task/data/makeSpikes.mat');
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
opts.collectFor = round(dataR.R(end,1) + 5000) / 1000;
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

[mrBrRea, d2Rea, startSRea] = deal(cell(1, length(areas)));
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
    aDataMatRea = round(sum(aDataMatRea, 2));
    startSRea{a} = nan(1, numWindows); mrBrRea{a} = nan(1, numWindows); d2Rea{a} = nan(1, numWindows);
    for w = 1:numWindows
        startIdx = (w - 1) * stepSamples + 1; endIdx = startIdx + winSamples - 1;
        startSRea{a}(w) = (startIdx + round(winSamples/2)-1) * optimalBinSizeRea(a);
        wPopActivity = aDataMatRea(startIdx:endIdx);
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

results = struct(); 
results.areas = areas; 
results.reach.mrBr = mrBrRea; 
results.reach.d2 = d2Rea; 
results.reach.startS = startSRea;
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
        if ~isempty(startSRea{a})
        xlim([startSRea{a}(1) startSRea{a}(end)])
        end
        title(sprintf('%s - d2 (blue) & mrBr (black)', areas{a})); xlabel('Time (s)'); grid on; set(gca, 'XTickLabelMode', 'auto'); set(gca, 'YTickLabelMode', 'auto');
    end
    sgtitle(sprintf('Reach-only d2 (blue, left) and mrBr (black, right) - win=%gs', slidingWindowSize));
    exportgraphics(gcf, fullfile(saveDir, sprintf('criticality_reach_ar_win%d.png', slidingWindowSize)), 'Resolution', 300);
end
end
