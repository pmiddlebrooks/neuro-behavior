%%
% Criticality Reach-only Script (d2 + mrBr)
% Analyzes reach data only; saves results to a folder named after the data file

paths = get_paths;

% Flags
loadExistingResults = false;
makePlots = true;

% Sliding window size (seconds)
slidingWindowSize = 3;

% Reach data file and save directory
reachDataFile = fullfile(paths.dropPath, 'reach_data/Copy_of_Y4_100623_Spiketimes_idchan_BEH.mat');
[~, dataBaseName, ~] = fileparts(reachDataFile);
saveDir = fullfile(paths.dropPath, 'reach_data', dataBaseName);
if ~exist(saveDir, 'dir'); mkdir(saveDir); end

resultsPath = fullfile(saveDir, sprintf('criticality_reach_ar_win%d.mat', slidingWindowSize));

% Options
opts = neuro_behavior_options;
opts.firingRateCheckTime = 5 * 60;
opts.collectStart = 0;
opts.collectFor = 45 * 60;

% Load reach data
dataR = load(reachDataFile);
[dataMatR, idLabels, areaLabels] = neural_matrix_mark_data(dataR, opts);
areas = {'M23', 'M56', 'DS', 'VS'};
idM23R = find(strcmp(areaLabels, 'M23'));
idM56R = find(strcmp(areaLabels, 'M56'));
idDSR = find(strcmp(areaLabels, 'DS'));
idVSR = find(strcmp(areaLabels, 'VS'));
idListRea = {idM23R, idM56R, idDSR, idVSR};
areasToTest = 1:4;

% Params
pcaFlag = 0; pcaFirstFlag = 1; nDim = 4;
thresholdFlag = 1; thresholdPct = 0.75;
candidateFrameSizes = [.02 .03 .04 0.05, .075, 0.1 .15];
candidateWindowSizes = [30, 45, 60, 90, 120];
windowSizes = repmat(slidingWindowSize, 1, 4);
pOrder = 10; critType = 2; d2StepSize = .02;

% Step 1-3: PCA optional and optimal bin/window search
reconstructedDataMatRea = cell(1, length(areas));
for a = areasToTest
    aID = idListRea{a}; thisDataMat = dataMatR(:, aID);
    if pcaFlag
        [coeff, score, ~, ~, explained, mu] = pca(thisDataMat);
        forDim = find(cumsum(explained) > 30, 1); forDim = max(3, min(6, forDim));
        nDim = 1:forDim;
        reconstructedDataMatRea{a} = score(:,nDim) * coeff(:,nDim)' + mu;
    else
        reconstructedDataMatRea{a} = thisDataMat;
    end
end

optimalBinSizeRea = zeros(1, length(areas));
optimalWindowSizeRea = zeros(1, length(areas));
for a = areasToTest
    thisDataMat = reconstructedDataMatRea{a};
    [optimalBinSizeRea(a), optimalWindowSizeRea(a)] = find_optimal_bin_and_window(thisDataMat, candidateFrameSizes, candidateWindowSizes, 3, 50, 1000);
end

d2StepSizeRea = optimalBinSizeRea; d2WindowSizeRea = windowSizes;

% Analyze reach data
mrBrRea = cell(1, length(areas)); d2Rea = cell(1, length(areas)); startSRea = cell(1, length(areas));
for a = areasToTest
    aID = idListRea{a};
    stepSamples = round(d2StepSizeRea(a) / optimalBinSizeRea(a));
    winSamples = round(d2WindowSizeRea(a) / optimalBinSizeRea(a));
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
        result = branching_ratio_mr_estimation(wPopActivity);
        mrBrRea{a}(w) = result.branching_ratio;
        [varphi, ~] = myYuleWalker3(wPopActivity, pOrder);
        d2Rea{a}(w) = getFixedPointDistance2(pOrder, critType, varphi);
    end
end

% Save results
results = struct(); results.areas = areas; results.reach.mrBr = mrBrRea; results.reach.d2 = d2Rea; results.reach.startS = startSRea;
results.reach.optimalBinSize = optimalBinSizeRea; results.reach.optimalWindowSize = optimalWindowSizeRea;
results.reach.d2StepSize = d2StepSizeRea; results.reach.d2WindowSize = d2WindowSizeRea;
save(resultsPath, 'results'); fprintf('Saved reach-only d2/mrBr to %s\n', resultsPath);

% Plotting: 2 rows (d2, mrBr) x num areas (columns)
if makePlots
    figure(900); clf; set(gcf, 'Position', [100 100 1400 600]);
    for idx = 1:length(areasToTest)
        a = areasToTest(idx);
        % d2 (top)
        subplot(2, length(areasToTest), idx);
        hold on;
        plot(startSRea{a}, d2Rea{a}, '-', 'Color', [0 0 1], 'LineWidth', 2);
        title(sprintf('%s - d2', areas{a})); xlabel('Time (s)'); ylabel('d2'); grid on;
        % mrBr (bottom)
        subplot(2, length(areasToTest), length(areasToTest) + idx);
        hold on;
        plot(startSRea{a}, mrBrRea{a}, '-', 'Color', [0 0 0], 'LineWidth', 2);
        yline(1, 'k:', 'LineWidth', 1.5);
        title(sprintf('%s - mrBr', areas{a})); xlabel('Time (s)'); ylabel('mrBr'); grid on;
    end
    sgtitle(sprintf('Reach-only d2 (top) and mrBr (bottom) - win=%gs', slidingWindowSize));
    exportgraphics(gcf, fullfile(saveDir, sprintf('criticality_reach_ar_win%d.png', slidingWindowSize)), 'Resolution', 300);
end


