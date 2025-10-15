%%
% Criticality Reach-only Avalanche Script (dcc + kappa)
% Analyzes reach data only; saves results to a folder named after the data file

opts = neuro_behavior_options;
opts.minActTime = .16; opts.collectStart = 0; opts.minFiringRate = .05; opts.frameSize = .001;
paths = get_paths;

% Flags
loadExistingResults = false; makePlots = true;

% Window/step (seconds)
slidingWindowSize = 120; avStepSize = 20;

% Reach data file and save directory
reachDataFile = fullfile(paths.dropPath, 'reach_task/data/Copy_of_Y4_100623_Spiketimes_idchan_BEH.mat');
reachDataFile = fullfile(paths.dropPath, 'reach_task/data/makeSpikes.mat');
[~, dataBaseName, ~] = fileparts(reachDataFile);
saveDir = fullfile(paths.dropPath, 'reach_task/results', dataBaseName);
if ~exist(saveDir, 'dir'); mkdir(saveDir); end

resultsPath = fullfile(saveDir, sprintf('criticality_reach_av_win%d_step%d.mat', slidingWindowSize, avStepSize));

% Load reach data
dataR = load(reachDataFile);
[dataMatR, idLabels, areaLabels] = neural_matrix_mark_data(dataR, opts);
areas = {'M23','M56','DS','VS'};
idM23R = find(strcmp(areaLabels, 'M23')); idM56R = find(strcmp(areaLabels, 'M56'));
idDSR = find(strcmp(areaLabels, 'DS')); idVSR = find(strcmp(areaLabels, 'VS'));
idListRea = {idM23R, idM56R, idDSR, idVSR}; areasToTest = 1:4;

% Params
pcaFlag = 0; pcaFirstFlag = 1; nDim = 4; thresholdFlag = 1; thresholdPct = 0.75;
candidateFrameSizes = [.02 .03 .04 0.05, .075, 0.1 .15];
candidateWindowSizes = [30, 45, 60, 90, 120];

% Find optimal bin per area using spikes
reconstructedDataMatRea = cell(1, length(areas));
for a = areasToTest
    aID = idListRea{a}; reconstructedDataMatRea{a} = dataMatR(:, aID);
end
optimalBinSizeRea = zeros(1, length(areas));
for a = areasToTest
    [optimalBinSizeRea(a), ~] = find_optimal_bin_and_window(reconstructedDataMatRea{a}, candidateFrameSizes, candidateWindowSizes, 3, 50, 1000);
end

% Analyze reach with avalanche methods
dccRea = cell(1, length(areas)); kappaRea = cell(1, length(areas)); decadesRea = cell(1, length(areas)); startSRea_dcc = cell(1, length(areas));
for a = areasToTest
    aID = idListRea{a};
    aDataMatRea_dcc = neural_matrix_ms_to_frames(dataMatR(:, aID), optimalBinSizeRea(a));
    numTimePoints_dcc = size(aDataMatRea_dcc, 1);
    stepSamples_dcc = round(avStepSize / optimalBinSizeRea(a));
    winSamples_dcc = round(slidingWindowSize / optimalBinSizeRea(a));
    numWindows_dcc = floor((numTimePoints_dcc - winSamples_dcc) / stepSamples_dcc) + 1;

    if pcaFlag
        [coeff, score, ~, ~, explained, mu] = pca(aDataMatRea_dcc);
        forDim = find(cumsum(explained) > 30, 1); forDim = max(3, min(6, forDim));
        nDim = 1:forDim; aDataMatRea_dcc = score(:,nDim) * coeff(:,nDim)' + mu;
    end

    if thresholdFlag
        aDataMatRea_dcc = round(sum(aDataMatRea_dcc, 2));
        threshSpikes = thresholdPct * median(aDataMatRea_dcc);
        aDataMatRea_dcc(aDataMatRea_dcc < threshSpikes) = 0;
    else
        aDataMatRea_dcc = round(sum(aDataMatRea_dcc, 2));
    end

    dccRea{a} = nan(1, numWindows_dcc); kappaRea{a} = nan(1, numWindows_dcc); decadesRea{a} = nan(1, numWindows_dcc); startSRea_dcc{a} = nan(1, numWindows_dcc);
    for w = 1:numWindows_dcc
        startIdx = (w - 1) * stepSamples_dcc + 1; endIdx = startIdx + winSamples_dcc - 1;
        startSRea_dcc{a}(w) = (startIdx + round(winSamples_dcc/2)-1) * optimalBinSizeRea(a);
        wPopActivity = aDataMatRea_dcc(startIdx:endIdx);
        zeroBins = find(wPopActivity == 0);
        if length(zeroBins) > 1 && any(diff(zeroBins)>1)
            asdfMat = rastertoasdf2(wPopActivity', optimalBinSizeRea(a)*1000, 'CBModel', 'Spikes', 'DS');
            Av = avprops(asdfMat, 'ratio', 'fingerprint');
            [tau, ~, tauC, ~, alpha, ~, paramSD, decades] = avalanche_log(Av, 0);
            dccRea{a}(w) = distance_to_criticality(tau, alpha, paramSD);
            kappaRea{a}(w) = compute_kappa(Av.size);
            decadesRea{a}(w) = decades;
        end
    end
end

% Save results
results = struct(); results.areas = areas; results.reach.dcc = dccRea; results.reach.kappa = kappaRea; results.reach.decades = decadesRea; results.reach.startS_dcc = startSRea_dcc;
results.reach.optimalBinSize = optimalBinSizeRea; results.reach.slidingWindowSize = slidingWindowSize; results.reach.avStepSize = avStepSize;
save(resultsPath, 'results'); fprintf('Saved reach-only dcc/kappa to %s\n', resultsPath);

% Plotting: 3 rows (dcc, kappa, decades) x num areas (columns)
if makePlots
    figure(901); clf; set(gcf, 'Position', [100 100 1600 800]);
    for idx = 1:length(areasToTest)
        a = areasToTest(idx);
        % dcc (top)
        subplot(3, length(areasToTest), idx);
        hold on;
        plot(startSRea_dcc{a}, dccRea{a}, '-', 'Color', [1 0 0], 'LineWidth', 2);
        title(sprintf('%s - dcc', areas{a})); xlabel('Time (s)'); ylabel('dcc'); grid on;
        % kappa (middle)
        subplot(3, length(areasToTest), length(areasToTest) + idx);
        hold on;
        plot(startSRea_dcc{a}, kappaRea{a}, '-', 'Color', [0 0.6 0], 'LineWidth', 2);
        title(sprintf('%s - kappa', areas{a})); xlabel('Time (s)'); ylabel('kappa'); grid on;
        % decades (bottom)
        subplot(3, length(areasToTest), 2*length(areasToTest) + idx);
        hold on;
        plot(startSRea_dcc{a}, decadesRea{a}, '-', 'Color', [0.6 0 0.6], 'LineWidth', 2);
        title(sprintf('%s - decades', areas{a})); xlabel('Time (s)'); ylabel('decades'); grid on;
    end
    sgtitle(sprintf('Reach-only dcc (top), kappa (mid), decades (bottom) - win=%gs, step=%gs', slidingWindowSize, avStepSize));
    exportgraphics(gcf, fullfile(saveDir, sprintf('criticality_reach_av_win%d_step%d.png', slidingWindowSize, avStepSize)), 'Resolution', 300);
end


