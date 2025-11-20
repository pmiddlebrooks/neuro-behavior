%%
% Criticality Sliding Window Analysis Script (d2 + mrBr)
% Unified script for analyzing both reach data and naturalistic data
% Analyzes data using sliding window approach; saves results to data-specific folders

paths = get_paths;

% =============================    Configuration    =============================
% Data type selection
dataType = 'naturalistic';  % 'reach' or 'naturalistic'

% Sliding window size (seconds)
slidingWindowSize = 20;

% Flags
loadExistingResults = false;
makePlots = true;
plotCorrelations = false;

% Analysis flags
analyzeD2 = true;      % compute d2
analyzeMrBr = false;   % compute mrBr

% NEW: Modulation analysis flags
analyzeModulation = false;  % Set to true to split into modulated/unmodulated
modulationThreshold = 2;   % Standard deviations for modulation detection (legacy fallback)
modulationBinSize = nan;   % Bin size for modulation analysis
modulationBaseWindow = [-3, -2];    % Baseline time range [min max] in seconds relative to reach onset
modulationEventWindow = [-0.2, 0.6]; % Event time range [min max] in seconds relative to reach onset
modulationPlotFlag = false; % Set to true to generate modulation analysis plots

% Analysis parameters
minSegmentLength = 50;
minSpikesPerBin = 3;
maxSpikesPerBin = 50;
minBinsPerWindow = 1000;

% Areas to analyze
areasToTest = 1:4;
areasToPlot = areasToTest;

% PCA options
pcaFlag = 1;           % Set to 1 to use PCA
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
d2StepSize = repmat(.2,1,4);

% =============================    Data Loading    =============================
fprintf('\n=== Loading %s data ===\n', dataType);

% Create filename suffix based on PCA flag
if pcaFlag
    filenameSuffix = '_pca';
else
    filenameSuffix = '';
end

if strcmp(dataType, 'reach')
    % Load reach data
    reachDataFile = fullfile(paths.reachDataPath, 'Y4_06-Oct-2023 14_14_53_NeuroBeh.mat');
    % reachDataFile = fullfile(paths.reachDataPath, 'makeSpikes.mat');
    
    [~, dataBaseName, ~] = fileparts(reachDataFile);
    saveDir = fullfile(paths.dropPath, 'reach_task/results', dataBaseName);
    if ~exist(saveDir, 'dir'); mkdir(saveDir); end
    
    resultsPath = fullfile(saveDir, sprintf('criticality_sliding_window_ar%s_win%d.mat', filenameSuffix, slidingWindowSize));
    
    dataR = load(reachDataFile);
    
    opts = neuro_behavior_options;
    opts.frameSize = .001;
    opts.firingRateCheckTime = 5 * 60;
    opts.collectStart = 0;
    opts.collectEnd = round(min(dataR.R(end,1) + 5000, max(dataR.CSV(:,1)*1000)) / 1000);
    opts.minFiringRate = .1;
    opts.maxFiringRate = 70;
    
    [dataMat, idLabels, areaLabels] = neural_matrix_mark_data(dataR, opts);
    areas = {'M23', 'M56', 'DS', 'VS'};
    idM23 = find(strcmp(areaLabels, 'M23'));
    idM56 = find(strcmp(areaLabels, 'M56'));
    idDS = find(strcmp(areaLabels, 'DS'));
    idVS = find(strcmp(areaLabels, 'VS'));
    idMatIdx = {idM23, idM56, idDS, idVS};
    idLabel = {idLabels(idM23), idLabels(idM56), idLabels(idDS), idLabels(idVS)};

    % NEW: Load spike data for modulation analysis
    if analyzeModulation
        fprintf('\n=== Loading spike data for modulation analysis ===\n');
        % spikeData = spike_times_per_area_reach(opts);
        spikeData = dataR.CSV(:,1:2);
        fprintf('Loaded spike data: %d spikes from %d neurons\n', size(spikeData, 1), length(unique(spikeData(:,2))));
    end
    
elseif strcmp(dataType, 'naturalistic')
    % Load naturalistic data
    getDataType = 'spikes';
    opts.firingRateCheckTime = 5 * 60;
    opts.collectStart = 0 * 60; % seconds
    opts.collectEnd = 45 * 60; % seconds
    opts.minFiringRate = .05;
    get_standard_data
    
    areas = {'M23', 'M56', 'DS', 'VS'};
    idMatIdx = {idM23, idM56, idDS, idVS};
    idLabel = {idLabels(idM23), idLabels(idM56), idLabels(idDS), idLabels(idVS)};
    
    % Create save directory for naturalistic data
    saveDir = fullfile(paths.dropPath, 'criticality/results');
    if ~exist(saveDir, 'dir'); mkdir(saveDir); end
    resultsPath = fullfile(saveDir, sprintf('criticality_sliding_window_ar%s_win%d.mat', filenameSuffix, slidingWindowSize));
    
    % NEW: Load spike data for modulation analysis
    if analyzeModulation
        fprintf('\n=== Loading spike data for modulation analysis ===\n');
        % spikeData = spike_times_per_area(opts);
        spikeData = [spikeTimes, spikeClusters];
        fprintf('Loaded spike data: %d spikes from %d neurons\n', size(spikeData, 1), length(unique(spikeData(:,2))));
    end

else
    error('Invalid dataType. Must be ''reach'' or ''naturalistic''');
end
fprintf('%d M23\n%d M56\n%d DS\n%d VS\n', length(idM23), length(idM56), length(idDS), length(idVS))






% =============================    Analysis    =============================


% =============================    Modulation Analysis    =============================
if analyzeModulation
    modulationResults = perform_modulation_analysis(spikeData, areas, idLabel, areasToTest, dataType, dataR, opts, modulationBinSize, modulationBaseWindow, modulationEventWindow, modulationThreshold, modulationPlotFlag);
else
    modulationResults = cell(1, length(areas));
end

% =============================    Analysis    =============================
fprintf('\n=== %s Data Analysis ===\n', dataType);

% Adjust areasToTest based on which areas have data
% areasToTest = areasToTest(~cellfun(@isempty, idMatIdx));

% Step 1-2: Apply PCA to original data if requested
fprintf('\n--- Step 1-2: PCA on original data if requested ---\n');
reconstructedDataMat = cell(1, length(areas));
for a = areasToTest
    aID = idMatIdx{a};
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

% FIRST: Find optimal parameters for ALL neurons (original analysis)
for a = areasToTest
    thisDataMat = reconstructedDataMat{a};
    [optimalBinSize(a), optimalWindowSize(a)] = ...
        find_optimal_bin_and_window(thisDataMat, candidateFrameSizes, candidateWindowSizes, minSpikesPerBin, maxSpikesPerBin, minBinsPerWindow);
    fprintf('Area %s (all neurons): optimal bin size = %.3f s, optimal window size = %.1f s\n', areas{a}, optimalBinSize(a), optimalWindowSize(a));
end

% SECOND: If modulation analysis is enabled, find optimal parameters for modulated and unmodulated separately
if analyzeModulation
    % Initialize optimal parameters for modulated and unmodulated
    optimalBinSizeModulated = zeros(1, length(areas));
    optimalBinSizeUnmodulated = zeros(1, length(areas));
    optimalWindowSizeModulated = zeros(1, length(areas));
    optimalWindowSizeUnmodulated = zeros(1, length(areas));

    for a = areasToTest
        thisDataMat = reconstructedDataMat{a};

        if ~isempty(modulationResults{a})
            % Get modulated and unmodulated neuron indices
            modulatedNeurons = modulationResults{a}.neuronIds(modulationResults{a}.isModulated);
            unmodulatedNeurons = modulationResults{a}.neuronIds(~modulationResults{a}.isModulated);

            % Find indices in the data matrix
            modulatedIndices = ismember(idLabel{a}, modulatedNeurons);
            unmodulatedIndices = ismember(idLabel{a}, unmodulatedNeurons);

            % Analyze modulated neurons
            if sum(modulatedIndices) >= 5 % Need minimum neurons
                modulatedDataMat = thisDataMat(:, modulatedIndices);
                [optimalBinSizeModulated(a), optimalWindowSizeModulated(a)] = ...
                    find_optimal_bin_and_window(modulatedDataMat, candidateFrameSizes, candidateWindowSizes, minSpikesPerBin, maxSpikesPerBin, minBinsPerWindow);
            else
                optimalBinSizeModulated(a) = NaN;
                optimalWindowSizeModulated(a) = NaN;
            end

            % Analyze unmodulated neurons
            if sum(unmodulatedIndices) >= 5 % Need minimum neurons
                unmodulatedDataMat = thisDataMat(:, unmodulatedIndices);
                [optimalBinSizeUnmodulated(a), optimalWindowSizeUnmodulated(a)] = ...
                    find_optimal_bin_and_window(unmodulatedDataMat, candidateFrameSizes, candidateWindowSizes, minSpikesPerBin, maxSpikesPerBin, minBinsPerWindow);
            else
                optimalBinSizeUnmodulated(a) = NaN;
                optimalWindowSizeUnmodulated(a) = NaN;
            end

            fprintf('Area %s (modulated): bin=%.3f, (unmodulated): bin=%.3f\n', areas{a}, ...
                optimalBinSizeModulated(a), optimalBinSizeUnmodulated(a));
        else
            fprintf('Area %s: No modulation data available\n', areas{a});
        end
    end
end

% Use optimal bin sizes for each area
d2StepSizeData = d2StepSize; % optimalBinSize;
warning('Change step size back to optimalBinSize')
d2WindowSizeData = windowSizes;
validMask = isfinite(optimalBinSize(areasToTest)) & (optimalBinSize(areasToTest) > 0);
areasToTest = areasToTest(validMask);

% Initialize results
[popActivity, mrBr, d2, startS, popActivityWindows, popActivityFull] = ...
    deal(cell(1, length(areas)));

% NEW: Initialize results for modulated/unmodulated populations
if analyzeModulation
    [popActivityModulated, mrBrModulated, d2Modulated, startSModulated, popActivityWindowsModulated, popActivityFullModulated] = ...
        deal(cell(1, length(areas)));
    [popActivityUnmodulated, mrBrUnmodulated, d2Unmodulated, startSUnmodulated, popActivityWindowsUnmodulated, popActivityFullUnmodulated] = ...
        deal(cell(1, length(areas)));
end

for a = areasToTest
    fprintf('\nProcessing area %s (%s)...\n', areas{a}, dataType); 
    tic;
    aID = idMatIdx{a};
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

    % NEW: Analyze modulated and unmodulated populations separately
    if analyzeModulation && ~isempty(modulationResults{a})
        % Get modulated and unmodulated neuron indices
        modulatedNeurons = modulationResults{a}.neuronIds(modulationResults{a}.isModulated);
        unmodulatedNeurons = modulationResults{a}.neuronIds(~modulationResults{a}.isModulated);

        % Find indices in the data matrix
        modulatedIndices = ismember(idLabel{a}, modulatedNeurons);
        unmodulatedIndices = ismember(idLabel{a}, unmodulatedNeurons);

        % Analyze modulated neurons
        if sum(modulatedIndices) >= 5
            modulatedDataMat = aDataMat(:, modulatedIndices);
            popActivityModulated{a} = round(sum(modulatedDataMat, 2));
            [startSModulated{a}, mrBrModulated{a}, d2Modulated{a}, popActivityWindowsModulated{a}, popActivityFullModulated{a}] = ...
                deal(nan(1, numWindows));

            for w = 1:numWindows
                startIdx = (w - 1) * stepSamples + 1;
                endIdx = startIdx + winSamples - 1;
                startSModulated{a}(w) = (startIdx + round(winSamples/2)-1) * optimalBinSize(a);
                wPopActivity = popActivityModulated{a}(startIdx:endIdx);
                popActivityWindowsModulated{a}(w) = mean(wPopActivity);
                popActivityFullModulated{a}(w) = popActivityModulated{a}((startIdx + round(winSamples/2)-1));
                if analyzeMrBr
                    result = branching_ratio_mr_estimation(wPopActivity);
                    mrBrModulated{a}(w) = result.branching_ratio;
                else
                    mrBrModulated{a}(w) = nan;
                end
                if analyzeD2
                    [varphi, ~] = myYuleWalker3(wPopActivity, pOrder);
                    d2Modulated{a}(w) = getFixedPointDistance2(pOrder, critType, varphi);
                else
                    d2Modulated{a}(w) = nan;
                end
            end
        end

        % Analyze unmodulated neurons
        if sum(unmodulatedIndices) >= 5
            unmodulatedDataMat = aDataMat(:, unmodulatedIndices);
            popActivityUnmodulated{a} = round(sum(unmodulatedDataMat, 2));
            [startSUnmodulated{a}, mrBrUnmodulated{a}, d2Unmodulated{a}, popActivityWindowsUnmodulated{a}, popActivityFullUnmodulated{a}] = ...
                deal(nan(1, numWindows));

            for w = 1:numWindows
                startIdx = (w - 1) * stepSamples + 1;
                endIdx = startIdx + winSamples - 1;
                startSUnmodulated{a}(w) = (startIdx + round(winSamples/2)-1) * optimalBinSize(a);
                wPopActivity = popActivityUnmodulated{a}(startIdx:endIdx);
                popActivityWindowsUnmodulated{a}(w) = mean(wPopActivity);
                popActivityFullUnmodulated{a}(w) = popActivityUnmodulated{a}((startIdx + round(winSamples/2)-1));
                if analyzeMrBr
                    result = branching_ratio_mr_estimation(wPopActivity);
                    mrBrUnmodulated{a}(w) = result.branching_ratio;
                else
                    mrBrUnmodulated{a}(w) = nan;
                end
                if analyzeD2
                    [varphi, ~] = myYuleWalker3(wPopActivity, pOrder);
                    d2Unmodulated{a}(w) = getFixedPointDistance2(pOrder, critType, varphi);
                else
                    d2Unmodulated{a}(w) = nan;
                end
            end
        end
    end
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

% NEW: Compute correlations for modulated vs unmodulated populations
if analyzeModulation
    fprintf('\n=== Modulated vs Unmodulated Population Correlations ===\n');
    for a = areasToTest
        if ~isempty(modulationResults{a})
            fprintf('\n--- Area %s ---\n', areas{a});

            % Modulated population correlations
            if ~isempty(popActivityWindowsModulated{a}) && ~isempty(d2Modulated{a})
                validIdx = ~isnan(popActivityWindowsModulated{a}) & ~isnan(d2Modulated{a});
                if sum(validIdx) > 10
                    popAct = popActivityModulated{a}(validIdx);
                    d2Vals = d2Modulated{a}(validIdx);
                    [rPopD2, pPopD2] = corrcoef(popAct, d2Vals);
                    fprintf('Modulated: PopActivity vs d2: r=%.3f, p=%.3f (n=%d)\n', rPopD2(1,2), pPopD2(1,2), sum(validIdx));
                end
            end

            % Unmodulated population correlations
            if ~isempty(popActivityWindowsUnmodulated{a}) && ~isempty(d2Unmodulated{a})
                validIdx = ~isnan(popActivityWindowsUnmodulated{a}) & ~isnan(d2Unmodulated{a});
                if sum(validIdx) > 10
                    popAct = popActivityUnmodulated{a}(validIdx);
                    d2Vals = d2Unmodulated{a}(validIdx);
                    [rPopD2, pPopD2] = corrcoef(popAct, d2Vals);
                    fprintf('Unmodulated: PopActivity vs d2: r=%.3f, p=%.3f (n=%d)\n', rPopD2(1,2), pPopD2(1,2), sum(validIdx));
                end
            end
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
% results.popActivityWindows = popActivityWindows;
results.popActivity = popActivity;  % Store full population activity for each area
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

% NEW: Save modulation analysis results
if analyzeModulation
    results.analyzeModulation = true;
    results.modulationResults = modulationResults;
    results.modulationThreshold = modulationThreshold;
    results.modulationBinSize = modulationBinSize;
    results.modulationBaseWindow = modulationBaseWindow;
    results.modulationEventWindow = modulationEventWindow;
    results.modulationPlotFlag = modulationPlotFlag;

    % Save modulated/unmodulated results
    results.popActivityModulated = popActivityWindowsModulated;
    results.mrBrModulated = mrBrModulated;
    results.d2Modulated = d2Modulated;
    results.startSModulated = startSModulated;
    results.popActivityFullModulated = popActivityFullModulated;

    results.popActivityUnmodulated = popActivityWindowsUnmodulated;
    results.mrBrUnmodulated = mrBrUnmodulated;
    results.d2Unmodulated = d2Unmodulated;
    results.startSUnmodulated = startSUnmodulated;
    results.popActivityFullUnmodulated = popActivityFullUnmodulated;

    % Save optimal parameters for each population
    results.optimalBinSizeModulated = optimalBinSizeModulated;
    results.optimalBinSizeUnmodulated = optimalBinSizeUnmodulated;
    results.optimalWindowSizeModulated = optimalWindowSizeModulated;
    results.optimalWindowSizeUnmodulated = optimalWindowSizeUnmodulated;
else
    results.analyzeModulation = false;
    % Initialize empty arrays for modulated/unmodulated when not analyzed
    results.optimalBinSizeModulated = nan(1, length(areas));
    results.optimalBinSizeUnmodulated = nan(1, length(areas));
    results.optimalWindowSizeModulated = nan(1, length(areas));
    results.optimalWindowSizeUnmodulated = nan(1, length(areas));
end

save(resultsPath, 'results'); 
fprintf('Saved %s d2/mrBr to %s\n', dataType, resultsPath);

%% =============================    Plotting    =============================
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
    
    % Create all plots using functions
    if strcmp(dataType, 'reach')
        reachOnsetTimes = dataR.R(:,1) / 1000; % Convert from ms to seconds
    else
        reachOnsetTimes = []; % No reach onsets for naturalistic data
    end
    plot_criticality_timeseries(startS, d2, mrBr, popActivityWindows, popActivity, areasToTest, areas, dataType, slidingWindowSize, saveDir, targetPos, analyzeD2, analyzeMrBr, optimalBinSize, reachOnsetTimes, filenameSuffix);

    if plotCorrelations
        plot_criticality_correlations(popActivityWindows, popActivityFull, d2, areasToPlot, areas, dataType, slidingWindowSize, saveDir, targetPos, filenameSuffix);
    end
    if analyzeModulation
        plot_modulated_vs_unmodulated(d2Modulated, d2Unmodulated, startSModulated, startSUnmodulated, popActivityWindowsModulated, popActivityWindowsUnmodulated, areasToTest, areasToPlot, areas, dataType, slidingWindowSize, saveDir, targetPos, analyzeD2, reachOnsetTimes, filenameSuffix);
    end
end

fprintf('\n=== %s Analysis Complete ===\n', dataType);

%% =============================    Function Definitions    =============================

function modulationResults = perform_modulation_analysis(spikeData, areas, idLabel, areasToTest, dataType, dataR, opts, modulationBinSize, modulationBaseWindow, modulationEventWindow, modulationThreshold, modulationPlotFlag)
% Perform modulation analysis for all areas
fprintf('\n=== Performing modulation analysis ===\n');

modulationResults = cell(1, length(areas));

% Define alignment times for modulation analysis
if strcmp(dataType, 'reach')
    % Use reach start times as alignment points
    alignTimes = dataR.R(:,1) / 1000;
else
    % For naturalistic data, use regular intervals
    disp('Code for what behaviors you want to align to for naturalistic')
    return
    totalTime = opts.collectEnd;
    alignTimes = (modulationEventWindow/2):(modulationEventWindow*2):(totalTime - modulationEventWindow/2);
end

% Perform modulation analysis for each area
for a = areasToTest
    fprintf('\nAnalyzing modulation for area %s...\n', areas{a});

    % Get neuron IDs for this area
    areaNeuronIds = idLabel{a};

    % Filter spike data for this area
    areaSpikeMask = ismember(spikeData(:,2), areaNeuronIds);
    areaSpikeData = spikeData(areaSpikeMask, :);

    if size(areaSpikeData, 1) < 100 % Need minimum spikes for analysis
        fprintf('Insufficient spikes in area %s for modulation analysis\n', areas{a});
        modulationResults{a} = [];
        continue;
    end

    % Set up modulation analysis options
    modOpts = struct();
    modOpts.binSize = modulationBinSize;
    modOpts.baseWindow = modulationBaseWindow;
    modOpts.eventWindow = modulationEventWindow;
    modOpts.alignTimes = alignTimes;
    modOpts.threshold = modulationThreshold;
    modOpts.plotFlag = modulationPlotFlag;

    % Run modulation analysis
    % try
    modulationResults{a} = spike_modulation(areaSpikeData, modOpts);
    fprintf('Area %s: %d/%d neurons modulated (%.1f%%)\n', areas{a}, ...
        sum(modulationResults{a}.isModulated), length(modulationResults{a}.neuronIds), ...
        100*sum(modulationResults{a}.isModulated)/length(modulationResults{a}.neuronIds));
    % catch ME
    %     fprintf('Error in modulation analysis for area %s: %s\n', areas{a}, ME.message);
    %     modulationResults{a} = [];
    % end
end
end

function plot_criticality_timeseries(startS, d2, mrBr, popActivityWindows, popActivity, areasToTest, areas, dataType, slidingWindowSize, saveDir, targetPos, analyzeD2, analyzeMrBr, optimalBinSize, reachOnsetTimes, filenameSuffix)
% Plot time series of criticality measures
    figure(900); clf;
    set(gcf, 'Units', 'pixels');
    set(gcf, 'Position', targetPos);
    numRows = length(areasToTest);
ha = tight_subplot(numRows, 1, [0.05 0.04], [0.03 0.08], [0.08 0.04]);

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
        
    % Add normalized population activity
        if ~isempty(popActNorm) && any(~isnan(popActNorm))
            yyaxis left;
            plot(startS{a}, popActNorm, '-', 'Color', [1 0 0], 'LineWidth', 1, 'LineStyle', ':'); 
        end
        
    % Add full population activity (normalized) for comparison
        if ~isempty(popActivity{a}) && any(~isnan(popActivity{a}))
            timeIndices = (1:length(popActivity{a})) * optimalBinSize(a);
            popActFullNorm = (popActivity{a} - min(popActivity{a}(~isnan(popActivity{a})))) / ...
                           (max(popActivity{a}(~isnan(popActivity{a}))) - min(popActivity{a}(~isnan(popActivity{a}))));
            popActFullNorm(isnan(popActivity{a})) = nan;
            
            yyaxis left;
            plot(timeIndices, popActFullNorm, '-', 'Color', [0.8 0.4 0.4], 'LineWidth', 0.5, 'LineStyle', '-.'); 
        end

    % Add vertical lines at reach onsets (only for reach data)
    if ~isempty(reachOnsetTimes) && strcmp(dataType, 'reach')
        yyaxis left;
        % Filter reach onsets to only show those within the current plot's time range
        if ~isempty(startS{a})
            plotTimeRange = [startS{a}(1), startS{a}(end)];
            reachOnsetsInRange = reachOnsetTimes(reachOnsetTimes >= plotTimeRange(1) & reachOnsetTimes <= plotTimeRange(2));

            if ~isempty(reachOnsetsInRange)
                for i = 1:length(reachOnsetsInRange)
                    xline(reachOnsetsInRange(i), 'Color', [0.5 0.5 0.5], 'LineWidth', 0.8, 'LineStyle', '--', 'Alpha', 0.7);
                end
            end
        end
    end
        
        if ~isempty(startS{a})
        xlim([startS{a}(1) startS{a}(end)])
        end
        title(sprintf('%s - d2 (blue), mrBr (black), PopActWin (red dotted), PopActFull (brown dash-dot)', areas{a})); 
        xlabel('Time (s)'); grid on; set(gca, 'XTickLabelMode', 'auto'); set(gca, 'YTickLabelMode', 'auto');
    end

if ~isempty(reachOnsetTimes) && strcmp(dataType, 'reach')
    sgtitle(sprintf('%s d2 (blue, left) and mrBr (black, right) with reach onsets (gray dashed) - win=%gs', dataType, slidingWindowSize));
else
    sgtitle(sprintf('%s d2 (blue, left) and mrBr (black, right) - win=%gs', dataType, slidingWindowSize));
end
    exportgraphics(gcf, fullfile(saveDir, sprintf('criticality_%s_ar%s_win%d.png', dataType, filenameSuffix, slidingWindowSize)), 'Resolution', 300);
end
    
function plot_criticality_correlations(popActivityWindows, popActivityFull, d2, areasToPlot, areas, dataType, slidingWindowSize, saveDir, targetPos, filenameSuffix)
% Plot correlation scatter plots
    figure(901); clf;
    set(gcf, 'Units', 'pixels');
    set(gcf, 'Position', targetPos);
    
    numAreas = length(areasToPlot);
    numRows = 2;
    numCols = 4;
ha = tight_subplot(numRows, numCols, [0.05 0.04], [0.03 0.08], [0.08 0.04]);
    
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
                
            % Add regression line
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
                
            % Add regression line
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
    exportgraphics(gcf, fullfile(saveDir, sprintf('criticality_%s_correlations%s_win%d.png', dataType, filenameSuffix, slidingWindowSize)), 'Resolution', 300);
    fprintf('Saved %s correlation scatter plots to: %s\n', dataType, fullfile(saveDir, sprintf('criticality_%s_correlations%s_win%d.png', dataType, filenameSuffix, slidingWindowSize)));
end

function plot_modulated_vs_unmodulated(d2Modulated, d2Unmodulated, startSModulated, startSUnmodulated, popActivityWindowsModulated, popActivityWindowsUnmodulated, areasToTest, areasToPlot, areas, dataType, slidingWindowSize, saveDir, targetPos, analyzeD2, reachOnsetTimes, filenameSuffix)
% Plot modulated vs unmodulated comparisons
plotCorrelations = false;

% Time series comparison
figure(902); clf;
set(gcf, 'Units', 'pixels');
set(gcf, 'Position', targetPos);

numRows = length(areasToTest);
ha = tight_subplot(numRows, 1, [0.08 0.04], [0.03 0.08], [0.08 0.04]);

for idx = 1:length(areasToTest)
    a = areasToTest(idx);
    axes(ha(idx)); hold on;

    if analyzeD2
        % Plot modulated population
        if ~isempty(d2Modulated{a}) && any(~isnan(d2Modulated{a}))
            plot(startSModulated{a}, d2Modulated{a}, '-', 'Color', [1 0 0], 'LineWidth', 2, 'DisplayName', 'Modulated');
        end

        % Plot unmodulated population
        if ~isempty(d2Unmodulated{a}) && any(~isnan(d2Unmodulated{a}))
            plot(startSUnmodulated{a}, d2Unmodulated{a}, '-', 'Color', [0 0 1], 'LineWidth', 2, 'DisplayName', 'Unmodulated');
        end

        ylabel('d2'); grid on;
        legend({'Modulated', 'Unmodulated'}, 'Location', 'best', 'AutoUpdate', 'off');
        % Add vertical lines at reach onsets (only for reach data)
        if ~isempty(reachOnsetTimes) && strcmp(dataType, 'reach')
                    for i = 1:length(reachOnsetTimes)
                        xline(reachOnsetTimes(i), 'Color', [0.5 0.5 0.5], 'LineWidth', 0.8, 'LineStyle', '--', 'Alpha', 0.7);
                    end
        end

    end

    title(sprintf('%s - Modulated (red) vs Unmodulated (blue) d2', areas{a}));
    xlabel('Time (s)'); grid on; set(gca, 'XTickLabelMode', 'auto'); set(gca, 'YTickLabelMode', 'auto');
end

sgtitle(sprintf('%s Modulated neurons d2 win=%gs', dataType, slidingWindowSize));
exportgraphics(gcf, fullfile(saveDir, sprintf('criticality_%s_modulated_vs_unmodulated%s_win%d.png', dataType, filenameSuffix, slidingWindowSize)), 'Resolution', 300);

if plotCorrelations

% Scatter plots comparing modulated vs unmodulated
figure(903); clf;
set(gcf, 'Units', 'pixels');
set(gcf, 'Position', targetPos);

numAreas = length(areasToPlot);
numRows = 2;
numCols = numAreas;
ha = tight_subplot(numRows, numCols, [0.08 0.04], [0.08 0.1], [0.06 0.04]);

for idx = 1:numAreas
    a = areasToPlot(idx);

    % Row 1: Modulated population scatter
    axes(ha(idx));
    if ~isempty(popActivityWindowsModulated{a}) && ~isempty(d2Modulated{a})
        validIdx = ~isnan(popActivityWindowsModulated{a}) & ~isnan(d2Modulated{a});
        if sum(validIdx) > 5
            xData = popActivityWindowsModulated{a}(validIdx);
            yData = d2Modulated{a}(validIdx);
            scatter(xData, yData, 20, 'filled', 'MarkerFaceColor', [1 0 0], 'MarkerFaceAlpha', 0.6);
            [r, p] = corrcoef(xData, yData);
            title(sprintf('%s Modulated\nr=%.3f, p=%.3f', areas{a}, r(1,2), p(1,2)));

            % Add regression line
            hold on;
            p_fit = polyfit(xData, yData, 1);
            x_fit = linspace(min(xData), max(xData), 100);
            y_fit = polyval(p_fit, x_fit);
            plot(x_fit, y_fit, 'r-', 'LineWidth', 2);
        else
            title(sprintf('%s Modulated\nInsufficient data', areas{a}));
        end
    end
    xlabel('PopActivity'); ylabel('d2');
    grid on;

    % Row 2: Unmodulated population scatter
    axes(ha(numAreas + idx));
    if ~isempty(popActivityWindowsUnmodulated{a}) && ~isempty(d2Unmodulated{a})
        validIdx = ~isnan(popActivityWindowsUnmodulated{a}) & ~isnan(d2Unmodulated{a});
        if sum(validIdx) > 5
            xData = popActivityWindowsUnmodulated{a}(validIdx);
            yData = d2Unmodulated{a}(validIdx);
            scatter(xData, yData, 20, 'filled', 'MarkerFaceColor', [0 0 1], 'MarkerFaceAlpha', 0.6);
            [r, p] = corrcoef(xData, yData);
            title(sprintf('%s Unmodulated\nr=%.3f, p=%.3f', areas{a}, r(1,2), p(1,2)));

            % Add regression line
            hold on;
            p_fit = polyfit(xData, yData, 1);
            x_fit = linspace(min(xData), max(xData), 100);
            y_fit = polyval(p_fit, x_fit);
            plot(x_fit, y_fit, 'b-', 'LineWidth', 2);
        else
            title(sprintf('%s Unmodulated\nInsufficient data', areas{a}));
        end
    end
    xlabel('PopActivity'); ylabel('d2');
    grid on;
end

sgtitle(sprintf('%s Modulated vs Unmodulated Population Activity vs Criticality', dataType));
exportgraphics(gcf, fullfile(saveDir, sprintf('criticality_%s_modulated_unmodulated_scatter%s_win%d.png', dataType, filenameSuffix, slidingWindowSize)), 'Resolution', 300);
end
end
