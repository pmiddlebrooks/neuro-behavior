%%
% Criticality Sliding Window Analysis Script (d2 + mrBr)
% Unified script for analyzing both reach data and spontaneous data
% Analyzes data using sliding window approach; saves results to data-specific folders
%
% Note: Data should be loaded by criticality_sliding_data_prep.m before running this script
% Expected workspace variables: dataType, dataMat, areas, idMatIdx, idLabel, opts, saveDir, 
%                                dataR (for reach), startBlock2 (for reach), reachStart (for reach), 
%                                reachClass (for reach), sessionName (for reach), spikeData (if analyzeModulation)

% =============================    Configuration    =============================


% Flags
loadExistingResults = false;
makePlots = true;
plotCorrelations = false;

% Analysis flags
analyzeD2 = true;      % compute d2
analyzeMrBr = false;   % compute mrBr

% NEW: Permutation testing flags
enablePermutations = true;  % Set to true to perform circular permutation testing
nShuffles = 3;  % Number of circular permutations to perform

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

% Bin/window size selection mode
useOptimalBinWindowFunction = true;  % Set to true to use find_optimal_bin_and_window.m
                                     % Set to false to manually define optimalBinSize and optimalWindowSize below
% Manual bin/window size definitions (used only if useOptimalBinWindowFunction = false)
% Define as arrays with one value per area: [M23, M56, DS, VS]
optimalBinSize = repmat(binSize, 1, length(areas));      % Bin sizes in seconds (e.g., 10 ms = 0.01 s)
optimalWindowSize = repmat(slidingWindowSize, 1, length(areas));          % Window sizes in seconds

% Areas to analyze
areasToPlot = areasToTest;

% PCA options
pcaFlag = 0;           % Set to 1 to use PCA
pcaFirstFlag = 1;      % Use first nDim dimensions if 1, last nDim if 0
nDim = 4;              % Number of PCA dimensions to use

% Threshold options
thresholdFlag = 1;     % Set to 1 to use threshold method
thresholdPct = 1;   % Threshold as percentage of median

% Optimal bin/window size search parameters
pOrder = 10;
critType = 2;

% =============================    Data Validation    =============================
% Verify that required variables are in workspace (loaded by criticality_sliding_data_prep.m)
requiredVars = {'dataType', 'dataMat', 'areas', 'idMatIdx', 'idLabel', 'opts', 'saveDir'};
for i = 1:length(requiredVars)
    if ~exist(requiredVars{i}, 'var')
        error('Required variable %s not found in workspace. Please run criticality_sliding_data_prep.m first.', requiredVars{i});
    end
end

% Create filename suffix based on PCA flag
if pcaFlag
    filenameSuffix = '_pca';
else
    filenameSuffix = '';
end

% Create results path
if strcmp(dataType, 'reach')
    if ~exist('sessionName', 'var') || isempty(sessionName)
        error('sessionName must be defined for reach data');
    end
    resultsPath = fullfile(saveDir, sprintf('criticality_sliding_window_ar%s_win%d_%s.mat', filenameSuffix, slidingWindowSize, sessionName));
else
    resultsPath = fullfile(saveDir, sprintf('criticality_sliding_window_ar%s_win%d.mat', filenameSuffix, slidingWindowSize));
end

% Load spike data for modulation analysis if needed (fallback if not loaded by prep script)
if analyzeModulation
    if isempty(spikeData) || ~exist('spikeData', 'var')
        fprintf('\n=== Loading spike data for modulation analysis ===\n');
        if strcmp(dataType, 'reach')
            if ~exist('dataR', 'var')
                error('dataR must be available for reach data modulation analysis');
            end
            spikeData = dataR.CSV(:,1:2);
        else
            % For spontaneous/schall, spikeData should have been loaded by prep script
            error('spikeData must be loaded by criticality_sliding_data_prep.m for modulation analysis');
        end
        fprintf('Loaded spike data: %d spikes from %d neurons\n', size(spikeData, 1), length(unique(spikeData(:,2))));
    end
end






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
if useOptimalBinWindowFunction
    % Calculate optimal parameters using function
    optimalBinSize = zeros(1, length(areas));
    optimalWindowSize = zeros(1, length(areas));
    
    % FIRST: Find optimal parameters for ALL neurons (original analysis)
    for a = areasToTest
        thisDataMat = reconstructedDataMat{a};
        thisFiringRate = sum(thisDataMat(:) / (size(thisDataMat, 1)/1000));
        [optimalBinSize(a), optimalWindowSize(a)] = ...
            find_optimal_bin_and_window(thisFiringRate, minSpikesPerBin, minBinsPerWindow);
        % optimalBinSize(a) = max(optimalBinSize(a), .02);
    end
d2StepSize = optimalBinSize * 2;
else
    % Use manually defined values
    fprintf('Using manually defined bin and window sizes:\n');
end
    for a = areasToTest
        fprintf('Area %s: bin size = %.3f s, window size = %.1f s, step size = %.3f\n', areas{a}, optimalBinSize(a), optimalWindowSize(a), d2StepSize(a));
    end

% SECOND: If modulation analysis is enabled, find optimal parameters for modulated and unmodulated separately
if analyzeModulation
    % Initialize optimal parameters for modulated and unmodulated
    optimalBinSizeModulated = zeros(1, length(areas));
    optimalBinSizeUnmodulated = zeros(1, length(areas));
    optimalWindowSizeModulated = zeros(1, length(areas));
    optimalWindowSizeUnmodulated = zeros(1, length(areas));

    if useOptimalBinWindowFunction
        % Calculate optimal parameters using function
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
                    mFiringRate = sum(modulatedDataMat(:) / (size(modulatedDataMat, 1)/1000));
                    [optimalBinSizeModulated(a), optimalWindowSizeModulated(a)] = ...
                        find_optimal_bin_and_window(mFiringRate, minSpikesPerBin, minBinsPerWindow);
                else
                    optimalBinSizeModulated(a) = NaN;
                    optimalWindowSizeModulated(a) = NaN;
                end

                % Analyze unmodulated neurons
                if sum(unmodulatedIndices) >= 5 % Need minimum neurons
                    unmodulatedDataMat = thisDataMat(:, unmodulatedIndices);
                    uFiringRate = sum(unmodulatedDataMat(:) / (size(unmodulatedDataMat, 1)/1000));
                    [optimalBinSizeUnmodulated(a), optimalWindowSizeUnmodulated(a)] = ...
                        find_optimal_bin_and_window(uFiringRate, minSpikesPerBin, minBinsPerWindow);
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
    else
        % Use manually defined values for modulated/unmodulated (same as all neurons)
        for a = areasToTest
            optimalBinSizeModulated(a) = optimalBinSize(a);
            optimalBinSizeUnmodulated(a) = optimalBinSize(a);
            optimalWindowSizeModulated(a) = optimalWindowSize(a);
            optimalWindowSizeUnmodulated(a) = optimalWindowSize(a);
            fprintf('Area %s (modulated/unmodulated): using manual bin=%.3f, window=%.1f\n', areas{a}, ...
                optimalBinSize(a), optimalWindowSize(a));
        end
    end
end

% Use optimal bin sizes for each area
d2StepSizeData = d2StepSize; % optimalBinSize;
warning('Change step size back to optimalBinSize')
d2WindowSizeData = optimalWindowSize;

% Initialize results
[popActivity, mrBr, d2, startS, popActivityWindows, popActivityFull] = ...
    deal(cell(1, length(areas)));

% NEW: Initialize permutation results
if enablePermutations
    d2Permuted = cell(1, length(areas));  % Store all permutation results [nWindows x nShuffles]
    mrBrPermuted = cell(1, length(areas)); % Store all permutation results [nWindows x nShuffles]
    for a = 1:length(areas)
        d2Permuted{a} = [];
        mrBrPermuted{a} = [];
    end
end

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
    % popActivity{a} = round(sum(aDataMat, 2));
    popActivity{a} = mean(aDataMat, 2);
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
    
    % NEW: Perform circular permutations if enabled
    if enablePermutations
        fprintf('  Running %d circular permutations per window for area %s...\n', nShuffles, areas{a});
        ticPerm = tic;
        
        % Initialize storage for permutation results [nWindows x nShuffles]
        d2Permuted{a} = nan(numWindows, nShuffles);
        mrBrPermuted{a} = nan(numWindows, nShuffles);
        
        % Get original binned data matrix (before PCA if applied)
        originalDataMat = neural_matrix_ms_to_frames(dataMat(:, aID), optimalBinSize(a));
        nNeurons = size(originalDataMat, 2);
        
        % Permute each window independently (accounting for drift)
        for w = 1:numWindows
            startIdx = (w - 1) * stepSamples + 1; 
            endIdx = startIdx + winSamples - 1;
            
            % Extract this window's data
            windowData = originalDataMat(startIdx:endIdx, :);
            winSamples_window = size(windowData, 1);
            
            % For each shuffle, permute this window's data independently
            for shuffle = 1:nShuffles
                % Circularly permute each neuron independently within this window
                permutedWindowData = windowData;
                for n = 1:nNeurons
                    % Random circular shift for this neuron within this window
                    shiftAmount = randi([1, winSamples_window]);
                    permutedWindowData(:, n) = circshift(permutedWindowData(:, n), shiftAmount);
                end
                
                % Apply PCA if needed (compute PCA on this permuted window separately)
                if pcaFlag
                    [coeffPerm, scorePerm, ~, ~, explainedPerm, muPerm] = pca(permutedWindowData);
                    forDimPerm = find(cumsum(explainedPerm) > 30, 1); 
                    forDimPerm = max(3, min(6, forDimPerm));
                    nDimPerm = 1:forDimPerm; 
                    permutedWindowData = scorePerm(:,nDimPerm) * coeffPerm(:,nDimPerm)' + muPerm;
                end
                
                % Calculate population activity for this permuted window
                wPopActivityPerm = mean(permutedWindowData, 2);
                
                if analyzeMrBr
                    resultPerm = branching_ratio_mr_estimation(wPopActivityPerm);
                    mrBrPermuted{a}(w, shuffle) = resultPerm.branching_ratio;
                end
                
                if analyzeD2
                    [varphiPerm, ~] = myYuleWalker3(wPopActivityPerm, pOrder);
                    d2Permuted{a}(w, shuffle) = getFixedPointDistance2(pOrder, critType, varphiPerm);
                end
            end
            
            if mod(w, max(1, round(numWindows/10))) == 0
                fprintf('    Completed %d/%d windows (%.1f min elapsed)\n', w, numWindows, toc(ticPerm)/60);
            end
        end
        fprintf('  Permutations completed in %.1f minutes\n', toc(ticPerm)/60);
    end

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
            % popActivityModulated{a} = round(sum(modulatedDataMat, 2));
            popActivityModulated{a} = mean(modulatedDataMat, 2);
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
            % popActivityUnmodulated{a} = round(sum(unmodulatedDataMat, 2));
            popActivityUnmodulated{a} = mean(unmodulatedDataMat, 2);
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

% NEW: Save permutation results
if enablePermutations
    results.enablePermutations = true;
    results.nShuffles = nShuffles;
    results.d2Permuted = d2Permuted;
    results.mrBrPermuted = mrBrPermuted;
    
    % Calculate mean and SEM for permutations
    d2PermutedMean = cell(1, length(areas));
    d2PermutedSEM = cell(1, length(areas));
    mrBrPermutedMean = cell(1, length(areas));
    mrBrPermutedSEM = cell(1, length(areas));
    
    for a = 1:length(areas)
        if ~isempty(d2Permuted{a})
            d2PermutedMean{a} = nanmean(d2Permuted{a}, 2);
            d2PermutedSEM{a} = nanstd(d2Permuted{a}, 0, 2) / sqrt(nShuffles);
        else
            d2PermutedMean{a} = [];
            d2PermutedSEM{a} = [];
        end
        if ~isempty(mrBrPermuted{a})
            mrBrPermutedMean{a} = nanmean(mrBrPermuted{a}, 2);
            mrBrPermutedSEM{a} = nanstd(mrBrPermuted{a}, 0, 2) / sqrt(nShuffles);
        else
            mrBrPermutedMean{a} = [];
            mrBrPermutedSEM{a} = [];
        end
    end
    
    results.d2PermutedMean = d2PermutedMean;
    results.d2PermutedSEM = d2PermutedSEM;
    results.mrBrPermutedMean = mrBrPermutedMean;
    results.mrBrPermutedSEM = mrBrPermutedSEM;
else
    results.enablePermutations = false;
    results.nShuffles = 0;
    d2PermutedMean = cell(1, length(areas));
    d2PermutedSEM = cell(1, length(areas));
    mrBrPermutedMean = cell(1, length(areas));
    mrBrPermutedSEM = cell(1, length(areas));
end

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

% =============================    Plotting    =============================
if makePlots
    % Extract filename prefix for titles and filenames
    if exist('dataBaseName', 'var') && ~isempty(dataBaseName)
        filePrefix = dataBaseName(1:min(8, length(dataBaseName)));
    elseif strcmp(dataType, 'reach') && exist('sessionName', 'var') && ~isempty(sessionName)
        filePrefix = sessionName(1:min(8, length(sessionName)));
    elseif exist('saveDir', 'var') && ~isempty(saveDir)
        [~, dirName, ~] = fileparts(saveDir);
        filePrefix = dirName(1:min(8, length(dirName)));
    else
        filePrefix = '';
    end
    
    % Detect monitors and size figure to full screen (prefer second monitor if present)
    monitorPositions = get(0, 'MonitorPositions');
    monitorOne = monitorPositions(1, :);
    monitorTwo = monitorPositions(size(monitorPositions, 1), :);
    if size(monitorPositions, 1) >= 2
        targetPos = monitorTwo;
    else
        targetPos = monitorOne;
    end
    
    
    % ========== Plot 1: Time series of criticality measures ==========
    figure(909); clf;
    set(gcf, 'Units', 'pixels');
    set(gcf, 'Position', targetPos);
    numRows = length(areasToTest) + 1;  % Add one row for combined d2 plot
    ha = tight_subplot(numRows, 1, [0.035 0.04], [0.03 0.08], [0.08 0.04]);

    % Define colors for each area
    areaColors = {[1 0.6 0.6], [0 .8 0], [0 0 1], [1 .4 1]};  % Red, Green, Blue, Magenta for M23, M56, DS, VS
    
    % First row: Plot all d2 traces together
    axes(ha(1)); hold on;
    if analyzeD2
        % Find common time range across all areas
        allStartS = [];
        for a = areasToTest
            if ~isempty(startS{a})
                allStartS = [allStartS, startS{a}];
            end
        end
        if ~isempty(allStartS)
            xLimitsCombined = [min(allStartS), max(allStartS)];
            
            % Collect all d2 values to find maximum
            allD2Values = [];
            for idx = 1:length(areasToTest)
                a = areasToTest(idx);
                if ~isempty(d2{a})
                    allD2Values = [allD2Values; d2{a}(:)];
                end
            end
            
            % Plot d2 for each area with different colors
            for idx = 1:length(areasToTest)
                a = areasToTest(idx);
                if ~isempty(d2{a}) && ~isempty(startS{a})
                    plot(startS{a}, d2{a}, '-', 'Color', areaColors{a}, 'LineWidth', 2, 'DisplayName', areas{a});
                end
            end
            if exist('startBlock2', 'var') && strcmp(dataType, 'reach')
                xline(startBlock2, 'Color', [.8 0 0], 'LineWidth', 3);
            end
                     
            xlim(xLimitsCombined);
            % Set y-axis from 0 to max of all d2 signals
                maxD2 = max(allD2Values(~isnan(allD2Values)));
                minD2 = min(allD2Values(~isnan(allD2Values)));
                ylim([minD2 maxD2]);
            ylabel('d2');
            title('All Areas - d2');
            legend('Location', 'best');
            grid on;
            % Ensure y-axis tick labels are visible
            set(gca, 'YTickLabelMode', 'auto');
            set(gca, 'YTickMode', 'auto');
        end
    end

    % Subsequent rows: Individual area plots
    for idx = 1:length(areasToTest)
        a = areasToTest(idx); 
        axes(ha(idx + 1)); hold on;  % +1 because first row is the combined plot
        
        if analyzeD2
            yyaxis left; 
            % Plot real data first (so permutation line appears on top)
            plot(startS{a}, d2{a}, '-', 'Color', [0 0 1], 'LineWidth', 2, 'DisplayName', 'Real data'); 
            ylabel('d2', 'Color', [0 0 1]); ylim([0 0.5]);
            % Ensure y-axis tick labels are visible
            set(gca, 'YTickLabelMode', 'auto');
            set(gca, 'YTickMode', 'auto');
            
            % Plot permutation mean ± std per window if available
            if enablePermutations && exist('d2Permuted', 'var') && ~isempty(d2Permuted{a})
                % Calculate mean and std for each window across shuffles
                permutedMean = nanmean(d2Permuted{a}, 2);  % Mean across shuffles for each window
                permutedStd = nanstd(d2Permuted{a}, 0, 2);  % Std across shuffles for each window
                
                % Find valid indices (where we have data)
                validIdx = ~isnan(permutedMean) & ~isnan(permutedStd) & ~isnan(startS{a}(:));
                if any(validIdx)
                    xFill = startS{a}(validIdx);
                    yMean = permutedMean(validIdx);
                    yStd = permutedStd(validIdx);
                    
                    % Ensure row vectors for fill
                    if iscolumn(xFill); xFill = xFill'; end
                    if iscolumn(yMean); yMean = yMean'; end
                    if iscolumn(yStd); yStd = yStd'; end
                    
                    % Plot shaded region (mean ± std)
                    fill([xFill, fliplr(xFill)], ...
                         [yMean + yStd, fliplr(yMean - yStd)], ...
                         [0.7 0.7 1], 'FaceAlpha', 0.3, 'EdgeColor', 'none', 'DisplayName', 'Permuted mean ± std');
                    
                    % Plot mean line
                    plot(startS{a}(validIdx), permutedMean(validIdx), '-', 'Color', [0.5 0.5 1], 'LineWidth', 1.5, 'LineStyle', '--', 'DisplayName', 'Permuted mean');
                end
            end
        end
        
        % Plot population activity on right y-axis
        if ~isempty(popActivityWindows{a}) && any(~isnan(popActivityWindows{a}))
            yyaxis right;
            plot(startS{a}, popActivityWindows{a}, '-', 'Color', [0 0 0], 'LineWidth', 2, 'DisplayName', 'PopActivity Windows');
            ylabel('PopActivity Windows', 'Color', [0 0 0]); ylim('auto');
            % Ensure right y-axis tick labels are visible
            set(gca, 'YTickLabelMode', 'auto');
            set(gca, 'YTickMode', 'auto');
        end
        
        % COMMENTED OUT: mrBr plotting code (preserved for future use)
        % if analyzeMrBr
        %     yyaxis right; 
        %     % Plot permutation mean ± SEM if available
        %     if enablePermutations && exist('mrBrPermutedMean', 'var') && ~isempty(mrBrPermutedMean{a}) && ~isempty(mrBrPermutedSEM{a}) && any(~isnan(mrBrPermutedMean{a}))
        %         validIdx = ~isnan(mrBrPermutedMean{a}) & ~isnan(mrBrPermutedSEM{a});
        %         if any(validIdx)
        %             xFill = startS{a}(validIdx);
        %             yMean = mrBrPermutedMean{a}(validIdx);
        %             ySEM = mrBrPermutedSEM{a}(validIdx);
        %             % Ensure row vectors for fill
        %             if iscolumn(xFill); xFill = xFill'; end
        %             if iscolumn(yMean); yMean = yMean'; end
        %             if iscolumn(ySEM); ySEM = ySEM'; end
        %             % Shaded region for SEM
        %             fill([xFill, fliplr(xFill)], ...
        %                  [yMean + ySEM, fliplr(yMean - ySEM)], ...
        %                  [0.8 0.8 0.8], 'FaceAlpha', 0.3, 'EdgeColor', 'none', 'DisplayName', 'Permuted mean ± SEM');
        %             % Mean line
        %             plot(startS{a}(validIdx), mrBrPermutedMean{a}(validIdx), '-', 'Color', [0.5 0.5 0.5], 'LineWidth', 1.5, 'LineStyle', '--', 'DisplayName', 'Permuted mean');
        %         end
        %     end
        %     % Plot real data
        %     plot(startS{a}, mrBr{a}, '-', 'Color', [0 0 0], 'LineWidth', 2, 'DisplayName', 'Real data'); 
        %     yline(1, 'k:', 'LineWidth', 1.5); 
        %     ylabel('mrBr', 'Color', [0 0 0]); ylim('auto');
        % end

        % Add vertical lines at reach onsets (only for reach data)
        if strcmp(dataType, 'reach')
            yyaxis left;
            % Filter reach onsets to only show those within the current plot's time range
            if ~isempty(startS{a})
                plotTimeRange = [startS{a}(1), startS{a}(end)];
                reachOnsetsInRange = reachStart(reachStart >= plotTimeRange(1) & reachStart <= plotTimeRange(2));

                if ~isempty(reachOnsetsInRange)
                    for i = 1:length(reachOnsetsInRange)
                        xline(reachOnsetsInRange(i), 'Color', [0.5 0.5 0.5], 'LineWidth', 0.8, 'LineStyle', '--', 'Alpha', 0.7);
                    end
                    if exist('startBlock2', 'var')
                        xline(startBlock2, 'Color', [1 0 0], 'LineWidth', 3);
                    end
                end
            end
        end
        % Add vertical lines at reach onsets (only for choice countermanding data)
        if strcmp(dataType, 'schall')
            yyaxis left;
            % Filter reach onsets to only show those within the current plot's time range
            if ~isempty(startS{a})
                plotTimeRange = [startS{a}(1), startS{a}(end)];
                reachOnsetsInRange = responseOnset(responseOnset >= plotTimeRange(1) & responseOnset <= plotTimeRange(2));

                if ~isempty(reachOnsetsInRange)
                    for i = 1:length(reachOnsetsInRange)
                        xline(reachOnsetsInRange(i), 'Color', [0.5 0.5 0.5], 'LineWidth', 0.8, 'LineStyle', '--', 'Alpha', 0.7);
                    end
                end
            end
        end
        % Add a line depicting which trial type it is
        if strcmp(dataType, 'hong')
            yyaxis left;
plot(T.startTime_oe, T.trialType/8, 'Color', [1 .3 .3], 'LineWidth', 2, 'LineStyle', '-')           
        end
        
        if ~isempty(startS{a})
            xlim([startS{a}(1) startS{a}(end)])
        end
        title(sprintf('%s - d2 (blue, left) and PopActivity Windows (red, right)', areas{a})); 
        xlabel('Time (s)'); grid on; 
        set(gca, 'XTickLabelMode', 'auto'); 
        % Ensure both left and right y-axes have visible tick labels
        yyaxis left;
        set(gca, 'YTickLabelMode', 'auto');
        set(gca, 'YTickMode', 'auto');
        if ~isempty(popActivityWindows{a}) && any(~isnan(popActivityWindows{a}))
            yyaxis right;
            set(gca, 'YTickLabelMode', 'auto');
            set(gca, 'YTickMode', 'auto');
        end
    end

    % Extract filename prefix for title
    if strcmp(dataType, 'reach') && exist('sessionName', 'var') && ~isempty(sessionName)
        filePrefix = sessionName(1:min(8, length(sessionName)));
    else
        filePrefix = '';
    end
    
    if strcmp(dataType, 'reach')
        if ~isempty(filePrefix)
            sgtitle(sprintf('[%s] %s d2 (blue, left) and PopActivity Windows (red, right) with reach onsets (gray dashed) - win=%gs', filePrefix, dataType, slidingWindowSize));
        else
            sgtitle(sprintf('%s d2 (blue, left) and PopActivity Windows (red, right) with reach onsets (gray dashed) - win=%gs', dataType, slidingWindowSize));
        end
    else
        if ~isempty(filePrefix)
            sgtitle(sprintf('[%s] %s d2 (blue, left) and PopActivity Windows (red, right) - win=%gs', filePrefix, dataType, slidingWindowSize));
        else
            sgtitle(sprintf('%s d2 (blue, left) and PopActivity Windows (red, right) - win=%gs', dataType, slidingWindowSize));
        end
    end
    if ~isempty(filePrefix)
        exportgraphics(gcf, fullfile(saveDir, sprintf('%s_criticality_%s_ar%s_win%d.png', filePrefix, dataType, filenameSuffix, slidingWindowSize)), 'Resolution', 300);
    else
        exportgraphics(gcf, fullfile(saveDir, sprintf('criticality_%s_ar%s_win%d.png', dataType, filenameSuffix, slidingWindowSize)), 'Resolution', 300);
    end

    % ========== Plot 2: Correlation scatter plots ==========
    if plotCorrelations
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
        
        if ~isempty(filePrefix)
            sgtitle(sprintf('[%s] %s Population Activity vs Criticality Correlations - win=%gs', filePrefix, dataType, slidingWindowSize));
        else
            sgtitle(sprintf('%s Population Activity vs Criticality Correlations - win=%gs', dataType, slidingWindowSize));
        end
        if ~isempty(filePrefix)
            exportgraphics(gcf, fullfile(saveDir, sprintf('%s_criticality_%s_correlations%s_win%d.png', filePrefix, dataType, filenameSuffix, slidingWindowSize)), 'Resolution', 300);
            fprintf('Saved %s correlation scatter plots to: %s\n', dataType, fullfile(saveDir, sprintf('%s_criticality_%s_correlations%s_win%d.png', filePrefix, dataType, filenameSuffix, slidingWindowSize)));
        else
            exportgraphics(gcf, fullfile(saveDir, sprintf('criticality_%s_correlations%s_win%d.png', dataType, filenameSuffix, slidingWindowSize)), 'Resolution', 300);
            fprintf('Saved %s correlation scatter plots to: %s\n', dataType, fullfile(saveDir, sprintf('criticality_%s_correlations%s_win%d.png', dataType, filenameSuffix, slidingWindowSize)));
        end
    end

    % ========== Plot 3: Modulated vs unmodulated ==========
    if analyzeModulation
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
                if ~isempty(reachStart) && strcmp(dataType, 'reach')
                    for i = 1:length(reachStart)
                        xline(reachStart(i), 'Color', [0.5 0.5 0.5], 'LineWidth', 0.8, 'LineStyle', '--', 'Alpha', 0.7);
                    end
                end
            end

            title(sprintf('%s - Modulated (red) vs Unmodulated (blue) d2', areas{a}));
            xlabel('Time (s)'); grid on; set(gca, 'XTickLabelMode', 'auto'); set(gca, 'YTickLabelMode', 'auto');
        end

        if ~isempty(filePrefix)
            sgtitle(sprintf('[%s] %s Modulated neurons d2 win=%gs', filePrefix, dataType, slidingWindowSize));
        else
            sgtitle(sprintf('%s Modulated neurons d2 win=%gs', dataType, slidingWindowSize));
        end
        if ~isempty(filePrefix)
            exportgraphics(gcf, fullfile(saveDir, sprintf('%s_criticality_%s_modulated_vs_unmodulated%s_win%d.png', filePrefix, dataType, filenameSuffix, slidingWindowSize)), 'Resolution', 300);
        else
            exportgraphics(gcf, fullfile(saveDir, sprintf('criticality_%s_modulated_vs_unmodulated%s_win%d.png', dataType, filenameSuffix, slidingWindowSize)), 'Resolution', 300);
        end
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
    % For spontaneous data, use regular intervals
    disp('Code for what behaviors you want to align to for spontaneous')
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

