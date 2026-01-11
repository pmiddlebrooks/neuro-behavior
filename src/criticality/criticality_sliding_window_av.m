%%
% Criticality Sliding Window Avalanche Analysis Script (dcc + kappa)
% Unified script for analyzing both reach data and spontaneous data
% Analyzes data using sliding window avalanche approach; saves results to data-specific folders

% Note: Data should be loaded by criticality_sliding_data_prep.m before running this script

paths = get_paths;

% =============================    Configuration    =============================

% Flags
loadExistingResults = false;
makePlots = true;

% Permutation testing flags
enablePermutations = true;  % Set to true to perform circular permutation testing
nShuffles = 3;  % Number of circular permutations to perform

% Analysis parameters
minNeurons = 10;
minSpikesPerBin = 4;
maxSpikesPerBin = 50;
minBinsPerWindow = 1000;


% PCA options
pcaFlag = 0;           % Set to 1 to use PCA
pcaFirstFlag = 1;      % Use first nDim dimensions if 1, last nDim if 0
nDim = 4;              % Number of PCA dimensions to use

% Threshold options
thresholdFlag = 1;     % Set to 1 to use threshold method
thresholdPct = 0.75;   % Threshold as percentage of median
thresholdPct = 1;   % Threshold as percentage of median

% Optimal bin/window size search parameters
candidateFrameSizes = [.02 .03 .04 0.05, .075, 0.1 .15];
candidateWindowSizes = [30, 45, 60, 90, 120];


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
    resultsPath = fullfile(saveDir, sprintf('criticality_sliding_window_av%s_win%d_%s.mat', filenameSuffix, slidingWindowSize, sessionName));
else
    resultsPath = fullfile(saveDir, sprintf('criticality_sliding_window_av%s_win%d.mat', filenameSuffix, slidingWindowSize));
end

% =============================    Analysis    =============================
fprintf('\n=== %s Data Avalanche Analysis ===\n', dataType);

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
for a = areasToTest
    thisDataMat = reconstructedDataMat{a};
        thisFiringRate = sum(thisDataMat(:)) / (size(thisDataMat, 1)/1000);
        [optimalBinSize(a), optimalWindowSize(a)] = ...
            find_optimal_bin_and_window(thisFiringRate, minSpikesPerBin, minBinsPerWindow);
        % optimalBinSize(a) = max(optimalBinSize(a), .02);
    fprintf('Area %s: optimal bin size = %.3f s, optimal window size = %.1f s\n', areas{a}, optimalBinSize(a), optimalWindowSize(a));
end

% Filter areas based on valid optimal bin sizes
validMask = length(idMatIdx{a}) >= minNeurons & (optimalBinSize(areasToTest) <= max(candidateFrameSizes));
areasToTest = areasToTest(validMask);

% Initialize results
[dcc, kappa, decades, startS, tau, alpha, paramSD] = deal(cell(1, length(areas)));

% Initialize permutation results
if enablePermutations
    dccPermuted = cell(1, length(areas));  % Store all permutation results [nWindows x nShuffles]
    kappaPermuted = cell(1, length(areas));
    decadesPermuted = cell(1, length(areas));
    tauPermuted = cell(1, length(areas));
    alphaPermuted = cell(1, length(areas));
    paramSDPermuted = cell(1, length(areas));
    for a = 1:length(areas)
        dccPermuted{a} = [];
        kappaPermuted{a} = [];
        decadesPermuted{a} = [];
        tauPermuted{a} = [];
        alphaPermuted{a} = [];
        paramSDPermuted{a} = [];
    end
end

for a = areasToTest
    fprintf('\nProcessing area %s (%s)...\n', areas{a}, dataType); 
    tic;
    aID = idMatIdx{a};
    
    % Step 4: Bin the original data for dcc/kappa analysis
    aDataMat_dcc = neural_matrix_ms_to_frames(dataMat(:, aID), optimalBinSize(a));
    numTimePoints_dcc = size(aDataMat_dcc, 1);
    stepSamples_dcc = round(avStepSize / optimalBinSize(a));
    winSamples_dcc = round(slidingWindowSize / optimalBinSize(a));
    numWindows_dcc = floor((numTimePoints_dcc - winSamples_dcc) / stepSamples_dcc) + 1;

    % Step 5-6: Apply PCA to binned data and project back to neural space
    if pcaFlag
        [coeff, score, ~, ~, explained, mu] = pca(aDataMat_dcc);
        forDim = find(cumsum(explained) > 30, 1); 
        forDim = max(3, min(6, forDim));
        nDim = 1:forDim; 
        aDataMat_dcc = score(:,nDim) * coeff(:,nDim)' + mu;
    end

    % Step 7: Calculate population activity (no global thresholding - threshold per window)
    aDataMat_dcc = mean(aDataMat_dcc, 2);
    % aDataMat_dcc = sum(aDataMat_dcc, 2);

    % Initialize arrays for dcc/kappa/decades/tau/alpha/paramSD (size by number of windows)
    dcc{a} = nan(1, numWindows_dcc);
    kappa{a} = nan(1, numWindows_dcc);
    decades{a} = nan(1, numWindows_dcc);
    tau{a} = nan(1, numWindows_dcc);
    alpha{a} = nan(1, numWindows_dcc);
    paramSD{a} = nan(1, numWindows_dcc);
    startS{a} = nan(1, numWindows_dcc);

    % Step 8: Process each window for dcc and kappa
    for w = 1:numWindows_dcc
        startIdx = (w - 1) * stepSamples_dcc + 1;
        endIdx = startIdx + winSamples_dcc - 1;
        startS{a}(w) = (startIdx + round(winSamples_dcc/2)-1) * optimalBinSize(a);

        % Calculate population activity for this window
        wPopActivity = aDataMat_dcc(startIdx:endIdx);
        
        % Apply thresholding using median of this window
        threshSpikes = median(wPopActivity);
        wPopActivity(wPopActivity < threshSpikes) = 0;

        % Avalanche analysis for dcc and kappa
        % Find avalanches in the window
        zeroBins = find(wPopActivity == 0);
        if length(zeroBins) > 1 && any(diff(zeroBins)>1)

            % Old code:
            % tic
            % % Create avalanche data
            % asdfMat = rastertoasdf2(wPopActivity', optimalBinSize(a)*1000, 'CBModel', 'Spikes', 'DS');
            % Av = avprops(asdfMat, 'ratio', 'fingerprint');
            % 
            % % Calculate avalanche parameters
            % [tau, ~, tauC, ~, alpha, ~, paramSD, decades_val] = avalanche_log(Av, 0)
% toc

            % Woody's new code:
            % tic
[sizes, durs] = getAvalanches(wPopActivity', .5, 1);
gof = .8;
plotAv = 0;
[tauVal, plrS, minavS, maxavS, ~, ~, ~] = plfit2023(sizes, gof, plotAv, 0);
[alphaVal, plrD, minavD, maxavD, ~, ~, ~] = plfit2023(durs, gof, plotAv, 0);
[paramSDVal, sigmaNuZInvStd, logCoeff] = size_given_duration(sizes, durs, 'durmin', minavD, 'durmax', maxavD);
% toc

            % dcc (distance to criticality from avalanche analysis)
            dcc{a}(w) = distance_to_criticality(tauVal, alphaVal, paramSDVal);

            % kappa (avalanche shape parameter)
            % kappa{a}(w) = compute_kappa(Av.size);
            kappa{a}(w) = compute_kappa(sizes);

            % decades (log10 of avalanche size range)
            % decades{a}(w) = decades_val;
            decades{a}(w) = plrS;
            
            % Store tau, alpha, and paramSD
            tau{a}(w) = tauVal;
            alpha{a}(w) = alphaVal;
            paramSD{a}(w) = paramSDVal;
        end
    end

    fprintf('Area %s completed in %.1f minutes\n', areas{a}, toc/60);
    
    % Perform circular permutations if enabled
    if enablePermutations
        fprintf('  Running %d circular permutations per window for area %s...\n', nShuffles, areas{a});
        ticPerm = tic;
        
        % Initialize storage for permutation results [nWindows x nShuffles]
        dccPermuted{a} = nan(numWindows_dcc, nShuffles);
        kappaPermuted{a} = nan(numWindows_dcc, nShuffles);
        decadesPermuted{a} = nan(numWindows_dcc, nShuffles);
        tauPermuted{a} = nan(numWindows_dcc, nShuffles);
        alphaPermuted{a} = nan(numWindows_dcc, nShuffles);
        paramSDPermuted{a} = nan(numWindows_dcc, nShuffles);
        
        % Get original binned data matrix (before PCA if applied)
        originalDataMat = neural_matrix_ms_to_frames(dataMat(:, aID), optimalBinSize(a));
        nNeurons = size(originalDataMat, 2);
        
        % Permute each window independently (accounting for drift)
        for w = 1:numWindows_dcc
            startIdx = (w - 1) * stepSamples_dcc + 1; 
            endIdx = startIdx + winSamples_dcc - 1;
            
            % Extract this window's data
            windowData = originalDataMat(startIdx:endIdx, :);
            winSamples = size(windowData, 1);
            
            % For each shuffle, permute this window's data independently
            for shuffle = 1:nShuffles
                % Circularly permute each neuron independently within this window
                permutedWindowData = windowData;
                for n = 1:nNeurons
                    % Random circular shift for this neuron within this window
                    shiftAmount = randi([1, winSamples]);
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
                
                % Apply thresholding using median of this window
                threshSpikes = median(wPopActivityPerm);
                wPopActivityPerm(wPopActivityPerm < threshSpikes) = 0;
                
                % Avalanche analysis for permuted data
                zeroBins = find(wPopActivityPerm == 0);
                if length(zeroBins) > 1 && any(diff(zeroBins)>1)
                    [sizesPerm, dursPerm] = getAvalanches(wPopActivityPerm', .5, 1);
                    gof = .8;
                    plotAv = 0;
                    [tauPerm, plrSPerm, minavSPerm, maxavSPerm, ~, ~, ~] = plfit2023(sizesPerm, gof, plotAv, 0);
                    [alphaPerm, plrDPerm, minavDPerm, maxavDPerm, ~, ~, ~] = plfit2023(dursPerm, gof, plotAv, 0);
                    [paramSDPerm, sigmaNuZInvStdPerm, logCoeffPerm] = size_given_duration(sizesPerm, dursPerm, 'durmin', minavDPerm, 'durmax', maxavDPerm);
                    
                    % Calculate metrics for permuted data
                    dccPermuted{a}(w, shuffle) = distance_to_criticality(tauPerm, alphaPerm, paramSDPerm);
                    kappaPermuted{a}(w, shuffle) = compute_kappa(sizesPerm);
                    decadesPermuted{a}(w, shuffle) = plrSPerm;
                    tauPermuted{a}(w, shuffle) = tauPerm;
                    alphaPermuted{a}(w, shuffle) = alphaPerm;
                    paramSDPermuted{a}(w, shuffle) = paramSDPerm;
                end
            end
            
            if mod(w, max(1, round(numWindows_dcc/10))) == 0
                fprintf('    Completed %d/%d windows (%.1f min elapsed)\n', w, numWindows_dcc, toc(ticPerm)/60);
            end
        end
        fprintf('  Permutations completed in %.1f minutes\n', toc(ticPerm)/60);
    end
end

% =============================    Save Results    =============================
results = struct(); 
results.dataType = dataType;
results.areas = areas; 
results.dcc = dcc; 
results.kappa = kappa; 
results.decades = decades;
results.tau = tau;
results.alpha = alpha;
results.paramSD = paramSD;
results.startS = startS;
results.optimalBinSize = optimalBinSize; 
results.optimalWindowSize = optimalWindowSize;
results.params.slidingWindowSize = slidingWindowSize;
results.params.avStepSize = avStepSize;
results.params.pcaFlag = pcaFlag;
results.params.pcaFirstFlag = pcaFirstFlag;
results.params.nDim = nDim;
results.params.thresholdFlag = thresholdFlag;
results.params.thresholdPct = thresholdPct;

% Save permutation results
if enablePermutations
    results.enablePermutations = true;
    results.nShuffles = nShuffles;
    results.dccPermuted = dccPermuted;
    results.kappaPermuted = kappaPermuted;
    results.decadesPermuted = decadesPermuted;
    results.tauPermuted = tauPermuted;
    results.alphaPermuted = alphaPermuted;
    results.paramSDPermuted = paramSDPermuted;
    
    % Calculate mean and SEM for permutations
    dccPermutedMean = cell(1, length(areas));
    dccPermutedSEM = cell(1, length(areas));
    kappaPermutedMean = cell(1, length(areas));
    kappaPermutedSEM = cell(1, length(areas));
    decadesPermutedMean = cell(1, length(areas));
    decadesPermutedSEM = cell(1, length(areas));
    tauPermutedMean = cell(1, length(areas));
    tauPermutedSEM = cell(1, length(areas));
    alphaPermutedMean = cell(1, length(areas));
    alphaPermutedSEM = cell(1, length(areas));
    paramSDPermutedMean = cell(1, length(areas));
    paramSDPermutedSEM = cell(1, length(areas));
    
    for a = 1:length(areas)
        if ~isempty(dccPermuted{a})
            dccPermutedMean{a} = nanmean(dccPermuted{a}, 2);
            dccPermutedSEM{a} = nanstd(dccPermuted{a}, 0, 2) / sqrt(nShuffles);
        else
            dccPermutedMean{a} = [];
            dccPermutedSEM{a} = [];
        end
        if ~isempty(kappaPermuted{a})
            kappaPermutedMean{a} = nanmean(kappaPermuted{a}, 2);
            kappaPermutedSEM{a} = nanstd(kappaPermuted{a}, 0, 2) / sqrt(nShuffles);
        else
            kappaPermutedMean{a} = [];
            kappaPermutedSEM{a} = [];
        end
        if ~isempty(decadesPermuted{a})
            decadesPermutedMean{a} = nanmean(decadesPermuted{a}, 2);
            decadesPermutedSEM{a} = nanstd(decadesPermuted{a}, 0, 2) / sqrt(nShuffles);
        else
            decadesPermutedMean{a} = [];
            decadesPermutedSEM{a} = [];
        end
        if ~isempty(tauPermuted{a})
            tauPermutedMean{a} = nanmean(tauPermuted{a}, 2);
            tauPermutedSEM{a} = nanstd(tauPermuted{a}, 0, 2) / sqrt(nShuffles);
        else
            tauPermutedMean{a} = [];
            tauPermutedSEM{a} = [];
        end
        if ~isempty(alphaPermuted{a})
            alphaPermutedMean{a} = nanmean(alphaPermuted{a}, 2);
            alphaPermutedSEM{a} = nanstd(alphaPermuted{a}, 0, 2) / sqrt(nShuffles);
        else
            alphaPermutedMean{a} = [];
            alphaPermutedSEM{a} = [];
        end
        if ~isempty(paramSDPermuted{a})
            paramSDPermutedMean{a} = nanmean(paramSDPermuted{a}, 2);
            paramSDPermutedSEM{a} = nanstd(paramSDPermuted{a}, 0, 2) / sqrt(nShuffles);
        else
            paramSDPermutedMean{a} = [];
            paramSDPermutedSEM{a} = [];
        end
    end
    
    results.dccPermutedMean = dccPermutedMean;
    results.dccPermutedSEM = dccPermutedSEM;
    results.kappaPermutedMean = kappaPermutedMean;
    results.kappaPermutedSEM = kappaPermutedSEM;
    results.decadesPermutedMean = decadesPermutedMean;
    results.decadesPermutedSEM = decadesPermutedSEM;
    results.tauPermutedMean = tauPermutedMean;
    results.tauPermutedSEM = tauPermutedSEM;
    results.alphaPermutedMean = alphaPermutedMean;
    results.alphaPermutedSEM = alphaPermutedSEM;
    results.paramSDPermutedMean = paramSDPermutedMean;
    results.paramSDPermutedSEM = paramSDPermutedSEM;
else
    results.enablePermutations = false;
    results.nShuffles = 0;
end

save(resultsPath, 'results'); 
fprintf('Saved %s dcc/kappa to %s\n', dataType, resultsPath);

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
    
    % First, collect all data to determine axis limits (including permuted data)
    allStartS = [];
    allDcc = [];
    allTau = [];
    allAlpha = [];
    allParamSD = [];
    allDecades = [];
    
    for idx = 1:length(areasToTest)
        a = areasToTest(idx);
        if ~isempty(startS{a})
            allStartS = [allStartS, startS{a}(~isnan(startS{a}))];
        end
        if ~isempty(dcc{a})
            allDcc = [allDcc, dcc{a}(~isnan(dcc{a}))];
        end
        if ~isempty(tau{a})
            allTau = [allTau, tau{a}(~isnan(tau{a}))];
        end
        if ~isempty(alpha{a})
            allAlpha = [allAlpha, alpha{a}(~isnan(alpha{a}))];
        end
        if ~isempty(paramSD{a})
            allParamSD = [allParamSD, paramSD{a}(~isnan(paramSD{a}))];
        end
        if ~isempty(decades{a})
            allDecades = [allDecades, decades{a}(~isnan(decades{a}))];
        end
        
        % Include permuted data in axis limits if available
        if enablePermutations
            % dcc permuted data
            if exist('dccPermuted', 'var') && ~isempty(dccPermuted{a})
                permutedMean = nanmean(dccPermuted{a}, 2);
                permutedStd = nanstd(dccPermuted{a}, 0, 2);
                validIdx = ~isnan(permutedMean) & ~isnan(permutedStd);
                if any(validIdx)
                    permutedVals = [permutedMean(validIdx) + permutedStd(validIdx); permutedMean(validIdx) - permutedStd(validIdx)];
                    allDcc = [allDcc(:); permutedVals(:)];
                end
            end
            
            % tau permuted data
            if exist('tauPermuted', 'var') && ~isempty(tauPermuted{a})
                permutedMean = nanmean(tauPermuted{a}, 2);
                permutedStd = nanstd(tauPermuted{a}, 0, 2);
                validIdx = ~isnan(permutedMean) & ~isnan(permutedStd);
                if any(validIdx)
                    permutedVals = [permutedMean(validIdx) + permutedStd(validIdx); permutedMean(validIdx) - permutedStd(validIdx)];
                    allTau = [allTau(:); permutedVals(:)];
                end
            end
            
            % alpha permuted data
            if exist('alphaPermuted', 'var') && ~isempty(alphaPermuted{a})
                permutedMean = nanmean(alphaPermuted{a}, 2);
                permutedStd = nanstd(alphaPermuted{a}, 0, 2);
                validIdx = ~isnan(permutedMean) & ~isnan(permutedStd);
                if any(validIdx)
                    permutedVals = [permutedMean(validIdx) + permutedStd(validIdx); permutedMean(validIdx) - permutedStd(validIdx)];
                    allAlpha = [allAlpha(:); permutedVals(:)];
                end
            end
            
            % paramSD permuted data
            if exist('paramSDPermuted', 'var') && ~isempty(paramSDPermuted{a})
                permutedMean = nanmean(paramSDPermuted{a}, 2);
                permutedStd = nanstd(paramSDPermuted{a}, 0, 2);
                validIdx = ~isnan(permutedMean) & ~isnan(permutedStd);
                if any(validIdx)
                    permutedVals = [permutedMean(validIdx) + permutedStd(validIdx); permutedMean(validIdx) - permutedStd(validIdx)];
                    allParamSD = [allParamSD(:); permutedVals(:)];
                end
            end
            
            % decades permuted data
            if exist('decadesPermuted', 'var') && ~isempty(decadesPermuted{a})
                permutedMean = nanmean(decadesPermuted{a}, 2);
                permutedStd = nanstd(decadesPermuted{a}, 0, 2);
                validIdx = ~isnan(permutedMean) & ~isnan(permutedStd);
                if any(validIdx)
                    permutedVals = [permutedMean(validIdx) + permutedStd(validIdx); permutedMean(validIdx) - permutedStd(validIdx)];
                    allDecades = [allDecades(:); permutedVals(:)];
                end
            end
        end
    end
    
    % Determine axis limits
    if ~isempty(allStartS)
        xMin = min(allStartS);
        xMax = max(allStartS);
    else
        xMin = 0;
        xMax = 1;
    end
    
    % Set dcc y-limit max
    dccMaxPlot = 1;
    if ~isempty(allDcc)
        yMinDcc = 0;
        yMaxDcc = min(max(allDcc), dccMaxPlot);
        % Add small padding
        yRangeDcc = yMaxDcc - yMinDcc;
        yMaxDcc = min(yMaxDcc + 0.05 * yRangeDcc, dccMaxPlot);
    else
        yMinDcc = 0;
        yMaxDcc = dccMaxPlot;
    end
    
    % Determine y-limits for tau, alpha, paramSD (all on same plot)
    allParams = [allTau, allAlpha, allParamSD];
    if ~isempty(allParams)
        yMinParams = min(allParams(:));
        yMaxParams = max(allParams(:));
        % Add small padding
        yRangeParams = yMaxParams - yMinParams;
        yMinParams = yMinParams - 0.05 * yRangeParams;
        yMaxParams = yMaxParams + 0.05 * yRangeParams;
    else
        yMinParams = 0;
        yMaxParams = 1;
    end
    
    if ~isempty(allDecades)
        yMinDecades = min(allDecades);
        yMaxDecades = max(allDecades);
        % Add small padding
        yRangeDecades = yMaxDecades - yMinDecades;
        yMinDecades = yMinDecades - 0.05 * yRangeDecades;
        yMaxDecades = yMaxDecades + 0.05 * yRangeDecades;
    else
        yMinDecades = 0;
        yMaxDecades = 1;
    end
    
    % Avalanche analysis plots: 3 rows (dcc, tau/alpha/paramSD, decades) x num areas (columns)
    figure(903); clf; 
    set(gcf, 'Position', targetPos);
    
    for idx = 1:length(areasToTest)
        a = areasToTest(idx);
        
        % dcc (top row)
        subplot(3, length(areasToTest), idx);
        hold on;
        plot(startS{a}, dcc{a}, '-', 'Color', [1 0 0], 'LineWidth', 2);
        
        % Plot permutation mean ± std per window if available
        if enablePermutations && exist('dccPermuted', 'var') && ~isempty(dccPermuted{a})
            % Calculate mean and std for each window across shuffles
            permutedMean = nanmean(dccPermuted{a}, 2);  % Mean across shuffles for each window
            permutedStd = nanstd(dccPermuted{a}, 0, 2);  % Std across shuffles for each window
            
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
        
        title(sprintf('%s - dcc', areas{a})); 
        xlabel('Time (s)'); ylabel('dcc'); grid on;
        xlim([xMin, xMax]);
        ylim([yMinDcc, dccMaxPlot]);
        
        % tau, alpha, paramSD (middle row) - all on same subplot
        subplot(3, length(areasToTest), length(areasToTest) + idx);
        hold on;
        
        % Plot tau
        if ~isempty(tau{a})
            plot(startS{a}, tau{a}, '-', 'Color', [1 0.5 0], 'LineWidth', 2, 'DisplayName', 'tau');
        end
        
        % Plot alpha
        if ~isempty(alpha{a})
            plot(startS{a}, alpha{a}, '-', 'Color', [0 0.8 0], 'LineWidth', 2, 'DisplayName', 'alpha');
        end
        
        % Plot paramSD
        if ~isempty(paramSD{a})
            plot(startS{a}, paramSD{a}, '-', 'Color', [0 0 1], 'LineWidth', 2, 'DisplayName', 'paramSD');
        end
        
        % Plot permutation mean ± std per window for tau if available
        if enablePermutations && exist('tauPermuted', 'var') && ~isempty(tauPermuted{a})
            permutedMean = nanmean(tauPermuted{a}, 2);
            permutedStd = nanstd(tauPermuted{a}, 0, 2);
            validIdx = ~isnan(permutedMean) & ~isnan(permutedStd) & ~isnan(startS{a}(:));
            if any(validIdx)
                xFill = startS{a}(validIdx);
                yMean = permutedMean(validIdx);
                yStd = permutedStd(validIdx);
                if iscolumn(xFill); xFill = xFill'; end
                if iscolumn(yMean); yMean = yMean'; end
                if iscolumn(yStd); yStd = yStd'; end
                fill([xFill, fliplr(xFill)], ...
                     [yMean + yStd, fliplr(yMean - yStd)], ...
                     [1 0.8 0.6], 'FaceAlpha', 0.2, 'EdgeColor', 'none', 'DisplayName', 'tau Permuted mean ± std');
                plot(startS{a}(validIdx), permutedMean(validIdx), '-', 'Color', [1 0.7 0.3], 'LineWidth', 1, 'LineStyle', '--', 'DisplayName', 'tau Permuted mean');
            end
        end
        
        % Plot permutation mean ± std per window for alpha if available
        if enablePermutations && exist('alphaPermuted', 'var') && ~isempty(alphaPermuted{a})
            permutedMean = nanmean(alphaPermuted{a}, 2);
            permutedStd = nanstd(alphaPermuted{a}, 0, 2);
            validIdx = ~isnan(permutedMean) & ~isnan(permutedStd) & ~isnan(startS{a}(:));
            if any(validIdx)
                xFill = startS{a}(validIdx);
                yMean = permutedMean(validIdx);
                yStd = permutedStd(validIdx);
                if iscolumn(xFill); xFill = xFill'; end
                if iscolumn(yMean); yMean = yMean'; end
                if iscolumn(yStd); yStd = yStd'; end
                fill([xFill, fliplr(xFill)], ...
                     [yMean + yStd, fliplr(yMean - yStd)], ...
                     [0.6 0.9 0.6], 'FaceAlpha', 0.2, 'EdgeColor', 'none', 'DisplayName', 'alpha Permuted mean ± std');
                plot(startS{a}(validIdx), permutedMean(validIdx), '-', 'Color', [0.3 0.7 0.3], 'LineWidth', 1, 'LineStyle', '--', 'DisplayName', 'alpha Permuted mean');
            end
        end
        
        % Plot permutation mean ± std per window for paramSD if available
        if enablePermutations && exist('paramSDPermuted', 'var') && ~isempty(paramSDPermuted{a})
            permutedMean = nanmean(paramSDPermuted{a}, 2);
            permutedStd = nanstd(paramSDPermuted{a}, 0, 2);
            validIdx = ~isnan(permutedMean) & ~isnan(permutedStd) & ~isnan(startS{a}(:));
            if any(validIdx)
                xFill = startS{a}(validIdx);
                yMean = permutedMean(validIdx);
                yStd = permutedStd(validIdx);
                if iscolumn(xFill); xFill = xFill'; end
                if iscolumn(yMean); yMean = yMean'; end
                if iscolumn(yStd); yStd = yStd'; end
                fill([xFill, fliplr(xFill)], ...
                     [yMean + yStd, fliplr(yMean - yStd)], ...
                     [0.6 0.6 0.9], 'FaceAlpha', 0.2, 'EdgeColor', 'none', 'DisplayName', 'paramSD Permuted mean ± std');
                plot(startS{a}(validIdx), permutedMean(validIdx), '-', 'Color', [0.3 0.3 0.7], 'LineWidth', 1, 'LineStyle', '--', 'DisplayName', 'paramSD Permuted mean');
            end
        end
        
        title(sprintf('%s - tau (orange), alpha (green), paramSD (blue)', areas{a})); 
        xlabel('Time (s)'); ylabel('Value'); grid on;
        xlim([xMin, xMax]);
        ylim([yMinParams, yMaxParams]);
        legend('Location', 'best');
        
        % decades (bottom row)
        subplot(3, length(areasToTest), 2*length(areasToTest) + idx);
        hold on;
        plot(startS{a}, decades{a}, '-', 'Color', [0.6 0 0.6], 'LineWidth', 2);
        
        % Plot permutation mean ± std per window if available
        if enablePermutations && exist('decadesPermuted', 'var') && ~isempty(decadesPermuted{a})
            % Calculate mean and std for each window across shuffles
            permutedMean = nanmean(decadesPermuted{a}, 2);  % Mean across shuffles for each window
            permutedStd = nanstd(decadesPermuted{a}, 0, 2);  % Std across shuffles for each window
            
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
        
        title(sprintf('%s - decades', areas{a})); 
        xlabel('Time (s)'); ylabel('decades'); grid on;
        xlim([xMin, xMax]);
        ylim([yMinDecades, yMaxDecades]);
    end
    
    % Extract filename prefix for title
    if exist('dataBaseName', 'var') && ~isempty(dataBaseName)
        filePrefix = dataBaseName(1:min(8, length(dataBaseName)));
    elseif exist('saveDir', 'var') && ~isempty(saveDir)
        [~, dirName, ~] = fileparts(saveDir);
        filePrefix = dirName(1:min(8, length(dirName)));
    else
        filePrefix = '';
    end
    
    if ~isempty(filePrefix)
        sgtitle(sprintf('[%s] %s dcc (top), tau/alpha/paramSD (mid), decades (bottom) - win=%gs, step=%gs', filePrefix, dataType, slidingWindowSize, avStepSize), 'interpreter', 'none');
    else
        sgtitle(sprintf('%s dcc (top), tau/alpha/paramSD (mid), decades (bottom) - win=%gs, step=%gs', dataType, slidingWindowSize, avStepSize), 'interpreter', 'none');
    end
    if ~isempty(filePrefix)
        exportgraphics(gcf, fullfile(saveDir, sprintf('%s_criticality_%s_av_win%d_step%d.png', filePrefix, dataType, slidingWindowSize, avStepSize)), 'Resolution', 300);
        fprintf('Saved %s avalanche plots to: %s\n', dataType, fullfile(saveDir, sprintf('%s_criticality_%s_av_win%d_step%d.png', filePrefix, dataType, slidingWindowSize, avStepSize)));
    else
        exportgraphics(gcf, fullfile(saveDir, sprintf('criticality_%s_av_win%d_step%d.png', dataType, slidingWindowSize, avStepSize)), 'Resolution', 300);
        fprintf('Saved %s avalanche plots to: %s\n', dataType, fullfile(saveDir, sprintf('criticality_%s_av_win%d_step%d.png', dataType, slidingWindowSize, avStepSize)));
    end
end

fprintf('\n=== %s Avalanche Analysis Complete ===\n', dataType);

%% ==============================================     Helper Functions     ==============================================

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
