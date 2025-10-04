%%
% Criticality Comparison Script
% Compares different measures of criticality within each brain area
% Uses sliding window analysis on Mark's reach data and naturalistic data
% Measures: mrBr (MR branching ratio), d2 (distance to criticality),
%           dcc (distance to criticality from avalanche analysis), kappa
%
% Update controls:
%   loadExistingResults - load existing saved results to update selectively
%   updateD2MrBr        - run and update d2/mrBr analyses only if true
%   updateDccKappa      - run and update dcc/kappa analyses only if true
%   makePlots           - generate plots if true
%   runCorrelation      - compute correlation matrices if true

%%
opts = neuro_behavior_options;
opts.minActTime = .16;
opts.collectStart = 0 * 60 * 60; % seconds
opts.minFiringRate = .05;
opts.frameSize = .001;

paths = get_paths;

% =============================    Update/Run Flags    =============================
loadExistingResults = true;   % load existing results file to preserve untouched fields
updateD2MrBr = true;          % run d2/mrBr analyses
updateDccKappa = false;       % run dcc/kappa analyses
makePlots = false;            % create comparison plots
runCorrelation = false;       % compute correlation matrices

% Load existing results if requested
slidingWindowSize = 1;        % For d2, use a small window to try to optimize temporal resolution

resultsBasePath = fullfile(paths.dropPath, 'criticality/criticality_compare_results.mat');
resultsPathWin = fullfile(paths.dropPath, sprintf('criticality/criticality_compare_results_win%gs.mat', slidingWindowSize));
results = struct();
if loadExistingResults
    if exist(resultsPathWin, 'file')
        loaded = load(resultsPathWin);
    elseif exist(resultsBasePath, 'file')
        loaded = load(resultsBasePath);
    else
        loaded = struct();
    end
    if isfield(loaded, 'results')
        results = loaded.results;
    end
end

% Criticality parameter ranges for reference
tauRange = [1.2 2.5];
alphaRange = [1.5 2.2];
paramSDRange = [1.3 1.7];

% Monitor setup
monitorPositions = get(0, 'MonitorPositions');
monitorOne = monitorPositions(1, :);
monitorTwo = monitorPositions(size(monitorPositions, 1), :);

%% ==============================================     Data Loading     ==============================================

%% Naturalistic data
getDataType = 'spikes';
opts.firingRateCheckTime = 5 * 60;
opts.collectStart = 0 * 60; % seconds
opts.collectFor = 45 * 60; % seconds
get_standard_data

areas = {'M23', 'M56', 'DS', 'VS'};
idListNat = {idM23, idM56, idDS, idVS};

%
% Mark's reach data
% dataR = load(fullfile(paths.dropPath, 'reach_data/Y4_100623_Spiketimes_idchan_BEH.mat'));
dataR = load(fullfile(paths.dropPath, 'reach_data/Copy_of_Y4_100623_Spiketimes_idchan_BEH.mat'));
[dataMatR, idLabels, areaLabels, rmvNeurons] = neural_matrix_mark_data(dataR, opts);

% Get data until 1 sec after the last reach ending.
% cutOff = round((dataR.R(end,2) + 1000) / 1000 / opts.frameSize);
% dataMatR = dataMatR(1:cutOff,:);

idM23R = find(strcmp(areaLabels, 'M23'));
idM56R = find(strcmp(areaLabels, 'M56'));
idDSR = find(strcmp(areaLabels, 'DS'));
idVSR = find(strcmp(areaLabels, 'VS'));

idListRea = {idM23R, idM56R, idDSR, idVSR};

%% ==============================================     Analysis Parameters     ==============================================

% PCA options
pcaFlag = 0;           % Set to 1 to use PCA
pcaFirstFlag = 1;      % Use first nDim dimensions if 1, last nDim if 0
nDim = 4;              % Number of PCA dimensions to use

% Threshold options
thresholdFlag = 1;     % Set to 1 to use threshold method
thresholdPct = 0.75;   % Threshold as percentage of median

% Optimal bin/window size search parameters
% candidateFrameSizes = [0.004, 0.01, 0.02, 0.05, .075, 0.1];
% candidateFrameSizes = [0.01, 0.02, .03 .04 0.05, .075, 0.1];
candidateFrameSizes = [.02 .03 .04 0.05, .075, 0.1 .15]; % seconds
candidateWindowSizes = [30, 45, 60, 90, 120]; % seconds
% slidingWindowSize = 3;% For d2, use a small window to try to optimize temporal resolution
windowSizes = repmat(slidingWindowSize, 1, 4); % For d2, use a small window to try to optimize temporal resolution
minSpikesPerBin = 3;
maxSpikesPerBin = 30;
minBinsPerWindow = 1000;


% For randomized comparison (for significance testing)
nShuffles = 5;

% AR model parameters for d2 calculation
pOrder = 10;
critType = 2;
d2StepSize = .02;

% Separate parameters for dcc/kappa analysis
dccStepSize = round(3 * 60); % seconds - user specified
dccWindowSize = 6 * 60; % seconds - user specified

 %% ==============================================     Naturalistic Data Analysis     ==============================================

areasToTest = 1:4;

if updateD2MrBr
    fprintf('\n=== Naturalistic Data d2/mrbr Analysis ===\n');

    % Step 1-2: Apply PCA to original data to determine nDim and project back to neural space
    fprintf('\n--- Step 1-2: PCA on original data if requested ---\n');
    reconstructedDataMat = cell(1, length(areas));
    for a = areasToTest
        aID = idListNat{a};
        thisDataMat = dataMat(:, aID);

        if pcaFlag
            [coeff, score, ~, ~, explained, mu] = pca(thisDataMat);
            forDim = find(cumsum(explained) > 30, 1);
            forDim = max(3, forDim);
            forDim = min(6, forDim);
            if pcaFirstFlag
                fprintf('Area %s: Using PCA first %d dimensions\n', areas{a}, forDim);
                nDim = 1:forDim;
            else
                fprintf('Area %s: Using PCA last %d dimensions\n', areas{a}, size(score, 2) - forDim + 1);
                nDim = forDim+1:size(score, 2);
            end
            reconstructedDataMat{a} = score(:,nDim) * coeff(:,nDim)' + mu;
        else
            reconstructedDataMat{a} = thisDataMat;
        end
    end

    % Step 3: Find optimal parameters using reconstructed data
    fprintf('\n--- Step 3: Finding optimal parameters ---\n');
    optimalBinSizeNat = zeros(1, length(areas));
    optimalWindowSizeNat = zeros(1, length(areas));
    for a = areasToTest
        thisDataMat = reconstructedDataMat{a};
        [optimalBinSizeNat(a), optimalWindowSizeNat(a)] = ...
            find_optimal_bin_and_window(thisDataMat, candidateFrameSizes, candidateWindowSizes, minSpikesPerBin, maxSpikesPerBin, minBinsPerWindow);
        fprintf('Area %s: optimal bin size = %.3f s, optimal window size = %.1f s\n', areas{a}, optimalBinSizeNat(a), optimalWindowSizeNat(a));
    end

    % Use optimal bin sizes for each area (no longer unified)
    % Set d2StepSize to optimal bin size for each area
    % Prepare per-area step/window sizes for d2
    d2StepSizeNat = optimalBinSizeNat;
    d2WindowSizeNat = optimalWindowSizeNat;
    d2WindowSizeNat = windowSizes;

    fprintf('Using area-specific optimal parameters:\n');
    for a = areasToTest
        fprintf('  Area %s: bin size = %.3f s, window size = %.1f s, d2StepSize = %.3f s\n', ...
            areas{a}, optimalBinSizeNat(a), optimalWindowSizeNat(a), d2StepSizeNat(a));
    end

    % Initialize results for naturalistic data
    mrBrNat = cell(1, length(areas));
    d2Nat = cell(1, length(areas));
    dccNat = cell(1, length(areas));
    kappaNat = cell(1, length(areas));
    startSNat = cell(1, length(areas));

    for a = areasToTest %length(areas)
        fprintf('\nProcessing area %s (Naturalistic)...\n', areas{a});
        tic;

        aID = idListNat{a};
        stepSamples = round(d2StepSizeNat(a) / optimalBinSizeNat(a));
        winSamples = round(d2WindowSizeNat(a) / optimalBinSizeNat(a));

        % Step 4: Bin the original data using area-specific optimal bin size
        aDataMatNat = neural_matrix_ms_to_frames(dataMat(:, aID), optimalBinSizeNat(a));
        numTimePoints = size(aDataMatNat, 1);
        numWindows = floor((numTimePoints - winSamples) / stepSamples) + 1;

        % Step 5-6: Apply PCA to binned data and project back to neural space
        if pcaFlag
            [coeff, score, ~, ~, explained, mu] = pca(aDataMatNat);
            forDim = find(cumsum(explained) > 30, 1);
            forDim = max(3, forDim);
            forDim = min(6, forDim);
            if pcaFirstFlag
                nDim = 1:forDim;
            else
                nDim = forDim+1:size(score, 2);
            end
            aDataMatNat = score(:,nDim) * coeff(:,nDim)' + mu;
        end

        % Step 7: Apply thresholding if needed
        % if thresholdFlag
        %     aDataMatNat = round(sum(aDataMatNat, 2));
        %     threshSpikes = thresholdPct * median(aDataMatNat);
        %     aDataMatNat(aDataMatNat < threshSpikes) = 0;
        %     fprintf('Area %s: Threshold = %.3f, Prop zeros = %.3f\n', areas{a}, threshSpikes, sum(aDataMatNat == 0) / length(aDataMatNat));
        % else
        aDataMatNat = round(sum(aDataMatNat, 2));
        % end

        % Initialize arrays for this area (size by number of windows)
        startSNat{a} = nan(1, numWindows);
        mrBrNat{a} = nan(1, numWindows);
        d2Nat{a} = nan(1, numWindows);

        % Step 8: Process each window for mrBr and d2
            kMax = round(10 / optimalBinSizeNat(a));
            warning('Using default kMax for br_mr_estimation')
        for w = 1:numWindows
            startIdx = (w - 1) * stepSamples + 1;
            endIdx = startIdx + winSamples - 1;
            centerIdx = startIdx + floor((endIdx - startIdx)/2);
            startSNat{a}(w) = (startIdx + round(winSamples/2)-1) * optimalBinSizeNat(a); % center time of window

            % Extract window data and
            % Calculate population activity for MR and d2
            wPopActivity = aDataMatNat(startIdx:endIdx);

            % MR branching ratio
            % result = branching_ratio_mr_estimation(wPopActivity, kMax);
            result = branching_ratio_mr_estimation(wPopActivity);
            mrBrNat{a}(w) = result.branching_ratio;

            % d2 (distance to criticality from AR model)
            [varphi, ~] = myYuleWalker3(wPopActivity, pOrder);
            d2Val = getFixedPointDistance2(pOrder, critType, varphi);
            d2Nat{a}(w) = d2Val;
        end

        fprintf('Area %s completed in %.1f minutes\n', areas{a}, toc/60);
    end
end % updateD2MrBr


% Separate loop for dcc and kappa analysis
if updateDccKappa
    fprintf('\n=== Naturalistic Data dcc/kappa Analysis ===\n');
    for a = areasToTest
        fprintf('\nProcessing dcc/kappa for area %s (Naturalistic)...\n', areas{a});
        tic;

        aID = idListNat{a};

        % Step 4: Bin the original data for dcc/kappa analysis
        % Ensure optimalBinSizeNat available when only dcc/kappa is updated
        if ~exist('optimalBinSizeNat', 'var') && isfield(results, 'naturalistic') && isfield(results.naturalistic, 'optimalBinSize')
            optimalBinSizeNat = results.naturalistic.optimalBinSize;
        end
        aDataMatNat_dcc = neural_matrix_ms_to_frames(dataMat(:, aID), optimalBinSizeNat(a));
        numTimePoints_dcc = size(aDataMatNat_dcc, 1);
        stepSamples_dcc = round(dccStepSize / optimalBinSizeNat(a));
        winSamples_dcc = round(dccWindowSize / optimalBinSizeNat(a));
        numWindows_dcc = floor((numTimePoints_dcc - winSamples_dcc) / stepSamples_dcc) + 1;

        % Step 5-6: Apply PCA to binned data and project back to neural space
        if pcaFlag
            [coeff, score, ~, ~, explained, mu] = pca(aDataMatNat_dcc);
            forDim = find(cumsum(explained) > 30, 1);
            forDim = max(3, forDim);
            forDim = min(6, forDim);
            if pcaFirstFlag
                nDim = 1:forDim;
            else
                nDim = forDim+1:size(score, 2);
            end
            aDataMatNat_dcc = score(:,nDim) * coeff(:,nDim)' + mu;
        end

        % Step 7: Apply thresholding if needed
        if thresholdFlag
            aDataMatNat_dcc = round(sum(aDataMatNat_dcc, 2));
            threshSpikes = thresholdPct * median(aDataMatNat_dcc);
            aDataMatNat_dcc(aDataMatNat_dcc < threshSpikes) = 0;
        else
            aDataMatNat_dcc = round(sum(aDataMatNat_dcc, 2));
        end

        % Initialize arrays for dcc/kappa/decades (size by number of windows)
        dccNat{a} = nan(1, numWindows_dcc);
        kappaNat{a} = nan(1, numWindows_dcc);
        decadesNat{a} = nan(1, numWindows_dcc);
        startSNat_dcc{a} = nan(1, numWindows_dcc);

        % Step 8: Process each window for dcc and kappa
        for w = 1:numWindows_dcc
            startIdx = (w - 1) * stepSamples_dcc + 1;
            endIdx = startIdx + winSamples_dcc - 1;
            centerIdx = startIdx + floor((endIdx - startIdx)/2);
            startSNat_dcc{a}(w) = (startIdx + round(winSamples_dcc/2)-1) * optimalBinSizeNat(a);

            % Calculate population activity for this window
            wPopActivity = aDataMatNat_dcc(startIdx:endIdx);

            % Avalanche analysis for dcc and kappa
            % Find avalanches in the window
            zeroBins = find(wPopActivity == 0);
            if length(zeroBins) > 1 && any(diff(zeroBins)>1)
                % Create avalanche data
                asdfMat = rastertoasdf2(wPopActivity', optimalBinSizeNat(a)*1000, 'CBModel', 'Spikes', 'DS');
                Av = avprops(asdfMat, 'ratio', 'fingerprint');

                % Calculate avalanche parameters
                [tau, ~, tauC, ~, alpha, ~, paramSD, decades] = avalanche_log(Av, 0);

                % dcc (distance to criticality from avalanche analysis)
                dccNat{a}(w) = distance_to_criticality(tau, alpha, paramSD);

                % kappa (avalanche shape parameter)
                kappaNat{a}(w) = compute_kappa(Av.size);

                % decades (log10 of avalanche size range)
                decadesNat{a}(w) = decades;
            end
        end

        fprintf('Area %s dcc/kappa completed in %.1f minutes\n', areas{a}, toc/60);
    end
end % updateDccKappa

% ==============================================     Mark's Reach Data Analysis     ==============================================

if updateD2MrBr
    fprintf('\n=== Mark''s Reach Data d2/mrbr Analysis ===\n');

    % Step 1-2: Apply PCA to original data to determine nDim and project back to neural space
    fprintf('\n--- Step 1-2: PCA on original data if requested ---\n');
    reconstructedDataMatRea = cell(1, length(areas));
    for a = areasToTest
        aID = idListRea{a};
        thisDataMat = dataMatR(:, aID);

        if pcaFlag
            [coeff, score, ~, ~, explained, mu] = pca(thisDataMat);
            forDim = find(cumsum(explained) > 30, 1);
            forDim = max(3, forDim);
            forDim = min(6, forDim);
            if pcaFirstFlag
                fprintf('Area %s: Using PCA first %d dimensions\n', areas{a}, forDim);
                nDim = 1:forDim;
            else
                fprintf('Area %s: Using PCA last %d dimensions\n', areas{a}, size(score, 2) - forDim + 1);
                nDim = forDim+1:size(score, 2);
            end
            reconstructedDataMatRea{a} = score(:,nDim) * coeff(:,nDim)' + mu;
        else
            reconstructedDataMatRea{a} = thisDataMat;
        end
    end

    % Step 3: Find optimal parameters using reconstructed data
    fprintf('\n--- Step 3: Finding optimal parameters ---\n');
    optimalBinSizeRea = zeros(1, length(areas));
    optimalWindowSizeRea = zeros(1, length(areas));
    for a = areasToTest % 1:length(areas)
        thisDataMat = reconstructedDataMatRea{a};
        [optimalBinSizeRea(a), optimalWindowSizeRea(a)] = find_optimal_bin_and_window(thisDataMat, candidateFrameSizes, candidateWindowSizes, minSpikesPerBin, maxSpikesPerBin, minBinsPerWindow);
        fprintf('Area %s: optimal bin size = %.3f s, optimal window size = %.1f s\n', areas{a}, optimalBinSizeRea(a), optimalWindowSizeRea(a));
    end

    % Use optimal bin sizes for each area (no longer unified)
    % Set d2StepSize to optimal bin size for each area
    d2StepSizeRea = optimalBinSizeRea;
    d2WindowSizeRea = optimalWindowSizeRea;
    d2WindowSizeRea = windowSizes;

    fprintf('Using area-specific optimal parameters for reach data:\n');
    for a = areasToTest
        fprintf('  Area %s: bin size = %.3f s, window size = %.1f s, d2StepSize = %.3f s\n', ...
            areas{a}, optimalBinSizeRea(a), optimalWindowSizeRea(a), d2StepSizeRea(a));
    end

    % Initialize results for reach data
    mrBrRea = cell(1, length(areas));
    d2Rea = cell(1, length(areas));
    dccRea = cell(1, length(areas));
    kappaRea = cell(1, length(areas));
    startSRea = cell(1, length(areas));

    for a = areasToTest %1:length(areas)
        fprintf('\nProcessing area %s (Reach)...\n', areas{a});
        tic;

        aID = idListRea{a};
        stepSamples = round(d2StepSizeRea(a) / optimalBinSizeRea(a));
        winSamples = round(d2WindowSizeRea(a) / optimalBinSizeRea(a));

        % Step 4: Bin the original data using area-specific optimal bin size
        aDataMatRea = neural_matrix_ms_to_frames(dataMatR(:, aID), optimalBinSizeRea(a));
        numTimePoints = size(aDataMatRea, 1);
        numWindows = floor((numTimePoints - winSamples) / stepSamples) + 1;

        % Step 5-6: Apply PCA to binned data and project back to neural space
        if pcaFlag
            [coeff, score, ~, ~, explained, mu] = pca(aDataMatRea);
            forDim = find(cumsum(explained) > 30, 1);
            forDim = max(3, forDim);
            forDim = min(6, forDim);
            if pcaFirstFlag
                nDim = 1:forDim;
            else
                nDim = forDim+1:size(score, 2);
            end
            aDataMatRea = score(:,nDim) * coeff(:,nDim)' + mu;
        end

        % Step 7: Apply thresholding if needed
        % if thresholdFlag
        %     aDataMatRea = round(sum(aDataMatRea, 2));
        %     threshSpikes = thresholdPct * median(aDataMatRea);
        %     aDataMatRea(aDataMatRea < threshSpikes) = 0;
        %     fprintf('Area %s: Threshold = %.3f, Prop zeros = %.3f\n', areas{a}, threshSpikes, sum(aDataMatRea == 0) / length(aDataMatRea));
        % else
        aDataMatRea = round(sum(aDataMatRea, 2));
        % end

        % Initialize arrays for this area (size by number of windows)
        startSRea{a} = nan(1, numWindows);
        mrBrRea{a} = nan(1, numWindows);
        d2Rea{a} = nan(1, numWindows);

        % Step 8: Process each window for mrBr and d2
            kMax = round(10 / optimalBinSizeRea(a));
            warning('Using default kMax for br_mr_estimation')
        for w = 1:numWindows
            startIdx = (w - 1) * stepSamples + 1;
            endIdx = startIdx + winSamples - 1;
            centerIdx = startIdx + floor((endIdx - startIdx)/2);
            startSRea{a}(w) = (startIdx + round(winSamples/2)-1) * optimalBinSizeRea(a);

            % Extract window data and
            % Calculate population activity for MR and d2
            wPopActivity = aDataMatRea(startIdx:endIdx);

            % MR branching ratio
            % result = branching_ratio_mr_estimation(wPopActivity, kMax);
            result = branching_ratio_mr_estimation(wPopActivity);
            mrBrRea{a}(w) = result.branching_ratio;

            % d2 (distance to criticality from AR model)
            [varphi, ~] = myYuleWalker3(wPopActivity, pOrder);
            d2Val = getFixedPointDistance2(pOrder, critType, varphi);
            d2Rea{a}(w) = d2Val;
        end

        fprintf('Area %s completed in %.1f minutes\n', areas{a}, toc/60);
    end
end % updateD2MrBr


% Separate loop for dcc and kappa analysis (reach data)
if updateDccKappa
    fprintf('\n=== Reach Data dcc/kappa Analysis ===\n');
    for a = areasToTest
        fprintf('\nProcessing dcc/kappa for area %s (Reach)...\n', areas{a});
        tic;

        aID = idListRea{a};

        % Step 4: Bin the original data for dcc/kappa analysis
        % Ensure optimalBinSizeRea available when only dcc/kappa is updated
        if ~exist('optimalBinSizeRea', 'var') && isfield(results, 'naturalistic') && isfield(results.naturalistic, 'optimalBinSize')
            optimalBinSizeRea = results.naturalistic.optimalBinSize;
        end
        aDataMatRea_dcc = neural_matrix_ms_to_frames(dataMatR(:, aID), optimalBinSizeRea(a));
        numTimePoints_dcc = size(aDataMatRea_dcc, 1);
        stepSamples_dcc = round(dccStepSize / optimalBinSizeRea(a));
        winSamples_dcc = round(dccWindowSize / optimalBinSizeRea(a));
        numWindows_dcc = floor((numTimePoints_dcc - winSamples_dcc) / stepSamples_dcc) + 1;

        % Step 5-6: Apply PCA to binned data and project back to neural space
        if pcaFlag
            [coeff, score, ~, ~, explained, mu] = pca(aDataMatRea_dcc);
            forDim = find(cumsum(explained) > 30, 1);
            forDim = max(3, forDim);
            forDim = min(6, forDim);
            if pcaFirstFlag
                nDim = 1:forDim;
            else
                nDim = forDim+1:size(score, 2);
            end
            aDataMatRea_dcc = score(:,nDim) * coeff(:,nDim)' + mu;
        end

        % Step 7: Apply thresholding if needed
        if thresholdFlag
            aDataMatRea_dcc = round(sum(aDataMatRea_dcc, 2));
            threshSpikes = thresholdPct * median(aDataMatRea_dcc);
            aDataMatRea_dcc(aDataMatRea_dcc < threshSpikes) = 0;
        else
            aDataMatRea_dcc = round(sum(aDataMatRea_dcc, 2));
        end

        % Initialize arrays for dcc/kappa/decades (size by number of windows)
        dccRea{a} = nan(1, numWindows_dcc);
        kappaRea{a} = nan(1, numWindows_dcc);
        decadesRea{a} = nan(1, numWindows_dcc);
        startSRea_dcc{a} = nan(1, numWindows_dcc);

        % Step 8: Process each window for dcc and kappa
        for w = 1:numWindows_dcc
            startIdx = (w - 1) * stepSamples_dcc + 1;
            endIdx = startIdx + winSamples_dcc - 1;
            centerIdx = startIdx + floor((endIdx - startIdx)/2);
            startSRea_dcc{a}(w) = (startIdx + round(winSamples_dcc/2)-1) * optimalBinSizeRea(a);

            % Calculate population activity for this window
            wPopActivity = aDataMatRea_dcc(startIdx:endIdx);

            % Avalanche analysis for dcc and kappa
            % Find avalanches in the window
            zeroBins = find(wPopActivity == 0);
            if length(zeroBins) > 1 && any(diff(zeroBins)>1)
                % Create avalanche data
                asdfMat = rastertoasdf2(wPopActivity', optimalBinSizeRea(a)*1000, 'CBModel', 'Spikes', 'DS');
                Av = avprops(asdfMat, 'ratio', 'fingerprint');

                % Calculate avalanche parameters
                [tau, ~, tauC, ~, alpha, ~, paramSD, decades] = avalanche_log(Av, 0);

                % dcc (distance to criticality from avalanche analysis)
                dccRea{a}(w) = distance_to_criticality(tau, alpha, paramSD);

                % kappa (avalanche shape parameter)
                kappaRea{a}(w) = compute_kappa(Av.size);

                % decades (log10 of avalanche size range)
                decadesRea{a}(w) = decades;
            end
        end

        fprintf('Area %s dcc/kappa completed in %.1f minutes\n', areas{a}, toc/60);
    end
end % updateDccKappa

% ==============================================     Plotting Results     ==============================================

% Create comparison plots for each area with all measures
measures = {'mrBr', 'd2', 'dcc', 'kappa', 'decades'};
measureNames = {'MR Branching Ratio', 'Distance to Criticality (d2)', 'Distance to Criticality (dcc)', 'Kappa', 'Decades'};

% Colors for different measures
measureColors = {'k', 'b', 'r', [0 0.75 0], [0.5 0 0.5]};

% Create one figure per area
if makePlots
    for a = areasToTest
        figure(100 + a); clf;
        set(gcf, 'Position', monitorTwo);

        % Use tight_subplot for 5x1 layout
        ha = tight_subplot(5, 1, [0.05 0.02], [0.1 0.05], [0.1 0.05]);

        % Plot d2
        axes(ha(1));
        hold on;
        plot(startSNat{a}/60, d2Nat{a}, '-', 'Color', 'b', 'LineWidth', 2, 'MarkerSize', 4);
        plot(startSRea{a}/60, d2Rea{a}, '--', 'Color', 'b', 'LineWidth', 2, 'MarkerSize', 4);
        ylabel('Distance to Criticality (d2)', 'FontSize', 14);
        title(sprintf('%s - Distance to Criticality (d2)', areas{a}), 'FontSize', 14);
        legend({'Naturalistic', 'Reach'}, 'Location', 'best', 'FontSize', 14);
        grid on;
        set(gca, 'XTickLabel', [], 'FontSize', 14); % Remove x-axis labels for all but bottom subplot
        set(gca, 'YTickLabelMode', 'auto');  % Enable Y-axis labels
        xlim([opts.collectStart/60 (opts.collectStart+opts.collectFor)/60]);

        % Plot mrBr
        axes(ha(2));
        hold on;
        plot(startSNat{a}/60, mrBrNat{a}, '-', 'Color', 'k', 'LineWidth', 2, 'MarkerSize', 4);
        plot(startSRea{a}/60, mrBrRea{a}, '--', 'Color', 'k', 'LineWidth', 2, 'MarkerSize', 4);
        ylabel('MR Branching Ratio', 'FontSize', 14);
        title(sprintf('%s - MR Branching Ratio', areas{a}), 'FontSize', 14);
        legend({'Naturalistic', 'Reach'}, 'Location', 'best', 'FontSize', 14);
        grid on;
        set(gca, 'XTickLabel', [], 'FontSize', 14); % Remove x-axis labels for all but bottom subplot
        set(gca, 'YTickLabelMode', 'auto');  % Enable Y-axis labels
        xlim([opts.collectStart/60 (opts.collectStart+opts.collectFor)/60]);

        % Plot dcc
        axes(ha(3));
        hold on;
        plot(startSNat_dcc{a}/60, dccNat{a}, '-o', 'Color', 'r', 'LineWidth', 2, 'MarkerSize', 4);
        plot(startSRea_dcc{a}/60, dccRea{a}, '--s', 'Color', 'r', 'LineWidth', 2, 'MarkerSize', 4);
        ylabel('Distance to Criticality (dcc)', 'FontSize', 14);
        title(sprintf('%s - Distance to Criticality (dcc)', areas{a}), 'FontSize', 14);
        legend({'Naturalistic', 'Reach'}, 'Location', 'best', 'FontSize', 14);
        grid on;
        set(gca, 'XTickLabel', [], 'FontSize', 14); % Remove x-axis labels for all but bottom subplot
        set(gca, 'YTickLabelMode', 'auto');  % Enable Y-axis labels
        xlim([opts.collectStart/60 (opts.collectStart+opts.collectFor)/60]);

        % Plot kappa
        axes(ha(4));
        hold on;
        plot(startSNat_dcc{a}/60, kappaNat{a}, '-o', 'Color', [0 0.75 0], 'LineWidth', 2, 'MarkerSize', 4);
        plot(startSRea_dcc{a}/60, kappaRea{a}, '--s', 'Color', [0 0.75 0], 'LineWidth', 2, 'MarkerSize', 4);
        ylabel('Kappa', 'FontSize', 14);
        title(sprintf('%s - Kappa', areas{a}), 'FontSize', 14);
        legend({'Naturalistic', 'Reach'}, 'Location', 'best', 'FontSize', 14);
        grid on;
        set(gca, 'XTickLabel', [], 'FontSize', 14); % Remove x-axis labels for all but bottom subplot
        set(gca, 'YTickLabelMode', 'auto');  % Enable Y-axis labels
        xlim([opts.collectStart/60 (opts.collectStart+opts.collectFor)/60]);

        % Plot decades
        axes(ha(5));
        hold on;
        plot(startSNat_dcc{a}/60, decadesNat{a}, '-o', 'Color', [0.5 0 0.5], 'LineWidth', 2, 'MarkerSize', 4);
        plot(startSRea_dcc{a}/60, decadesRea{a}, '--s', 'Color', [0.5 0 0.5], 'LineWidth', 2, 'MarkerSize', 4);
        ylabel('Decades', 'FontSize', 14);
        title(sprintf('%s - Decades', areas{a}), 'FontSize', 14);
        legend({'Naturalistic', 'Reach'}, 'Location', 'best', 'FontSize', 14);
        grid on;
        xlabel('Minutes', 'FontSize', 14); % Only add xlabel to bottom subplot
        set(gca, 'YTickLabelMode', 'auto', 'FontSize', 14);  % Enable Y-axis labels
        set(gca, 'XTickLabelMode', 'auto');  % Enable Y-axis labels
        xlim([opts.collectStart/60 (opts.collectStart+opts.collectFor)/60]);

        sgtitle(sprintf('Criticality Measures Comparison - %s', areas{a}), 'FontSize', 14);

        % Save PNG using exportgraphics
        filename = fullfile(paths.dropPath, sprintf('criticality/criticality_comparison_%s_win%gs.png', areas{a}, slidingWindowSize));
        exportgraphics(gcf, filename, 'Resolution', 300);
        fprintf('Saved area plot to: %s\n', filename);

    end
end
% ==============================================     Correlation Analysis     ==============================================

if runCorrelation

    % Calculate correlations between d2 and mrBr (original sliding window timebase)
    fprintf('\n=== Correlation Analysis: d2 and mrBr ===\n');
    corrMatNat_d2mrBr = cell(1, length(areas));
    corrMatRea_d2mrBr = cell(1, length(areas));

    for a = areasToTest
        % Naturalistic data
        Xnat = [d2Nat{a}(:), mrBrNat{a}(:)];
        corrMatNat_d2mrBr{a} = corr(Xnat, 'Rows', 'pairwise');

        % Reach data
        Xrea = [d2Rea{a}(:), mrBrRea{a}(:)];
        corrMatRea_d2mrBr{a} = corr(Xrea, 'Rows', 'pairwise');
    end

    % Calculate correlations between dcc, kappa, and decades (dcc sliding window timebase)
    fprintf('\n=== Correlation Analysis: dcc, kappa, and decades ===\n');
    corrMatNat_dccKappaDecades = cell(1, length(areas));
    corrMatRea_dccKappaDecades = cell(1, length(areas));

    for a = areasToTest
        % Naturalistic data
        Xnat = [dccNat{a}(:), kappaNat{a}(:), decadesNat{a}(:)];
        corrMatNat_dccKappaDecades{a} = corr(Xnat, 'Rows', 'pairwise');

        % Reach data
        Xrea = [dccRea{a}(:), kappaRea{a}(:), decadesRea{a}(:)];
        corrMatRea_dccKappaDecades{a} = corr(Xrea, 'Rows', 'pairwise');
    end

    % Set colorbar scale for all correlation matrices
    cmin = -1;
    cmax = 1;

    % Plot correlation matrices for each area
    figure(201); clf;
    set(gcf, 'Position', monitorTwo);
    for a = areasToTest
        % d2/mrBr correlations
        subplot(2, length(areasToTest), a - areasToTest(1) + 1);
        imagesc(corrMatNat_d2mrBr{a});
        colorbar;
        title(sprintf('%s (Nat d2/mrBr)', areas{a}));
        xticks(1:2); yticks(1:2);
        xticklabels({'d2','mrBr'}); yticklabels({'d2','mrBr'});
        axis square;
        caxis([cmin cmax]); % Set consistent color axis

        subplot(2, length(areasToTest), length(areasToTest) + (a - areasToTest(1) + 1));
        imagesc(corrMatRea_d2mrBr{a});
        colorbar;
        title(sprintf('%s (Rea d2/mrBr)', areas{a}));
        xticks(1:2); yticks(1:2);
        xticklabels({'d2','mrBr'}); yticklabels({'d2','mrBr'});
        axis square;
        caxis([cmin cmax]); % Set consistent color axis
    end
    sgtitle('Correlation Matrices: d2 and mrBr');

    % Save PNG using exportgraphics
    filename = fullfile(paths.dropPath, sprintf('criticality/correlation_matrices_d2_mrbr_win%gs.png', slidingWindowSize));
    exportgraphics(gcf, filename, 'Resolution', 300);
    fprintf('Saved d2/mrBr correlation plot to: %s\n', filename);

    figure(202); clf;
    set(gcf, 'Position', monitorTwo);
    for a = areasToTest
        % dcc/kappa/decades correlations
        subplot(2, length(areasToTest), a - areasToTest(1) + 1);
        imagesc(corrMatNat_dccKappaDecades{a});
        colorbar;
        title(sprintf('%s (Nat dcc/kappa/decades)', areas{a}));
        xticks(1:3); yticks(1:3);
        xticklabels({'dcc','kappa','decades'}); yticklabels({'dcc','kappa','decades'});
        axis square;
        caxis([cmin cmax]); % Set consistent color axis

        subplot(2, length(areasToTest), length(areasToTest) + (a - areasToTest(1) + 1));
        imagesc(corrMatRea_dccKappaDecades{a});
        colorbar;
        title(sprintf('%s (Rea dcc/kappa/decades)', areas{a}));
        xticks(1:3); yticks(1:3);
        xticklabels({'dcc','kappa','decades'}); yticklabels({'dcc','kappa','decades'});
        axis square;
        caxis([cmin cmax]); % Set consistent color axis
    end
    sgtitle('Correlation Matrices: dcc, kappa, and decades');

    % Save PNG using exportgraphics
    filename = fullfile(paths.dropPath, sprintf('criticality/correlation_matrices_dcc_kappa_decades_win%gs.png', slidingWindowSize));
    exportgraphics(gcf, filename, 'Resolution', 300);
    fprintf('Saved dcc/kappa/decades correlation plot to: %s\n', filename);
end % runCorrelation

% ==============================================     Save Results     ==============================================

% Save all results (merge into existing if loaded)
if ~isfield(results, 'areas') || isempty(results.areas)
    results.areas = areas;
end
if ~exist('measures', 'var')
    measures = {'mrBr', 'd2', 'dcc', 'kappa', 'decades'};
end
if ~exist('measureNames', 'var')
    measureNames = {'MR Branching Ratio', 'Distance to Criticality (d2)', 'Distance to Criticality (dcc)', 'Kappa', 'Decades'};
end
results.measures = measures;
results.measureNames = measureNames;
results.naturalistic.collectStart = opts.collectStart;
results.naturalistic.collectFor = opts.collectFor;

% Naturalistic data results (conditional updates)
if exist('optimalBinSizeNat', 'var')
    results.naturalistic.optimalBinSize = optimalBinSizeNat;
end
if exist('d2StepSizeNat', 'var')
    results.reach.d2StepSize = d2StepSizeNat;
end
if exist('d2WindowSizeNat', 'var')
    results.naturalistic.d2WindowSize = d2WindowSizeNat;
end
if exist('optimalWindowSizeNat', 'var')
    results.naturalistic.optimalWindowSize = optimalWindowSizeNat;
end
if updateD2MrBr && exist('mrBrNat', 'var')
    results.naturalistic.mrBr = mrBrNat;
end
if updateD2MrBr && exist('d2Nat', 'var')
    results.naturalistic.d2 = d2Nat;
end
if updateD2MrBr && exist('startSNat', 'var')
    results.naturalistic.startS = startSNat;
end
if updateDccKappa && exist('dccNat', 'var')
    results.naturalistic.dcc = dccNat;
end
if updateDccKappa && exist('kappaNat', 'var')
    results.naturalistic.kappa = kappaNat;
end
if updateDccKappa && exist('decadesNat', 'var')
    results.naturalistic.decades = decadesNat;
end

% Reach data results (conditional updates)
if exist('optimalBinSizeRea', 'var')
    results.reach.optimalBinSize = optimalBinSizeRea;
end
if exist('d2StepSizeRea', 'var')
    results.reach.d2StepSize = d2StepSizeRea;
end
if exist('d2WindowSizeRea', 'var')
    results.naturalistic.d2WindowSize = d2WindowSizeRea;
end
if exist('optimalWindowSizeRea', 'var')
    results.reach.optimalWindowSize = optimalWindowSizeRea;
end
if updateD2MrBr && exist('mrBrRea', 'var')
    results.reach.mrBr = mrBrRea;
end
if updateD2MrBr && exist('d2Rea', 'var')
    results.reach.d2 = d2Rea;
end
if updateD2MrBr && exist('startSRea', 'var')
    results.reach.startS = startSRea;
end
if updateDccKappa && exist('dccRea', 'var')
    results.reach.dcc = dccRea;
end
if updateDccKappa && exist('kappaRea', 'var')
    results.reach.kappa = kappaRea;
end
if updateDccKappa && exist('decadesRea', 'var')
    results.reach.decades = decadesRea;
end

% Correlation matrices (conditional updates based on flags)
if runCorrelation && exist('corrMatNat_d2mrBr', 'var')
    results.naturalistic.corrMat = corrMatNat_d2mrBr;
end
if runCorrelation && exist('corrMatNat_dccKappaDecades', 'var')
    results.naturalistic.corrMatDccKappaDecades = corrMatNat_dccKappaDecades;
end
if runCorrelation && exist('corrMatRea_d2mrBr', 'var')
    results.reach.corrMat = corrMatRea_d2mrBr;
end
if runCorrelation && exist('corrMatRea_dccKappaDecades', 'var')
    results.reach.corrMatDccKappaDecades = corrMatRea_dccKappaDecades;
end

% Analysis parameters
results.params.pcaFlag = pcaFlag;
results.params.pcaFirstFlag = pcaFirstFlag;
results.params.nDim = nDim;
results.params.thresholdFlag = thresholdFlag;
results.params.thresholdPct = thresholdPct;
results.params.nShuffles = nShuffles;
results.params.pOrder = pOrder;
results.params.critType = critType;
results.params.dccStepSize = dccStepSize;
results.params.dccWindowSize = dccWindowSize;

% Save to file (window-size specific)
save(resultsPathWin, 'results');

fprintf('\nAnalysis complete! Results saved to %s\n', resultsPathWin); 






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



