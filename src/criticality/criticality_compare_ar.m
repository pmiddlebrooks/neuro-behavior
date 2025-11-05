%%
% Criticality Comparison Script
% Compares different measures of criticality within each brain area
% Uses sliding window analysis on Mark's reach data and naturalistic data
% Measures: mrBr (MR branching ratio), d2 (distance to criticality),
%
% Update controls:
%   loadExistingResults - load existing saved results to update selectively
%   updateD2MrBr        - run and update d2/mrBr analyses only if true
%   makePlots           - generate plots if true
%   runCorrelation      - compute correlation matrices if true

%%
paths = get_paths;

% =============================    Update/Run Flags    =============================
loadExistingResults = false;   % load existing results file to preserve untouched fields
updateD2MrBr = true;          % run d2/mrBr analyses
makePlots = true;            % create comparison plots
runCorrelation = true;       % compute correlation matrices

% Sliding window size
slidingWindowSize = 3;        % For d2, use a small window to try to optimize temporal resolution
% slidingWindowSize = 60;        % For d2, use a small window to try to optimize temporal resolution

% Determine save directory based on loaded data file name
% reachDataFile = fullfile(paths.dropPath, 'reach_data/Copy_of_Y4_100623_Spiketimes_idchan_BEH.mat');
reachDataFile = fullfile(paths.dropPath, 'reach_data/AB6_03-Apr-2025 13_34_09_NeuroBeh.mat');

[~, dataBaseName, ~] = fileparts(reachDataFile);
saveDir = fullfile(paths.dropPath, 'reach_data', dataBaseName);
if ~exist(saveDir, 'dir')
    mkdir(saveDir);
end

% Load existing results if requested
resultsPathD2MrBr = fullfile(saveDir, sprintf('criticality_compare_ar_win%d.mat', slidingWindowSize));
results = struct();
if loadExistingResults
    if exist(resultsPathD2MrBr, 'file')
        loaded = load(resultsPathD2MrBr);
    else
        loaded = struct();
    end
    if isfield(loaded, 'results')
        results = loaded.results;
    end
end

%% ==============================================     Data Loading     ==============================================

% Naturalistic data
getDataType = 'spikes';
opts.firingRateCheckTime = 5 * 60;
opts.collectStart = 0 * 60; % seconds
opts.collectEnd = 45 * 60; % seconds
opts.minFiringRate = .05;
get_standard_data

areas = {'M23', 'M56', 'DS', 'VS'};
idListNat = {idM23, idM56, idDS, idVS};

%
% Mark's reach data
% dataR = load(fullfile(paths.dropPath, 'reach_data/Y4_100623_Spiketimes_idchan_BEH.mat'));
dataR = load(reachDataFile);
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
candidateFrameSizes = [.02 .03 .04 0.05, .075, 0.1 .15]; % seconds
candidateWindowSizes = [30, 45, 60, 90, 120]; % seconds
windowSizes = repmat(slidingWindowSize, 1, 4); % For d2, use a small window to try to optimize temporal resolution
minSpikesPerBin = 3;
maxSpikesPerBin = 50;
minBinsPerWindow = 1000;


% For randomized comparison (for significance testing)
nShuffles = 5;

% AR model parameters for d2 calculation
pOrder = 10;
critType = 2;
d2StepSize = .02;


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




% ==============================================     Save Results     ==============================================

% Save all results (merge into existing if loaded)
if ~isfield(results, 'areas') || isempty(results.areas)
    results.areas = areas;
end
if ~exist('measures', 'var')
    measures = {'mrBr', 'd2'};
end
if ~exist('measureNames', 'var')
    measureNames = {'MR Branching Ratio', 'Distance to Criticality (d2)'};
end
results.measures = measures;
results.measureNames = measureNames;
results.naturalistic.collectStart = opts.collectStart;
results.naturalistic.collectFor = opts.collectEnd;

% Naturalistic data results (conditional updates)
if exist('optimalBinSizeNat', 'var')
    results.naturalistic.optimalBinSize = optimalBinSizeNat;
end
if exist('d2StepSizeNat', 'var')
    results.naturalistic.d2StepSize = d2StepSizeNat;
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

% Reach data results (conditional updates)
if exist('optimalBinSizeRea', 'var')
    results.reach.optimalBinSize = optimalBinSizeRea;
end
if exist('d2StepSizeRea', 'var')
    results.reach.d2StepSize = d2StepSizeRea;
end
if exist('d2WindowSizeRea', 'var')
    results.reach.d2WindowSize = d2WindowSizeRea;
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

% Correlation matrices (conditional updates based on flags)
if runCorrelation && exist('corrMatNat_d2mrBr', 'var')
    results.naturalistic.corrMat = corrMatNat_d2mrBr;
end
if runCorrelation && exist('corrMatRea_d2mrBr', 'var')
    results.reach.corrMat = corrMatRea_d2mrBr;
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

% Save to file (in data-specific folder)
save(resultsPathD2MrBr, 'results');

fprintf('\nD2/MrBr analysis complete! Results saved to %s\n', resultsPathD2MrBr);
