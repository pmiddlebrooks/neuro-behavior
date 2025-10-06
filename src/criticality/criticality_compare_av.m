%%
% Criticality Avalanche Analysis Script
% Analyzes dcc (distance to criticality from avalanche analysis) and kappa measures
% Uses sliding window analysis on Mark's reach data and naturalistic data
% Measures: dcc (distance to criticality from avalanche analysis), kappa
%
% Update controls:
%   loadExistingResults - load existing saved results to update selectively
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
loadExistingResults = false;   % load existing results file to preserve untouched fields
updateDccKappa = true;        % run dcc/kappa analyses
makePlots = true;             % create comparison plots
runCorrelation = false;       % compute correlation matrices

% Sliding window and step size
slidingWindowSize = 2 * 60; % seconds - user specified
avStepSize = 20; %round(.5 * 60); % seconds - user specified

% Determine save directory based on loaded data file name
reachDataFile = fullfile(paths.dropPath, 'reach_data/Copy_of_Y4_100623_Spiketimes_idchan_BEH.mat');
[~, dataBaseName, ~] = fileparts(reachDataFile);
saveDir = fullfile(paths.dropPath, 'reach_data', dataBaseName);
if ~exist(saveDir, 'dir')
    mkdir(saveDir);
end

% Load existing results if requested
resultsPathAvalanche = fullfile(saveDir, sprintf('criticality_compare_av_win%d_step%d.mat', slidingWindowSize, avStepSize));
results = struct();
if loadExistingResults
    if exist(resultsPathAvalanche, 'file')
        loaded = load(resultsPathAvalanche);
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
opts.collectFor = 45 * 60; % seconds
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


%% ==============================================     Naturalistic Data Analysis     ==============================================

areasToTest = 1:4;

if updateDccKappa
    fprintf('\n=== Naturalistic Data dcc/kappa Analysis ===\n');
    
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
        stepSamples_dcc = round(avStepSize / optimalBinSizeNat(a));
        winSamples_dcc = round(slidingWindowSize / optimalBinSizeNat(a));
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

if updateDccKappa
    fprintf('\n=== Reach Data dcc/kappa Analysis ===\n');
    
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
    for a = areasToTest
        thisDataMat = reconstructedDataMatRea{a};
        [optimalBinSizeRea(a), optimalWindowSizeRea(a)] = find_optimal_bin_and_window(thisDataMat, candidateFrameSizes, candidateWindowSizes, minSpikesPerBin, maxSpikesPerBin, minBinsPerWindow);
        fprintf('Area %s: optimal bin size = %.3f s, optimal window size = %.1f s\n', areas{a}, optimalBinSizeRea(a), optimalWindowSizeRea(a));
    end
    
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
        stepSamples_dcc = round(avStepSize / optimalBinSizeRea(a));
        winSamples_dcc = round(slidingWindowSize / optimalBinSizeRea(a));
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

%% ==============================================     Save Results     ==============================================

% Save all results (merge into existing if loaded)
if ~isfield(results, 'areas') || isempty(results.areas)
    results.areas = areas;
end
if ~exist('measures', 'var')
    measures = {'dcc', 'kappa', 'decades'};
end
if ~exist('measureNames', 'var')
    measureNames = {'Distance to Criticality (dcc)', 'Kappa', 'Decades'};
end
results.measures = measures;
results.measureNames = measureNames;
results.naturalistic.collectStart = opts.collectStart;
results.naturalistic.collectFor = opts.collectFor;

% Naturalistic data results (conditional updates)
if exist('optimalBinSizeNat', 'var')
    results.naturalistic.optimalBinSize = optimalBinSizeNat;
end
if exist('optimalWindowSizeNat', 'var')
    results.naturalistic.optimalWindowSize = optimalWindowSizeNat;
end
if exist('slidingWindowSize', 'var')
    results.naturalistic.slidingWindowSize = slidingWindowSize;
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
if updateDccKappa && exist('startSNat_dcc', 'var')
    results.naturalistic.startS_dcc = startSNat_dcc;
end

% Reach data results (conditional updates)
if exist('optimalBinSizeRea', 'var')
    results.reach.optimalBinSize = optimalBinSizeRea;
end
if exist('optimalWindowSizeRea', 'var')
    results.reach.optimalWindowSize = optimalWindowSizeRea;
end
if exist('slidingWindowSize', 'var')
    results.naturalistic.slidingWindowSize = slidingWindowSize;
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
if updateDccKappa && exist('startSRea_dcc', 'var')
    results.reach.startS_dcc = startSRea_dcc;
end

% Correlation matrices (conditional updates based on flags)
if runCorrelation && exist('corrMatNat_dccKappaDecades', 'var')
    results.naturalistic.corrMatDccKappaDecades = corrMatNat_dccKappaDecades;
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
results.params.dccStepSize = avStepSize;
results.params.slidingWindowSize = slidingWindowSize;

% Save to file (in data-specific folder)
save(resultsPathAvalanche, 'results');

fprintf('\nAvalanche analysis complete! Results saved to %s\n', resultsPathAvalanche);

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
