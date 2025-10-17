%%
% Criticality Sliding Window Avalanche Analysis Script (dcc + kappa)
% Unified script for analyzing both reach data and naturalistic data
% Analyzes data using sliding window avalanche approach; saves results to data-specific folders

paths = get_paths;

% =============================    Configuration    =============================
% Data type selection
dataType = 'reach';  % 'reach' or 'naturalistic'

% Sliding window and step size (seconds)
slidingWindowSize = 120;  % seconds - user specified
avStepSize = 27;          % seconds - user specified

% Flags
loadExistingResults = false;
makePlots = true;

% Analysis parameters
minSpikesPerBin = 3;
maxSpikesPerBin = 50;
minBinsPerWindow = 1000;

% Areas to analyze
areasToTest = 1:4;

% PCA options
pcaFlag = 0;           % Set to 1 to use PCA
pcaFirstFlag = 1;      % Use first nDim dimensions if 1, last nDim if 0
nDim = 4;              % Number of PCA dimensions to use

% Threshold options
thresholdFlag = 1;     % Set to 1 to use threshold method
thresholdPct = 0.75;   % Threshold as percentage of median

% Optimal bin/window size search parameters
candidateFrameSizes = [.02 .03 .04 0.05, .075, 0.1 .15];
candidateWindowSizes = [30, 45, 60, 90, 120];

% =============================    Data Loading    =============================
fprintf('\n=== Loading %s data ===\n', dataType);

if strcmp(dataType, 'reach')
    % Load reach data
    reachDataFile = fullfile(paths.reachDataPath, 'Y4_06-Oct-2023 14_14_53_NeuroBeh.mat');
    % reachDataFile = fullfile(paths.reachDataPath, 'makeSpikes.mat');
    
    [~, dataBaseName, ~] = fileparts(reachDataFile);
    saveDir = fullfile(paths.dropPath, 'reach_task/results', dataBaseName);
    if ~exist(saveDir, 'dir'); mkdir(saveDir); end
    
    resultsPath = fullfile(saveDir, sprintf('criticality_sliding_window_av_win%d_step%d.mat', slidingWindowSize, avStepSize));
    
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
    resultsPath = fullfile(saveDir, sprintf('criticality_sliding_window_av_win%d_step%d.mat', slidingWindowSize, avStepSize));
    
else
    error('Invalid dataType. Must be ''reach'' or ''naturalistic''');
end

% =============================    Analysis    =============================
fprintf('\n=== %s Data Avalanche Analysis ===\n', dataType);

% Adjust areasToTest based on which areas have data
areasToTest = areasToTest(~cellfun(@isempty, idList));

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

% Filter areas based on valid optimal bin sizes
validMask = isfinite(optimalBinSize(areasToTest)) & (optimalBinSize(areasToTest) > 0);
areasToTest = areasToTest(validMask);

% Initialize results
[dcc, kappa, decades, startS] = deal(cell(1, length(areas)));

for a = areasToTest
    fprintf('\nProcessing area %s (%s)...\n', areas{a}, dataType); 
    tic;
    aID = idList{a};
    
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

    % Step 7: Apply thresholding if needed
    if thresholdFlag
        aDataMat_dcc = round(sum(aDataMat_dcc, 2));
        threshSpikes = thresholdPct * median(aDataMat_dcc);
        aDataMat_dcc(aDataMat_dcc < threshSpikes) = 0;
    else
        aDataMat_dcc = round(sum(aDataMat_dcc, 2));
    end

    % Initialize arrays for dcc/kappa/decades (size by number of windows)
    dcc{a} = nan(1, numWindows_dcc);
    kappa{a} = nan(1, numWindows_dcc);
    decades{a} = nan(1, numWindows_dcc);
    startS{a} = nan(1, numWindows_dcc);

    % Step 8: Process each window for dcc and kappa
    for w = 1:numWindows_dcc
        startIdx = (w - 1) * stepSamples_dcc + 1;
        endIdx = startIdx + winSamples_dcc - 1;
        startS{a}(w) = (startIdx + round(winSamples_dcc/2)-1) * optimalBinSize(a);

        % Calculate population activity for this window
        wPopActivity = aDataMat_dcc(startIdx:endIdx);

        % Avalanche analysis for dcc and kappa
        % Find avalanches in the window
        zeroBins = find(wPopActivity == 0);
        if length(zeroBins) > 1 && any(diff(zeroBins)>1)
            % Create avalanche data
            asdfMat = rastertoasdf2(wPopActivity', optimalBinSize(a)*1000, 'CBModel', 'Spikes', 'DS');
            Av = avprops(asdfMat, 'ratio', 'fingerprint');

            % Calculate avalanche parameters
            [tau, ~, tauC, ~, alpha, ~, paramSD, decades_val] = avalanche_log(Av, 0);

            % dcc (distance to criticality from avalanche analysis)
            dcc{a}(w) = distance_to_criticality(tau, alpha, paramSD);

            % kappa (avalanche shape parameter)
            kappa{a}(w) = compute_kappa(Av.size);

            % decades (log10 of avalanche size range)
            decades{a}(w) = decades_val;
        end
    end

    fprintf('Area %s completed in %.1f minutes\n', areas{a}, toc/60);
end

% =============================    Save Results    =============================
results = struct(); 
results.dataType = dataType;
results.areas = areas; 
results.dcc = dcc; 
results.kappa = kappa; 
results.decades = decades;
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

save(resultsPath, 'results'); 
fprintf('Saved %s dcc/kappa to %s\n', dataType, resultsPath);

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
    
    % Avalanche analysis plots: 3 rows (dcc, kappa, decades) x num areas (columns)
    figure(901); clf; 
    set(gcf, 'Position', targetPos);
    
    for idx = 1:length(areasToTest)
        a = areasToTest(idx);
        
        % dcc (top row)
        subplot(3, length(areasToTest), idx);
        hold on;
        plot(startS{a}, dcc{a}, '-', 'Color', [1 0 0], 'LineWidth', 2);
        title(sprintf('%s - dcc', areas{a})); 
        xlabel('Time (s)'); ylabel('dcc'); grid on;
        
        % kappa (middle row)
        subplot(3, length(areasToTest), length(areasToTest) + idx);
        hold on;
        plot(startS{a}, kappa{a}, '-', 'Color', [0 0.6 0], 'LineWidth', 2);
        title(sprintf('%s - kappa', areas{a})); 
        xlabel('Time (s)'); ylabel('kappa'); grid on;
        
        % decades (bottom row)
        subplot(3, length(areasToTest), 2*length(areasToTest) + idx);
        hold on;
        plot(startS{a}, decades{a}, '-', 'Color', [0.6 0 0.6], 'LineWidth', 2);
        title(sprintf('%s - decades', areas{a})); 
        xlabel('Time (s)'); ylabel('decades'); grid on;
    end
    
    sgtitle(sprintf('%s dcc (top), kappa (mid), decades (bottom) - win=%gs, step=%gs', dataType, slidingWindowSize, avStepSize));
    exportgraphics(gcf, fullfile(saveDir, sprintf('criticality_%s_av_win%d_step%d.png', dataType, slidingWindowSize, avStepSize)), 'Resolution', 300);
    fprintf('Saved %s avalanche plots to: %s\n', dataType, fullfile(saveDir, sprintf('criticality_%s_av_win%d_step%d.png', dataType, slidingWindowSize, avStepSize)));
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
