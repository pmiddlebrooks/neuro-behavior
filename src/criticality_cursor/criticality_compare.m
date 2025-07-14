%%
% Criticality Comparison Script
% Compares different measures of criticality within each brain area
% Uses sliding window analysis on Mark's reach data and naturalistic data
% Measures: mrBr (MR branching ratio), d2 (distance to criticality), 
%           dcc (distance to criticality from avalanche analysis), kappa

%%
opts = neuro_behavior_options;
opts.minActTime = .16;
opts.collectStart = 0 * 60 * 60; % seconds
opts.minFiringRate = .05;
opts.frameSize = .001;

paths = get_paths;

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
opts.collectFor = 10 * 60; % seconds
get_standard_data

areas = {'M23', 'M56', 'DS', 'VS'};
idListNat = {idM23, idM56, idDS, idVS};

%% Mark's reach data
dataR = load(fullfile(paths.dropPath, 'reach_data/Y4_100623_Spiketimes_idchan_BEH.mat'));
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
nDim = 3;              % Number of PCA dimensions to use

% Threshold options
thresholdFlag = 1;     % Set to 1 to use threshold method
thresholdPct = 0.75;   % Threshold as percentage of median

% Optimal bin/window size search parameters
% candidateFrameSizes = [0.004, 0.01, 0.02, 0.05, .075, 0.1];
% candidateFrameSizes = [0.01, 0.02, .03 .04 0.05, .075, 0.1];
candidateFrameSizes = [.03 .04 0.05, .075, 0.1];
candidateWindowSizes = [30, 45, 60, 90, 120];
minSpikesPerBin = 5;
maxSpikesPerBin = 20;
minBinsPerWindow = 1000;

% Sliding window parameters
stepSize = 2; % seconds
nShuffles = 5;

% AR model parameters for d2 calculation
pOrder = 10;
critType = 2;

%% ==============================================     Naturalistic Data Analysis     ==============================================

areasToTest = 2:3;

fprintf('\n=== Naturalistic Data Analysis ===\n');

% Find optimal parameters for each area
optimalBinSizeNat = zeros(1, length(areas));
optimalWindowSizeNat = zeros(1, length(areas));
for a = areasToTest
    aID = idListNat{a};
    thisDataMat = dataMat(:, aID);
    [optimalBinSizeNat(a), optimalWindowSizeNat(a)] = ...
        find_optimal_bin_and_window(thisDataMat, candidateFrameSizes, candidateWindowSizes, minSpikesPerBin, maxSpikesPerBin, minBinsPerWindow);
    fprintf('Area %s: optimal bin size = %.3f s, optimal window size = %.1f s\n', areas{a}, optimalBinSizeNat(a), optimalWindowSizeNat(a));
end

% Find maximum bin size and corresponding maximum window size
[maxBinSizeNat, maxBinIdx] = max(optimalBinSizeNat);
% Find areas that have the maximum bin size
areasWithMaxBin = optimalBinSizeNat == maxBinSizeNat;
% Among those areas, find the maximum window size
maxWindowSizeNat = max(optimalWindowSizeNat(areasWithMaxBin));

fprintf('Using unified parameters for all areas: bin size = %.3f s, window size = %.1f s\n', maxBinSizeNat, maxWindowSizeNat);

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
    stepSamples = round(stepSize / maxBinSizeNat);
    winSamples = round(maxWindowSizeNat / maxBinSizeNat);
    
    % Bin the data
    aDataMatNat = neural_matrix_ms_to_frames(dataMat(:, aID), maxBinSizeNat);
    numTimePoints = size(aDataMatNat, 1);
    numWindows = floor((numTimePoints - winSamples) / stepSamples) + 1;
    
    % Apply PCA if requested
    if pcaFlag
        [coeff, score, ~, ~, explained, mu] = pca(aDataMatNat);
        forDim = find(cumsum(explained) > 30, 1);
        forDim = max(3, forDim);
        forDim = min(6, forDim);
        if pcaFirstFlag
            fprintf('Using PCA first %d dimensions\n', forDim);
            nDim = 1:forDim;
        else
            fprintf('Using PCA last %d dimensions\n', forDim);
            nDim = forDim+1:size(score, 2);
        end
        aDataMatNat = score(:,nDim) * coeff(:,nDim)' + mu;
    end
    
    % Initialize arrays for this area
    mrBrNat{a} = nan(1, numWindows);
    d2Nat{a} = nan(1, numWindows);
    dccNat{a} = nan(1, numWindows);
    kappaNat{a} = nan(1, numWindows);
    startSNat{a} = nan(1, numWindows);
    
    % Process each window
    for w = 1:numWindows
        startIdx = (w - 1) * stepSamples + 1;
        endIdx = startIdx + winSamples - 1;
        startSNat{a}(w) = (startIdx + round(winSamples/2)-1) * maxBinSizeNat; % the middle of the window counts as the time
        
        % Extract window data and        
        % Calculate population activity for MR and d2
        wPopActivity = round(sum(aDataMatNat(startIdx:endIdx, :), 2));
        
        % MR branching ratio
        kMax = round(10 / maxBinSizeNat);
        result = branching_ratio_mr_estimation(wPopActivity, kMax);
        mrBrNat{a}(w) = result.branching_ratio;
        
        % d2 (distance to criticality from AR model)
        [varphi, ~] = myYuleWalker3(wPopActivity, pOrder);
        d2Nat{a}(w) = getFixedPointDistance2(pOrder, critType, varphi);
        
        % Avalanche analysis for dcc and kappa
        % Apply threshold if requested
            wDataMatAv = wPopActivity;
        if thresholdFlag
            threshSpikes = thresholdPct * median(wPopActivity);
            wDataMatAv(wDataMatAv < threshSpikes) = 0;
        end
        % Find avalanches in the window       
        zeroBins = find(sum(wDataMatAv, 2) == 0);
        if length(zeroBins) > 1 && any(diff(zeroBins)>1)
            % Create avalanche data
            asdfMat = rastertoasdf2(wDataMatAv', maxBinSizeNat*1000, 'CBModel', 'Spikes', 'DS');
            Av = avprops(asdfMat, 'ratio', 'fingerprint');
            
            % Calculate avalanche parameters
            [tau, ~, tauC, ~, alpha, ~, paramSD, ~] = avalanche_log(Av, 0);
            
            % dcc (distance to criticality from avalanche analysis)
            dccNat{a}(w) = distance_to_criticality(tau, alpha, paramSD);
            
            % kappa (avalanche shape parameter)
            kappaNat{a}(w) = compute_kappa(Av.size);
        end
    end
    
    fprintf('Area %s completed in %.1f minutes\n', areas{a}, toc/60);
end

%% ==============================================     Mark's Reach Data Analysis     ==============================================

fprintf('\n=== Mark''s Reach Data Analysis ===\n');

% Find optimal parameters for each area
optimalBinSizeRea = zeros(1, length(areas));
optimalWindowSizeRea = zeros(1, length(areas));
for a = areasToTest % 1:length(areas)
    aID = idListRea{a};
    thisDataMat = dataMatR(:, aID);
    [optimalBinSizeRea(a), optimalWindowSizeRea(a)] = find_optimal_bin_and_window(thisDataMat, candidateFrameSizes, candidateWindowSizes, minSpikesPerBin, maxSpikesPerBin, minBinsPerWindow);
    fprintf('Area %s: optimal bin size = %.3f s, optimal window size = %.1f s\n', areas{a}, optimalBinSizeRea(a), optimalWindowSizeRea(a));
end

% Find maximum bin size and corresponding maximum window size
[maxBinSizeRea, maxBinIdx] = max(optimalBinSizeRea);
% Find areas that have the maximum bin size
areasWithMaxBin = optimalBinSizeRea == maxBinSizeRea;
% Among those areas, find the maximum window size
maxWindowSizeRea = max(optimalWindowSizeRea(areasWithMaxBin));

fprintf('Using unified parameters for all areas: bin size = %.3f s, window size = %.1f s\n', maxBinSizeRea, maxWindowSizeRea);

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
    stepSamples = round(stepSize / maxBinSizeRea);
    winSamples = round(maxWindowSizeRea / maxBinSizeRea);
    
    % Bin the data
    aDataMatRea = neural_matrix_ms_to_frames(dataMatR(:, aID), maxBinSizeRea);
    numTimePoints = size(aDataMatRea, 1);
    numWindows = floor((numTimePoints - winSamples) / stepSamples) + 1;
    
    % Apply PCA if requested
    if pcaFlag
        [coeff, score, ~, ~, explained, mu] = pca(aDataMatRea);
        forDim = find(cumsum(explained) > 30, 1);
        forDim = max(3, forDim);
        forDim = min(6, forDim);
        if pcaFirstFlag
            fprintf('Using PCA first %d dimensions\n', forDim);
            nDim = 1:forDim;
        else
            fprintf('Using PCA last %d dimensions\n', forDim);
            nDim = forDim+1:size(score, 2);
        end
        aDataMatRea = score(:,nDim) * coeff(:,nDim)' + mu;
    end
    
    % Initialize arrays for this area
    mrBrRea{a} = nan(1, numWindows);
    d2Rea{a} = nan(1, numWindows);
    dccRea{a} = nan(1, numWindows);
    kappaRea{a} = nan(1, numWindows);
    startSRea{a} = nan(1, numWindows);
    
    % Process each window
    for w = 1:numWindows
        startIdx = (w - 1) * stepSamples + 1;
        endIdx = startIdx + winSamples - 1;
        startSRea{a}(w) = (startIdx + round(winSamples/2)-1) * maxBinSizeRea;
        
        % Extract window data and        
        % Calculate population activity for MR and d2
        wPopActivity = round(sum(aDataMatRea(startIdx:endIdx, :), 2));
                       
        % MR branching ratio
        kMax = round(10 / maxBinSizeRea);
        result = branching_ratio_mr_estimation(wPopActivity, kMax);
        mrBrRea{a}(w) = result.branching_ratio;
        
        % d2 (distance to criticality from AR model)
        [varphi, ~] = myYuleWalker3(wPopActivity, pOrder);
        d2Rea{a}(w) = getFixedPointDistance2(pOrder, critType, varphi);
        
        % Avalanche analysis for dcc and kappa
        % Apply threshold if requested
            wDataMatAv = wPopActivity;
        if thresholdFlag
            threshSpikes = thresholdPct * median(wPopActivity);
            wDataMatAv(wDataMatAv < threshSpikes) = 0;
        end
        % Find avalanches in the window       
        zeroBins = find(sum(wDataMatAv, 2) == 0);
        if length(zeroBins) > 1 && any(diff(zeroBins)>1)
            % Create avalanche data
            asdfMat = rastertoasdf2(wDataMatAv', maxBinSizeRea*1000, 'CBModel', 'Spikes', 'DS');
            Av = avprops(asdfMat, 'ratio', 'fingerprint');
            
            % Calculate avalanche parameters
            [tau, ~, tauC, ~, alpha, ~, paramSD, ~] = avalanche_log(Av, 0);
            
            % dcc (distance to criticality from avalanche analysis)
            dccRea{a}(w) = distance_to_criticality(tau, alpha, paramSD);
            
            % kappa (avalanche shape parameter)
            kappaRea{a}(w) = compute_kappa(Av.size);
        end
    end
    
    fprintf('Area %s completed in %.1f minutes\n', areas{a}, toc/60);
end

%% ==============================================     Plotting Results     ==============================================

% Create comparison plots for each area with all measures
measures = {'mrBr', 'd2', 'dcc', 'kappa'};
measureNames = {'MR Branching Ratio', 'Distance to Criticality (d2)', 'Distance to Criticality (dcc)', 'Kappa'};

% Colors for different measures
measureColors = {'k', 'b', 'r', [0 0.75 0]};

% Create one figure per area
for a = 1:length(areas)
    figure(100 + a); clf;
    set(gcf, 'Position', monitorTwo);
    
    % Use tight_subplot for 4x1 layout
    ha = tight_subplot(4, 1, [0.05 0.02], [0.1 0.05], [0.1 0.05]);
    
    for m = 1:length(measures)
        axes(ha(m));
        hold on;
        
        % Get data for this measure
        switch measures{m}
            case 'mrBr'
                dataNat = mrBrNat{a};
                dataRea = mrBrRea{a};
            case 'd2'
                dataNat = d2Nat{a};
                dataRea = d2Rea{a};
            case 'dcc'
                dataNat = dccNat{a};
                dataRea = dccRea{a};
            case 'kappa'
                dataNat = kappaNat{a};
                dataRea = kappaRea{a};
        end
        
        % Plot naturalistic data
        plot(startSNat{a}/60, dataNat, '-o', 'Color', measureColors{m}, 'LineWidth', 2, 'MarkerSize', 4);
        
        % Plot reach data
        plot(startSRea{a}/60, dataRea, '--s', 'Color', measureColors{m}, 'LineWidth', 2, 'MarkerSize', 4);
        
        ylabel(measureNames{m});
        title(sprintf('%s - %s', areas{a}, measureNames{m}));
        legend({'Naturalistic', 'Reach'}, 'Location', 'best');
        grid on;
        
        % Only add xlabel to bottom subplot
        if m == length(measures)
            xlabel('Minutes');
        end
    end
    
    sgtitle(sprintf('Criticality Measures Comparison - %s', areas{a}));
end

%% ==============================================     Correlation Analysis     ==============================================

% Calculate correlations between measures for each area and dataset
fprintf('\n=== Correlation Analysis ===\n');

% Combine all measures for correlation analysis
allMeasuresNat = cell(1, length(areas));
allMeasuresRea = cell(1, length(areas));

for a = 1:length(areas)
    % Get minimum length across all measures for this area
    lengths = [length(mrBrNat{a}), length(d2Nat{a}), length(dccNat{a}), length(kappaNat{a})];
    minLen = min(lengths);
    
    % Naturalistic data
    allMeasuresNat{a} = [mrBrNat{a}(1:minLen); d2Nat{a}(1:minLen); dccNat{a}(1:minLen); kappaNat{a}(1:minLen)]';
    
    % Reach data
    lengths = [length(mrBrRea{a}), length(d2Rea{a}), length(dccRea{a}), length(kappaRea{a})];
    minLen = min(lengths);
    allMeasuresRea{a} = [mrBrRea{a}(1:minLen); d2Rea{a}(1:minLen); dccRea{a}(1:minLen); kappaRea{a}(1:minLen)]';
end

% Calculate correlation matrices
corrMatNat = cell(1, length(areas));
corrMatRea = cell(1, length(areas));

for a = 1:length(areas)
    % Remove rows with NaN values
    validIdx = ~any(isnan(allMeasuresNat{a}), 2);
    if sum(validIdx) > 10  % Need sufficient data points
        corrMatNat{a} = corrcoef(allMeasuresNat{a}(validIdx, :));
    else
        corrMatNat{a} = nan(4, 4);
    end
    
    validIdx = ~any(isnan(allMeasuresRea{a}), 2);
    if sum(validIdx) > 10
        corrMatRea{a} = corrcoef(allMeasuresRea{a}(validIdx, :));
    else
        corrMatRea{a} = nan(4, 4);
    end
end

% Plot correlation matrices
figure(200); clf;
set(gcf, 'Position', monitorTwo);

for a = 1:length(areas)
    % Naturalistic correlations
    subplot(2, 4, a);
    imagesc(corrMatNat{a});
    colorbar;
    title(sprintf('%s (Naturalistic)', areas{a}));
    xticks(1:4); yticks(1:4);
    xticklabels(measures); yticklabels(measures);
    axis square;
    
    % Reach correlations
    subplot(2, 4, a+4);
    imagesc(corrMatRea{a});
    colorbar;
    title(sprintf('%s (Reach)', areas{a}));
    xticks(1:4); yticks(1:4);
    xticklabels(measures); yticklabels(measures);
    axis square;
end

sgtitle('Correlation Matrices Between Criticality Measures');

%% ==============================================     Save Results     ==============================================

% Save all results
results = struct();
results.areas = areas;
results.measures = measures;
results.measureNames = measureNames;

% Naturalistic data results
results.naturalistic.mrBr = mrBrNat;
results.naturalistic.d2 = d2Nat;
results.naturalistic.dcc = dccNat;
results.naturalistic.kappa = kappaNat;
results.naturalistic.startS = startSNat;
results.naturalistic.optimalBinSize = optimalBinSizeNat;
results.naturalistic.optimalWindowSize = optimalWindowSizeNat;
results.naturalistic.unifiedBinSize = maxBinSizeNat;
results.naturalistic.unifiedWindowSize = maxWindowSizeNat;
results.naturalistic.corrMat = corrMatNat;

% Reach data results
results.reach.mrBr = mrBrRea;
results.reach.d2 = d2Rea;
results.reach.dcc = dccRea;
results.reach.kappa = kappaRea;
results.reach.startS = startSRea;
results.reach.optimalBinSize = optimalBinSizeRea;
results.reach.optimalWindowSize = optimalWindowSizeRea;
results.reach.unifiedBinSize = maxBinSizeRea;
results.reach.unifiedWindowSize = maxWindowSizeRea;
results.reach.corrMat = corrMatRea;

% Analysis parameters
results.params.pcaFlag = pcaFlag;
results.params.pcaFirstFlag = pcaFirstFlag;
results.params.nDim = nDim;
results.params.thresholdFlag = thresholdFlag;
results.params.thresholdPct = thresholdPct;
results.params.stepSize = stepSize;
results.params.nShuffles = nShuffles;
results.params.pOrder = pOrder;
results.params.critType = critType;

% Save to file
save(fullfile(paths.dropPath, 'criticality_compare_results.mat'), 'results');

fprintf('\nAnalysis complete! Results saved to criticality_compare_results.mat\n'); 






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



