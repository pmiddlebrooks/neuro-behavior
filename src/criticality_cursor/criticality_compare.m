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
opts.collectFor = 30 * 60; % seconds
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

% Separate parameters for dcc/kappa analysis
dccBinSize = 0.05; % seconds - user specified
dccStepSize = round(2.5 * 60); % seconds - user specified  
dccWindowSize = 5 * 60; % seconds - user specified

%% ==============================================     Naturalistic Data Analysis     ==============================================

areasToTest = 2:4;

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
    
    % Initialize arrays for this area
    startSNat{a} = nan(1, numTimePoints);
    mrBrNat{a} = nan(1, numTimePoints);
    d2Nat{a} = nan(1, numTimePoints);
    
    % Process each window for mrBr and d2
    for w = 1:numWindows
        startIdx = (w - 1) * stepSamples + 1;
        endIdx = startIdx + winSamples - 1;
        centerIdx = startIdx + floor((endIdx - startIdx)/2);
        startSNat{a}(centerIdx) = (startIdx + round(winSamples/2)-1) * maxBinSizeNat; % the middle of the window counts as the time
        
        % Extract window data and        
        % Calculate population activity for MR and d2
        wPopActivity = round(sum(aDataMatNat(startIdx:endIdx, :), 2));
        
        % MR branching ratio
        kMax = round(10 / maxBinSizeNat);
        result = branching_ratio_mr_estimation(wPopActivity, kMax);
        mrBrNat{a}(centerIdx) = result.branching_ratio;
        
        % d2 (distance to criticality from AR model)
        [varphi, ~] = myYuleWalker3(wPopActivity, pOrder);
        d2Val = getFixedPointDistance2(pOrder, critType, varphi);
        d2Nat{a}(centerIdx) = d2Val;
    end
    
    fprintf('Area %s completed in %.1f minutes\n', areas{a}, toc/60);
end

% Separate loop for dcc and kappa analysis
fprintf('\n=== Naturalistic Data dcc/kappa Analysis ===\n');
for a = areasToTest
    fprintf('\nProcessing dcc/kappa for area %s (Naturalistic)...\n', areas{a});
    tic;
    
    aID = idListNat{a};
    
    % Bin the data for dcc/kappa analysis
    aDataMatNat_dcc = neural_matrix_ms_to_frames(dataMat(:, aID), dccBinSize);
    numTimePoints_dcc = size(aDataMatNat_dcc, 1);
    stepSamples_dcc = round(dccStepSize / dccBinSize);
    winSamples_dcc = round(dccWindowSize / dccBinSize);
    numWindows_dcc = floor((numTimePoints_dcc - winSamples_dcc) / stepSamples_dcc) + 1;
    
    % Initialize arrays for dcc/kappa
    dccNat{a} = nan(1, numTimePoints_dcc);
    kappaNat{a} = nan(1, numTimePoints_dcc);
    startSNat_dcc{a} = nan(1, numWindows_dcc);
    
    % Process each window for dcc and kappa
    for w = 1:numWindows_dcc
        startIdx = (w - 1) * stepSamples_dcc + 1;
        endIdx = startIdx + winSamples_dcc - 1;
        centerIdx = startIdx + floor((endIdx - startIdx)/2);
        startSNat_dcc{a}(w) = (startIdx + round(winSamples_dcc/2)-1) * dccBinSize;
        
        % Calculate population activity for this window
        wPopActivity = round(sum(aDataMatNat_dcc(startIdx:endIdx, :), 2));
        
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
            asdfMat = rastertoasdf2(wDataMatAv', dccBinSize*1000, 'CBModel', 'Spikes', 'DS');
            Av = avprops(asdfMat, 'ratio', 'fingerprint');
            
            % Calculate avalanche parameters
            [tau, ~, tauC, ~, alpha, ~, paramSD, ~] = avalanche_log(Av, 0);
            
            % dcc (distance to criticality from avalanche analysis)
            dccNat{a}(centerIdx) = distance_to_criticality(tau, alpha, paramSD);
            
            % kappa (avalanche shape parameter)
            kappaNat{a}(centerIdx) = compute_kappa(Av.size);
        end
    end
    
    fprintf('Area %s dcc/kappa completed in %.1f minutes\n', areas{a}, toc/60);
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
    
    % Initialize arrays for this area
    startSRea{a} = nan(1, numTimePoints);
    mrBrRea{a} = nan(1, numTimePoints);
    d2Rea{a} = nan(1, numTimePoints);
    
    % Process each window for mrBr and d2
    for w = 1:numWindows
        startIdx = (w - 1) * stepSamples + 1;
        endIdx = startIdx + winSamples - 1;
        centerIdx = startIdx + floor((endIdx - startIdx)/2);
        startSRea{a}(centerIdx) = (startIdx + round(winSamples/2)-1) * maxBinSizeRea;
        
        % Extract window data and        
        % Calculate population activity for MR and d2
        wPopActivity = round(sum(aDataMatRea(startIdx:endIdx, :), 2));
                       
        % MR branching ratio
        kMax = round(10 / maxBinSizeRea);
        result = branching_ratio_mr_estimation(wPopActivity, kMax);
        mrBrRea{a}(centerIdx) = result.branching_ratio;
        
        % d2 (distance to criticality from AR model)
        [varphi, ~] = myYuleWalker3(wPopActivity, pOrder);
        d2Val = getFixedPointDistance2(pOrder, critType, varphi);
        d2Rea{a}(centerIdx) = d2Val;
    end
    
    fprintf('Area %s completed in %.1f minutes\n', areas{a}, toc/60);
end

% Separate loop for dcc and kappa analysis (reach data)
fprintf('\n=== Reach Data dcc/kappa Analysis ===\n');
for a = areasToTest
    fprintf('\nProcessing dcc/kappa for area %s (Reach)...\n', areas{a});
    tic;
    
    aID = idListRea{a};
    
    % Bin the data for dcc/kappa analysis
    aDataMatRea_dcc = neural_matrix_ms_to_frames(dataMatR(:, aID), dccBinSize);
    numTimePoints_dcc = size(aDataMatRea_dcc, 1);
    stepSamples_dcc = round(dccStepSize / dccBinSize);
    winSamples_dcc = round(dccWindowSize / dccBinSize);
    numWindows_dcc = floor((numTimePoints_dcc - winSamples_dcc) / stepSamples_dcc) + 1;
    
    % Initialize arrays for dcc/kappa
    dccRea{a} = nan(1, numTimePoints_dcc);
    kappaRea{a} = nan(1, numTimePoints_dcc);
    startSRea_dcc{a} = nan(1, numWindows_dcc);
    
    % Process each window for dcc and kappa
    for w = 1:numWindows_dcc
        startIdx = (w - 1) * stepSamples_dcc + 1;
        endIdx = startIdx + winSamples_dcc - 1;
        centerIdx = startIdx + floor((endIdx - startIdx)/2);
        startSRea_dcc{a}(w) = (startIdx + round(winSamples_dcc/2)-1) * dccBinSize;
        
        % Calculate population activity for this window
        wPopActivity = round(sum(aDataMatRea_dcc(startIdx:endIdx, :), 2));
        
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
            asdfMat = rastertoasdf2(wDataMatAv', dccBinSize*1000, 'CBModel', 'Spikes', 'DS');
            Av = avprops(asdfMat, 'ratio', 'fingerprint');
            
            % Calculate avalanche parameters
            [tau, ~, tauC, ~, alpha, ~, paramSD, ~] = avalanche_log(Av, 0);
            
            % dcc (distance to criticality from avalanche analysis)
            dccRea{a}(centerIdx) = distance_to_criticality(tau, alpha, paramSD);
            
            % kappa (avalanche shape parameter)
            kappaRea{a}(centerIdx) = compute_kappa(Av.size);
        end
    end
    
    fprintf('Area %s dcc/kappa completed in %.1f minutes\n', areas{a}, toc/60);
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
    
    % Plot mrBr
    axes(ha(1));
    hold on;
    validIdx = ~isnan(mrBrNat{a});
    plot(startSNat{a}(validIdx)/60, mrBrNat{a}(validIdx), '-', 'Color', 'k', 'LineWidth', 2, 'MarkerSize', 4);
    validIdx = ~isnan(mrBrRea{a});
    plot(startSRea{a}(validIdx)/60, mrBrRea{a}(validIdx), '--', 'Color', 'k', 'LineWidth', 2, 'MarkerSize', 4);
    ylabel('MR Branching Ratio');
    title(sprintf('%s - MR Branching Ratio', areas{a}));
    legend({'Naturalistic', 'Reach'}, 'Location', 'best');
    grid on;
    set(gca, 'XTickLabel', []); % Remove x-axis labels for all but bottom subplot
    set(gca, 'YTickLabelMode', 'auto');  % Enable Y-axis labels
    
    % Plot d2
    axes(ha(2));
    hold on;
    validIdx = ~isnan(d2Nat{a});
    plot(startSNat{a}(validIdx)/60, d2Nat{a}(validIdx), '-o', 'Color', 'b', 'LineWidth', 2, 'MarkerSize', 4);
    validIdx = ~isnan(d2Rea{a});
    plot(startSRea{a}(validIdx)/60, d2Rea{a}(validIdx), '--s', 'Color', 'b', 'LineWidth', 2, 'MarkerSize', 4);
    ylabel('Distance to Criticality (d2)');
    title(sprintf('%s - Distance to Criticality (d2)', areas{a}));
    legend({'Naturalistic', 'Reach'}, 'Location', 'best');
    grid on;
    set(gca, 'XTickLabel', []); % Remove x-axis labels for all but bottom subplot
    set(gca, 'YTickLabelMode', 'auto');  % Enable Y-axis labels
    
    % Plot dcc
    axes(ha(3));
    hold on;
    validIdx = ~isnan(dccNat{a});
    plot(startSNat_dcc{a}/60, dccNat{a}, '-o', 'Color', 'r', 'LineWidth', 2, 'MarkerSize', 4);
    validIdx = ~isnan(dccRea{a});
    plot(startSRea_dcc{a}/60, dccRea{a}, '--s', 'Color', 'r', 'LineWidth', 2, 'MarkerSize', 4);
    ylabel('Distance to Criticality (dcc)');
    title(sprintf('%s - Distance to Criticality (dcc)', areas{a}));
    legend({'Naturalistic', 'Reach'}, 'Location', 'best');
    grid on;
    set(gca, 'XTickLabel', []); % Remove x-axis labels for all but bottom subplot
     set(gca, 'YTickLabelMode', 'auto');  % Enable Y-axis labels
   
    % Plot kappa
    axes(ha(4));
    hold on;
    validIdx = ~isnan(kappaNat{a});
    plot(startSNat_dcc{a}/60, kappaNat{a}, '-o', 'Color', [0 0.75 0], 'LineWidth', 2, 'MarkerSize', 4);
    validIdx = ~isnan(kappaRea{a});
    plot(startSRea_dcc{a}/60, kappaRea{a}, '--s', 'Color', [0 0.75 0], 'LineWidth', 2, 'MarkerSize', 4);
    ylabel('Kappa');
    title(sprintf('%s - Kappa', areas{a}));
    legend({'Naturalistic', 'Reach'}, 'Location', 'best');
    grid on;
    xlabel('Minutes'); % Only add xlabel to bottom subplot
    set(gca, 'YTickLabelMode', 'auto');  % Enable Y-axis labels
    set(gca, 'XTickLabelMode', 'auto');  % Enable Y-axis labels
    
    sgtitle(sprintf('Criticality Measures Comparison - %s', areas{a}));
end

%% ==============================================     Correlation Analysis     ==============================================

% Calculate correlations between d2 and mrBr (original sliding window timebase)
fprintf('\n=== Correlation Analysis: d2 and mrBr ===\n');
corrMatNat_d2mrBr = cell(1, length(areas));
corrMatRea_d2mrBr = cell(1, length(areas));

for a = areasToTest
    % Get minimum length for d2 and mrBr
    minLenNat = min(length(d2Nat{a}), length(mrBrNat{a}));
    minLenRea = min(length(d2Rea{a}), length(mrBrRea{a}));
    
    % Naturalistic data
    Xnat = [d2Nat{a}(1:minLenNat); mrBrNat{a}(1:minLenNat)]';
    validIdx = ~any(isnan(Xnat), 2);
    if sum(validIdx) > 10
        corrMatNat_d2mrBr{a} = corrcoef(Xnat(validIdx, :));
    else
        corrMatNat_d2mrBr{a} = nan(2, 2);
    end
    
    % Reach data
    Xrea = [d2Rea{a}(1:minLenRea); mrBrRea{a}(1:minLenRea)]';
    validIdx = ~any(isnan(Xrea), 2);
    if sum(validIdx) > 10
        corrMatRea_d2mrBr{a} = corrcoef(Xrea(validIdx, :));
    else
        corrMatRea_d2mrBr{a} = nan(2, 2);
    end
end

% Calculate correlations between dcc and kappa (dcc sliding window timebase)
fprintf('\n=== Correlation Analysis: dcc and kappa ===\n');
corrMatNat_dccKappa = cell(1, length(areas));
corrMatRea_dccKappa = cell(1, length(areas));

for a = areasToTest
    % Get minimum length for dcc and kappa
    minLenNat = min(length(dccNat{a}), length(kappaNat{a}));
    minLenRea = min(length(dccRea{a}), length(kappaRea{a}));
    
    % Naturalistic data
    Xnat = [dccNat{a}(1:minLenNat); kappaNat{a}(1:minLenNat)]';
    validIdx = ~any(isnan(Xnat), 2);
    if sum(validIdx) > 10
        corrMatNat_dccKappa{a} = corrcoef(Xnat(validIdx, :));
    else
        corrMatNat_dccKappa{a} = nan(2, 2);
    end
    
    % Reach data
    Xrea = [dccRea{a}(1:minLenRea); kappaRea{a}(1:minLenRea)]';
    validIdx = ~any(isnan(Xrea), 2);
    if sum(validIdx) > 10
        corrMatRea_dccKappa{a} = corrcoef(Xrea(validIdx, :));
    else
        corrMatRea_dccKappa{a} = nan(2, 2);
    end
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

figure(202); clf;
set(gcf, 'Position', monitorTwo);
for a = areasToTest
    % dcc/kappa correlations
    subplot(2, length(areasToTest), a - areasToTest(1) + 1);
    imagesc(corrMatNat_dccKappa{a});
    colorbar;
    title(sprintf('%s (Nat dcc/kappa)', areas{a}));
    xticks(1:2); yticks(1:2);
    xticklabels({'dcc','kappa'}); yticklabels({'dcc','kappa'});
    axis square;
    caxis([cmin cmax]); % Set consistent color axis
    
    subplot(2, length(areasToTest), length(areasToTest) + (a - areasToTest(1) + 1));
    imagesc(corrMatRea_dccKappa{a});
    colorbar;
    title(sprintf('%s (Rea dcc/kappa)', areas{a}));
    xticks(1:2); yticks(1:2);
    xticklabels({'dcc','kappa'}); yticklabels({'dcc','kappa'});
    axis square;
    caxis([cmin cmax]); % Set consistent color axis
end
sgtitle('Correlation Matrices: dcc and kappa');

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
results.naturalistic.corrMat = corrMatNat_d2mrBr; % Correlation matrices are d2/mrBr

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
results.reach.corrMat = corrMatRea_d2mrBr; % Correlation matrices are d2/mrBr

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
results.params.dccBinSize = dccBinSize;
results.params.dccStepSize = dccStepSize;
results.params.dccWindowSize = dccWindowSize;

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



