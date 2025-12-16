%%       MR Estimator
% Adapted from Wilting and Preismann 2018
% Github python repo: https://github.com/jwilting/WiltingPriesemann2018/tree/master

% Steps:
% 1.  Make a spike count time series from your data. Let's call the result 'popActivity'
% For rodents, Wilting/Prieemann use 4ms bins and kmax = 2500ms

% 2.

% 3.

% One simple control to compare against is if you randomize the time order
% of points from your time series and then repeat steps 2 and 3.  This
% control tells you how far from criticality white noise is for the model order you chose.

%%
opts = neuro_behavior_options;
opts.minActTime = .16;
opts.minFiringRate = .05;

paths = get_paths;

tauRange = [1.2 2.5];
alphaRange = [1.5 2.2];
paramSDRange = [1.3 1.7];



monitorPositions = get(0, 'MonitorPositions');
monitorOne = monitorPositions(1, :); % Just use single monitor if you don't have second one
monitorTwo = monitorPositions(size(monitorPositions, 1), :); % Just use single monitor if you don't have second one


%% MR Estimator constants
opts.frameSize = .001;






%%   ====================================     Mark Reach Task: Sliding window       ==============================================
%% Mark's reach data
opts.frameSize = 0.001; % Set to 1 ms as requested

% Load data as before
dataR = load(fullfile(paths.dropPath, 'reach_data/Y4_100623_Spiketimes_idchan_BEH.mat'));
Fs = 1000; % dataR is in ms (1000 Hz)
opts.collectStart = 5 * 60; % seconds
opts.collectEnd = 25 * 60; % seconds
[dataMatR, idLabels, areaLabels, rmvNeurons] = reach_neural_matrix(dataR, opts);

% % Get data until 1 sec after the last reach ending.
% cutOff = round((dataR.R(end,2) + 1000) / 1000 / opts.frameSize);
% cutOff = round(35 * 60 / opts.frameSize);
% dataMatR = dataMatR(1:cutOff,:);

idM23R = find(strcmp(areaLabels, 'M23'));
idM56R = find(strcmp(areaLabels, 'M56'));
idDSR = find(strcmp(areaLabels, 'DS'));
idVSR = find(strcmp(areaLabels, 'VS'));
fprintf('%d M23\n%d M56\n%d DS\n%d VS\n', length(idM23R), length(idM56R), length(idDSR), length(idVSR))

areas = {'M23','M56', 'DS', 'VS'};
idList = {idM23R, idM56R, idDSR, idVSR};


%% ===================== Optimal frameSize and windowSize determination =====================
% Recommendations: frameSize should yield ~5-20 spikes/bin, windowSize should yield >=100-200 bins per window
candidateFrameSizes = [0.004, 0.01, 0.02, 0.05, .075, 0.1]; % 4ms, 10ms, 20ms, 50ms, 100ms
candidateWindowSizes = [30, 45, 60, 90, 120]; % in seconds
minSpikesPerBin = 5;
maxSpikesPerBin = 20;
minBinsPerWindow = 1000;

optimalBinSize = zeros(1, length(areas));
optimalWindowSize = zeros(1, length(areas));
for a = 1:length(areas)
    aID = idList{a};
    thisDataMat = dataMatR(:, aID);
    [optimalBinSize(a), optimalWindowSize(a)] = find_optimal_bin_and_window(thisDataMat, candidateFrameSizes, candidateWindowSizes, minSpikesPerBin, maxSpikesPerBin, minBinsPerWindow);
    fprintf('Area %s: optimal frame/bin size = %.3f s, optimal window size = %.1f s\n', areas{a}, optimalBinSize(a), optimalWindowSize(a));
end

% Find maximum bin size and corresponding maximum window size
[maxBinSize, maxBinIdx] = max(optimalBinSize);
% Find areas that have the maximum bin size
areasWithMaxBin = find(optimalBinSize == maxBinSize);
% Among those areas, find the maximum window size
maxWindowSize = max(optimalWindowSize(areasWithMaxBin));

fprintf('Using unified parameters for all areas: bin size = %.3f s, window size = %.1f s\n', maxBinSize, maxWindowSize);

%%    With optimal parameters, run each area
% Define sliding window parameters
debugFlag = 0;

stepSize = 2; %1 * 60; % (in seconds)

nShuffles = 10;

% Initialize variables
brMrRea = cell(1, length(areas));
brMrReaR = cell(1, length(areas));
popActivity = cell(1, length(areas));
startS = cell(1, length(areas));
% Initialize padded vectors
padded_brMrRea = cell(1, length(areas));

trialIdx2 = ismember(dataR.block(:, 3), 3:4);
block2First = find(trialIdx2, 1);
block2Start = dataR.R(block2First, 1) / 1000 / opts.frameSize;  % Make sure this is correct

corrVals = [2 4];
errVals = [1 3];
corrIdx = ismember(dataR.block(:,3), corrVals);
errIdx = ismember(dataR.block(:,3), errVals);
reachCorr = dataR.R(corrIdx, 1) / 1000 / opts.frameSize;  % Make sure this is correct
reachErr = dataR.R(errIdx, 1) / 1000 / opts.frameSize;  % Make sure this is correct


for a = 1 : length(areas)
    fprintf('Area %s\n', areas{a})
    tic
    aID = idList{a};

    % Convert window and step size to samples (in frames)
    stepSamples = round(stepSize / maxBinSize);
    winSamples = round(maxWindowSize / maxBinSize);

    aDataMatR = neural_matrix_ms_to_frames(dataMatR(:, aID), maxBinSize);

    numTimePoints = size(aDataMatR, 1);
    % Preallocate or store outputs if needed
    numWindows = floor((numTimePoints - winSamples) / stepSamples) + 1;

    popActivity{a} = sum(aDataMatR, 2);
    kMax = round(10 / maxBinSize);  % kMax in seconds, specific for this area's optimal frame size
    brMrRea{a} = nan(1, numWindows);
    brMrReaR{a} = nan(1, numWindows);
    startS{a} = nan(1, numWindows);
    padded_brMrRea{a} = nan(1, numTimePoints);
    for w = 1:numWindows
        startIdx = (w - 1) * stepSamples + 1;
        endIdx = startIdx + winSamples - 1;
        centerIdx = startIdx + floor((endIdx - startIdx)/2);
        startS{a}(w) = (startIdx + round(winSamples/2)-1) * maxBinSize;  % center of window
        wPopActivity = popActivity{a}(startIdx:endIdx);

        if debugFlag
            % Debug: Check if data is valid for MR estimation
            if any(isnan(wPopActivity)) || any(isinf(wPopActivity))
                fprintf('Warning: Window %d for area %s contains NaN or Inf values\n', w, areas{a});
                continue;
            end

            if all(wPopActivity == 0)
                fprintf('Warning: Window %d for area %s contains all zeros\n', w, areas{a});
                continue;
            end

            if length(wPopActivity) < 100
                fprintf('Warning: Window %d for area %s has too few samples (%d)\n', w, areas{a}, length(wPopActivity));
                continue;
            end

            % Check data statistics
            meanAct = mean(wPopActivity);
            stdAct = std(wPopActivity);
            fprintf('Window %d, Area %s: mean=%.3f, std=%.3f, length=%d\n', w, areas{a}, meanAct, stdAct, length(wPopActivity));
        end

        result = branching_ratio_mr_estimation(wPopActivity, kMax);
        brMrRea{a}(w) = result.branching_ratio;

        brShuff = zeros(nShuffles, 1);
        for nShuff = 1 : nShuffles
            % Shuffle data for comparison
            shuffledData = shift_shuffle_neurons(aDataMatR);

            wPopActivityR = sum(shuffledData(startIdx:endIdx,:), 2);

            resultR = branching_ratio_mr_estimation(wPopActivityR, kMax);
            brShuff(nShuff) = resultR.branching_ratio;
        end
        brMrReaR{a}(w) = mean(brShuff);
        padded_brMrRea{a}(centerIdx) = brMrRea{a}(w);

    end
    toc
end


%%
figure(60); clf; hold on;
plot(startS{1}/60, brMrRea{1}, '-k', 'lineWidth', 2);
plot(startS{2}/60, brMrRea{2}, '-b', 'lineWidth', 2);
plot(startS{3}/60, brMrRea{3}, '-r', 'lineWidth', 2);
% plot(startS{4}/60, brMrRea{4}, '-', 'color', [0 .75 0], 'lineWidth', 2);

plot(startS{1}/60, brMrReaR{1}, '--k');
plot(startS{2}/60, brMrReaR{2}, '--b');
plot(startS{3}/60, brMrReaR{3}, '--r');
% plot(startS{4}/60, brMrReaR{4}, '--', 'color', [0 .75 0]);
xline(block2Start/60, 'linewidth', 2)
legend({'M23', 'M56', 'DS', 'VS'}, 'Location','northwest')
xlabel('Minutes')
ylabel('MR estimate')

% h1 = plot(reachCorr*opts.frameSize/60, 1, '.', 'color', [0 .5 0], 'MarkerSize', 30);
% h2 = plot(reachErr*opts.frameSize/60, 1.002, '.', 'color', [.3 .3 .3], 'MarkerSize', 30);
% set(h1, 'HandleVisibility', 'off');
% set(h2, 'HandleVisibility', 'off');
title(['Reach Data ', num2str(maxWindowSize), ' sec window, ' num2str(stepSize), ' sec steps'])
xlim([0 opts.collectEnd/60])













%%   ====================================     Natrualistic: Sliding window       ==============================================
%% Naturalistic data
getDataType = 'spikes';
opts.firingRateCheckTime = 5 * 60;
opts.collectStart = 5 * 60; % seconds
opts.collectEnd = 25 * 60; % seconds
get_standard_data

areas = {'M23', 'M56', 'DS', 'VS'};
idList = {idM23, idM56, idDS, idVS};


%%
% Use the same candidate frame/bin and window sizes as above
[candidateBinSizes, candidateWindowSizesNat] = deal(candidateFrameSizes, candidateWindowSizes); % For clarity
optimalBinSizeNat = zeros(1, length(areas));
optimalWindowSizeNat = zeros(1, length(areas));
for a = 1:length(areas)
    aID = idList{a};
    thisDataMat = dataMat(:, aID);
    [optimalBinSizeNat(a), optimalWindowSizeNat(a)] = find_optimal_bin_and_window(thisDataMat, candidateBinSizes, candidateWindowSizesNat, minSpikesPerBin, maxSpikesPerBin, minBinsPerWindow);
    fprintf('Area %s: optimal frame/bin size = %.3f s, optimal window size = %.1f s\n', areas{a}, optimalBinSizeNat(a), optimalWindowSizeNat(a));
end

% Find maximum bin size and corresponding maximum window size
[maxBinSizeNat, maxBinIdx] = max(optimalBinSizeNat);
% Find areas that have the maximum bin size
areasWithMaxBin = find(optimalBinSizeNat == maxBinSizeNat);
% Among those areas, find the maximum window size
maxWindowSizeNat = max(optimalWindowSizeNat(areasWithMaxBin));

fprintf('Using unified parameters for all areas: bin size = %.3f s, window size = %.1f s\n', maxBinSizeNat, maxWindowSizeNat);

%%
% Define sliding window parameters
stepSize = 2; %1 * 60; %  (in seconds)

% Initialize variables for MR estimation
nShuffles = 10;
brMrNat = cell(1, length(areas));
brMrRNat = cell(1, length(areas));
popActivityNat = cell(1, length(areas));
startSNat = cell(1, length(areas));
% Initialize padded vectors
padded_brMrNat = cell(1, length(areas));

for a = 1:length(areas)
    fprintf('Area %s (Naturalistic)\n', areas{a})
    tic
    aID = idList{a};

    % Convert window and step size to samples
    stepSamples = round(stepSize / maxBinSizeNat);
    winSamples = round(maxWindowSizeNat / maxBinSizeNat);
    

    % Bin the data for this area using optimal bin size
    aDataMatNat = neural_matrix_ms_to_frames(dataMat(:, aID), maxBinSizeNat);
    numTimePointsNat = size(aDataMatNat, 1);
    numWindowsNat = floor((numTimePointsNat - winSamples) / stepSamples) + 1;

    popActivityNat{a} = sum(aDataMatNat, 2);
    kMax = round(10 / maxBinSizeNat);  % kMax in seconds, specific for this area's bin size
    brMrNat{a} = nan(1, numWindowsNat);
    brMrRNat{a} = nan(1, numWindowsNat);
    startSNat{a} = nan(1, numWindowsNat);
    padded_brMrNat{a} = nan(1, numTimePointsNat);
    for w = 1:numWindowsNat
        startIdx = (w - 1) * stepSamples + 1;
        endIdx = startIdx + winSamples - 1;
        centerIdx = startIdx + floor((endIdx - startIdx)/2);
        startSNat{a}(w) = (startIdx + round(winSamples/2)-1) * maxBinSizeNat;  % center of window
        wPopActivity = popActivityNat{a}(startIdx:endIdx);
        result = branching_ratio_mr_estimation(wPopActivity, kMax);
        brMrNat{a}(w) = result.branching_ratio;
        brMrRNat{a}(w) = nan; % keep as before
        padded_brMrNat{a}(centerIdx) = brMrNat{a}(w);
        brShuff = zeros(nShuffles, 1);
        for nShuff = 1:nShuffles
            shuffledData = shift_shuffle_neurons(aDataMatNat);
            wPopActivityR = sum(shuffledData(startIdx:endIdx,:), 2);
            resultR = branching_ratio_mr_estimation(wPopActivityR, kMax);
            brShuff(nShuff) = resultR.branching_ratio;
        end
        brMrRNat{a}(w) = mean(brShuff);
    end
    toc
end

%% Plotting for Naturalistic MR estimation
figure(62); clf; hold on;
plot(startSNat{1}/60, brMrNat{1}, '-k', 'lineWidth', 2);
plot(startSNat{2}/60, brMrNat{2}, '-b', 'lineWidth', 2);
plot(startSNat{3}/60, brMrNat{3}, '-r', 'lineWidth', 2);
% plot(startSNat{4}/60, brMrNat{4}, '-', 'color', [0 .75 0], 'lineWidth', 2);

plot(startSNat{1}/60, brMrRNat{1}, '--k');
plot(startSNat{2}/60, brMrRNat{2}, '--b');
plot(startSNat{3}/60, brMrRNat{3}, '--r');
% plot(startSNat{4}/60, brMrRNat{4}, '--', 'color', [0 .75 0]);
legend({'M23', 'M56', 'DS', 'VS'}, 'Location','northwest')
xlabel('Minutes')
ylabel('MR estimate')
title(['Naturalistic Data ', num2str(maxWindowSizeNat), ' sec window, ' num2str(stepSize), ' sec steps'])
xlim([0 opts.collectEnd/60])



%% ========== Test how brMrNat changes as a function of number of neurons in DS ==========

% Select DS area (a=3)
a = 2;
fprintf('\nTesting effect of neuron count on brMrNat for %s (a=%d)\n', areas{a}, a);

% Option: use fixed optimal parameters (true) or find optimal for each subpopulation (false)
useFixedOptimalParams = true; % Set to false to find optimal parameters for each subpopulation

% Get neuron IDs for DS and randomly permute
allDSNeurons = idList{a};
numNeurons = length(allDSNeurons);
rng('shuffle'); % For randomness
permNeurons = allDSNeurons(randperm(numNeurons));

% Define 5 increasing neuron counts from half to max, in equal steps
minCount = ceil(numNeurons/2);
maxCount = numNeurons;
numSteps = 3;
neuronCounts = round(linspace(minCount, maxCount, numSteps));

% Preallocate results
brMrNat_Area = cell(1, numSteps);
brMrRNat_Area = cell(1, numSteps);
startSNat_Area = cell(1, numSteps);
usedNeuronCounts = zeros(1, numSteps);
optimalBinSizes_Area = zeros(1, numSteps);
optimalWindowSizes_Area = zeros(1, numSteps);

% Find fixed optimal parameters if using fixed approach
if useFixedOptimalParams
    firstCount = neuronCounts(1);
    firstNeurons = permNeurons(1:firstCount);
    dataMatDS_first = dataMat(:, firstNeurons);
    [optimalBinSize_fixed, optimalWindowSize_fixed] = find_optimal_bin_and_window(dataMatDS_first, ...
        candidateFrameSizes, candidateWindowSizes, minSpikesPerBin, maxSpikesPerBin, minBinsPerWindow);
    
    if optimalBinSize_fixed == 0 || optimalWindowSize_fixed == 0
        warning('No optimal bin/window size found for first iteration (%d neurons in DS). Skipping all iterations.', firstCount);
    else
        fprintf('Using fixed optimal bin size = %.3f s and window size = %.1f s for all iterations\n', optimalBinSize_fixed, optimalWindowSize_fixed);
    end
end

% Single loop for all iterations
for i = 1:numSteps
    currCount = neuronCounts(i);
    usedNeuronCounts(i) = currCount;
    currNeurons = permNeurons(1:currCount);
    dataMatDS = dataMat(:, currNeurons);
    
    % Determine optimal parameters for this iteration
    if useFixedOptimalParams
        if optimalBinSize_fixed == 0 || optimalWindowSize_fixed == 0
            % Skip this iteration if no optimal parameters found
            brMrNat_Area{i} = nan;
            brMrRNat_Area{i} = nan;
            startSNat_Area{i} = nan;
            optimalBinSizes_Area(i) = nan;
            optimalWindowSizes_Area(i) = nan;
            continue;
        end
        optimalBinSize = optimalBinSize_fixed;
        optimalWindowSize = optimalWindowSize_fixed;
    else
        % Find optimal parameters for this subpopulation
        [optimalBinSize, optimalWindowSize] = find_optimal_bin_and_window(dataMatDS, ...
            candidateFrameSizes, candidateWindowSizes, minSpikesPerBin, maxSpikesPerBin, minBinsPerWindow);
        fprintf('Using optimal bin size = %.3f s and window size = %.1f s\n', optimalBinSize, optimalWindowSize);
        
        if optimalBinSize == 0 || optimalWindowSize == 0
            warning('No optimal bin/window size found for %d neurons in. Skipping.', currCount);
            brMrNat_Area{i} = nan;
            brMrRNat_Area{i} = nan;
            startSNat_Area{i} = nan;
            optimalBinSizes_Area(i) = nan;
            optimalWindowSizes_Area(i) = nan;
            continue;
        end
    end
    
    % Store optimal parameters for this iteration
    optimalBinSizes_Area(i) = optimalBinSize;
    optimalWindowSizes_Area(i) = optimalWindowSize;
    
    % Bin the data and compute MR estimates
    aDataMatNat = neural_matrix_ms_to_frames(dataMatDS, optimalBinSize);
    numTimePointsNat = size(aDataMatNat, 1);
    
    % Calculate window parameters
    stepSamples = round(stepSize / optimalBinSize);
    winSamples = round(optimalWindowSize / optimalBinSize);
    numWindowsNat = floor((numTimePointsNat - winSamples) / stepSamples) + 1;
    
    popActivityNat = sum(aDataMatNat, 2);
    kMax = round(10 / optimalBinSize);
    
    brMrNat_Area{i} = nan(1, numWindowsNat);
    brMrRNat_Area{i} = nan(1, numWindowsNat);
    startSNat_Area{i} = nan(1, numWindowsNat);
    
    for w = 1:numWindowsNat
        startIdx = (w - 1) * stepSamples + 1;
        endIdx = startIdx + winSamples - 1;
        centerIdx = startIdx + floor((endIdx - startIdx)/2);
        startSNat_Area{i}(w) = (startIdx + round(winSamples/2)-1) * optimalBinSize;  % center of window
        wPopActivity = popActivityNat(startIdx:endIdx);
        result = branching_ratio_mr_estimation(wPopActivity, kMax);
        brMrNat_Area{i}(w) = result.branching_ratio;
        brShuff = zeros(nShuffles, 1);
        for nShuff = 1:nShuffles
            shuffledData = shift_shuffle_neurons(aDataMatNat);
            wPopActivityR = sum(shuffledData(startIdx:endIdx,:), 2);
            resultR = branching_ratio_mr_estimation(wPopActivityR, kMax);
            brShuff(nShuff) = resultR.branching_ratio;
        end
        brMrRNat_Area{i}(w) = mean(brShuff);
    end
    
    % Print progress with appropriate message
    if useFixedOptimalParams
        fprintf('Done for %d neurons in DS (step %d/%d)\n', currCount, i, numSteps);
    else
        fprintf('Done for %d neurons in DS (step %d/%d) with optimal bin=%.3f, window=%.1f\n', currCount, i, numSteps, optimalBinSize, optimalWindowSize);
    end
end

%% Plotting: brMrNat as a function of time for different neuron counts in DS
figure(66); clf; hold on;
colors = lines(numSteps);
for i = 1:numSteps
    if ~isnan(brMrNat_Area{i})
        plot(startSNat_Area{i}/60, brMrNat_Area{i}, '-', 'Color', colors(i,:), 'LineWidth', 2, ...
            'DisplayName', sprintf('%d neurons', usedNeuronCounts(i)));
    end
end
xlabel('Minutes');
ylabel('MR estimate');
title(sprintf('%s: MR estimate vs. time for increasing neuron counts', areas{a}));
legend('show');
xlim([0 opts.collectEnd/60]);










%%   ====================================     Mark Task: Block1 vs Block2       ==============================================

areas = {'M23', 'M56', 'DS', 'VS'};
idList = {idM23R, idM56R, idDSR, idVSR};

isiMult = 10; % Multiple of mean ISI to determine minimum bin size
pOrder = 10; % Order parameter for the autoregressor model
critType = 2;
optBinSize = nan(length(areas), 1);
[d21, d22, d21R, d22R, varNoise1, varNoise2, varNoise1R, varNoise2R] = ...
    deal(nan(length(areas), 1));

trialIdx1 = ismember(dataR.block(:, 3), 1:2);
trialIdx2 = ismember(dataR.block(:, 3), 3:4);
block1Last = find(trialIdx1, 1, 'last');
block2First = find(trialIdx2, 1);
block1End = dataR.R(block1Last, 2);
block2Start = dataR.R(block2First, 1);

shuffledData1 = shift_shuffle_neurons(dataMatR(1:block1End,:));
shuffledData2 = shift_shuffle_neurons(dataMatR(block2First:end,:));

for a = 1 : length(areas)
    fprintf('Area %s\n', areas{a})
    tic
    aID = idList{a};

    optBinSize(a) = isiMult * round(mean(diff(find(sum(dataMatR(:, aID), 2))))) / 1000;
    optBinSize(a) = .05;

    popActivity1 = neural_matrix_ms_to_frames(sum(dataMatR(1:block1End, aID), 2), optBinSize(a));
    popActivity2 = neural_matrix_ms_to_frames(sum(dataMatR(block2First:end, aID), 2), optBinSize(a));
    % popActivity2 = neural_matrix_ms_to_frames(sum(dataMatR(block2First:block2First+5*60*1000, aID), 2), optBinSize(a));

    [varphi, varNoise1(a)] = myYuleWalker3(popActivity1, pOrder);
    d21(a) =getFixedPointDistance2(pOrder, critType, varphi);
    [varphi, varNoise2(a)] = myYuleWalker3(popActivity2, pOrder);
    d22(a) =getFixedPointDistance2(pOrder, critType, varphi);

    % popActivity1R = popActivity1(randperm(length(popActivity1)));
    % popActivity2R = popActivity2(randperm(length(popActivity2)));
    popActivity1R = sum(shuffledData1(:, aID), 2);
    popActivity2R = sum(shuffledData2(:, aID), 2);

    [varphi, varNoise1R(a)] = myYuleWalker3(popActivity1R, pOrder);
    d21R(a) =getFixedPointDistance2(pOrder, critType, varphi);
    [varphi, varNoise2R(a)] = myYuleWalker3(popActivity2R, pOrder);
    d22R(a) =getFixedPointDistance2(pOrder, critType, varphi);

end

[d21 d21R d22 d22R]

%% =================        NATURALISTIC DATA              ================
isiMult = 10; % Multiple of mean ISI to determine minimum bin size
pOrder = 10; % Order parameter for the autoregressor model
critType = 2;
[optBinSize, d2N, d2NR, varNoiseN, varNoiseNR] = ...
    deal(nan(length(areas), 1));

shuffledData = shift_shuffle_neurons(dataMat);

for a = 1 : length(areas)
    fprintf('Area %s\n', areas{a})
    tic
    aID = idList{a};

    optBinSize(a) = isiMult * round(mean(diff(find(sum(dataMat(:, aID), 2))))) / 1000;
    optBinSize(a) = .05;
    popActivityN = neural_matrix_ms_to_frames(sum(dataMat(:, aID), 2), optBinSize(a));

    [varphi, varNoiseN(a)] = myYuleWalker3(popActivityN, pOrder);
    d2N(a) =getFixedPointDistance2(pOrder, critType, varphi);

    % popActivityR = popActivityN(randperm(length(popActivityN)));
    popActivityR = sum(shuffledData(:, aID), 2);

    [varphi, varNoiseNR(a)] = myYuleWalker3(popActivityR, pOrder);
    d2NR(a) =getFixedPointDistance2(pOrder, critType, varphi);

end

%%
figure(45); clf
plot(1:4, d21, 'ok', 'LineWidth', 2)
hold on
plot(1:4, d22, 'or', 'LineWidth', 2)
plot(1:4, d2N, 'ob', 'LineWidth', 2)
plot(1:4, d21R, '*k')
plot(1:4, d22R, '*r')
plot(1:4, d2NR, '*b')
legend({'Block 1', 'Block 2', 'Naturalistic', 'B1 Rand', 'B2 Rand', 'Nat Rand'})
xlim([.5 4.5])
title('Distance to Criticality')
xticks(1:4)
xticklabels({'M23', 'M56', 'DS', 'VS'})
ylabel('Distance to criticality')









function shuffledData = shift_shuffle_neurons(dataMat)
% shift_shuffle_neurons circularly shifts each neuron's activity by a random
% amount between 1 and half the duration of the data.
%
% dataMat: [time x neurons] matrix
% shuffledData: matrix of the same size with shuffled neuron time courses

[numTimePoints, numNeurons] = size(dataMat);
maxShift = floor(numTimePoints / 2);

% Initialize output
shuffledData = zeros(size(dataMat));

for n = 1:numNeurons
    % Generate random shift between 1 and half the duration
    shiftAmount = randi([1, maxShift]);

    % Circularly shift the neuron's time series
    shuffledData(:, n) = circshift(dataMat(:, n), shiftAmount);
end
end



function maxEig = computeMaxEigenvalue(varphi)
% Computes the maximum eigenvalue of the AR model's companion matrix
% INPUT:
%   varphi - AR model coefficients (vector of length equal to model order)
% OUTPUT:
%   maxEig - maximum (absolute) eigenvalue of the companion matrix

p = length(varphi);  % AR model order

% Construct the companion matrix
C = zeros(p);
C(1, :) = varphi(:)';  % First row is the AR coefficients
C(2:end, 1:end-1) = eye(p-1);  % Sub-diagonal identity matrix

% Compute eigenvalues
eigVals = eig(C);

% Return the maximum absolute eigenvalue (spectral radius)
maxEig = max(abs(eigVals));
end
