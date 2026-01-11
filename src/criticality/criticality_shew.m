%%
% Steps:
% 1.  Make a spike count time series from your data.  (Use a time bin that
% is at least 10x the mean ISI as a rule of thumb).  Let's call the result 'popActivity'

% 2.  Fit an AR model to the data.  You can use the attached function
% called myYuleWalker3.  Example syntax for fitting a 10th order model:
% [varphi, varNoise] = myYuleWalker3(popActivity, 10);

% 3.  Calculate distance to criticality (d2).  Use the other attached
% function called getFixedPointDistance2.   Example syntax: d2 =getFixedPointDistance2(10, 2, varphi);
%
% One simple control to compare against is if you randomize the time order
% of points from your time series and then repeat steps 2 and 3.  This
% control tells you how far from criticality white noise is for the model order you chose.

%%
opts = neuro_behavior_options;
opts.minActTime = .16;
opts.collectStart = 0 * 60 * 60; % seconds
opts.collectEnd = 1*60 * 60; % seconds
opts.minFiringRate = .05;
opts.frameSize = .001;

paths = get_paths;

tauRange = [1.2 2.5];
alphaRange = [1.5 2.2];
paramSDRange = [1.3 1.7];



monitorPositions = get(0, 'MonitorPositions');
monitorOne = monitorPositions(1, :); % Just use single monitor if you don't have second one
monitorTwo = monitorPositions(size(monitorPositions, 1), :); % Just use single monitor if you don't have second one



%% Spontaneous data
getDataType = 'spikes';
opts.firingRateCheckTime = 5 * 60;
opts.collectEnd = 45 * 60; % seconds
get_standard_data

areas = {'M23', 'M56', 'DS', 'VS'};
idList = {idM23, idM56, idDS, idVS};


%% Mark's reach data
dataR = load(fullfile(paths.dropPath, 'reach_data/Y4_100623_Spiketimes_idchan_BEH.mat'));
[dataMatR, idLabels, areaLabels, rmvNeurons] = reach_neural_matrix(dataR, opts);

% Get data until 1 sec after the last reach ending.
cutOff = round((dataR.R(end,2) + 1000) / 1000 / opts.frameSize);
dataMatR = dataMatR(1:cutOff,:);

% Ensure dataMatR is same size as dataMat
idM23R = find(strcmp(areaLabels, 'M23'));
idM56R = find(strcmp(areaLabels, 'M56'));
idDSR = find(strcmp(areaLabels, 'DS'));
idVSR = find(strcmp(areaLabels, 'VS'));

areas = {'M23', 'M56', 'DS', 'VS'};
idList = {idM23R, idM56R, idDSR, idVSR};










%%   ====================================     Mark Reach Task: Sliding window (mirrored from MR estimator)   ==============================================
% Optimal bin/window size search for reach data
candidateFrameSizes = [0.004, 0.01, 0.02, 0.05, .075, 0.1];
candidateWindowSizes = [30, 45, 60, 90, 120];
minSpikesPerBin = 3;
maxSpikesPerBin = 20;
minBinsPerWindow = 500;

optimalBinSizeRea = zeros(1, length(areas));
optimalWindowSizeRea = zeros(1, length(areas));
for a = 1:length(areas)
    aID = idList{a};
    thisDataMat = dataMatR(:, aID);
    [optimalBinSizeRea(a), optimalWindowSizeRea(a)] = find_optimal_bin_and_window(thisDataMat, candidateFrameSizes, candidateWindowSizes, minSpikesPerBin, maxSpikesPerBin, minBinsPerWindow);
    fprintf('Area %s: optimal frame/bin size = %.3f s, optimal window size = %.1f s\n', areas{a}, optimalBinSizeRea(a), optimalWindowSizeRea(a));
end

% Find maximum bin size and corresponding maximum window size
[maxBinSizeRea, maxBinIdx] = max(optimalBinSizeRea);
% Find areas that have the maximum bin size
areasWithMaxBin = find(optimalBinSizeRea == maxBinSizeRea);
% Among those areas, find the maximum window size
maxWindowSizeRea = max(optimalWindowSizeRea(areasWithMaxBin));

fprintf('Using unified parameters for all areas: bin size = %.3f s, window size = %.1f s\n', maxBinSizeRea, maxWindowSizeRea);

% Sliding window d2 analysis for reach data
stepSize = 2; % seconds
nShuffles = 10;
d2Rea = cell(1, length(areas));
d2ReaR = cell(1, length(areas));
popActivityRea = cell(1, length(areas));
startSRea = cell(1, length(areas));
% Initialize padded vectors
padded_d2Rea = cell(1, length(areas));
pOrder = 10;
critType = 2;

for a = 1:length(areas)
    aID = idList{a};
    stepSamples = round(stepSize / maxBinSizeRea);
    winSamples = round(maxWindowSizeRea / maxBinSizeRea);
    aDataMatR = neural_matrix_ms_to_frames(dataMatR(:, aID), maxBinSizeRea);
    numTimePoints = size(aDataMatR, 1);
    numWindows = floor((numTimePoints - winSamples) / stepSamples) + 1;
    popActivityRea{a} = sum(aDataMatR, 2);
    d2Rea{a} = nan(1, numWindows);
    d2ReaR{a} = nan(1, numWindows);
    startSRea{a} = nan(1, numWindows);
    padded_d2Rea{a} = nan(1, numTimePoints);
    for w = 1:numWindows
        startIdx = (w - 1) * stepSamples + 1;
        endIdx = startIdx + winSamples - 1;
        centerIdx = startIdx + floor((endIdx - startIdx)/2);
        startSRea{a}(w) = (startIdx + round(winSamples/2)-1) * maxBinSizeRea;
        wPopActivity = popActivityRea{a}(startIdx:endIdx);
        [varphi, ~] = myYuleWalker3(wPopActivity, pOrder);
        d2Rea{a}(w) = getFixedPointDistance2(pOrder, critType, varphi);
        d2Shuff = zeros(nShuffles, 1);
        for nShuff = 1:nShuffles
            shuffledData = shift_shuffle_neurons(aDataMatR);
            wPopActivityR = sum(shuffledData(startIdx:endIdx,:), 2);
            [varphi, ~] = myYuleWalker3(wPopActivityR, pOrder);
            d2Shuff(nShuff) = getFixedPointDistance2(pOrder, critType, varphi);
        end
        d2ReaR{a}(w) = mean(d2Shuff);
        padded_d2Rea{a}(centerIdx) = d2Rea{a}(w);
    end
end

% Plotting for reach d2
figure(160); clf; hold on;
plot(startSRea{1}/60, d2Rea{1}, '-ok', 'lineWidth', 2);
plot(startSRea{2}/60, d2Rea{2}, '-ob', 'lineWidth', 2);
plot(startSRea{3}/60, d2Rea{3}, '-or', 'lineWidth', 2);
% plot(startSRea{4}/60, d2Rea{4}, '-o', 'color', [0 .75 0], 'lineWidth', 2);
plot(startSRea{1}/60, d2ReaR{1}, '*k');
plot(startSRea{2}/60, d2ReaR{2}, '*b');
plot(startSRea{3}/60, d2ReaR{3}, '*r');
% plot(startSRea{4}/60, d2ReaR{4}, '*', 'color', [0 .75 0]);
xlabel('Minutes'); ylabel('d2 estimate');
legend({'M23', 'M56', 'DS', 'VS'}, 'Location','northwest');
title(['Reach Data ', num2str(maxWindowSizeRea), ' sec window, ' num2str(stepSize), ' sec steps']);
xlim([0 opts.collectEnd/60]);





%%   ====================================     Spontaneous: Sliding window (mirrored from MR estimator)   ==============================================
% Optimal bin/window size search for spontaneous data
[candidateBinSizes, candidateWindowSizesNat] = deal(candidateFrameSizes, candidateWindowSizes);
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

% Sliding window d2 analysis for spontaneous data
stepSize = 1; % seconds
nShuffles = 10;
d2Nat = cell(1, length(areas));
d2NatR = cell(1, length(areas));
popActivityNat = cell(1, length(areas));
startSNat = cell(1, length(areas));
% Initialize padded vectors
padded_d2Nat = cell(1, length(areas));
pOrder = 10;
critType = 2;

for a = 1:length(areas)
    aID = idList{a};
    tic
    fprintf('Area %s (Spontaneous)\n', areas{a})
    stepSamples = round(stepSize / maxBinSizeNat);
    winSamples = round(maxWindowSizeNat / maxBinSizeNat);
    aDataMatNat = neural_matrix_ms_to_frames(dataMat(:, aID), maxBinSizeNat);
    numTimePointsNat = size(aDataMatNat, 1);
    numWindowsNat = floor((numTimePointsNat - winSamples) / stepSamples) + 1;
    popActivityNat{a} = sum(aDataMatNat, 2);
    d2Nat{a} = nan(1, numWindowsNat);
    d2NatR{a} = nan(1, numWindowsNat);
    startSNat{a} = nan(1, numWindowsNat);
    padded_d2Nat{a} = nan(1, numTimePointsNat);
    for w = 1:numWindowsNat
        startIdx = (w - 1) * stepSamples + 1;
        endIdx = startIdx + winSamples - 1;
        centerIdx = startIdx + floor((endIdx - startIdx)/2);
        startSNat{a}(w) = (startIdx + round(winSamples/2)-1) * maxBinSizeNat;
        wPopActivity = popActivityNat{a}(startIdx:endIdx);
        [varphi, ~] = myYuleWalker3(wPopActivity, pOrder);
        d2Nat{a}(w) = getFixedPointDistance2(pOrder, critType, varphi);
        d2Shuff = zeros(nShuffles, 1);
        for nShuff = 1:nShuffles
            shuffledData = shift_shuffle_neurons(aDataMatNat);
            wPopActivityR = sum(shuffledData(startIdx:endIdx,:), 2);
            [varphi, ~] = myYuleWalker3(wPopActivityR, pOrder);
            d2Shuff(nShuff) = getFixedPointDistance2(pOrder, critType, varphi);
        end
        d2NatR{a}(w) = mean(d2Shuff);
        padded_d2Nat{a}(centerIdx) = d2Nat{a}(w);
    end
    toc
end

% Plotting for spontaneous d2
figure(161); clf; hold on;
plot(startSNat{1}/60, d2Nat{1}, '-ok', 'lineWidth', 2);
plot(startSNat{2}/60, d2Nat{2}, '-ob', 'lineWidth', 2);
plot(startSNat{3}/60, d2Nat{3}, '-or', 'lineWidth', 2);
% plot(startSNat{4}/60, d2Nat{4}, '-o', 'color', [0 .75 0], 'lineWidth', 2);
plot(startSNat{1}/60, d2NatR{1}, '*k');
plot(startSNat{2}/60, d2NatR{2}, '*b');
plot(startSNat{3}/60, d2NatR{3}, '*r');
% plot(startSNat{4}/60, d2NatR{4}, '*', 'color', [0 .75 0]);
xlabel('Minutes'); ylabel('d2 estimate');
legend({'M23', 'M56', 'DS', 'VS'}, 'Location','northwest');
title(['Spontaneous Data ', num2str(maxWindowSizeNat), ' sec window, ' num2str(stepSize), ' sec steps']);
xlim([0 opts.collectEnd/60]);


%% ========== Test how d2Nat changes as a function of number of neurons in area (mirrored from MR estimator) ==========
% Select area (e.g., a = 2 for M56)
a = 2;
fprintf('\nTesting effect of neuron count on d2Nat for %s (a=%d)\n', areas{a}, a);
useFixedOptimalParams = false; % Set to false to find optimal parameters for each subpopulation

allAreaNeurons = idList{a};
numNeurons = length(allAreaNeurons);
rng('shuffle');
permNeurons = allAreaNeurons(randperm(numNeurons));
minCount = ceil(numNeurons/2);
maxCount = numNeurons;
numSteps = 3;
neuronCounts = round(linspace(minCount, maxCount, numSteps));
d2Nat_Area = cell(1, numSteps);
d2NatR_Area = cell(1, numSteps);
startSNat_Area = cell(1, numSteps);
usedNeuronCounts = zeros(1, numSteps);
optimalBinSizes_Area = zeros(1, numSteps);
optimalWindowSizes_Area = zeros(1, numSteps);
if useFixedOptimalParams
    firstCount = neuronCounts(1);
    firstNeurons = permNeurons(1:firstCount);
    dataMatArea_first = dataMat(:, firstNeurons);
    [optimalBinSize_fixed, optimalWindowSize_fixed] = find_optimal_bin_and_window(dataMatArea_first, candidateFrameSizes, candidateWindowSizes, minSpikesPerBin, maxSpikesPerBin, minBinsPerWindow);
    if optimalBinSize_fixed == 0 || optimalWindowSize_fixed == 0
        warning('No optimal bin/window size found for first iteration (%d neurons in area). Skipping all iterations.', firstCount);
    else
        fprintf('Using fixed optimal bin size = %.3f s and window size = %.1f s for all iterations\n', optimalBinSize_fixed, optimalWindowSize_fixed);
    end
end
for i = 1:numSteps
    currCount = neuronCounts(i);
    usedNeuronCounts(i) = currCount;
    currNeurons = permNeurons(1:currCount);
    dataMatArea = dataMat(:, currNeurons);
    if useFixedOptimalParams
        if optimalBinSize_fixed == 0 || optimalWindowSize_fixed == 0
            d2Nat_Area{i} = nan;
            d2NatR_Area{i} = nan;
            startSNat_Area{i} = nan;
            optimalBinSizes_Area(i) = nan;
            optimalWindowSizes_Area(i) = nan;
            continue;
        end
        optimalBinSize = optimalBinSize_fixed;
        optimalWindowSize = optimalWindowSize_fixed;
    else
        [optimalBinSize, optimalWindowSize] = find_optimal_bin_and_window(dataMatArea, candidateFrameSizes, candidateWindowSizes, minSpikesPerBin, maxSpikesPerBin, minBinsPerWindow);
        fprintf('Using optimal bin size = %.3f s and window size = %.1f s\n', optimalBinSize, optimalWindowSize);
        if optimalBinSize == 0 || optimalWindowSize == 0
            warning('No optimal bin/window size found for %d neurons in. Skipping.', currCount);
            d2Nat_Area{i} = nan;
            d2NatR_Area{i} = nan;
            startSNat_Area{i} = nan;
            optimalBinSizes_Area(i) = nan;
            optimalWindowSizes_Area(i) = nan;
            continue;
        end
    end
    optimalBinSizes_Area(i) = optimalBinSize;
    optimalWindowSizes_Area(i) = optimalWindowSize;
    aDataMatNat = neural_matrix_ms_to_frames(dataMat(:, currNeurons), optimalBinSize);
    numTimePointsNat = size(aDataMatNat, 1);
    stepSamples = round(stepSize / optimalBinSize);
    winSamples = round(optimalWindowSize / optimalBinSize);
    numWindowsNat = floor((numTimePointsNat - winSamples) / stepSamples) + 1;
    popActivityNat = sum(aDataMatNat, 2);
    d2Nat_Area{i} = nan(1, numWindowsNat);
    d2NatR_Area{i} = nan(1, numWindowsNat);
    startSNat_Area{i} = nan(1, numWindowsNat);
    for w = 1:numWindowsNat
        startIdx = (w - 1) * stepSamples + 1;
        endIdx = startIdx + winSamples - 1;
        startSNat_Area{i}(w) = (startIdx + round(winSamples/2)-1) * optimalBinSize;
        wPopActivity = popActivityNat(startIdx:endIdx);
        [varphi, ~] = myYuleWalker3(wPopActivity, pOrder);
        d2Nat_Area{i}(w) = getFixedPointDistance2(pOrder, critType, varphi);
        d2Shuff = zeros(nShuffles, 1);
        for nShuff = 1:nShuffles
            shuffledData = shift_shuffle_neurons(aDataMatNat);
            wPopActivityR = sum(shuffledData(startIdx:endIdx,:), 2);
            [varphi, ~] = myYuleWalker3(wPopActivityR, pOrder);
            d2Shuff(nShuff) = getFixedPointDistance2(pOrder, critType, varphi);
        end
        d2NatR_Area{i}(w) = mean(d2Shuff);
    end
    if useFixedOptimalParams
        fprintf('Done for %d neurons in %s (step %d/%d)\n', currCount, areas{a}, i, numSteps);
    else
        fprintf('Done for %d neurons in %s (step %d/%d) with optimal bin=%.3f, window=%.1f\n', currCount, areas{a}, i, numSteps, optimalBinSize, optimalWindowSize);
    end
end
figure(163); clf; hold on;
colors = lines(numSteps);
for i = 1:numSteps
    if ~isnan(d2Nat_Area{i})
        plot(startSNat_Area{i}/60, d2Nat_Area{i}, '-', 'Color', colors(i,:), 'LineWidth', 2, ...
            'DisplayName', sprintf('%d neurons', usedNeuronCounts(i)));
    end
end
xlabel('Minutes'); ylabel('d2 estimate');
title(sprintf('%s: d2 estimate vs. time for increasing neuron counts', areas{a}));
legend('show');
xlim([0 opts.collectEnd/60]);

% ========== End mirrored sections ==========
















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
legend({'Block 1', 'Block 2', 'Spontaneous', 'B1 Rand', 'B2 Rand', 'Nat Rand'})
xlim([.5 4.5])
title('Distance to Criticality')
xticks(1:4)
xticklabels({'M23', 'M56', 'DS', 'VS'})
ylabel('Distance to criticality')
































%%   ====================================     Mark Task: Sliding window       ==============================================
% Load the data above (Mark's data)

isiMult = 10; % Multiple of mean ISI to determine minimum bin size
pOrder = 10; % Order parameter for the autoregressor model
critType = 2;
binSize = .04;

% Define sliding window parameters
windowSize = 1 * 60; % (in seconds)
stepSize = 2; %1 * 60; % (in seconds)
Fs = 1000; % dataR is in ms

% Convert window and step size to samples
winSamples = round(windowSize * Fs);
stepSamples = round(stepSize * Fs);
numTimePoints = size(dataMatR, 1);

% Preallocate or store outputs if needed
numWindows = floor((numTimePoints - winSamples) / stepSamples) + 1;


% Initialize variables
woody = struct();
optBinSize = nan(length(areas), 1);
[d2,d2R, varNoise, varNoiseR] = ...
    deal(nan(numWindows, length(areas)));
startS = nan(numWindows, 1);
popActivity = cell(4,1);

trialIdx2 = ismember(dataR.block(:, 3), 3:4);
block2First = find(trialIdx2, 1);
block2Start = dataR.R(block2First, 1) / Fs;

corrVals = [2 4];
errVals = [1 3];
corrIdx = ismember(dataR.block(:,3), corrVals);
errIdx = ismember(dataR.block(:,3), errVals);
reachCorr = dataR.R(corrIdx, 1) / Fs;
reachErr = dataR.R(errIdx, 1) / Fs;

% Shuffle data for comparison
shuffledData = shift_shuffle_neurons(dataMatR);

for a = 1 : length(areas)
    fprintf('Area %s\n', areas{a})
    tic
    aID = idList{a};

    % Remove crazy high multi-units
    aRmv = max(dataMatR(:,aID)) > 2;
    aID(aRmv) = [];

    optBinSize(a) = isiMult * round(mean(diff(find(sum(dataMatR(:, aID), 2))))) / 1000;
    % optBinSize(a) = binSize;

    popActivity{a} = neural_matrix_ms_to_frames(sum(dataMatR(:, aID), 2), optBinSize(a));
    for w = 1:numWindows
        startIdx = (w - 1) * stepSamples + 1;
        endIdx = startIdx + winSamples - 1;

        startS(w) = (startIdx + round(winSamples/2)-1) / Fs;  % center of window
        wPopActivity = neural_matrix_ms_to_frames(sum(dataMatR(startIdx:endIdx, aID), 2), optBinSize(a));

        [varphi, varNoise(w,a)] = myYuleWalker3(wPopActivity, pOrder);
        d2(w,a) =getFixedPointDistance2(pOrder, critType, varphi);

        % popActivityR = popActivity(randperm(length(popActivity)));
        wPopActivityR = neural_matrix_ms_to_frames(sum(shuffledData(startIdx:endIdx, aID), 2), optBinSize(a));


        [varphi, varNoiseR(w,a)] = myYuleWalker3(wPopActivityR, pOrder);
        d2R(w,a) =getFixedPointDistance2(pOrder, critType, varphi);

    end
    toc
end
% [d21 d21R d22 d22R]
woody.popActivity = popActivity;
woody.startS = startS;
woody.d2 = d2;
woody.d2R = d2R;
woody.optBinSize = optBinSize;

%%
% woody40R = woody;
woodyOptR = woody;

%%
mins = startS/60;
figure(56); clf; hold on;
plot(mins, d2(:,1), 'ok', 'lineWidth', 2);
plot(mins, d2(:,2), 'ob', 'lineWidth', 2);
plot(mins, d2(:,3), 'or', 'lineWidth', 2);
plot(mins, d2(:,4), 'o', 'color', [0 .75 0], 'lineWidth', 2);
% plot(mins, d2(:,1), '-k', 'lineWidth', 2);
% plot(mins, d2(:,2), '-b', 'lineWidth', 2);
% plot(mins, d2(:,3), '-r', 'lineWidth', 2);
% plot(mins, d2(:,4), '-', 'color', [0 .75 0], 'lineWidth', 2);
plot(mins, d2R(:,1), '*k');
plot(mins, d2R(:,2), '*b');
plot(mins, d2R(:,3), '*r');
plot(mins, d2R(:,4), '*', 'color', [0 .75 0]);
xline(block2Start/60, 'linewidth', 2)
legend({'M23', 'M56', 'DS', 'VS'}, 'Location','northwest')
xlabel('Minutes')
ylabel('Distance to criticality')

h1 = plot(reachCorr/60, 0, '.', 'color', [0 .5 0], 'MarkerSize', 30);
h2 = plot(reachErr/60, -.02, '.', 'color', [.3 .3 .3], 'MarkerSize', 30);
set(h1, 'HandleVisibility', 'off');
set(h2, 'HandleVisibility', 'off');
title(['Reach Data ', num2str(windowSize), ' sec window, ' num2str(stepSize), ' sec steps'])
xlim([0 45])

%%
figure(14); clf
areaIdx = 2;
subplot(2,1,1)
plot(startS/60, d2(:,areaIdx), 'or', 'lineWidth', 2);
xlim([0 45])
subplot(2,1,2)
popSmooth = movmean(popActivity{areaIdx}, 21);
popMin = (0:length(popSmooth)-1) * optBinSize(areaIdx) / 60;
% plot(popMin, popSmooth);
plot(popMin, popActivity{areaIdx});
hold on;
h1 = plot(reachCorr/60,15, '.', 'color', [0 .5 0], 'MarkerSize', 10);
h2 = plot(reachErr/60, 16, '.', 'color', [.3 .3 .3], 'MarkerSize', 10);
set(h1, 'HandleVisibility', 'off');
set(h2, 'HandleVisibility', 'off');

xlim([0 45])



%%
figure(12);
popMin = (0:length(popActivity{areaIdx})-1) * optBinSize(areaIdx) / 60;
% imagesc(startS/60, 1:length(idDSR), dataMatR(:,idDSR)')
imagesc(popMin, 1:length(idDSR), dataMatR(:,idDSR)')
% Customize x-axis ticks
% xtickPositions = linspace(1, round(size(dataMatR, 1)/Fs/60), 5);  % Choose 5 tick positions evenly
% xticks(xtickPositions);  % Set tick positions
%
% % Set corresponding timestamp labels
% xticklabels(arrayfun(@(x) sprintf('%.2f', x), startS(round(xtickPositions)/60), 'UniformOutput', false));

%%
maxEig = computeMaxEigenvalue(varphi)


















%%   ====================================     Natrualistic: Sliding window       ==============================================


isiMult = 10; % Multiple of mean ISI to determine minimum bin size
pOrder = 10; % Order parameter for the autoregressor model
critType = 2;
binSize = .04;

% Define sliding window parameters
windowSize = 1 * 60; % (in seconds)
stepSize = 1; %1 * 60; %  (in seconds)
Fs = 1/opts.frameSize; % data is in ms

% Convert window and step size to samples
winSamples = round(windowSize * Fs);
stepSamples = round(stepSize * Fs);
numTimePoints = size(dataMat, 1);

% Preallocate or store outputs if needed
numWindows = floor((numTimePoints - winSamples) / stepSamples) + 1;


% Initialize variables
optBinSize = nan(length(areas), 1);
[d2,d2R, varNoise, varNoiseR] = ...
    deal(nan(numWindows, length(areas)));
startS = nan(numWindows, 1);
popActivity = cell(length(areas), 1);

shuffledData = shift_shuffle_neurons(dataMat);

for a = 1 : length(areas)
    fprintf('Area %s\n', areas{a})
    tic
    aID = idList{a};
    optBinSize(a) = isiMult * round(mean(diff(find(sum(dataMat(:, aID), 2))))) / 1000;
    optBinSize(a) = binSize;

    popActivity{a} = neural_matrix_ms_to_frames(sum(dataMat(:, aID), 2), optBinSize(a));

    for w = 1:numWindows
        startIdx = (w - 1) * stepSamples + 1;
        endIdx = startIdx + winSamples - 1;

        startS(w) = (startIdx + round(winSamples/2)) / Fs / 60;  % center of window
        wPopActivity = neural_matrix_ms_to_frames(sum(dataMat(startIdx:endIdx, aID), 2), optBinSize(a));

        [varphi, varNoise(w,a)] = myYuleWalker3(wPopActivity, pOrder);
        d2(w,a) =getFixedPointDistance2(pOrder, critType, varphi);

        % popActivityR = popActivity(randperm(length(popActivity)));
        wPopActivityR = neural_matrix_ms_to_frames(sum(shuffledData(startIdx:endIdx, aID), 2), optBinSize(a));

        [varphi, varNoiseR(w,a)] = myYuleWalker3(wPopActivityR, pOrder);
        d2R(w,a) =getFixedPointDistance2(pOrder, critType, varphi);

    end
    toc
end
% [d21 d21R d22 d22R]

woody.popActivity = popActivity;
woody.startS = startS;
woody.d2 = d2;
woody.d2R = d2R;
woody.optBinSize = optBinSize;

%%
woody40N = woody;
% woodyOptN = woody;

%%
save([paths.dropPath, 'criticality_sliding_woody.mat'], 'woody40R', 'woodyOptR', 'woody40N', 'woodyOptN');

%%
mins = startS/60;
figure(50); clf; hold on;
plot(mins, d2(:,1), 'ok', 'lineWidth', 2);
plot(mins, d2(:,2), 'ob', 'lineWidth', 2);
plot(mins, d2(:,3), 'or', 'lineWidth', 2);
plot(mins, d2(:,4), 'o', 'color', [0 .75 0], 'lineWidth', 2);
plot(mins, d2R(:,1), '*k');
plot(mins, d2R(:,2), '*b');
plot(mins, d2R(:,3), '*r');
plot(mins, d2R(:,4), '*', 'color', [0 .75 0]);
legend({'M23', 'M56', 'DS', 'VS'}, 'Location','northwest')
xlabel('Minutes')
ylabel('Distance to criticality')
title('Spontaneous')
























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

