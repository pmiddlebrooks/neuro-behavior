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
opts.collectFor = 1*60 * 60; % seconds

paths = get_paths;

tauRange = [1.2 2.5];
alphaRange = [1.5 2.2];
paramSDRange = [1.3 1.7];





monitorPositions = get(0, 'MonitorPositions');
monitorOne = monitorPositions(1, :); % Just use single monitor if you don't have second one
monitorTwo = monitorPositions(size(monitorPositions, 1), :); % Just use single monitor if you don't have second one



%% Naturalistic data
opts = neuro_behavior_options;
opts.frameSize = .001;
opts.minFiringRate = .05;
getDataType = 'spikes';
opts.collectStart = 0 * 60 * 60; % seconds
opts.collectFor = 1 * 60 * 60;
opts.firingRateCheckTime = 5 * 60;
get_standard_data

areas = {'M23', 'M56', 'DS', 'VS'};
idList = {idM23, idM56, idDS, idVS};

%% Mark's reach data
dataR = load(fullfile(paths.dropPath, 'reach_data/Y4_100623_Spiketimes_idchan_BEH.mat'));
[dataMatR, idLabels, areaLabels, rmvNeurons] = neural_matrix_mark_data(dataR, opts);
% Ensure dataMatR is same size as dataMat
idM23R = find(strcmp(areaLabels, 'M23'));
idM56R = find(strcmp(areaLabels, 'M56'));
idDSR = find(strcmp(areaLabels, 'DS'));
idVSR = find(strcmp(areaLabels, 'VS'));

areas = {'M23', 'M56', 'DS', 'VS'};
idList = {idM23R, idM56R, idDSR, idVSR};












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
































%%   ====================================     Mark Task: Sliding window       ==============================================


isiMult = 10; % Multiple of mean ISI to determine minimum bin size
pOrder = 10; % Order parameter for the autoregressor model
critType = 2;

% Define sliding window parameters
windowSize = 1 * 60; % 10 minutes (in seconds)
stepSize = 2; %1 * 60; % 2 minutes (in seconds)
Fs = 1000; % dataR is in ms

% Convert window and step size to samples
winSamples = round(windowSize * Fs);
stepSamples = round(stepSize * Fs);
numTimePoints = size(dataMatR, 1);

% Preallocate or store outputs if needed
numWindows = floor((numTimePoints - winSamples) / stepSamples) + 1;


% Initialize variables
optBinSize = nan(length(areas), 1);
[d2,d2R, varNoise, varNoiseR] = ...
    deal(nan(numWindows, length(areas)));
startMin = nan(numWindows, 1);

trialIdx2 = ismember(dataR.block(:, 3), 3:4);
block2First = find(trialIdx2, 1);
block2Start = dataR.R(block2First, 1);
block2StartMin = block2Start / Fs / 60;

shuffledData = shift_shuffle_neurons(dataMatR);

for a = 1 : length(areas)
    fprintf('Area %s\n', areas{a})
    tic
    aID = idList{a};
    optBinSize(a) = isiMult * round(mean(diff(find(sum(dataMatR(:, aID), 2))))) / 1000;
optBinSize(a) = .05;

for w = 1:numWindows
    startIdx = (w - 1) * stepSamples + 1;
    endIdx = startIdx + winSamples - 1;

    startMin(w) = startIdx / 1000 / 60;
    popActivity = neural_matrix_ms_to_frames(sum(dataMatR(startIdx:endIdx, aID), 2), optBinSize(a));

    [varphi, varNoise(w,a)] = myYuleWalker3(popActivity, pOrder);
    d2(w,a) =getFixedPointDistance2(pOrder, critType, varphi);

    % popActivityR = popActivity(randperm(length(popActivity)));
    popActivityR = sum(shuffledData(startIdx:endIdx, aID), 2);


    [varphi, varNoiseR(w,a)] = myYuleWalker3(popActivityR, pOrder);
    d2R(w,a) =getFixedPointDistance2(pOrder, critType, varphi);

end
toc
end
% [d21 d21R d22 d22R]
%
figure(52); clf; hold on;
plot(startMin, d2(:,1), 'ok', 'lineWidth', 2);
plot(startMin, d2(:,2), 'ob', 'lineWidth', 2);
plot(startMin, d2(:,3), 'or', 'lineWidth', 2);
plot(startMin, d2(:,4), 'o', 'color', [0 .75 0], 'lineWidth', 2);
plot(startMin, d2R(:,1), '*k');
plot(startMin, d2R(:,2), '*b');
plot(startMin, d2R(:,3), '*r');
plot(startMin, d2R(:,4), '*', 'color', [0 .75 0]);
xline(block2StartMin)
legend({'M23', 'M56', 'DS', 'VS'}, 'Location','northwest')
xlabel('Minutes')
ylabel('Distance to criticality')










%%   ====================================     Natrualistic: Sliding window       ==============================================


isiMult = 10; % Multiple of mean ISI to determine minimum bin size
pOrder = 10; % Order parameter for the autoregressor model
critType = 2;

% Define sliding window parameters
windowSize = 1 * 60; % 10 minutes (in seconds)
stepSize = 5; %1 * 60; % 2 minutes (in seconds)
Fs = 1/opts.frameSize; % dataR is in ms

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
startMin = nan(numWindows, 1);

shuffledData = shift_shuffle_neurons(dataMat);

for a = 1 : length(areas)
    fprintf('Area %s\n', areas{a})
    tic
    aID = idList{a};
    optBinSize(a) = isiMult * round(mean(diff(find(sum(dataMat(:, aID), 2))))) / 1000;
optBinSize(a) = .05;

for w = 1:numWindows
    startIdx = (w - 1) * stepSamples + 1;
    endIdx = startIdx + winSamples - 1;

    startMin(w) = startIdx / 1000 / 60;
    popActivity = neural_matrix_ms_to_frames(sum(dataMat(startIdx:endIdx, aID), 2), optBinSize(a));

    [varphi, varNoise(w,a)] = myYuleWalker3(popActivity, pOrder);
    d2(w,a) =getFixedPointDistance2(pOrder, critType, varphi);

    % popActivityR = popActivity(randperm(length(popActivity)));
    popActivityR = sum(shuffledData(startIdx:endIdx, aID), 2);

    [varphi, varNoiseR(w,a)] = myYuleWalker3(popActivityR, pOrder);
    d2R(w,a) =getFixedPointDistance2(pOrder, critType, varphi);

end
toc
end
% [d21 d21R d22 d22R]
%
figure(51); clf; hold on;
plot(startMin, d2(:,1), 'ok', 'lineWidth', 2);
plot(startMin, d2(:,2), 'ob', 'lineWidth', 2);
plot(startMin, d2(:,3), 'or', 'lineWidth', 2);
plot(startMin, d2(:,4), 'o', 'color', [0 .75 0], 'lineWidth', 2);
plot(startMin, d2R(:,1), '*k');
plot(startMin, d2R(:,2), '*b');
plot(startMin, d2R(:,3), '*r');
plot(startMin, d2R(:,4), '*', 'color', [0 .75 0]);
legend({'M23', 'M56', 'DS', 'VS'}, 'Location','northwest')
xlabel('Minutes')
ylabel('Distance to criticality')
title('Naturalistic')
























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
