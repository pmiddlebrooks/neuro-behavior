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
opts.minFiringRate = .05;
opts.frameSize = .001;

paths = get_paths;

tauRange = [1.2 2.5];
alphaRange = [1.5 2.2];
paramSDRange = [1.3 1.7];



monitorPositions = get(0, 'MonitorPositions');
monitorOne = monitorPositions(1, :); % Just use single monitor if you don't have second one
monitorTwo = monitorPositions(size(monitorPositions, 1), :); % Just use single monitor if you don't have second one



%% Naturalistic data
getDataType = 'spikes';
opts.firingRateCheckTime = 5 * 60;
opts.collectFor = 45 * 60; % seconds
get_standard_data

areas = {'M23', 'M56', 'DS', 'VS'};
idList = {idM23, idM56, idDS, idVS};


%% Mark's reach data
dataR = load(fullfile(paths.dropPath, 'reach_data/Y4_100623_Spiketimes_idchan_BEH.mat'));
[dataMatR, idLabels, areaLabels, rmvNeurons] = neural_matrix_mark_data(dataR, opts);

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
