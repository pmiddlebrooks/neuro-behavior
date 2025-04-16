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
opts.collectFor = 4*60 * 60; % seconds

paths = get_paths;

tauRange = [1.2 2.5];
alphaRange = [1.5 2.2];
paramSDRange = [1.3 1.7];





monitorPositions = get(0, 'MonitorPositions');
monitorOne = monitorPositions(1, :); % Just use single monitor if you don't have second one
monitorTwo = monitorPositions(size(monitorPositions, 1), :); % Just use single monitor if you don't have second one


%%   ====================================     Mark Task: Block1 vs Block2       ==============================================
opts = neuro_behavior_options;
opts.frameSize = .001;
opts.minFiringRate = .1;
getDataType = 'spikes';
opts.collectFor = 43 * 60;
opts.firingRateCheckTime = 5 * 60;
get_standard_data

%% Mark's reach data
dataR = load(fullfile(paths.dropPath, 'reach_data/Y4_100623_Spiketimes_idchan_BEH.mat'));
[dataMatR, idLabels, areaLabels, rmvNeurons] = neural_matrix_mark_data(dataR, opts);
% Ensure dataMatR is same size as dataMat
dataMatR = dataMatR(1:size(dataMat, 1),:);
idM23R = find(strcmp(areaLabels, 'M23'));
idM56R = find(strcmp(areaLabels, 'M56'));
idDSR = find(strcmp(areaLabels, 'DS'));
idVSR = find(strcmp(areaLabels, 'VS'));


%%
areas = {'M23', 'M56', 'DS', 'VS'};
idList = {idM23R, idM56R, idDSR, idVSR};

isiMult = 10; % Multiple of mean ISI to determine minimum bin size
pOrder = 10; % Order parameter for the autoregressor model
critType = 2;
optBinSize = nan(length(areas), 1);
[d21, d22, d21R, d22R] = ...
    deal(nan(length(areas), 1));

trialIdx1 = ismember(dataR.block(:, 3), 1:2);
trialIdx2 = ismember(dataR.block(:, 3), 3:4);
block1Last = find(trialIdx1, 1, 'last');
block2First = find(trialIdx2, 1);
block1End = dataR.R(block1Last, 2);
block2Start = dataR.R(block2First, 1);


for a = 1 : length(areas)
    fprintf('Area %s\n', areas{a})
    tic
    aID = idList{a};

    optBinSize(a) = isiMult * round(mean(diff(find(sum(dataMatR(:, aID), 2))))) / 1000;

    popActivity1 = neural_matrix_ms_to_frames(sum(dataMatR(1:block1End, aID), 2), optBinSize(a));
    popActivity2 = neural_matrix_ms_to_frames(sum(dataMatR(block2First:end, aID), 2), optBinSize(a));

    [varphi, varNoise] = myYuleWalker3(popActivity1, pOrder);
    d21(a) =getFixedPointDistance2(pOrder, critType, varphi);
    [varphi, varNoise] = myYuleWalker3(popActivity2, pOrder);
    d22(a) =getFixedPointDistance2(pOrder, critType, varphi);

    popActivity1R = popActivity1(randperm(length(popActivity1)));
    popActivity2R = popActivity2(randperm(length(popActivity2)));

    [varphi, varNoise] = myYuleWalker3(popActivity1R, pOrder);
    d21R(a) =getFixedPointDistance2(pOrder, critType, varphi);
    [varphi, varNoise] = myYuleWalker3(popActivity2R, pOrder);
    d22R(a) =getFixedPointDistance2(pOrder, critType, varphi);

end



%% =================        NATURALISTIC DATA              ================
areas = {'M23', 'M56', 'DS', 'VS'};
[optBinSize, d2, d2R] = ...
    deal(nan(length(areas), 1));
idList = {idM23, idM56, idDS, idVS};


for a = 1 : length(areas)
    fprintf('Area %s\n', areas{a})
    tic
    aID = idList{a};

    optBinSize(a) = isiMult * round(mean(diff(find(sum(dataMat(:, aID), 2))))) / 1000;

    popActivity1 = neural_matrix_ms_to_frames(sum(dataMat(:, aID), 2), optBinSize(a));
    % popActivity2 = neural_matrix_ms_to_frames(sum(dataMat(block2First:end, aID), 2), optBinSize(a));

    [varphi, varNoise] = myYuleWalker3(popActivity1, pOrder);
    d2(a) =getFixedPointDistance2(pOrder, critType, varphi);
    % [varphi, varNoise] = myYuleWalker3(popActivity2, pOrder);
    % d22(a) =getFixedPointDistance2(pOrder, critType, varphi);

    popActivity1R = popActivity1(randperm(length(popActivity1)));
    % popActivity2R = popActivity2(randperm(length(popActivity2)));

    [varphi, varNoise] = myYuleWalker3(popActivity1R, pOrder);
    d2R(a) =getFixedPointDistance2(pOrder, critType, varphi);
    % [varphi, varNoise] = myYuleWalker3(popActivity2R, pOrder);
    % d22R(a) =getFixedPointDistance2(pOrder, critType, varphi);

end





