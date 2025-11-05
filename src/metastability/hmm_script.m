%%
opts = neuro_behavior_options;
opts.minActTime = .16;
opts.minFiringRate = .05;
opts.frameSize = .001;

paths = get_paths;

monitorPositions = get(0, 'MonitorPositions');
monitorOne = monitorPositions(1, :); % Just use single monitor if you don't have second one
monitorTwo = monitorPositions(size(monitorPositions, 1), :); % Just use single monitor if you don't have second one






%%           ==========================         WHICH DATA DO YOU WANT TO ANALYZE?        =================================
% Naturalistic data
getDataType = 'spikes';
opts.collectStart = 0 * 60 * 60; % seconds
opts.collectEnd = 45 * 60; % seconds
opts.firingRateCheckTime = 5 * 60;
opts.frameSize = .03;
opts.frameSize = .001;
% opts.windowSize = .1;
% opts.method = 'useOverlap';
get_standard_data

areas = {'M23', 'M56', 'DS', 'VS'};
idList = {idM23, idM56, idDS, idVS};

dataMatMain = dataMat;

%% Mark's reach data
dataR = load(fullfile(paths.dropPath, 'reach_data/Y4_100623_Spiketimes_idchan_BEH.mat'));

opts.frameSize = .001;
% opts.windowSize = .1;
% opts.method = 'useOverlap';
[dataMatR, idLabels, areaLabels, rmvNeurons] = neural_matrix_mark_data(dataR, opts);

% Get data until 1 sec after the last reach ending.
cutOff = round((dataR.R(end,2) + 1000) / 1000 / opts.frameSize);
dataMatR = dataMatR(1:cutOff,:);

% Ensure dataMatR is same size as dataMat
idM23 = find(strcmp(areaLabels, 'M23'));
idM56 = find(strcmp(areaLabels, 'M56'));
idDS = find(strcmp(areaLabels, 'DS'));
idVS = find(strcmp(areaLabels, 'VS'));

areas = {'M23', 'M56', 'DS', 'VS'};
idList = {idM23, idM56, idDS, idVS};

halfSet = round(size(dataMatR, 1) / 2);
dataMatMain = dataMatR(1:halfSet, :);











%%  Fit a gaussian HMM in PCA space (up to a certain explained variance)
idMain = idM56;

expThresh = 70;   % percent explained variance to collect pca component scores for.

[coeff, score, ~, ~, explained, mu] = pca(dataMatMain(:, idMain));
nDim = find(cumsum(explained) > expThresh, 1);

opts.stateRange = 3:30;
opts.numReps = 5;
opts.numFolds = 3;
opts.margLikMethod = 'laplace'; %importance
opts.numSamples = 100;
opts.selectBy = 'margLik';
opts.plotFlag = 1;

[bestModel, bestNumStates, stateSeq, allModels, allLogL, allBIC, allMargLik] = ...
    fit_gaussian_hmm(score(:,1:nDim), opts);

%%
posteriorProb = posterior(bestModel, score(:,1:nDim));

% Plot posterior probability of each state across time
figure;
imagesc(posteriorProb'); colorbar;
xlabel('Time'); ylabel('State');
title('Posterior Probabilities of States');










