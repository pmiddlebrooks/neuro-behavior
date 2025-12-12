%%
% Initialize paths
paths = get_paths;

% Initialize options structure
opts = neuro_behavior_options;
opts.frameSize = .001;
opts.firingRateCheckTime = 5 * 60;
opts.collectStart = 0;
opts.minFiringRate = .05;
opts.maxFiringRate = 200;

sessionName =  'AB6_03-Apr-2025 13_34_09_NeuroBeh.mat';

% Load reach data
reachDataFile = fullfile(paths.reachDataPath, sessionName);

[~, dataBaseName, ~] = fileparts(reachDataFile);
saveDir = fullfile(paths.dropPath, 'reach_task/results', dataBaseName);
if ~exist(saveDir, 'dir'); mkdir(saveDir); end

dataR = load(reachDataFile);
opts.collectEnd = round(min(dataR.R(end,1) + 5000, max(dataR.CSV(:,1)*1000)) / 1000);

% Load reach spike data
[dataMat, idLabels, areaLabels] = neural_matrix_mark_data(dataR, opts);
areas = {'M23', 'M56', 'DS', 'VS'};
idM23 = find(strcmp(areaLabels, 'M23'));
idM56 = find(strcmp(areaLabels, 'M56'));
idDS = find(strcmp(areaLabels, 'DS'));
idVS = find(strcmp(areaLabels, 'VS'));
idMatIdx = {idM23, idM56, idDS, idVS};
idLabel = {idLabels(idM23), idLabels(idM56), idLabels(idDS), idLabels(idVS)};

popAct = mean(dataMat(:,idM56), 2);
timePoints = opts.frameSize * (1:length(popAct));

%% Behavior data
 % block or BlockWithErrors - columns are BlockNum Correct? ReachClassification1-4 ReachClassification-20-20
 %     -rows are reaches
 %     classification1-4:
 %     - 1 - block 1 error
 %     - 2 - block 1 Rew
 %     - 3 - block 2 error
 %     - 4 - block 2 rew
 %     - 5 - block 3 error
 %     - 6 - block 3 rew
 % 
 % 
 %     classification-20-20:
 %     - -10 - block 1 error below
 %     -  1   - block 1 Rew
 %     - 10 - block 1 error above
 %     - -20 - block 2 error below
 %     -  2   - block 2 Rew
 %     - 20 - block 2 error above

 reachClass = dataR.Block(:,3);
startBlock2Idx = reachStart(find(ismember(reachClass, [3 4]), 1));
reachStart = dataR.R(:,1);
reachStop = dataR.R(:,2);
reachAmp = dataR.R(:,3); % Amplitude of each reach (distance from 0)

% NIDATA indices
ERtraceIdx = 4;
SOLtraceIdx = 5;
lickSensorIdx = 6;
jsXIdx = 7;
jsYIdx = 8;

% Continuous traces normalized to max
jsTrace = zscore(dataR.JSNIpos); % Continuous absolute value of amplitude trace of joystick jsTrace = sqrt(x^2 + y^2)
jsX = zscore(dataR.NIDATA(jsXIdx,2:end)); % Continuous absolute value of amplitude trace of joystick jsTrace = sqrt(x^2 + y^2)
jsY = zscore(dataR.NIDATA(jsYIdx,2:end)); % Continuous absolute value of amplitude trace of joystick jsTrace = sqrt(x^2 + y^2)
jsAmp = sqrt(dataR.NIDATA(jsXIdx,2:end).^2 + dataR.NIDATA(8,2:end).^2);
jsAmp = movmean(jsAmp, 61);
jsAmp = zscore(jsAmp); % Continuous absolute value of amplitude trace of joystick jsTrace = sqrt(x^2 + y^2)

% jsTrace = traces.JSNIpos ./ max(traces.JSNIpos); % Continuous absolute value of amplitude trace of joystick jsTrace = sqrt(x^2 + y^2)
errTrace = dataR.NIDATA(ERtraceIdx,2:end) ./ max(dataR.NIDATA(ERtraceIdx,2:end)); % Continuous voltage trace of error light (for error trials)
solTrace = dataR.NIDATA(SOLtraceIdx,2:end) ./ max(dataR.NIDATA(SOLtraceIdx,2:end)); % Continous voltage trace of solenoid (for correct rewarded trials)

% Lick times
filtLick = bandpass(dataR.NIDATA(lickSensorIdx,2:end), [1 10], 1000); 
filtLick = abs(filtLick); 
filtLick(filtLick>300)=300;
filtLick = movmean(filtLick, 101);
filtLick = zscore(filtLick);
% Event times
errTimes = dataR.ERtimes/1000; % Error light turns on (errors)
solTimes = dataR.SOLtimes/1000; % Reward solenoid opens (correct)
