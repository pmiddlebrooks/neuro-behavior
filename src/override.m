%%
opts = neuro_behavior_options;
opts.minActTime = .16;
opts.collectStart = 0 * 60 * 60; % seconds
opts.collectEnd = 30 * 60; % seconds

paths = get_paths;
%% Need to find an appropriate bin size for each brain area
opts.frameSize = .005; % DS
% opts.frameSize = .01; % M56

opts.useOverlappingBins = 0;
opts.windowSize = .2;
opts.stepSize = opts.frameSize;

getDataType = 'spikes';
get_standard_data

idSelect = idDS;
% idSelect = idM56;

%%
sumSpikes = sum(dataMat(:, idSelect), 2);
figure(91);
histogram(sumSpikes)
sum(sumSpikes == 0) / length(sumSpikes)



%% go to demoempdata.m

asdfMat = rastertoasdf2(dataMat(:,idSelect)', opts.frameSize*1000, 'CBModel', 'Spikes', 'DS')
Av = avprops(asdfMat, 'ratio', 'fingerprint');

% Go to demoempdata.m to run rest of codes using Av












%% Mark's reaching data
opts.frameSize = .005; % DS
opts.frameSize = .01; % DS
opts.frameSize = .0075; % M56
opts.minFiringRate = .1;

data = load(fullfile(paths.saveDataPath, 'reach_data/Y4_100623_Spiketimes_idchan.mat'));
[dataMat, idLabels, areaLabels, rmvNeurons] = reach_neural_matrix(data, opts);

idM23 = find(strcmp(areaLabels, 'M23'));
idM56 = find(strcmp(areaLabels, 'M56'));
idDS = find(strcmp(areaLabels, 'DS'));
idVS = find(strcmp(areaLabels, 'VS'));
fprintf('%d M23\n%d M56\n%d DS\n%d VS\n', length(idM23), length(idM56), length(idDS), length(idVS))

%%
idSelect = idDS;
idSelect = idM56;

%
sumSpikes = sum(dataMat(:, idSelect), 2);
figure(91);
histogram(sumSpikes, 'Normalization', 'pdf')
% sum(sumSpikes == 0) / length(sumSpikes)


%% go to demoempdata.m

asdfMat = rastertoasdf2(dataMat(:,idSelect)', opts.frameSize*1000, 'CBModel', 'Spikes', 'DS')


Av = avprops(asdfMat, 'ratio', 'fingerprint');

% Go to demoempdata.m to run rest of codes using Av










%% Set variables and get data
opts.minFiringRate = .1;
opts.collectEnd = 45 * 60;
getDataType = 'spikes';

nIter = 20;
nSubsample = 20;
binSizes = [.005, .01, .025 .05];
% binSizes = [.005, .05];
% binSizes = [.005];

% reach data
dataR = load(fullfile(paths.saveDataPath, 'reach_data/Y4_100623_Spiketimes_idchan.mat'));

brPeak = zeros(nIter, 2, length(binSizes));
tau = brPeak;
alpha = tau;
sigmaNuZInvSD = tau;
%% Set variables and get data
opts.minFiringRate = .1;
opts.collectEnd = 5 * 60;
getDataType = 'spikes';
opts.frameSize = .001;
get_standard_data


% Mark's reach data
dataR = load(fullfile(paths.dropPath, 'reach_data/Y4_100623_Spiketimes_idchan.mat'));
[dataMatR, idLabels, areaLabels, rmvNeurons] = reach_neural_matrix(dataR, opts);
% Ensure dataMatR is same size as dataMat
dataMatR = dataMatR(1:size(dataMat, 1),:);
idM23R = find(strcmp(areaLabels, 'M23'));
idM56R = find(strcmp(areaLabels, 'M56'));
idDSR = find(strcmp(areaLabels, 'DS'));
idVSR = find(strcmp(areaLabels, 'VS'));



%%    Is spontaneous data closer to criticality than reach data?
nIter = 1;
nSubsample = 20;

areas = {'M23', 'M56', 'DS', 'VS'};
[brPeak, tau, tauC, alpha, sigmaNuZInvSD, optBinSize, Av] = cell(length(areas), 1);


for a = 1 : length(areas)
    % brPeak = zeros(nIter, 2, length(binSizes));
    switch areas{a}
        case 'M23'
            aID = idM23;
            aIDR = idM23R;
        case 'M56'
            aID = idM56;
            aIDR = idM56R;
        case 'DS'
            aID = idDS;
            aIDR = idDSR;
        case 'VS'
            aID = idVS;
            aIDR = idVSR;
    end
    tic
    for iter = 1:nIter


        % spontaneous data
        % --------------------

        % Randomize a subsample of neurons
        idSelect = aID(randperm(length(aID), nSubsample));

        % Find optimal bin size for this group of neurons (to nearest ms)
        optBinSize{a}(iter, 1) = round(mean(diff(find(sum(dataMat(:, idSelect), 2))))) / 1000;
        if optBinSize{a}(iter, 1) == 0; optBinSize{a}(iter, 1) = .001; end
        fprintf('\nNatural\tIteration: %d\tBinSize: %.3f\n', iter, optBinSize{a}(iter, 1))

        dataMatNat = neural_matrix_ms_to_frames(dataMat(:, idSelect), optBinSize{a}(iter, 1));
        asdfMat = rastertoasdf2(dataMatNat', optBinSize{a}(iter, 1)*1000, 'CBModel', 'Spikes', 'area');
        Av{a}(1) = avprops(asdfMat, 'ratio', 'fingerprint');

        [brPeak{a}(iter, 1), tau{a}(iter, 1), tauC{a}(iter, 1), alpha{a}(iter, 1), sigmaNuZInvSD{a}(iter, 1)] = avalanche_log(Av{a}(1));



        % Mark reach data
        % --------------------

        % Randomize a subsample of neurons
        idSelect = aIDR(randperm(length(aIDR), nSubsample));

        % Find optimal bin size for this group of neurons
        optBinSize{a}(iter, 2) = round(mean(diff(find(sum(dataMatR(:, idSelect), 2))))) / 1000;
        if optBinSize{a}(iter, 2) == 0; optBinSize{a}(iter, 2) = .001; end
        fprintf('\nReach\tIteration: %d\tBinSize: %.3f\n', iter, optBinSize{a}(iter, 2))
        dataMatReach = neural_matrix_ms_to_frames(dataMatR(:, idSelect), optBinSize{a}(iter, 2));

        asdfMatR = rastertoasdf2(dataMatReach', optBinSize{a}(iter, 2)*1000, 'CBModel', 'Spikes', 'area');
        Av{a}(2) = avprops(asdfMatR, 'ratio', 'fingerprint');

        [brPeak{a}(iter, 2), tau{a}(iter, 2), tauC{a}(iter, 2), alpha{a}(iter, 2), sigmaNuZInvSD{a}(iter, 2)] = avalanche_log(Av{a}(2));

        fprintf('\n\nInteration %d\t %.1f\n\n', iter, toc/60)

    end

    slack_code_done
end
fileName = fullfile(paths.dropPath, 'avalanche_analyses.mat');
save(fileName, 'Av', 'brPeak', 'tau', 'alpha', 'sigmaNuZInvSD', 'optBinSize', 'areas', 'nSubsample', '-append')


%%    Is spontaneous data closer to criticality than reach data?
nIter = 10;
nSubsample = 20;
% binSizes = [.005, .01, .015, .02];
binSizes = [.01, .015, .02 .025];
% binSizes = [.005, .05];
% binSizes = [.005];

% brPeak = zeros(nIter, 2, length(binSizes));
[brPeak, tau, tauC, alpha, sigmaNuZInvSD, optBinSize] = deal(zeros(nIter, 2, length(binSizes)));


tic
for b = 1 : length(binSizes)
    for iter = 1:nIter
        % fprintf('\nIteration: %d\tBinSize: %.3f\t Time Elapsed: %.1f min\n', iter, binSizes(b), toc/60)

        opts.frameSize = binSizes(b);

        % Spontaneoush data4
        get_standard_data

        % Randomize a subsample of neurons
        idSelect = idVS(randperm(length(idVS), nSubsample));

        % Find optimal bin size for this group of neurons (to nearest ms)
        % optBinSize(iter, 1) = round(mean(diff(find(sum(dataMat(:, idSelect), 2))))) / 1000;
        % if optBinSize(iter, 1) == 0; optBinSize(iter, 1) = .001; end
        fprintf('\nNatural\tIteration: %d\tBinSize: %.3f\n', iter, opts.frameSize)

        asdfMat = rastertoasdf2(dataMat(:,idSelect)', opts.frameSize*1000, 'CBModel', 'Spikes', 'DS');
        Av = avprops(asdfMat, 'ratio', 'fingerprint');

        [brPeak(iter, 1, b), tau(iter, 1, b), tauC(iter, 1, b), alpha(iter, 1, b), sigmaNuZInvSD(iter, 1, b)] = avalanche_log(Av);



        dataR = load(fullfile(paths.dropPath, 'reach_data/Y4_100623_Spiketimes_idchan.mat'));
        dataMatR = reach_neural_matrix(dataR, opts);
        % Randomize a subsample of neurons
        idSelect = idVSR(randperm(length(idVSR), nSubsample));

        % Find optimal bin size for this group of neurons
        % optBinSize(iter, 2) = round(mean(diff(find(sum(dataMatR(:, idSelect), 2))))) / 1000;
        % if optBinSize(iter, 2) == 0; optBinSize(iter, 2) = .001; end
        fprintf('\nReach\tIteration: %d\tBinSize: %.3f\n', iter, opts.frameSize)

        asdfMatR = rastertoasdf2(dataMatR(:,idSelect)', opts.frameSize*1000, 'CBModel', 'Spikes', 'DS');
        AvR = avprops(asdfMatR, 'ratio', 'fingerprint');

        [brPeak(iter, 2, b), tau(iter, 2, b), tauC(iter, 2, b), alpha(iter, 2, b), sigmaNuZInvSD(iter, 2, b)] = avalanche_log(AvR);

        fprintf('\n\nInteration %d\t %.1f\n\n', iter, toc/60)

    end
end


%
brPeakVS = brPeak;
tauVS = tau;
tauCVS = tauC;
alphaVS = alpha;
sigmaNuZInvSDVS = sigmaNuZInvSD;

fileName = fullfile(paths.dropPath, 'criticality_parameters.mat');
save(fileName, 'brPeakVS', 'tauVS', 'tauCVS', 'alphaVS', 'sigmaNuZInvSDVS', '-append')

slack_code_done






















%%   =============     Test transition vs. within-bout criticality BETWEEN behaviors.   =============
opts.minFiringRate = .1;
opts.collectStart = 0;
opts.collectEnd = 45 * 60;
getDataType = 'spikes';

opts.frameSize = .001;
get_standard_data

%%
preTime = .2;
postTime = .2;

minBR = 0; maxBR = 2.5;
edges = minBR : .1 : maxBR;
centers = edges(1:end-1) + diff(edges) / 2;


areas = {'M23', 'M56', 'DS', 'VS'};
[tau, tauC, alpha, sigmaNuZInvSD] = deal(nan(length(areas), length(analyzeBhv), 2));
brHist = nan(length(areas), length(analyzeBhv), length(centers), 2);
optBinSize = nan(length(areas), 1);
idList = {idM23, idM56, idDS, idVS};

for a = 1 : length(areas)
    fprintf('Area %s\n', areas{a})
    tic
    aID = idList{a};

    % Find optimal bin size for avalannce analyses
    optBinSize = ceil(mean(diff(find(sum(dataMat(:, aID), 2))))) / 1000;
    if optBinSize == 0; optBinSize = .001; end


    % Make the matrices of the transitions and within-bouts at 1 kHz
    transWindow = round((-preTime/opts.frameSize : postTime/opts.frameSize - 1));

    for b = 1 : length(analyzeCodes)
        fprintf('Behavior: %s\n', analyzeBhv{b})
        bID = analyzeCodes(b);

        % First row of zeros so first spike counts as avalanche
    transMat = zeros(1, length(aID));
    withinMat = transMat;

        % Transition indices
        preIndAll = find(diff(bhvID) ~= 0); % 1 frame prior to all behavior transitions
        preInd = preIndAll(bhvID(preIndAll+1) == bID);

        for bout = 1 : length(preInd)-1
            % fprintf('bout %d\n', bout)


            % Collect transition avalanche data
            transInd = preInd(bout) + transWindow;
            % Only use transitions that are consitent behaviors going into and out
            % of the transition
            if transInd(1) > 0 && sum(diff(bhvID(transInd)) == 0) == length(transWindow) - 2

                % Convert the 1 kHz dataMats to the optimal bin size for avalanche analysis
                tempMat = neural_matrix_ms_to_frames(dataMat(transInd, aID), optBinSize);
                % Find avalances within the data
                zeroBins = find(sum(tempMat, 2) == 0);
                if length(zeroBins) > 1 && any(diff(zeroBins)>1)
                    transMat = [transMat; tempMat(zeroBins(1) : zeroBins(end)-1, :)];
                end
            end
            preIndNext = preIndAll(find(preIndAll > preInd(bout), 1));
            % Collect within-bout avalanche data
            withinInd = preInd(bout) + transWindow(end) + 1 : preIndNext + transWindow(1) - 1;
            % If there is any within-bout data...
            if ~isempty(withinInd)
                % Convert the 1 kHz dataMats to the optimal bin size for avalanche analysis
                tempMat = neural_matrix_ms_to_frames(dataMat(withinInd, aID), optBinSize);
                zeroBins = find(sum(tempMat, 2) == 0);
                if length(zeroBins) > 1 && any(diff(zeroBins)>1)
                    withinMat = [withinMat; tempMat(zeroBins(1) : zeroBins(end)-1, :)];
                end
            end

        end
    transMat = [transMat; zeros(1, size(transMat, 2))]; % add a final row of zeros to close out last avalanche
    withinMat = [withinMat; zeros(1, size(withinMat, 2))]; % add a final row of zeros to close out last avalanche




        % dataMatT = neural_matrix_ms_to_frames(transMat, optBinSize);
        asdfMat = rastertoasdf2(transMat', optBinSize*1000, 'CBModel', 'Spikes', 'trans');
        Av(a, b, 1) = avprops(asdfMat, 'ratio', 'fingerprint');
        brHist(a,b,:,1) = histcounts(Av(a, b, 1).branchingRatio, edges, 'Normalization','pdf');

        [tau(a, b, 1), tauC(a, b, 1), alpha(a, b, 1), sigmaNuZInvSD(a, b, 1)] = avalanche_log(Av(a, b, 1), 0);


        % dataMatW = neural_matrix_ms_to_frames(withinMat, optBinSize);
        asdfMat = rastertoasdf2(withinMat', optBinSize*1000, 'CBModel', 'Spikes', 'within');
        Av(a, b, 2) = avprops(asdfMat, 'ratio', 'fingerprint');
        brHist(a,b,:,2) = histcounts(Av(a, b, 2).branchingRatio, edges, 'Normalization','pdf');
        [tau(a, b, 2), tauC(a, b, 2), alpha(a, b, 2), sigmaNuZInvSD(a, b, 2)] = avalanche_log(Av(a, b, 2), 0);
    end
end
% fileName = fullfile(paths.dropPath, 'avalanche_analyses.mat');
% save(fileName, 'Av', 'brPeak', 'tau', 'alpha', 'sigmaNuZInvSD', 'optBinSize', 'areas', 'nSubsample', '-append')


%%
area = 2;
fig = figure(37); clf;
set(fig, 'Position', monitorTwo);
linewidth = 2;
ha = tight_subplot(2, ceil(length(analyzeCodes)/2), [0.05 0.02], [0.15 0.1], [0.07 0.05]);  % [gap, lower margin, upper margin]
for b = 1 : length(analyzeCodes)
    axes(ha(b));
    hold on; grid on;
plot(centers(2:end), reshape(brHist(area,b,2:end,1), length(centers(2:end)), 1), 'k', 'LineWidth',linewidth)
plot(centers(2:end), reshape(brHist(area,b,2:end,2), length(centers(2:end)), 1), 'r', 'LineWidth',linewidth)
title(analyzeBhv{b}, 'Interpreter','none')
    set(ha(b), 'XTickLabelMode', 'auto');  % Enable Y-axis labels
    if b == 1
        legend({'Trans', 'Within'})
    end
end
sgtitle([areas{1}, ' Branching Ratios Transitions vs Within-Bout'])
%%
figure(38); clf;
tauRange = [1.3 1.6];
alphaRange = [1.8 2.2];
sigmaRange = [1.2 1.5];
ha = tight_subplot(1, length(areas), [0.05 0.05], [0.15 0.1], [0.07 0.05]);  % [gap, lower margin, upper margin]
for a = 1 : length(areas)
    % Activate subplot
    axes(ha(a));
    hold on;
    plot(1, tau(a,1), 'ok', 'LineWidth', linewidth)
    plot(1, tau(a,2), 'or', 'LineWidth', linewidth)

    plot(2, tauC(a,1), 'ok', 'LineWidth', linewidth)
    plot(2, tauC(a,2), 'or', 'LineWidth', linewidth)

    plot(3, alpha(a,1), 'ok', 'LineWidth', linewidth)
    plot(3, alpha(a,2), 'or', 'LineWidth', linewidth)

    plot(4, sigmaNuZInvSD(a,1), 'ok', 'LineWidth', linewidth)
    plot(4, sigmaNuZInvSD(a,2), 'or', 'LineWidth', linewidth)

    plot([1 1], tauRange, 'b', 'LineWidth', linewidth)
    plot([2 2], tauRange, 'b', 'LineWidth', linewidth)
    plot([3 3], alphaRange, 'b', 'LineWidth', linewidth)
    plot([4 4], sigmaRange, 'b', 'LineWidth', linewidth)
    xticks(1:4)
    xticklabels({'tau', 'tauC', 'alpha', 'sigmaNuZInvSD'})
    xlim([.5 4.5])
    ylim([.5 5])
    grid on;
    title(areas{a})
end
legend({'ITI', 'Reaches'})
sgtitle('Avalanche Params Reaching vs ITI')



































%%   =======================     Sliding window of criticality parameters   =======================
% Slide a window from 1sec prior to 100ms after each time point
% Get criticality parameters at each time piont
% Assess metastability of criticality over time
opts.frameSize = .001;
opts.minFiringRate = .1;
getDataType = 'spikes';
opts.collectEnd = 30 * 60;
opts.firingRateCheckTime = 5 * 60;
get_standard_data

%%
preTime = 2;
postTime = .2;
stepSize = 0.15;       % Step size in seconds
numSteps = floor((size(dataMat, 1) / 1000 - stepSize) / stepSize) - 1;

% Initialize variables
areas = {'M23', 'M56', 'DS', 'VS'};
[brPeak, tau, tauC, alpha, sigmaNuZInvSD] = deal(cell(length(areas), 1));
idList = {idM23, idM56, idDS, idVS};
% Branching ratio histogram values
minBR = 0; maxBR = 4;
edges = minBR : .1 : maxBR;
centers = edges(1:end-1) + diff(edges) / 2;

% poolID = parpool(4, 'IdleTimeout', Inf);
% parfor a = 1  : length(areas)
for a = 1 : length(areas)
    tic
    aID = idList{a};
    optBinSize = ceil(mean(diff(find(sum(dataMat(:, aID), 2))))) / 1000;
    if optBinSize == 0; optBinSize = .001; end
    dataMatNat = neural_matrix_ms_to_frames(dataMat(:, aID), optBinSize);

    transWindow = (floor(-preTime/optBinSize) : ceil(postTime/optBinSize) - 1);

    % % Calculate number of windows to preallocate
    % numWindows = floor((size(dataMatNat, 1) - 1) / stepRows) + 1;

    [iTau, iTauC, iAlpha, iSigmaNuZInvSD] = deal(nan(numSteps, 1));

    iBrHist = nan(numSteps, length(centers));

    % Find firsst valid index to start collecting windowed data
    firstIdx = ceil(preTime / stepSize);
    for i = firstIdx: numSteps %-transWindow(1): size(dataMatNat, 1) - transWindow(end)
        % Find the index in dataMatNat for this step
        iRealTime = i * stepSize;
        iIdx = round(iRealTime / optBinSize);
        % Find the index in dataMatNat for this step
        % iIdx = -transWindow(1) + round((i-1) * stepRows);
        fprintf('\t%s\t %d of %d: %.3f  finished \n', areas{a}, i, numSteps, i/numSteps)
        % Make sure there are avalanches in the window
        zeroBins = find(sum(dataMatNat(iIdx + transWindow + 1, :), 2) == 0);
        if length(zeroBins) > 1 && any(diff(zeroBins)>1)

            asdfMat = rastertoasdf2(dataMatNat(iIdx + transWindow + 1, :)', optBinSize*1000, 'CBModel', 'Spikes', 'DS');
            Av = avprops(asdfMat, 'ratio', 'fingerprint');
            iBrHist(i,:) = histcounts(Av.branchingRatio, edges);
            [iTau(i), iTauC(i), iAlpha(i), iSigmaNuZInvSD(i)] = avalanche_log(Av, 0);
        end

    end
    fprintf('\nArea %s\t %.1f\n\n', areas{a}, toc/60)

    brPeak{a} = iBrHist;
    tau{a} = iTau;
    tauC{a} = iTauC;
    alpha{a} = iAlpha;
    sigmaNuZInvSD{a} = iSigmaNuZInvSD;
end
% delete(poolID)
%%
fileName = fullfile(paths.dropPath, 'avalanche_data_30min.mat');
% save(fileName, 'Av', 'brPeak', 'tau', 'alpha', 'sigmaNuZInvSD', 'optBinSize', 'areas', 'nSubsample', '-append')
% save(fileName, 'Av', 'brPeak', 'tau', 'alpha', 'sigmaNuZInvSD', 'optBinSize', 'areas')
save(fileName, 'brPeak', 'tau', 'alpha', 'sigmaNuZInvSD', 'optBinSize', 'centers', 'stepSize', 'preTime', 'postTme', 'areas')



























%% =======================    Mark's reaching vs ITI     =======================
opts.minFiringRate = .1;
opts.collectEnd = 45 * 60;
getDataType = 'spikes';
opts.frameSize = .001;

% Load Mark's reach data and make it a ms neural data matrix
dataR = load(fullfile(paths.dropPath, 'reach_data/Y4_100623_Spiketimes_idchan_BEH.mat'));
[dataMatR, idLabels, areaLabels, rmvNeurons] = reach_neural_matrix(dataR, opts);
idM23R = find(strcmp(areaLabels, 'M23'));
idM56R = find(strcmp(areaLabels, 'M56'));
idDSR = find(strcmp(areaLabels, 'DS'));
idVSR = find(strcmp(areaLabels, 'VS'));

% timeEmJs = dataR.und_time_EM_JS;
% align arduino time with neural time
% timeEmJs(:,1) = timeEmJs(:,1) + dataR.ArduinoOffset;



%%
preTime = .2; % ms before reach onset
postTime = 2;

areas = {'M23', 'M56', 'DS', 'VS'};


% Branching ratio histogram values
minBR = 0; maxBR = 4;
edges = minBR : .1 : maxBR;
centers = edges(1:end-1) + diff(edges) / 2;


[tau, tauC, alpha, sigmaNuZInvSD] = deal(nan(length(areas), 2));
brHist = nan(length(areas), length(centers), 2);
optBinSize = nan(length(areas), 1);
idList = {idM23R, idM56R, idDSR, idVSR};

% Find all reach starts
block1Corr = 2;
block2Corr = 3;
block1Cor = dataR.block(:, 3) == block1Corr;

for a = 1 : length(areas)
    fprintf('Area %s\n', areas{a})
    tic
    aID = idList{a};

    optBinSize(a) = ceil(mean(diff(find(sum(dataMatR(:, aID), 2))))) / 1000;
    if optBinSize(a) == 0; optBinSize(a) = .001; end
reachWindow = floor(-preTime/optBinSize(a)) : ceil(postTime/optBinSize(a)) - 1;
baseWindow = reachWindow - floor(3/optBinSize(a)) - 1;

    dataMatReach = neural_matrix_ms_to_frames(dataMatR(:, aID), optBinSize(a));


    rStarts = round(dataR.R(block1Cor,1)/optBinSize(a)/1000);  % in frame time

    % initialize data mats: start with a row of zeros so first spike counts as
    % avalanche
    baseMat = zeros(1, size(dataMatReach, 2));
    reachMat = zeros(1, size(dataMatReach, 2));
    for i = 1 : length(rStarts) - 1


        % baseline matrix
        iBaseMat = dataMatReach(rStarts(i) + baseWindow, :);
        % Find avalances within the data
        zeroBins = find(sum(iBaseMat, 2) == 0);
        if length(zeroBins) > 1 && any(diff(zeroBins)>1)
            baseMat = [baseMat; iBaseMat(zeroBins(1) : zeroBins(end)-1, :)];
        end

        % reach data matrix
        iReachMat = dataMatReach(rStarts(i) + reachWindow, :);
        % Find avalances within the data
        zeroBins = find(sum(iReachMat, 2) == 0);
        if length(zeroBins) > 1 && any(diff(zeroBins)>1)
            reachMat = [reachMat; iReachMat(zeroBins(1) : zeroBins(end)-1, :)];
        end

    end
    baseMat = [baseMat; zeros(1, size(baseMat, 2))]; % add a final row of zeros to close out last avalanche
    reachMat = [reachMat; zeros(1, size(reachMat, 2))]; % add a final row of zeros to close out last avalanche

    plotFlag = 0;

    asdfMat = rastertoasdf2(baseMat', optBinSize*1000, 'CBModel', 'Spikes', 'reach');
    Av(a, 1) = avprops(asdfMat, 'ratio', 'fingerprint');
    brHist(a,:, 1) = histcounts(Av(a, 1).branchingRatio, edges, 'Normalization','pdf');
    [tau(a, 1), tauC(a, 1), alpha(a, 1), sigmaNuZInvSD(a, 1)] = avalanche_log(Av(a, 1), plotFlag);


    %
    asdfMat = rastertoasdf2(reachMat', optBinSize*1000, 'CBModel', 'Spikes', 'ITI');
    Av(a, 2) = avprops(asdfMat, 'ratio', 'fingerprint');
    brHist(a,:, 2) = histcounts(Av(a, 2).branchingRatio, edges, 'Normalization','pdf');
    [tau(a, 2), tauC(a, 2), alpha(a, 2), sigmaNuZInvSD(a, 2)] = avalanche_log(Av(a, 2), plotFlag);

    % [tauB, alphaB, sigmaNuZInvSDB]
    % [tau, alpha, sigmaNuZInvSD]
end


%%
area = 4;
figure(34); clf; hold on;
linewidth = 2;
% for a = 1 : length(areas)
plot(centers(2:end), brHist(area,2:end,1), 'k', 'LineWidth',linewidth)
plot(centers(2:end), brHist(area,2:end,2), 'r', 'LineWidth',linewidth)
% end
%%
figure(35); clf;
tauRange = [1.3 1.6];
alphaRange = [1.8 2.2];
sigmaRange = [1.2 1.5];
ha = tight_subplot(1, length(areas), [0.05 0.05], [0.15 0.1], [0.07 0.05]);  % [gap, lower margin, upper margin]
for a = 1 : length(areas)
    % Activate subplot
    axes(ha(a));
    hold on;
    plot(1, tau(a,1), 'ok', 'LineWidth', linewidth)
    plot(1, tau(a,2), 'or', 'LineWidth', linewidth)

    plot(2, tauC(a,1), 'ok', 'LineWidth', linewidth)
    plot(2, tauC(a,2), 'or', 'LineWidth', linewidth)

    plot(3, alpha(a,1), 'ok', 'LineWidth', linewidth)
    plot(3, alpha(a,2), 'or', 'LineWidth', linewidth)

    plot(4, sigmaNuZInvSD(a,1), 'ok', 'LineWidth', linewidth)
    plot(4, sigmaNuZInvSD(a,2), 'or', 'LineWidth', linewidth)

    plot([1 1], tauRange, 'b', 'LineWidth', linewidth)
    plot([2 2], tauRange, 'b', 'LineWidth', linewidth)
    plot([3 3], alphaRange, 'b', 'LineWidth', linewidth)
    plot([4 4], sigmaRange, 'b', 'LineWidth', linewidth)
    xticks(1:4)
    xticklabels({'tau', 'tauC', 'alpha', 'sigmaNuZInvSD'})
    xlim([.5 4.5])
    ylim([.5 5])
    grid on;
    title(areas{a})
end
legend({'ITI', 'Reaches'})
sgtitle('Avalanche Params Reaching vs ITI')






















%% =======================    Behavior scale-free?     =======================
% Define behaviors to include (input as a vector)
bhvInclude = analyzeCodes           ; % Modify this as needed
someBhvs = 'groom';
bhvInclude = codes(contains(behaviors, someBhvs));

% Load behavior labels (bhvID) - Ensure bhvID is a column vector
bhvID = bhvID(:);

% Filter bhvID to only include specified behaviors
include_mask = ismember(bhvID, bhvInclude);
bhvFiltered = bhvID(include_mask);

% Compute behavior durations (time spent in each behavior state)
change_points = find(diff([NaN; bhvFiltered]) ~= 0); % Find transitions
durations = diff(change_points); % Compute durations

% Fit power-law distribution using Clauset's plfit (ensure you have plfit.m)
[alpha, xmin, L] = plfit(durations);

% Compare with exponential and lognormal distributions
% Exponential fit
lambda = 1/mean(durations);
L_exp = sum(log(lambda * exp(-lambda * durations)));

% Lognormal fit
mu = mean(log(durations));
sigma = std(log(durations));
L_lognorm = sum(log((1./(durations * sigma * sqrt(2*pi))) .* exp(-((log(durations)-mu).^2) / (2*sigma^2))));

% Compare likelihoods
R_exp = 2 * (L - L_exp);
R_lognorm = 2 * (L - L_lognorm);

% Compute p-values
p_exp = 1 - chi2cdf(R_exp, 1);
p_lognorm = 1 - chi2cdf(R_lognorm, 1);

% Display results
fprintf('Power-law exponent (alpha): %.3f\n', alpha);
fprintf('Comparison with exponential: R=%.3f, p=%.3f\n', R_exp, p_exp);
fprintf('Comparison with lognormal: R=%.3f, p=%.3f\n', R_lognorm, p_lognorm);

% Plot the distribution
figure;
% Define logarithmic bins
binEdges = logspace(log10(min(durations)), log10(max(durations)), 20); % Adjust number of bins
% Compute histogram
[counts, edges] = histcounts(durations, binEdges, 'Normalization', 'pdf');
% Compute bin centers
binCenters = sqrt(edges(1:end-1) .* edges(2:end));
bar(binCenters, counts, 'histc');
hold on;
plot(sort(durations), (xmin ./ sort(durations)).^alpha, 'r', 'LineWidth', 2);
set(gca, 'XScale', 'log', 'YScale', 'log');
xlabel('Duration'); ylabel('Probability Density');
title('Scale-Free Test: Behavior Durations');
legend('Empirical Data', 'Power-Law Fit');








%%
function [tau, tauC, alpha, sigmaNuZInvSD] = avalanche_log(Av, plotFlag)

if plotFlag == 1
    plotFlag = 'plot';
else
    plotFlag = 'nothing';
end

% size distribution (SZ)
[tau, xminSZ, xmaxSZ, sigmaSZ, pSZ, pCritSZ, ksDR, DataSZ] =...
    avpropvals(Av.size, 'size', plotFlag);
tau = cell2mat(tau);

% size distribution (SZ) with cutoffs
tauC = nan;
UniqSizes = unique(Av.size);
Occurances = hist(Av.size,UniqSizes);
% AllowedSizes = UniqSizes(Occurances >= 20);
AllowedSizes = UniqSizes(Occurances >= 10);
% AllowedSizes = UniqSizes(Occurances >= 2);
AllowedSizes(AllowedSizes < 4) = [];
% AllowedSizes(AllowedSizes < 3) = [];
if length(AllowedSizes) > 1
    LimSize = Av.size(ismember(Av.size,AllowedSizes));
    [tauC, xminSZ, xmaxSZ, sigmaSZ, pSZ, pCritSZ, DataSZ] =...
        avpropvals(LimSize, 'size', plotFlag);
    tauC = cell2mat(tauC);
end

% duration distribution (DR)
[alpha, xminDR, xmaxDR, sigmaDR, pDR, pCritDR, ksDR, DataDR] =...
    avpropvals(Av.duration, 'duration', plotFlag);
alpha = cell2mat(alpha);

% size given duration distribution (SD)
[sigmaNuZInvSD, waste, waste, sigmaSD] = avpropvals({Av.size, Av.duration},...
    'sizgivdur', 'durmin', xminDR{1}, 'durmax', xmaxDR{1}, plotFlag);
end

