%%
opts = neuro_behavior_options;
opts.minActTime = .16;
opts.collectStart = 0 * 60 * 60; % seconds
opts.collectFor = 30 * 60; % seconds

paths = get_paths;

tauRange = [1.2 2.5];
alphaRange = [1.5 2.2];
sigmaRange = [1.3 1.7];




















%%   ====================================       Naturalistic: Check a few min of each area       ==============================================
opts = neuro_behavior_options;
opts.frameSize = .001;
opts.minFiringRate = .1;
getDataType = 'spikes';
opts.collectFor = 10 * 60;
opts.firingRateCheckTime = 2 * 60;
get_standard_data

%%
pcaFlag = 1;
pcaFirstFlag = 0;
thresholdFlag = 1;
thresholdBinSize = .05;


% Initialize variables
areas = {'M23', 'M56', 'DS', 'VS'};
[optBinSize, tau, tauC, alpha, sigmaNuZInvSD] = deal(nan(length(areas), 1));
idList = {idM23, idM56, idDS, idVS};
% Branching ratio histogram values
minBR = 0; maxBR = 3;
edges = minBR : .1 : maxBR;
centers = edges(1:end-1) + diff(edges) / 2;
brHist = nan(length(areas), length(centers));

tic
for a = 1 : length(areas)
    aID = idList{a};

    % If using the threshold method
    if thresholdFlag
        optBinSize(a) = thresholdBinSize;
    else
        optBinSize(a) = optimal_bin_size(dataMat(:,aID));
    end
    fprintf('-------------\tArea: %s\n', areas{a})

    dataMatNat = neural_matrix_ms_to_frames(dataMat(:, aID), optBinSize(a));

    if pcaFlag
        [coeff, score, ~, ~, explained, mu] = pca(dataMatNat);
        forDim = find(cumsum(explained) > 30, 1);
        forDim = max(3, forDim);
        forDim = min(6, forDim);
        if pcaFirstFlag
            fprintf('Using PCA first %d\n', forDim)
            nDim = 1:forDim;
        else
            fprintf('Using PCA Last many from %d to %d\n', forDim+1, size(score, 2))
            nDim = forDim+1:size(score, 2);
        end
        dataMatNat = score(:,nDim) * coeff(:,nDim)' + mu;
    end

    if thresholdFlag
        dataMatNat = round(sum(dataMatNat, 2));
        threshPct = .08;
        % threshSpikes = threshPct * max(dataMatReach);
        threshSpikes = .8*median(dataMatNat);
        dataMatNat(dataMatNat < threshSpikes) = 0;
        fprintf('Using Threshold method \tBinsize: %.3f\tProp zeros: %.3f\n', optBinSize(a), sum(dataMatNat == 0) / length(dataMatNat))
    end

    asdfMat = rastertoasdf2(dataMatNat', optBinSize(a)*1000, 'CBModel', 'Spikes', 'DS');
    Av(a) = avprops(asdfMat, 'ratio', 'fingerprint');
    br = Av(a).branchingRatio;

    brHist(a,:) = histcounts(br(br>0), edges, 'Normalization', 'pdf');
    [tau(a), tauC(a), alpha(a), sigmaNuZInvSD(a), decades(a)] = avalanche_log(Av(a), 0);

end
toc/60
%%
figure(36); clf;
ha = tight_subplot(1, length(areas), [0.05 0.05], [0.15 0.1], [0.07 0.05]);  % [gap, lower margin, upper margin]
for a = 1 : length(areas)
    axes(ha(a));
    hold on;
    linewidth = 2;
    % for a = 1 : length(areas)
    plot(centers(2:end), brHist(a,2:end), 'k', 'LineWidth',linewidth)
    % plot(centers(2:end), brHist(a,2:end,2), 'r', 'LineWidth',linewidth)
    xlim([0 2])
    grid on;
    set(ha(a), 'XTickLabelMode', 'auto');  % Enable Y-axis labels
    title([areas{a}])
end
legend({'ITI', 'Reach'})
sgtitle('Branching Ratio PDFs')
copy_figure_to_clipboard
%
pause
figure(37); clf;
ha = tight_subplot(1, length(areas), [0.05 0.05], [0.15 0.1], [0.07 0.05]);  % [gap, lower margin, upper margin]
for a = 1 : length(areas)
    % Activate subplot
    axes(ha(a));
    hold on;
    plot(1, tau(a), 'ok', 'LineWidth', linewidth)
    % plot(1, tau(a,2), 'or', 'LineWidth', linewidth)

    plot(2, tauC(a), 'ok', 'LineWidth', linewidth)
    % plot(2, tauC(a,2), 'or', 'LineWidth', linewidth)

    plot(3, alpha(a), 'ok', 'LineWidth', linewidth)
    % plot(3, alpha(a,2), 'or', 'LineWidth', linewidth)

    plot(4, sigmaNuZInvSD(a), 'ok', 'LineWidth', linewidth)
    % plot(4, sigmaNuZInvSD(a,2), 'or', 'LineWidth', linewidth)

    plot([1 1], tauRange, 'b', 'LineWidth', 1)
    plot([2 2], tauRange, 'b', 'LineWidth', 1)
    plot([3 3], alphaRange, 'b', 'LineWidth', 1)
    plot([4 4], sigmaRange, 'b', 'LineWidth', 1)
    xticks(1:4)
    xticklabels({'tau', 'tauC', 'alpha', 'sigmaNuZInvSD'})
    xlim([.5 4.5])
    ylim([.5 5])
    grid on;
    title(areas{a})
end
% legend({'ITI', 'Reaches'})
sgtitle('Avalanche Params Reaching vs ITI')
copy_figure_to_clipboard



















%% Set variables and get data
opts.minFiringRate = .1;
opts.collectFor = 45 * 60;
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
opts.collectFor = 5 * 60;
getDataType = 'spikes';
opts.frameSize = .001;
get_standard_data


% Mark's reach data
dataR = load(fullfile(paths.dropPath, 'reach_data/Y4_100623_Spiketimes_idchan.mat'));
[dataMatR, idLabels, areaLabels, rmvNeurons] = neural_matrix_mark_data(dataR, opts);
% Ensure dataMatR is same size as dataMat
dataMatR = dataMatR(1:size(dataMat, 1),:);
idM23R = find(strcmp(areaLabels, 'M23'));
idM56R = find(strcmp(areaLabels, 'M56'));
idDSR = find(strcmp(areaLabels, 'DS'));
idVSR = find(strcmp(areaLabels, 'VS'));



%%    Is naturalistic data closer to criticality than reach data?
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


        % Naturalistic data
        % --------------------

        % Randomize a subsample of neurons
        idSelect = aID(randperm(length(aID), nSubsample));

        % Find optimal bin size for this group of neurons (to nearest ms)
        optBinSize{a} = optimal_bin_size(dataMat(:, idSelect));
        fprintf('\nNatural\tIteration: %d\tBinSize: %.3f\n', iter, optBinSize{a}(iter, 1))

        dataMatNat = neural_matrix_ms_to_frames(dataMat(:, idSelect), optBinSize{a}(iter, 1));
        asdfMat = rastertoasdf2(dataMatNat', optBinSize{a}(iter, 1)*1000, 'CBModel', 'Spikes', 'area');
        Av{a}(1) = avprops(asdfMat, 'ratio', 'fingerprint');

        [brPeak{a}(iter, 1), tau{a}(iter, 1), tauC{a}(iter, 1), alpha{a}(iter, 1), sigmaNuZInvSD{a}(iter, 1), decades{a}(iter, 1)] = avalanche_log(Av{a}(1));



        % Mark reach data
        % --------------------

        % Randomize a subsample of neurons
        idSelect = aIDR(randperm(length(aIDR), nSubsample));

        % Find optimal bin size for this group of neurons
        optBinSize{a}(iter, 2) = optimal_bin_size(dataMatR(:, idSelect));
        fprintf('\nReach\tIteration: %d\tBinSize: %.3f\n', iter, optBinSize{a}(iter, 2))
        dataMatReach = neural_matrix_ms_to_frames(dataMatR(:, idSelect), optBinSize{a}(iter, 2));

        asdfMatR = rastertoasdf2(dataMatReach', optBinSize{a}(iter, 2)*1000, 'CBModel', 'Spikes', 'area');
        Av{a}(2) = avprops(asdfMatR, 'ratio', 'fingerprint');

        [brPeak{a}(iter, 2), tau{a}(iter, 2), tauC{a}(iter, 2), alpha{a}(iter, 2), sigmaNuZInvSD{a}(iter, 2), decades{a}(iter, 2)] = avalanche_log(Av{a}(2));

        fprintf('\n\nInteration %d\t %.1f\n\n', iter, toc/60)

    end

    slack_code_done
end
fileName = fullfile(paths.dropPath, 'avalanche_analyses.mat');
save(fileName, 'Av', 'brPeak', 'tau', 'alpha', 'sigmaNuZInvSD', 'optBinSize', 'areas', 'nSubsample', '-append')


%%    Is naturalistic data closer to criticality than reach data?
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

        % Naturalistich data4
        get_standard_data

        % Randomize a subsample of neurons
        idSelect = idVS(randperm(length(idVS), nSubsample));

        % Find optimal bin size for this group of neurons (to nearest ms)
        fprintf('\nNatural\tIteration: %d\tBinSize: %.3f\n', iter, opts.frameSize)

        asdfMat = rastertoasdf2(dataMat(:,idSelect)', opts.frameSize*1000, 'CBModel', 'Spikes', 'DS');
        Av = avprops(asdfMat, 'ratio', 'fingerprint');

        [brPeak(iter, 1, b), tau(iter, 1, b), tauC(iter, 1, b), alpha(iter, 1, b), sigmaNuZInvSD(iter, 1, b), decades(iter, 1, b)] = avalanche_log(Av);



        dataR = load(fullfile(paths.dropPath, 'reach_data/Y4_100623_Spiketimes_idchan.mat'));
        dataMatR = neural_matrix_mark_data(dataR, opts);
        % Randomize a subsample of neurons
        idSelect = idVSR(randperm(length(idVSR), nSubsample));

        % Find optimal bin size for this group of neurons
        fprintf('\nReach\tIteration: %d\tBinSize: %.3f\n', iter, opts.frameSize)

        asdfMatR = rastertoasdf2(dataMatR(:,idSelect)', opts.frameSize*1000, 'CBModel', 'Spikes', 'DS');
        AvR = avprops(asdfMatR, 'ratio', 'fingerprint');

        [brPeak(iter, 2, b), tau(iter, 2, b), tauC(iter, 2, b), alpha(iter, 2, b), sigmaNuZInvSD(iter, 2, b), decades(iter, 2, b)] = avalanche_log(AvR);

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
opts = neuro_behavior_options;
opts.minFiringRate = .1;
opts.collectStart = 0;
opts.collectFor = 45 * 60;
getDataType = 'spikes';

opts.frameSize = .001;
get_standard_data

%%
pcaFlag = 0;
pcaFirstFlag = 0;
thresholdFlag = 1;
thresholdBinSize = .02;

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
    % If using the threshold method
    if thresholdFlag
        optBinSize(a) = thresholdBinSize;
    else
        optBinSize(a) = optimal_bin_size(dataMat(:, aID));
    end


    % Make the matrices of the transitions and within-bouts at 1 kHz
    transWindow = round((-preTime/opts.frameSize : postTime/opts.frameSize - 1));

    for b = 1 : length(analyzeCodes)
        fprintf('Behavior: %s\n', analyzeBhv{b})
        bID = analyzeCodes(b);



        % First row of zeros so first spike counts as avalanche
        if thresholdFlag
            transMat = 0;
        else
            transMat = zeros(1, length(aID));
        end
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
                tempMat = neural_matrix_ms_to_frames(dataMat(transInd, aID), optBinSize(a));
                if pcaFlag
                    [coeff, score, ~, ~, explained, mu] = pca(tempMat);
                    forDim = find(cumsum(explained) > 30, 1);
                    forDim = max(3, forDim);
                    forDim = min(6, forDim);
                    if pcaFirstFlag
                        fprintf('Using PCA first %d\n', forDim)
                        nDim = 1:forDim;
                    else
                        fprintf('Using PCA Last many from %d to %D\n', forDim+1, size(score, 2))
                        nDim = forDim+1:size(score, 2);
                    end
                    tempMat = score(:,nDim) * coeff(:,nDim)' + mu;
                end

                if thresholdFlag
                    tempMat = round(sum(tempMat, 2));
                    threshPct = .08;
                    % threshSpikes = threshPct * max(dataMatReach);
                    threshSpikes = .8*median(dataMatNat);
                    tempMat(tempMat < threshSpikes) = 0;
                    fprintf('Using Threshold method \tBinsize: %.3f\tProp zeros: %.3f\n', optBinSize(a), sum(dataMatNat == 0) / length(dataMatNat))
                end
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
                tempMat = neural_matrix_ms_to_frames(dataMat(withinInd, aID), optBinSize(a));
                if thresholdFlag
                    tempMat = sum(tempMat, 2);
                    threshPct = .08;
                    % threshSpikes = threshPct * max(dataMatReach);
                    threshSpikes = .9*median(tempMat);
                    tempMat(tempMat < threshSpikes) = 0;
                end
                zeroBins = find(sum(tempMat, 2) == 0);
                if length(zeroBins) > 1 && any(diff(zeroBins)>1)
                    withinMat = [withinMat; tempMat(zeroBins(1) : zeroBins(end)-1, :)];
                end
            end

        end
        transMat = [transMat; zeros(1, size(transMat, 2))]; % add a final row of zeros to close out last avalanche
        withinMat = [withinMat; zeros(1, size(withinMat, 2))]; % add a final row of zeros to close out last avalanche




        % dataMatT = neural_matrix_ms_to_frames(transMat, optBinSize);
        asdfMat = rastertoasdf2(transMat', optBinSize(a)*1000, 'CBModel', 'Spikes', 'trans');
        Av(a, b, 1) = avprops(asdfMat, 'ratio', 'fingerprint');
        br = Av(a, b, 1).branchingRatio;
        brHist(a,b,:,1) = histcounts(br(br>0), edges, 'Normalization','pdf');

        [tau(a, b, 1), tauC(a, b, 1), alpha(a, b, 1), sigmaNuZInvSD(a, b, 1), decades(a, b, 1)] = avalanche_log(Av(a, b, 1), 0);


        % dataMatW = neural_matrix_ms_to_frames(withinMat, optBinSize);
        asdfMat = rastertoasdf2(withinMat', optBinSize(a)*1000, 'CBModel', 'Spikes', 'within');
        Av(a, b, 2) = avprops(asdfMat, 'ratio', 'fingerprint');
        br = Av(a, b, 2).branchingRatio;
        brHist(a,b,:,2) = histcounts(br(br>0), edges, 'Normalization','pdf');
        [tau(a, b, 2), tauC(a, b, 2), alpha(a, b, 2), sigmaNuZInvSD(a, b, 2), decades(a, b, 2)] = avalanche_log(Av(a, b, 2), 0);
    end
end
% fileName = fullfile(paths.dropPath, 'avalanche_analyses.mat');
% save(fileName, 'Av', 'brPeak', 'tau', 'alpha', 'sigmaNuZInvSD', 'optBinSize', 'areas', 'nSubsample', '-append')


%%
a = 4;
fig = figure(37); clf;
set(fig, 'Position', monitorTwo);
linewidth = 2;
ha = tight_subplot(2, ceil(length(analyzeCodes)/2), [0.05 0.02], [0.15 0.1], [0.07 0.05]);  % [gap, lower margin, upper margin]
for b = 1 : length(analyzeCodes)
    axes(ha(b));
    hold on; grid on;
    plot(centers(2:end), reshape(brHist(a,b,2:end,1), length(centers(2:end)), 1), 'k', 'LineWidth',linewidth)
    plot(centers(2:end), reshape(brHist(a,b,2:end,2), length(centers(2:end)), 1), 'r', 'LineWidth',linewidth)
    title(analyzeBhv{b}, 'Interpreter','none')
    set(ha(b), 'XTickLabelMode', 'auto');  % Enable Y-axis labels
    xlim([0 2])
    if b == 1
        legend({'Trans', 'Within'})
    end
end
sgtitle([areas{a}, ' Branching Ratios Transitions vs Within-Bout'])
copy_figure_to_clipboard
%%
% a = 2;
fig2 = figure(38); clf;
set(fig2, 'Position', monitorTwo);

ha = tight_subplot(2, ceil(length(analyzeCodes)/2), [0.05 0.02], [0.15 0.1], [0.07 0.05]);  % [gap, lower margin, upper margin]
for b = 1 : length(analyzeCodes)
    % Activate subplot
    axes(ha(b));
    % for a = 1 : length(areas)
    hold on;

    plot(1, tau(a,b,1), 'ok', 'LineWidth', linewidth)
    plot(1, tau(a,b,2), 'or', 'LineWidth', linewidth)

    plot(2, tauC(a,b,1), 'ok', 'LineWidth', linewidth)
    plot(2, tauC(a,b,2), 'or', 'LineWidth', linewidth)

    plot(3, alpha(a,b,1), 'ok', 'LineWidth', linewidth)
    plot(3, alpha(a,b,2), 'or', 'LineWidth', linewidth)

    plot(4, sigmaNuZInvSD(a,b,1), 'ok', 'LineWidth', linewidth)
    plot(4, sigmaNuZInvSD(a,b,2), 'or', 'LineWidth', linewidth)

    plot([1 1], tauRange, 'b', 'LineWidth', 1)
    plot([2 2], tauRange, 'b', 'LineWidth', 1)
    plot([3 3], alphaRange, 'b', 'LineWidth', 1)
    plot([4 4], sigmaRange, 'b', 'LineWidth', 1)

    xticks(1:4)
    xticklabels({'tau', 'tauC', 'alpha', 'sigmaNuZInvSD'})
    xlim([.5 4.5])
    ylim([.5 5])
    set(ha(b), 'YTickLabelMode', 'auto');  % Enable Y-axis labels
    grid on;
    title(analyzeBhv{b}, 'interpreter', 'none')
    if b == 1
        legend({'Trans', 'Within'})
    end
end
sgtitle([areas{a}, ' Avalanche Params Transitions vs Within-Bout'])
copy_figure_to_clipboard








































%%   ================================            Sliding window of criticality parameters          ================================
% Slide a window from 1sec prior to 100ms after each time point
% Get criticality parameters at each time piont
% Assess metastability of criticality over time
opts = neuro_behavior_options;
opts.frameSize = .001;
opts.minFiringRate = .1;
getDataType = 'spikes';
opts.collectStart = 3*60*60;
opts.collectFor = 10 * 60;
opts.firingRateCheckTime = 5 * 60;
get_standard_data

%%
pcaFlag = 0;
pcaFirstFlag = 0;
thresholdFlag = 1;
thresholdBinSize = .015;

preTime = 3;
postTime = .2;
stepSize = 0.1;       % Step size in seconds
numSteps = floor((size(dataMat, 1) / 1000 - stepSize) / stepSize) - 1;

% Initialize variables
areas = {'M23', 'M56', 'DS', 'VS'};
[brPeak, tau, tauC, alpha, sigmaNuZInvSD, decades] = deal(cell(length(areas), 1));
optBinSize = nan(length(areas), 1);
idList = {idM23, idM56, idDS, idVS};
% Branching ratio histogram values
minBR = 0; maxBR = 2.5;
edges = minBR : .1 : maxBR;
centers = edges(1:end-1) + diff(edges) / 2;

% poolID = parpool(4, 'IdleTimeout', Inf);
% parfor a = 1  : length(areas)
for a = 1 : length(areas)
    tic
    aID = idList{a};

    % Find optimal bin size for avalanche analyses
    % If using the threshold method
    if thresholdFlag
        optBinSize(a) = thresholdBinSize;
    else
        optBinSize(a) = optimal_bin_size(dataMat(:,aID));
    end
    fprintf('-------------\tArea: %s\n', areas{a})

    dataMatNat = neural_matrix_ms_to_frames(dataMat(:, aID), optBinSize(a));

    if pcaFlag
        [coeff, score, ~, ~, explained, mu] = pca(dataMatNat);
        forDim = find(cumsum(explained) > 30, 1);
        forDim = max(3, forDim);
        forDim = min(6, forDim);
        if pcaFirstFlag
            fprintf('Using PCA first %d\n', forDim)
            nDim = 1:forDim;
        else
            fprintf('Using PCA Last many from %d to %D\n', forDim+1, size(score, 2))
            nDim = forDim+1:size(score, 2);
        end
        dataMatNat = score(:,nDim) * coeff(:,nDim)' + mu;
    end

    if thresholdFlag
        dataMatNat = round(sum(dataMatNat, 2));
        threshPct = .08;
        % threshSpikes = threshPct * max(dataMatReach);
        threshSpikes = .8*median(dataMatNat);
        dataMatNat(dataMatNat < threshSpikes) = 0;
        fprintf('Using Threshold method \tBinsize: %.3f\tProp zeros: %.3f\n', optBinSize(a), sum(dataMatNat == 0) / length(dataMatNat))
    end


    transWindow = (floor(-preTime/optBinSize(a)) : ceil(postTime/optBinSize(a)) - 1);

    % % Calculate number of windows to preallocate
    % numWindows = floor((size(dataMatNat, 1) - 1) / stepRows) + 1;

    [iTau, iTauC, iAlpha, iSigmaNuZInvSD, iDecades] = deal(nan(numSteps, 1));

    iBrHist = nan(numSteps, length(centers));

    % Find firsst valid index to start collecting windowed data
    firstIdx = ceil(preTime / stepSize);
    for i = firstIdx: numSteps %-transWindow(1): size(dataMatNat, 1) - transWindow(end)
        % Find the index in dataMatNat for this step
        iRealTime = i * stepSize;
        iIdx = round(iRealTime / optBinSize(a));
        % Find the index in dataMatNat for this step
        % iIdx = -transWindow(1) + round((i-1) * stepRows);
        fprintf('\t%s\t %d of %d: %.3f  finished \n', areas{a}, i, numSteps, i/numSteps)
        % Make sure there are avalanches in the window
        zeroBins = find(sum(dataMatNat(iIdx + transWindow + 1, :), 2) == 0);
        if length(zeroBins) > 1 && any(diff(zeroBins)>1)

            asdfMat = rastertoasdf2(dataMatNat(iIdx + transWindow + 1, :)', optBinSize(a)*1000, 'CBModel', 'Spikes', 'DS');
            Av = avprops(asdfMat, 'ratio', 'fingerprint');
            iBrHist(i,:) = histcounts(Av.branchingRatio, edges, 'Normalization', 'pdf');
            [iTau(i), iTauC(i), iAlpha(i), iSigmaNuZInvSD(i), iDecades(i)] = avalanche_log(Av, 0);
        end

    end
    fprintf('\nArea %s\t %.1f\n\n', areas{a}, toc/60)

    brPeak{a} = iBrHist;
    tau{a} = iTau;
    tauC{a} = iTauC;
    alpha{a} = iAlpha;
    sigmaNuZInvSD{a} = iSigmaNuZInvSD;
    decades{a} = iDecades;
end
% delete(poolID)
%%
fileName = fullfile(paths.dropPath, 'avalanche_data_10min.mat');
save(fileName, 'brPeak', 'tau', 'tauC', 'alpha', 'sigmaNuZInvSD', 'optBinSize', 'centers', 'stepSize', 'preTime', 'postTime', 'areas')






























%% ==============================================             Mark's reaching vs ITI             ==============================================
opts.minFiringRate = .1;
getDataType = 'spikes';
opts.frameSize = .001;

% Load Mark's reach data and make it a ms neural data matrix
dataR = load(fullfile(paths.dropPath, 'reach_data/Y4_100623_Spiketimes_idchan_BEH.mat'));
[dataMatR, idLabels, areaLabels, rmvNeurons] = neural_matrix_mark_data(dataR, opts);
idM23R = find(strcmp(areaLabels, 'M23'));
idM56R = find(strcmp(areaLabels, 'M56'));
idDSR = find(strcmp(areaLabels, 'DS'));
idVSR = find(strcmp(areaLabels, 'VS'));

% timeEmJs = dataR.und_time_EM_JS;
% align arduino time with neural time
% timeEmJs(:,1) = timeEmJs(:,1) + dataR.ArduinoOffset;



%%
pcaFlag = 1;
thresholdFlag = 1;
thresholdBinSize = .05;

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
block1Err = 1;
block1Corr = 2;
block2Err = 3;
block2Corr = 4;
trialIdx = dataR.block(:, 3) == block2Err;

for a = 1 : length(areas)
    fprintf('Area %s\n', areas{a})
    tic
    aID = idList{a};

    % If using the threshold method
    if thresholdFlag
        optBinSize(a) = thresholdBinSize;
    else
        optBinSize(a) = optimal_bin_size(dataMatR(:, aID));
    end

    reachWindow = floor(-preTime/optBinSize(a)) : ceil(postTime/optBinSize(a)) - 1;
    baseWindow = reachWindow - floor(3/optBinSize(a)) - 1;

    dataMatReach = neural_matrix_ms_to_frames(dataMatR(:, aID), optBinSize(a));

    if pcaFlag
        [coeff, score, ~, ~, explained, mu] = pca(dataMatReach);
        forDim = find(cumsum(explained) > 30, 1);
        forDim = max(3, forDim);
        forDim = min(6, forDim);
        if pcaFirstFlag
            fprintf('Using PCA first %d\n', forDim)
            nDim = 1:forDim;
        else
            fprintf('Using PCA Last many from %d to %D\n', forDim+1, size(score, 2))
            nDim = forDim+1:size(score, 2);
        end
        dataMatReach = score(:,nDim) * coeff(:,nDim)' + mu;
    end

    if thresholdFlag
        dataMatReach = round(sum(dataMatReach, 2));
        threshPct = .08;
        % threshSpikes = threshPct * max(dataMatReach);
        threshSpikes = .8*median(dataMatReach);
        dataMatReach(dataMatReach < threshSpikes) = 0;
    end

    rStarts = round(dataR.R(trialIdx,1)/optBinSize(a)/1000);  % in frame time

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

    asdfMat = rastertoasdf2(baseMat', optBinSize(a)*1000, 'CBModel', 'Spikes', 'reach');
    Av(a, 1) = avprops(asdfMat, 'ratio', 'fingerprint');
    br = Av(a, 1).branchingRatio;
    brHist(a,:, 1) = histcounts(br(br>0), edges, 'Normalization','pdf');
    [tau(a, 1), tauC(a, 1), alpha(a, 1), sigmaNuZInvSD(a, 1), decades(a, 1)] = avalanche_log(Av(a, 1), plotFlag);


    %
    asdfMat = rastertoasdf2(reachMat', optBinSize(a)*1000, 'CBModel', 'Spikes', 'ITI');
    Av(a, 2) = avprops(asdfMat, 'ratio', 'fingerprint');
    br = Av(a, 2).branchingRatio;
    brHist(a,:, 2) = histcounts(br(br>0), edges, 'Normalization','pdf');
    [tau(a, 2), tauC(a, 2), alpha(a, 2), sigmaNuZInvSD(a, 2), decades(a, 2)] = avalanche_log(Av(a, 2), plotFlag);

    % [tauB, alphaB, sigmaNuZInvSDB]
    % [tau, alpha, sigmaNuZInvSD]
end

%%
figure(36); clf;
ha = tight_subplot(1, length(areas), [0.05 0.05], [0.15 0.1], [0.07 0.05]);  % [gap, lower margin, upper margin]
for a = 1 : length(areas)
    axes(ha(a));
    hold on;
    linewidth = 2;
    % for a = 1 : length(areas)
    plot(centers(2:end), brHist(a,2:end,1), 'k', 'LineWidth',linewidth)
    plot(centers(2:end), brHist(a,2:end,2), 'r', 'LineWidth',linewidth)
    xlim([0 2])
    grid on;
    set(ha(a), 'XTickLabelMode', 'auto');  % Enable Y-axis labels
    title([areas{a}])
end
legend({'ITI', 'Reach'})
sgtitle('Branching Ratio PDFs')
copy_figure_to_clipboard
%
pause
figure(37); clf;
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

    plot([1 1], tauRange, 'b', 'LineWidth', 1)
    plot([2 2], tauRange, 'b', 'LineWidth', 1)
    plot([3 3], alphaRange, 'b', 'LineWidth', 1)
    plot([4 4], sigmaRange, 'b', 'LineWidth', 1)
    xticks(1:4)
    xticklabels({'tau', 'tauC', 'alpha', 'sigmaNuZInvSD'})
    xlim([.5 4.5])
    ylim([.5 5])
    grid on;
    title(areas{a})
end
legend({'ITI', 'Reaches'})
sgtitle('Avalanche Params Reaching vs ITI')
copy_figure_to_clipboard



















%% ==============================================             Mark's Sliding Window
opts.minFiringRate = .1;
getDataType = 'spikes';
opts.frameSize = .001;

% Load Mark's reach data and make it a ms neural data matrix
dataR = load(fullfile(paths.dropPath, 'reach_data/Y4_100623_Spiketimes_idchan_BEH.mat'));
[dataMatR, idLabels, areaLabels, rmvNeurons] = neural_matrix_mark_data(dataR, opts);
idM23R = find(strcmp(areaLabels, 'M23'));
idM56R = find(strcmp(areaLabels, 'M56'));
idDSR = find(strcmp(areaLabels, 'DS'));
idVSR = find(strcmp(areaLabels, 'VS'));

%%

% howLong = 10 * 60;
% dataMatR = dataMatR(1:(howLong * 1000), :);


pcaFlag = 0;
pcaFirstFlag = 0;
thresholdFlag = 1;
thresholdBinSize = .02;

preTime = 3;
postTime = .2;
stepSize = 0.1;       % Step size in seconds
numSteps = floor((size(dataMatR, 1) / 1000 - stepSize) / stepSize) - 1;

% Initialize variables
areas = {'M23', 'M56', 'DS', 'VS'};
[brPeak, tau, tauC, alpha, sigmaNuZInvSD, decades] = deal(cell(length(areas), 1));
optBinSize = nan(length(areas), 1);
idList = {idM23R, idM56R, idDSR, idVSR};
% Branching ratio histogram values
minBR = 0; maxBR = 2.5;
edges = minBR : .1 : maxBR;
centers = edges(1:end-1) + diff(edges) / 2;

% poolID = parpool(4, 'IdleTimeout', Inf);
% parfor a = 1  : length(areas)
for a = 1 : length(areas)
    tic
    aID = idList{a};

    % Find optimal bin size for avalanche analyses
    % If using the threshold method
    if thresholdFlag
        optBinSize(a) = thresholdBinSize;
    else
        optBinSize(a) = optimal_bin_size(dataMatR(:,aID));
    end
    fprintf('-------------\tArea: %s\n', areas{a})

    dataMatNat = neural_matrix_ms_to_frames(dataMatR(:, aID), optBinSize(a));

    if pcaFlag
        [coeff, score, ~, ~, explained, mu] = pca(dataMatNat);
        forDim = find(cumsum(explained) > 30, 1);
        forDim = max(3, forDim);
        forDim = min(6, forDim);
        if pcaFirstFlag
            fprintf('Using PCA first %d\n', forDim)
            nDim = 1:forDim;
        else
            fprintf('Using PCA Last many from %d to %D\n', forDim+1, size(score, 2))
            nDim = forDim+1:size(score, 2);
        end
        dataMatNat = score(:,nDim) * coeff(:,nDim)' + mu;
    end

    if thresholdFlag
        dataMatNat = round(sum(dataMatNat, 2));
        threshPct = .08;
        % threshSpikes = threshPct * max(dataMatReach);
        threshSpikes = .8*median(dataMatNat);
        dataMatNat(dataMatNat < threshSpikes) = 0;
        fprintf('Using Threshold method \tBinsize: %.3f\tProp zeros: %.3f\n', optBinSize(a), sum(dataMatNat == 0) / length(dataMatNat))
    end


    transWindow = (floor(-preTime/optBinSize(a)) : ceil(postTime/optBinSize(a)) - 1);

    % % Calculate number of windows to preallocate
    % numWindows = floor((size(dataMatNat, 1) - 1) / stepRows) + 1;

    [iTau, iTauC, iAlpha, iSigmaNuZInvSD, iDecades] = deal(nan(numSteps, 1));

    iBrHist = nan(numSteps, length(centers));

    % Find firsst valid index to start collecting windowed data
    firstIdx = ceil(preTime / stepSize);
    for i = firstIdx: numSteps %-transWindow(1): size(dataMatNat, 1) - transWindow(end)
        % Find the index in dataMatNat for this step
        iRealTime = i * stepSize;
        iIdx = round(iRealTime / optBinSize(a));
        % Find the index in dataMatNat for this step
        % iIdx = -transWindow(1) + round((i-1) * stepRows);
        fprintf('\t%s\t %d of %d: %.3f  finished \n', areas{a}, i, numSteps, i/numSteps)
        % Make sure there are avalanches in the window
        zeroBins = find(sum(dataMatNat(iIdx + transWindow + 1, :), 2) == 0);
        if length(zeroBins) > 1 && any(diff(zeroBins)>1)

            asdfMat = rastertoasdf2(dataMatNat(iIdx + transWindow + 1, :)', optBinSize(a)*1000, 'CBModel', 'Spikes', 'DS');
            Av = avprops(asdfMat, 'ratio', 'fingerprint');
            iBrHist(i,:) = histcounts(Av.branchingRatio, edges, 'Normalization', 'pdf');
            [iTau(i), iTauC(i), iAlpha(i), iSigmaNuZInvSD(i), iDecades(i)] = avalanche_log(Av, 0);
        end

    end
    fprintf('\nArea %s\t %.1f\n\n', areas{a}, toc/60)

    brPeak{a} = iBrHist;
    tau{a} = iTau;
    tauC{a} = iTauC;
    alpha{a} = iAlpha;
    sigmaNuZInvSD{a} = iSigmaNuZInvSD;
    decades{a} = iDecades;
end
% delete(poolID)



















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













%% =======================    Fake data     =======================
% run fakeTrialWiseData

% Make a 1000hz neural matrix
dataFake = zeros(round(max(nnAll(:)) * 1000), size(nnAll, 1));
for n = 1 : size(nnAll, 1)
    nSpikes = round(nnAll(n,:)* 1000);
    nSpikes(isnan(nSpikes)) = [];
    dataFake(nSpikes, n) = 1;
end

%%
pcaFlag = 0;
pcaFirstFlag = 1;
thresholdFlag = 1;
thresholdBinSize = .05;

preTime = .2; % ms before reach onset
postTime = 2;

areas = {'Fake'};


% Branching ratio histogram values
minBR = 0; maxBR = 4;
edges = minBR : .1 : maxBR;
centers = edges(1:end-1) + diff(edges) / 2;


[tau, tauC, alpha, sigmaNuZInvSD, decades] = deal(nan(length(areas), 2));
brHist = nan(length(areas), length(centers), 2);
optBinSize = nan(length(areas), 1);
idList = {1:size(dataFake, 2)};

for a = 1 : length(areas)
    fprintf('Area %s\n', areas{a})
    tic
    aID = idList{a};

    % If using the threshold method
    if thresholdFlag
        optBinSize(a) = thresholdBinSize;
    else
        optBinSize(a) = optimal_bin_size(dataFake(:, aID));
    end

    reachWindow = floor(-preTime/optBinSize(a)) : ceil(postTime/optBinSize(a)) - 1;
    baseWindow = reachWindow - floor(3/optBinSize(a)) - 1;

    dataMatReach = neural_matrix_ms_to_frames(dataFake(:, aID), optBinSize(a));

    if pcaFlag
        [coeff, score, ~, ~, explained, mu] = pca(dataMatReach);
        forDim = find(cumsum(explained) > 30, 1);
        forDim = max(3, forDim);
        forDim = min(6, forDim);
        if pcaFirstFlag
            fprintf('Using PCA first %d\n', forDim)
            nDim = 1:forDim;
        else
            fprintf('Using PCA Last many from %d to %D\n', forDim+1, size(score, 2))
            nDim = forDim+1:size(score, 2);
        end
        dataMatReach = score(:,nDim) * coeff(:,nDim)' + mu;
    end

    if thresholdFlag
        dataMatReach = round(sum(dataMatReach, 2));
        threshPct = .08;
        % threshSpikes = threshPct * max(dataMatReach);
        threshSpikes = .9*median(dataMatReach);
        dataMatReach(dataMatReach < threshSpikes) = 0;
    end
    % rStarts = 1000 * (10:10:590);  % in frame time
    rStarts = round((10:10:590)/optBinSize(a));  % in frame time

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

    asdfMat = rastertoasdf2(baseMat', optBinSize(a)*1000, 'CBModel', 'Spikes', 'reach');
    Av(a, 1) = avprops(asdfMat, 'ratio', 'fingerprint');
    br = Av(a, 1).branchingRatio;
    brHist(a,:, 1) = histcounts(br(br>0), edges, 'Normalization','pdf');
    [tau(a, 1), tauC(a, 1), alpha(a, 1), sigmaNuZInvSD(a, 1), decades(a, 1)] = avalanche_log(Av(a, 1), plotFlag);


    %
    asdfMat = rastertoasdf2(reachMat', optBinSize(a)*1000, 'CBModel', 'Spikes', 'ITI');
    Av(a, 2) = avprops(asdfMat, 'ratio', 'fingerprint');
    br = Av(a, 2).branchingRatio;
    brHist(a,:, 2) = histcounts(br(br>0), edges, 'Normalization','pdf');
    [tau(a, 2), tauC(a, 2), alpha(a, 2), sigmaNuZInvSD(a, 2), decades(a, 2)] = avalanche_log(Av(a, 2), plotFlag);

    % [tauB, alphaB, sigmaNuZInvSDB]
    % [tau, alpha, sigmaNuZInvSD]
end

%%
figure(36); clf;
ha = tight_subplot(1, length(areas), [0.05 0.05], [0.15 0.1], [0.07 0.05]);  % [gap, lower margin, upper margin]
for a = 1 : length(areas)
    axes(ha(a));
    hold on;
    linewidth = 2;
    % for a = 1 : length(areas)
    plot(centers(2:end), brHist(a,2:end,1), 'k', 'LineWidth',linewidth)
    plot(centers(2:end), brHist(a,2:end,2), 'r', 'LineWidth',linewidth)
    xlim([0 2])
    grid on;
    set(ha(a), 'XTickLabelMode', 'auto');  % Enable Y-axis labels
    title([areas{a}])
end
legend({'ITI', 'Reach'})
sgtitle('Branching Ratio PDFs')
copy_figure_to_clipboard
%
pause
figure(37); clf;
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

    plot([1 1], tauRange, 'b', 'LineWidth', 1)
    plot([2 2], tauRange, 'b', 'LineWidth', 1)
    plot([3 3], alphaRange, 'b', 'LineWidth', 1)
    plot([4 4], sigmaRange, 'b', 'LineWidth', 1)
    xticks(1:4)
    xticklabels({'tau', 'tauC', 'alpha', 'sigmaNuZInvSD'})
    xlim([.5 4.5])
    ylim([.5 5])
    grid on;
    title(areas{a})
end
legend({'ITI', 'Reaches'})
sgtitle('Avalanche Params Reaching vs ITI')
copy_figure_to_clipboard












%%

function [tau, tauC, alpha, sigmaNuZInvSD, decades] = avalanche_log(Av, plotFlag)


if plotFlag == 1
    plotFlag = 'plot';
else
    plotFlag = 'nothing';
end

% size distribution (SZ)
[tau, xminSZ, xmaxSZ, sigmaSZ, pSZ, pCritSZ, ksDR, DataSZ] =...
    avpropvals(Av.size, 'size', plotFlag);
tau = cell2mat(tau);

decades = log10(cell2mat(xmaxSZ)/cell2mat(xminSZ));

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
% decades = log10(xmaxSZ/xminSZ);

% duration distribution (DR)
if length(unique(Av.duration)) > 1
    [alpha, xminDR, xmaxDR, sigmaDR, pDR, pCritDR, ksDR, DataDR] =...
        avpropvals(Av.duration, 'duration', plotFlag);
    alpha = cell2mat(alpha);
    % size given duration distribution (SD)
    [sigmaNuZInvSD, waste, waste, sigmaSD] = avpropvals({Av.size, Av.duration},...
        'sizgivdur', 'durmin', xminDR{1}, 'durmax', xmaxDR{1}, plotFlag);
else
    alpha = nan;
    sigmaNuZInvSD = nan;
end
end



function optBinSize = optimal_bin_size(dataMat)
spikeFrames = sum(dataMat, 2);
positiveIndices = find(spikeFrames > 0);
checkISI = repelem(positiveIndices, spikeFrames(positiveIndices));
optBinSize = ceil(mean(diff(checkISI))) / 1000;
if optBinSize == 0; optBinSize = .001; end

% Old version:
%     optBinSize{a}(iter, 1) = round(mean(diff(find(sum(dataMat(:, idSelect), 2))))) / 1000;
% if optBinSize{a}(iter, 1) == 0; optBinSize{a}(iter, 1) = .001; end

end





