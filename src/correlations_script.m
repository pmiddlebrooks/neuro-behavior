%% get desired file paths
computerDriveName = 'ROSETTA'; %'ROSETTA'; % 'Z' or 'home'
paths = get_paths(computerDriveName);


opts = neuro_behavior_options;

animal = 'ag25290';
sessionBhv = '112321_1';
sessionNrn = '112321';
if strcmp(sessionBhv, '112321_1')
    sessionSave = '112321';
end


%% Run this, then go to spiking_script and get the behavior and neural data matrix
opts.frameSize = .05; % 50 ms framesize for now
opts.collectFor = 60*60; % Get an hour of data


%% Which dataMat do you want to use?
% dataMatuse = dataMat;
dataMatUse = dataMatZ;




%% Create a 2-D data matrix of stacked peri-event start time windows (time X neuron)
bhv = 'locomotion';
bhvCode = analyzeCodes(strcmp(analyzeBhv, bhv));

periEventTime = -.2 : opts.frameSize : .2; % seconds around onset
periWindow = round(periEventTime(1:end-1) / opts.frameSize); % frames around onset (remove last frame)
preEventTime = -.8 : opts.frameSize : -.4; % seconds before onset
preWindow = round(preEventTime(1:end-1) / opts.frameSize); % frames around onset (remove last frame)


bhvStartFrames = floor(dataBhv.bhvStartTime(dataBhv.bhvID == bhvCode) ./ opts.frameSize);
bhvStartFrames(bhvStartFrames < abs(preWindow(1))+1) = [];
bhvStartFrames(bhvStartFrames > size(dataMat, 1) - periWindow(end)-1) = [];

nTrial = length(bhvStartFrames);


dataMatPeri = [];
dataMatPre = [];
for j = 1 : nTrial
    dataMatPeri = [dataMatPeri; dataMat(bhvStartFrames(j) + periWindow ,:)];
    dataMatPre = [dataMatPre; dataMat(bhvStartFrames(j) + preWindow ,:)];
end

%% Get data mat for M56 and size-matched subset of DS
dataMatM56Peri = dataMatPeri(:, strcmp(areaLabels, 'M56'));
dataMatDSPeri =  dataMatPeri(:, strcmp(areaLabels, 'DS'));
dataMatM56Pre = dataMatPre(:, strcmp(areaLabels, 'M56'));
dataMatDSPre =  dataMatPre(:, strcmp(areaLabels, 'DS'));
% subDSIdx = randperm(size(dataMatDS, 2), size(dataMatM56, 2));
% dataMatDSSub = dataMatDS(:, subDSIdx);

%%      Overall correlations across the whole recording

edges = -.1 : .005 : .1;
    binCenters = (edges(1:end-1) + edges(2:end)) / 2;

% M56
m56CorrPeri = tril(corr(dataMatM56Peri), -1);
N = histcounts(m56CorrPeri(:), edges, 'Normalization', 'pdf');
subplot(2,2,1)
bar(binCenters, N, 'hist')
[mean(m56CorrPeri(:)), std(m56CorrPeri(:))]

m56CorrPre = tril(corr(dataMatM56Pre), -1);
N = histcounts(m56CorrPre(:), edges, 'Normalization', 'pdf');
subplot(2,2,3)
bar(binCenters, N, 'hist')
[mean(m56CorrPre(:)), std(m56CorrPre(:))]

% DS
dsCorrPeri = tril(corr(dataMatDSPeri), -1);
N = histcounts(dsCorrPeri(:), edges, 'Normalization', 'pdf');
subplot(2,2,2)
bar(binCenters, N, 'hist')
[mean(dsCorrPeri(:)), std(dsCorrPeri(:))]

dsCorrPre = tril(corr(dataMatDSPre), -1);
N = histcounts(dsCorrPre(:), edges, 'Normalization', 'pdf');
subplot(2,2,4)
bar(binCenters, N, 'hist')
[mean(dsCorrPre(:)), std(dsCorrPre(:))]








%% Cross-Correlation across areas (across the whole session)

% [c,lags] = corrcoef(dataMatM56,dataMatDSSub);
[c,lags] = xcorr(dataMatM56(:,1),dataMatDSSub(:,1)); % this is for example first neuron in each brain area
subplot(1,2,1)
stem(lags,c)

crossC = corr(dataMatM56, dataMatDS);
edges = -1 : .01 : 1;
    binCenters = (edges(1:end-1) + edges(2:end)) / 2;
N = histcounts(crossC(:), edges, 'Normalization', 'pdf');
subplot(1,2,2)
bar(binCenters, N, 'hist')
mean(crossC(:))






%% Cross-correlations for each behavior between pairwise M56 and DS (their full populations)

% Create a 3-D psth data matrix of stacked peri-event start time windows (time X neuron X trial)

periEventTime = -.3 : opts.frameSize : .3; % seconds around onset
dataWindow = round(periEventTime(1:end-1) / opts.frameSize); % frames around onset (remove last frame)
eventMat = cell(length(analyzeBhv), 1);
for iBhv = 1 : length(analyzeBhv)
    bhvCode = analyzeCodes(strcmp(analyzeBhv, analyzeBhv{iBhv}));

    bhvStartFrames = floor(dataBhv.bhvStartTime(dataBhv.bhvID == bhvCode) ./ opts.frameSize);
    bhvStartFrames(bhvStartFrames < dataWindow(end) + 1) = [];
    bhvStartFrames(bhvStartFrames > size(dataMat, 1) - dataWindow(end)) = [];

    nTrial = length(bhvStartFrames);

    iEventMat = zeros(length(dataWindow), size(dataMat, 2), nTrial); % peri-event time X neurons X nTrial
    for j = 1 : nTrial
        iEventMat(:,:,j) = dataMat(bhvStartFrames(j) + dataWindow ,:);
    end
    eventMat{iBhv} = iEventMat;
end





%%  Cross-correlations
% maxLag = 5;
meanLagPerBhv = cell(length(analyzeBhv), 1);
m56Ind = find(strcmp(areaLabels, 'M56'));
dsInd = find(strcmp(areaLabels, 'DS'));
for iBhv = 1 : length(analyzeBhv)
    meanLag = zeros(length(m56Ind), length(dsInd)); % collects mean pairwise lags for each behavior (lags averaged across all onset times for this behavior)

    % Across all M56 neurons
    for m = 1 : length(m56Ind)

        % Across all DS neurons
        for d = 1 : length(dsInd)

            % Acorss all trials
            nLag = []; % Avg pairwise lag across all trials for this behavior
            for n = 1 : size(eventMat{iBhv}, 3) % Across all onsets for this behavior

                if sum(eventMat{iBhv}(:,m56Ind(m), n)) && sum(eventMat{iBhv}(:,dsInd(d), n)) % Only use trials if both neurons (from each area) have at least on spike
                    % [c,lags] = xcorr(eventMat{iBhv}(:,m56Ind(m), iBhv), eventMat{iBhv}(:,dsInd(d), iBhv), 'normalized');
                    [c,lags] = xcorr(eventMat{iBhv}(:,m56Ind(m), n), eventMat{iBhv}(:,dsInd(d), n), 'normalized');
                    % stem(lags,c)
                    nLag = [nLag; sum(lags(:) .* c) / sum(c)];

                    % figure(55);
                    % plot(lags, c)
                end

            end
            meanLag(m, d) = mean(nLag);

        end

    end
    meanLagPerBhv{iBhv} = meanLag;

    % edges = dataWindow;
    edges = -8 : .1 : 7;
    N = histcounts(meanLagPerBhv{iBhv}(:), edges, 'Normalization', 'pdf');
    binCenters = (edges(1:end-1) + edges(2:end)) / 2;

    figure(44)
    cla
    hold on;
    % bar(edges(1:end-1), normHist, 'hist')
    bar(binCenters, N, 'hist')
    xlim([dataWindow(1) dataWindow(end)])
    xlim([-4 4])

    % Mean lag across all pairwise neurons
    binCenters = edges(1:end-1);
    meanX = sum(binCenters .* N) / sum(N)
    xline(meanX, 'k', 'linewidth', 2)
    % Median
    cumulativeSum = cumsum(N);
    medianIndex = find(cumulativeSum >= sum(N) / 2, 1);
    medianX = binCenters(medianIndex)
    xline(medianX, 'r', 'linewidth', 2)

    title(['Cross-correlation between M56 and DS: ', analyzeBhv{iBhv}], 'interpreter', 'none')

    if plotGauss
        % Gaussian mixture model
        data = meanLagPerBhv{iBhv}(:);
        % Fit a Gaussian mixture model with 2 components
        gm = fitgmdist(data, 2);

        % Define the range for plotting
        x_range = linspace(min(data), max(data), 1000);

        % Evaluate the pdf of the GMM
        y_gmm = pdf(gm, x_range');

        % Evaluate the pdfs of the individual components
        % Normalize the component pdfs to match the scale of the GMM pdf
        y_comp1 = normpdf(x_range, gm.mu(1), sqrt(gm.Sigma(:,:,1)));
        y_comp2 = normpdf(x_range, gm.mu(2), sqrt(gm.Sigma(:,:,2)));
        y_comp1 = y_comp1 * gm.ComponentProportion(1);
        y_comp2 = y_comp2 * gm.ComponentProportion(2);

        % Plot the GMM pdf
        plot(x_range, y_gmm, 'r', 'LineWidth', 2);

        % Plot the individual component pdfs
        plot(x_range, y_comp1, '--r', 'LineWidth', 2);
        plot(x_range, y_comp2, '--r', 'LineWidth', 2);


        % GMModel = fitgmdist(meanLagPerBhv{iBhv}(:),2);
        % mu = GMModel.mu;
        % sigma = GMModel.Sigma(:);
        % gmwt = GMModel.ComponentProportion;
        % x = linspace(-20,20,1000);
        % pdfValues = pdf(GMModel, x');
        % plot(x, pdfValues, 'k', 'LineWidth', 3);
        % pdf1 = normpdf(x, mu(1), sqrt(sigma(1)));
        % pdf2 = normpdf(x, mu(2), sqrt(sigma(2)));
        % plot(x, pdf1*gmwt(1), 'r', 'LineWidth', 3)
        % plot(x, pdf2*gmwt(2), 'r', 'LineWidth', 3)
    end
end























%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%       Signal and noise correlations
%       across pairwise M56 and DS neurons across behvaiors (mean per-event response of each neuron for each behavior

%  Get relevant data
periEventTime = -.2 : opts.frameSize : .2; % seconds around onset
dataWindow = periEventTime(1:end-1) / opts.frameSize; % frames around onset (remove last frame)

m56Ind = find(strcmp(areaLabels, 'M56'));
dsInd = find(strcmp(areaLabels, 'DS'));

meanSpikes = zeros(length(analyzeBhv), size(dataMat, 2));
spikesPerTrial = cell(length(analyzeBhv), 1);
for iBhv = 1 : length(analyzeBhv)
    bhvCode = analyzeCodes(strcmp(analyzeBhv, analyzeBhv{iBhv}));

    bhvStartFrames = floor(dataBhv.bhvStartTime(dataBhv.bhvID == bhvCode) ./ opts.frameSize);
    bhvStartFrames(bhvStartFrames < 10) = [];
    bhvStartFrames(bhvStartFrames > size(dataMat, 1) - 10) = [];

    nTrial = length(bhvStartFrames);

    iEventMat = zeros(nTrial, size(dataMat, 2)); % nTrial X nNeurons
    for j = 1 : nTrial
        iEventMat(j,:) = sum(dataMat(bhvStartFrames(j) + dataWindow ,:));
    end

    meanSpikes(iBhv, :) = mean(iEventMat, 1);
    spikesPerTrial{iBhv} = iEventMat;
end

%% Signal correlations
xlimConst = [-1 1];

% -------------------------
% M56 signal correlations
% -------------------------
[rho,pval] = corr(meanSpikes(:, m56Ind));
returnIdx = tril(true(length(m56Ind)), -1);

edges = -1 : .01 : 1;
binCenters = (edges(1:end-1) + edges(2:end)) / 2;
% xVal = edges(1:end-1);
N = histcounts(rho(returnIdx), edges, 'Normalization', 'pdf');
% N = histcounts(rho(returnIdx), edges);
% normHist = N / sum(N);

% Plot
subplot(1,3,1)
cla
hold on
bar(binCenters, N, 'hist')

% Mean correlation
meanX = sum(binCenters .* N) / sum(N);
plot([meanX meanX], [0 .005], 'g', 'linewidth', 2)
% Median correlation
cumulativeSum = cumsum(N);
medianIndex = find(cumulativeSum >= sum(N) / 2, 1);
medianX = binCenters(medianIndex);
plot([medianX medianX], [0 .005], 'r', 'linewidth', 2)
xlim(xlimConst)

title(['M56 Signal Correlations'], 'interpreter', 'none')

% -------------------------
% DS signal correlations
% -------------------------
[rho,pval] = corr(meanSpikes(:, dsInd));
returnIdx = tril(true(length(dsInd)), -1);

N = histcounts(rho(returnIdx), edges, 'Normalization', 'pdf');
% N = histcounts(rho(returnIdx), edges);
% normHist = N / sum(N);

% Plot
subplot(1,3,2)
cla
hold on
bar(binCenters, N, 'hist')

% Mean correlation
meanX = sum(binCenters .* N) / sum(N);
plot([meanX meanX], [0 .005], 'g', 'linewidth', 2)
% Median correlation
cumulativeSum = cumsum(N);
medianIndex = find(cumulativeSum >= sum(N) / 2, 1);
medianX = binCenters(medianIndex);
plot([medianX medianX], [0 .005], 'r', 'linewidth', 2)
xlim(xlimConst)

title(['DS Signal Correlations'], 'interpreter', 'none')

% -------------------------
% M56-DS areas signal correlations
% -------------------------
[rho,pval] = corr(meanSpikes(:, m56Ind), meanSpikes(:, dsInd));

N = histcounts(rho(:), edges, 'Normalization', 'pdf');
% N = histcounts(rho(:), edges);
% normHist = N / sum(N);

% Plot
subplot(1,3,3)
cla
hold on
bar(binCenters, N, 'hist')

% Mean correlation
meanX = sum(binCenters .* N) / sum(N);
plot([meanX meanX], [0 .005], 'g', 'linewidth', 2)
% Median correlation
cumulativeSum = cumsum(N);
medianIndex = find(cumulativeSum >= sum(N) / 2, 1);
medianX = binCenters(medianIndex);
plot([medianX medianX], [0 .005], 'r', 'linewidth', 2)
xlim(xlimConst)
title(['M56-DS Signal Correlations'], 'interpreter', 'none')

if plotGauss
    % Gaussian mixture model
    data = rho(:);
    % Fit a Gaussian mixture model with 2 components
    gm = fitgmdist(data, 2);

    % Define the range for plotting
    x_range = linspace(min(data), max(data), 1000);

    % Evaluate the pdf of the GMM
    y_gmm = pdf(gm, x_range');

    % Evaluate the pdfs of the individual components
    % Normalize the component pdfs to match the scale of the GMM pdf
    y_comp1 = normpdf(x_range, gm.mu(1), sqrt(gm.Sigma(:,:,1)));
    y_comp2 = normpdf(x_range, gm.mu(2), sqrt(gm.Sigma(:,:,2)));
    y_comp1 = y_comp1 * gm.ComponentProportion(1);
    y_comp2 = y_comp2 * gm.ComponentProportion(2);

    % Plot the GMM pdf
    plot(x_range, y_gmm, 'r', 'LineWidth', 2);

    % Plot the individual component pdfs
    plot(x_range, y_comp1, '--r', 'LineWidth', 2);
    plot(x_range, y_comp2, '--r', 'LineWidth', 2);


    %     %Gaussian mixture model
    % GMModel = fitgmdist(rho(:),2);
    % mu = GMModel.mu;
    % sigma = GMModel.Sigma(:);
    % gmwt = GMModel.ComponentProportion;
    % x = linspace(edges(1),edges(end),1000);
    % pdfValues = pdf(GMModel, x');
    % plot(x, pdfValues, 'k', 'LineWidth', 3);
    % pdf1 = normpdf(x, mu(1), sqrt(sigma(1)));
    % pdf2 = normpdf(x, mu(2), sqrt(sigma(2)));
    % plot(x, pdf1*gmwt(1), 'r', 'LineWidth', 3)
    % plot(x, pdf2*gmwt(2), 'r', 'LineWidth', 3)
end

%     imagesc(rho)
% colormap(bluewhitered)
% colorbar


%% Noise correlations


noiseCorrM56 = zeros(length(m56Ind), length(m56Ind), length(analyzeBhv));
noiseCorrDS = zeros(length(dsInd), length(dsInd), length(analyzeBhv));
noiseCorrCross = zeros(length(m56Ind), length(dsInd), length(analyzeBhv));
edges = -1 : .01 : 1;
binCenters = (edges(1:end-1) + edges(2:end)) / 2;
xVal = edges(1:end-1);
xlimConst = [-.6 .6];
for iBhv = 1 : length(analyzeBhv)

    % -------------------------
    % M56 areas correlations
    % -------------------------
    [iCorr iPval] = corr(spikesPerTrial{iBhv}(:, m56Ind));
    noiseCorrM56(:,:,iBhv) = iCorr;
    returnIdx = tril(true(length(m56Ind)), -1);

    N = histcounts(iCorr(returnIdx), edges, 'Normalization', 'pdf');
    % N = histcounts(iCorr(returnIdx), edges);
    % normHist = N / sum(N);

    subplot(1,3,1)
    cla
    hold on
    bar(binCenters, N, 'hist')

    % Mean correlation
    meanX = sum(binCenters .* N) / sum(N);
    plot([meanX meanX], [0 .5], 'g', 'linewidth', 2)
    % Median correlation
    cumulativeSum = cumsum(N);
    medianIndex = find(cumulativeSum >= sum(N) / 2, 1);
    medianX = binCenters(medianIndex);
    plot([medianX medianX], [0 .5], 'r', 'linewidth', 2)
    xlim(xlimConst)
    title(['M56 Noise Correlations for ', analyzeBhv{iBhv}], 'interpreter', 'none')

    % -------------------------
    % DS correlations
    % -------------------------
    [iCorr iPval] = corr(spikesPerTrial{iBhv}(:, dsInd));
    noiseCorrDS(:,:,iBhv) = iCorr;
    returnIdx = tril(true(length(dsInd)), -1);

    N = histcounts(iCorr(returnIdx), edges, 'Normalization', 'pdf');
    % N = histcounts(iCorr(returnIdx), edges);
    % normHist = N / sum(N);

    subplot(1,3,2)
    cla
    hold on
    bar(binCenters, N, 'hist')

    % Mean correlation
    meanX = sum(binCenters .* N) / sum(N);
    plot([meanX meanX], [0 .5], 'g', 'linewidth', 2)
    % Median correlation
    cumulativeSum = cumsum(N);
    medianIndex = find(cumulativeSum >= sum(N) / 2, 1);
    medianX = binCenters(medianIndex);
    plot([medianX medianX], [0 .5], 'r', 'linewidth', 2)
    xlim(xlimConst)
    title(['DS Noise Correlations for ', analyzeBhv{iBhv}], 'interpreter', 'none')

    % -------------------------
    % Across areas correlations
    % -------------------------
    [iCorr iPval] = corr(spikesPerTrial{iBhv}(:, m56Ind), spikesPerTrial{iBhv}(:, dsInd));
    noiseCorrCross(:,:,iBhv) = iCorr;

    N = histcounts(noiseCorrCross(:,:,iBhv), edges, 'Normalization', 'pdf');
    % N = histcounts(noiseCorrCross(:,:,iBhv), edges);
    % normHist = N / sum(N);

    subplot(1,3,3)
    cla
    hold on
    bar(binCenters, N, 'hist')

    % Mean correlation
    meanX = sum(binCenters .* N) / sum(N);
    plot([meanX meanX], [0 .5], 'g', 'linewidth', 2)
    % Median correlation
    cumulativeSum = cumsum(N);
    medianIndex = find(cumulativeSum >= sum(N) / 2, 1);
    medianX = binCenters(medianIndex);
    plot([medianX medianX], [0 .5], 'r', 'linewidth', 2)
    xlim(xlimConst)
    title(['M56-DS Noise Correlations for ', analyzeBhv{iBhv}], 'interpreter', 'none')
    % imagesc(noiseCorr(:,:,iBhv))
    % colorbar
    % colormap(bluewhitered)
end























%% For each behavior, what are the most common preceding behaviors?
dataBhv.prevID = [nan; dataBhv.bhvID(1:end-1)];
dataBhv.prevDur = [nan; dataBhv.bhvDur(1:end-1)];
dataBhv.prevStartTime = [nan; dataBhv.bhvStartTime(1:end-1)];
dataBhvTruncate = dataBhv(3:end-3, :); % Truncate a few behaviors so we can look back and ahead in time a bit
validBhvTruncate = opts.validBhv(3:end-3,:);

minCurrDur = .15;
minBeforeDur = .15; % previous behavior must last within a range (sec)
maxBeforeDur = 2;
minNumBoutsPrev = 20;

% sequenceNames = cell(length(analyzeCodes));
sequenceNames = {};
% startTimes = cell(length(analyzeCodes));
seqStartTimes = {};
for iCurr = 1 : length(analyzeCodes)

    % Start here: For each behavior:
    % - Get the noise correlations for the current behavior (like above)
    % and all (valid) previous behaviors
    % - compare them (subtract?)
    for jPrev = 1 : length(analyzeCodes)
        if iCurr ~= jPrev
            % Make sure this sequence passes requisite criteria
            currIdx = dataBhvTruncate.bhvID == analyzeCodes(iCurr) & ...
                validBhvTruncate(:, opts.bhvCodes == analyzeCodes(iCurr));
            goodSeqIdx = currIdx & ...
                dataBhvTruncate.prevID == analyzeCodes(jPrev) & ...
                dataBhvTruncate.prevDur >= minBeforeDur & ...
                dataBhvTruncate.prevDur <= maxBeforeDur;
            if sum(goodSeqIdx)
                sequenceNames = [sequenceNames; [analyzeBhv{iCurr}, ' after ', analyzeBhv{jPrev}]];
                seqStartTimes = [seqStartTimes; [dataBhvTruncate.bhvStartTime(goodSeqIdx), dataBhvTruncate.prevStartTime(goodSeqIdx)]];
            end
        end

        % STart here: For each sequence
        % - if there are enough trials
        % - get the noise correlations for the previous behavior and the
        % current behavior
        % - compare them (subtract?)

        % Also, compare all current behavior noise correlations with those
        % of previous then current correlations (how to do this with
        % uneven numbers in the distributions?
    end
end
%%
nTrial = cell2mat(cellfun(@(x) size(x, 1), seqStartTimes, 'UniformOutput', false));

[~, i] = sort(nTrial, 'descend');
nTrial = nTrial(i);
sequenceNames = sequenceNames(i);
seqStartTimes = seqStartTimes(i);
seqStartTimes(1:10)
sequenceNames(1:10)

% Use all sequences with at least 40 trials
over40 = find(nTrial >= 40, 1, 'last');
%%
m56Ind = find(strcmp(areaLabels, 'M56'));
dsInd = find(strcmp(areaLabels, 'DS'));

periEventTime = -.2 : opts.frameSize : .2; % seconds around onset
dataWindow = periEventTime(1:end-1) / opts.frameSize; % frames around onset (remove last frame)
    edges = -1 : .02 : 1;
    binCenters = (edges(1:end-1) + edges(2:end)) / 2;

for seq = 1 : over40
    seqStr = strsplit(sequenceNames{seq});
    bhvCurr = analyzeCodes(strcmp(analyzeBhv, seqStr{1}));
    bhvPrev = analyzeCodes(strcmp(analyzeBhv, seqStr{3}));

    goodStarts = dataBhvTruncate.bhvID == bhvCurr & validBhvTruncate(:, opts.bhvCodes == bhvCurr);
    allStarts = floor(dataBhvTruncate.bhvStartTime(goodStarts) ./ opts.frameSize);
    seqStarts = floor(seqStartTimes{seq} ./ opts.frameSize);


    allSpikesCurr = zeros(length(allStarts), size(dataMat, 2));
    for i = 1 : length(allStarts)
        allSpikesCurr(i,:) = sum(dataMat(allStarts(i) + dataWindow, :));
    end

    seqSpikesCurr = zeros(size(seqStarts, 1), size(dataMat, 2));
    seqSpikesPrev = zeros(size(seqStarts, 1), size(dataMat, 2));
    for i = 1 : size(seqStarts, 1)
        seqSpikesCurr(i,:) = sum(dataMat(seqStarts(i, 1) + dataWindow, :));
        seqSpikesPrev(i,:) = sum(dataMat(seqStarts(i, 2) + dataWindow, :));
    end

    % Within area correlations
    % M56
    returnIdx = tril(true(length(m56Ind)), -1);
    rho = corr(allSpikesCurr(:, m56Ind));
    allM56Corr = rho(returnIdx);
    rho = corr(seqSpikesCurr(:, m56Ind));
    seqM56CorrCurr = rho(returnIdx);
    rho = corr(seqSpikesPrev(:, m56Ind));
    seqM56CorrPrev = rho(returnIdx);

    % Test whether distributions are different
    [p,h,~] = ranksum(allM56Corr, seqM56CorrCurr)

    % Plot the distributions
    subplot(1,2,1)
    cla
    N = histcounts(allM56Corr, edges, 'Normalization', 'pdf');
    % N = histcounts(allM56Corr, edges);
    % normHist = N / sum(N);
    bar(binCenters, N, 'b', 'FaceAlpha', .5);%, 'hist')
    hold on
    xlim([-.5 .5])
    N = histcounts(seqM56CorrCurr, edges, 'Normalization', 'pdf');
    % N = histcounts(seqM56CorrCurr, edges);
    % normHist = N / sum(N);
    bar(binCenters, N, 'r', 'FaceAlpha', .5);%, 'hist')
    yl = ylim;
    plot([median(allM56Corr) median(allM56Corr)], [.9*yl(2) yl(2)], 'b', 'linewidth', 4)
    plot([median(seqM56CorrCurr) median(seqM56CorrCurr)], [.9*yl(2) yl(2)], 'r', 'linewidth', 4)
title(['M56 correlations: ', sequenceNames{seq}], 'interpreter', 'none')
    % xline(median(seqM56CorrCurr), 'r', 'linewidth', 4)
    %



    % Across area correlations
    [allM56DSCorr, ~] = corr(allSpikesCurr(:, m56Ind), allSpikesCurr(:, dsInd));
    [seqM56DSCorrCurr, ~] = corr(seqSpikesCurr(:, m56Ind), seqSpikesCurr(:, dsInd));
    [seqM56DSCorrPrev, ~] = corr(seqSpikesPrev(:, m56Ind), seqSpikesPrev(:, dsInd));

        % Test whether distributions are different
    [p,h,~] = ranksum(allM56DSCorr(:), seqM56DSCorrCurr(:)) % Different means?

    % Different variances?
    % F = var(allM56DSCorr(:)) / var(seqM56DSCorrCurr(:));
    F = var(seqM56DSCorrCurr(:)) / var(allM56DSCorr(:));
    df1 = length(seqM56DSCorrCurr(:)) - 1;
    df2 = length(allM56DSCorr(:)) - 1;
        % Calculating the p-value using the F cumulative distribution function.
    % The '1 - ' part calculates the right-tailed probability.
    pValue = 1 - fcdf(F, df1, df2);

    % Displaying the results.
    fprintf('F statistic: %f\n', F);
    fprintf('p-value: %f\n', pValue);



    % Plot the distributions
    subplot(1,2,2)
    cla
    N = histcounts(allM56DSCorr(:), edges, 'Normalization', 'pdf');
    % N = histcounts(allM56DSCorr(:), edges);
    % normHist = N / sum(N);
    bar(binCenters, N, 'b', 'FaceAlpha', .5);%, 'hist')
    hold on
    xlim([-.5 .5])
    N = histcounts(seqM56DSCorrCurr(:), edges, 'Normalization', 'pdf');
    % N = histcounts(seqM56DSCorrCurr(:), edges);
    % normHist = N / sum(N);
    bar(binCenters, N, 'r', 'FaceAlpha', .5);%, 'hist')
    yl = ylim;
    plot([median(allM56DSCorr(:)) median(allM56DSCorr(:))], [.9*yl(2) yl(2)], 'b', 'linewidth', 4)
    plot([median(seqM56DSCorrCurr(:)) median(seqM56DSCorrCurr(:))], [.9*yl(2) yl(2)], 'r', 'linewidth', 4)
title(['M56 - DS correlations: ', sequenceNames{seq}], 'interpreter', 'none')
end























%%  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                           CCA Canonical Correlation Analysis
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


