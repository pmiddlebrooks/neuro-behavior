%% get desired file paths
computerDriveName = 'ROSETTA'; %'ROSETTA'; % 'Z' or 'home'
paths = get_paths(computerDriveName)


opts = neuro_behavior_options;

animal = 'ag25290';
sessionBhv = '112321_1';
sessionNrn = '112321';
if strcmp(sessionBhv, '112321_1')
    sessionSave = '112321';
end


%% Run this, then go to spiking_script and get the behavior and data matrix
opts.frameSize = .05; % 50 ms framesize for now
opts.collectFor = 60*60; % Get an hour of data


%% Which dataMat do you want to use?
% dataMatuse = dataMat;
dataMatUse = dataMatZ;




%% Create a 2-D data matrix of stacked peri-event start time windows (time X neuron)
bhv = 'locomotion';
bhvCode = analyzeCodes(strcmp(analyzeBhv, bhv));

periEventTime = -.4 : opts.frameSize : .4; % seconds around onset
dataWindow = round(periEventTime(1:end-1) / opts.frameSize); % frames around onset (remove last frame)


bhvStartFrames = floor(dataBhv.bhvStartTime(dataBhv.bhvID == bhvCode) ./ opts.frameSize);
bhvStartFrames(bhvStartFrames < dataWindow(end)+1) = [];
bhvStartFrames(bhvStartFrames > size(dataMat, 1) - dataWindow(end)-1) = [];

nTrial = length(bhvStartFrames);

% psths = zeros(sum(nrnInd), length(dataWindow), nTrial);

% eventMat = zeros(length(dataWindow), size(dataMat, 2), nTrial); % peri-event time X neurons X nTrial
dataMatEvent = [];
for j = 1 : nTrial
    dataMatEvent = [dataMatEvent; dataMat(bhvStartFrames(j) + dataWindow ,:)];
    % eventMat(:,:,j) = dataMat(bhvStartFrames(j) + dataWindow ,:);
end

%% Get data mat for M56 and size-matched subset of DS
dataMatM56 = dataMatEvent(:, strcmp(areaLabels, 'M56'));
dataMatDS =  dataMatEvent(:, strcmp(areaLabels, 'DS'));
subDSIdx = randperm(size(dataMatDS, 2), size(dataMatM56, 2));
dataMatDSSub = dataMatDS(:, subDSIdx);

%%      Overall correlations across the whole recording

m56Corr = tril(corr(dataMatM56), -1);
edges = -1 : .01 : 1;
N = histcounts(m56Corr(:), edges);
normHist = N / sum(N);
subplot(1,2,1)
bar(edges(1:end-1), normHist, 'hist')
mean(m56Corr(:))

dsCorr = tril(corr(dataMatDS), -1);
edges = -1 : .01 : 1;
N = histcounts(dsCorr(:), edges);
normHist = N / sum(N);
subplot(1,2,2)
bar(edges(1:end-1), normHist, 'hist')
mean(dsCorr(:))

%% Correlation across areas (across the whole session)

% [c,lags] = corrcoef(dataMatM56,dataMatDSSub);
[c,lags] = xcorr(dataMatM56(:,1),dataMatDSSub(:,1));
subplot(1,2,1)
stem(lags,c)

crossC = corr(dataMatM56, dataMatDS);
edges = -1 : .01 : 1;
N = histcounts(crossC(:), edges);
normHist = N / sum(N);
subplot(1,2,2)
bar(edges(1:end-1), normHist, 'hist')
mean(crossC(:))






%% Cross-correlations for each behavior between pairwise M56 and DS (their full populations)

% Create a 3-D data matrix of stacked peri-event start time windows (time X neuron X trial)

periEventTime = -1 : opts.frameSize : 1; % seconds around onset
dataWindow = round(periEventTime(1:end-1) / opts.frameSize); % frames around onset (remove last frame)
eventMat = cell(length(analyzeBhv), 1);
for iBhv = 1 : length(analyzeBhv)
    bhvCode = analyzeCodes(strcmp(analyzeBhv, analyzeBhv{iBhv}));

    bhvStartFrames = floor(dataBhv.bhvStartTime(dataBhv.bhvID == bhvCode) ./ opts.frameSize);
    bhvStartFrames(bhvStartFrames < dataWindow(end)) = [];
    bhvStartFrames(bhvStartFrames > size(dataMat, 1) - dataWindow(end)) = [];

    nTrial = length(bhvStartFrames);

    % psths = zeros(sum(nrnInd), length(dataWindow), nTrial);

    iEventMat = zeros(length(dataWindow), size(dataMat, 2), nTrial); % peri-event time X neurons X nTrial
    % dataMatEvent = [];
    for j = 1 : nTrial
        % dataMatEvent = [dataMatEvent; dataMat(bhvStartFrames(j) + dataWindow ,:)];
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
    meanLag = [];
    for m = 1 : length(m56Ind)
        for d = 1 : length(dsInd)
            if sum(eventMat{iBhv}(:,m56Ind(m), iBhv)) && sum(eventMat{iBhv}(:,dsInd(d), iBhv))
                [c,lags] = xcorr(eventMat{iBhv}(:,m56Ind(m), iBhv), eventMat{iBhv}(:,dsInd(d), iBhv), 'normalized');
                % stem(lags,c)
                meanLag = [meanLag; sum(lags(:) .* c) / sum(c)];
            end
        end

    end
    meanLagPerBhv{iBhv} = meanLag;

    edges = dataWindow;
    N = histcounts(meanLagPerBhv{iBhv}, edges);
    normHist = N / sum(N);
    figure(23)
    cla
    hold on;
    bar(edges(1:end-1), normHist, 'hist')
    xlim([dataWindow(1) dataWindow(end)])

    % Mean correlation
    xVal = edges(1:end-1);
    meanX = sum(xVal .* normHist) / sum(normHist)
    xline(meanX, 'k', 'linewidth', 2)
    % Median
    cumulativeSum = cumsum(normHist);
    medianIndex = find(cumulativeSum >= sum(normHist) / 2, 1);
    medianX = xVal(medianIndex)
    xline(medianX, 'r', 'linewidth', 2)

    title(['Cross-correlation between M56 and DS: ', analyzeBhv{iBhv}], 'interpreter', 'none')

    % Gaussian mixture model
    GMModel = fitgmdist(meanLagPerBhv{iBhv},2);
    mu = GMModel.mu;
    sigma = GMModel.Sigma(:);
    gmwt = GMModel.ComponentProportion;
    x = linspace(-20,20,1000);
    pdfValues = pdf(GMModel, x');
    plot(x, pdfValues, 'k', 'LineWidth', 3);
    pdf1 = normpdf(x, mu(1), sqrt(sigma(1)));
    pdf2 = normpdf(x, mu(2), sqrt(sigma(2)));
    plot(x, pdf1*gmwt(1), 'r', 'LineWidth', 3)
    plot(x, pdf2*gmwt(2), 'r', 'LineWidth', 3)

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
    % dataMatEvent = [];
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
xVal = edges(1:end-1);
N = histcounts(rho(returnIdx), edges);
normHist = N / sum(N);

% Plot
subplot(1,3,1)
cla
hold on
bar(xVal, normHist, 'hist')

% Mean correlation
meanX = sum(xVal .* normHist) / sum(normHist);
plot([meanX meanX], [0 .005], 'g', 'linewidth', 2)
% Median correlation
cumulativeSum = cumsum(normHist);
medianIndex = find(cumulativeSum >= sum(normHist) / 2, 1);
medianX = xVal(medianIndex);
plot([medianX medianX], [0 .005], 'r', 'linewidth', 2)
xlim(xlimConst)

title(['M56 Signal Correlations'], 'interpreter', 'none')

% -------------------------
% DS signal correlations
% -------------------------
[rho,pval] = corr(meanSpikes(:, dsInd));
returnIdx = tril(true(length(dsInd)), -1);

edges = -1 : .01 : 1;
xVal = edges(1:end-1);
N = histcounts(rho(returnIdx), edges);
normHist = N / sum(N);

% Plot
subplot(1,3,2)
cla
hold on
bar(xVal, normHist, 'hist')

% Mean correlation
meanX = sum(xVal .* normHist) / sum(normHist);
plot([meanX meanX], [0 .005], 'g', 'linewidth', 2)
% Median correlation
cumulativeSum = cumsum(normHist);
medianIndex = find(cumulativeSum >= sum(normHist) / 2, 1);
medianX = xVal(medianIndex);
plot([medianX medianX], [0 .005], 'r', 'linewidth', 2)
xlim(xlimConst)

title(['DS Signal Correlations'], 'interpreter', 'none')

% -------------------------
% M56-DS areas signal correlations
% -------------------------
[rho,pval] = corr(meanSpikes(:, m56Ind), meanSpikes(:, dsInd));

edges = -1 : .01 : 1;
xVal = edges(1:end-1);
N = histcounts(rho(:), edges);
normHist = N / sum(N);

% Plot
subplot(1,3,3)
cla
hold on
bar(xVal, normHist, 'hist')

% Mean correlation
meanX = sum(xVal .* normHist) / sum(normHist);
plot([meanX meanX], [0 .005], 'g', 'linewidth', 2)
% Median correlation
cumulativeSum = cumsum(normHist);
medianIndex = find(cumulativeSum >= sum(normHist) / 2, 1);
medianX = xVal(medianIndex);
plot([medianX medianX], [0 .005], 'r', 'linewidth', 2)
xlim(xlimConst)

%Gaussian mixture model
GMModel = fitgmdist(rho(:),2);
mu = GMModel.mu;
sigma = GMModel.Sigma(:);
gmwt = GMModel.ComponentProportion;
x = linspace(edges(1),edges(end),1000);
pdfValues = pdf(GMModel, x');
plot(x, pdfValues, 'k', 'LineWidth', 3);
pdf1 = normpdf(x, mu(1), sqrt(sigma(1)));
pdf2 = normpdf(x, mu(2), sqrt(sigma(2)));
plot(x, pdf1*gmwt(1), 'r', 'LineWidth', 3)
plot(x, pdf2*gmwt(2), 'r', 'LineWidth', 3)

title(['M56-DS Signal Correlations'], 'interpreter', 'none')

%     imagesc(rho)
% colormap(bluewhitered)
% colorbar

%% Noise correlations
noiseCorrM56 = zeros(length(m56Ind), length(m56Ind), length(analyzeBhv));
noiseCorrDS = zeros(length(dsInd), length(dsInd), length(analyzeBhv));
noiseCorrCross = zeros(length(m56Ind), length(dsInd), length(analyzeBhv));
edges = -1 : .01 : 1;
xVal = edges(1:end-1);
xlimConst = [-.6 .6];
for iBhv = 1 : length(analyzeBhv)
    % -------------------------
    % M56 areas correlations
    % -------------------------
    [iCorr iPval] = corr(spikesPerTrial{iBhv}(:, m56Ind));
    noiseCorrM56(:,:,iBhv) = iCorr;
    returnIdx = tril(true(length(m56Ind)), -1);

    N = histcounts(iCorr(returnIdx), edges);
    normHist = N / sum(N);

    subplot(1,3,1)
    cla
    hold on
    bar(xVal, normHist, 'hist')

    % Mean correlation
    meanX = sum(xVal .* normHist) / sum(normHist);
    plot([meanX meanX], [0 .01], 'g', 'linewidth', 2)
    % Median correlation
    cumulativeSum = cumsum(normHist);
    medianIndex = find(cumulativeSum >= sum(normHist) / 2, 1);
    medianX = xVal(medianIndex);
    plot([medianX medianX], [0 .01], 'r', 'linewidth', 2)
    xlim(xlimConst)
    title(['M56 Noise Correlations for ', analyzeBhv{iBhv}], 'interpreter', 'none')

    % -------------------------
    % DS correlations
    % -------------------------
    [iCorr iPval] = corr(spikesPerTrial{iBhv}(:, dsInd));
    noiseCorrDS(:,:,iBhv) = iCorr;
    returnIdx = tril(true(length(dsInd)), -1);

    N = histcounts(iCorr(returnIdx), edges);
    normHist = N / sum(N);

    subplot(1,3,2)
    cla
    hold on
    bar(xVal, normHist, 'hist')

    % Mean correlation
    meanX = sum(xVal .* normHist) / sum(normHist);
    plot([meanX meanX], [0 .01], 'g', 'linewidth', 2)
    % Median correlation
    cumulativeSum = cumsum(normHist);
    medianIndex = find(cumulativeSum >= sum(normHist) / 2, 1);
    medianX = xVal(medianIndex);
    plot([medianX medianX], [0 .01], 'r', 'linewidth', 2)
    xlim(xlimConst)
    title(['DS Noise Correlations for ', analyzeBhv{iBhv}], 'interpreter', 'none')

    % -------------------------
    % Across areas correlations
    % -------------------------
    [iCorr iPval] = corr(spikesPerTrial{iBhv}(:, m56Ind), spikesPerTrial{iBhv}(:, dsInd));
    noiseCorrCross(:,:,iBhv) = iCorr;

    N = histcounts(noiseCorrCross(:,:,iBhv), edges);
    normHist = N / sum(N);

    subplot(1,3,3)
    cla
    hold on
    bar(xVal, normHist, 'hist')

    % Mean correlation
    meanX = sum(xVal .* normHist) / sum(normHist);
    plot([meanX meanX], [0 .01], 'g', 'linewidth', 2)
    % Median correlation
    cumulativeSum = cumsum(normHist);
    medianIndex = find(cumulativeSum >= sum(normHist) / 2, 1);
    medianX = xVal(medianIndex);
    plot([medianX medianX], [0 .01], 'r', 'linewidth', 2)
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

minCurrDur = .2;
minBeforeDur = .2; % previous behavior must last within a range (sec)
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
                seqStartTimes = [seqStartTimes; [dataBhvTruncate.prevStartTime(goodSeqIdx), dataBhvTruncate.bhvStartTime(goodSeqIdx)]];
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

%%
bhvCurr = analyzeCodes(strcmp(analyzeBhv, 'investigate_2'));
bhvPrev = analyzeCodes(strcmp(analyzeBhv, 'locomotion'));
bhvCurr = analyzeCodes(strcmp(analyzeBhv, 'locomotion'));
bhvPrev = analyzeCodes(strcmp(analyzeBhv, 'investigate_2'));
seqStr = ['locomotion after investigate_2'];

m56Ind = find(strcmp(areaLabels, 'M56'));
dsInd = find(strcmp(areaLabels, 'DS'));

periEventTime = -.2 : opts.frameSize : .2; % seconds around onset
dataWindow = periEventTime(1:end-1) / opts.frameSize; % frames around onset (remove last frame)

goodStarts = dataBhvTruncate.bhvID == bhvCurr & validBhvTruncate(:, opts.bhvCodes == bhvCurr);
allStarts = floor(dataBhvTruncate.bhvStartTime(goodStarts) ./ opts.frameSize);
seqStarts = floor(seqStartTimes{strcmp(sequenceNames, seqStr)} ./ opts.frameSize);

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

[p,h,stats] = ranksum(allM56Corr, seqM56CorrCurr)
% Plot the distributions
clf
subplot(1,2,1)
edges = -1 : .01 : 1;
N = histcounts(allM56Corr, edges);
normHist = N / sum(N);
bar(edges(1:end-1), normHist, 'b', 'FaceAlpha', .7);%, 'hist')
hold on
xlim([-.5 .5])
N = histcounts(seqM56CorrCurr, edges);
normHist = N / sum(N);
bar(edges(1:end-1), normHist, 'r', 'FaceAlpha', .7);%, 'hist')
yl = ylim;
plot([median(allM56Corr) median(allM56Corr)], [.9*yl(2) yl(2)], 'b', 'linewidth', 4)
plot([median(seqM56CorrCurr) median(seqM56CorrCurr)], [.9*yl(2) yl(2)], 'r', 'linewidth', 4)
% xline(median(seqM56CorrCurr), 'r', 'linewidth', 4)
%


% Across area correlations
[allM56DSCorr, ~] = corr(allSpikesCurr(:, m56Ind), allSpikesCurr(:, dsInd));
[seqM56DSCorrCurr, ~] = corr(seqSpikesCurr(:, m56Ind), seqSpikesCurr(:, dsInd));
[seqM56DSCorrPrev, ~] = corr(seqSpikesPrev(:, m56Ind), seqSpikesPrev(:, dsInd));

[p,h,stats] = ranksum(allM56DSCorr(:), seqM56DSCorrCurr(:))


% Plot the distributions
subplot(1,2,2)
edges = -1 : .01 : 1;
N = histcounts(allM56DSCorr(:), edges);
normHist = N / sum(N);
bar(edges(1:end-1), normHist, 'b', 'FaceAlpha', .7);%, 'hist')
hold on
xlim([-.5 .5])
N = histcounts(seqM56DSCorrCurr(:), edges);
normHist = N / sum(N);
bar(edges(1:end-1), normHist, 'r', 'FaceAlpha', .7);%, 'hist')
yl = ylim;
plot([median(allM56DSCorr(:)) median(allM56DSCorr(:))], [.9*yl(2) yl(2)], 'b', 'linewidth', 4)
plot([median(seqM56DSCorrCurr(:)) median(seqM56DSCorrCurr(:))], [.9*yl(2) yl(2)], 'r', 'linewidth', 4)

