%% Create a 2-D data matrix of stacked peri-event start time windows (time X neuron)
bhv = 'locomotion';
bhvCode = analyzeCodes(strcmp(analyzeBhv, bhv));

periEventTime = -.5 : opts.frameSize : .5; % seconds around onset
dataWindow = round(periEventTime(1:end-1) / opts.frameSize); % frames around onset (remove last frame)


bhvStartFrames = floor(dataBhv.bhvStartTime(dataBhv.bhvID == bhvCode) ./ opts.frameSize);
bhvStartFrames(bhvStartFrames < dataWindow(end)) = [];
bhvStartFrames(bhvStartFrames > size(dataMat, 1) - dataWindow(end)) = [];

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

%%

% [c,lags] = corrcoef(dataMatM56,dataMatDSSub);
[c,lags] = xcorr(dataMatM56(:,1),dataMatDSSub(:,1));
stem(lags,c)

m56Corr = tril(corr(dataMatM56), -1);
edges = -1 : .01 : 1;
N = histcounts(m56Corr(:), edges);
normHist = N / sum(N);
bar(edges(1:end-1), normHist, 'hist')
mean(m56Corr(:))

% Cross correlation
crossC = corr(dataMatM56, dataMatDS);






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
    bar(edges(1:end-1), normHist, 'hist')
    xlim([dataWindow(1) dataWindow(end)])

    title(['Cross-correlation between M56 and DS: ', analyzeBhv{iBhv}], 'interpreter', 'none')
end





%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%       Signal and noise correlations
%       across pairwise M56 and DS neurons across behvaiors (mean per-event response of each neuron for each behavior
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
[rho,pval] = corr(meanSpikes(:, m56Ind), meanSpikes(:, dsInd));

    edges = -1 : .01 : 1;
    xVal = edges(1:end-1);
    N = histcounts(rho(:), edges);
    normHist = N / sum(N);
    bar(xVal, normHist, 'hist')

    % Mean correlation
    meanX = sum(xVal .* normHist) / sum(normHist);
    xline(meanX, 'k', 'linewidth', 2)

    % Median correlation
    cumulativeSum = cumsum(normHist);
    medianIndex = find(cumulativeSum >= sum(normHist) / 2, 1);
    medianX = xVal(medianIndex);
    xline(medianX, 'b', 'linewidth', 2)

    title(['Singal Correlations for M56 and DS'], 'interpreter', 'none')

%     imagesc(rho)
% colormap(bluewhitered)
% colorbar

%% Noise correlations
noiseCorr = zeros(length(m56Ind), length(dsInd), length(analyzeBhv));
for iBhv = 1 : length(analyzeBhv)
    [iCorr iPval] = corr(spikesPerTrial{iBhv}(:, m56Ind), spikesPerTrial{iBhv}(:, dsInd));
    noiseCorr(:,:,iBhv) = iCorr;

    edges = -1 : .01 : 1;
    xVal = edges(1:end-1);
    N = histcounts(noiseCorr(:,:,iBhv), edges);
    normHist = N / sum(N);
    bar(xVal, normHist, 'hist')

    % Mean correlation
    meanX = sum(xVal .* normHist) / sum(normHist);
    xline(meanX, 'k', 'linewidth', 2)

    % Median correlation
    cumulativeSum = cumsum(normHist);
    medianIndex = find(cumulativeSum >= sum(normHist) / 2, 1);
    medianX = xVal(medianIndex);
    xline(medianX, 'b', 'linewidth', 2)

    title(['Noise Correlations for ', analyzeBhv{iBhv}], 'interpreter', 'none')
    % imagesc(noiseCorr(:,:,iBhv))
    % colorbar
    % colormap(bluewhitered)
end


%% For each behavior, what are the most common preceding behaviors?
dataBhv.prevID = [nan; dataBhv.bhvID(1:end-1)];
dataBhv.prevDur = [nan; dataBhv.bhvDur(1:end-1)];
dataBhv.prevStartTime = [nan; dataBhv.bhvStartTime(1:end-1)];
dataBhvTruncate = dataBhv(3:end-3, :); % Truncate a few behaviors so we can look back and ahead in time a bit

minBeforeDur = .2; % previous behavior must last within a range (sec)
maxBeforeDur = 1;
minNumBoutsPrev = 20;

sequenceNames = cell(length(analyzeCodes));
startTimes = cell(length(analyzeCodes));
for iCurr = 1 : length(analyzeCodes)

    % Start here: For each behavior:
    % - Get the noise correlations for the current behavior (like above)
    % and all (valid) previous behaviors
    % - compare them (subtract?)
    for jPrev = 1 : length(analyzeCodes)
        if iCurr ~= jPrev
            sequenceNames{iCurr, jPrev} = [analyzeBhv{jPrev}, ' then ', analyzeBhv{iCurr}];
            % Make sure this sequence passes requisite criteria
            currIdx = dataBhvTruncate.bhvID == analyzeCodes(iCurr);
            goodSeqIdx = currIdx & ...
                dataBhvTruncate.prevID == analyzeCodes(jPrev) & ...
                dataBhvTruncate.prevDur >= minBeforeDur & ...
                dataBhvTruncate.prevDur <= maxBeforeDur;
            if sum(goodSeqIdx)
            startTimes{iCurr, jPrev} = [dataBhvTruncate.prevStartTime(goodSeqIdx), dataBhvTruncate.bhvStartTime(goodSeqIdx)];
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



