%% Get data from get_standard_data

opts = neuro_behavior_options;
opts.frameSize = .05; % 50 ms framesize for now
opts.collectFor = 60*60; % Get 45 min
opts.minActTime = .16;

get_standard_data


%% Create a 2-D data matrix of stacked peri-event start time windows (time X neuron)
% bhv = 'locomotion';
for iBhv = 1 : length(analyzeCodes)
    bhvCode = analyzeCodes(iBhv);

    periEventTime = -.2 : opts.frameSize : .2; % seconds around onset
    periWindow = round(periEventTime(1:end-1) / opts.frameSize); % frames around onset (remove last frame)
    preEventTime = -1 : opts.frameSize : -.6; % seconds before onset
    preWindow = round(preEventTime(1:end-1) / opts.frameSize); % frames around onset (remove last frame)


    bhvStartFrames = 1 + floor(dataBhv.StartTime(dataBhv.ID == bhvCode) ./ opts.frameSize);
    bhvStartFrames(bhvStartFrames < abs(preWindow(1))+1) = [];
    bhvStartFrames(bhvStartFrames > size(dataMat, 1) - periWindow(end)-1) = [];

    nTrial = length(bhvStartFrames);


    dataMatPeri = [];
    dataMatPre = [];
    for j = 1 : nTrial
        dataMatPeri = [dataMatPeri; dataMat(bhvStartFrames(j) + periWindow ,:)];
        dataMatPre = [dataMatPre; dataMat(bhvStartFrames(j) + preWindow ,:)];
    end

    % Get data mat for M56 and size-matched subset of DS
    dataMatM56Peri = dataMatPeri(:, strcmp(areaLabels, 'M56'));
    dataMatDSPeri =  dataMatPeri(:, strcmp(areaLabels, 'DS'));
    dataMatM56Pre = dataMatPre(:, strcmp(areaLabels, 'M56'));
    dataMatDSPre =  dataMatPre(:, strcmp(areaLabels, 'DS'));
    % subDSIdx = randperm(size(dataMatDS, 2), size(dataMatM56, 2));
    % dataMatDSSub = dataMatDS(:, subDSIdx);

    %     Overall correlations across the whole recording

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

    sgtitle(['Peri and Pre within area correlations: ', analyzeBhv(iBhv)])
end







%% Which area has modulation first? Cross-correlation across behaviors
% For each behavior onset, determine the cross-correlation of mean spiking across neurons between M56 and DS
% plotCodes = codes(codes>-1);
% plotBhv = behaviors(codes>-1);

fig = figure(88); clf
set(fig, 'Position', monitorTwo);
nPlot = length(analyzeCodes);
[ax, pos] = tight_subplot(2, ceil(nPlot/2), [.08 .02], .1);
% hold all;
colors = colors_for_behaviors(analyzeCodes);

    zTime = -3 : opts.frameSize : 2;  % zscore on a 5sec window peri-onset
    zWindow = round(zTime(1:end-1) / opts.frameSize);
    zStartInd = find(zTime == 0);
    fullTime = -1 : opts.frameSize : 1; % seconds around onset
    fullWindow = round(fullTime(1:end-1) / opts.frameSize); % frames around onset w.r.t. zWindow (remove last frame)
nPermute = 1000;


for iBhv = 1 : length(analyzeCodes)
    % Get mean peri-event spiking data from the two areas for this behavior
    iPeriM56 = mean(mean(eventMatZ{iBhv}(:, idM56, :), 2), 3);
    iPeriDS = mean(mean(eventMatZ{iBhv}(:, idDS, :), 2), 3);
    % iPeriM56 = mean(eventMat{iBhv}(14:end, idM56, 1), 2);
    % iPeriDS = mean(eventMat{iBhv}(14:end, idDS, 1), 2);

    % Get randomized windows of spiking data from the two areas for
    % comparison
    % iPeriM56Rand = zeros(length(zWindow), length(idM56), size(eventMatZ{iBhv}, 3));
    % iPeriDSRand = zeros(length(zWindow), length(idDS), size(eventMatZ{iBhv}, 3));
    iPeriM56Rand = zeros(length(zWindow), length(idM56), nPermute);
    iPeriDSRand = zeros(length(zWindow), length(idDS), nPermute);
       % Loop to extract n random windows
        % Randomly select the starting index of the window
    startIdx = randperm(size(dataMat, 1)-length(zWindow) - 2) - zWindow(1) + 1;
    % for j = 1:size(eventMatZ{iBhv}, 3)
    for j = 1:nPermute

        % Extract the window from dataMat
        iPeriM56Rand(:, :, j) = dataMat(startIdx(j) + zWindow - 1, idM56);
        iPeriDSRand(:, :, j) = dataMat(startIdx(j) + zWindow - 1, idDS);

    end
    iMeanM56Rand = mean(mean(iPeriM56Rand, 3), 1);
    iMeanDSRand = mean(mean(iPeriDSRand, 3), 1);
    iStdM56Rand = std(mean(iPeriM56Rand, 3), [], 1);
    iStdDSRand = std(mean(iPeriDSRand, 3), [], 1);
    iM56ZRand = (iPeriM56Rand - iMeanM56Rand) ./ iStdM56Rand;
    iDSZRand = (iPeriDSRand - iMeanDSRand) ./ iStdDSRand;

    iPeriM56RandZ = mean(mean(iM56ZRand(zStartInd + fullWindow, :, :), 2), 3);
    iPeriDSRandZ = mean(mean(iDSZRand(zStartInd + fullWindow, :, :), 2), 3);

    [c, lags] = xcorr(iPeriM56, iPeriDS, 25, 'normalized');
    [cRand, ~] = xcorr(iPeriM56RandZ, iPeriDSRandZ, 25, 'normalized');
    % [cM56, ~] = xcorr(iPeriM56, iPeriM56, 30, 'normalized');
    % [cDS, ~] = xcorr(iPeriDS, iPeriDS, 30, 'normalized');
    R = corrcoef(iPeriM56, iPeriDS);
    RRand = corrcoef(iPeriM56RandZ, iPeriDSRandZ);

    axes(ax(iBhv)); % hold on;
    plot(lags, cRand, '--', 'color', [.5 .5 .5], 'lineWidth', 2)
    hold on;
    plot(lags, c, 'color', colors(iBhv,:), 'lineWidth', 3)
    % plot(lags, cM56, '--k', 'lineWidth', 2)
    % plot(lags, cDS, '--r', 'lineWidth', 2)
    xline(0)
    yline(0)

        title(analyzeBhv{iBhv}, 'interpreter', 'none');

    % title(num2str(corr(mean(dataMat(plotFrames, idM56), 2), mean(dataMat(plotFrames, idDS), 2))))
end
sgtitle('M56 X DS Cross-correlations peri-behavior-transitions (z-scored)')

%% Across some time, regardless of aligning to transitions
randWindow = randi([50 70000]) + fullWindow;
firstFrames = find(diff(bhvIDMat) ~= 0) + 1; % 1 frame prior to all behavior transitions
preIndID = bhvIDMat(firstFrames);

plotFrames = firstFrames(19) + fullWindow;
% plotFrames = 40001:44350;
meanM56 = mean(dataMat(plotFrames, idM56), 2);
meanDS = mean(dataMat(plotFrames, idDS), 2);
meanM56Z = zscore(mean(dataMat(plotFrames, idM56), 2));
meanDSZ = zscore(mean(dataMat(plotFrames, idDS), 2));
% plot(meanM56);
% hold on
% plot(meanDS, 'r');
    % [c, lags] = xcorr(meanM56, meanDS, 50, 'normalized');
    [cZ, lags] = xcorr(meanM56Z, meanDSZ, 30, 'normalized');
    % [c, lags] = xcorr(meanM56, meanDS, 50);
    % [c, lags] = xcorr(meanM56, meanDS(randperm(length(meanDS))), 50);
    % R = corrcoef(meanM56, meanDS)
    % [c, lags] = xcorr(meanM56, meanDS, 50, 'normalized');
    % R = corrcoef(meanM56, meanDS);
    figure(89); clf; hold on;
    % plot(lags, c, 'linewidth', 2)
    plot(lags, cZ, '--b', 'linewidth', 2)
    xline(0)







%% Cross-Correlation across areas (across the whole session)

% % [c,lags] = corrcoef(dataMatM56,dataMatDSSub);
% [c,lags] = xcorr(dataMatM56(:,1),dataMatDSSub(:,1)); % this is for example first neuron in each brain area
% subplot(1,2,1)
% stem(lags,c)
%
% crossC = corr(dataMatM56, dataMatDS);
% edges = -1 : .01 : 1;
% binCenters = (edges(1:end-1) + edges(2:end)) / 2;
% N = histcounts(crossC(:), edges, 'Normalization', 'pdf');
% subplot(1,2,2)
% bar(binCenters, N, 'hist')
% mean(crossC(:))
%










%% Cross-correlations for each behavior between pairwise M56 and DS (their full populations)

% Create a 3-D psth data matrix of stacked peri-event start time windows (time X neuron X trial)

periEventTime = -.3 : opts.frameSize : .3; % seconds around onset
dataWindow = round(periEventTime(1:end-1) / opts.frameSize); % frames around onset (remove last frame)
eventMat = cell(length(analyzeBhv), 1);
for iBhv = 1 : length(analyzeBhv)
    bhvCode = analyzeCodes(strcmp(analyzeBhv, analyzeBhv{iBhv}));

    bhvStartFrames = 1 + floor(dataBhv.StartTime(dataBhv.ID == bhvCode) ./ opts.frameSize);
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
maxLag = 3;
meanLagPerBhv = cell(length(analyzeBhv), 1);
for iBhv = 1 : length(analyzeBhv)
    % meanLag = zeros(length(idM56), length(idDS)); % collects mean pairwise lags for each behavior (lags averaged across all onset times for this behavior)
    meanLag = nan(length(idM56), length(idDS)); % collects mean pairwise lags for each behavior (lags averaged across all onset times for this behavior)

    % Across all M56 neurons
    for m = 1 : length(idM56)

        % Across all DS neurons
        for d = 1 : length(idDS)



            %     % Across each trials
            %     nLag = []; % Avg pairwise lag across all trials for this behavior
            %     for n = 1 : size(eventMat{iBhv}, 3) % Across all onsets for this behavior
            %
            %         if sum(eventMat{iBhv}(:,idM56(m), n)) && sum(eventMat{iBhv}(:,idDS(d), n)) % Only use trials if both neurons (from each area) have at least on spike
            %             [c,lags] = xcorr(eventMat{iBhv}(:,idM56(m), n), eventMat{iBhv}(:,idDS(d), n), maxLag, 'normalized');
            %
            %             % nLag = [nLag; sum(lags(:) .* c) / sum(c)];
            %
            %             % take lag index of max c
            %             nMax = max(c);
            %             maxIdx = find(c == nMax);
            %             if length(maxIdx) > 1
            %                         randomIndex = randi(length(maxIdx));
            % index = maxIdx(randomIndex);
            %             else
            %                 index = maxIdx;
            %             end
            %             nLag = [nLag; lags(index)];
            %
            %             % figure(55);
            %             % plot(lags, c)
            %         end
            %
            %     end
            %     meanLag(m, d) = mean(nLag);
            %

            % Average psths across trials
            [c,lags] = xcorr(mean(eventMat{iBhv}(:,idM56(m),:), 3), mean(eventMat{iBhv}(:,idDS(d), :), 3), maxLag, 'normalized');

            % take lag index of max c
            nMax = max(c);
            maxIdx = find(c == nMax);
            if length(maxIdx) > 1
                randomIndex = randi(length(maxIdx));
                index = maxIdx(randomIndex);
            else
                index = maxIdx;
            end
            if ~isempty(index)
                meanLag(m, d) = lags(index);
            end
        end

    end
    meanLagPerBhv{iBhv} = meanLag;


    figure(52)
    subplot(1,2,1)
    cla
    plot(lags, c)

    subplot(1,2,2)
    cla

    edges = -maxLag : 1 : maxLag;
    N = histcounts(meanLagPerBhv{iBhv}(:), edges, 'Normalization', 'pdf');
    binCenters = (edges(1:end-1) + edges(2:end)) / 2;

    hold on;
    % bar(edges(1:end-1), normHist, 'hist')
    bar(binCenters, N, 'hist')
    xlim([dataWindow(1) dataWindow(end)])
    xlim([-maxLag maxLag])

    % Mean lag across all pairwise neurons
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

%%
dataBhvTrunc = dataBhv(3:end-2, :);
validBhvTrunc = validBhv(3:end-2,:);

%% If you want to subsample bouts (match the number of bouts for each behavior...
% Do you want to subsample to match the number of bouts?
matchBouts = 0;
%%
nBout = zeros(length(analyzeCodes), 1);
for i = 1 : length(analyzeCodes)
    nBout(i) = sum(dataBhvTrunc.ID == analyzeCodes(i) & dataBhvTrunc.Valid);
end
nSample = min(nBout);


%% Get relevant data (Using dataMat/dataMatZ
periEventTime = -.1 : opts.frameSize : .1; % seconds around onset
corrWindow = periEventTime(1:end-1) / opts.frameSize; % frames around onset (remove last frame)

meanSpikes = zeros(length(analyzeBhv), size(dataMat, 2));
meanSpikesZ = meanSpikes;
spikesPerTrial = cell(length(analyzeBhv), 1);
spikesPerTrialZ = cell(length(analyzeBhv), 1);
for iBhv = 1 : length(analyzeBhv)
    bhvCode = analyzeCodes(strcmp(analyzeBhv, analyzeBhv{iBhv}));

    iStartFrames = 1 + floor(dataBhvTrunc.StartTime(dataBhvTrunc.ID == bhvCode & dataBhvTrunc.Valid) ./ opts.frameSize);

    if matchBouts
        iRand = randperm(length(iStartFrames));
        iStartFrames = iStartFrames(iRand(1:nSample));
    end
    nTrial = length(iStartFrames);

    iEventMat = zeros(nTrial, size(dataMat, 2)); % nTrial X nNeurons
    iMeanMatZ = zeros(nTrial, size(dataMat, 2)); % nTrial X nNeurons
    for j = 1 : nTrial
        iEventMat(j,:) = sum(dataMat(iStartFrames(j) + corrWindow ,:), 1);
        iMeanMatZ(j,:) = mean(dataMatZ(iStartFrames(j) + corrWindow ,:), 1);
    end

    meanSpikes(iBhv, :) = mean(iEventMat, 1);
    meanSpikesZ(iBhv, :) = mean(iMeanMatZ, 1);
    spikesPerTrial{iBhv} = iEventMat;
    spikesPerTrialZ{iBhv} = iMeanMatZ;

end

%%  Get relevant data: Using eventMat from spiking_script.m

periEventTime = -.1 : opts.frameSize : .1; % seconds around onset
corrWindow = fullStartInd + periEventTime(1:end-1) / opts.frameSize; % frames around onset (remove last frame)

meanSpikes = zeros(length(analyzeBhv), size(eventMat{1}, 2));
meanSpikesZ = meanSpikes;
spikesPerTrial = cell(length(analyzeBhv), 1);
spikesPerTrialZ = cell(length(analyzeBhv), 1);
for iBhv = 1 : length(analyzeBhv)

    if matchBouts
        iRand = randperm(size(eventMatZ{iBhv}, 3));
        iTrialInd = iRand(1:nSample);
    else
        iTrialInd = 1:size(eventMatZ{iBhv}, 3);
    end
    spikesPerTrialZ{iBhv} = mean(eventMatZ{iBhv}(corrWindow, :, iTrialInd));
    spikesPerTrialZ{iBhv} = permute(spikesPerTrialZ{iBhv}, [3 2 1]);
    meanSpikesZ(iBhv, :) = mean(mean(eventMatZ{iBhv}(corrWindow, :, iTrialInd), 1), 3);
end

%%

%% Signal correlations
plotFlag = 1;

xlimConst = [-1 1];

% M56 signal correlations
% -------------------------
[rhoM56,pval] = corr(meanSpikesZ(:, idM56));

% DS signal correlations
% -------------------------
[rhoDS,pval] = corr(meanSpikesZ(:, idDS));

% M56-DS areas signal correlations
% -------------------------
[rhoX,pval] = corr(meanSpikesZ(:, idM56), meanSpikesZ(:, idDS));

% Plotting
% -------------------------
if plotFlag
    plotGauss = 0;
    edges = -1 : .05 : 1;
    binCenters = (edges(1:end-1) + edges(2:end)) / 2;

    % M56
    % -------------------------
    returnIdx = tril(true(length(idM56)), -1);
    N = histcounts(rhoM56(returnIdx), edges, 'Normalization', 'pdf');

    figure(49)
    clf
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


    % DS
    % -------------------------
    returnIdx = tril(true(length(idDS)), -1);
    N = histcounts(rhoDS(returnIdx), edges, 'Normalization', 'pdf');

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


    % M56 X DS
    % -------------------------
    N = histcounts(rhoX(:), edges, 'Normalization', 'pdf');

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

    % Gaussian mixture model
    if plotGauss
        data = rhoX(:);
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

        % if plotGauss
        %         %Gaussian mixture model
        %     GMModel = fitgmdist(rho(:),2);
        %     mu = GMModel.mu;
        %     sigma = GMModel.Sigma(:);
        %     gmwt = GMModel.ComponentProportion;
        %     x = linspace(edges(1),edges(end),1000);
        %     pdfValues = pdf(GMModel, x');
        %     plot(x, pdfValues, 'k', 'LineWidth', 3);
        %     pdf1 = normpdf(x, mu(1), sqrt(sigma(1)));
        %     pdf2 = normpdf(x, mu(2), sqrt(sigma(2)));
        %     plot(x, pdf1*gmwt(1), 'r', 'LineWidth', 3)
        %     plot(x, pdf2*gmwt(2), 'r', 'LineWidth', 3)
        % end
    end
end



%% Noise correlations
plotFlag = 1;

noiseCorrM56 = zeros(length(idM56), length(idM56), length(analyzeBhv));
noiseCorrDS = zeros(length(idDS), length(idDS), length(analyzeBhv));
noiseCorrX = zeros(length(idM56), length(idDS), length(analyzeBhv));
edges = -1 : .01 : 1;
binCenters = (edges(1:end-1) + edges(2:end)) / 2;
xVal = edges(1:end-1);
xlimConst = [-.6 .6];
for iBhv = 1 : length(analyzeBhv)

    % M56 areas correlations
    % -------------------------
    [iCorrM56 iPval] = corr(spikesPerTrialZ{iBhv}(:, idM56));
    noiseCorrM56(:,:,iBhv) = iCorrM56;

    % DS correlations
    % -------------------------
    [iCorrDS iPval] = corr(spikesPerTrialZ{iBhv}(:, idDS));
    noiseCorrDS(:,:,iBhv) = iCorrDS;


    % M56 X DS correlations
    % -------------------------
    [iCorrX iPval] = corr(spikesPerTrialZ{iBhv}(:, idM56), spikesPerTrialZ{iBhv}(:, idDS));
    noiseCorrX(:,:,iBhv) = iCorrX;



    % -------------------------
    % Plotting
    % -------------------------
    if plotFlag

        % M56
        % ----------
        returnIdx = tril(true(length(idM56)), -1);
        N = histcounts(iCorrM56(returnIdx), edges, 'Normalization', 'pdf');

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

        %  DS
        % ----------
        returnIdx = tril(true(length(idDS)), -1);
        N = histcounts(iCorrDS(returnIdx), edges, 'Normalization', 'pdf');

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


        % M56 X DS
        % ----------
        N = histcounts(noiseCorrX(:,:,iBhv), edges, 'Normalization', 'pdf');

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
end








%%    Pairwise correlations as a function of pairwise peri-onset modulation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
plotFlag = 1;
idX = [idM56 idDS];

% Get the neuron indices of the correlation matrix
returnIdxM56 = tril(true(length(idM56)), -1);
[rowM56, colM56] = find(returnIdxM56);
returnIdxDS = tril(true(length(idDS)), -1);
[rowDS, colDS] = find(returnIdxDS);
% returnIdxX = ones(length(idM56), length(idDS));
% [rowX, colX] = find(returnIdxX);
[rowX, colX] = ndgrid(idM56, idDS);
idxXPairs = [rowX(:) colX(:)];

noiseCorrM56Pair = zeros(length(rowM56), length(analyzeCodes));
neuronMod1M56 = zeros(length(rowM56), length(analyzeCodes));
neuronMod2M56 = zeros(length(rowM56), length(analyzeCodes));
noiseCorrDSPair = zeros(length(rowDS), length(analyzeCodes));
neuronMod1DS = zeros(length(rowDS), length(analyzeCodes));
neuronMod2DS = zeros(length(rowDS), length(analyzeCodes));
noiseCorrXPair = zeros(size(idxXPairs, 1), length(analyzeCodes));
neuronMod1X = zeros(size(idxXPairs, 1), length(analyzeCodes));
neuronMod2X = zeros(size(idxXPairs, 1), length(analyzeCodes));

% Loop through behaviors to get correlations and neural modulations
for iBhv = 1 : length(analyzeCodes)
    iCorr = noiseCorrM56(:,:,iBhv);
    noiseCorrM56Pair(:, iBhv) = iCorr(returnIdxM56);
    iCorr = noiseCorrDS(:,:,iBhv);
    noiseCorrDSPair(:, iBhv) = iCorr(returnIdxDS);
    iCorr = noiseCorrX(:,:,iBhv);
    noiseCorrXPair(:, iBhv) = iCorr(:);

    % Go through each pair and compare the correlation value to each
    % neuron's behavior-related modulation
    for k = 1 : size(noiseCorrM56Pair, 1)
        neuronMod1M56(k, iBhv) = meanSpikesZ(iBhv, idM56(rowM56(k)));
        neuronMod2M56(k, iBhv) = meanSpikesZ(iBhv, idM56(colM56(k)));
    end
    for k = 1 : size(noiseCorrDSPair, 1)
        neuronMod1DS(k, iBhv) = meanSpikesZ(iBhv, idDS(rowDS(k)));
        neuronMod2DS(k, iBhv) = meanSpikesZ(iBhv, idDS(colDS(k)));
    end
    for k = 1 : size(noiseCorrXPair, 1)
        % neuronMod1X(k, iBhv) = meanSpikesZ(iBhv, idX(rowX(k)));
        % neuronMod2X(k, iBhv) = meanSpikesZ(iBhv, idX(colX(k)));
        neuronMod1X(k, iBhv) = meanSpikesZ(iBhv, idxXPairs(k, 1));
        neuronMod2X(k, iBhv) = meanSpikesZ(iBhv, idxXPairs(k, 2));
    end
    for m = 1 : length(idM56)
        for d = 1 : length(idDS)

        end
    end
end


if plotFlag
    plotBhv = 'X'; % M56 DS X

    % Get monitor positions and size
    % monitorPositions = get(0, 'MonitorPositions');
    % secondMonitorPosition = monitorPositions(size(monitorPositions, 1), :); % Just use single monitor if you don't have second one
    % Create a maximized figure on the second monitor
    fig = figure(70);
    clf
    % set(fig, 'Position', secondMonitorPosition);
    nPlot = length(analyzeCodes);
    [ax, pos] = tight_subplot(ceil(nPlot/4), ceil(nPlot/4));
    colors = colors_for_behaviors(analyzeCodes);

    switch plotBhv
        case 'M56'
            maxCorr = max(noiseCorrM56Pair(:));
            minCorr = min(noiseCorrM56Pair(:));
            maxMod = max([neuronMod1M56(:); neuronMod2M56(:)]);
            minMod = min([neuronMod1M56(:); neuronMod2M56(:)]);
            for iBhv = 1 : length(analyzeCodes)
                axes(ax(iBhv))
                hold on
                xlim([minMod maxMod])
                ylim([minCorr maxCorr]);
                for k = 1 : length(noiseCorrM56Pair)
                    plot([neuronMod1M56(k, iBhv) neuronMod2M56(k, iBhv)], [noiseCorrM56Pair(k, iBhv) noiseCorrM56Pair(k, iBhv)], 'color', colors(iBhv,:), 'linewidth', 2)
                end
                xline(0, 'linewidth', 2);
                yline(0, 'linewidth', 2);
                title(analyzeBhv{iBhv}, 'interpreter', 'none')
                if iBhv == 9
                    ylabel('Correlation')
                    xlabel('Pairwise modulation')
                end

            end
        case 'DS'
            maxCorr = max(noiseCorrDSPair(:));
            minCorr = min(noiseCorrDSPair(:));
            maxMod = max([neuronMod1DS(:); neuronMod2DS(:)]);
            minMod = min([neuronMod1DS(:); neuronMod2DS(:)]);
            for iBhv = 1 : length(analyzeCodes)
                axes(ax(iBhv))
                hold on
                xlim([minMod maxMod])
                ylim([minCorr maxCorr]);
                for k = 1 : length(noiseCorrDSPair)
                    plot([neuronMod1DS(k, iBhv) neuronMod2DS(k, iBhv)], [noiseCorrDSPair(k, iBhv) noiseCorrDSPair(k, iBhv)], 'color', colors(iBhv,:), 'linewidth', 2)
                end
                xline(0, 'linewidth', 2);
                yline(0, 'linewidth', 2);
                title(analyzeBhv{iBhv}, 'interpreter', 'none')
                if iBhv == 9
                    ylabel('Correlation')
                    xlabel('Pairwise modulation')
                end

            end
        case 'X'
            maxCorr = max(noiseCorrXPair(:));
            minCorr = min(noiseCorrXPair(:));
            maxMod = max([neuronMod1X(:); neuronMod2X(:)]);
            minMod = min([neuronMod1X(:); neuronMod2X(:)]);
            for iBhv = 1 : length(analyzeCodes)
                axes(ax(iBhv))
                hold on
                xlim([minMod maxMod])
                ylim([minCorr maxCorr]);
                for k = 1 : length(noiseCorrXPair)
                    plot([neuronMod1X(k, iBhv) neuronMod2X(k, iBhv)], [noiseCorrXPair(k, iBhv) noiseCorrXPair(k, iBhv)], 'color', colors(iBhv,:), 'linewidth', 2)
                end
                xline(0, 'linewidth', 2);
                yline(0, 'linewidth', 2);
                title(analyzeBhv{iBhv}, 'interpreter', 'none')
                if iBhv == 9
                    ylabel('Correlation')
                    xlabel('Pairwise modulation')
                end

            end
    end
end




%%        Are the pairwise correlations consistent across behaviors, for each pair?
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% M56 individual pairs
returnIdx = tril(true(length(idM56)), -1);
signalCorrM56Pair = rhoM56(returnIdx);
figure(44)
for k = 650 : length(noiseCorrM56Pair)
    clf
    hold on
    plot(neuronMod1M56(k,:), 'b', 'linewidth', 2)
    plot(neuronMod2M56(k,:), 'color', [0 .5 1], 'linewidth', 2)
    yline(signalCorrM56Pair(k), 'g', 'linewidth', 2)
    plot(noiseCorrM56Pair(k,:), 'r', 'linewidth', 2)
    yline(0, '--k')
    legend({'neuron1 Tune', 'neuron2 Tune', 'Signal Corr', 'Noise Corr'})
end

%% DS individual pairs
returnIdx = tril(true(length(idDS)), -1);
signalCorrDSPair = rhoDS(returnIdx);
figure(44)
for k = 1 : length(noiseCorrDSPair)
    clf
    hold on
    plot(neuronMod1DS(k,:), 'b', 'linewidth', 2)
    plot(neuronMod2DS(k,:), 'color', [0 .5 1], 'linewidth', 2)
    yline(signalCorrDSPair(k), 'g', 'linewidth', 2)
    plot(noiseCorrDSPair(k,:), 'r', 'linewidth', 2)
    yline(0, '--k')
    legend({'neuron1 Tune', 'neuron2 Tune', 'Signal Corr', 'Noise Corr'})
end

%% M56 X DS individual pairs
signalCorrXPair = rhoX(:);
figure(44)
for k = 1 : length(noiseCorrXPair)
    clf
    hold on
    plot(neuronMod1X(k,:), 'b', 'linewidth', 2)
    plot(neuronMod2X(k,:), 'color', [0 .5 1], 'linewidth', 2)
    yline(signalCorrXPair(k), 'g', 'linewidth', 2)
    plot(noiseCorrXPair(k,:), 'r', 'linewidth', 2)
    yline(0, '--k')
    legend({'neuron1 Tune', 'neuron2 Tune', 'Signal Corr', 'Noise Corr'})
    pause
end



%%              Plot Noise Correlation distributions for pairs

% Get signal correlation to draw a correlation line
returnIdxM56 = tril(true(length(idM56)), -1);
[rowM56, colM56] = find(returnIdxM56);   % indices of each neuron in the pair
signalCorrM56Pair = rhoM56(returnIdxM56);


for k = 550 : length(signalCorrM56Pair)
    figure(455)
    clf;

    subplot(1,2,2)
    hold on
    plot(neuronMod1M56(k,:), 'b', 'linewidth', 2)
    plot(neuronMod2M56(k,:), 'color', [0 .5 1], 'linewidth', 2)
    yline(signalCorrM56Pair(k), 'g', 'linewidth', 2)
    plot(noiseCorrM56Pair(k,:), 'r', 'linewidth', 2)
    yline(0, '--k')
    legend({'neuron1 Tune', 'neuron2 Tune', 'Signal Corr', 'Noise Corr'})



    subplot(1,2,1)
    axis square
    hold on;
    yline(0, '--k')
    xline(0, '--k')


    for iBhv = 1 : length(analyzeCodes)
        colors = colors_for_behaviors(analyzeCodes);
        % Get the trial to trial spike counts for these two neurons
        n1 = spikesPerTrialZ{iBhv}(:, idM56(rowM56(k)));
        n2 = spikesPerTrialZ{iBhv}(:, idM56(colM56(k)));


        % Calculate the regression slope and intercept
        slope = corr(n1, n2);

        % coefficients = polyfit(n1, n2, 1);
        % slope = coefficients(1);
        % intercept = coefficients(2);

        % Calculate mean and std
        meanN1 = mean(n1);
        meanN2 = mean(n2);
        stdN1 = std(n1);
        stdN2 = std(n2);

        % Generate regression line points
        % xLine = linspace(meanN1 - stdN1, meanN1 + stdN1, 100);
        % yLine = slope * xLine + intercept;

        % Plotting the regression line
        % plot(xLine, yLine, 'b-', 'LineWidth', 2);

        % Generate ellipse
        theta = linspace(0, 2 * pi, 100);
        xEllipse = stdN1 * cos(theta);
        yEllipse = stdN2 * sin(theta);

        % Angle for rotation
        angle = atan(slope);

        % Rotate the ellipse
        cosA = cos(angle);
        sinA = sin(angle);
        xRot = cosA * xEllipse - sinA * yEllipse;
        yRot = sinA * xEllipse + cosA * yEllipse;

        % Plot ellipse
        % plot(xRot, yRot, 'b-', 'LineWidth', 2);
        fill(meanN1 + xRot, meanN2 + yRot, colors(iBhv,:), 'FaceAlpha', 0.2, 'EdgeColor', colors(iBhv,:)); % Plot the filled and rotated ellipse

        % Make axes square and have same span
        xLimit = xlim;
        yLimit = ylim;
        if min(meanN1 + xRot) < xLimit(1); xLimit(1) = min(meanN1 + xRot); end
        if max(meanN1 + xRot) > xLimit(2); xLimit(2) = max(meanN1 + xRot); end
        if min(meanN2 + yRot) < yLimit(1); yLimit(1) = min(meanN2 + yRot); end
        if max(meanN2 + yRot) > yLimit(2); yLimit(2) = max(meanN2 + yRot); end

        % Calculate span
        xSpan = xLimit(2) - xLimit(1);
        ySpan = yLimit(2) - yLimit(1);

        % Ensure both axes have the same span
        if xSpan > ySpan
            yMid = mean(yLimit);
            yLimit = [yMid - xSpan / 2, yMid + xSpan / 2];
        elseif ySpan > xSpan
            xMid = mean(xLimit);
            xLimit = [xMid - ySpan / 2, xMid + ySpan / 2];
        end

        % Set the limits
        xlim(xLimit);
        ylim(yLimit);
    end
    % Draw the signal correlation line
    kCorr = signalCorrM56Pair(k);
    xLineLim = xlim;
    xSigLine = linspace(xLineLim(1), xLineLim(2), 100);
    ySigLine = kCorr * xSigLine;
    plot(xSigLine, ySigLine, 'k', 'linewidth', 2)

    % legend({'neuron1 Tune', 'neuron2 Tune', 'Signal Corr', 'Noise Corr'})

end








%% Individual pair correlations of positively modulated neurons: Examine the correlations between pairs of modulated neurons in each behavior:
% Do positively modulated neurons have pos/neg signal and noise
% correlations?

% Get posMod from neurobehavior_tuning_script

idInd = idM56;

colors = colors_for_behaviors(analyzeCodes);
for iBhv = 1 : length(analyzeCodes)
    % Which neurons are positively tuned for this behavior?
    iPosTuneInd = intersect(idInd, find(posMod(iBhv, :)));

    % What is their signal correlation?
    [sigRho,pval] = corr(meanSpikesZ(:, iPosTuneInd));
    returnIdx = tril(true(length(iPosTuneInd)), -1);
    [row, col] = find(returnIdx);   % indices of each neuron in the pair
    iSigCorr = sigRho(returnIdx);

    % Loop through all pairs of neurons
    for k = 1 : length(row)
        kSigCorr = iSigCorr(k); % The signal correlation for this pair of neurons

        % Noise correlation for this pair of (postively tuned) neurons for this behavior
        n1 = spikesPerTrialZ{iBhv}(:, iPosTuneInd(row(k)));
        n2 = spikesPerTrialZ{iBhv}(:, iPosTuneInd(col(k)));

        % Calculate the regression slope
        kNoiseCorr = corr(n1, n2);

        % Calculate mean and std
        meanN1 = mean(n1);
        meanN2 = mean(n2);
        stdN1 = std(n1);
        stdN2 = std(n2);

        % Generate ellipse
        theta = linspace(0, 2 * pi, 100);
        xEllipse = stdN1 * cos(theta);
        yEllipse = stdN2 * sin(theta);

        % Angle for rotation
        angle = atan(kNoiseCorr);

        % Rotate the ellipse
        cosA = cos(angle);
        sinA = sin(angle);
        xRot = cosA * xEllipse - sinA * yEllipse;
        yRot = sinA * xEllipse + cosA * yEllipse;


        figure(456)
        clf;
        % Plot ellipse
        subplot(1,2,1)
        hold on
        fill(meanN1 + xRot, meanN2 + yRot, colors(iBhv,:), 'FaceAlpha', 0.4, 'EdgeColor', colors(iBhv,:)); % Plot the filled and rotated ellipse
        scatter(n1, n2)
        ylim([min([n1; n2]-5) max([n1; n2])])
        xlim([min([n1; n2]-5) max([n1; n2])])
        title('Spikes per bout')
        xlabel('Neuron 1')
        title('Neuron 2')
        axis square
        yline(0, '--k')

        subplot(1,2,2)
        bar([1 2], [kSigCorr kNoiseCorr])
        ylim([-1 1])
        xticklabels({'Signal Corr', 'Noise Corr'})


        sgtitle('Pairs of pos. tuned neurons: Signal and Noise correlations')
    end
end



%% Distribution of signal and noise correlations for positively tuned neurons in each behavior
idInd = idM56;
edges = -.8 : .05 : .8;
binCenters = (edges(1:end-1) + edges(2:end)) / 2;

% Get monitor positions and size
% monitorPositions = get(0, 'MonitorPositions');
% secondMonitorPosition = monitorPositions(size(monitorPositions, 1), :); % Just use single monitor if you don't have second one
% Create a maximized figure on the second monitor
fig = figure(760);
clf
% set(fig, 'Position', secondMonitorPosition);
nPlot = length(analyzeCodes);
[ax, pos] = tight_subplot(ceil(nPlot/4), ceil(nPlot/4), [.04 .02]);
colors = colors_for_behaviors(analyzeCodes);

for iBhv = 1 : length(analyzeCodes)

    % Which neurons are positively tuned for this behavior?
    iPosTuneInd = intersect(idInd, find(posMod(iBhv, :)));

    returnIdx = tril(true(length(iPosTuneInd)), -1);
    [row, col] = find(returnIdx);   % indices of each neuron in the pair

    % What is their signal correlation across behaviors?
    [sigRho,pval] = corr(meanSpikesZ(:, iPosTuneInd));
    iSigCorr = sigRho(returnIdx);

    % What is their noise correlation for this behavior?
    noiseRho = corr(spikesPerTrialZ{iBhv}(:, iPosTuneInd));
    iNoiseCorr = noiseRho(returnIdx);

    axes(ax(iBhv)); hold on

    N = histcounts(iNoiseCorr, edges, 'Normalization', 'pdf');
    bar(binCenters, N, 'FaceColor', colors(iBhv,:), 'BarWidth', 1);

    N = histcounts(iSigCorr, edges, 'Normalization', 'pdf');
    bar(binCenters, N, 'FaceColor', [.5 .5 .5], 'BarWidth', 1, 'FaceAlpha', .6);

    ca = gca;
    % ca.YTickLabel = ca.YTick;
    ca.XTickLabel = ca.XTick;
    title(analyzeBhv{iBhv}, 'interpreter', 'none')
    if iBhv == 1
        legend({'Noise', 'Signal'})
    end
end
sgtitle('Positively tuned neuron pairs: Noise and Signal correlations')


%% Distribution of signal - noise correlations for positively tuned neurons in each behavior
idInd = idM56;
edges = -.8 : .05 : .8;
binCenters = (edges(1:end-1) + edges(2:end)) / 2;

% Get monitor positions and size
% monitorPositions = get(0, 'MonitorPositions');
% secondMonitorPosition = monitorPositions(size(monitorPositions, 1), :); % Just use single monitor if you don't have second one
% Create a maximized figure on the second monitor
fig = figure(761); clf
% set(fig, 'Position', secondMonitorPosition);
nPlot = length(analyzeCodes);
[ax, pos] = tight_subplot(ceil(nPlot/4), ceil(nPlot/4), [.04 .02]);
colors = colors_for_behaviors(analyzeCodes);

for iBhv = 1 : length(analyzeCodes)

    % Which neurons are positively tuned for this behavior?
    iPosTuneInd = intersect(idInd, find(posMod(iBhv, :)));

    returnIdx = tril(true(length(iPosTuneInd)), -1);
    [row, col] = find(returnIdx);   % indices of each neuron in the pair

    % What is their signal correlation across behaviors?
    [sigRho,pval] = corr(meanSpikesZ(:, iPosTuneInd));
    iSigCorr = sigRho(returnIdx);

    % What is their noise correlation for this behavior?
    noiseRho = corr(spikesPerTrialZ{iBhv}(:, iPosTuneInd));
    iNoiseCorr = noiseRho(returnIdx);

    axes(ax(iBhv)); hold on

    N = histcounts(iSigCorr - iNoiseCorr, edges, 'Normalization', 'pdf');
    bar(binCenters, N, 'FaceColor', colors(iBhv,:), 'BarWidth', 1);

    ca = gca;
    % ca.YTickLabel = ca.YTick;
    ca.XTickLabel = ca.XTick;
    title(analyzeBhv{iBhv}, 'interpreter', 'none')
    if iBhv == 1
        legend({'Signal - Noise'})
    end
end
sgtitle('Positively tuned neuron pairs: Signal minus Noise correlations')




%% Distribution of signal and noise correlations for untuned neurons in each behavior
edges = -.8 : .05 : .8;
binCenters = (edges(1:end-1) + edges(2:end)) / 2;

% Get monitor positions and size
monitorPositions = get(0, 'MonitorPositions');
secondMonitorPosition = monitorPositions(size(monitorPositions, 1), :); % Just use single monitor if you don't have second one

% Create a maximized figure on the second monitor
fig = figure(761);
clf
set(fig, 'Position', secondMonitorPosition);
nPlot = length(analyzeCodes);
[ax, pos] = tight_subplot(ceil(nPlot/4), ceil(nPlot/4), [.04 .02]);
colors = colors_for_behaviors(analyzeCodes);

for iBhv = 1 : length(analyzeCodes)

    % Which neurons are not tuned for this behavior?
    iNotTuneInd = intersect(idInd, find(notMod(iBhv, :)));

    returnIdx = tril(true(length(iNotTuneInd)), -1);
    [row, col] = find(returnIdx);   % indices of each neuron in the pair

    % What is their signal correlation across behaviors?
    [sigRho,pval] = corr(meanSpikesZ(:, iNotTuneInd));
    iSigCorr = sigRho(returnIdx);

    % What is their noise correlation for this behavior?
    noiseRho = corr(spikesPerTrialZ{iBhv}(:, iNotTuneInd));
    iNoiseCorr = noiseRho(returnIdx);

    axes(ax(iBhv)); hold on

    N = histcounts(iNoiseCorr, edges, 'Normalization', 'pdf');
    bar(binCenters, N, 'FaceColor', colors(iBhv,:), 'BarWidth', 1);

    N = histcounts(iSigCorr, edges, 'Normalization', 'pdf');
    % bar(binCenters, N, 'FaceColor', [.5 1 .5], 'BarWidth', 1, 'FaceAlpha', .5);
    bar(binCenters, N, 'FaceColor', [.5 .5 .5], 'BarWidth', 1, 'FaceAlpha', .6);

    ca = gca;
    % ca.YTickLabel = ca.YTick;
    ca.XTickLabel = ca.XTick;
    title(analyzeBhv{iBhv}, 'interpreter', 'none')
    if iBhv == 1
        legend({'Noise', 'Signal'})
    end
end
sgtitle('Untuned neuron pairs: Noise and Signal correlations')

%% Distribution of signal and noise correlations for tuned vs. untuned neurons in each behavior
edges = -.7 : .05 : .7;
binCenters = (edges(1:end-1) + edges(2:end)) / 2;

% Get monitor positions and size
% monitorPositions = get(0, 'MonitorPositions');
% secondMonitorPosition = monitorPositions(size(monitorPositions, 1), :); % Just use single monitor if you don't have second one
% Create a maximized figure on the second monitor
fig = figure(762);
clf
% set(fig, 'Position', secondMonitorPosition);
nPlot = length(analyzeCodes);
[ax, pos] = tight_subplot(ceil(nPlot/4), ceil(nPlot/4), [.04 .02]);
colors = colors_for_behaviors(analyzeCodes);

for iBhv = 1 : length(analyzeCodes)

    % Which neurons are positively/not tuned for this behavior?
    iPosTuneInd = intersect(idInd, find(posMod(iBhv, :)));
    iNotTuneInd = intersect(idInd, find(notMod(iBhv, :)));
    iNegTuneInd = intersect(idInd, find(negMod(iBhv, :)));
    a = length(iPosTuneInd) + length(iNotTuneInd) + length(iNegTuneInd);

    % What is their signal correlation across behaviors?
    [sigRho,pval] = corr(meanSpikesZ(:, iPosTuneInd), meanSpikesZ(:, iNotTuneInd));
    iSigCorr = sigRho(:);

    % What is their noise correlation for this behavior?
    noiseRho = corr(spikesPerTrialZ{iBhv}(:, iPosTuneInd), spikesPerTrialZ{iBhv}(:, iNotTuneInd));
    iNoiseCorr = noiseRho(:);

    axes(ax(iBhv)); hold on

    N = histcounts(iNoiseCorr, edges, 'Normalization', 'pdf');
    bar(binCenters, N, 'FaceColor', colors(iBhv,:), 'BarWidth', 1);

    N = histcounts(iSigCorr, edges, 'Normalization', 'pdf');
    bar(binCenters, N, 'FaceColor', [.5 .5 .5], 'BarWidth', 1, 'FaceAlpha', .6);

    ca = gca;
    % ca.YTickLabel = ca.YTick;
    ca.XTickLabel = ca.XTick;
    title(analyzeBhv{iBhv}, 'interpreter', 'none')
    if iBhv == 1
        legend({'Noise', 'Signal'})
    end
end
sgtitle('Tuned vs Untuned neuron pairs: Noise and Signal correlations')















%% Regressions for the correlations on all the pair-wise behaviors (
figure(54);
clf
hold on
for iBhv = 1 : length(analyzeCodes)-1
    for jBhv = 2 : length(analyzeCodes)
        if iBhv == jBhv
            continue
        end
        x = noiseCorrM56Pair(:,iBhv);
        y = noiseCorrM56Pair(:,jBhv);
        scatter(x,y)
        x = [ones(length(x), 1), x];
        b = x\y;

        % Define regression line function
        regressionLine = @(u) b(1) + b(2)*u;

        % Plot regression line
        fplot(regressionLine, [min(x(:,2)), max(x(:,2))]);

        % Calculate correlation coefficient
        r = corrcoef(x(:,2), y);
        r = r(1, 2);

    end
end

% [p,tbl,stats] = anova1(noiseCorrM56Pair(1,:))

%%
% imagesc(noiseCorrM56Pair)
figure(71);
clf
hold on;
clearAx = 1;
for iPair = 1 : size(noiseCorrM56Pair, 1)
    if clearAx == 5
        cla
        clearAx = 1;
    end
    yline(0)
    plot(noiseCorrM56Pair(iPair,:), 'linewidth', 2)
    [h, p] = ttest(noiseCorrM56Pair(iPair,:))
    clearAx = clearAx + 1;
end











%%  Noise Correlations for sequences vs. all behaviors (going into behavior X, coming from behavior Y or from all different behaviors)
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%    For each behavior, what are the most common preceding behaviors?

[seqStartTimes, seqCodes, seqNames] = behavior_sequences(dataBhv, analyzeCodes, analyzeBhv);

% Use all sequences with at least 40 trials
nTrial = cell2mat(cellfun(@(x) size(x, 1), seqStartTimes, 'UniformOutput', false));
[~, i] = sort(nTrial, 'descend');
nTrial = nTrial(i);
over40 = find(nTrial >= 40, 1, 'last');


%%

periEventTime = -.2 : opts.frameSize : .2; % seconds around onset
dataWindow = periEventTime(1:end-1) / opts.frameSize; % frames around onset (remove last frame)
edges = -1 : .02 : 1;
binCenters = (edges(1:end-1) + edges(2:end)) / 2;

for seq = 1 : over40
    seqStr = strsplit(seqNames{seq});
    bhvCurr = analyzeCodes(strcmp(analyzeBhv, seqStr{3}));
    bhvPrev = analyzeCodes(strcmp(analyzeBhv, seqStr{1}));

    goodStarts = dataBhvTruncate.ID == bhvCurr & validBhvTruncate(:, opts.bhvCodes == bhvCurr);
    allStarts = 1 + floor(dataBhvTruncate.StartTime(goodStarts) ./ opts.frameSize);
    seqStarts = 1 + floor(seqStartTimes{seq} ./ opts.frameSize);

    % take random number-matched subsample of allStarts
    allStartsSub = allStarts(randperm(length(allStarts)));
    allStartsSub = allStartsSub(1:size(seqStarts, 1));

    nonseqStarts = setdiff(allStarts, seqStarts(:,2));
    if length(nonseqStarts) > size(seqStarts, 1)
        nonseqStarts = nonseqStarts(randperm(size(seqStarts, 1)));
    elseif length(nonseqStarts) < size(seqStarts, 1)
        subIdx = randperm(length(nonseqStarts));
        seqStarts = seqStarts(subIdx,:);
    end

    allSpikesCurr = zeros(length(allStarts), size(dataMat, 2));
    for i = 1 : length(allStarts)
        allSpikesCurr(i,:) = sum(dataMat(allStarts(i) + dataWindow, :));
    end

    nonSeqSpikesCurr = zeros(size(seqStarts, 1), size(dataMat, 2));
    allSpikesCurrSub = zeros(size(seqStarts, 1), size(dataMat, 2));
    seqSpikesCurr = zeros(size(seqStarts, 1), size(dataMat, 2));
    seqSpikesPrev = zeros(size(seqStarts, 1), size(dataMat, 2));
    for i = 1 : size(seqStarts, 1)
        allSpikesCurrSub(i,:) = sum(dataMat(allStartsSub(i, 1) + dataWindow, :)); % trial-matched subset
        nonSeqSpikesCurr(i,:) = sum(dataMat(nonseqStarts(i, 1) + dataWindow, :)); % trial-matched subset
        seqSpikesCurr(i,:) = sum(dataMat(seqStarts(i, 2) + dataWindow, :));
        seqSpikesPrev(i,:) = sum(dataMat(seqStarts(i, 1) + dataWindow, :));
    end

    % Within area correlations
    % ------------------------
    % M56
    returnIdx = tril(true(length(idM56)), -1);
    rho = corr(allSpikesCurr(:, idM56));
    allM56Corr = rho(returnIdx);
    rho = corr(nonSeqSpikesCurr(:, idM56));  % trial-matched non-sequence behaviors
    nonSeqM56Corr = rho(returnIdx);
    rho = corr(seqSpikesCurr(:, idM56));
    seqM56CorrCurr = rho(returnIdx);
    rho = corr(seqSpikesPrev(:, idM56));
    seqM56CorrPrev = rho(returnIdx);

    % Test whether distributions are different
    [p,h,~] = ranksum(allM56Corr, seqM56CorrCurr)

    % Plot the distributions
    subplot(1,3,1)
    cla
    % N = histcounts(allM56Corr, edges, 'Normalization', 'pdf');
    N = histcounts(nonSeqM56Corr, edges, 'Normalization', 'pdf');
    bar(binCenters, N, 'b', 'FaceAlpha', .5);%, 'hist')
    hold on
    xlim([-.5 .5])
    N = histcounts(seqM56CorrCurr, edges, 'Normalization', 'pdf');
    bar(binCenters, N, 'r', 'FaceAlpha', .5);%, 'hist')
    yl = ylim;
    plot([median(allM56Corr) median(allM56Corr)], [.9*yl(2) yl(2)], 'b', 'linewidth', 4)
    plot([median(seqM56CorrCurr) median(seqM56CorrCurr)], [.9*yl(2) yl(2)], 'r', 'linewidth', 4)
    title(['M56 correlations: ', sequenceNames{seq}], 'interpreter', 'none')

    % ------------------------
    % DS
    returnIdx = tril(true(length(idDS)), -1);
    rho = corr(allSpikesCurr(:, idDS));
    allDSCorr = rho(returnIdx);
    rho = corr(nonSeqSpikesCurr(:, idDS));  % trial-matched non-sequence behaviors
    nonSeqDSCorr = rho(returnIdx);
    rho = corr(seqSpikesCurr(:, idDS));
    seqDSCorrCurr = rho(returnIdx);
    rho = corr(seqSpikesPrev(:, idDS));
    seqDSCorrPrev = rho(returnIdx);

    % Test whether distributions are different
    [p,h,~] = ranksum(allDSCorr, seqDSCorrCurr)

    % Plot the distributions
    subplot(1,3,2)
    cla
    % N = histcounts(allDSCorr, edges, 'Normalization', 'pdf');
    N = histcounts(nonSeqDSCorr, edges, 'Normalization', 'pdf');
    bar(binCenters, N, 'b', 'FaceAlpha', .5);%, 'hist')
    hold on
    xlim([-.5 .5])
    N = histcounts(seqDSCorrCurr, edges, 'Normalization', 'pdf');
    bar(binCenters, N, 'r', 'FaceAlpha', .5);%, 'hist')
    yl = ylim;
    plot([median(allDSCorr) median(allDSCorr)], [.9*yl(2) yl(2)], 'b', 'linewidth', 4)
    plot([median(seqDSCorrCurr) median(seqDSCorrCurr)], [.9*yl(2) yl(2)], 'r', 'linewidth', 4)
    title(['DS correlations: ', sequenceNames{seq}], 'interpreter', 'none')



    % Across area correlations
    % ------------------------
    [allM56DSCorr, ~] = corr(allSpikesCurr(:, idM56), allSpikesCurr(:, idDS));
    % trial-matched behaviors subsample
    [allM56DSCorrSub, ~] = corr(allSpikesCurrSub(:, idM56), allSpikesCurrSub(:, idDS));
    % trial-matched non-sequence behaviors subsample
    [nonSeqM56DSCorr, ~] = corr(nonSeqSpikesCurr(:, idM56), nonSeqSpikesCurr(:, idDS));
    [seqM56DSCorrCurr, ~] = corr(seqSpikesCurr(:, idM56), seqSpikesCurr(:, idDS));
    [seqM56DSCorrPrev, ~] = corr(seqSpikesPrev(:, idM56), seqSpikesPrev(:, idDS));

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
    subplot(1,3,3)
    cla
    % N = histcounts(allM56DSCorr(:), edges, 'Normalization', 'pdf');
    % trial-matched behaviors subsample
    % N = histcounts(allM56DSCorrSub(:), edges, 'Normalization', 'pdf'); %
    % trial-matched non-sequence behaviors subsample
    N = histcounts(nonSeqM56DSCorr(:), edges, 'Normalization', 'pdf');
    bar(binCenters, N, 'b', 'FaceAlpha', .5);%, 'hist')
    hold on
    xlim([-.5 .5])
    N = histcounts(seqM56DSCorrCurr(:), edges, 'Normalization', 'pdf');
    bar(binCenters, N, 'r', 'FaceAlpha', .5);%, 'hist')
    yl = ylim;
    plot([median(allM56DSCorr(:)) median(allM56DSCorr(:))], [.9*yl(2) yl(2)], 'b', 'linewidth', 4)
    plot([median(seqM56DSCorrCurr(:)) median(seqM56DSCorrCurr(:))], [.9*yl(2) yl(2)], 'r', 'linewidth', 4)
    title(['M56 - DS correlations: ', sequenceNames{seq}], 'interpreter', 'none')
end










































%%  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                           CCA Canonical Correlation Analysis
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
bhvCode = 15;

% Define the column subsets for CCA
subset1Columns = idM56; % Example subset 1
subset2Columns = idDS; % Example subset 2

% Extract subsets from neuralMatrix
subsetM56 = neuralMatrix(behaviorID == bhvCode, subset1Columns);
subsetDS = neuralMatrix(behaviorID == bhvCode, subset2Columns);
fitMat = neuralMatrix(behaviorID == bhvCode, :);

%%
% Fit CCA model on training data
[A, B, r, U, V] = canoncorr(subsetM56, subsetDS);

%% Train and test the model

[A, B, meanRSquared] = cca_with_crossvalidation(fitMat, subset1Columns, subset2Columns);


%%
% Remove the first and last bouts from the dataBhv table
dataBhvTrunc = dataBhv(2:end-1, :);
validBhvTrunc = validBhv(2:end-1,:);


periEventTime = -.3 : opts.frameSize : .5; % seconds around onset
dataWindow = round(periEventTime(1:end-1) / opts.frameSize); % frames around onset (remove last frame)

dataBhvInd = dataBhvTrunc.ID == bhvCode & validBhvTrunc(:, codes == bhvCode);
iStartFrames = 1 + floor(dataBhvTrunc.StartTime(dataBhvInd) ./ opts.frameSize);

for i = 1 : length(iStartFrames)
    neuralMatrix1 = dataMatZ(iStartFrames(i) + dataWindow, idM56);
    neuralMatrix2 = dataMatZ(iStartFrames(i) + dataWindow, idDS);
    plot_projected_data(neuralMatrix1, neuralMatrix2, A, B)

end
%%


% Perform Canonical Correlation Analysis
[A, B, r, U, V] = canoncorr(subsetM56, subsetDS);

% Display R-squared values
rSquared = r.^2;
disp('R-squared values:');
disp(rSquared);

% Project data onto the first 4 canonical variables
U_projected = U(:, 1:4);
V_projected = V(:, 1:4);

% Create 2x2 subplot using tight_subplot (or MATLAB's built-in subplot)
figure;
ha = tight_subplot(2, 2, [0.1 0.1], [0.1 0.1], [0.1 0.1]);

axes(ha(1));
hold on
% plot(U_projected(:, 1), V_projected(:, 1), 'o');
plot(U_projected(:, 1), 'b');
plot(V_projected(:, 1), 'k');
title('Canonical Variables 1');

axes(ha(2));
plot(U_projected(:, 2), V_projected(:, 2), 'o');
title('Canonical Variables 2');

axes(ha(3));
plot(U_projected(:, 3), V_projected(:, 3), 'o');
title('Canonical Variables 3');

axes(ha(4));
plot(U_projected(:, 4), V_projected(:, 4), 'o');
title('Canonical Variables 4');




function [A, B, meanRSquared] = cca_with_crossvalidation(neuralMatrix, subset1Columns, subset2Columns)
% Perform 5-fold cross-validation
n = size(neuralMatrix, 1);
cv = cvpartition(n, 'KFold', 5);
rSquaredValues = zeros(cv.NumTestSets, 4); % Store R^2 for first 4 canonical vars

for i = 1:cv.NumTestSets
    trainIdx = cv.training(i);
    testIdx = cv.test(i);

    % Training and Test subsets
    subset1Train = neuralMatrix(trainIdx, subset1Columns);
    subset2Train = neuralMatrix(trainIdx, subset2Columns);
    subset1Test = neuralMatrix(testIdx, subset1Columns);
    subset2Test = neuralMatrix(testIdx, subset2Columns);

    % Fit CCA model on training data
    [A, B, rT, U, V] = canoncorr(subset1Train, subset2Train);

    % Project test data onto the canonical variables
    U_test = subset1Test * A;
    V_test = subset2Test * B;

    % Compute R-squared for the first 4 canonical variables
    for j = 1:4
        r = corr(U_test(:, j), V_test(:, j))^2;
        rSquaredValues(i, j) = r;
    end
end

% Average R-squared values across folds
meanRSquared = mean(rSquaredValues, 1);

% Display mean R-squared values
disp('Mean R-squared values across folds:');
disp(meanRSquared);
end



function plot_projected_data(neuralMatrix1, neuralMatrix2, A, B)
% Project the data onto the first 4 canonical variables
U1_projected = neuralMatrix1 * A(:, 1:4);
U2_projected = neuralMatrix2 * B(:, 1:4);

% Create 2x2 subplot
figure;
ha = tight_subplot(2, 2, [0.1 0.1], [0.1 0.1], [0.1 0.1]);
time = 1:size(U1_projected, 1); % Assuming equal time points for both matrices

for i = 1:4
    axes(ha(i));
    plot(time, U1_projected(:, i), 'b', time, U2_projected(:, i), 'r');
    title(['Canonical Variable ' num2str(i)]);
    xlabel('Time');
    ylabel('Projection Value');
    legend('Matrix1', 'Matrix2');
end
end
