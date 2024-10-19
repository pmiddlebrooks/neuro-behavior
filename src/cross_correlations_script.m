%% Get data from get_standard_data

opts = neuro_behavior_options;
opts.frameSize = .05; % 50 ms framesize for now
opts.collectFor = 60*60; % Get 45 min
opts.minActTime = .16;

get_standard_data



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
maxlag = 25;

    xCorrData = cell(iBhv, 1);
    xCorrRand = xCorrData;

for iBhv = 1 : length(analyzeCodes)
    
    % Get mean peri-event spiking data from the two areas for this behavior
    iPeriM56 = mean(eventMatZ{iBhv}(:, idM56, :), 2);
    iPeriDS = mean(eventMatZ{iBhv}(:, idDS, :), 2);


    % Get randomized windows of spiking data from the two areas for
    % comparison
    iPeriM56Rand = zeros(length(zWindow), length(idM56), size(eventMatZ{iBhv}, 3));
    iPeriDSRand = zeros(length(zWindow), length(idDS), size(eventMatZ{iBhv}, 3));
    % iPeriM56Rand = zeros(length(zWindow), length(idM56), nPermute);
    % iPeriDSRand = zeros(length(zWindow), length(idDS), nPermute);
       % Loop to extract n random windows
        % Randomly select the starting index of the window
    startIdx = randperm(size(dataMat, 1)-length(zWindow) - 2) - zWindow(1) + 1;

    for j = 1:size(eventMatZ{iBhv}, 3)
    % for j = 1:nPermute

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

    iPeriM56RandZ = mean(iM56ZRand(zStartInd + fullWindow, :, :), 2);
    iPeriDSRandZ = mean(iDSZRand(zStartInd + fullWindow, :, :), 2);



    % For each bout, get xcorr of average population responses between
    % areas:

    for j = 1:size(eventMatZ{iBhv}, 3)
        [xCorrData{iBhv}(:, j), lags] = xcorr(iPeriM56(:,:,j), iPeriDS(:,:,j), maxlag, 'normalized');
        [xCorrRand{iBhv}(:, j), ~] = xcorr(iPeriM56RandZ(:,:,j), iPeriDSRandZ(:,:,j), maxlag, 'normalized');
    end

    xCorrMeanData(:, iBhv) = mean(xCorrData{iBhv}, 2);
    xCorrMeanRand(:, iBhv) = mean(xCorrRand{iBhv}, 2);
    
    % [c, lags] = xcorr(iPeriM56, iPeriDS, maxlag, 'normalized');
    % [cRand, ~] = xcorr(iPeriM56RandZ, iPeriDSRandZ, maxlag, 'normalized');
    % R = corrcoef(iPeriM56, iPeriDS);
    % RRand = corrcoef(iPeriM56RandZ, iPeriDSRandZ);

    axes(ax(iBhv)); % hold on;
    % plot(lags, cRand, '--', 'color', [.5 .5 .5], 'lineWidth', 2)
    plot(lags, xCorrMeanRand(:, iBhv), '--', 'color', [.5 .5 .5], 'lineWidth', 2)
    hold on;
    % plot(lags, c, 'color', colors(iBhv,:), 'lineWidth', 3)
    plot(lags, xCorrMeanData(:, iBhv), 'color', colors(iBhv,:), 'lineWidth', 3)
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

