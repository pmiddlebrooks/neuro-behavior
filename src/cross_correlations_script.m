%% Get data from get_standard_data

opts = neuro_behavior_options;
opts.frameSize = .05; % 50 ms framesize for now
opts.collectFor = 60*60; % Get 45 min
opts.minActTime = .16;

get_standard_data
bhvID = double(bhvIDMat);
[dataBhv, bhvIDMat] = curate_behavior_labels(dataBhv, opts);


%% Which area has modulation first? Cross-correlation across behaviors
% For each behavior onset, determine the cross-correlation of mean spiking across neurons between M56 and DS
% plotCodes = codes(codes>-1);
% plotBhv = behaviors(codes>-1);


zTime = -3 : opts.frameSize : 2;  % zscore on a 5sec window peri-onset
zWindow = round(zTime(1:end-1) / opts.frameSize);
zStartInd = find(zTime == 0);
fullTime = -.5 : opts.frameSize : .5; % seconds around onset
fullWindow = round(fullTime(1:end-1) / opts.frameSize); % frames around onset w.r.t. zWindow (remove last frame)
maxLag = 30;

xCorrData = cell(iBhv, 1);
xCorrRand = xCorrData;


starts = find(diff(bhvID) ~= 0) + 1;
starts(starts < -zWindow(1)+1 | starts > length(bhvID) - zWindow(end)-1) = []; % Remove starts too close to begin/end of session
% Preallocate a cell array to store the xcorr results
xcorrData = cell(length(analyzeCodes), 1);
xcorrRand = cell(length(analyzeCodes), 1);
xcorrDataPop = cell(length(analyzeCodes), 1);
xcorrRandPop = cell(length(analyzeCodes), 1);

xcorrDataPopMean = zeros(maxLag*2+1, length(analyzeCodes));
xcorrDataPopStd = xcorrDataPopMean;
for iBhv = 1 : length(analyzeCodes)
tic

    % Get the starting indices of every bout for this behavior
    iStarts = intersect(starts, find(bhvID == analyzeCodes(iBhv)));

    % Initialize spike data matrices
    iM56Data = zeros(length(fullWindow), length(idM56), length(iStarts));
    iDSData = zeros(length(fullWindow), length(idDS), length(iStarts));
    % iM56Rand = iM56Data;
    % iDSRand = iDSData;
    % startRand = randperm(size(dataMat, 1)-length(zWindow) - 2) - zWindow(1) + 1;

    for j = 1 : length(iStarts)
        jZWindowM56 = zscore(dataMat(iStarts(j) + zWindow, idM56), [], 1);
        iM56Data(:, :, j) = jZWindowM56(zStartInd + fullWindow, :);
        jZWindowDS = zscore(dataMat(iStarts(j) + zWindow, idDS), [], 1);
        iDSData(:, :, j) = jZWindowDS(zStartInd + fullWindow, :);

        % Get randomized windows of spiking data from the two areas for
        % comparison
        % jZWindowM56Rand = zscore(dataMat(startRand(j) + zWindow, idM56), [], 1);
        % iM56Rand(:, :, j) = jZWindowM56Rand(zStartInd + fullWindow, :);
        % jZWindowDSRand = zscore(dataMat(startRand(j) + zWindow, idDS), [], 1);
        % iDSRand(:, :, j) = jZWindowDSRand(zStartInd + fullWindow, :);



        % Do xcorrs for pairwise single neuron responses between the areas

        % Loop over each pair of variables (columns)
        for x = 1:length(idM56)
            for y = 1:length(idDS)

                % Perform cross-correlation with maxLag
                [cData, lags] = xcorr(iM56Data(:, x, j), iDSData(:, y, j), maxLag, 'normalized');
                % [cRand, ~] = xcorr(iM56Rand(:, x, j), iDSRand(:, y, j), maxLag, 'normalized');

                % Store the result in the cell array
                xcorrData{iBhv}(:, x, y, j) = cData;
                % xcorrRand{iBhv}(:, x, y) = cRand;
            end
        end

        % Do xcorrs for mean population responses between the areas
        [xcorrDataPop{iBhv}(:, j), ~] = xcorr(mean(iM56Data(:,:,j), 2), mean(iDSData(:,:,j), 2), maxLag, 'normalized');
        % [xcorrRandPop{iBhv}(:, j), ~] = xcorr(mean(iM56Rand(:,:,j), 2), mean(iDSRand(:,:,j), 2), maxlag, 'normalized');


    end

    xcorrDataPopMean(:, iBhv) = mean(xcorrDataPop{iBhv}, 2);
    xcorrDataPopStd(:, iBhv) = std(xcorrDataPop{iBhv}, [], 2);
    % xcorrRandPopMean(:, iBhv) = mean(xcorrRandPop{iBhv}, 2);
    toc
    a = whos('xcorrData');
    a.bytes / 10e6
end
toc

%
fig = figure(88); clf
set(fig, 'Position', monitorTwo);
nPlot = length(analyzeCodes);
[ax, pos] = tight_subplot(2, ceil(nPlot/2), [.08 .02], .1);
% hold all;
colors = colors_for_behaviors(analyzeCodes);

for iBhv = 1 : length(analyzeCodes)
    axes(ax(iBhv)); % hold on;
    % plot(lags, cRand, '--', 'color', [.5 .5 .5], 'lineWidth', 2)
    % plot(lags, xCorrMeanRand(:, iBhv), '--', 'color', [.5 .5 .5], 'lineWidth', 2)
    % hold on;
    % plot(lags, c, 'color', colors(iBhv,:), 'lineWidth', 3)
    plot(lags*opts.frameSize, xcorrDataPopMean(:, iBhv), 'color', colors(iBhv,:), 'lineWidth', 3)
    % plot(lags, cM56, '--k', 'lineWidth', 2)
    % plot(lags, cDS, '--r', 'lineWidth', 2)
    xline(0)
    yline(0)

    title(analyzeBhv{iBhv}, 'interpreter', 'none');
end
sgtitle('M56 X DS Cross-correlations peri-behavior-transitions (z-scored)')

%
%
% % Get mean peri-event spiking data from the two areas for this behavior
% iPeriM56 = mean(eventMatZ{iBhv}(:, idM56, :), 2);
% iPeriDS = mean(eventMatZ{iBhv}(:, idDS, :), 2);
%
%
% % Get randomized windows of spiking data from the two areas for
% % comparison
% iPeriM56Rand = zeros(length(zWindow), length(idM56), size(eventMatZ{iBhv}, 3));
% iPeriDSRand = zeros(length(zWindow), length(idDS), size(eventMatZ{iBhv}, 3));
% % iPeriM56Rand = zeros(length(zWindow), length(idM56), nPermute);
% % iPeriDSRand = zeros(length(zWindow), length(idDS), nPermute);
% % Loop to extract n random windows
% % Randomly select the starting index of the window
% startIdx = randperm(size(dataMat, 1)-length(zWindow) - 2) - zWindow(1) + 1;
%
% for j = 1:size(eventMatZ{iBhv}, 3)
%     % for j = 1:nPermute
%
%     % Extract the window from dataMat
%     iPeriM56Rand(:, :, j) = dataMat(startIdx(j) + zWindow - 1, idM56);
%     iPeriDSRand(:, :, j) = dataMat(startIdx(j) + zWindow - 1, idDS);
%
% end
% iMeanM56Rand = mean(mean(iPeriM56Rand, 3), 1);
% iMeanDSRand = mean(mean(iPeriDSRand, 3), 1);
% iStdM56Rand = std(mean(iPeriM56Rand, 3), [], 1);
% iStdDSRand = std(mean(iPeriDSRand, 3), [], 1);
% iM56ZRand = (iPeriM56Rand - iMeanM56Rand) ./ iStdM56Rand;
% iDSZRand = (iPeriDSRand - iMeanDSRand) ./ iStdDSRand;
%
% iPeriM56RandZ = mean(iM56ZRand(zStartInd + fullWindow, :, :), 2);
% iPeriDSRandZ = mean(iDSZRand(zStartInd + fullWindow, :, :), 2);
%
%
%
% % For each bout, get xcorr of average population responses between
% % areas:
%
% for j = 1:size(eventMatZ{iBhv}, 3)
%     [xCorrData{iBhv}(:, j), lags] = xcorr(iPeriM56(:,:,j), iPeriDS(:,:,j), maxlag, 'normalized');
%     [xCorrRand{iBhv}(:, j), ~] = xcorr(iPeriM56RandZ(:,:,j), iPeriDSRandZ(:,:,j), maxlag, 'normalized');
% end
%
% xCorrMeanData(:, iBhv) = mean(xCorrData{iBhv}, 2);
% xCorrMeanRand(:, iBhv) = mean(xCorrRand{iBhv}, 2);
%
% % [c, lags] = xcorr(iPeriM56, iPeriDS, maxlag, 'normalized');
% % [cRand, ~] = xcorr(iPeriM56RandZ, iPeriDSRandZ, maxlag, 'normalized');
% % R = corrcoef(iPeriM56, iPeriDS);
% % RRand = corrcoef(iPeriM56RandZ, iPeriDSRandZ);


% title(num2str(corr(mean(dataMat(plotFrames, idM56), 2), mean(dataMat(plotFrames, idDS), 2))))
% end

% %% Across some time, regardless of aligning to transitions
% randWindow = randi([50 70000]) + fullWindow;
% firstFrames = find(diff(bhvIDMat) ~= 0) + 1; % 1 frame prior to all behavior transitions
% preIndID = bhvIDMat(firstFrames);
% 
% plotFrames = firstFrames(19) + fullWindow;
% % plotFrames = 40001:44350;
% meanM56 = mean(dataMat(plotFrames, idM56), 2);
% meanDS = mean(dataMat(plotFrames, idDS), 2);
% meanM56Z = zscore(mean(dataMat(plotFrames, idM56), 2));
% meanDSZ = zscore(mean(dataMat(plotFrames, idDS), 2));
% % plot(meanM56);
% % hold on
% % plot(meanDS, 'r');
% % [c, lags] = xcorr(meanM56, meanDS, 50, 'normalized');
% [cZ, lags] = xcorr(meanM56Z, meanDSZ, 30, 'normalized');
% % [c, lags] = xcorr(meanM56, meanDS, 50);
% % [c, lags] = xcorr(meanM56, meanDS(randperm(length(meanDS))), 50);
% % R = corrcoef(meanM56, meanDS)
% % [c, lags] = xcorr(meanM56, meanDS, 50, 'normalized');
% % R = corrcoef(meanM56, meanDS);
% figure(89); clf; hold on;
% % plot(lags, c, 'linewidth', 2)
% plot(lags, cZ, '--b', 'linewidth', 2)
% xline(0)
% 
