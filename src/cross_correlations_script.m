%% Get data from get_standard_data

opts = neuro_behavior_options;
opts.frameSize = .02; % 50 ms framesize for now
opts.collectFor = 60*60; % Get 45 min
opts.minActTime = .16;

getDataType = 'all';

get_standard_data
bhvID = double(bhvIDMat);
[dataBhv, bhvIDMat] = curate_behavior_labels(dataBhv, opts);


monitorPositions = get(0, 'MonitorPositions');
if exist('/Users/paulmiddlebrooks/Projects/', 'dir')
    monitorPositions = flipud(monitorPositions);
end
monitorOne = monitorPositions(1, :); % Just use single monitor if you don't have second one
monitorTwo = monitorPositions(size(monitorPositions, 1), :); % Just use single monitor if you don't have second one

%% Which area has modulation first? Cross-correlation across behaviors
% For each behavior onset, determine the cross-correlation of mean spiking across neurons between M56 and DS
% plotCodes = codes(codes>-1);
% plotBhv = behaviors(codes>-1);


zTime = -3 : opts.frameSize : 2;  % zscore on a 5sec window peri-onset
zWindow = round(zTime(1:end-1) / opts.frameSize);
zStartInd = find(zTime == 0);
fullTime = -1 : opts.frameSize : 1; % seconds around onset
fullWindow = round(fullTime(1:end-1) / opts.frameSize); % frames around onset w.r.t. zWindow (remove last frame)


maxLag = min(50, length(fullWindow)-1);
% maxLag = round(length(fullWindow)*1.5);
% Once xcorr is performed, what window do you want to plot/analyze?
if opts.frameSize >= .05
xcorrTime = -.5 : opts.frameSize : .5;
else
xcorrTime = -.3 : opts.frameSize : .3;
end
    xcorrWindow = round(xcorrTime / opts.frameSize);
cDataZeroInd = (maxLag * 2 + 2)/2; % When xcorr is performed, find index in cData where lag = 0


starts = find(diff(bhvID) ~= 0) + 1;
starts(starts < -zWindow(1)+1 | starts > length(bhvID) - zWindow(end)-1) = []; % Remove starts too close to begin/end of session
% Preallocate a cell array to store the xcorr results
spikeDataM56 = cell(length(analyzeCodes), 1);
spikeDataDS = cell(length(analyzeCodes), 1);
xcorrData = cell(length(analyzeCodes), 1);
% xcorrRand = cell(length(analyzeCodes), 1);
xcorrDataPop = cell(length(analyzeCodes), 1);
% xcorrRandPop = cell(length(analyzeCodes), 1);

xcorrDataMeanPsth = cell(length(analyzeCodes), 1);

xcorrDataPopMean = zeros(length(xcorrWindow), length(analyzeCodes));
xcorrDataPopStd = xcorrDataPopMean;
xcorrDataMeanPsthMean = xcorrDataPopMean;


for iBhv = 1 : length(analyzeCodes)
% for iBhv = length(analyzeCodes)
    fprintf('\n%s\n  ', analyzeBhv{iBhv})
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
        spikeDataM56{iBhv}(:, :, j) = jZWindowM56(zStartInd + fullWindow, :);
        jZWindowDS = zscore(dataMat(iStarts(j) + zWindow, idDS), [], 1);
        iDSData(:, :, j) = jZWindowDS(zStartInd + fullWindow, :);
        spikeDataDS{iBhv}(:, :, j) = jZWindowDS(zStartInd + fullWindow, :);

        % Get randomized windows of spiking data from the two areas for
        % comparison
        % jZWindowM56Rand = zscore(dataMat(startRand(j) + zWindow, idM56), [], 1);
        % iM56Rand(:, :, j) = jZWindowM56Rand(zStartInd + fullWindow, :);
        % jZWindowDSRand = zscore(dataMat(startRand(j) + zWindow, idDS), [], 1);
        % iDSRand(:, :, j) = jZWindowDSRand(zStartInd + fullWindow, :);



        % Do xcorrs for pairwise single neuron responses between the areas

        % Loop over each pair of neurons for each bout
        for x = 1:length(idM56)
            for y = 1:length(idDS)

                % Only xcorr if there are any spikes
                if sum(iM56Data(:, x, j)) || sum(iDSData(:, x, j))
                    % Perform cross-correlation with maxLag
                    [cData, lags] = xcorr(iM56Data(:, x, j), iDSData(:, y, j), maxLag, 'normalized');
                    % [cData, lags] = xcorr(iM56Data(:, x, j), iDSData(:, y, j), maxLag);
                    % [cRand, ~] = xcorr(iM56Rand(:, x, j), iDSRand(:, y, j), maxLag, 'normalized');

                    % Store the result in the cell array
                    xcorrData{iBhv}(:, x, y, j) = cData(cDataZeroInd + xcorrWindow);
                    % xcorrRand{iBhv}(:, x, y) = cRand;

                else
                    % Lots of individual bouts won't have any spikes for lots
                    % of individual neurons. If that's the case, return nans
                    % for the xcorr:
                    xcorrData{iBhv}(:, x, y, j) = nan(length(xcorrWindow), 1);
                end
            end
        end

        % Do xcorrs for mean population responses between the areas (each
        % bout averaged across neurons)
        [cData, ~] = xcorr(mean(iM56Data(:,:,j), 2), mean(iDSData(:,:,j), 2), maxLag, 'normalized');
        % [cData, ~] = xcorr(mean(iM56Data(:,:,j), 2), mean(iDSData(:,:,j), 2), maxLag);
        xcorrDataPop{iBhv}(:, j) = cData(cDataZeroInd + xcorrWindow);
        % [xcorrRandPop{iBhv}(:, j), ~] = xcorr(mean(iM56Rand(:,:,j), 2), mean(iDSRand(:,:,j), 2), maxlag, 'normalized');

    end

        xcorrDataPopMean(:, iBhv) = mean(xcorrDataPop{iBhv}, 2);
    xcorrDataPopStd(:, iBhv) = std(xcorrDataPop{iBhv}, [], 2);
    % xcorrRandPopMean(:, iBhv) = mean(xcorrRandPop{iBhv}, 2);

    % Do xcorrs for mean neuron psths (each neuron averaged across bouts)
    for x = 1:length(idM56)
        for y = 1:length(idDS)
            % xcorr the mean psths between all neurons
            [cData, lags] = xcorr(mean(spikeDataM56{iBhv}(:, x, :), 3), mean(spikeDataDS{iBhv}(:, y, :), 3), ...
                maxLag, 'normalized');
            % [cData, lags] = xcorr(mean(spikeDataM56{iBhv}(:, x, :), 3), mean(spikeDataDS{iBhv}(:, y, :), 3), ...
            %     maxLag);
            xcorrDataMeanPsth{iBhv}(:, x, y) = cData(cDataZeroInd + xcorrWindow);

        end
    end

    xcorrDataMeanPsthMean(:, iBhv) = mean(mean(xcorrDataMeanPsth{iBhv}, 2), 3);

    toc
    a = whos('xcorrData');
    a.bytes / 10e6
end
toc








%%
xcorrDataPopMean10 = xcorrDataPopMean;

%%
xcorrDataPopMean = xcorrDataPopMean01;
%% Plot mean of trial-wise xcorrs between averaged populations psths in each area
fig = figure(89); clf
set(fig, 'Position', monitorTwo);
nPlot = length(analyzeCodes);
[ax, pos] = tight_subplot(2, ceil(nPlot/2), [.08 .02], .1);
% hold all;
colors = colors_for_behaviors(analyzeCodes);
plotSec = xcorrTime(end) + .001;
lagSec = xcorrWindow * opts.frameSize;
plotWindowIdx = find(lagSec > -plotSec & lagSec < plotSec);
ymax = max(xcorrDataPopMean(:));
ymin = min(xcorrDataPopMean(:));

for iBhv = 1 : length(analyzeCodes)
    axes(ax(iBhv)); % hold on;
    % plot(lags, cRand, '--', 'color', [.5 .5 .5], 'lineWidth', 2)
    % plot(lags, xCorrMeanRand(:, iBhv), '--', 'color', [.5 .5 .5], 'lineWidth', 2)
    % hold on;
    % plot(lags, c, 'color', colors(iBhv,:), 'lineWidth', 3)

    plot(xcorrWindow(plotWindowIdx)*opts.frameSize, xcorrDataPopMean(plotWindowIdx, iBhv), 'color', colors(iBhv,:), 'lineWidth', 3)
    % plot(lags, cM56, '--k', 'lineWidth', 2)
    % plot(lags, cDS, '--r', 'lineWidth', 2)
    xline(0)
    yline(0)
    ylim([ymin ymax])
    xlim([-plotSec plotSec]);
    title(analyzeBhv{iBhv}, 'interpreter', 'none');
end
figTitle = sprintf('XCORR (M56 X DS) behavior-transitions, Pop avg across bouts, frame=%.2f', opts.frameSize);
sgtitle(figTitle)
set(gcf, 'PaperOrientation', 'landscape');
print('-dpdf', fullfile(paths.figurePath, [figTitle, '.pdf']), '-bestfit')


%% Plot mean xcorrs of pairwise neuron average psths
fig = figure(98); clf
set(fig, 'Position', monitorTwo);
nPlot = length(analyzeCodes);
[ax, pos] = tight_subplot(2, ceil(nPlot/2), [.08 .02], .1);
% hold all;
colors = colors_for_behaviors(analyzeCodes);
plotSec = xcorrTime(end) + .001;
lagSec = xcorrWindow * opts.frameSize;
plotWindowIdx = find(lagSec > -plotSec & lagSec < plotSec);
ymax = max(xcorrDataMeanPsthMean(:));
ymin = min(xcorrDataMeanPsthMean(:));


for iBhv = 1 : length(analyzeCodes)
    axes(ax(iBhv)); % hold on;
    % plot(lags, cRand, '--', 'color', [.5 .5 .5], 'lineWidth', 2)
    % plot(lags, xCorrMeanRand(:, iBhv), '--', 'color', [.5 .5 .5], 'lineWidth', 2)
    % hold on;
    % plot(lags, c, 'color', colors(iBhv,:), 'lineWidth', 3)

    plot(xcorrWindow(plotWindowIdx)*opts.frameSize, xcorrDataMeanPsthMean(plotWindowIdx, iBhv), 'color', colors(iBhv,:), 'lineWidth', 3)
    % plot(lags, cM56, '--k', 'lineWidth', 2)
    % plot(lags, cDS, '--r', 'lineWidth', 2)
    xline(0)
    yline(0)
    ylim([ymin ymax])
    xlim([-plotSec plotSec]);
    title(analyzeBhv{iBhv}, 'interpreter', 'none');
end
figTitle = sprintf('XCORR (M56 X DS) behavior-transitions, PSTH avg across bouts, frame=%.2f', opts.frameSize);
sgtitle(figTitle)
set(gcf, 'PaperOrientation', 'landscape');
print('-dpdf', fullfile(paths.figurePath, [figTitle, '.pdf']), '-bestfit')















%% Non-negative matrix factorization to find common distinct patterns of xcorrs among pairs across individual bouts
warning('currently only doing nnmf for locomotion- test on others as well')
% resize xcorrData so every column is a pair of neurons, every row is a lag
% of the xcorr
bhvIdx = 7;

[nLag, nM56, nDS, nBout] = size(xcorrData{bhvIdx});
% Reshape the matrix to (n, x*y*j)
nnmfData = reshape(xcorrData{bhvIdx}, nLag, nM56 * nDS * nBout);
% Only do nnmf on bouts with spiking data in both neuron's of the pair
zeroBoutPair = isnan(nnmfData(1,:));
nnmfData = nnmfData(:, ~zeroBoutPair);

% Keep track of original indices
% Create index arrays for x, y, and j indices for each element
[originalM56, originalDS, originalBout] = ndgrid(1:nM56, 1:nDS, 1:nBout);

% Reshape each array to a 1D vector and combine into a 3 x numElements matrix
indexMatrix = [originalM56(:)'; originalDS(:)'; originalBout(:)'];
indexMatrix = indexMatrix(:, ~zeroBoutPair);

% Ensure no negative values in the pattern.
minXcorr = min(xcorrData{bhvIdx}(:));
nnmfData = nnmfData + abs(minXcorr);

% Define the range of rank values to test
rankRange = 1:8;
W = cell(length(rankRange), 1);
H = W;
reconstructionErrors = zeros(size(rankRange));  % Store reconstruction errors


for iRank = 1:length(rankRange)
    % Current rank
    rank = rankRange(iRank);

    % Perform NNMF with the current rank
    [iW, iH] = nnmf(nnmfData, rank);
    W{iRank} = iW;
    H{iRank} = iH;
    % Calculate the reconstruction error (Frobenius norm)
    reconstructedData = W{iRank} * H{iRank};
    reconstructionErrors(iRank) = norm(nnmfData - reconstructedData, 'fro');
end

%% Plot reconstruction error across ranks
figure(112);
plot(rankRange, reconstructionErrors, '-o', 'lineWidth', 2);
xlabel('Rank');
ylabel('Reconstruction Error (Frobenius Norm)');
title('Reconstruction Error vs. Rank');

% Finding the elbow point
[~, elbowIdx] = min(diff(reconstructionErrors));
elbowRank = rankRange(elbowIdx);
disp(['Suggested Rank (Elbow Point): ', num2str(elbowRank)]);


%% Plot basis functions for each rank
fig = figure(111); clf
nPlot = 8; %length(rankRange);
fun = @sRGB_to_OKLab;
colors = maxdistcolor(nPlot, fun);
[ax, pos] = tight_subplot(2, ceil(nPlot/2), [.08 .02], .1);
for i = 1:nPlot
    % Current rank
    rank = rankRange(i);

    % Plot each basis function (each column in W) in a single plot
    % subplot(1, length(rankRange), i);  % Create subplot for each rank
axes(ax(i));
% Apply colors to each line using set function
set(gca, 'ColorOrder', colors(1:i,:), 'NextPlot', 'replacechildren');

plot(xcorrWindow, W{i}, 'linewidth', 2);  % Plot columns of W to visualize basis functions
    title(['Rank = ', num2str(rank)]);
    xlabel('Lags (frames)');
    ylabel('Basis Function Value');
    xline(0)
    legend
    % pause(1);  % Pause for inspection before moving to the next rank
end
titleN = sprintf('NNMF Basis Functions- %s', analyzeBhv{bhvIdx});
sgtitle(titleN, 'interpreter', 'none')
    print('-dpdf', fullfile(paths.figurePath, [titleN, '.pdf']), '-bestfit')




    %% Which basis functions dominate for each pair of neurons? 
rankIdxToTest = 4;

% Initialize cell array to store percentage dominance for each (originalM56, originalDS) pair
percentageDominance = cell(nM56, nDS);
domBasisIdx = zeros(nM56, nDS);
tic
% Loop through each unique (originalM56, originalDS) pair
for m = 1:nM56
    for d = 1:nDS
        % Get indices in indexMatrix where originalM56 == m and originalDS == d
        relevantIndices = find(indexMatrix(1, :) == m & indexMatrix(2, :) == d);
        
        % Extract the H values for this (m, d) pair across all originalBouts
        H_subset = H{rankIdxToTest}(:, relevantIndices);
        
        % Determine the dominant basis function for each Bout
        [~, dominantBasisForBouts] = max(H_subset, [], 1);
        
        % Calculate the percentage dominance of each basis function within this pair
        counts = histcounts(dominantBasisForBouts, 1:(size(H_subset, 1)+1));
        [~, domBasisIdx(m, d)] = max(counts);
        percentageDominance{m, d} = (counts / length(dominantBasisForBouts)) * 100;
    end
end
toc

% Which basis functions are most prevelantly dominant among the pairs of neurons, in terms of 
figure(219);
histogram(domBasisIdx(:))
xlabel('Basis Function')
ylabel('Counts of neuron pairs')
title('Dominant basis functions per neuron pairs across bouts')



% How well-separated are the basis functions in terms of how much each contributes to the pairs of neurons?

separationRatios = zeros(size(percentageDominance));

for i = 1:numel(percentageDominance)
    sortedValues = sort(percentageDominance{i}, 'descend');  % Sort in descending order
    if numel(sortedValues) >= 2
        separationRatios(i) = sortedValues(1) / sortedValues(2);  % Ratio of highest to next-highest
    else
        separationRatios(i) = NaN;  % Handle cases with fewer values if applicable
    end
end
figure(216); 
% histogram(separationRatios, [1 : .02 : 3])
histogram(separationRatios(:))
title('NNMF Basis functions separation ratios per neuron pair')
xlabel('Separation ratio')
ylabel('Counts')











%% Non-negative matrix factorization to find common distinct patterns of xcorrs among pairs averaged across bouts
% warning('currently only doing nnmf for locomotion- test on others as well')

bhvIdx = 14;
[nLag, nM56, nDS, nBout] = size(xcorrData{bhvIdx});

        xcorrPairs = nanmean(xcorrData{bhvIdx}, 4);
        % xcorrPairs = mean(xcorrData{bhvIdx}, 4);

%
[nLag, nM56, nDS] = size(xcorrPairs);

% Reshape the matrix to (n, x*y*j)
nnmfData = reshape(xcorrPairs, nLag, nM56 * nDS);
% Only do nnmf on bouts with spiking data in both neuron's of the pair
% zeroBoutPair = isnan(nnmfData(1,:));
% nnmfData = nnmfData(:, ~zeroBoutPair);

% Keep track of original indices
% Create index arrays for x, y, and j indices for each element
[originalM56, originalDS] = ndgrid(1:nM56, 1:nDS);

% Reshape each array to a 1D vector and combine into a 3 x numElements matrix
indexMatrix = [originalM56(:)'; originalDS(:)'];
indexMatrix = indexMatrix(:, ~zeroBoutPair);

% Ensure no negative values in the pattern.
minXcorr = min(nnmfData(:));
nnmfData = nnmfData + abs(minXcorr);

% Define the range of rank values to test
rankRange = 3:6;
W = cell(length(rankRange), 1);
H = W;
reconstructionErrors = zeros(size(rankRange));  % Store reconstruction errors


for iRank = 1:length(rankRange)
    % Current rank
    rank = rankRange(iRank);

    % Perform NNMF with the current rank
    [iW, iH] = nnmf(nnmfData, rank);
    W{iRank} = iW;
    H{iRank} = iH;
    % Calculate the reconstruction error (Frobenius norm)
    reconstructedData = W{iRank} * H{iRank};
    reconstructionErrors(iRank) = norm(nnmfData - reconstructedData, 'fro');
end

%%                      TESTS FOR WHETHER THE CORRELATION FUNCTIONS ARE STRUCTURED OR FLAT

%% Use magnitude threshold

t = max(abs(xcorrPairs), [], 1);
t = reshape(t, nM56, nDS);
figure(); imagesc(t)
colorbar

%% Plot variance of the pair-wise averaged correlation functions

% figure(722);
t = var(xcorrPairs, 1);
t = reshape(t, nM56, nDS);
figure(722); imagesc(t)
% reshape nnmf
colorbar
% for i = 1 : 1000
%      clf; 
% plot(nnmfData(:,i))
% hold on; yline(mean(nnmfData(:,i)))
% hold on; yline(mean(nnmfData(:,i)) + 2*std(nnmfData(:,i)))
% hold on; yline(mean(nnmfData(:,i)) - 2*std(nnmfData(:,i)))
% end

%% Plot autocorrelation of the pairwise averaged correlation functions

autoCorrPairs = zeros(nM56, nDS);
for m = 1 : nM56
    for d = 1 : nDS

 acf = autocorr(xcorrPairs(:,m,d));
 autoCorrPairs(m, d) = mean(abs(acf));
    end
end
figure(); imagesc(autoCorrPairs)
colorbar

%%     Max-min range

t = max(xcorrPairs, [], 1) - min(xcorrPairs, [], 1);
t = reshape(t, nM56, nDS);
figure(); imagesc(t)
% reshape nnmf
colorbar






%% Plot reconstruction error across ranks
figure(112);
plot(rankRange, reconstructionErrors, '-o', 'lineWidth', 2);
xlabel('Rank');
ylabel('Reconstruction Error (Frobenius Norm)');
title('Reconstruction Error vs. Rank');

% Finding the elbow point
[~, elbowIdx] = min(diff(reconstructionErrors));
elbowRank = rankRange(elbowIdx);
disp(['Suggested Rank (Elbow Point): ', num2str(elbowRank)]);


%% Plot basis functions for each rank
fig = figure(111); clf
nPlot = 8; %length(rankRange);
fun = @sRGB_to_OKLab;
colors = maxdistcolor(nPlot, fun);
[ax, pos] = tight_subplot(2, ceil(nPlot/2), [.08 .02], .1);
for i = 1:nPlot
    % Current rank
    rank = rankRange(i);

    % Plot each basis function (each column in W) in a single plot
    % subplot(1, length(rankRange), i);  % Create subplot for each rank
axes(ax(i));
% Apply colors to each line using set function
set(gca, 'ColorOrder', colors(1:i,:), 'NextPlot', 'replacechildren');

plot(xcorrWindow, W{i}, 'linewidth', 2);  % Plot columns of W to visualize basis functions
    title(['Rank = ', num2str(rank)]);
    xlabel('Lags (frames)');
    ylabel('Basis Function Value');
    xline(0)
    legend
    % pause(1);  % Pause for inspection before moving to the next rank
end
titleN = sprintf('NNMF Basis Functions- %s', analyzeBhv{bhvIdx});
sgtitle(titleN, 'interpreter', 'none')
    print('-dpdf', fullfile(paths.figurePath, [titleN, '.pdf']), '-bestfit')




    %% Which basis functions dominate across pairs of neurons? 
rankIdxToTest = 3;

% Initialize cell array to store percentage dominance for each (originalM56, originalDS) pair
percentageDominance = cell(nM56, nDS);
domBasisIdx = zeros(nM56, nDS);        
        
        % Determine the dominant basis function for each Bout
        [~, dominantBasisForPair] = max(H{rankIdxToTest}, [], 1);
        
        % Calculate the percentage dominance of each basis function within this pair
        counts = histcounts(dominantBasisForPair, 1:rankIdxToTest+1);
        % % counts = histcounts(dominantBasisForPair, 1:(size(H_subset, 1)+1));
        % [~, domBasisIdx(m, d)] = max(counts);
        % percentageDominance{m, d} = (counts / length(dominantBasisForPair)) * 100;

% Which basis functions are most prevelantly dominant among the pairs of neurons, in terms of 
figure(219);
bar(1:rankIdxToTest, counts)
xlabel('Basis Function')
ylabel('Counts of neuron pairs')
title('Dominant basis functions among neuron pairs')



% How well-separated are the basis functions in terms of how much each contributes to the pairs of neurons?
    sortedValues = sort(H{rankIdxToTest}, 'descend');  % Sort in descending order

separationRatios = sortedValues(1,:) ./ sortedValues(2,:);

figure(216); 
% histogram(separationRatios, [1 : .02 : 3])
histogram(separationRatios)
title('NNMF Basis functions separation ratios among neuron pairs')
xlabel('Separation ratio')
ylabel('Counts')

















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
