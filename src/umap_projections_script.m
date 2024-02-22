% Get data from get_standard_data
cd 'E:/Projects/toolboxes/umapFileExchange (4.4)/umap/'
opts = neuro_behavior_options;
opts.collectStart = 0 * 60; % seconds
opts.collectFor = 45 * 60; % seconds
opts.frameSize = .1;

getDataType = 'all';
get_standard_data


%%
%%                           Dim-reduction and clustering
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Make a modified dataMat with big (e.g. 400ms) bins
binSize = .1;
nPerBin = round(binSize / opts.frameSize);
nBin = floor(size(dataMat, 1) / nPerBin);

% If n is not exactly divisible by k, you might want to handle the remainder
% For this example, we'll trim the excess
% dataMatTrimmed = dataMat(1 : nBin * nPerBin, :);

% Reshape and sum
dataMatReshaped = reshape(dataMat, nPerBin, nBin, size(dataMat, 2));
dataMatMod = squeeze(sum(dataMatReshaped, 1));

bhvIDReshaped = reshape(bhvIDMat, nPerBin, nBin);
% labels = bhvIDReshaped(floor(nPerBin/2) + 1, :)';
labels = bhvIDReshaped(end, :)';

%%
idInd = cell2mat(idAll); area = 'All';
idInd = idM56; area = 'M56';
idInd = idDS; area = 'DS';

% Only use part of the dataMat (if binSize is small)
nFrame = floor(size(dataMatMod, 1) / 1);
frameWindow = 1 : nFrame;
% windowStart = floor(size(dataMatMod, 1) / 1);
% frameWindow = windowStart : windowStart + nFrame - 1;

colors = colors_for_behaviors(codes);








%% t-SNE for all behaviors
projections = tsne(dataMatMod(frameWindow, idInd),'Algorithm','exact');

%%
hfig = figure(230);
colorsForPlot = arrayfun(@(x) colors(x,:), labels(frameWindow) + 2, 'UniformOutput', false);
colorsForPlot = vertcat(colorsForPlot{:}); % Convert cell array to a matrix
% scatter3(projections(:,1), projections(:,2), 1:nFrame, [], colorsForPlot, 'linewidth', 2);
scatter(projections(:,1), projections(:,2), [], colorsForPlot, 'linewidth', 2);

title(['t-SNE ' area, ' binSize = ', num2str(binSize)])
saveas(gcf, fullfile(paths.figurePath, ['t-sne ' area, ' binsize ', num2str(binSize), '.png']), 'png')
% Map labels to RGB colors


%% Classify using HDBSCAN
clusterer = HDBSCAN(projections);
clusterer.minpts = 4; %tends to govern cluster number  %was 3? with all neurons
clusterer.minclustsize = 11; %governs accuracy  %was 4? with all neurons
clusterer.fit_model(); 			% trains a cluster hierarchy
clusterer.get_best_clusters(); 	% finds the optimal "flat" clustering scheme
clusterer.get_membership();		% assigns cluster labels to the points in X
figure(828); clusterer.plot_clusters();
title(['t-SNE ' area, ' binSize = ', num2str(binSize)])
saveas(gcf, fullfile(paths.figurePath, ['t-sne HDBSCAN ' area, ' binsize ', num2str(binSize), '.png']), 'png')







%% 2-D UMAP for all behaviors
cd 'E:/Projects/toolboxes/umapFileExchange (4.4)/umap/'
[projections, umap, clusterIdentifiers, extras] = run_umap(dataMatMod(frameWindow, idInd));
pause(2); close
%%
% Create a figure window that fills the screen
monitorPositions = get(0, 'MonitorPositions');
secondMonitorPosition = monitorPositions(size(monitorPositions, 1), :); % Just use single monitor if you don't have second one
fig = figure(223); clf
set(fig, 'Position', secondMonitorPosition);

colorsForPlot = arrayfun(@(x) colors(x,:), labels(frameWindow) + 2, 'UniformOutput', false);
colorsForPlot = vertcat(colorsForPlot{:}); % Convert cell array to a matrix
scatter3(projections(:,1), projections(:,2), 1:size(projections,1), [], colorsForPlot, 'linewidth', 2);
% scatter(projections(:,1), projections(:,2), [], colorsForPlot, 'linewidth', 2);
xlabel('D1'); ylabel('D2'); zlabel('D3')
title(['UMAP ' area, ' binSize = ', num2str(binSize)])
saveas(gcf, fullfile(paths.figurePath, ['umap ' area, ' binsize ', num2str(binSize), '.png']), 'png')
%% Plot a short time span at a time to see how the dots are connected
figure(231)
colorsForPlot = arrayfun(@(x) colors(x,:), labels(frameWindow) + 2, 'UniformOutput', false);
colorsForPlot = vertcat(colorsForPlot{:}); % Convert cell array to a matrix

plotFrames = 45 / opts.frameSize; %
nPlot = floor(size(projections, 1) / plotFrames);

for iPlot = 1 : nPlot
    clf; hold on;
    iStart = 1 + (iPlot - 1) * plotFrames;
    iSpan = iStart : iStart + plotFrames - 1;
    scatter3(projections(iSpan,1), projections(iSpan,2), iSpan, 50, colorsForPlot(iSpan, :), 'linewidth', 3);
    plot3(projections(iSpan,1), projections(iSpan,2), iSpan, 'k', 'linewidth', 1);
    grid on;
end
title(['UMAP ' area, ' binSize = ', num2str(binSize)])
% saveas(gcf, fullfile(paths.figurePath, ['umap ' area, ' binsize ', num2str(binSize), '.png']), 'png')


%% Classify using HDBSCAN
clusterer = HDBSCAN(projections);
clusterer.minpts = 4; %tends to govern cluster number  %was 3? with all neurons
clusterer.minclustsize = 11; %governs accuracy  %was 4? with all neurons
clusterer.fit_model(); 			% trains a cluster hierarchy
clusterer.get_best_clusters(); 	% finds the optimal "flat" clustering scheme
clusterer.get_membership();		% assigns cluster labels to the points in X
figure(823); clusterer.plot_clusters();
title(['umap HDBSCAN ,' area, ' binSize = ', num2str(binSize)])
saveas(gcf, fullfile(paths.figurePath, ['umap HDBSCAN ,' area, ' binsize ', num2str(binSize), '.png']), 'png')








%% DO UMAP on just the transitions

periTime = .1 : opts.frameSize : .3;
periWindow = round(periTime(1:end-1) / opts.frameSize); % frames around onset (remove last frame)

dataMatOnsets = [];
labels = [];
dataStartFrame = [];
for i = 3 : length(dataBhv.StartFrame)
    if dataBhv.DurFrame(i) > 0
        dataMatOnsets = [dataMatOnsets; sum(dataMat(dataBhv.StartFrame(i) + periWindow, :), 1)];
        labels = [labels; dataBhv.ID(i)];
        dataStartFrame = [dataStartFrame; dataBhv.StartFrame(i)];
    end
end

%% 2-D UMAP for all behavior onsets
[projections, umap, clusterIdentifiers, extras] = run_umap(dataMatOnsets(:, idInd));
pause(2); close;

%%
% Create a figure window that fills the screen
monitorPositions = get(0, 'MonitorPositions');
secondMonitorPosition = monitorPositions(size(monitorPositions, 1), :); % Just use single monitor if you don't have second one
fig = figure(232); clf;
set(fig, 'Position', secondMonitorPosition);

colorsForPlot = arrayfun(@(x) colors(x,:), labels + 2, 'UniformOutput', false);
colorsForPlot = vertcat(colorsForPlot{:}); % Convert cell array to a matrix
scatter3(projections(:,1), projections(:,2), 1:size(projections,1), [], colorsForPlot, 'linewidth', 2);
% scatter(projections(:,1), projections(:,2), [], colorsForPlot, 'linewidth', 2);
title(['UMAP Onsets ' area, ' binSize = ', num2str(binSize)])
grid on;
% saveas(gcf, fullfile(paths.figurePath, ['umap ' area, ' binsize ', num2str(binSize), '.png']), 'png')


%% Plot a short time span at a time to see how the dots are connected
figure(233)
colorsForPlot = arrayfun(@(x) colors(x,:), labels + 2, 'UniformOutput', false);
colorsForPlot = vertcat(colorsForPlot{:}); % Convert cell array to a matrix

plotFrames = 1 / opts.frameSize; %
nPlot = floor(size(projections, 1) / opts.frameSize / plotFrames);

for iPlot = 1 : nPlot
    clf; hold on;
    iStart = 1 + (iPlot - 1) * plotFrames;
    iSpan = dataStartFrame >= iStart & dataStartFrame < iStart + plotFrames - 1;
    scatter3(projections(iSpan,1), projections(iSpan,2), dataStartFrame(iSpan), 50, colorsForPlot(iSpan, :), 'linewidth', 3);
    plot3(projections(iSpan,1), projections(iSpan,2), dataStartFrame(iSpan), 'k', 'linewidth', 1);
    grid on;
end
title(['UMAP Onsets ' area, ' binSize = ', num2str(binSize)])
% saveas(gcf, fullfile(paths.figurePath, ['umap ' area, ' binsize ', num2str(binSize), '.png']), 'png')








%% UMAP for all behaviors, alllowing more than 2 dimensions
[projections, umap, clusterIdentifiers, extras] = run_umap(dataMatMod(frameWindow, idInd), 'n_components', 6);
pause(2); close
%% Plot dimensions
d1 = 1;
d2 = 2;
figure(230); clf
colorsForPlot = arrayfun(@(x) colors(x,:), labels(frameWindow) + 2, 'UniformOutput', false);
colorsForPlot = vertcat(colorsForPlot{:}); % Convert cell array to a matrix
% scatter3(projections(:,d1), projections(:,d2), 1:size(projections,1), [], colorsForPlot, 'linewidth', 2);
scatter3(projections(:,d1), projections(:,d2), projections(:,3), [], colorsForPlot, 'linewidth', 2);
% scatter(projections(:,d1), projections(:,d2), [], colorsForPlot, 'linewidth', 2);
title(['UMAP ' area, ' binSize = ', num2str(binSize)])
xlabel('D1'); ylabel('D2'); zlabel('D3')
saveas(gcf, fullfile(paths.figurePath, ['umap ' area, ' binsize ', num2str(binSize), '.png']), 'png')
%% Plot a short time span at a time to see how the dots are connected
figure(231)
colorsForPlot = arrayfun(@(x) colors(x,:), labels(frameWindow) + 2, 'UniformOutput', false);
colorsForPlot = vertcat(colorsForPlot{:}); % Convert cell array to a matrix

plotFrames = 45 / opts.frameSize; %
nPlot = floor(size(projections, 1) / plotFrames);

for iPlot = 1 : nPlot
    clf; hold on;
    iStart = 1 + (iPlot - 1) * plotFrames;
    iSpan = iStart : iStart + plotFrames - 1;
    scatter3(projections(iSpan,1), projections(iSpan,2), iSpan, 50, colorsForPlot(iSpan, :), 'linewidth', 3);
    plot3(projections(iSpan,1), projections(iSpan,2), iSpan, 'k', 'linewidth', 1);
    grid on;
end
title(['UMAP ' area, ' binSize = ', num2str(binSize)])
% saveas(gcf, fullfile(paths.figurePath, ['umap ' area, ' binsize ', num2str(binSize), '.png']), 'png')








%% Histograms of the different behaviors per dimension, in UMAP space

% Create a figure window that fills the screen
monitorPositions = get(0, 'MonitorPositions');
secondMonitorPosition = monitorPositions(size(monitorPositions, 1), :); % Just use single monitor if you don't have second one
fig = figure(234); clf;
set(fig, 'Position', secondMonitorPosition);
sgtitle('UMAP Dimension histograms')
% Loop through behaviors to plot historgram of each
for jBhv = 1 : length(codes)
    for iDim = 1 : 8
        figure(234)
        subplot(4,2,iDim); hold on;

        % edges = floor(min(longTraj(iDim,:))) : .5 : ceil(max(longTraj(iDim,:)));
        edges = min(projections(:, iDim)) : .2 : max(projections(:, iDim));
        binCenters = (edges(1:end-1) + edges(2:end)) / 2;

        % xxxx
        % N = histcounts(reduction(bhvIDMod == codes(jBhv), iDim), edges, 'Normalization', 'pdf');
        N = histcounts(projections(labels == codes(jBhv), iDim), edges);
        plot(binCenters, N./sum(N), 'Color', colors(jBhv,:), 'lineWidth', 3)
        title(['Dimension ', num2str(iDim)])
        xline(0)
    end
end







%% Plot individual behaviors. Use heatmap color scheme to differentiate beginning --> end of bouts
% Create a figure window that fills the screen
monitorPositions = get(0, 'MonitorPositions');
secondMonitorPosition = monitorPositions(size(monitorPositions, 1), :); % Just use single monitor if you don't have second one
fig = figure(250); clf; hold on;
set(fig, 'Position', secondMonitorPosition);

bhvCode = 1;
bhvCode = codes(find(strcmp(behaviors, 'face_groom_1')));
bhvInd = labels == bhvCode;
bhvStarts = find([0; diff(bhvInd)] == 1);
bhvEnds = find([0; diff(bhvInd)] == -1) - 1;

for iBout = 1 : length(bhvStarts)
    iInd = bhvStarts(iBout) : bhvEnds(iBout);
    iColors = three_color_heatmap([0 0 1], [.7 .7 .7], [1 0 0], length(iInd));
    plot3(projections(iInd, 1), projections(iInd, 2), iInd, 'color', [.5 .5 .5])
    scatter3(projections(iInd, 1), projections(iInd, 2), iInd, 40, iColors, 'linewidth', 2)
end







%%                      Compare common behavioral sequences
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Get an hour
cd 'E:/Projects/toolboxes/umapFileExchange (4.4)/umap/'

opts = neuro_behavior_options;
opts.collectStart = 0 * 60; % seconds
opts.collectFor = 2 * 60 * 60; % seconds
opts.frameSize = .1;

getDataType = 'all';
get_standard_data

%% Run UMAPto get projectsions in low-D space

idInd = idDS;
nComponents = 3;
[projections, umap, clusterIdentifiers, extras] = run_umap(dataMat(:, idInd), 'n_components', nComponents);
pause(5); close



%%
nSeq = 3;
requireValid = [0 1 0];
% requireValid = [1 1];
% requireValid = [0 0 0];
[uniqueSequences, sequenceIndices] = find_unique_sequences(dataBhv, nSeq, requireValid);

uniqueSequences(1:40)
cellfun(@length, sequenceIndices(1:40))


%% Get just the triplets with a particular behavior in the middle

middleBhv = 6;
middleIdx = find(cellfun(@(x) x(2) == middleBhv, uniqueSequences)); % Sequences in uniqueSequences with middle behavior = middleBhv
uniqueSequences(middleIdx)
cellfun(@length, sequenceIndices(middleIdx))

%%
figure(929); clf; hold on; grid on;
colors = colors_for_behaviors(codes);
middleColor = colors(middleBhv+2,:);
for i = 1 : length(middleIdx)
    iFirstBhv = uniqueSequences{middleIdx(i)}(1);
    iLastBhv = uniqueSequences{middleIdx(i)}(end);
    iFirstColor = colors(iFirstBhv+2,:);
    iLastColor = colors(iLastBhv+2,:);
    iDataBhvIdx = sequenceIndices{middleIdx(i)};
    % Loop through each instance of this sequence and plot from 1st to 2nd
    % to 3rd behavior
    for j = 1 : length(iDataBhvIdx)
        start1 = dataBhv.StartFrame(iDataBhvIdx(j));
        start2 = dataBhv.StartFrame(iDataBhvIdx(j) + 1);
        start3 = dataBhv.StartFrame(iDataBhvIdx(j) + 2);

        plot3(projections(start1:start2, 1), projections(start1:start2, 2), projections(start1:start2, 3), 'color', [.7 .7 .7])
        plot3(projections(start2:start3, 1), projections(start2:start3, 2), projections(start2:start3, 3), 'color', [.7 .7 .7])
        scatter3(projections(start1, 1), projections(start1, 2), projections(start1, 3), 60, iFirstColor, 'LineWidth', 2)
        scatter3(projections(start2, 1), projections(start2, 2), projections(start2, 3), 100, middleColor, 'LineWidth', 2)
        scatter3(projections(start3, 1), projections(start3, 2), projections(start3, 3), 60, iLastColor, 'LineWidth', 2)
    end
end


%%
[seqStartTimes, seqCodes, seqNames] = behavior_sequences(dataBhv, analyzeCodes, analyzeBhv)

% for a given behavior, make a histogram of the behaviors that precede it.
bhv = 6;





