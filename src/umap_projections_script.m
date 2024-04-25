% Get data from get_standard_data
cd 'E:/Projects/toolboxes/umapFileExchange (4.4)/umap/'
opts = neuro_behavior_options;
opts.collectStart = 0 * 60; % seconds
opts.collectFor = 45 * 60; % seconds
% opts.frameSize = .1;

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
[projections, umap, clusterIdentifiers, extras] = run_umap(dataMatMod(frameWindow, idInd), 'n_components', 3);
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













%%                      Sequences: Compare common behavioral sequences
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Get data
cd 'E:/Projects/toolboxes/umapFileExchange (4.4)/umap/'

opts = neuro_behavior_options;
opts.collectStart = 0 * 60; % seconds
opts.collectFor = 2 * 60 * 60; % seconds
opts.frameSize = .15;

getDataType = 'all';
get_standard_data

%% Curate behavior labels if desired
[dataBhv, bhvIDMat] = curate_behavior_labels(dataBhv, opts);


%%
nSeq = 3;
requireValid = [0 1 0];
% requireValid = [1 0];
% requireValid = [0 0 0];
[uniqueSequences, sequenceIndices] = find_unique_sequences(dataBhv, nSeq, requireValid);

uniqueSequences(1:50)
cellfun(@length, sequenceIndices(1:50))

startBhvIdx = 1; % Which behavior in the sequence to plot as the start point. (The index in the sequence)

%%
grooms = 5:12;
matches = find(cellfun(@(x) all(ismember(x, grooms)), uniqueSequences));
uniqueSequences(matches(1:10))
cellfun(@length, sequenceIndices(matches(1:10)))

%% Which sequences to plot
% seqIdx = [1 2 14 18];
seqIdx = matches(1:4);

nTrial = min(30, min(cellfun(@length, sequenceIndices(seqIdx))));

%% Run UMAPto get projectsions in low-D space
idInd = idM56;
nComponents = 3;
[projectionsM56, ~, ~, ~] = run_umap(dataMat(:, idInd), 'n_components', nComponents);
pause(5); close

%
idInd = idDS;
[projectionsDS, ~, ~, ~] = run_umap(dataMat(:, idInd), 'n_components', nComponents);
pause(5); close



%% For sequences of 2 or 3
uniqueSequences(seqIdx)

% colors = [1 0 0; 0 0 1; 0 .7 0; 0 0 .15];
colors = colors_for_behaviors(codes);
figure(520); clf; hold on; title(['UMAP M56, D=', num2str(nComponents)]);
figure(521); clf; hold on; title(['UMAP DS, D=', num2str(nComponents)]);
for i = 1 : length(seqIdx)
    bhv1 = uniqueSequences{seqIdx(i)}(1);
    bhv2 = uniqueSequences{seqIdx(i)}(2);
    bhv3 = uniqueSequences{seqIdx(i)}(3);
    color1 = colors(bhv1+2,:);
    color2 = colors(bhv2+2,:);
    color3 = colors(bhv3+2,:);


    for j = 1 : nTrial
        start1 = dataBhv.StartFrame(sequenceIndices{i}(j));
        start2 = dataBhv.StartFrame(sequenceIndices{i}(j)+1);
        start3 = dataBhv.StartFrame(sequenceIndices{i}(j)+2);

        %  M56 data
        figure(520);
        plot3(projectionsM56(start1:start3, 1), projectionsM56(start1:start3, 2), projectionsM56(start1:start3, 3), '.-', 'Color', [.5 .5 .5], 'LineWidth', 1, 'MarkerSize', 10')
        scatter3(projectionsM56(start1, 1), projectionsM56(start1, 2), projectionsM56(start1, 3), 100, color1, 'filled')
        scatter3(projectionsM56(start2, 1), projectionsM56(start2, 2), projectionsM56(start2, 3), 100, color2, 'filled')
        scatter3(projectionsM56(start3, 1), projectionsM56(start3, 2), projectionsM56(start3, 3), 100, color3, 'filled')
        grid on;
        xlabel('D1'); ylabel('D2'); zlabel('D3')

        %  DS data
        figure(521)
        plot3(projectionsDS(start1:start3, 1), projectionsDS(start1:start3, 2), projectionsDS(start1:start3, 3), '.-', 'Color', [.5 .5 .5], 'LineWidth', 1, 'MarkerSize', 10')
        scatter3(projectionsDS(start1, 1), projectionsDS(start1, 2), projectionsDS(start1, 3), 100, color1, 'filled')
        scatter3(projectionsDS(start2, 1), projectionsDS(start2, 2), projectionsDS(start2, 3), 100, color2, 'filled')
        scatter3(projectionsDS(start3, 1), projectionsDS(start3, 2), projectionsDS(start3, 3), 100, color2, 'filled')
        xlabel('D1'); ylabel('D2'); zlabel('D3')
        grid on;
    end
end






%% Find sequences of 3 and collect their indices
nSeq = 3;
requireValid = [0 1 0];
% requireValid = [0 0 0];
[uniqueSequences, sequenceIndices] = find_unique_sequences(dataBhv, nSeq, requireValid);

minSeq = 7;
bhvList = 5:12;

seqList = {};
seqInd = {};
for i = 1 : length(uniqueSequences)
    if any(ismember(uniqueSequences{i}, bhvList)) && ~ismember(-1, uniqueSequences{i}) % If any interesting behaviors are part of the sequence...
        if length(sequenceIndices{i}) >= minSeq % If there enough instances of this sequence, collect it
            seqList = [seqList, uniqueSequences{i}];
            seqInd = [seqInd, sequenceIndices{i}];

            fprintf('Sequence: %s\t %s\t %s\n', behaviors{codes == uniqueSequences{i}(1)}, ...
                behaviors{codes == uniqueSequences{i}(2)}, ...
                behaviors{codes == uniqueSequences{i}(3)});
            fprintf('Number: %d\n', length(sequenceIndices{i}))
        else
            continue
        end
    end
end

%%
colors = colors_for_behaviors(codes);
figure(420); clf; hold on; title('UMAP M56');
figure(421); clf; hold on; title('UMAP DS');
for i = 3 : length(seqList)
    bhv1 = seqList{i}(1);
    bhv2 = seqList{i}(2);
    bhv3 = seqList{i}(3);
    color1 = colors(bhv1+2,:);
    color2 = colors(bhv2+2,:);
    color3 = colors(bhv3+2,:);


    for j = 1 : length(seqInd{i})
        start1 = dataBhv.StartFrame(seqInd{i}(j));
        start2 = dataBhv.StartFrame(seqInd{i}(j)+1);
        start3 = dataBhv.StartFrame(seqInd{i}(j)+2);

        %  M56 data
        figure(420);
        plot3(projectionsM56(start1:start2, 1), projectionsM56(start1:start2, 2), projectionsM56(start1:start2, 3), '.-', 'Color', color1, 'LineWidth', 2, 'MarkerSize', 10')
        plot3(projectionsM56(start2:start3, 1), projectionsM56(start2:start3, 2), projectionsM56(start2:start3, 3), '.-', 'Color', color2, 'LineWidth', 2, 'MarkerSize', 10')
        scatter3(projectionsM56(start1, 1), projectionsM56(start1, 2), projectionsM56(start1, 3), 100, color1, 'filled')
        scatter3(projectionsM56(start2, 1), projectionsM56(start2, 2), projectionsM56(start2, 3), 100, color2, 'filled')
        scatter3(projectionsM56(start3, 1), projectionsM56(start3, 2), projectionsM56(start3, 3), 100, color3, 'filled')
        grid on;
        xlabel('D1'); ylabel('D2'); zlabel('D3')

        %  DS data
        figure(421)
        plot3(projectionsDS(start1:start2, 1), projectionsDS(start1:start2, 2), projectionsDS(start1:start2, 3), '.-', 'Color', color1, 'LineWidth', 2, 'MarkerSize', 10')
        plot3(projectionsDS(start2:start3, 1), projectionsDS(start2:start3, 2), projectionsDS(start2:start3, 3), '.-', 'Color', color2, 'LineWidth', 2, 'MarkerSize', 10')
        scatter3(projectionsDS(start1, 1), projectionsDS(start1, 2), projectionsDS(start1, 3), 100, color1, 'filled')
        scatter3(projectionsDS(start2, 1), projectionsDS(start2, 2), projectionsDS(start2, 3), 100, color2, 'filled')
        scatter3(projectionsDS(start3, 1), projectionsDS(start3, 2), projectionsDS(start3, 3), 100, color3, 'filled')
        xlabel('D1'); ylabel('D2'); zlabel('D3')
        grid on;
    end
end







%% For sequences of 3: Get just the triplets with a particular behavior in the middle

middleBhv = 15;
middleIdx = find(cellfun(@(x) x(2) == middleBhv, uniqueSequences)); % Sequences in uniqueSequences with middle behavior = middleBhv
uniqueSequences(middleIdx)
cellfun(@length, sequenceIndices(middleIdx))

%%
figure(929); clf; hold on; grid on;
colors = colors_for_behaviors(codes);
middleColor = colors(middleBhv+2,:);
for i = 1 : length(middleIdx)
    bhv1 = uniqueSequences{middleIdx(i)}(1);
    bhv2 = uniqueSequences{middleIdx(i)}(end);
    iFirstColor = colors(bhv1+2,:);
    iLastColor = colors(bhv2+2,:);
    iDataBhvIdx = sequenceIndices{middleIdx(i)};
    uniqueSequences{middleIdx(i)}
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





















%%                     Compare neuro-behavior in UMAP spaces
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
cd 'E:/Projects/toolboxes/umapFileExchange (4.4)/umap/'

opts = neuro_behavior_options;
opts.minActTime = .16;
opts.collectStart = 0 * 60; % seconds
opts.collectFor = .5 * 60 * 60; % seconds
opts.frameSize = .15;

getDataType = 'all';
get_standard_data
%%
[dataBhv, bhvIDMat] = curate_behavior_labels(dataBhv, opts);

%% Run UMAPto get projections in low-D space
nComponents = 8;

idInd = idM56;
[projM56, ~, ~, ~] = run_umap(dataMat(:, idInd), 'n_components', nComponents, 'randomize', false);
pause(3); close
%
idInd = idDS;
rng(1); % 'shuffle'
randSub = randperm(length(idDS), length(idM56));
idInd = idDS(randSub);
[projDS, ~, ~, ~] = run_umap(dataMat(:, idInd), 'n_components', nComponents, 'randomize', false);
pause(3); close


%% --------------------------------------------
% Shift behavior label w.r.t. neural to account for neuro-behavior latency
shiftSec = 0;
shiftFrame = ceil(shiftSec / opts.frameSize);

bhvID = bhvIDMat(1+shiftFrame:end); % Shift bhvIDMat to account for time shift
projectionsM56 = projM56(1:end-shiftFrame, :); % Remove shiftFrame frames from projections to accoun for time shift in bhvIDMat
projectionsDS = projDS(1:end-shiftFrame, :);


%% --------------------------------------------
% Plot FULL TIME OF ALL BEHAVIORS
dimPlot = [1 2 3];

colors = colors_for_behaviors(codes);
colorsForPlot = arrayfun(@(x) colors(x,:), bhvID + 2, 'UniformOutput', false);
colorsForPlot = vertcat(colorsForPlot{:}); % Convert cell array to a matrix

figure(220); clf; hold on;
titleM = ['UMAP M56 ', num2str(nComponents), 'D,  bin = ', num2str(opts.frameSize), ' shift = ', num2str(shiftSec)];
title(titleM)
if nComponents > 2
    scatter3(projectionsM56(:, dimPlot(1)), projectionsM56(:, dimPlot(2)), projectionsM56(:, dimPlot(3)), 60, colorsForPlot, 'LineWidth', 2)
elseif nComponents == 2
    scatter(projectionsM56(:, dimPlot(1)), projectionsM56(:, dimPlot(2)), 60, colorsForPlot, 'LineWidth', 2)
end
grid on;
xlabel('D1'); ylabel('D2'); zlabel('D3')
% saveas(gcf, fullfile(paths.figurePath, [titleM, '.png']), 'png')


figure(221); clf; hold on;
titleD = ['UMAP DS ', num2str(nComponents), 'D, bin = ', num2str(opts.frameSize), ' shift = ', num2str(shiftSec)];
title(titleD)
if nComponents > 2
    scatter3(projectionsDS(:, dimPlot(1)), projectionsDS(:, dimPlot(2)), projectionsDS(:, dimPlot(3)), 60, colorsForPlot, 'LineWidth', 2)
elseif nComponents == 2
    scatter(projectionsDS(:, dimPlot(1)), projectionsDS(:, dimPlot(2)), 60, colorsForPlot, 'LineWidth', 2)
end
grid on;
xlabel('D1'); ylabel('D2'); zlabel('D3')
% saveas(gcf, fullfile(paths.figurePath, [titleD, '.png']), 'png')









%% --------------------------------------------
% Plot just the TRANSITIONS with individual behavior labels
dimPlot = [1 2 3];

behaviorsPlot = {'investigate_1', 'head_groom'};
behaviorsPlot = {'contra_itch', 'paw_groom'};
behaviorsPlot = {'locomotion', 'contra_orient', 'ipsi_orient'};
behaviorsPlot = {'contra_itch', 'rear'};
behaviorsPlot = {'investigate_2'};
% behaviorsPlot = {'paw_groom'};
behaviorsPlot = {'locomotion'};
% behaviorsPlot = {'face_groom_1'};


colors = colors_for_behaviors(codes);
periEventTime = -opts.frameSize : opts.frameSize : 0; % seconds around onset
% periEventTime = -opts.frameSize : opts.frameSize : opts.frameSize; % seconds around onset
dataWindow = floor(periEventTime(1:end-1) / opts.frameSize); % frames around onset (remove last frame)

figure(230); clf; hold on;
titleM = ['UMAP M56 ', num2str(nComponents), 'D,  bin = ', num2str(opts.frameSize), ' shift = ', num2str(shiftSec)];
title(titleM)
grid on;
xlabel('D1'); ylabel('D2'); zlabel('D3')

figure(231); clf; hold on;
titleD = ['UMAP DS ', num2str(nComponents), 'D, bin = ', num2str(opts.frameSize), ' shift = ', num2str(shiftSec)];
title(titleD)
grid on;
xlabel('D1'); ylabel('D2'); zlabel('D3')

for i = 1 : length(behaviorsPlot)
    bhvPlot = find(strcmp(behaviors, behaviorsPlot{i})) - 2;

    allInd = bhvID == bhvPlot; % all labeled target behaviors
    firstInd = 1 + find(diff(allInd) == 1); % first frames of all target behaviors

    transitionsInd = zeros(length(dataWindow) * length(firstInd), 1);
    for j = 1 : length(firstInd)
        % Calculate the start index in the expanded array
        startIndex = (j-1) * length(dataWindow) + 1;

        % Add dataWindow to the current element of firstInd and store it in the correct position
        transitionsInd(startIndex:startIndex + length(dataWindow) - 1) = firstInd(j) + dataWindow;
    end

    figure(230)
    if nComponents > 2
        scatter3(projectionsM56(transitionsInd, dimPlot(1)), projectionsM56(transitionsInd, dimPlot(2)), projectionsM56(transitionsInd, dimPlot(3)), 60, colors(bhvPlot+2,:), 'LineWidth', 2)
    elseif nComponents == 2
        scatter3(projectionsM56(transitionsInd, dimPlot(1)), projectionsM56(transitionsInd, dimPlot(2)), opts.frameSize * (transitionsInd-1), 60, colors(bhvPlot+2,:), 'LineWidth', 2)
        % plot3(projectionsM56(transitionsInd, dimPlot(1)), projectionsM56(transitionsInd, dimPlot(2)), opts.frameSize * (transitionsInd-1), 60, '-', 'color', [.6 .6 .6])
        % scatter(projectionsM56(transitionsInd, dimPlot(1)), projectionsM56(transitionsInd, dimPlot(2)), 60, colors(bhvPlot+2,:), 'LineWidth', 2)
    end

    figure(231)
    if nComponents > 2
        scatter3(projectionsDS(transitionsInd, dimPlot(1)), projectionsDS(transitionsInd, dimPlot(2)), projectionsDS(transitionsInd, dimPlot(3)), 60, colors(bhvPlot+2,:), 'LineWidth', 2)
    elseif nComponents == 2
        scatter3(projectionsDS(transitionsInd, dimPlot(1)), projectionsDS(transitionsInd, dimPlot(2)), opts.frameSize * (transitionsInd-1), 60, colors(bhvPlot+2,:), 'LineWidth', 2)
        % plot3(projectionsDS(transitionsInd, dimPlot(1)), projectionsDS(transitionsInd, dimPlot(2)), opts.frameSize * (transitionsInd-1), '-', 'color', [.6 .6 .6])
        % scatter(projectionsDS(transitionsInd, dimPlot(1)), projectionsDS(transitionsInd, dimPlot(2)), 60, colors(bhvPlot+2,:), 'LineWidth', 2)
    end
end

figure(230)
% saveas(gcf, fullfile(paths.figurePath, [titleM, '.png']), 'png')
figure(231)
% saveas(gcf, fullfile(paths.figurePath, [titleD, '.png']), 'png')


%% Use a square to Find time indices according to an area within low-D space
% area1D1Window = [0 3];
% area1D2Window = [2 6];
%
% area2D1Window = [-4 2];
% area2D2Window = [-3 2];
%
% area1Ind = bhvID == bhvPlot &...
%     (projectionsDS(:,1) >= area1D1Window(1) & projectionsDS(:,1) <= area1D1Window(2)) & ...
%     (projectionsDS(:,2) >= area1D2Window(1) & projectionsDS(:,2) <= area1D2Window(2));
% area1Time = (find(area1Ind)-1) * opts.frameSize;
%
% area2Ind = bhvID == bhvPlot &...
%     (projectionsDS(:,1) >= area2D1Window(1) & projectionsDS(:,1) <= area2D1Window(2)) & ...
%     (projectionsDS(:,2) >= area2D2Window(1) & projectionsDS(:,2) <= area2D2Window(2));
% area2Time = (find(area2Ind)-1) * opts.frameSize;


%% Use a circle instead of a square
selectFrom = 'M56';
selectFrom = 'DS';
switch selectFrom
    case 'M56'
        projSelect = projectionsM56;
        projProject = projectionsDS;
        idSelect = idM56;
    case 'DS'
        projSelect = projectionsDS;
        projProject = projectionsM56;
        idSelect = idDS;
end        

clear center radius
center{1} = [-1.25 -1];
center{2} = [-4.5 1.75];
center{3} = [0 4];
radius{1} = 1;
radius{2} = 1.5;
radius{3} = 2;
% center{1} = [-2.5 -8];
% center{2} = [0 0];
% center{3} = [1 -6];
% radius{1} = 2;
% radius{2} = 2;
% radius{3} = 2;

colors = three_color_heatmap([1 0 0],[0 .7 0], [0 0 1], 3);

bhvFrame = cell(length(center), 1);
prevBhv = cell(length(center), 1);

% Number of points to define the circle
theta = linspace(0, 2*pi, 100);  % Generate 100 points along the circle

plotTime = -3 : opts.frameSize : 3; % seconds around onset
plotFrame = round(plotTime(1:end-1) / opts.frameSize);
matSelect = cell(length(center), 1);

for i = 1 : length(center)
if strcmp(selectFrom, 'M56')
    figure(230)
else
    figure(231)
end
    % Parametric equation of the circle
    x = center{i}(1) + radius{i} * cos(theta);
    y = center{i}(2) + radius{i} * sin(theta);
    plot(x, y, 'color', colors(i,:), 'lineWidth', 4);  % Plot the circle with a blue line


    % Calculate the Euclidean distance between the point and the center of the circle
    iDistance = sqrt((projSelect(:, 1) - center{i}(1)).^2 + (projSelect(:, 2) - center{i}(2)).^2);

    % Determine if the point is inside the circle (including on the boundary)
    isInside = iDistance <= radius{i};

    bhvInd = zeros(length(projSelect), 1);
    bhvInd(transitionsInd) = 1;

    bhvFrame{i} = find(isInside & bhvInd);
    % Get rid of instances too close to beginning/end (to enable plotting
    % them)
    bhvFrame{i}((bhvFrame{i} < -plotFrame(1)) | bhvFrame{i} > size(dataMat, 1) - plotFrame(end)) = [];

    prevBhv{i} = bhvID(bhvFrame{i});

    % Plot the data in the other area
if strcmp(selectFrom, 'M56')
    figure(231)
else
    figure(230)
end
    scatter3(projProject(bhvFrame{i}, dimPlot(1)), projProject(bhvFrame{i}, dimPlot(2)), opts.frameSize * (bhvFrame{i}-1), 60, colors(i,:), 'LineWidth', 2)

    % Collect the neural data peri-bout start
    nBout = length(bhvFrame{i});
    iMatSelect = zeros(length(idSelect), length(plotFrame), nBout);
    for n = 1 : nBout
        iMatSelect(:, :, n) = dataMat((bhvFrame{i}(n) - dataWindow) + plotFrame, idSelect)';
    end
    matSelect{i} = iMatSelect;
end

n = 1:10;
% t = seconds([bhvTime1(n) bhvTime2(n) bhvTime3(n) bhvTime4(n) bhvTime5(n)] .* opts.frameSize);
t = seconds([bhvFrame{1}(n) bhvFrame{2}(n) bhvFrame{3}(n)] .* opts.frameSize);
t.Format = 'hh:mm:ss.S'

%% Want to get all the behaviors just before this one, for each group in UMAP space
figure(11)
for i = 1 : length(center)
    [uniqueElements, ~, idx] = unique(prevBhv{i});
    counts = accumarray(idx, 1);

    % Create the bar graph
   subplot(1, length(center), i);  % Open a new figure window
    bar(uniqueElements, counts, 'FaceColor', colors(i,:));  % Create a bar plot
end


%% Plot PSTHS of the various groups
figure(3235);

allMat = [];
for i = 1 : length(center)
    allMat = cat(3, allMat, matSelect{i});
end
meanMat = mean(allMat, 3);
meanWindow = mean(meanMat, 2);
stdWindow = std(meanMat, [], 2);

zMat = cell(length(center), 1);
zMeanMat = cell(length(center), 1);
for i = 1 : length(center)

    % z-score the psth data
    zMat{i} = (matSelect{i} - meanWindow) ./ stdWindow;
    zMeanMat{i} = mean(zMat{i}, 3);
    subplot(1, length(center), i)
    % imagesc(mean(matSelect{i}, 3))
    imagesc(zMeanMat{i})
    colormap(bluewhitered_custom([-4 4]))
    title(num2str(i))

end
% colorbar

figure(423);
diffMat12 = zMeanMat{1} - zMeanMat{2};
diffMat13 = zMeanMat{1} - zMeanMat{3};
diffMat23 = zMeanMat{2} - zMeanMat{3};
% diffMat12 = mean(matSelect{1}, 3) - mean(matSelect{2}, 3);
% diffMat13 = mean(matSelect{1}, 3) - mean(matSelect{3}, 3);
% diffMat23 = mean(matSelect{2}, 3) - mean(matSelect{3}, 3);
colorBarSpan = [-4 4];
subplot(1,3,1)
    imagesc(diffMat12)
    colormap(bluewhitered_custom(colorBarSpan))
    title('1 - 2')
subplot(1,3,2)
    imagesc(diffMat13)
    colormap(bluewhitered_custom(colorBarSpan))
    title('1 - 3')
subplot(1,3,3)
    imagesc(diffMat23)
    colormap(bluewhitered_custom(colorBarSpan))
    title('2 - 3')


testFrame = (-plotFrame(1) + 1) + (-1:0);
zTest1 = mean(zMeanMat{1}(:, testFrame), 2);
zTest2 = mean(zMeanMat{2}(:, testFrame), 2);
zTest3 = mean(zMeanMat{3}(:, testFrame), 2);
[h,p] = ttest(zTest1, zTest2)
[h,p] = ttest(zTest1, zTest3)
[h,p] = ttest(zTest2, zTest3)







%% Plot in M56
figure(230); clf; hold on; grid on;
if nComponents == 3
    scatter3(projectionsM56(transitionsInd, dimPlot(1)), projectionsM56(transitionsInd, dimPlot(2)), projectionsM56(transitionsInd, dimPlot(3)), 60, colors(bhvPlot+2,:), 'LineWidth', 2)
elseif nComponents == 2
    scatter3(projectionsM56(bhvTime1, dimPlot(1)), projectionsM56(bhvTime1, dimPlot(2)), opts.frameSize * (bhvTime1-1), 60, colors(bhvPlot+2,:), 'LineWidth', 2)
    scatter3(projectionsM56(bhvTime2, dimPlot(1)), projectionsM56(bhvTime2, dimPlot(2)), opts.frameSize * (bhvTime2-1), 60, colors(bhvPlot+2,:), 'LineWidth', 2)
    scatter3(projectionsM56(bhvTime3, dimPlot(1)), projectionsM56(bhvTime3, dimPlot(2)), opts.frameSize * (bhvTime3-1), 60, colors(bhvPlot+2,:), 'LineWidth', 2)
    % plot3(projectionsM56(transitionsInd, dimPlot(1)), projectionsM56(transitionsInd, dimPlot(2)), opts.frameSize * (transitionsInd-1), 60, '-', 'color', [.6 .6 .6])
    % scatter(projectionsM56(transitionsInd, dimPlot(1)), projectionsM56(transitionsInd, dimPlot(2)), 60, colors(bhvPlot+2,:), 'LineWidth', 2)
end










%% --------------------------------------------
% Plot particular behaviors etc - FULL TIME during behaviors
dimPlot = [1 2 3];

behaviorsPlot = {'investigate_2', 'locomotion', 'contra_itch'};
behaviorsPlot = {'locomotion'};
colors = colors_for_behaviors(codes);

figure(230); clf; hold on;
titleM = ['UMAP M56 ', num2str(nComponents), 'D,  bin = ', num2str(opts.frameSize), ' shift = ', num2str(shiftSec)];
title(titleM)
grid on;
xlabel('D1'); ylabel('D2'); zlabel('D3')

figure(231); clf; hold on;
titleD = ['UMAP DS ', num2str(nComponents), 'D, bin = ', num2str(opts.frameSize), ' shift = ', num2str(shiftSec)];
title(titleD)
grid on;
xlabel('D1'); ylabel('D2'); zlabel('D3')

for i = 1 : length(behaviorsPlot)
    bhvPlot = find(strcmp(behaviors, behaviorsPlot{i})) - 2;

    plotInd = bhvID == bhvPlot;

    figure(230)
    if nComponents == 3
        scatter3(projectionsM56(plotInd, dimPlot(1)), projectionsM56(plotInd, dimPlot(2)), projectionsM56(plotInd, dimPlot(3)), 60, colors(bhvPlot+2,:), 'LineWidth', 2)
    elseif nComponents == 2
        scatter3(projectionsM56(plotInd, dimPlot(1)), projectionsM56(plotInd, dimPlot(2)), opts.frameSize * (find(plotInd)-1), 60, colors(bhvPlot+2,:), 'LineWidth', 2)
    end
    % saveas(gcf, fullfile(paths.figurePath, [titleM, '.png']), 'png')


    figure(231)
    if nComponents == 3
        scatter3(projectionsDS(plotInd, dimPlot(1)), projectionsDS(plotInd, dimPlot(2)), projectionsDS(plotInd, dimPlot(3)), 60, colors(bhvPlot+2,:), 'LineWidth', 2)
    elseif nComponents == 2
        scatter3(projectionsDS(plotInd, dimPlot(1)), projectionsDS(plotInd, dimPlot(2)), opts.frameSize * (find(plotInd)-1), 60, colors(bhvPlot+2,:), 'LineWidth', 2)
    end
    % saveas(gcf, fullfile(paths.figurePath, [titleD, '.png']), 'png')


end


%% Use a circle instead of a square
bhvPlot = find(strcmp(behaviors, 'locomotion')) - 2;

area1Cen = [-3 -9];
area2Cen = [3 -2];
% area3Cen = [-3.2 -2.8];
% area4Cen = [0 -5.5];
% area5Cen = [-.25 -2];
radius = 1.5;

% Calculate the Euclidean distance between the point and the center of the circle
distance1 = sqrt((projectionsDS(:, 1) - area1Cen(1)).^2 + (projectionsDS(:, 2) - area1Cen(2)).^2);
distance2 = sqrt((projectionsDS(:, 1) - area2Cen(1)).^2 + (projectionsDS(:, 2) - area2Cen(2)).^2);
% distance3 = sqrt((projectionsDS(:, 1) - area3Cen(1)).^2 + (projectionsDS(:, 2) - area3Cen(2)).^2);
% distance4 = sqrt((projectionsDS(:, 1) - area4Cen(1)).^2 + (projectionsDS(:, 2) - area4Cen(2)).^2);
% distance5 = sqrt((projectionsDS(:, 1) - area5Cen(1)).^2 + (projectionsDS(:, 2) - area5Cen(2)).^2);

% Determine if the point is inside the circle (including on the boundary)
isInside1 = distance1 <= radius;
isInside2 = distance2 <= radius;
% isInside3 = distance3 <= radius;
% isInside4 = distance4 <= radius;
% isInside5 = distance5 <= radius;

bhvInd = bhvID == bhvPlot;

bhvTime1 = find(isInside1 & bhvInd);
bhvTime2 = find(isInside2 & bhvInd);
% bhvTime3 = find(isInside3 & bhvInd);
% bhvTime4 = find(isInside4 & bhvInd);
% bhvTime5 = find(isInside5 & bhvInd);

n = 1:20;
% [bhvTime1(n) bhvTime2(n) bhvTime3(n) bhvTime4(n) bhvTime5(n)] .* opts.frameSize
t = seconds([bhvTime1(n) bhvTime2(n)] .* opts.frameSize);
t.Format = 'hh:mm:ss.S'

bhvInd = find(bhvPlot == dataBhv.ID);














%%                     Activities:  Compare common behavioral activities
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% --------------------------------------------
% Get stretches of activities
opts.minLength = ceil(2 / opts.frameSize); % Mininum length in sec for a sequence to count
opts.minBhvPct = 90; % Minimum percent in the sequence that has to be within the requested behaviors
opts.maxNonBhv = ceil(.5 / opts.frameSize); % Max consecutive time of a non-requested behavior allowed within the sequence
opts.minBtwnSeq = ceil(.5 / opts.frameSize); % Minimum time between qualifying sequences
opts.minDistinctBhv = 2; % Sequence must have at least this many distinct behaviors

possibleBhv = {5:12, [0:2, 13:14], 13:15};
possibleBhv = {5:12, [0:2], 13:15};
% possibleBhv = {5:12, [0:2, 13:15]};
% possibleBhv = {15};

matchSeq = {};
startIdx = {};
bhvSeq = {};
idx = {};

for i = 1 : length(possibleBhv)
    opts.possibleBhv = possibleBhv{i};
    [matchSeq{i}, startIdx{i}, bhvSeq{i}] = find_matching_sequences(bhvID, opts);
    % [matchSeq(i), startIdx{i}, bhvSeq(i)] = find_matching_sequences(bhvID, opts)
    idx{i} = randperm(length(matchSeq{i}));
end
nSample = min(cellfun(@length, idx));
% opts.possibleBhv = [5:12];
% [matchSeqGroom, startIdxGroom, bhvSeqGroom] = find_matching_sequences(bhvID, opts);
%
% opts.possibleBhv = [0 1 2 13 14 15];
% [matchSeqMove, startIdxMove, bhvSeqMove] = find_matching_sequences(bhvID, opts);

% --------------------------------------------
% Plot the coarse-grained activities 3D
% Choose which indices of activities to plot
for i = 1 : length(matchSeq)
    randIdx{i} = randperm(length(matchSeq{i}));
end
% groomIdx = randperm(length(matchSeqGroom));
% moveIdx = randperm(length(matchSeqMove));

% Plot Groom vs Move activities

% colors = colors_for_behaviors(codes);
% colorGroom = [0 0 1];
% colorMove = [0 .7 0];
colors = {[0 0 1], [0 .7 0], [1 .3 0]};
% colors = {[0 0 1], [0 .7 0]};
figure(420); clf; hold on; title(['UMAP M56, D:', num2str(nComponents), '  bin:', num2str(opts.frameSize), '  shift:', num2str(shiftSec)], 'interpreter', 'none');
figure(421); clf; hold on; title(['UMAP DS, D:', num2str(nComponents), '  bin:', num2str(opts.frameSize), '  shift:', num2str(shiftSec)], 'interpreter', 'none');
for i = 1 : nSample % length(groomIdx)

    for j = 1 : length(matchSeq)
        jStart = startIdx{j}(randIdx{j}(i));
        jStop = startIdx{j}(randIdx{j}(i)) + length(matchSeq{j}{i});

        %     startGroom = startIdxGroom(groomIdx(i));
        % stopGroom = startIdxGroom(groomIdx(i)) + length(matchSeqGroom{groomIdx(i)}) - 1;
        % startMove = startIdxMove(moveIdx(i));
        % stopMove = startIdxMove(moveIdx(i)) + length(matchSeqMove{moveIdx(i)}) - 1;

        figure(420);
        % plot3(projectionsM56(jStart:jStop, dimPlot(1)), projectionsM56(jStart:jStop, dimPlot(2)), projectionsM56(jStart:jStop, dimPlot(3)), '.:', 'Color', colors{j}, 'LineWidth', 1, 'MarkerSize', 15')
        plot3(projectionsM56(jStart:jStop, dimPlot(1)), projectionsM56(jStart:jStop, dimPlot(2)), projectionsM56(jStart:jStop, dimPlot(3)), 'o:', 'Color', colors{j}, 'LineWidth', 1, 'MarkerSize', 7')
        % plot3(projectionsM56(startGroom:stopGroom, 1), projectionsM56(startGroom:stopGroom, 2), projectionsM56(startGroom:stopGroom, 3), '.:', 'Color', colorGroom, 'LineWidth', 1, 'MarkerSize', 15')
        % plot3(projectionsM56(startMove:stopMove, 1), projectionsM56(startMove:stopMove, 2), projectionsM56(startMove:stopMove, 3), '.:', 'Color', colorMove, 'LineWidth', 1, 'MarkerSize', 15')
        % scatter3(projectionsM56(startGroom:stopGroom, 1), projectionsM56(startGroom:stopGroom, 2), projectionsM56(startGroom:stopGroom, 3), 60, colorsForPlot(startGroom:stopGroom,:), 'LineWidth', 2)
        % scatter3(projectionsM56(startMove:stopMove, 1), projectionsM56(startMove:stopMove, 2), projectionsM56(startMove:stopMove, 3), 60, colorsForPlot(startMove:stopMove,:), 'LineWidth', 2)
        grid on;
        xlabel('D1'); ylabel('D2'); zlabel('D3')

        figure(421);
        % plot3(projectionsDS(jStart:jStop, dimPlot(1)), projectionsDS(jStart:jStop, dimPlot(2)), projectionsDS(jStart:jStop, dimPlot(3)), '.:', 'Color', colors{j}, 'LineWidth', 1, 'MarkerSize', 15')
        plot3(projectionsDS(jStart:jStop, dimPlot(1)), projectionsDS(jStart:jStop, dimPlot(2)), projectionsDS(jStart:jStop, dimPlot(3)), 'o:', 'Color', colors{j}, 'LineWidth', 1, 'MarkerSize', 7')
        % plot3(projectionsDS(startGroom:stopGroom, 1), projectionsDS(startGroom:stopGroom, 2), projectionsDS(startGroom:stopGroom, 3), '.:', 'Color', colorGroom, 'LineWidth', 1, 'MarkerSize', 15')
        % plot3(projectionsDS(startMove:stopMove, 1), projectionsDS(startMove:stopMove, 2), projectionsDS(startMove:stopMove, 3), '.:', 'Color', colorMove, 'LineWidth', 1, 'MarkerSize', 15')
        xlabel('D1'); ylabel('D2'); zlabel('D3')
        grid on;

    end
end

%% Make movie of the 3-D activity projections
fig = figure(420);
figName = ['Groom_vs_Move_M56_bin_', num2str(opts.frameSize), '_shift_', num2str(shiftSec), '_3D'];
% Set up the video writer for MP4 format
videoFileName = ['E:\Projects\neuro-behavior\docs\', figName, '.mp4'];
v = VideoWriter(videoFileName, 'MPEG-4');
v.FrameRate = 10; % Lower numbers will slow down the playback
open(v);

% Define the rotation steps
azimuthSteps = 0:360;
elevationSteps = linspace(0, 360, numel(azimuthSteps));

% Rotate the view and capture frames
for i = 1:length(azimuthSteps)
    % Calculate the current azimuth and elevation
    azimuth = azimuthSteps(i);
    elevation = -30 + 30*sin(deg2rad(elevationSteps(i))); % Example elevation change

    % Update the view angle
    view(azimuth, elevation);

    % Capture the frame
    frame = getframe(fig);

    % Write the frame to the video
    writeVideo(v, frame);
end

% Close the video file
close(v);

% Display completion message
disp(['Rotating 3D plot saved to ' videoFileName]);

pause(5)

fig = figure(421);
figName = ['Groom_vs_Move_DS_bin_', num2str(opts.frameSize), '_shift_', num2str(shiftSec), '_3D'];
% Set up the video writer for MP4 format
videoFileName = ['E:\Projects\neuro-behavior\docs\', figName, '.mp4'];
v = VideoWriter(videoFileName, 'MPEG-4');
v.FrameRate = 5; % Lower numbers will slow down the playback
open(v);

% Define the rotation steps
azimuthSteps = 0:360;
elevationSteps = linspace(0, 360, numel(azimuthSteps));

% Rotate the view and capture frames
for i = 1:length(azimuthSteps)
    % Calculate the current azimuth and elevation
    azimuth = azimuthSteps(i);
    elevation = -30 + 30*sin(deg2rad(elevationSteps(i))); % Example elevation change

    % Update the view angle
    view(azimuth, elevation);

    % Capture the frame
    frame = getframe(fig);

    % Write the frame to the video
    writeVideo(v, frame);
end

% Close the video file
close(v);

% Display completion message
disp(['Rotating 3D plot saved to ' videoFileName]);


%% Plot the coarse-grained activities 2D through time
% Choose which indices of activities to plot
groomIdx = randperm(length(matchSeqGroom));
moveIdx = randperm(length(matchSeqMove));

% Plot Groom vs Move activities

% colors = colors_for_behaviors(codes);
colorGroom = [0 0 1];
colorMove = [0 .7 0];
figure(430); clf; hold on; title(['UMAP M56, D:', num2str(nComponents), '  bin:', num2str(opts.frameSize), '  shift:', num2str(shiftSec)], 'interpreter', 'none');
figure(431); clf; hold on; title(['UMAP DS, D:', num2str(nComponents), '  bin:', num2str(opts.frameSize), '  shift:', num2str(shiftSec)], 'interpreter', 'none');
for i = 1 : length(groomIdx)

    startGroom = startIdxGroom(groomIdx(i));
    stopGroom = startIdxGroom(groomIdx(i)) + length(matchSeqGroom{groomIdx(i)}) - 1;
    startMove = startIdxMove(moveIdx(i));
    stopMove = startIdxMove(moveIdx(i)) + length(matchSeqMove{moveIdx(i)}) - 1;

    figure(430);
    plot3(projectionsM56(startGroom:stopGroom, 1), projectionsM56(startGroom:stopGroom, 2), startGroom:stopGroom, '.:', 'Color', colorGroom, 'LineWidth', 1, 'MarkerSize', 15')
    plot3(projectionsM56(startMove:stopMove, 1), projectionsM56(startMove:stopMove, 2), startMove:stopMove, '.:', 'Color', colorMove, 'LineWidth', 1, 'MarkerSize', 15')
    grid on;
    xlabel('D1'); ylabel('D2'); zlabel('Time')

    figure(431);
    plot3(projectionsDS(startGroom:stopGroom, 1), projectionsDS(startGroom:stopGroom, 2), startGroom:stopGroom, '.:', 'Color', colorGroom, 'LineWidth', 1, 'MarkerSize', 15')
    plot3(projectionsDS(startMove:stopMove, 1), projectionsDS(startMove:stopMove, 2), startMove:stopMove, '.:', 'Color', colorMove, 'LineWidth', 1, 'MarkerSize', 15')
    xlabel('D1'); ylabel('D2'); zlabel('Time')
    grid on;


end

%% save rotating videos
fig = figure(430);
figName = ['Groom_vs_Move_M56_bin_', num2str(opts.frameSize), '_shift_', num2str(shiftSec), '_2DTime'];
% Set up the video writer for MP4 format
videoFileName = ['E:\Projects\neuro-behavior\docs\', figName, '.mp4'];
v = VideoWriter(videoFileName, 'MPEG-4');
v.FrameRate = 10; % Lower numbers will slow down the playback
open(v);

% Define the rotation steps
azimuthSteps = 0:360;
elevationSteps = linspace(0, 360, numel(azimuthSteps));

% Rotate the view and capture frames
for i = 1:length(azimuthSteps)
    % Calculate the current azimuth and elevation
    azimuth = azimuthSteps(i);
    elevation = -30 + 30*sin(deg2rad(elevationSteps(i))); % Example elevation change

    % Update the view angle
    view(azimuth, elevation);

    % Capture the frame
    frame = getframe(fig);

    % Write the frame to the video
    writeVideo(v, frame);
end

% Close the video file
close(v);

% Display completion message
disp(['Rotating 3D plot saved to ' videoFileName]);

pause(5)

fig = figure(431);
figName = ['Groom_vs_Move_DS_bin_', num2str(opts.frameSize), '_shift_', num2str(shiftSec), '_2DTime'];
% Set up the video writer for MP4 format
videoFileName = ['E:\Projects\neuro-behavior\docs\', figName, '.mp4'];
v = VideoWriter(videoFileName, 'MPEG-4');
v.FrameRate = 5; % Lower numbers will slow down the playback
open(v);

% Define the rotation steps
azimuthSteps = 0:360;
elevationSteps = linspace(0, 360, numel(azimuthSteps));

% Rotate the view and capture frames
for i = 1:length(azimuthSteps)
    % Calculate the current azimuth and elevation
    azimuth = azimuthSteps(i);
    elevation = -30 + 30*sin(deg2rad(elevationSteps(i))); % Example elevation change

    % Update the view angle
    view(azimuth, elevation);

    % Capture the frame
    frame = getframe(fig);

    % Write the frame to the video
    writeVideo(v, frame);
end

% Close the video file
close(v);

% Display completion message
disp(['Rotating 3D plot saved to ' videoFileName]);






%% Short vs long locomotions
shortLocStart = dataBhv.StartFrame(dataBhv.ID == 15 & dataBhv.Dur < .5);
longLocStart = dataBhv.StartFrame(dataBhv.ID == 15 & dataBhv.Dur > 1.5);

shortIdx = randperm(length(shortLocStart));


periEventTime = -.3 : opts.frameSize : .3; % seconds around onset
dataWindow = floor(periEventTime(1:end-1) / opts.frameSize); % frames around onset (remove last frame)

colorShort = [0 .6 0];
colorLong = [0 0 0];
figure(620); clf; hold on; title(['UMAP M56, D:', num2str(nComponents), '  bin:', num2str(opts.frameSize), '  shift:', num2str(shiftSec)], 'interpreter', 'none');
figure(621); clf; hold on; title(['UMAP DS, D:', num2str(nComponents), '  bin:', num2str(opts.frameSize), '  shift:', num2str(shiftSec)], 'interpreter', 'none');

for i = 1 : length(longLocStart)


    figure(620);
    plot3(projectionsM56(shortLocStart(shortIdx(i)) + dataWindow, dimPlot(1)), projectionsM56(shortLocStart(shortIdx(i)) + dataWindow, dimPlot(2)), projectionsM56(shortLocStart(shortIdx(i)) + dataWindow, dimPlot(3)), 'o:', 'Color', colorShort, 'LineWidth', 1, 'MarkerSize', 7')
    plot3(projectionsM56(longLocStart(i) + dataWindow, dimPlot(1)), projectionsM56(longLocStart(i) + dataWindow, dimPlot(2)), projectionsM56(longLocStart(i) + dataWindow, dimPlot(3)), 'o:', 'Color', colorLong, 'LineWidth', 1, 'MarkerSize', 7')
    grid on;
    xlabel('D1'); ylabel('D2'); zlabel('D3')

    figure(621);
    plot3(projectionsDS(shortLocStart(shortIdx(i)) + dataWindow, dimPlot(1)), projectionsDS(shortLocStart(shortIdx(i)) + dataWindow, dimPlot(2)), projectionsDS(shortLocStart(shortIdx(i)) + dataWindow, dimPlot(3)), 'o:', 'Color', colorShort, 'LineWidth', 1, 'MarkerSize', 7')
    plot3(projectionsDS(longLocStart(i) + dataWindow, dimPlot(1)), projectionsDS(longLocStart(i) + dataWindow, dimPlot(2)), projectionsDS(longLocStart(i) + dataWindow, dimPlot(3)), 'o:', 'Color', colorLong, 'LineWidth', 1, 'MarkerSize', 7')
    xlabel('D1'); ylabel('D2'); zlabel('D3')
    grid on;

end


%% Single behaviors - check when they are in different umap spaces