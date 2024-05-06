%%                     Compare neuro-behavior in UMAP spaces
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
cd 'E:/Projects/toolboxes/umapFileExchange (4.4)/umap/'

opts = neuro_behavior_options;
opts.minActTime = .16;
opts.collectStart = 0 * 60; % seconds
opts.collectFor = 1.5 * 60 * 60; % seconds
opts.frameSize = .1;

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
rng(1);
% rng('shuffle');
% randSub = randperm(length(idDS), length(idM56));
% idInd = idDS(randSub);
[projDS, ~, ~, ~] = run_umap(dataMat(:, idInd), 'n_components', nComponents, 'randomize', true);
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


% figure(221+i); clf; hold on;
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
%






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
% behaviorsPlot = {'contra_itch'};
behaviorsPlot = {'dive_scrunch'};


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


%% --------------------------------------------
% Compare, for a single behavior, the frame before and after behavior start
% time (frame)
dimPlot = [1 2 3];

behaviorsPlot = {'investigate_1', 'head_groom'};
behaviorsPlot = {'contra_itch', 'paw_groom'};
behaviorsPlot = {'locomotion', 'contra_orient', 'ipsi_orient'};
behaviorsPlot = {'contra_itch', 'rear'};
behaviorsPlot = {'investigate_2'};
% behaviorsPlot = {'paw_groom'};
behaviorsPlot = {'locomotion'};
behaviorsPlot = {'contra_itch'};


colors = [0 0 0; 1 0 0];
frameShift = [0 opts.frameSize];
periEventTime = -opts.frameSize : opts.frameSize : 0; % seconds around onset

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

for k = 1 : length(frameShift)
    periEventTime = periEventTime + frameShift(k);
    dataWindow = floor(periEventTime(1:end-1) / opts.frameSize); % frames around onset (remove last frame)


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
            scatter3(projectionsM56(transitionsInd, dimPlot(1)), projectionsM56(transitionsInd, dimPlot(2)), projectionsM56(transitionsInd, dimPlot(3)), 60, colors(k,:), 'LineWidth', 2)
        elseif nComponents == 2
            scatter3(projectionsM56(transitionsInd, dimPlot(1)), projectionsM56(transitionsInd, dimPlot(2)), opts.frameSize * (transitionsInd-1), 60, colors(k,:), 'LineWidth', 2)
            % plot3(projectionsM56(transitionsInd, dimPlot(1)), projectionsM56(transitionsInd, dimPlot(2)), opts.frameSize * (transitionsInd-1), 60, '-', 'color', [.6 .6 .6])
        end

        figure(231)
        if nComponents > 2
            scatter3(projectionsDS(transitionsInd, dimPlot(1)), projectionsDS(transitionsInd, dimPlot(2)), projectionsDS(transitionsInd, dimPlot(3)), 60, colors(k,:), 'LineWidth', 2)
        elseif nComponents == 2
            scatter3(projectionsDS(transitionsInd, dimPlot(1)), projectionsDS(transitionsInd, dimPlot(2)), opts.frameSize * (transitionsInd-1), 60, colors(k,:), 'LineWidth', 2)
            % plot3(projectionsDS(transitionsInd, dimPlot(1)), projectionsDS(transitionsInd, dimPlot(2)), opts.frameSize * (transitionsInd-1), '-', 'color', [.6 .6 .6])
        end
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



%% Choose which area to select data from (to analyze clusters in that area, and/or to project same time points in another area
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


%% Use HDBSCAN to select data points for analaysis. Play with this until you're happy with the groups
nDimUse = 3; % use the first nDimUse dimensions for HDBSCAN

% Classify using HDBSCAN
clusterer = HDBSCAN(projSelect(transitionsInd, 1:nDimUse));
clusterer.minpts = 2; %tends to govern cluster number  %was 3? with all neurons
clusterer.minclustsize = 20; %governs accuracy  %was 4? with all neurons

clusterer.fit_model(); 			% trains a cluster hierarchy
clusterer.get_best_clusters(); 	% finds the optimal "flat" clustering scheme
clusterer.get_membership();		% assigns cluster labels to the points in X
figure(823); 
h = clusterer.plot_clusters();
% Change appearance if you want
% h.Marker = 'o'
            % set( h.Parent,'tickdir','out','box','off','color','k','xcolor','w','ycolor','w' );

% title(['umap HDBSCAN ,' area, ' binSize = ', num2str(binSize)])
% saveas(gcf, fullfile(paths.figurePath, ['umap HDBSCAN ,' area, ' binsize ', num2str(binSize), '.png']), 'png')

%% HDBSCAN defined some groups. Now select those group data from the umap projections
nGroup = max(clusterer.labels);
bhvFrame = cell(nGroup, 1);
plotTime = -3 : opts.frameSize : 3; % seconds around onset (to plot PSTHs below)
plotFrame = round(plotTime(1:end-1) / opts.frameSize);
matSelect = cell(nGroup, 1);

for i = 1 : nGroup

        % Choose only behavior of interest
    bhvInd = zeros(length(projSelect), 1);
    bhvInd(transitionsInd) = 1;

        bhvFrame{i} = transitionsInd(clusterer.labels == i);
    % Get rid of instances too close to beginning/end (to enable plotting
    % them)
    bhvFrame{i}((bhvFrame{i} < -plotFrame(1)) | bhvFrame{i} > size(dataMat, 1) - plotFrame(end)) = [];

        % Collect the neural data peri-bout start
    nBout = length(bhvFrame{i});
    iMatSelect = zeros(length(idSelect), length(plotFrame), nBout);
    for n = 1 : nBout
        iMatSelect(:, :, n) = dataMat((bhvFrame{i}(n) - dataWindow) + plotFrame, idSelect)';
    end
    matSelect{i} = iMatSelect;

end



%% Alt: Do it by hand.  Use a circle or sphere to select data points for analysis (instead of HDBSCAN)
clear center radius
% center{1} = [-.25 -.5];
% center{2} = [-4.5 1];
% center{3} = [.5 4];
center{1} = [2.25 1 7];
center{2} = [2 -.7 3.5];
center{3} = [0 1.5 5.7];
radius{1} = 1;
radius{2} = 1;
radius{3} = 1;
% center{1} = [-2.5 -8];
% center{2} = [0 0];
% center{3} = [1 -6];
% radius{1} = 2;
% radius{2} = 2;
% radius{3} = 2;

nGroup = length(center);
colors = three_color_heatmap([1 0 0],[0 .7 0], [0 0 1], 3);

bhvFrame = cell(nGroup, 1);
prevBhv = cell(nGroup, 1);

% Number of points to define the circle
theta = linspace(0, 2*pi, 100);  % Generate 100 points along the circle

plotTime = -3 : opts.frameSize : 3; % seconds around onset (to plot PSTHs below)
plotFrame = round(plotTime(1:end-1) / opts.frameSize);
matSelect = cell(nGroup, 1);

for i = 1 : nGroup
    if strcmp(selectFrom, 'M56')
        figure(230)
    else
        figure(231)
    end

    if length(center{1}) == 2
    % Parametric equation of the circle
    x = center{i}(1) + radius{i} * cos(theta);
    y = center{i}(2) + radius{i} * sin(theta);
    plot(x, y, 'color', colors(i,:), 'lineWidth', 4);  % Plot the circle with a blue line

    % Calculate the Euclidean distance between the point and the center of the circle
    iDistance = sqrt((projSelect(:, 1) - center{i}(1)).^2 + (projSelect(:, 2) - center{i}(2)).^2);
    elseif length(center{1}) == 3
            [x, y, z] = sphere(50);
        x = x * radius{i} + center{i}(1);  % Scale and translate the sphere
        y = y * radius{i} + center{i}(2);  % Scale and translate the sphere
        z = z * radius{i} + center{i}(3);  % Scale and translate the sphere

            surf(x, y, z, 'FaceColor', colors(i,:), 'FaceAlpha', 0.2, 'EdgeColor', 'none');  % Translucent sphere

    % Calculate the Euclidean distance between the point and the center of the circle
    iDistance = sqrt((projSelect(:, 1) - center{i}(1)).^2 + (projSelect(:, 2) - center{i}(2)).^2 + (projSelect(:, 3) - center{i}(3)).^2);
    end

    % Determine if the point is inside the circle (including on the boundary)
    isInside = iDistance <= radius{i};

    % Choose only behavior of interest
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
for i = 1 : nGroup
    [uniqueElements, ~, idx] = unique(prevBhv{i});
    counts = accumarray(idx, 1);

    % Create the bar graph
    subplot(1, nGroup, i);  % Open a new figure window
    bar(uniqueElements, counts, 'FaceColor', colors(i,:));  % Create a bar plot
end


%% Plot PSTHS of the various groups
figure(3235);

allMat = [];
for i = 1 : nGroup
    allMat = cat(3, allMat, matSelect{i});
end
meanMat = mean(allMat, 3);
meanWindow = mean(meanMat, 2);
stdWindow = std(meanMat, [], 2);

zMat = cell(nGroup, 1);
zMeanMat = cell(nGroup, 1);
for i = 1 : nGroup

    % z-score the psth data
    zMat{i} = (matSelect{i} - meanWindow) ./ stdWindow;
    zMeanMat{i} = mean(zMat{i}, 3);
    subplot(1, nGroup, i)
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














%%                  RUN UMAP / HDBSCAN clusters in GPFA
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Build a set of neural matrices for training gpfa

% Go through the data and get all the relevant indices in dataBhv
nTrial = min(cellfun(@length, bhvFrame)); % Use same amount of trials per "condition"
dataBhvInd = cell(length(bhvFrame), 1);
for i = 1 : length(bhvFrame)
    bhvFrame{i} = bhvFrame{i}(randperm(length(bhvFrame{i}), nTrial));
    dataBhvInd{i} = zeros(nTrial, 1);
    for j = 1 : nTrial
        % Find the dataBhv index for this particular start time
        dataBhvInd{i}(j) = find(strcmp(dataBhv.Name, behaviorsPlot) & dataBhv.StartFrame == (bhvFrame{i}(j)+1));
    end
end

%% Get a new dataMat and dataBvh for gpfa
opts.frameSize = .001;
get_standard_data
cd('E:/Projects/toolboxes/gpfa_v0203/')
startup

%%
clear dat
preTime = .2;
postTime = .2;
periEventTime = -preTime : opts.frameSize : postTime; % seconds around onset
dataWindow = floor(periEventTime(1:end-1) / opts.frameSize); % frames around onset (remove last frame)

for i = 1 : length(bhvFrame)
    for j = 1 : nTrial
        % Find the dataBhv index for this particular start time
        jStartFrame = floor(dataBhv.StartTime(dataBhvInd{i}(j)) / opts.frameSize);
        trialIdx = (i-1) * nTrial + j;
        dat(trialIdx).trialId = i;
        dat(trialIdx).spikes = dataMat(jStartFrame + dataWindow, idSelect)';
    end
end


%%
method = 'gpfa';
runIdx = 1;
binWidth = 20; % ms
startInd = ceil(1000* preTime / binWidth); % When the behavior starts (first frame)

% Select number of latent dimensions
xDim = 8;
kernSD = 30;

% Extract neural trajectories
result = neuralTraj(runIdx, dat, 'method', method, 'xDim', xDim,...
    'kernSDList', kernSD, 'seglength', 50, 'binWidth', binWidth);
% NOTE: This function does most of the heavy lifting.

% Orthonormalize neural trajectories
[estParams, seqTrain] = postprocess(result, 'kernSD', kernSD);
% NOTE: The importance of orthnormalization is described on
%       pp.621-622 of Yu et al., J Neurophysiol, 2009.



%% Plot the gpfa fits
figure(430); clf; grid on; hold on
xlabel('D1'); ylabel('D2'); zlabel('D3')
for i = 1 : length(seqTrain)
    trialID = seqTrain(i).trialId;
    plot3(seqTrain(i).xorth(1,:), seqTrain(i).xorth(2,:), seqTrain(i).xorth(3,:), '.-', 'Color', colors(trialID,:), 'LineWidth', 2, 'MarkerSize', 10')
    scatter3(seqTrain(i).xorth(1,1), seqTrain(i).xorth(2,1), seqTrain(i).xorth(3,1), 40, colors(trialID,:), 'filled')
    scatter3(seqTrain(i).xorth(1,startInd), seqTrain(i).xorth(2,startInd), seqTrain(i).xorth(3,startInd), 100, colors(trialID,:), 'filled')
end






















%% Plot all behaviors in light gray, full time of a behavior in black, and transitions into that behavior in green (or some other color)

behaviorsPlot = {'locomotion'};
behaviorsPlot = {'contra_itch'};
% behaviorsPlot = {'investigate_2'};
behaviorsPlot = {'face_groom_2'};
for i = 1 : length(behaviors)
    behaviorsPlot = behaviors(i);

    % --------------------------------------------
    % Plot FULL TIME OF ALL BEHAVIORS - in light gray, sampling every X
    dimPlot = [1 2 3];

    colorsForPlot = [.8 .8 .8];
    colorsForPlot = [.5 .5 .5];
    alphaVal = .3;
    propSample = .3; % subsample the data to keep the plot clean (but still sample the whole space)
    randInd = randperm(length(bhvID), floor(length(bhvID) * propSample));

    figure(520); clf; hold on;
    titleM = [behaviors{i}, '  UMAP M56 ', num2str(nComponents), 'D,  bin = ', num2str(opts.frameSize), ' shift = ', num2str(shiftSec)];
    title(titleM, 'interpreter', 'none')
    if nComponents > 2
        scatter3(projectionsM56(randInd, dimPlot(1)), projectionsM56(randInd, dimPlot(2)), projectionsM56(randInd, dimPlot(3)), 60, colorsForPlot, 'LineWidth', 1, 'MarkerEdgeAlpha', alphaVal)
    elseif nComponents == 2
        scatter(projectionsM56(randInd, dimPlot(1)), projectionsM56(randInd, dimPlot(2)), 60, colorsForPlot, 'LineWidth', 1, 'MarkerEdgeAlpha', alphaVal)
    end
    grid on;
    xlabel('D1'); ylabel('D2'); zlabel('D3')
    % saveas(gcf, fullfile(paths.figurePath, [titleM, '.png']), 'png')


    % figure(221+i); clf; hold on;
    figure(521); clf; hold on;
    titleD = [behaviors{i}, '  UMAP DS ', num2str(nComponents), 'D, bin = ', num2str(opts.frameSize), ' shift = ', num2str(shiftSec)];
    title(titleD, 'interpreter', 'none')
    if nComponents > 2
        scatter3(projectionsDS(randInd, dimPlot(1)), projectionsDS(randInd, dimPlot(2)), projectionsDS(randInd, dimPlot(3)), 60, colorsForPlot, 'LineWidth', 1, 'MarkerEdgeAlpha', alphaVal)
    elseif nComponents == 2
        scatter(projectionsDS(randInd, dimPlot(1)), projectionsDS(randInd, dimPlot(2)), 60, colorsForPlot, 'LineWidth', 1, 'MarkerEdgeAlpha', alphaVal)
    end
    grid on;
    xlabel('D1'); ylabel('D2'); zlabel('D3')
    % saveas(gcf, fullfile(paths.figurePath, [titleD, '.png']), 'png')
    %


    %

    % --------------------------------------------
    % Plot FULL TIME OF Single behavior
    dimPlot = [1 2 3];

    colors = colors_for_behaviors(codes);
    bhvColor = find(strcmp(behaviors, behaviorsPlot));
    fullColor = [0 0 0];

    bhvPlot = find(strcmp(behaviors, behaviorsPlot)) - 2;
    allInd = bhvID == bhvPlot; % all labeled target behaviors

    figure(520);
    if nComponents > 2
        scatter3(projectionsM56(allInd, dimPlot(1)), projectionsM56(allInd, dimPlot(2)), projectionsM56(allInd, dimPlot(3)), 60, fullColor, 'LineWidth', 2)
    elseif nComponents == 2
        scatter(projectionsM56(allInd, dimPlot(1)), projectionsM56(allInd, dimPlot(2)), 60, fullColor, 'LineWidth', 2)
    end
    % saveas(gcf, fullfile(paths.figurePath, [titleM, '.png']), 'png')


    figure(521);
    if nComponents > 2
        scatter3(projectionsDS(allInd, dimPlot(1)), projectionsDS(allInd, dimPlot(2)), projectionsDS(allInd, dimPlot(3)), 60, fullColor, 'LineWidth', 2)
    elseif nComponents == 2
        scatter(projectionsDS(allInd, dimPlot(1)), projectionsDS(allInd, dimPlot(2)), 60, fullColor, 'LineWidth', 2)
    end
    % saveas(gcf, fullfile(paths.figurePath, [titleD, '.png']), 'png')


    % --------------------------------------------
    % Plot just the TRANSITIONS with individual behavior labels
    dimPlot = [1 2 3];

    startColor = [0 1 0];
    periEventTime = -opts.frameSize : opts.frameSize : 0; % seconds around onset
    dataWindow = floor(periEventTime(1:end-1) / opts.frameSize); % frames around onset (remove last frame)


    firstInd = 1 + find(diff(allInd) == 1); % first frames of all target behaviors

    transitionsInd = zeros(length(dataWindow) * length(firstInd), 1);
    for j = 1 : length(firstInd)
        % Calculate the start index in the expanded array
        startIndex = (j-1) * length(dataWindow) + 1;

        % Add dataWindow to the current element of firstInd and store it in the correct position
        transitionsInd(startIndex:startIndex + length(dataWindow) - 1) = firstInd(j) + dataWindow;
    end

    figure(520)
    if nComponents > 2
        scatter3(projectionsM56(transitionsInd, dimPlot(1)), projectionsM56(transitionsInd, dimPlot(2)), projectionsM56(transitionsInd, dimPlot(3)), 60, startColor, 'LineWidth', 2)
    elseif nComponents == 2
        scatter3(projectionsM56(transitionsInd, dimPlot(1)), projectionsM56(transitionsInd, dimPlot(2)), opts.frameSize * (transitionsInd-1), 60, startColor, 'LineWidth', 2)
        % plot3(projectionsM56(transitionsInd, dimPlot(1)), projectionsM56(transitionsInd, dimPlot(2)), opts.frameSize * (transitionsInd-1), 60, '-', 'color', [.6 .6 .6])
        % scatter(projectionsM56(transitionsInd, dimPlot(1)), projectionsM56(transitionsInd, dimPlot(2)), 60, colors(bhvPlot+2,:), 'LineWidth', 2)
    end

    figure(521)
    if nComponents > 2
        scatter3(projectionsDS(transitionsInd, dimPlot(1)), projectionsDS(transitionsInd, dimPlot(2)), projectionsDS(transitionsInd, dimPlot(3)), 60, startColor, 'LineWidth', 2)
    elseif nComponents == 2
        scatter3(projectionsDS(transitionsInd, dimPlot(1)), projectionsDS(transitionsInd, dimPlot(2)), opts.frameSize * (transitionsInd-1), 60, startColor, 'LineWidth', 2)
        % plot3(projectionsDS(transitionsInd, dimPlot(1)), projectionsDS(transitionsInd, dimPlot(2)), opts.frameSize * (transitionsInd-1), '-', 'color', [.6 .6 .6])
        % scatter(projectionsDS(transitionsInd, dimPlot(1)), projectionsDS(transitionsInd, dimPlot(2)), 60, colors(bhvPlot+2,:), 'LineWidth', 2)
    end

end

















