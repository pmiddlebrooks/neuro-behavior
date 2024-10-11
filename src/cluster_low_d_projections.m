% cluster_low_d_projections

%%
[projSelect, ~, uIdx, ~] = run_umap(dataMat(:, idSelect), 'n_components', iDim, 'randomize', false);
close

%% dbscan

% To determine epsilon and minpts, see matlab help
% https://www.mathworks.com/help/stats/dbscan-clustering.html

% minpts
% To select a value for minpts, consider a value greater than or equal to one plus the number of dimensions of the input data [1]. For example, for an n-by-p matrix X, set the value of 'minpts' greater than or equal to p+1.
% [1] Ester, M., H.-P. Kriegel, J. Sander, and X. Xiaowei. “A density-based algorithm for discovering clusters in large spatial databases with noise.” In Proceedings of the Second International Conference on Knowledge Discovery in Databases and Data Mining, 226-231. Portland, OR: AAAI Press, 1996.
minpts = size(projSelect, 2) + 1; % Default is 5
minpts = 2; % Default is 5

%%
% epsilon
kD = pdist2(projSelect,projSelect,'euc','Smallest',minpts);
% Plot the k-distance graph.
plot(sort(kD(end,:)));
title('k-distance graph')
xlabel('Points sorted with 50th nearest distances')
ylabel('50th nearest distances')
grid
%%
epsilon = .2; % Default is 0.6
% epsilon = 1; % Default is 0.6


%%
idx = dbscan(projSelect(:,1:2), epsilon, minpts);

clusters = unique(idx);
nCluster = length(clusters)
% scatter3


%% Plot results
fig = figure(916); clf
set(fig, 'Position', monitorTwo);
[ax, pos] = tight_subplot(1,2, [.08 .02], .1);

dim1 = 1;
dim2 = 2;
dim3 = 3;
% plot dbscan assignments
axes(ax(1))
gscatter(projSelect(:,dim1),projSelect(:,dim2),idx, hsv(length(clusters)));
% compare with behavior labels
axes(ax(2))
scatter(projSelect(:,dim1),projSelect(:,dim2), 50, colorsForPlot, '.'); %, 'filled', '.');

%%
fig = figure(916); clf
set(fig, 'Position', monitorTwo);
[ax, pos] = tight_subplot(1,2, [.08 .02], .1);
azimuth = 60;  % Angle for rotation around the z-axis
elevation = 20;  % Angle for elevation
% Set the viewing angle
dim1 = 1;
dim2 = 2;
dim3 = 3;



dimFit = 3:8;
minpts = 8 : 8 : 64;
epsilon = .04 : .02 : .2;
for kDim = 1 : length(dimFit)
    for iMin = 1:length(minpts)
        for jEps = 1 : length(epsilon)
            idx = dbscan(projSelect(:,1:dimFit(kDim)), epsilon(jEps), minpts(iMin));
            idx(idx == -1) = 0;
            clusters = unique(idx);
            nCluster(kDim, iMin, jEps) = length(clusters);
            pctOutlier(kDim, iMin, jEps) = sum(idx == 0) / length(idx);

            if length(clusters) > 5 && length(clusters) < 32
                % plot dbscan assignments
                axes(ax(1))
                % gscatter(projSelect(:,dim1),projSelect(:,dim2),idx, hsv(length(clusters)));
                dbscanClusterColors = hsv(length(clusters)+1);
                dbscanColors = arrayfun(@(x) dbscanClusterColors(x,:), idx+1, 'UniformOutput', false);
                dbscanColors = vertcat(dbscanColors{:}); % Convert cell array to a matrix
                scatter3(projSelect(:,dim1),projSelect(:,dim2),projSelect(:,dim3), 50, dbscanColors, '.');
                view(azimuth, elevation);
                sgtitle(sprintf('Dim: %d\t minpts: %d\t epsilon: %.2f\t nCluster: %d\t pctOutlier: %.2f', dimFit(kDim), minpts(iMin), epsilon(jEps), length(clusters), sum(idx == 0) / length(idx)))

                % compare with behavior labels
                axes(ax(2))
                scatter3(projSelect(:,dim1),projSelect(:,dim2),projSelect(:,dim3), 50, colorsForPlot, '.'); %, 'filled', '.');
                view(azimuth, elevation);
                fprintf('DBSCAN: Dim: %d\t minpts: %d\t epsilon: %.2f\t nCluster: %d\t pctOutlier: %.2f', dimFit(kDim), minpts(iMin), epsilon(jEps), length(clusters), sum(idx == 0) / length(idx))
                disp('pause')
            end
        end
    end
end















%% hdbscan
% Classify using HDBSCAN
% Fit a model on a subset of the data if you have too many points
maxFrames = 20000;
if size(projSelect, 1) <= maxFrames
    clusterer = HDBSCAN(projSelect); % creates a class
else
    fitFrames = randperm(size(projSelect, 1), maxFrames);
    clusterer = HDBSCAN(projSelect(fitFrames,:)); % creates a class
end
clusterer.minpts = 3; %size(projSelect, 2) + 1; %tends to govern cluster number  %was 3? with all neurons
clusterer.minclustsize = 50; %governs accuracy  %was 4? with all neurons
clusterer.minClustNum = 5;
clusterer.outlierThresh = .95;
clusterer.fit_model(); 		% trains a cluster hierarchy
clusterer.get_best_clusters(); 	% finds the optimal "flat" clustering scheme
clusterer.get_membership();		% assigns cluster labels to the points in X
figure(828); clusterer.plot_clusters();
% title(['t-SNE ' area, ' binSize = ', num2str(binSize)])
% saveas(gcf, fullfile(paths.figurePath, ['t-sne HDBSCAN ' area, ' binsize ', num2str(binSize), '.png']), 'png')

unique(clusterer.labels)

fprintf('%d labeled, %d unlabeled\n', sum(clusterer.labels > 0), sum(clusterer.labels == 0))
%%
if size(projSelect, 1) > maxFrames
    labels = clusterer.predict(projSelect);
end


fig = figure(916); clf
set(fig, 'Position', monitorTwo);
[ax, pos] = tight_subplot(1,2, [.08 .02], .1);
azimuth = 60;  % Angle for rotation around the z-axis
elevation = 20;  % Angle for elevation
% Set the viewing angle
dim1 = 1;
dim2 = 2;
dim3 = 3;


%%  Test a bunch of parameters in hdbscan
fig = figure(916); clf
set(fig, 'Position', monitorTwo);
[ax, pos] = tight_subplot(1,2, [.08 .02], .1);
azimuth = 60;  % Angle for rotation around the z-axis
elevation = 20;  % Angle for elevation
% Set the viewing angle
dim1 = 1;
dim2 = 2;
dim3 = 3;


dimFit = 3:8;
minpts = 3 : 20;
minClusterSize = [2 3 4 8 16 32 64];
% Fit a model on a subset of the data if you have too many points
% dimFit = 4;
% minpts = 8;
% minClusterSize = 4;

maxFrames = 20000;

nCluster = zeros(length(dimFit), length(minpts), length(minClusterSize));
pctOutlier = zeros(length(dimFit), length(minpts), length(minClusterSize));

for kDim = 1 : length(dimFit)

    if size(projSelect, 1) <= maxFrames
        fitFrames = 1 : size(projSelect, 1);
        clusterer = HDBSCAN(projSelect(:,1:dimFit(kDim))); % creates a class
    else
        fitFrames = randperm(size(projSelect, 1), maxFrames);
        clusterer = HDBSCAN(projSelect(fitFrames,1:dimFit(kDim))); % creates a class
    end

    for iMin = 1:length(minpts)

        for jMinClust = 1 : length(minClusterSize)
            fprintf('\nDim: %d\t minpts: %d\t minClust: %d\t nCluster: %d\t pctOutlier: %.2f\n', dimFit(kDim), minpts(iMin), minClusterSize(jMinClust), length(clusters), sum(idx == 0) / length(idx))

            clusterer.minpts = minpts(iMin); %size(projSelect, 2) + 1; %tends to govern cluster number  %was 3? with all neurons
            clusterer.minclustsize = minClusterSize(jMinClust); %governs accuracy  %was 4? with all neurons
            clusterer.minClustNum = 5;
            % clusterer.outlierThresh = .95;
            clusterer.fit_model(); 		% trains a cluster hierarchy
            clusterer.get_best_clusters(); 	% finds the optimal "flat" clustering scheme
            clusterer.get_membership();		% assigns cluster labels to the points in X


            idx = clusterer.labels;
            clusters = unique(idx);
            % Create a map from unique integers to their ranks
            rankMap = containers.Map(clusters, 1:length(clusters));
            % Replace each integer in the original vector with its corresponding rank
            rankedVec = arrayfun(@(x) rankMap(x), idx);

            iPctOutlier = sum(idx == 0) / length(idx);


            % Is this a fairly good clusterer??
            nCluster(kDim, iMin, jMinClust) = length(clusters);
            pctOutlier(kDim, iMin, jMinClust) = iPctOutlier;


            % plot dbscan assignments
            axes(ax(1))
            % gscatter(projSelect(:,dim1),projSelect(:,dim2),idx, hsv(length(clusters)));
            dbscanClusterColors = hsv(length(clusters));
            dbscanColors = arrayfun(@(x) dbscanClusterColors(x,:), rankedVec, 'UniformOutput', false);
            dbscanColors = vertcat(dbscanColors{:}); % Convert cell array to a matrix
            scatter3(projSelect(fitFrames,dim1),projSelect(fitFrames,dim2),projSelect(fitFrames,dim3), 50, dbscanColors, '.');
            view(azimuth, elevation);
            sgtitle(sprintf('HDBSCAN: Dim: %d\t minpts: %d\t minClust: %d\t nCluster: %d\t pctOutlier: %.2f', dimFit(kDim), minpts(iMin), minClusterSize(jMinClust), length(clusters), sum(idx == 0) / length(idx)))

            % compare with behavior labels
            axes(ax(2))
            scatter3(projSelect(:,dim1),projSelect(:,dim2),projSelect(:,dim3), 50, colorsForPlot, '.'); %, 'filled', '.');
            view(azimuth, elevation);
            if length(clusters) > 6 && length(clusters) < 30 && iPctOutlier <= 20
                disp('pause')
            end
        end
    end
end
