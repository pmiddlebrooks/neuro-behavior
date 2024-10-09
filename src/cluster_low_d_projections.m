% cluster_low_d_projections

%%
[projSelect, ~, idx, ~] = run_umap(dataMat(:, idSelect), 'n_components', iDim, 'randomize', false);


%% dbscan

% To determine epsilon and minpts, see matlab help
% https://www.mathworks.com/help/stats/dbscan-clustering.html

% minpts
% To select a value for minpts, consider a value greater than or equal to one plus the number of dimensions of the input data [1]. For example, for an n-by-p matrix X, set the value of 'minpts' greater than or equal to p+1.
% [1] Ester, M., H.-P. Kriegel, J. Sander, and X. Xiaowei. “A density-based algorithm for discovering clusters in large spatial databases with noise.” In Proceedings of the Second International Conference on Knowledge Discovery in Databases and Data Mining, 226-231. Portland, OR: AAAI Press, 1996.
minpts = size(projSelect, 2) + 1; % Default is 5
minpts = 5; % Default is 5


% epsilon
kD = pdist2(projSelect,projSelect,'euc','Smallest',minpts);
% Plot the k-distance graph.
plot(sort(kD(end,:)));
title('k-distance graph')
xlabel('Points sorted with 50th nearest distances')
ylabel('50th nearest distances')
grid
%%
epsilon = .6; % Default is 0.6
% epsilon = 1; % Default is 0.6


%%
idx = dbscan(projSelect, epsilon, minpts);

clusters = unique(idx)
nCluster = length(clusters)
% scatter3


% Plot results
    fig = figure(916); clf
    set(fig, 'Position', monitorTwo);
    [ax, pos] = tight_subplot(1,2, [.08 .02], .1);

    dim1 = 1;
dim2 = 2;
% plot dbscan assignments
axes(ax(1))
gscatter(projSelect(:,dim1),projSelect(:,dim2),idx, hsv(length(clusters)));
% compare with behavior labels
axes(ax(2))
scatter(projSelect(:,dim1),projSelect(:,dim2), 50, colorsForPlot, '.'); %, 'filled', '.');















%% hdbscan

% Classify using HDBSCAN
clusterer = HDBSCAN(projSelect(:,1:8)); % creates a class
clusterer.minpts = 4; %size(projSelect, 2) + 1; %tends to govern cluster number  %was 3? with all neurons
clusterer.minclustsize = 30; %governs accuracy  %was 4? with all neurons
clusterer.minClustNum = 5;
clusterer.fit_model(); 			% trains a cluster hierarchy
clusterer.get_best_clusters(); 	% finds the optimal "flat" clustering scheme
clusterer.get_membership();		% assigns cluster labels to the points in X
figure(828); clusterer.plot_clusters();
% title(['t-SNE ' area, ' binSize = ', num2str(binSize)])
% saveas(gcf, fullfile(paths.figurePath, ['t-sne HDBSCAN ' area, ' binsize ', num2str(binSize), '.png']), 'png')

unique(clusterer.labels)
