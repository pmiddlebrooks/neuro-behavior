%% Temporal Mapper on Session Spiking Data
% Loads a session of spiking data (spike times and clusters) from one of the
% session types (spontaneous, reach, hong, schall), restricts to a user-defined
% time window, and runs the Temporal Mapper toolbox for topological data analysis.
%
% Data loading follows data_prep/load_spike_times.m. Binning uses
% bin_spikes_with_overlap.m (overlapping or non-overlapping by stepSize).
%
% Prerequisites:
%   - sessionType and sessionName set in workspace (run choose_task_and_session.m
%     first, or set them manually).
%   - Temporal Mapper toolbox at: .../toolboxes/tmapper2 (relative to repo).
%
% User-defined:
%   - timeWindowSec: [tStart, tEnd] in seconds to analyze
%   - binSize, stepSize: window length and step (seconds); stepSize < binSize for overlap
%   - areasToUse: cell of area names to include (e.g. {'M23','M56'}); leave [] for all
%   - usePca: if true, X = first nDim PCs per area concatenated; if false, X = z-scored rates
%   - nDim: number of PCA dimensions per area when usePca is true
%   - tmapper params: k, d, texclude, maxdistprct, maxdist (optional)


% k (number of nearest neighbors)
% Meaning: For each time point, the algorithm adds edges to its k nearest neighbors in state space (after excluding the temporal neighborhood; see texclude).
% Effect:
% Larger k → more edges per node → denser recurrence graph, more "similar" states connected, smoother flow.
% Smaller k → sparser graph, only the closest states connect → more selective, can break up the graph or highlight strong recurrences.

% d (compression / filter threshold)
% Meaning: In filtergraph, original nodes are merged into one simplified node if the shortest path length between them (in both directions when reciprocal is true) is &lt; d. So d is a "path-length" threshold for collapsing the graph.
% Effect:
% Larger d → more nodes merged → fewer nodes in the simplified graph, stronger compression, coarser summary of dynamics.
% Smaller d → less merging → more nodes in the simplified graph, finer resolution, more detail in the transition structure.

% texclude (timeExcludeRange)
% Meaning: For each time index, the next texclude time points are treated as "temporal neighborhood" and cannot be chosen as k-NN (spatial) neighbors. So recurrence is only between states that are close in state space but not necessarily close in time.
% Effect:
% Larger texclude → stricter separation of "time" vs "recurrence" → only recurrences that are well separated in time; fewer edges, less trivial recurrence from nearby time.
% Smaller texclude → more recurrence can come from nearby time points.

% maxdistprct (percentile cap on neighbor distance)
% Meaning: An edge is allowed only if the distance is below the maxdistprct percentile of all pairwise distances. The code uses the minimum of this percentile and maxdist as the final threshold.
% Effect:
% Lower maxdistprct (e.g. 80) → only relatively close neighbors can be edges → sparser, stricter recurrence.
% Higher maxdistprct (e.g. 95 or 100) → more distant neighbors allowed → denser graph.

% maxdist (absolute distance cap)
% Meaning: No edge is allowed if the distance is above maxdist. The effective threshold is min( percentile(D, maxdistprct), maxdist).
% Effect:
% Smaller maxdist → stricter, fewer edges.
% Larger maxdist → less restrictive; with maxdistprct you can still cap by percentile.

% Summary for your settings (k=20, d=30, texclude=30):
% k=20 gives a fairly dense k-NN graph (20 neighbors per time point).
% d=30 merges nodes within path length 30, so you get a strongly compressed summary graph.
% texclude=30 means only recurrences at least 30 time bins apart in time are counted, which helps avoid trivial recurrence from consecutive or nearby bins.

%% ===== Paths =====
basePath = fileparts(mfilename('fullpath'));
srcPath = fullfile(basePath, '..');
tmapperRoot = fullfile(fileparts(srcPath), '..', 'toolboxes', 'tmapper2');
tmapperTools = fullfile(tmapperRoot, 'tmapper_tools');
dataPrepPath = fullfile(srcPath, 'data_prep');
swUtilsPath = fullfile(srcPath, 'sliding_window_prep', 'utils');
reachTaskPath = fullfile(srcPath, 'reach_task');
figureToolsPath = fullfile(fileparts(srcPath), '..', 'figure_tools');

addpath('/Users/paulmiddlebrooks/Projects/toolboxes/tmapper2')

addpath(srcPath);   % for bin_spikes_with_overlap.m, load_data.m, colors_for_behaviors.m
addpath(tmapperTools);
if exist(dataPrepPath, 'dir'), addpath(dataPrepPath); end
if exist(swUtilsPath, 'dir'), addpath(swUtilsPath); end
if exist(reachTaskPath, 'dir'), addpath(reachTaskPath); end
if exist(figureToolsPath, 'dir'), addpath(figureToolsPath); end

paths = get_paths;
opts = neuro_behavior_options;
opts.firingRateCheckTime = 3 * 60;
opts.minFiringRate = 0.5;
opts.maxFiringRate = 100;

%% ===== User: session and time window =====
% Set session via choose_task_and_session.m or here:
%   sessionType = 'spontaneous'; sessionName = 'ey042822';
  sessionType = 'reach';       sessionName = 'Y4_06-Oct-2023 14_14_53_NeuroBeh';
%   sessionType = 'schall';     sessionName = 'jp124n04';
%   sessionType = 'hong';       sessionName = '';  % Hong uses fixed files

if ~exist('sessionType', 'var') || ~exist('sessionName', 'var')
    error('Define sessionType and sessionName (e.g. run choose_task_and_session.m first).');
end

% Time window to analyze (seconds)
timeWindowSec = [300, 600];   % e.g. 5 minutes
% Bin size (window length) and step size for state matrix (seconds).
% If stepSize omitted or stepSize == binSize, uses non-overlapping bins.
% If stepSize < binSize, uses overlapping bins (more time points).
binSize = 0.05;
stepSize = 0.05;             % set < binSize for overlapping bins (e.g. 0.05)
% Areas to include: cell of area names (e.g. {'M23', 'M56'}, {'FEF'}, {'S1', 'SC'}).
% Leave empty [] to use all areas.
areasToUse = {'M23', 'M56', 'DS', 'VS'};            % e.g. {'M23', 'M56'} for reach/spontaneous; {'FEF'} for schall; {'S1','SC'} for hong
areasToUse = {'M56'};            % e.g. {'M23', 'M56'} for reach/spontaneous; {'FEF'} for schall; {'S1','SC'} for hong
% areasToUse = {'DS'};            % e.g. {'M23', 'M56'} for reach/spontaneous; {'FEF'} for schall; {'S1','SC'} for hong
% areasToUse = {'FEF'};            % e.g. {'M23', 'M56'} for reach/spontaneous; {'FEF'} for schall; {'S1','SC'} for hong
% PCA: if true and multiple areas, use first nDim PCs per area concatenated as X; if single area, first nDim PCs.
% If false, X is z-scored binned rates (no PCA).
usePca = false;
nDim = 5;   % number of PCA dimensions per area when usePca is true

%% ===== Load spike data =====
opts.collectStart = timeWindowSec(1);
opts.collectEnd = timeWindowSec(2);       % load full session; we crop to timeWindowSec below
% if strcmp(sessionType, 'reach') || strcmp(sessionType, 'hong')
%     opts.collectEnd = [];
% end

fprintf('Loading %s session: %s\n', sessionType, sessionName);
spikeData = load_spike_times(sessionType, paths, sessionName, opts);

% Restrict to user time window
tStart = timeWindowSec(1);
tEnd = timeWindowSec(2);

mask = spikeData.spikeTimes >= tStart & spikeData.spikeTimes < tEnd;
spikeTimesWin = spikeData.spikeTimes(mask);
spikeClustersWin = spikeData.spikeClusters(mask);
neuronIDs = spikeData.neuronIDs;
neuronAreas = spikeData.neuronAreas;

% Restrict to selected areas (if specified)
if ~isempty(areasToUse)
    if ischar(areasToUse) || isstring(areasToUse)
        areasToUse = cellstr(areasToUse);
    end
    areaMask = ismember(neuronAreas, areasToUse);
    neuronIDs = neuronIDs(areaMask);
    neuronAreas = neuronAreas(areaMask);
    if isempty(neuronIDs)
        error('No neurons in selected areas %s. Available: %s.', ...
            strjoin(areasToUse, ', '), strjoin(spikeData.areaLabelsUnique, ', '));
    end
    fprintf('Using %d neurons from areas: %s\n', length(neuronIDs), strjoin(areasToUse, ', '));
else
    fprintf('Using all %d neurons (areas: %s)\n', length(neuronIDs), strjoin(spikeData.areaLabelsUnique, ', '));
end

%% ===== Bin spikes into state matrix =====
timeRange = [tStart, tEnd];
if exist('stepSize', 'var') && stepSize < binSize && stepSize > 0
    [dataMat, t] = bin_spikes_with_overlap(spikeTimesWin, spikeClustersWin, neuronIDs, timeRange, binSize, stepSize);
else
    [dataMat, t] = bin_spikes_with_overlap(spikeTimesWin, spikeClustersWin, neuronIDs, timeRange, binSize);
end
numBins = size(dataMat, 1);
numNeurons = size(dataMat, 2);

% State matrix X for temporal mapper
if usePca
    % Per-area PCA: first nDim dimensions of each area's PCA, concatenated
    areasPresent = unique(neuronAreas);
    if iscell(areasPresent) && ~isempty(areasPresent) && ischar(areasPresent{1})
        areasPresent = areasPresent(:);
    end
    XParts = cell(length(areasPresent), 1);
    for a = 1:length(areasPresent)
        areaName = areasPresent{a};
        if iscell(areaName), areaName = areaName{1}; end
        idx = find(strcmp(neuronAreas, areaName));
        Xa = zscore(double(dataMat(:, idx)));
        [~, scoreA, ~] = pca(Xa);
        nTake = min(nDim, size(scoreA, 2));
        XParts{a} = scoreA(:, 1:nTake);
    end
    X = [XParts{:}];
else
    X = zscore(double(dataMat));
end
%% ===== Optional: delay embedding (increase state dimension) =====
% Uncomment to embed with delay (e.g. 10 bins back)
% delayBins = 10;
% if numBins > delayBins
%     X = [X(1:end-delayBins, :)  X(delayBins+1:end, :)];
%     t = t(delayBins+1:end);
% end

%% ===== Distance matrix (recurrence structure) =====
p = 2;   % Euclidean
D = pdist2(X, X, 'minkowski', p);

%% ===== Temporal Mapper parameters =====
k = 8;  % MZ says 3<=k<=10  connections to k nearest neighbors. 
d = 4;  % MZ says  2<=d<=5  compression rate. loops below this parameter is absorbed into nodes. 
texclude = 2;       % MZ says tx<=5  temporal neighborhood excluded from recurrence (time bins)
maxdistprct = 95; % maximal distance between neighbors by percentile.
maxdist = 0.5; % maximal distance between neighbors by absolute value;
maxdistprct = 99; % maximal distance between neighbors by percentile.
maxdist = .8; % maximal distance between neighbors by absolute value;

% Color nodes by behavior labels (reach/spontaneous) or mean rate (other session types)
if strcmp(sessionType, 'reach')
    reachDataFile = fullfile(paths.reachDataPath, [sessionName, '.mat']);
    optsBhv = struct('frameSize', stepSize, 'collectStart', tStart, 'collectEnd', tEnd);
    bhvID = define_reach_bhv_labels(reachDataFile, optsBhv);
    colorvar = zeros(length(t), 1);
    for i = 1:length(t)
        binIdx = floor((t(i) - tStart) / stepSize) + 1;
        if binIdx >= 1 && binIdx <= length(bhvID)
            colorvar(i) = bhvID(binIdx);
        else
            colorvar(i) = NaN;
        end
    end
    colorLabel = 'bhv ID (reach)';
    % Map reach bhv IDs to 1-based indices and get colors from distinguishable_colors
    valid = ~isnan(colorvar);
    uniqueIds = unique(colorvar(valid));
    behaviorCmap = distinguishable_colors(length(uniqueIds));
    colorvarIdx = nan(size(colorvar));
    for i = 1:length(colorvar)
        if ~isnan(colorvar(i))
            colorvarIdx(i) = find(uniqueIds == colorvar(i), 1);
        end
    end
    colorvar = colorvarIdx;
elseif strcmp(sessionType, 'spontaneous')
    try
        pathParts = strsplit(sessionName, filesep);
        if ~isempty(pathParts{1}) && length(pathParts{1}) >= 2
            subDir = pathParts{1}(1:2);
        elseif length(sessionName) >= 2
            subDir = sessionName(1:2);
        else
            subDir = '';
        end
        optsBhv = opts;
        optsBhv.dataPath = fullfile(paths.spontaneousDataPath, subDir);
        optsBhv.sessionName = sessionName;
        optsBhv.collectStart = 0;
        optsBhv.collectEnd = max(tEnd, max(t));
        if ~isfield(optsBhv, 'fsBhv') || isempty(optsBhv.fsBhv)
            optsBhv.fsBhv = 60;
        end
        dataBhv = load_data(optsBhv, 'behavior');
        if isempty(dataBhv) || size(dataBhv, 1) < 1
            error('No behavior data');
        end
        bhvBinSize = 1 / optsBhv.fsBhv;
        nBhvBins = ceil(optsBhv.collectEnd / bhvBinSize);
        dataBhv.StartFrame = 1 + round(dataBhv.StartTime / bhvBinSize);
        bhvID = zeros(nBhvBins, 1);
        for i = 1:size(dataBhv, 1) - 1
            iInd = dataBhv.StartFrame(i) : dataBhv.StartFrame(i+1) - 1;
            if iInd(1) <= nBhvBins
                iInd = iInd(iInd <= nBhvBins);
                bhvID(iInd) = dataBhv.ID(i);
            end
        end
        iInd = dataBhv.StartFrame(end) : nBhvBins;
        if ~isempty(iInd) && iInd(1) <= nBhvBins
            bhvID(iInd) = dataBhv.ID(end);
        end
        colorvar = zeros(length(t), 1);
        for i = 1:length(t)
            binIdx = floor(t(i) / bhvBinSize) + 1;
            if binIdx >= 1 && binIdx <= length(bhvID)
                colorvar(i) = bhvID(binIdx);
            else
                colorvar(i) = NaN;
            end
        end
        colorLabel = 'bhv ID (spont)';
        % Map bhv IDs to 1-based indices and get colors from colors_for_behaviors
        valid = ~isnan(colorvar);
        uniqueIds = unique(colorvar(valid));
        behaviorCmap = colors_for_behaviors(uniqueIds);
        colorvarIdx = nan(size(colorvar));
        for i = 1:length(colorvar)
            if ~isnan(colorvar(i))
                colorvarIdx(i) = find(uniqueIds == colorvar(i), 1);
            end
        end
        colorvar = colorvarIdx;
    catch
        colorvar = mean(X, 2);
        colorLabel = 'mean rate (z)';
        behaviorCmap = [];
        warning('Could not load spontaneous behavior labels; using mean rate for color.');
    end
else
    colorvar = mean(X, 2);
    colorLabel = 'mean rate (z)';
    behaviorCmap = [];
end

tidx = (1:length(t))';
disp('Computing knn graph...');
tic
[g, par] = tknndigraph(D, k, tidx, ...
    'timeExcludeRange', texclude, ...
    'maxNeighborDistPrct', maxdistprct, ...
    'maxNeighborDist', maxdist);
toc

disp('Simplifying knn graph...');
tic
[g_simp, members, nodesize, D_simp] = filtergraph(g, d, 'reciprocal', true);
toc

% ===== Plot transition network and recurrence =====
plotArgs = {'nodesizerange', [1, 10], 'colorlabel', colorLabel, ...
    'labelmethod', 'median', 'nodesizemode', 'log'};
if ~isempty(behaviorCmap)
    plotArgs = [plotArgs, {'cmap', behaviorCmap}];
end
[a1, a2, ~, ~, hg, D_geo] = plotgraphtcm(g_simp, colorvar, t, members, plotArgs{:});
areaStr = '';
if ~isempty(areasToUse)
    if iscell(areasToUse)
        areaStr = [' areas: ', strjoin(areasToUse, ', ')];
    else
        areaStr = [' areas: ', char(areasToUse)];
    end
end
pcaStr = '';
if usePca
    pcaStr = sprintf(', %d PC/area', nDim);
end
title(a1, {['Temporal Mapper: ', sessionType, ' / ', sessionName, areaStr, pcaStr]; ...
    sprintf('k=%d, d=%d, tx=%d', k, d, texclude); ...
    sprintf('window [%.1f, %.1f] s, bin=%.3f s, step=%.3f s, n=%d neurons', tStart, tEnd, binSize, stepSize, numNeurons)})

%% ===== Optional: recurrence plot =====
figure;
imagesc(t, t, D);
axis square;
xlabel('Time (s)');
ylabel('Time (s)');
title('Recurrence (L-2 distance)');
colorbar;
colormap hot;




%% Loop through variable ranges

