% Here's the data structure (spikeData.mat) that includes everything you should need in matlab format. 
% sp.st: spike times for every spike for every unit
% sp.clu: cluster ID# for each spike time corresponding to sp.st
% sp.spikeDepths: depth for each spike corresponding to sp.st
% sp.cids: list of cluster IDs
% sp.cgs: kilosort/phy designation of good/multi/noise units
% 
% Also included here is the table of just the single units
% (T_allUnits2.mat), which lists the cluster ID and corresponding cluster
% depth. Depth > 2000 is S1 (1250um laminar probe spanning the full S1
% cortical depth), < 2000 is SC (2 shanks, 750um span)   

%%
paths = get_paths;
% sp: spike depths spikeDepths
load(fullfile(paths.dropPath, 'hong/data', 'spikeData.mat'));

% load T_allUnits: table of single units used
load(fullfile(paths.dropPath, 'hong/data', 'T_allUnits2.mat'));

% Load T: behavior table for the session
load(fullfile(paths.dropPath, 'hong/data', 'behaviorTable.mat'));

% spike clusters, spike times
% sc = readNPY(fullfile(paths.dropPath, 'hong', 'spike_clusters.npy'));
% st = readNPY(fullfile(paths.dropPath, 'hong', 'spike_times.npy'));
% 
% cg = readtable(fullfile(paths.dropPath, 'hong', 'cluster_group.tsv'), 'FileType', 'text', 'Delimiter', '\t');

%%
opts = neuro_behavior_options;
opts.frameSize = .001;
opts.collectStart = 0;
opts.collectEnd = min(T.startTime_oe(end)+max(diff(T.startTime_oe)), max(sp.st));
opts.minFiringRate = .1;

data.sp = sp;
data.ci = T_allUnits;

[dataMat, idLabels, areaLabels, rmvNeurons] = neural_matrix_hong(data, opts);

    areas = {'S1', 'SC'};
    idS1 = find(strcmp(areaLabels, 'S1'));
    idSC = find(strcmp(areaLabels, 'SC'));
    idMatIdx = {idS1, idSC};
    idLabel = {idLabels(idS1), idLabels(idSC)};
    % Print summary
    fprintf('%d S1\n', length(idS1));
    fprintf('%d SC\n', length(idSC));
