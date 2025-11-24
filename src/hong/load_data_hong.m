%%
paths = get_paths;
% behavior table T
load(fullfile(paths.dropPath, 'hong', 'behaviorTable'));

% spike depths spikeDepths
load(fullfile(paths.dropPath, 'hong', 'spikeDepths.mat'));

% spike clusters, spike times
sc = readNPY(fullfile(paths.dropPath, 'hong', 'spike_clusters.npy'));
st = readNPY(fullfile(paths.dropPath, 'hong', 'spike_times.npy'));

cg = readtable(fullfile(paths.dropPath, 'hong', 'cluster_group.tsv'), 'FileType', 'text', 'Delimiter', '\t');

%%
opts = neuro_behavior_options;
opts.frameSize = .001;
opts.collectStart = 0;
opts.collectEnd = min(T.startTime_oe(end)+6, max(st));
opts.minFiringRate = .1;

data.spikeTimes = st;
data.spikeDepths = spikeDepths;
data.spikeClusters = sc;
data.ci = cg;

[dataMat, idLabels, areaLabels, rmvNeurons] = neural_matrix_hong(data, opts);
