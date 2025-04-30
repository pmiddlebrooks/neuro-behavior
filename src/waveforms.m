%%
opts = neuro_behavior_options;
opts.frameSize = .001;
opts.minFiringRate = .1;
getDataType = 'spikes';
opts.collectFor = 60 * 60;
opts.firingRateCheckTime = 5 * 60;
get_standard_data

areas = {'M23', 'M56', 'DS', 'VS'};
idList = {idM23, idM56, idDS, idVS};


%%
% wfPath = 'E:\Projects\neuro-behavior\data\raw_ephys\animal_ag25290\112321\recording1';
wfPath = '~/Projects/neuro-behavior/data/raw_ephys/animal_ag25290/112321/recording1';
load(fullfile(wfPath, 'waveforms.mat'));
wf = sp_waveforms;

getDataType = 'spikes';
animal = 'ag25290';
sessionBhv = '112321_1';
sessionNrn = '112321';
if strcmp(sessionBhv, '112321_1')
    sessionSave = '112321';
end
    nrnDataPath = strcat(paths.nrnDataPath, 'animal_',animal,'/', sessionNrn, '/');
    nrnDataPath = [nrnDataPath, 'recording1/'];
    opts.dataPath = nrnDataPath;

%%
data = load_data(opts, getDataType);


%%
idLabels(idM23)


%%

for iWave = 1 : length(sp_waveforms)

    % Get the column of the neural matrix to draw from
    dataMatIdx = find(idLabels == wv.unitID);

    % Classify waveform




end