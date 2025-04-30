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
wfPath = 'E:\Projects\neuro-behavior\data\raw_ephys\animal_ag25290\112321\recording1';
% wfPath = '~/Projects/neuro-behavior/data/raw_ephys/animal_ag25290/112321/recording1';
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



%% Analyze the waveforms of the valid neurons in each brain area
for a = 1:length(areas)
    [meanWf, fwhm, ptDur] = deal([]);

    for iWave = 1 : length(wf)

        % Get the column of the neural matrix to draw from
        dataMatIdx = find(idLabels == wf(iWave).unitID);

        % Find the unit in ci and get the channel
        ciIdx = find(data.ci.cluster_id == wf(iWave).unitID);
        if ismember(wf(iWave).unitID, idLabels(idList{a}))
            iWaveIdx = find(wf(iWave).channels == data.ci.ch(ciIdx));
            iMeanWf = reshape(wf(iWave).mean_wf(1, iWaveIdx, :), 1, 62);
            meanWf = [meanWf; iMeanWf];

            % Classify wavefo
            iFwhm = compute_fwhm(iMeanWf, opts.fsSpike);
            fwhm = [fwhm; iFwhm];

            iPt = compute_peak_to_trough(iMeanWf, opts.fsSpike);
            ptDur = [ptDur; iPt];
            % figure(44); clf
            % plot(iMeanWf);
            % disp('asdf')
        end


    end

    figure(98);
    subplot(2,2,a)
    edges = .02:.01:.4;
    histogram(fwhm,edges)
    title(areas(a))
    figure(99);
    subplot(2,2,a)
    edges = .02:.05:1.2;
    histogram(ptDur,edges)
    title(areas(a))
end
figure(98);
sgtitle('Full Width Half Max')
figure(99);
sgtitle('Peak to Trough')

























function fwhm = compute_fwhm(waveform, sampleRate)
% compute_fwhm calculates the Full Width at Half Maximum (FWHM) of a spike waveform.
% waveform: 1D vector of spike waveform (in volts or arbitrary units)
% sampleRate: in Hz (samples per second), e.g., 30000
% fwhm: result in milliseconds

% Find absolute peak (most extreme point)
[~, peakIdx] = max(abs(waveform));
peakVal = waveform(peakIdx);

% Determine half-max value
halfMax = 0.5 * peakVal;

% Find waveform crossings at half-max (interpolate for precision)
crossings = [];

for i = 1:length(waveform)-1
    if (waveform(i) - halfMax) * (waveform(i+1) - halfMax) < 0
        % Linear interpolation to find more accurate crossing point
        x1 = i; x2 = i + 1;
        y1 = waveform(i); y2 = waveform(i+1);
        slope = (y2 - y1);
        interpIdx = x1 + (halfMax - y1) / slope;
        crossings(end+1) = interpIdx;
    end
end

% Keep crossings that are on each side of the peak
pre = crossings(crossings < peakIdx);
post = crossings(crossings > peakIdx);

if isempty(pre) || isempty(post)
    warning('Could not find valid FWHM crossings on both sides of the peak.');
    fwhm = NaN;
    return;
end

% Compute FWHM in milliseconds
widthInSamples = post(1) - pre(end);
fwhm = (widthInSamples / sampleRate) * 1000;  % convert to ms
end

function ptDuration = compute_peak_to_trough(waveform, sampleRate)
% compute_peak_to_trough calculates the peak-to-trough duration for a spike waveform.
% waveform: 1D vector of spike waveform (e.g., in volts)
% sampleRate: sampling rate in Hz (e.g., 30000 for 30 kHz)
% ptDuration: duration in milliseconds

% Find absolute peak (max of abs(waveform))
[~, peakIdx] = max(abs(waveform));
peakVal = waveform(peakIdx);

% Determine direction: negative peak or positive peak
if peakVal < 0
    % Trough occurs first — find subsequent peak
    postSegment = waveform(peakIdx+1:end);
    [~, relIdx] = max(postSegment);
    troughIdx = peakIdx;
    peakIdx = peakIdx + relIdx;
else
    % Peak occurs first — find subsequent trough
    postSegment = waveform(peakIdx+1:end);
    [~, relIdx] = min(postSegment);
    peakIdx = peakIdx;
    troughIdx = peakIdx + relIdx;
end

% Calculate time difference in milliseconds
ptSamples = abs(peakIdx - troughIdx);
ptDuration = (ptSamples / sampleRate) * 1000;  % convert to ms
end

