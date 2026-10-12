function [lfpPerArea, areaNames, opts] = load_area_lfp(opts)
% LOAD_AREA_LFP - Load per-area LFP for a session collect window
%
% Variables:
%   opts - Options with dataPath, sessionName, collectStart, collectEnd.
%          opts.fsLfp is used only for legacy lfp.txt (default 1250 Hz).
%
% Goal:
%   Read the Neuropixels file written by process_np1_lfp (lfp.mat: lfpPerArea
%   and lfpMeta, already averaged, lowpassed, and resampled). Slice it to the
%   collect window and return one column per brain area. If lfp.mat is absent,
%   fall back to raw lfp.txt, flip channels so column 1 is the surface, average
%   the historical probe pairs, and lowpass at 300 Hz.
%
% Returns:
%   lfpPerArea - [nSamples x nAreas] double, sample 1 at collectStart
%   areaNames  - Cell of area names, one per column
%   opts       - Input opts with fsLfp set to the sampling rate of the matrix

sessionFolder = fullfile(opts.dataPath, opts.sessionName);
matPath = fullfile(sessionFolder, 'lfp.mat');
txtPath = fullfile(sessionFolder, 'lfp.txt');

if isfile(matPath)
    loaded = load(matPath, 'lfpPerArea', 'lfpMeta');
    if ~isfield(loaded, 'lfpPerArea') || isempty(loaded.lfpPerArea)
        error('load_area_lfp:MissingTrace', ...
            'lfp.mat has no lfpPerArea: %s', matPath);
    end
    if ~isfield(loaded, 'lfpMeta') || ~isfield(loaded.lfpMeta, 'fsOut') ...
            || isempty(loaded.lfpMeta.fsOut)
        error('load_area_lfp:MissingFs', ...
            'lfp.mat has no lfpMeta.fsOut: %s', matPath);
    end
    opts.fsLfp = double(loaded.lfpMeta.fsOut);
    if isfield(loaded.lfpMeta, 'brainAreas') && ~isempty(loaded.lfpMeta.brainAreas)
        areaNames = cellstr(loaded.lfpMeta.brainAreas);
        areaNames = areaNames(:)';
    else
        areaNames = {'M23', 'M56', 'DS', 'VS'};
    end
    if numel(areaNames) ~= size(loaded.lfpPerArea, 2)
        error('load_area_lfp:AreaCount', ...
            'lfp.mat has %d areas in lfpMeta and %d columns in lfpPerArea.', ...
            numel(areaNames), size(loaded.lfpPerArea, 2));
    end
    lfpPerArea = window_lfp_collect(double(loaded.lfpPerArea), opts.fsLfp, opts);
    fprintf('Loaded per-area LFP from lfp.mat (%d areas, %.0f Hz)\n', ...
        numel(areaNames), opts.fsLfp);
    return;
end

if ~isfile(txtPath)
    error('load_area_lfp:MissingFile', ...
        'No LFP file in %s. Expected lfp.mat from process_np1_lfp.', sessionFolder);
end

if ~isfield(opts, 'fsLfp') || isempty(opts.fsLfp)
    opts.fsLfp = 1250;
end
lfpData = readmatrix(txtPath);
lfpData = window_lfp_collect(lfpData, opts.fsLfp, opts);
lfpData = fliplr(lfpData);
lfpPerArea = [mean(lfpData(:, [3 5]), 2), mean(lfpData(:, [9 11]), 2), ...
    mean(lfpData(:, [19 23]), 2), mean(lfpData(:, [30 34]), 2)];
lfpPerArea = lowpass(lfpPerArea, 300, opts.fsLfp);
areaNames = {'M23', 'M56', 'DS', 'VS'};
fprintf('Loaded legacy lfp.txt and averaged probe pairs (%.0f Hz)\n', opts.fsLfp);
end

function data = window_lfp_collect(data, fsLfp, opts)
% WINDOW_LFP_COLLECT - Keep samples inside opts.collectStart / collectEnd
%
% Variables:
%   data   - [nSamples x nChannels] LFP
%   fsLfp  - Sampling rate (Hz)
%   opts   - collectStart (s, default 0) and collectEnd (s; [] = last sample)
%
% Goal:
%   Match the collect window used by load_data for lfp.txt. Sample 1 of the
%   returned matrix is collectStart. An empty collectEnd keeps the rest of
%   the recording.

collectStart = 0;
if isfield(opts, 'collectStart') && ~isempty(opts.collectStart)
    collectStart = opts.collectStart;
end
nSamples = size(data, 1);
startSample = round(1 + collectStart * fsLfp);
if startSample < 1
    startSample = 1;
end
if startSample > nSamples
    error('load_area_lfp:CollectStart', ...
        'collectStart %.3f s is past the end of the LFP (%.3f s).', ...
        collectStart, nSamples / fsLfp);
end

if ~isfield(opts, 'collectEnd') || isempty(opts.collectEnd)
    endSample = nSamples;
else
    endSample = round(opts.collectEnd * fsLfp);
    if endSample > nSamples
        endSample = nSamples;
    end
end
if endSample < startSample
    error('load_area_lfp:CollectEnd', ...
        'collectEnd is before collectStart for this LFP.');
end
data = data(startSample:endSample, :);
end
