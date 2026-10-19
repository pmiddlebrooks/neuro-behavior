%% LFP processing for subject ag25290
%
% Variables:
%   subjectName  - 'ag25290'
%   sessionType  - 'spontaneous'
%   brainAreas   - {'M23','M56','DS','VS'}
%   lowpassFreq  - 300 Hz, same cutoff as process_np1_lfp / process_np2_lfp
%   fsOut        - 1250 Hz output rate
%
% Goal:
%   The ag25290 lfp.mat files store every channel in a struct array lfp
%   (fields data, depth, samplerate). Depth is already surface-to-deep
%   (0 = surface), the same convention as lfp_state_script.m. Channels are
%   assigned to M23, M56, DS, and VS with those depth ranges. Each area
%   trace is the mean of the two sites nearest 1/3 and 2/3 of that range,
%   as in process_np1_lfp / process_np2_lfp. The result is low-passed at
%   300 Hz and written as lfpPerArea plus lfpMeta. The all-channel file is
%   kept as lfp_all_channels.mat.
%
% Usage:
%   lfp_processing_for_ag25290

subjectName = 'ag25290';
subjectName = 'ey4152';
sessionType = 'spontaneous';
brainAreas = {'M23', 'M56', 'DS', 'VS'};

thisDir = fileparts(mfilename('fullpath'));
addpath(thisDir);
addpath(fileparts(thisDir));
constants = np1_lfp_constants();
lowpassFreq = constants.lowpassFreq;
fsOut = constants.fsOut;

paths = get_paths();
subjectFolder = fullfile(paths.spontaneousDataPath, subjectName);
if ~isfolder(subjectFolder)
    error('lfp_processing_for_ag25290:NoSubject', ...
        'Subject folder not found: %s', subjectFolder);
end

sessionDirs = dir(subjectFolder);
nConverted = 0;
for iDir = 1:numel(sessionDirs)
    if sessionDirs(iDir).isdir && ~startsWith(sessionDirs(iDir).name, '.')
        sessionName = sessionDirs(iDir).name;
        sessionFolder = fullfile(subjectFolder, sessionName);
        didConvert = convert_ag25290_session(sessionFolder, sessionType, ...
            subjectName, sessionName, brainAreas, lowpassFreq, fsOut);
        nConverted = nConverted + didConvert;
    end
end
fprintf('Converted %d ag25290 session(s).\n', nConverted);

function didConvert = convert_ag25290_session(sessionFolder, sessionType, ...
    subjectName, sessionName, brainAreas, lowpassFreq, fsOut)
% CONVERT_AG25290_SESSION - One all-channel lfp.mat into lfpPerArea
%
% Variables:
%   sessionFolder - Folder that holds lfp.mat
%   sessionType, subjectName, sessionName - Identity stored in lfpMeta
%   brainAreas    - Area names, one per column of lfpPerArea
%   lowpassFreq   - Butterworth cutoff (Hz)
%   fsOut         - Sampling rate written to lfpMeta.fsOut
%
% Goal:
%   Average the two depth-matched channels in each area and replace lfp.mat
%   with the per-area file used by the other LFP pipelines.
%
% Returns:
%   didConvert - 1 when a file was written, 0 when the session was skipped

didConvert = 0;
matPath = fullfile(sessionFolder, 'lfp.mat');
backupPath = fullfile(sessionFolder, 'lfp_all_channels.mat');
sourcePath = all_channel_source(matPath, backupPath);
if isempty(sourcePath)
    return;
end

fprintf('\n%s\n', sessionName);
fprintf('  Reading all-channel LFP: %s\n', sourcePath);
loaded = load(sourcePath, 'lfp');
channelData = loaded.lfp;
clear loaded

[nSamples, fsIn, channelDepths] = channel_grid(channelData);
nChannels = numel(channelDepths);
areaPick = pick_area_channel_pairs(sessionFolder, brainAreas, channelDepths);
nAreas = numel(areaPick);
fprintf('  %d channels, %d samples at %.0f Hz\n', nChannels, nSamples, fsIn);

lfpPerArea = zeros(nSamples, nAreas, 'single');
for iArea = 1:nAreas
    channelPair = areaPick(iArea).channelIds;
    traceA = double(channelData(channelPair(1)).data(:));
    traceB = double(channelData(channelPair(2)).data(:));
    lfpPerArea(:, iArea) = single((traceA + traceB) / 2);
end
clear channelData traceA traceB

[bLp, aLp] = butter(4, lowpassFreq / (fsIn / 2), 'low');
for iArea = 1:nAreas
    lfpPerArea(:, iArea) = single(filtfilt(bLp, aLp, double(lfpPerArea(:, iArea))));
end
if abs(fsIn - fsOut) > 0.5
    fprintf('  Resampling %.0f Hz -> %.0f Hz\n', fsIn, fsOut);
    lfpPerArea = single(resample(double(lfpPerArea), fsOut, round(fsIn)));
end

lfpMeta = build_ag25290_meta(sessionType, subjectName, sessionName, areaPick, ...
    nChannels, nSamples, size(lfpPerArea, 1), fsIn, fsOut, lowpassFreq, matPath);

if strcmp(sourcePath, matPath) && ~isfile(backupPath)
    movefile(matPath, backupPath);
    fprintf('  Kept all-channel file at %s\n', backupPath);
end
save(matPath, 'lfpPerArea', 'lfpMeta', '-v7.3');
fprintf('  Wrote %d samples x %d areas to %s\n', size(lfpPerArea, 1), nAreas, matPath);
didConvert = 1;
end

function sourcePath = all_channel_source(matPath, backupPath)
% ALL_CHANNEL_SOURCE - Path of the struct-array LFP, if this session has one
%
% Returns:
%   sourcePath - lfp.mat or lfp_all_channels.mat, or '' when nothing to do

sourcePath = '';
if isfile(backupPath) && mat_has_variable(backupPath, 'lfp')
    sourcePath = backupPath;
    return;
end
if isfile(matPath) && mat_has_variable(matPath, 'lfp')
    sourcePath = matPath;
    return;
end
if isfile(matPath) && mat_has_variable(matPath, 'lfpPerArea')
    fprintf('Already per-area: %s\n', matPath);
end
end

function tf = mat_has_variable(matPath, variableName)
% MAT_HAS_VARIABLE - True when the MAT file lists variableName
info = whos('-file', matPath);
tf = any(strcmp({info.name}, variableName));
end

function [nSamples, fsIn, channelDepths] = channel_grid(channelData)
% CHANNEL_GRID - Shared length, sampling rate, and depth of each channel
%
% Variables:
%   channelData - Struct array, one element per channel
%
% Returns:
%   nSamples      - Samples in every channel
%   fsIn          - Sampling rate (Hz)
%   channelDepths - Depth (um) in the file's channel order

nChannels = numel(channelData);
nSamples = numel(channelData(1).data);
fsIn = double(channelData(1).samplerate);
channelDepths = zeros(1, nChannels);
for iChannel = 1:nChannels
    channelDepths(iChannel) = double(channelData(iChannel).depth);
    if numel(channelData(iChannel).data) ~= nSamples
        error('lfp_processing_for_ag25290:LengthMismatch', ...
            'Channel %d has %d samples; channel 1 has %d.', ...
            iChannel, numel(channelData(iChannel).data), nSamples);
    end
    if double(channelData(iChannel).samplerate) ~= fsIn
        error('lfp_processing_for_ag25290:RateMismatch', ...
            'Channel %d is %.0f Hz; channel 1 is %.0f Hz.', ...
            iChannel, double(channelData(iChannel).samplerate), fsIn);
    end
end
end

function areaPick = pick_area_channel_pairs(sessionFolder, brainAreas, channelDepths)
% PICK_AREA_CHANNEL_PAIRS - Two channels nearest 1/3 and 2/3 of each area
%
% Variables:
%   sessionFolder  - Used for session-specific brain_area_depths.mat
%   brainAreas     - Area names to keep, in column order
%   channelDepths  - Depth of each stored channel (um). 0 is the surface.
%
% Goal:
%   Use the same depth ranges as lfp_state_script.m (via
%   get_brain_area_depth_ranges) and the same 1/3 and 2/3 site choice as
%   select_area_lfp_channels.
%
% Returns:
%   areaPick - Struct array with areaName, depthRange, targetDepths,
%              channelIds (1-based index into the lfp struct), channelDepths

[m23, m56, cc, ds, vs, depthSource] = get_brain_area_depth_ranges(sessionFolder);
rangeMap = struct('M23', m23, 'M56', m56, 'CC', cc, 'DS', ds, 'VS', vs);
channelIds = 1:numel(channelDepths);
areaPick = struct('areaName', {}, 'depthRange', {}, 'targetDepths', {}, ...
    'channelIds', {}, 'channelDepths', {}, 'depthSource', {});

for iArea = 1:numel(brainAreas)
    areaName = brainAreas{iArea};
    if ~isfield(rangeMap, areaName)
        error('lfp_processing_for_ag25290:BadArea', ...
            'Unknown brain area "%s".', areaName);
    end
    depthRange = rangeMap.(areaName);
    inRange = channelDepths >= depthRange(1) & channelDepths <= depthRange(2);
    ids = channelIds(inRange);
    deps = channelDepths(inRange);
    if numel(ids) < 2
        error('lfp_processing_for_ag25290:TooFew', ...
            '%s [%d %d] um has %d LFP channels; need at least 2.', ...
            areaName, depthRange(1), depthRange(2), numel(ids));
    end
    span = depthRange(2) - depthRange(1);
    targetDepths = [depthRange(1) + span / 3, depthRange(1) + 2 * span / 3];
    pickedIds = zeros(1, 2);
    pickedDepths = zeros(1, 2);
    used = false(size(ids));
    for iTarget = 1:2
        dist = abs(deps - targetDepths(iTarget));
        dist(used) = inf;
        [~, idx] = min(dist);
        pickedIds(iTarget) = ids(idx);
        pickedDepths(iTarget) = deps(idx);
        used(idx) = true;
    end
    rec = struct();
    rec.areaName = areaName;
    rec.depthRange = depthRange;
    rec.targetDepths = targetDepths;
    rec.channelIds = pickedIds;
    rec.channelDepths = pickedDepths;
    rec.depthSource = depthSource;
    areaPick(end + 1) = rec; %#ok<AGROW>
    fprintf('  %s [%d %d] um  targets %.0f / %.0f  -> ch %d (%.0f um), ch %d (%.0f um)\n', ...
        areaName, depthRange(1), depthRange(2), targetDepths(1), targetDepths(2), ...
        pickedIds(1), pickedDepths(1), pickedIds(2), pickedDepths(2));
end
end

function lfpMeta = build_ag25290_meta(sessionType, subjectName, sessionName, areaPick, ...
    nChannels, nSamplesIn, nSamplesOut, fsIn, fsOut, lowpassFreq, matPath)
% BUILD_AG25290_META - lfpMeta fields used by load_area_lfp
%
% Variables:
%   areaPick - Output of pick_area_channel_pairs
%
% Goal:
%   Record the same area and sampling fields as the NP pipelines.

nAreas = numel(areaPick);
lfpMeta = struct();
lfpMeta.probeType = 'ag25290';
lfpMeta.sessionType = sessionType;
lfpMeta.subjectName = subjectName;
lfpMeta.sessionName = sessionName;
lfpMeta.brainAreas = {areaPick.areaName};
lfpMeta.depthSource = areaPick(1).depthSource;
lfpMeta.depthRanges = reshape([areaPick.depthRange], 2, nAreas)';
lfpMeta.targetDepths = reshape([areaPick.targetDepths], 2, nAreas)';
lfpMeta.channelIds = reshape([areaPick.channelIds], 2, nAreas)';
lfpMeta.channelDepths = reshape([areaPick.channelDepths], 2, nAreas)';
lfpMeta.depthConvention = '0 = surface, larger = deeper';
lfpMeta.nChannelsRaw = nChannels;
lfpMeta.fsRaw = fsIn;
lfpMeta.fsOut = fsOut;
lfpMeta.lowpassFreq = lowpassFreq;
lfpMeta.nSamplesRaw = nSamplesIn;
lfpMeta.nSamplesOut = nSamplesOut;
lfpMeta.units = 'asStored';
lfpMeta.dataClass = 'single';
lfpMeta.matPath = matPath;
end
