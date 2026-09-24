function matPath = process_np1_lfp(sessionType, subjectName, sessionName, varargin)
% PROCESS_NP1_LFP - Stitch NP1 LFP, average 2 channels per area, save lfp.mat
%
% Variables:
%   sessionType - 'spontaneous' or 'interval'
%   subjectName - Subject folder (e.g. 'ey9166')
%   sessionName - Session folder (e.g. 'ey9166_2026_04_09')
%   varargin    - Optional name-value pairs:
%       'brainAreas'        - Areas to store (default {'M23','M56','DS','VS'})
%       'overwrite'         - Rebuild lfp.mat if it exists (default false)
%       'lfpDataPath'       - Folder of np1-lfp_*.raw (default paths.lfpDataPath)
%       'outputFolder'      - Destination folder (default: spike session folder)
%       'chunkDurationSec'  - Raw samples per processing hop (default 30)
%       'overlapSec'        - Edge padding for filtfilt/resample (default 2)
%       'maxDurationSec'    - If nonempty, process only the first N seconds
%
% Goal:
%   Match np1-lfp_*.raw files to the session date, stitch split takes, and
%   for each requested brain area average the two raw channels nearest 1/3
%   and 2/3 of that area's depth range (same 3840-minus-phy_y convention as
%   load_session_cluster_info.m). Low-pass below 300 Hz, resample to 1000 Hz,
%   and save single-precision traces plus metadata in lfp.mat.
%
% Usage:
%   process_np1_lfp('interval', 'ey9166', 'ey9166_2026_04_09');
%   process_np1_lfp(..., 'brainAreas', {'M23','M56'});

thisDir = fileparts(mfilename('fullpath'));
srcDir = fileparts(thisDir);
addpath(thisDir);
addpath(srcDir);

if nargin < 3 || isempty(sessionType) || isempty(subjectName) || isempty(sessionName)
    [sessionType, subjectName, sessionName] = read_session_vars_from_workspace();
end
sessionType = char(sessionType);
subjectName = char(subjectName);
sessionName = char(sessionName);

constants = np1_lfp_constants();
p = inputParser;
addParameter(p, 'brainAreas', constants.defaultBrainAreas, @(x) iscell(x) || ischar(x) || isstring(x));
addParameter(p, 'overwrite', false, @(x) islogical(x) || isnumeric(x));
addParameter(p, 'lfpDataPath', '', @(x) ischar(x) || isstring(x) || isempty(x));
addParameter(p, 'outputFolder', '', @(x) ischar(x) || isstring(x) || isempty(x));
addParameter(p, 'chunkDurationSec', 30, @isnumeric);
addParameter(p, 'overlapSec', 2, @isnumeric);
addParameter(p, 'maxDurationSec', [], @(x) isempty(x) || isnumeric(x));
parse(p, varargin{:});
brainAreas = p.Results.brainAreas;
overwrite = logical(p.Results.overwrite);
chunkDurationSec = p.Results.chunkDurationSec;
overlapSec = p.Results.overlapSec;
maxDurationSec = p.Results.maxDurationSec;

paths = get_paths;
if isempty(p.Results.lfpDataPath)
    lfpDataPath = paths.lfpDataPath;
else
    lfpDataPath = char(p.Results.lfpDataPath);
end
sessionFolder = resolve_lfp_session_folder(sessionType, subjectName, sessionName, paths);
if isempty(p.Results.outputFolder)
    outputFolder = sessionFolder;
else
    outputFolder = char(p.Results.outputFolder);
end
if ~isfolder(outputFolder)
    mkdir(outputFolder);
end

matPath = fullfile(outputFolder, constants.outputFileName);
if isfile(matPath) && ~overwrite
    fprintf('lfp.mat already exists (set overwrite=true to rebuild):\n  %s\n', matPath);
    return
end

sessionDate = parse_session_recording_date(sessionName);
fileInfo = find_np1_lfp_files(lfpDataPath, sessionDate, sessionFolder);
areaChan = select_area_lfp_channels(sessionFolder, brainAreas);

nChannels = constants.nChannels;
nAreas = numel(areaChan);
fsRaw = constants.fsRaw;
fsOut = constants.fsOut;
lowpassFreq = constants.lowpassFreq;
adcOffset = constants.adcOffset;
uVPerBit = constants.uVPerBit;
matlabIdx = zeros(nAreas, 2);
for iArea = 1:nAreas
    matlabIdx(iArea, :) = areaChan(iArea).channelIds + 1;
end

nSamplesTotal = sum([fileInfo.nSamples]);
nSamplesUse = nSamplesTotal;
if ~isempty(maxDurationSec)
    nSamplesUse = min(nSamplesTotal, round(maxDurationSec * fsRaw));
end
nSamplesOut = round(nSamplesUse * fsOut / fsRaw);
estMb = nSamplesOut * nAreas * 4 / 1e6;

fprintf('\n=== NP1 LFP processing ===\n');
fprintf('Session: %s / %s (%s)\n', subjectName, sessionName, sessionType);
fprintf('Date: %s\n', sessionDate);
fprintf('Raw: %d ch @ %d Hz, %.2f min (using %.2f min)\n', ...
    nChannels, fsRaw, nSamplesTotal / fsRaw / 60, nSamplesUse / fsRaw / 60);
fprintf('Output: %d areas, lowpass < %d Hz, %d Hz, single, ~%.1f MB\n', ...
    nAreas, lowpassFreq, fsOut, estMb);
fprintf('Write: %s\n', matPath);

[bLp, aLp] = design_lfp_lowpass(lowpassFreq, fsRaw);
resampleRatio = [fsOut, fsRaw];
gcdRq = gcd(resampleRatio(1), resampleRatio(2));
pResamp = resampleRatio(1) / gcdRq;
qResamp = resampleRatio(2) / gcdRq;

chunkHop = round(chunkDurationSec * fsRaw);
chunkHop = chunkHop - mod(chunkHop, qResamp);
overlapN = round(overlapSec * fsRaw);
overlapN = overlapN - mod(overlapN, qResamp);
if chunkHop < qResamp
    error('process_np1_lfp:ShortChunk', 'chunkDurationSec is too short.');
end
if overlapN >= chunkHop
    overlapN = qResamp;
end

lfpPerArea = zeros(nSamplesOut, nAreas, 'single');
nWritten = 0;
startSample = 0;
tStart = tic;
while startSample < nSamplesUse
    endSample = min(startSample + chunkHop, nSamplesUse);
    readStart = max(0, startSample - overlapN);
    readEnd = min(nSamplesUse, endSample + overlapN);
    nRead = readEnd - readStart;
    nPad = mod(qResamp - mod(nRead, qResamp), qResamp);

    dataU16 = read_np1_lfp_samples(fileInfo, readStart, nRead, nChannels);
    areaUv = zeros(nRead, nAreas);
    for iArea = 1:nAreas
        chPair = double(dataU16(matlabIdx(iArea, :), :)');
        areaUv(:, iArea) = mean((chPair - adcOffset) * uVPerBit, 2);
    end
    if nPad > 0
        areaUv = [areaUv; repmat(areaUv(end, :), nPad, 1)];
    end
    dataFilt = filtfilt(bLp, aLp, areaUv);
    dataResamp = resample(dataFilt, pResamp, qResamp);

    outStart = round(startSample * fsOut / fsRaw);
    outEnd = round(endSample * fsOut / fsRaw);
    keepStart = round((startSample - readStart) * fsOut / fsRaw) + 1;
    nKeep = outEnd - outStart;
    keepEnd = keepStart + nKeep - 1;
    if keepEnd > size(dataResamp, 1)
        nKeep = size(dataResamp, 1) - keepStart + 1;
        keepEnd = keepStart + nKeep - 1;
    end
    if outStart + nKeep > size(lfpPerArea, 1)
        nKeep = size(lfpPerArea, 1) - outStart;
        keepEnd = keepStart + nKeep - 1;
    end
    lfpPerArea(outStart + (1:nKeep), :) = single(dataResamp(keepStart:keepEnd, :));
    nWritten = nWritten + nKeep;
    startSample = endSample;

    elapsedSec = toc(tStart);
    fracDone = endSample / nSamplesUse;
    if fracDone > 0
        etaSec = elapsedSec * (1 - fracDone) / fracDone;
        fprintf('  %.1f%%  wrote %d / %d samples  ETA %.1f min\n', ...
            100 * fracDone, nWritten, nSamplesOut, etaSec / 60);
    end
end

if nWritten < nSamplesOut
    lfpPerArea = lfpPerArea(1:nWritten, :);
end

lfpMeta = build_lfp_meta(sessionType, subjectName, sessionName, sessionDate, ...
    fileInfo, areaChan, constants, nSamplesUse, nWritten, matPath);
save(matPath, 'lfpPerArea', 'lfpMeta', '-v7.3');

fprintf('Done. Wrote %d samples (%.2f min) x %d areas to\n  %s\n', ...
    nWritten, nWritten / fsOut / 60, nAreas, matPath);
end

function lfpMeta = build_lfp_meta(sessionType, subjectName, sessionName, sessionDate, ...
    fileInfo, areaChan, constants, nSamplesUse, nWritten, matPath)
% BUILD_LFP_META - Session, channel, and sampling metadata for lfp.mat

nAreas = numel(areaChan);
lfpMeta = struct();
lfpMeta.sessionType = sessionType;
lfpMeta.subjectName = subjectName;
lfpMeta.sessionName = sessionName;
lfpMeta.sessionDate = sessionDate;
lfpMeta.sourceFiles = {fileInfo.name};
lfpMeta.sourceDurationsSec = [fileInfo.durationSec];
lfpMeta.brainAreas = {areaChan.areaName};
lfpMeta.depthSource = areaChan(1).depthSource;
lfpMeta.depthRanges = reshape([areaChan.depthRange], 2, nAreas)';
lfpMeta.targetDepths = reshape([areaChan.targetDepths], 2, nAreas)';
lfpMeta.channelIds = reshape([areaChan.channelIds], 2, nAreas)';
lfpMeta.channelDepths = reshape([areaChan.channelDepths], 2, nAreas)';
lfpMeta.nChannelsRaw = constants.nChannels;
lfpMeta.referenceChannelId = areaChan(1).chanMeta.referenceChannelId;
lfpMeta.channelGeometrySource = areaChan(1).chanMeta.source;
lfpMeta.depthConvention = areaChan(1).chanMeta.depthConvention;
lfpMeta.fsRaw = constants.fsRaw;
lfpMeta.fsOut = constants.fsOut;
lfpMeta.lowpassFreq = constants.lowpassFreq;
lfpMeta.adcOffset = constants.adcOffset;
lfpMeta.uVPerBit = constants.uVPerBit;
lfpMeta.nSamplesRaw = nSamplesUse;
lfpMeta.nSamplesOut = nWritten;
lfpMeta.units = 'uV';
lfpMeta.dataClass = 'single';
lfpMeta.matPath = matPath;
end

function [bLp, aLp] = design_lfp_lowpass(lowpassFreq, fsRaw)
% DESIGN_LFP_LOWPASS - 4th-order Butterworth lowpass coefficients
%
% Variables:
%   lowpassFreq - Cutoff in Hz
%   fsRaw       - Sampling rate of the raw LFP

nyquist = fsRaw / 2;
if lowpassFreq >= nyquist
    error('process_np1_lfp:BadCutoff', 'lowpassFreq must be < Nyquist (%.1f Hz).', nyquist);
end
[bLp, aLp] = butter(4, lowpassFreq / nyquist, 'low');
end

function [sessionType, subjectName, sessionName] = read_session_vars_from_workspace()
% READ_SESSION_VARS_FROM_WORKSPACE - sessionType / subject / session from base

needVars = {'sessionType', 'subjectName', 'sessionName'};
hasBase = true;
for iVar = 1:numel(needVars)
    if ~evalin('base', sprintf('exist(''%s'', ''var'')', needVars{iVar}))
        hasBase = false;
        break
    end
end
if ~hasBase
    error('process_np1_lfp:MissingVar', ...
        'sessionType, subjectName, and sessionName must be passed in or defined in the workspace.');
end
sessionType = evalin('base', 'sessionType');
subjectName = evalin('base', 'subjectName');
sessionName = evalin('base', 'sessionName');
end
