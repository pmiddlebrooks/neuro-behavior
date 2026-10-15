function fileInfo = find_np2_lfp_files(lfpDataPath, sessionDate, sessionFolder, probeId, nChannels)
% FIND_NP2_LFP_FILES - Locate and order extracted NP2 LFP files for one date
%
% Variables:
%   lfpDataPath   - Folder containing np2-<probe>-ephys_YYYY-MM-DDTHH_MM_SS*_lfp.raw
%   sessionDate   - 'YYYY-MM-DD' parsed from sessionName
%   sessionFolder - Optional spike session folder (for params.py timestamp)
%   probeId       - Probe letter in the file name (default 'a')
%   nChannels     - Channels per frame, used to turn bytes into sample counts
%
% Goal:
%   Return the 2500 Hz files produced by extract_lfp_np2.sh for this probe
%   and date, in time order. Wideband np2-*-ephys_*.raw files are ignored.
%   Segments such as _0_to_2_hrs_lfp then _2_to_4_hrs_lfp are stitched.
%   A truncated last frame is dropped with a warning.

if nargin < 2 || isempty(lfpDataPath) || isempty(sessionDate)
    error('find_np2_lfp_files:MissingArgs', 'lfpDataPath and sessionDate are required.');
end
if nargin < 3
    sessionFolder = '';
end
constants = np2_lfp_constants();
if nargin < 4 || isempty(probeId)
    probeId = constants.defaultProbeId;
end
if nargin < 5 || isempty(nChannels)
    nChannels = constants.nChannels;
end
probeId = char(probeId);

if ~isfolder(lfpDataPath)
    error('find_np2_lfp_files:NoFolder', 'LFP data folder not found: %s', lfpDataPath);
end

rawFiles = dir(fullfile(lfpDataPath, constants.filePattern));
if isempty(rawFiles)
    error('find_np2_lfp_files:NoRaw', ...
        'No %s files in %s. Run extract_lfp_np2.sh first.', ...
        constants.filePattern, lfpDataPath);
end

fileInfo = struct('name', {}, 'path', {}, 'timestamp', {}, 'probeId', {}, ...
    'segmentStartHour', {}, 'nBytes', {}, 'nSamples', {}, 'durationSec', {});
namePattern = ['np2-([A-Za-z0-9]+)-ephys_(\d{4}-\d{2}-\d{2})T(\d{2})_(\d{2})_(\d{2})', ...
    '(?:_(\d+)_to_(\d+)_hrs)?_lfp\.raw'];
bytesPerFrame = nChannels * constants.bytesPerSample;

for iFile = 1:numel(rawFiles)
    fileName = rawFiles(iFile).name;
    dateTok = regexp(fileName, namePattern, 'tokens', 'once');
    if isempty(dateTok)
        continue
    end
    if ~strcmpi(dateTok{1}, probeId)
        continue
    end
    fileDate = dateTok{2};
    if ~strcmp(fileDate, sessionDate)
        continue
    end
    timeStamp = sprintf('%sT%s_%s_%s', dateTok{2}, dateTok{3}, dateTok{4}, dateTok{5});
    nBytes = rawFiles(iFile).bytes;
    nExtra = mod(nBytes, bytesPerFrame);
    nSamples = floor(nBytes / bytesPerFrame);
    if nExtra ~= 0
        warning('find_np2_lfp_files:IncompleteFrame', ...
            ['%s size %d is not a whole number of %d-ch frames (%d leftover bytes). ', ...
            'Using %d complete samples and ignoring the truncated tail.'], ...
            fileName, nBytes, nChannels, nExtra, nSamples);
    end
    if nSamples < 1
        warning('find_np2_lfp_files:EmptyFile', 'Skipping %s: no complete LFP samples.', fileName);
        continue
    end
    rec = struct();
    rec.name = fileName;
    rec.path = fullfile(lfpDataPath, fileName);
    rec.timestamp = timeStamp;
    rec.probeId = dateTok{1};
    rec.segmentStartHour = segment_start_hour(fileName);
    rec.nBytes = nBytes;
    rec.nSamples = nSamples;
    rec.durationSec = nSamples / constants.fsRaw;
    fileInfo(end+1) = rec; %#ok<AGROW>
end

if isempty(fileInfo)
    error('find_np2_lfp_files:NoMatch', ...
        'No np2-%s-ephys_*_lfp.raw files in %s match date %s.', ...
        probeId, lfpDataPath, sessionDate);
end

sortKey = arrayfun(@(s) sprintf('%s_%06d', s.timestamp, s.segmentStartHour), ...
    fileInfo, 'UniformOutput', false);
[~, sortIdx] = sort(sortKey);
fileInfo = fileInfo(sortIdx);

spikeTimeStamp = read_np2_ephys_timestamp(sessionFolder, probeId);
if ~isempty(spikeTimeStamp)
    spikeDate = spikeTimeStamp(1:10);
    if strcmp(spikeDate, sessionDate)
        matchMask = strcmp({fileInfo.timestamp}, spikeTimeStamp);
        if any(matchMask)
            firstKeep = find(matchMask, 1, 'first');
            if firstKeep > 1
                droppedNames = strjoin({fileInfo(1:firstKeep-1).name}, ', ');
                fprintf(['Skipping earlier same-day LFP take(s) so timestamps ', ...
                    'match params.py np2-%s-ephys_%s: %s\n'], ...
                    probeId, spikeTimeStamp, droppedNames);
            end
            fileInfo = fileInfo(firstKeep:end);
        else
            fprintf(['params.py timestamp np2-%s-ephys_%s was not found among ', ...
                'LFP files; stitching all %s files.\n'], probeId, spikeTimeStamp, sessionDate);
        end
    end
end

fprintf('NP2 LFP files for %s probe %s (%d):\n', sessionDate, probeId, numel(fileInfo));
for iFile = 1:numel(fileInfo)
    fprintf('  %s  (%.2f min, %d samples)\n', ...
        fileInfo(iFile).name, fileInfo(iFile).durationSec / 60, fileInfo(iFile).nSamples);
end
end

function spikeTimeStamp = read_np2_ephys_timestamp(sessionFolder, probeId)
% READ_NP2_EPHYS_TIMESTAMP - np2-<probe>-ephys_YYYY-MM-DDTHH_MM_SS from params.py

spikeTimeStamp = '';
if isempty(sessionFolder) || ~isfolder(sessionFolder)
    return
end
paramsPath = fullfile(sessionFolder, 'params.py');
if ~isfile(paramsPath)
    return
end
paramsText = fileread(paramsPath);
pattern = sprintf('np2-%s-ephys_(\\d{4}-\\d{2}-\\d{2}T\\d{2}_\\d{2}_\\d{2})', probeId);
tok = regexpi(paramsText, pattern, 'tokens', 'once');
if ~isempty(tok)
    spikeTimeStamp = tok{1};
end
end

function startHour = segment_start_hour(fileName)
% SEGMENT_START_HOUR - Start hour from a _0_to_2_hrs_lfp suffix, or 0
%
% Variables:
%   fileName - Extracted LFP file name
%
% Goal:
%   Order _0_to_2_hrs_lfp before _2_to_4_hrs_lfp when one recording was split.

startHour = 0;
segTok = regexp(fileName, '_(\d+)_to_\d+_hrs_lfp\.raw$', 'tokens', 'once');
if isempty(segTok)
    return
end
startHour = str2double(segTok{1});
end
