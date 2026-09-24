function fileInfo = find_np1_lfp_files(lfpDataPath, sessionDate, sessionFolder)
% FIND_NP1_LFP_FILES - Locate and order np1-lfp_*.raw files for one date
%
% Variables:
%   lfpDataPath   - Folder containing np1-lfp_YYYY-MM-DDTHH_MM_SS*.raw
%   sessionDate   - 'YYYY-MM-DD' parsed from sessionName
%   sessionFolder - Optional spike session folder (for params.py timestamp)
%
% Goal:
%   Return a timestamp-sorted struct array of LFP files for that date.
%   Optional suffixes after the time (e.g. _0_to_2_hrs) are allowed.
%   Multiple files are stitched in chronological order, then by segment
%   start hour. Incomplete last frames (size not divisible by 384 ch) are
%   dropped with a warning. If params.py names an np1-spike_ file with a
%   matching timestamp, earlier aborted takes from the same day are dropped
%   so LFP lines up with spike times.

if nargin < 2 || isempty(lfpDataPath) || isempty(sessionDate)
    error('find_np1_lfp_files:MissingArgs', 'lfpDataPath and sessionDate are required.');
end
if nargin < 3
    sessionFolder = '';
end

constants = np1_lfp_constants();
if ~isfolder(lfpDataPath)
    error('find_np1_lfp_files:NoFolder', 'LFP data folder not found: %s', lfpDataPath);
end

rawFiles = dir(fullfile(lfpDataPath, constants.filePattern));
if isempty(rawFiles)
    error('find_np1_lfp_files:NoRaw', 'No %s files in %s', constants.filePattern, lfpDataPath);
end

fileInfo = struct('name', {}, 'path', {}, 'timestamp', {}, 'segmentStartHour', {}, ...
    'nBytes', {}, 'nSamples', {}, 'durationSec', {});
for iFile = 1:numel(rawFiles)
    fileName = rawFiles(iFile).name;
    dateTok = regexp(fileName, ...
        'np1-lfp_(\d{4}-\d{2}-\d{2})T(\d{2})_(\d{2})_(\d{2})(?:_(.*))?\.raw', ...
        'tokens', 'once');
    if isempty(dateTok)
        continue
    end
    fileDate = dateTok{1};
    if ~strcmp(fileDate, sessionDate)
        continue
    end
    timeStamp = sprintf('%sT%s_%s_%s', dateTok{1}, dateTok{2}, dateTok{3}, dateTok{4});
    nBytes = rawFiles(iFile).bytes;
    bytesPerFrame = constants.nChannels * constants.bytesPerSample;
    nExtra = mod(nBytes, bytesPerFrame);
    nSamples = floor(nBytes / bytesPerFrame);
    if nExtra ~= 0
        warning('find_np1_lfp_files:IncompleteFrame', ...
            ['%s size %d is not a whole number of 384-ch frames (%d leftover bytes). ', ...
            'Using %d complete samples and ignoring the truncated tail.'], ...
            fileName, nBytes, nExtra, nSamples);
    end
    if nSamples < 1
        warning('find_np1_lfp_files:EmptyFile', 'Skipping %s: no complete LFP samples.', fileName);
        continue
    end
    rec = struct();
    rec.name = fileName;
    rec.path = fullfile(lfpDataPath, fileName);
    rec.timestamp = timeStamp;
    rec.segmentStartHour = parse_segment_start_hour(dateTok);
    rec.nBytes = nBytes;
    rec.nSamples = nSamples;
    rec.durationSec = nSamples / constants.fsRaw;
    fileInfo(end+1) = rec; %#ok<AGROW>
end

if isempty(fileInfo)
    error('find_np1_lfp_files:NoMatch', ...
        'No np1-lfp_*.raw files in %s match date %s.', lfpDataPath, sessionDate);
end

sortKey = arrayfun(@(s) sprintf('%s_%06d', s.timestamp, s.segmentStartHour), fileInfo, 'UniformOutput', false);
[~, sortIdx] = sort(sortKey);
fileInfo = fileInfo(sortIdx);

spikeTimeStamp = read_spike_binary_timestamp(sessionFolder);
if ~isempty(spikeTimeStamp)
    spikeDate = spikeTimeStamp(1:10);
    if strcmp(spikeDate, sessionDate)
        matchMask = strcmp({fileInfo.timestamp}, spikeTimeStamp);
        if any(matchMask)
            firstKeep = find(matchMask, 1, 'first');
            if firstKeep > 1
                droppedNames = strjoin({fileInfo(1:firstKeep-1).name}, ', ');
                fprintf(['Skipping earlier same-day LFP take(s) so timestamps ', ...
                    'match params.py np1-spike_%s: %s\n'], spikeTimeStamp, droppedNames);
            end
            fileInfo = fileInfo(firstKeep:end);
        else
            fprintf(['params.py spike timestamp np1-spike_%s was not found among ', ...
                'LFP files; stitching all %s files.\n'], spikeTimeStamp, sessionDate);
        end
    else
        fprintf(['params.py spike file date %s does not match session date %s; ', ...
            'stitching LFP files by session date.\n'], spikeDate, sessionDate);
    end
end

fprintf('LFP files for %s (%d):\n', sessionDate, numel(fileInfo));
for iFile = 1:numel(fileInfo)
    fprintf('  %s  (%.2f min, %d samples)\n', ...
        fileInfo(iFile).name, fileInfo(iFile).durationSec / 60, fileInfo(iFile).nSamples);
end
end

function spikeTimeStamp = read_spike_binary_timestamp(sessionFolder)
% READ_SPIKE_BINARY_TIMESTAMP - np1-spike_YYYY-MM-DDTHH_MM_SS from params.py

spikeTimeStamp = '';
if isempty(sessionFolder)
    return
end
paramsPath = fullfile(sessionFolder, 'params.py');
if ~isfile(paramsPath)
    return
end
paramsText = fileread(paramsPath);
tok = regexp(paramsText, 'np1-spike_(\d{4}-\d{2}-\d{2}T\d{2}_\d{2}_\d{2})', 'tokens', 'once');
if ~isempty(tok)
    spikeTimeStamp = tok{1};
end
end

function startHour = parse_segment_start_hour(dateTok)
% PARSE_SEGMENT_START_HOUR - Start hour from suffix like 0_to_2_hrs

startHour = 0;
if numel(dateTok) < 5 || isempty(dateTok{5})
    return
end
segTok = regexp(dateTok{5}, '^(\d+)_to_(\d+)_hrs$', 'tokens', 'once');
if isempty(segTok)
    return
end
startHour = str2double(segTok{1});
end
