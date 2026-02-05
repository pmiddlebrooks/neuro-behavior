% Find and return sequences of groom-like behavior, with the time in seconds of each start and stop of the sequence, that fit the following criteria:
% - sequence must be at least minDur seconds long
% - sequence must contain at least propThreshold proportion of groom labels
% - no overlapping sequences
% - at least bufferSec seconds between the end of one sequence and the start of the next

% dataFull is a table (from behavior_labels*.csv) at opts.fsBhv sampling rate;
% each row is one sample. Columns:
%   Time    - time in seconds of each frame
%   Behavior - string label of behavior at that frame
%   Code    - integer code label of behavior at that frame

% Output:
%   sequences - cell array of label ids (Code) for each sequence, at native frame rate
%   times    - cell array of time stamps (Time) for the labels in sequences
%%
sessionName = 'ag112321_1';
pathParts = strsplit(sessionName, filesep);
% Use first two characters of first path component
subDir = pathParts{1}(1:2);

paths = get_paths;
opts = neuro_behavior_options;
opts.frameSize = opts.fsBhv;
opts.collectStart = 0;
opts.collectEnd = [];
opts.sessionName = sessionName;
opts.dataPath = fullfile(paths.spontaneousDataPath, subDir);

sessionFolder = fullfile(opts.dataPath, opts.sessionName);
% Behavioral data is stored with an assigned B-SOiD label every frame.
csvFiles = dir(fullfile(sessionFolder, 'behavior_labels*.csv'));
if isempty(csvFiles)
    error('No CSV file starting with "behavior_labels" found in %s', sessionFolder);
elseif length(csvFiles) > 1
    warning('Multiple CSV files starting with "behavior_labels" found. Using first: %s', csvFiles(1).name);
end
fileName = csvFiles(1).name;
dataFull = readtable(fullfile(sessionFolder, fileName));

if isempty(dataFull) || height(dataFull) == 0
    sequences = {};
    times = {};
    fprintf('No behavior data. Returning empty sequences and times.\n');
    return;
end

%% Configure criteria
minDur = 5; % Minimum duration in seconds
propThreshold = 0.99; % Proportion of groom labels required (0-1)
bufferSec = 5; % Min seconds between end of one sequence and start of the next

% Identify groom frames: any Behavior that contains "groom" (case-insensitive)
if iscell(dataFull.Behavior)
    isGroom = cellfun(@(x) contains(char(x), 'groom', 'IgnoreCase', true), dataFull.Behavior);
else
    isGroom = contains(string(dataFull.Behavior), 'groom', 'IgnoreCase', true);
end
groomIds = unique(dataFull.Code(isGroom));

% Sliding window in frames (dataFull is one row per sample at opts.fsBhv)
fsBhv = opts.fsBhv;
nFrames = height(dataFull);
nWin = round(minDur * fsBhv);  % window length in frames
if nWin < 1
    nWin = 1;
end
minGroomFrames = ceil(nWin * propThreshold);  % need at least this many groom frames in window
nBuffer = round(bufferSec * fsBhv);  % buffer in frames (between sequences only)

candidateStart = [];
candidateEnd = [];
for startIdx = 1 : (nFrames - nWin + 1)
    endIdx = startIdx + nWin - 1;
    if sum(isGroom(startIdx:endIdx)) >= minGroomFrames
        candidateStart = [candidateStart; startIdx]; %#ok<AGROW>
        candidateEnd = [candidateEnd; endIdx]; %#ok<AGROW>
    end
end

% Merge overlapping windows into non-overlapping sequences (by frame index)
intervalStartIdx = [];
intervalEndIdx = [];
if ~isempty(candidateStart)
    [sortedStart, sortOrder] = sort(candidateStart);
    sortedEnd = candidateEnd(sortOrder);
    intervalStartIdx = sortedStart(1);
    intervalEndIdx = sortedEnd(1);
    for k = 2:length(sortedStart)
        if sortedStart(k) <= intervalEndIdx(end)
            intervalEndIdx(end) = max(intervalEndIdx(end), sortedEnd(k));
        else
            intervalStartIdx(end+1) = sortedStart(k); %#ok<AGROW>
            intervalEndIdx(end+1) = sortedEnd(k); %#ok<AGROW>
        end
    end
end

% Extend each sequence to the last groom frame for which the criteria still hold,
% but ensure no overlap: each sequence must end before the next begins.
for s = 1:length(intervalStartIdx)
    a = intervalStartIdx(s);
    b = intervalEndIdx(s);
    % Latest allowed end: leave >= bufferSec between this sequence and the next
    if s < length(intervalStartIdx)
        maxEnd = intervalStartIdx(s + 1) - nBuffer - 1;
    else
        maxEnd = nFrames;  % last sequence can extend to end of data
    end
    if maxEnd < a
        intervalEndIdx(s) = a - 1;  % mark invalid (end before start); dropped below
        continue;
    end
    % End on a groom frame: use last groom in [a,b] as current end if any
    groomInSegment = find(isGroom(a:min(b, maxEnd)));
    if ~isempty(groomInSegment)
        endIdx = a - 1 + groomInSegment(end);
    else
        endIdx = min(b, maxEnd);
    end
    % Extend forward to last groom frame where window [e-nWin+1, e] still passes
    for e = (endIdx + 1):maxEnd
        if e - nWin + 1 < 1
            break;
        end
        if ~isGroom(e)
            continue;
        end
        if sum(isGroom((e - nWin + 1):e)) >= minGroomFrames
            endIdx = e;
        end
    end
    intervalEndIdx(s) = endIdx;
end

% Drop sequences that violate buffer (end before start or too short after capping)
keep = (intervalEndIdx >= intervalStartIdx) & ((intervalEndIdx - intervalStartIdx + 1) >= nWin);
intervalStartIdx = intervalStartIdx(keep);
intervalEndIdx = intervalEndIdx(keep);

% Build output: one cell per sequence, using frame indexing
sequences = cell(length(intervalStartIdx), 1);
times = cell(length(intervalStartIdx), 1);
for s = 1:length(intervalStartIdx)
    a = intervalStartIdx(s);
    b = intervalEndIdx(s);
    sequences{s} = dataFull.Code(a:b);
    times{s} = dataFull.Time(a:b);
end

% Report
fprintf('Groom IDs: %s\n', mat2str(groomIds(:)'));
fprintf('Found %d groom-like sequence(s) (minDur=%.1fs, propThreshold=%.2f).\n', length(sequences), minDur, propThreshold);
for s = 1:length(sequences)
    t0 = dataFull.Time(intervalStartIdx(s));
    t1 = dataFull.Time(intervalEndIdx(s));
    fprintf('  Sequence %d: %.2f - %.2f s, %d frames\n', s, t0, t1, numel(sequences{s}));
end


%%
%%
sessionName = 'ag112321_1';
        paths = get_paths;
% Use first two characters of first path component
    % Resolve data path and session folder
        pathParts = strsplit(sessionName, filesep);
        subDir = pathParts{1}(1:min(2, numel(pathParts{1})));
        dataPath = fullfile(paths.spontaneousDataPath, subDir);

    sessionFolder = fullfile(dataPath, sessionName);

    % Load behavior CSV
    csvFiles = dir(fullfile(sessionFolder, 'behavior_labels*.csv'));
    if length(csvFiles) > 1
        warning('spontaneous_behavior_sequences:multipleCSV', ...
            'Multiple behavior_labels CSV files found. Using: %s', csvFiles(1).name);
    end
    dataFull = readtable(fullfile(sessionFolder, csvFiles(1).name));

opts = neuro_behavior_options;
opts.frameSize = opts.fsBhv;
opts.collectStart = 0;
opts.collectEnd = [];
opts.sessionName = sessionName;
opts.dataPath = fullfile(paths.spontaneousDataPath, subDir);


opts.minDur = 2; % Minimum duration in seconds
opts.propThreshold = 0.95; % Proportion of groom labels required (0-1)
opts.bufferSec = 1.5; % Min seconds between end of one sequence and start of the next
opts.behaviorIds = 5:10;
% opts.behaviorIds = 0:15;
opts.nMinUniqueBhv = 3;
opts.fsBhv = 60;

[sequences, times] = spontaneous_behavior_sequences(dataFull, opts);
size(sequences)



%%
patOpts.minLength = 3;   % a sequence pattern of at least minLength 
patOpts.maxLength = 8;   % a sequence pattern of at most maxLength
patOpts.nPatterns = 20;  % top 10 recurring patterns
patOpts.nMinUniqueBhv = 2;
patOpts.anchorToFirstPattern = false;

patterns = find_common_behavior_patterns(sequences, times, patOpts);
patterns(1)
patterns(end)
%%
% Get unique Code–Behavior pairs (stable keeps first occurrence)
[uniqueCodes, ia] = unique(dataFull.Code, 'stable');
uniqueBehaviors = dataFull.Behavior(ia);

% Sort by Code
[sortedCodes, sortIdx] = sort(uniqueCodes);
sortedBehaviors = uniqueBehaviors(sortIdx);

% Display mapping
fprintf('Code\tBehavior\n');
for k = 1:numel(sortedCodes)
    fprintf('%d\t%s\n', sortedCodes(k), sortedBehaviors{k});
end

%%
bopts.fsBhv = 60;
bopts.smoothingWindow = .2;
bopts.summarize = true;
smoothedBhvID = behavior_label_smoothing(dataFull.Code, bopts);



%% Unique sequences
uOpts.patternLength = 2;
uOpts.nMin = 10;
uOpts.includeBhv = [5 6];
% uOpts.includeBhv = [0:2 15];
% uOpts.includeBhv = [14 15];
uOpts.includeIsFirst = true;
uOpts.noRepeat = false;
uOpts.nPreBuffer = .15 * opts.fsBhv;
uOpts.excludeBhv = [-1];
[uniqueSequences, sequenceTimes, sequenceIdx] = find_unique_sequences(dataFull, uOpts);
%%
% For criticality_behavior_sequences..
idx1 = 3;
idx2 = 10;
align1Times = sequenceTimes{idx1};
align2Times = sequenceTimes{idx2};
align1Name = mat2str(uniqueSequences{idx1});
align2Name = mat2str(uniqueSequences{idx2});

% Run criticality_behavior_sequences
criticality_behavior_sequences


%%
dimensionality_spontaneous_sequences_pr