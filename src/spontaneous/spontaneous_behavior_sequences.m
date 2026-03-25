%%  Load data
sessionName = 'ag112321_1';
pathParts = strsplit(sessionName, filesep);
% Use first two characters of first path component
subDir = pathParts{1}(1:2);

paths = get_paths;
opts.dataPath = fullfile(paths.spontaneousDataPath, subDir);

sessionFolder = fullfile(opts.dataPath, sessionName);
% Behavioral data is stored with an assigned B-SOiD label every frame.
csvFiles = dir(fullfile(sessionFolder, 'behavior_labels*.csv'));
if isempty(csvFiles)
    error('No CSV file starting with "behavior_labels" found in %s', sessionFolder);
elseif length(csvFiles) > 1
    warning('Multiple CSV files starting with "behavior_labels" found. Using first: %s', csvFiles(1).name);
end
fileName = csvFiles(1).name;
bhvMat = readtable(fullfile(sessionFolder, fileName));

%%  Smooth behavior labels if you want to
bopts.fsBhv = 60;
bopts.smoothingWindow = .25;
bopts.summarize = true;
smoothedBhvID = behavior_label_smoothing(bhvMat.Code, bopts);
bhvMat.Code = smoothedBhvID;


%% See the behavior-code pairs

% Get unique Code–Behavior pairs (stable keeps first occurrence)
[uniqueCodes, ia] = unique(bhvMat.Code, 'stable');
uniqueBehaviors = bhvMat.Behavior(ia);

% Sort by Code
[sortedCodes, sortIdx] = sortrows(uniqueCodes);
sortedBehaviors = uniqueBehaviors(sortIdx);

% Display mapping
fprintf('Code\tBehavior\n');
for k = 1:numel(sortedCodes)
    fprintf('%d\t%s\n', sortedCodes(k), sortedBehaviors{k});
end

%% Explore unique sequences
uOpts.patternLength = 3:6;
uOpts.nMin = 10;
% uOpts.includeBhv = [5 6];
% uOpts.includeBhv = [0:2 15];
% uOpts.includeBhv = [14 15];
uOpts.includeBhv = [5:12];
% uOpts.includeBhv = [5 8];
uOpts.firstBhv = [];
uOpts.includeIsFirst = true;
uOpts.noRepeat = false;  % Can behavior labels repeat (true) or not (false) in the sequence?
uOpts.preBufferSec = 0.5; % How many frames before the sequences must no t contain labels in includeBhv
uOpts.excludeBhv = [-1];
% uOpts.excludeBhv = [-1];
[uniqueSequences, sequenceTimes] = find_unique_sequences(bhvMat, uOpts);
[uniqueSequences', sequenceTimes']

%% Isolate and collect the sequences for analysis
% Choose which of the sequences to analyze, and use a for loop to build the
% data for analysis
alignOnIdx = 2; % Which behavior index (in the compressed sequence) do you want to align on for analysis?

useIdx = [1:3 7]; % indices into `uniqueSequences` to analyze

[alignTimes, sequences, sequenceNames] = deal(cell(1, length(useIdx)));
for i = 1:length(useIdx)
    alignTimes{i} = sequenceTimes{useIdx(i)};
    sequences{i} = uniqueSequences{useIdx(i)};
    sequenceNames{i} = mat2str(uniqueSequences{useIdx(i)});
end

