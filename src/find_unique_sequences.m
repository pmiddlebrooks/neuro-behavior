function [uniqueSequences, sequenceTimes, sequenceIdx] = find_unique_sequences(dataFull, opts)
% FIND_UNIQUE_SEQUENCES Find unique run-length compressed behavior patterns of fixed length.
%
% Variables:
%   dataFull  - Table with behavior data with fields Time, Behavior, and Code.
%               Code is the behavior ID per frame (column vector or table column).
%   opts      - Struct with fields:
%       patternLength - Number of distinct behaviors in a sequence (pattern length after
%                       run-length compression). Required.
%       nMin          - Minimum number of occurrences for a pattern to be included in
%                       the output (default 1).
%       includeBhv    - (Optional) Vector of behavior codes. Only sequences containing at
%                       least one of these behaviors are valid. If empty, include all.
%       includeIsFirst- (Optional) If true, valid sequences must begin with one of the
%                       labels in includeBhv and must not be directly preceded by any
%                       of the other labels in includeBhv (i.e. the prior run is not
%                       in includeBhv). Only applied when includeBhv is non-empty.
%                       Default false.
%       excludeBhv    - (Optional) Vector of behavior codes. Exclude any sequence that
%                       contains any of these behaviors. If empty, exclude none.
%
% Goal:
%   Compress the Code sequence into the behavior pattern (run-length sequence), e.g.
%   [5 5 5 6 6 4] -> [5 6 4]. Then find all unique patterns of length patternLength in
%   this compressed sequence, ranked by frequency. Only patterns with at least nMin
%   occurrences are returned. Optionally filter by includeBhv (pattern must contain at
%   least one), includeIsFirst (pattern must start with one of includeBhv and not be
%   directly preceded by any other label in includeBhv), and excludeBhv (pattern must
%   contain none). All frames are treated as valid.
%
% Returns:
%   uniqueSequences - Cell array of unique patterns (each a 1 x patternLength numeric
%                     vector), ranked by most frequent first.
%   sequenceTimes   - Cell array (same order as uniqueSequences); each element is a
%                     vector of start times (seconds) when that pattern begins
%                     (from dataFull.Time at the first frame of the pattern).
%   sequenceIdx     - Cell array (same order as uniqueSequences); each element is a
%                     vector of frame indices (into dataFull) of the first element of
%                     each occurrence of that pattern.
%
% Example:
%   opts.patternLength = 3;
%   opts.nMin = 2;
%   [uniqueSeqs, startTimes, startIdx] = find_unique_sequences(dataFull, opts);

    if nargin < 2 || isempty(opts) || ~isstruct(opts)
        error('find_unique_sequences:opts', 'opts struct is required.');
    end
    if ~isfield(opts, 'patternLength') || isempty(opts.patternLength) || ~isscalar(opts.patternLength) || opts.patternLength < 1
        error('find_unique_sequences:opts', 'opts.patternLength must be a positive scalar.');
    end
    patternLength = opts.patternLength;
    if isfield(opts, 'nMin') && ~isempty(opts.nMin)
        nMin = max(1, round(opts.nMin));
    else
        nMin = 1;
    end
    includeBhv = [];
    if isfield(opts, 'includeBhv') && ~isempty(opts.includeBhv)
        includeBhv = opts.includeBhv(:)';
    end
    excludeBhv = [];
    if isfield(opts, 'excludeBhv') && ~isempty(opts.excludeBhv)
        excludeBhv = opts.excludeBhv(:)';
    end
    includeIsFirst = false;
    if isfield(opts, 'includeIsFirst') && ~isempty(opts.includeIsFirst)
        includeIsFirst = logical(opts.includeIsFirst(1));
    end
    if isempty(dataFull) || ~istable(dataFull)
        error('find_unique_sequences:dataFull', 'dataFull must be a non-empty table with Time, Behavior, Code.');
    end
    if ~ismember('Code', dataFull.Properties.VariableNames)
        error('find_unique_sequences:dataFull', 'dataFull must contain a Code column.');
    end
    if ~ismember('Time', dataFull.Properties.VariableNames)
        error('find_unique_sequences:dataFull', 'dataFull must contain a Time column.');
    end

    codeVec = dataFull.Code(:);
    timeVec = dataFull.Time(:);
    nFrames = numel(codeVec);

    % Run-length compress Code into behavior pattern
    changeIdx = [true; diff(codeVec) ~= 0];
    compressedLabels = codeVec(changeIdx);
    runStartFrameIdx = find(changeIdx);
    nCompressed = numel(compressedLabels);

    allSequences = {};
    allTimes = {};
    allIndices = {};

    for i = 1 : (nCompressed - patternLength + 1)
        currentPattern = compressedLabels(i : i + patternLength - 1);

        if any(isnan(currentPattern))
            continue;
        end
        % includeBhv: pattern must contain at least one of these behaviors
        if ~isempty(includeBhv) && ~any(ismember(currentPattern, includeBhv))
            continue;
        end
        % includeIsFirst: pattern must begin with one of includeBhv and not be
        % directly preceded by any other label in includeBhv
        if includeIsFirst && ~isempty(includeBhv)
            if ~ismember(currentPattern(1), includeBhv)
                continue;
            end
            % Preceding run (if any) must not be in includeBhv
            if i > 1 && ismember(compressedLabels(i - 1), includeBhv)
                continue;
            end
        end
        % excludeBhv: pattern must not contain any of these behaviors
        if ~isempty(excludeBhv) && any(ismember(currentPattern, excludeBhv))
            continue;
        end

        seqStr = strjoin(arrayfun(@num2str, currentPattern(:)', 'UniformOutput', false), ',');
        seqIndex = find(strcmp(allSequences, seqStr));
        firstFrameOfPattern = runStartFrameIdx(i);
        firstTimeOfPattern = timeVec(firstFrameOfPattern);

        if isempty(seqIndex)
            allSequences{end + 1} = seqStr;
            allTimes{end + 1} = firstTimeOfPattern;
            allIndices{end + 1} = firstFrameOfPattern;
        else
            allTimes{seqIndex} = [allTimes{seqIndex}, firstTimeOfPattern];
            allIndices{seqIndex} = [allIndices{seqIndex}, firstFrameOfPattern];
        end
    end

    % Convert sequence strings back to numeric vectors
    uniqueSequences = cellfun(@(x) str2num(strrep(x, ',', ' ')), allSequences, 'UniformOutput', false);

    % Sort by frequency (descending)
    allTimesSorted = allTimes;
    allIndicesSorted = allIndices;
    [~, sortOrder] = sort(cellfun(@length, allTimesSorted), 'descend');
    uniqueSequences = uniqueSequences(sortOrder);
    allTimesSorted = allTimesSorted(sortOrder);
    allIndicesSorted = allIndicesSorted(sortOrder);

    % Keep only patterns with at least nMin occurrences
    keepIdx = cellfun(@length, allTimesSorted) >= nMin;
    uniqueSequences = uniqueSequences(keepIdx);
    sequenceTimes = allTimesSorted(keepIdx);
    sequenceIdx = allIndicesSorted(keepIdx);

end
