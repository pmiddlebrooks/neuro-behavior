function patterns = find_common_behavior_patterns(sequences, times, opts)
% FIND_COMMON_BEHAVIOR_PATTERNS Find recurring label patterns across behavior sequences.
%
% Variables:
%   sequences - Cell array of behavior label vectors (e.g. from spontaneous_behavior_sequences),
%               each element is a vector of integer codes (one per frame/bin).
%   times     - Cell array of time vectors corresponding to sequences (same shape as
%               the times output of spontaneous_behavior_sequences; each element is
%               a vector of time stamps per frame/bin).
%   opts      - Struct with fields:
%       minLength  - Minimum pattern length (in distinct labels, after run-length compression).
%       maxLength  - Maximum pattern length (in distinct labels).
%       nPatterns  - Maximum number of top patterns to return (by total occurrences).
%       nMinUniqueBhv - (optional) Minimum number of unique labels required in each pattern.
%                       If opts.behaviorIds is provided, only counts unique labels that are
%                       in behaviorIds. Otherwise, counts all unique labels in the pattern.
%                       Default: 1.
%       behaviorIds - (optional) Vector of behavior IDs. If provided, nMinUniqueBhv counts
%                     only unique labels from this set.
%       anchorToFirstPattern - (optional, logical) If true:
%           * For each pattern length from minLength to maxLength:
%             - Find the first sequence with a pattern of that length starting at index 1
%             - Use that pattern as a template
%             - Search all sequences for sequences that start with that exact pattern
%             - Each sequence contributes at most one match per pattern length
%           If false (default):
%           * Enumerate all patterns of each length across all sequences
%           * Patterns can occur anywhere within sequences and can repeat within sequences
%
% Goal:
%   Treat each input sequence as a series of *runs* of labels, e.g.
%       [5 5 5 6 6 4 4]  ->  [5 6 4]
%   Then, for each pattern length from minLength to maxLength:
%     - If anchorToFirstPattern is false: enumerate all contiguous subpatterns of that
%       length (e.g., [5 6], [6 4], [5 6 4]) and count occurrences across all sequences.
%     - If anchorToFirstPattern is true: find the first sequence with a pattern of that
%       length starting at index 1, use it as a template, and search all sequences for
%       sequences that start with that pattern (one match per sequence per pattern length).
%   Returns the top nPatterns most frequent patterns.
%
% Returns:
%   patterns - Struct array with fields:
%       .labels       - Row vector of labels defining the pattern (e.g. [5 6 4]).
%       .count        - Total number of occurrences across all sequences.
%       .occurrences  - N x 2 array of [seqIdx, startIdx] pairs, where:
%                       - seqIdx   is the index into the input sequences cell array
%                       - startIdx is the start index in the *compressed* label sequence
%                         for that seq (i.e., index into the run-length compressed labels)
%       .times        - 1 x N vector of time stamps (in seconds) giving the start time
%                       of each occurrence (from the corresponding times{seqIdx}).
%
% Example:
%   [seqs, ~] = spontaneous_behavior_sequences(sessionName, optsBhv);
%   patOpts.minLength = 2;
%   patOpts.maxLength = 4;
%   patOpts.nPatterns = 10;
%   patterns = find_common_behavior_patterns(seqs, patOpts);

    % Validate inputs
    if nargin < 3 || ~isstruct(opts)
        error('find_common_behavior_patterns:opts', 'opts struct is required.');
    end
    if ~iscell(sequences)
        error('find_common_behavior_patterns:sequences', 'sequences must be a cell array of label vectors.');
    end
    if ~iscell(times) || numel(times) ~= numel(sequences)
        error('find_common_behavior_patterns:times', 'times must be a cell array matching sequences in length.');
    end

    % Required options
    if ~isfield(opts, 'minLength') || isempty(opts.minLength)
        error('find_common_behavior_patterns:opts', 'opts.minLength (pattern length in labels) is required.');
    end
    if ~isfield(opts, 'maxLength') || isempty(opts.maxLength)
        error('find_common_behavior_patterns:opts', 'opts.maxLength (pattern length in labels) is required.');
    end

    minLength = max(1, opts.minLength);
    maxLength = max(minLength, opts.maxLength);

    % Default number of patterns to return
    if ~isfield(opts, 'nPatterns') || isempty(opts.nPatterns)
        nPatterns = 10;
    else
        nPatterns = opts.nPatterns;
    end

    % Minimum unique behavior labels per pattern
    if ~isfield(opts, 'nMinUniqueBhv') || isempty(opts.nMinUniqueBhv)
        nMinUniqueBhv = 1;
    else
        nMinUniqueBhv = opts.nMinUniqueBhv;
    end

    % Optional behaviorIds for filtering unique labels
    if isfield(opts, 'behaviorIds') && ~isempty(opts.behaviorIds)
        behaviorIds = opts.behaviorIds(:)';
        useBehaviorIdsFilter = true;
    else
        behaviorIds = [];
        useBehaviorIdsFilter = false;
    end

    % Option to anchor patterns to sequences that start with the same pattern
    anchorToFirst = isfield(opts, 'anchorToFirstPattern') && opts.anchorToFirstPattern;

    % Map from serialized pattern key -> struct with fields:
    %   .labels      - pattern labels (row vector)
    %   .count       - total occurrence count
    %   .occurrences - [seqIdx, startIdx] rows
    %   .times       - vector of time stamps for each occurrence
    patternMap = containers.Map('KeyType', 'char', 'ValueType', 'any');

    nSeq = numel(sequences);

    % Pre-compress all sequences and store metadata
    compressedSeqs = cell(1, nSeq);
    runStartIdxs = cell(1, nSeq);
    for seqIdx = 1:nSeq
        codeSeq = sequences{seqIdx};
        if isempty(codeSeq)
            compressedSeqs{seqIdx} = [];
            runStartIdxs{seqIdx} = [];
            continue;
        end
        codeSeq = codeSeq(:)';
        changeIdx = [true, diff(codeSeq) ~= 0];
        compressedLabels = codeSeq(changeIdx);
        compressedSeqs{seqIdx} = compressedLabels(:)';  % Ensure row vector
        runStartIdxs{seqIdx} = find(changeIdx);
    end

    % Iterate through pattern lengths from minLength to maxLength
    for patLen = minLength:maxLength
        if anchorToFirst
            % Find first sequence with a pattern of this length starting at index 1
            % The template is the first patLen labels of that sequence's compressed pattern
            template = [];
            for seqIdx = 1:nSeq
                compressedLabels = compressedSeqs{seqIdx};
                if ~isempty(compressedLabels) && numel(compressedLabels) >= patLen
                    template = compressedLabels(1:patLen);
                    % Ensure template is a row vector
                    template = template(:)';
                    break;
                end
            end

            % If no template found for this length, skip to next pattern length
            if isempty(template)
                continue;
            end

            % Create unique key for this template pattern
            templateKey = sprintf('%d_', template);

            % Search all sequences for sequences that start with this template
            % Each sequence can contribute at most one match (at index 1)
            for seqIdx = 1:nSeq
                compressedLabels = compressedSeqs{seqIdx};
                if isempty(compressedLabels) || numel(compressedLabels) < patLen
                    continue;
                end

                % Ensure compressedLabels is a row vector for comparison
                compressedLabels = compressedLabels(:)';

                % Check if this sequence starts with the template (must match exactly)
                % Extract the first patLen labels and compare
                seqStart = compressedLabels(1:patLen);
                if numel(compressedLabels) >= patLen && isequal(seqStart, template)
                    tSeq = times{seqIdx};
                    frameIdx = runStartIdxs{seqIdx}(1);
                    if ~isempty(tSeq) && frameIdx <= numel(tSeq)
                        tStart = tSeq(frameIdx);
                    else
                        tStart = NaN;
                    end

                    if patternMap.isKey(templateKey)
                        entry = patternMap(templateKey);
                        entry.count = entry.count + 1;
                        entry.occurrences(end+1, :) = [seqIdx, 1];
                        entry.times(end+1) = tStart;
                        patternMap(templateKey) = entry;
                    else
                        entry = struct();
                        entry.labels = template;
                        entry.count = 1;
                        entry.occurrences = [seqIdx, 1];
                        entry.times = tStart;
                        patternMap(templateKey) = entry;
                    end
                end
            end
        else
            % Enumerate all patterns of this length across all sequences
            % Patterns can occur anywhere and can repeat within sequences
            for seqIdx = 1:nSeq
                compressedLabels = compressedSeqs{seqIdx};
                if isempty(compressedLabels) || numel(compressedLabels) < patLen
                    continue;
                end

                L = numel(compressedLabels);
                tSeq = times{seqIdx};
                runStartIdx = runStartIdxs{seqIdx};

                % Enumerate all contiguous patterns of this length
                for startIdx = 1:(L - patLen + 1)
                    pat = compressedLabels(startIdx:startIdx + patLen - 1);
                    key = sprintf('%d_', pat);

                    % Determine start time from original times
                    frameIdx = runStartIdx(startIdx);
                    if ~isempty(tSeq) && frameIdx <= numel(tSeq)
                        tStart = tSeq(frameIdx);
                    else
                        tStart = NaN;
                    end

                    if patternMap.isKey(key)
                        entry = patternMap(key);
                        entry.count = entry.count + 1;
                        entry.occurrences(end+1, :) = [seqIdx, startIdx];
                        entry.times(end+1) = tStart;
                        patternMap(key) = entry;
                    else
                        entry = struct();
                        entry.labels = pat;
                        entry.count = 1;
                        entry.occurrences = [seqIdx, startIdx];
                        entry.times = tStart;
                        patternMap(key) = entry;
                    end
                end
            end
        end
    end

    % Convert map to struct array and sort by count (descending)
    keys = patternMap.keys;
    nKeys = numel(keys);

    if nKeys == 0
        patterns = struct('labels', {}, 'count', {}, 'occurrences', {}, 'times', {});
        return;
    end

    patternsAll = repmat(struct('labels', [], 'count', [], 'occurrences', [], 'times', []), 1, nKeys);
    counts = zeros(1, nKeys);
    keepPattern = true(1, nKeys);
    for k = 1:nKeys
        entry = patternMap(keys{k});
        patternsAll(k).labels = entry.labels;
        patternsAll(k).count = entry.count;
        patternsAll(k).occurrences = entry.occurrences;
        patternsAll(k).times = entry.times;
        counts(k) = entry.count;

        % Filter by minimum unique behavior labels
        if nMinUniqueBhv > 0
            uniqueLabelsInPattern = unique(entry.labels);
            if useBehaviorIdsFilter
                uniqueBhvInBehaviorIds = uniqueLabelsInPattern(ismember(uniqueLabelsInPattern, behaviorIds));
                nUnique = length(uniqueBhvInBehaviorIds);
            else
                nUnique = length(uniqueLabelsInPattern);
            end
            if nUnique < nMinUniqueBhv
                keepPattern(k) = false;
            end
        end
    end

    % Filter patterns that don't meet nMinUniqueBhv requirement
    patternsAll = patternsAll(keepPattern);
    counts = counts(keepPattern);

    if isempty(patternsAll)
        patterns = struct('labels', {}, 'count', {}, 'occurrences', {}, 'times', {});
        return;
    end

    [~, sortIdx] = sort(counts, 'descend');
    nKeep = min(nPatterns, length(patternsAll));
    patterns = patternsAll(sortIdx(1:nKeep));
end

