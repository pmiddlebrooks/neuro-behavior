function [matchingSequences, startIndices, bhvSequences] = find_matching_sequences(bhvID, opts)
% FIND_MATCHING_SEQUENCES Find behavior sequences matching label and duration criteria.
%
% Variables:
%   bhvID - Vector of categorical numerical variables representing behaviors (one per bin).
%   opts  - Struct with fields (all durations in **seconds**):
%       possibleBhv      - Vector of labels considered as "in" behaviors.
%       minBhvPct        - Minimum percentage of bins in a sequence that must be in
%                          possibleBhv (0–100).
%       minDistinctBhv   - Minimum number of distinct labels (from possibleBhv) required
%                          within a sequence.
%       fsBhv            - Sampling rate of behavior data in Hz (default 60).
%       minDurSeq        - Minimum sequence duration in seconds (required).
%       maxNonBhvDur     - Maximum duration (seconds) of any *consecutive* run of labels
%                          not in possibleBhv allowed inside a sequence (default 0).
%       minDurBtwnSeq    - Minimum duration (seconds) between separate qualifying sequences
%                          (default 0).
%
% Goal:
%   Return all sequences that:
%     - Are at least minDurSeq long,
%     - Have at least minBhvPct of their bins labeled with possibleBhv,
%     - Contain at least minDistinctBhv distinct labels from possibleBhv,
%     - Do not contain any consecutive non-possibleBhv run longer than maxNonBhvDur.
%
% Returns:
%   matchingSequences - Cell array of sequences (sub-vectors of bhvID) meeting criteria.
%   startIndices      - Start indices (into bhvID) for each sequence.
%   bhvSequences      - Cell array of the same sequences compressed to unique runs
%                       of labels (consecutive duplicates removed).

    % Unpack options
    possibleBhv = opts.possibleBhv;
    minBhvPct = opts.minBhvPct;
    minDistinctBhv = opts.minDistinctBhv;

    % Sample rate (Hz), default 60
    if isfield(opts, 'fsBhv') && ~isempty(opts.fsBhv)
        fsBhv = opts.fsBhv;
    elseif isfield(opts, 'fsBvh') && ~isempty(opts.fsBvh)
        fsBhv = opts.fsBvh;
    else
        fsBhv = 60;
    end

    % Durations (seconds) – all configuration is in seconds
    if isfield(opts, 'minDurSeq') && ~isempty(opts.minDurSeq)
        minDurSeq = opts.minDurSeq;
    else
        error('find_matching_sequences:opts', ...
            'opts.minDurSeq (seconds) must be provided.');
    end

    if isfield(opts, 'maxNonBhvDur') && ~isempty(opts.maxNonBhvDur)
        maxNonBhvDur = opts.maxNonBhvDur;
    else
        maxNonBhvDur = 0; % Default: no nonBhv allowed unless specified
    end

    if isfield(opts, 'minDurBtwnSeq') && ~isempty(opts.minDurBtwnSeq)
        minDurBtwnSeq = opts.minDurBtwnSeq;
    else
        minDurBtwnSeq = 0;
    end

    % Convert durations (s) to bin counts
    minLengthBins = max(1, round(minDurSeq * fsBhv));
    maxNonBhvBins = max(0, round(maxNonBhvDur * fsBhv));
    minBtwnSeqBins = max(0, round(minDurBtwnSeq * fsBhv));

    % Initialize variables for search
    matchingSequences = {};
    bhvSequences = {};
    startIndices = [];

    nBins = length(bhvID);
    i = 1;

    while i <= nBins - minLengthBins + 1
        % Require the sequence to start on a possible behavior
        if ~ismember(bhvID(i), possibleBhv)
            i = i + 1;
            continue;
        end

        nonBhvCount = 0;
        bhvCount = 0;
        distinctBhv = [];
        sequenceEnd = i;

        % Grow candidate sequence forward from i
        for j = i:nBins
            if ismember(bhvID(j), possibleBhv)
                bhvCount = bhvCount + 1;
                nonBhvCount = 0;
                distinctBhv = unique([distinctBhv, bhvID(j)]); %#ok<AGROW>
            else
                nonBhvCount = nonBhvCount + 1;
            end

            % If we exceed allowed consecutive nonBhv run, stop before this run
            if nonBhvCount > maxNonBhvBins
                sequenceEnd = j - nonBhvCount;
                break;
            else
                % Extend sequence to current index
                sequenceEnd = j;
            end
        end

        % Check minimum length (in bins)
        if sequenceEnd < i + minLengthBins - 1
            i = i + 1;
            continue;
        end

        lenBins = sequenceEnd - i + 1;
        if lenBins <= 0
            i = i + 1;
            continue;
        end

        bhvPct = bhvCount / lenBins;

        % Check overall criteria
        if bhvPct >= minBhvPct / 100 && length(distinctBhv) >= minDistinctBhv
            iSeq = bhvID(i:sequenceEnd);
            matchingSequences{end+1} = iSeq; %#ok<AGROW>

            % Compress to unique behavior runs
            changeIndices = [true; diff(iSeq) ~= 0];
            bhvSequences{end+1} = iSeq(changeIndices); %#ok<AGROW>

            startIndices(end+1) = i; %#ok<AGROW>
            % Enforce minimum separation between sequences
            i = sequenceEnd + minBtwnSeqBins;
        else
            i = i + 1;
        end
    end
end

