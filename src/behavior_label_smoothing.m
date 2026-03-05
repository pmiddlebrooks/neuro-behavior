function smoothedBhvID = behavior_label_smoothing(bhvID, opts)
% BEHAVIOR_LABEL_SMOOTHING Smooth categorical behavior labels by mode over a time window.
%
% Variables:
%   bhvID    - Vector of categorical behavior labels (integer or numeric codes).
%   opts     - Struct with fields:
%              .fsBhv           - Sampling rate of behavior data (Hz). Required.
%              .smoothingWindow - Time window in seconds over which to smooth. Required.
%              .summarize       - If true, print summary of re-labeled proportion and
%                                most common label switches (default false).
%
% Goal:
%   Apply sliding-window mode smoothing: at each sample, replace the label with the
%   mode (most frequent label) over a window of duration smoothingWindow centered on
%   that sample. At edges, the window is truncated to available samples.
%
% Returns:
%   smoothedBhvID - Vector of smoothed behavior labels (same size as bhvID).
%
% See also: spontaneous_behavior_sequences

    % Validate inputs
    if nargin < 2 || isempty(opts) || ~isstruct(opts)
        error('behavior_label_smoothing:opts', 'opts struct is required.');
    end
    if ~isfield(opts, 'fsBhv') || isempty(opts.fsBhv)
        error('behavior_label_smoothing:opts', 'opts.fsBhv is required.');
    end
    if ~isfield(opts, 'smoothingWindow') || isempty(opts.smoothingWindow)
        error('behavior_label_smoothing:opts', 'opts.smoothingWindow is required.');
    end

    wasRow = isrow(bhvID);
    bhvID = bhvID(:);
    nSamp = length(bhvID);
    nWin = round(opts.smoothingWindow * opts.fsBhv);
    nWin = max(1, nWin);
    halfWin = floor(nWin / 2);

    summarizeFlag = isfield(opts, 'summarize') && ~isempty(opts.summarize) && opts.summarize;

    % Sliding window mode at each sample
    smoothedBhvID = bhvID;
    for i = 1:nSamp
        low = max(1, i - halfWin);
        high = min(nSamp, i + halfWin);
        win = bhvID(low:high);
        smoothedBhvID(i) = mode(win);
    end

    if wasRow
        smoothedBhvID = smoothedBhvID';
    end

    if ~summarizeFlag
        return;
    end

    % Summary: proportion re-labeled and most common switches
    changed = (smoothedBhvID(:) ~= bhvID(:));
    nChanged = sum(changed);
    propRelabeled = nChanged / nSamp;

    fprintf('\n--- Behavior label smoothing summary ---\n');
    fprintf('  Proportion re-labeled: %d / %d (%.2f%%)\n', nChanged, nSamp, 100 * propRelabeled);

    if nChanged == 0
        fprintf('  No label switches to report.\n\n');
        return;
    end

    % Most common (old -> new) switches among changed samples
    oldLabels = bhvID(changed);
    newLabels = smoothedBhvID(changed);
    pairs = [oldLabels, newLabels];
    [uniquePairs, ~, ic] = unique(pairs, 'rows');
    counts = accumarray(ic, 1);

    [countsSorted, sortIdx] = sort(counts, 'descend');
    nReport = min(10, length(countsSorted));

    fprintf('  Most common label switches (old -> new):\n');
    for k = 1:nReport
        row = uniquePairs(sortIdx(k), :);
        fprintf('    %g -> %g  (count: %d)\n', row(1), row(2), countsSorted(k));
    end
    fprintf('\n');
end
