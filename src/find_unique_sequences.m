function [uniqueSequences, sequenceIndices] = find_unique_sequences(dataBhv, nSeq, validInd)
% ID: list of categorical behavior IDs
% nSeq: how many behaviors in a sequence?
% validInd: ensure these behaviors in the sequence are "valid" as per behavior_selection.m

if length(validInd) ~= nSeq
    error('validInd needs to be a 1XnSeq logical to determine which index/indices to ensure are "valid"')
end
% Initialize containers to hold sequences and their starting indices
allSequences = {};
allIndices = {};

% Iterate through ID to extract sequences of length n
for i = 1:length(dataBhv.ID) - nSeq + 1
    currentSeq = dataBhv.ID(i:i+nSeq-1); % Extract current sequence of length n


    % Skip any sequences that have NaNs
    if any(isnan(currentSeq)); continue; end
    % Only count the sequence if it's relevant behavior(s) is/are
    % valid.
    % If a behavior needs to be valid but it's not, skip this sequence
    skipSeq = 0;
    for j = 1 : length(validInd)
        if validInd(j) && ~dataBhv.Valid(i + j - 1)
            skipSeq = 1;
        end
        if skipSeq; break; end
    end
    if skipSeq; continue; end

    seqStr = strjoin(arrayfun(@num2str, currentSeq, 'UniformOutput', false), ','); % Convert to string for comparison

    % Check if this sequence is already recorded
    seqIndex = find(strcmp(allSequences, seqStr));
    if isempty(seqIndex)
        % New sequence found, add it to the list
        allSequences{end+1} = seqStr; % Add sequence string
        allIndices{end+1} = [i]; % Initialize with the current starting index
    else
        % Existing sequence, append the new starting index
        allIndices{seqIndex} = [allIndices{seqIndex}, i];
    end
end

% Convert sequences back to original format if necessary
% Here, we split each sequence string back into an array of numbers
uniqueSequences = cellfun(@(x) str2num(strrep(x, ',', ' ')), allSequences, 'UniformOutput', false);

% Output: uniqueSequences contains all unique sequences of length n,
% and sequenceIndices contains the starting indices of each instance of each sequence
sequenceIndices = allIndices;

[~, i] = sort(cellfun(@length, sequenceIndices), 'descend');
uniqueSequences = uniqueSequences(i);
sequenceIndices = sequenceIndices(i);

end
