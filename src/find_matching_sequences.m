function [matchingSequences, startIndices, bhvSequences] = find_matching_sequences(bhvID, opts)
%FINDMATCHINGSEQUENCES Finds sequences in a vector that match specified criteria,
% including containing at least 2 different categorical variables from possibleBhv.
%
% Inputs:
%   bhvID - A vector of categorical numerical variables representing behaviors.
%   opts - A structure containing the following fields:
%       possibleBhv - An array of labels that the sequences can contain, must start/end with,
%                     and must include at least 2 different labels from.
%       minLength - The minimum length that qualifies as a sequence.
%       minBhvPct - The minimum percentage of the sequence that must be composed
%                   of labels from possibleBhv.
%       maxNonBhv - The maximum number of consecutive elements not in possibleBhv
%                   allowed within a sequence.
%       minBtwnSeq - The minimum number of elements between separate qualifying sequences.
%
% Outputs:
%   matchingSequences - Cell array, each cell contains a vector representing a sequence
%                       that meets the specified criteria.
%   startIndices - Array containing the starting index within bhvID for each sequence
%                  found in matchingSequences.
%   bhvSequences - Cell array, each cell contains a vector representing a sequence
%                       of the single behavior labels in matchingSequences.


% Unpack options
possibleBhv = opts.possibleBhv;
minLength = opts.minLength;
minBhvPct = opts.minBhvPct;
maxNonBhv = opts.maxNonBhv;
minBtwnSeq = opts.minBtwnSeq;

% Initialize variables for search
matchingSequences = {};
bhvSequences = {};
startIndices = [];
i = 1;

while i <= length(bhvID) - minLength + 1
    sequenceEnd = i;
    nonBhvCount = 0;
    bhvCount = 0;

    % Check if the starting label is within possibleBhv
    if ~ismember(bhvID(i), possibleBhv)
        i = i + 1;
        continue;
    end

    distinctBhv = []; % To track distinct behaviors in possibleBhv within the sequence

    % Explore potential sequence starting from i
    for j = i:length(bhvID)
        if ismember(bhvID(j), possibleBhv)
            bhvCount = bhvCount + 1;
            nonBhvCount = 0; % Reset non-behavior count
            distinctBhv = unique([distinctBhv, bhvID(j)]); % Update distinct behaviors
        else
            nonBhvCount = nonBhvCount + 1;
        end

        % Break conditions
        if nonBhvCount > maxNonBhv
            sequenceEnd = j - nonBhvCount; % Adjust sequence end to exclude non-behavior stretch
            break;
        elseif j == length(bhvID) || sequenceEnd - i + 1 >= minLength && nonBhvCount == 1
            sequenceEnd = j; % Include current if it's the last or to check sequence ending with non-bhv
            break;
        end
    end

    % Ensure the sequence ends with a behavior from possibleBhv
    if ~ismember(bhvID(sequenceEnd), possibleBhv)
        i = i + 1;
        continue;
    end

    % Check if the sequence meets all criteria
    if sequenceEnd - i + 1 >= minLength && ...
            bhvCount / (sequenceEnd - i + 1) >= minBhvPct / 100 && ...
            length(distinctBhv) >= 2 % Ensure at least 2 distinct behaviors from possibleBhv
        iSeq = bhvID(i:sequenceEnd);
        matchingSequences{end+1} = iSeq;
        % Compress the sequence to the pattern of behaviors:
    changeIndices = [true; diff(iSeq) ~= 0];    
    bhvSequences{end+1} = iSeq(changeIndices);

        startIndices(end+1) = i;
        i = sequenceEnd + minBtwnSeq; % Move to the next potential start, ensuring minimum separation
    else
        i = i + 1; % Move to the next start point
    end
end
end
