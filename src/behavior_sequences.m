function [seqStartTimes, seqCodes, seqNames] = behavior_sequences(dataBhv, analyzeCodes, analyzeBhv)
%    For each behavior, what are the most common preceding behaviors?

dataBhv.prevID = [nan; dataBhv.bhvID(1:end-1)];
dataBhv.prevDur = [nan; dataBhv.bhvDur(1:end-1)];
dataBhv.prevStartTime = [nan; dataBhv.bhvStartTime(1:end-1)];
dataBhvTruncate = dataBhv(3:end-3, :); % Truncate a few behaviors so we can look back and ahead in time a bit

minCurrDur = .15;
minBeforeDur = .15; % previous behavior must last within a range (sec)
maxBeforeDur = 2;
minNumBoutsPrev = 20;

seqNames = {};
seqCodes = [];
seqStartTimes = {};
for iCurr = 1 : length(analyzeCodes)

    % For each behavior:
    % - Get the noise correlations for the current behavior (like above)
    % and all (valid) previous behaviors
    for jPrev = 1 : length(analyzeCodes)
        if iCurr ~= jPrev
            % Make sure this sequence passes requisite criteria
            currIdx = dataBhvTruncate.bhvID == analyzeCodes(iCurr) & ...
                dataBhvTruncate.bhvDur > minCurrDur;
            goodSeqIdx = currIdx & ...
                dataBhvTruncate.prevID == analyzeCodes(jPrev) & ...
                dataBhvTruncate.prevDur >= minBeforeDur & ...
                dataBhvTruncate.prevDur <= maxBeforeDur;
            if sum(goodSeqIdx)
                seqCodes = [seqCodes; [analyzeCodes(jPrev), analyzeCodes(iCurr)]];
                seqNames = [seqNames; [analyzeBhv{jPrev}, ' then ', analyzeBhv{iCurr}]];
                seqStartTimes = [seqStartTimes; [dataBhvTruncate.prevStartTime(goodSeqIdx), dataBhvTruncate.bhvStartTime(goodSeqIdx)]];
            end
        end

    end
end

% Sort them from most to least
nTrial = cell2mat(cellfun(@(x) size(x, 1), seqStartTimes, 'UniformOutput', false));

[~, i] = sort(nTrial, 'descend');
nTrial = nTrial(i);
seqNames = seqNames(i);
seqCodes = seqCodes(i,:);
seqStartTimes = seqStartTimes(i);