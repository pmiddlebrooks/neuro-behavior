%%
if strcmp(transOrWithin, 'within') && firstNFrames
withinRange = 1:3;
% withinRange = 2:4;

    % Initialize an empty array to store selected indices
    selectedIndices = [];
    
    % Find the changes in the bhvID vector (where consecutive integers break)
    starts = [find(diff(bhvID) ~= 0) + 1; length(bhvID) + 1];

    % Iterate through each stretch of consecutive integers
    for i = 1:length(starts)-1
        % Find the start and end indices of the current stretch
        % startIdx = starts(i) + withinRange(1) - 1;
        startIdx = starts(i);
        endIdx = starts(i+1) - 1;

        % Length of the current stretch of consecutive integers
        stretchLength = endIdx - startIdx + 1;


        % Edit code below to exclude any stretches shorter than withinRange

        % Check if the stretch is long enough for [x:n]
        if stretchLength >= withinRange(end) - withinRange(1) + 1
            % If yes, take the indices from x to n
            selectedIndices = [selectedIndices, startIdx + (withinRange(1)-1):(startIdx + withinRange(end) - 1)];
        else
            % If not, take the indices from x as close to n as possible
            if withinRange(1) <= stretchLength
                selectedIndices = [selectedIndices, startIdx + (withinRange(1)-1):endIdx];
            end
        end
    end

    % Remove indices that directly precede a change in bhvID
    excludeIdx = starts(2:end) - 1;
    svmInd = setdiff(selectedIndices, excludeIdx);
    svmID = bhvID(svmInd);

        transWithinLabel = [transWithinLabel, ' frame ', num2str(withinRange(1)), '-', num2str(withinRange(end))];

end
%%
if matchTransitionCount
    frameCounts = histcounts(bhvID(preInd + 1));
    warning('Make sure if you are comparing to multiple frames per transition, you account for it in within-behavior')

    for iBhv = 1 : length(frameCounts)
        iBhvInd = find(svmID == iBhv - 2);
        if length(iBhvInd) > frameCounts(iBhv)
            nRemove = length(iBhvInd) - frameCounts(iBhv);
            rmvBhvInd = iBhvInd(randperm(length(iBhvInd), nRemove));
            svmID(rmvBhvInd) = [];
            svmInd(rmvBhvInd) = [];
        end
    end
    transWithinLabel = [transWithinLabel, ' match transitions'];
end


%% IF YOU WANT, GET RID OF DATA FOR WHICH THE BOUTS ARE UNDER A MINIMUM NUMBER OF FRAMES
if minFramePerBout
    % Define the minimum number of frames
    nMinFrames = 6;  % The minimum number of consecutive repetitions

    % Find all unique integers in the vector
    uniqueInts = unique(bhvID);

    % Initialize a structure to hold the result indices for each unique integer
    rmvIndices = zeros(length(bhvID), 1);

    % Loop through each unique integer
    for iBhv = 1:length(uniqueInts)
        targetInt = uniqueInts(iBhv);  % The current integer to check

        % Find the indices where the target integer appears
        indices = find(bhvID == targetInt);

        % Loop through the found indices and check for repetitions
        startIdx = 1;
        while startIdx <= length(indices)
            endIdx = startIdx;

            % Check how long the sequence of consecutive targetInt values is
            while endIdx < length(indices) && indices(endIdx + 1) == indices(endIdx) + 1
                endIdx = endIdx + 1;
            end

            % If the sequence is shorter than nMinFrames, add all its indices to tempIndices
            if (endIdx - startIdx + 1) < nMinFrames
                rmvIndices(indices(startIdx:endIdx)) = 1;
            end

            % Move to the next sequence
            startIdx = endIdx + 1;
        end
    end
    rmvIndices = find(rmvIndices);

    % Remove any frames of behaviors that lasted less than nMinFrames
    rmvSvmInd = intersect(svmInd, rmvIndices);
    svmInd = setdiff(svmInd, rmvSvmInd);
    svmID = bhvID(svmInd);

    transWithinLabel = [transWithinLabel, ', minBoutDur ', num2str(nMinFrames)];
end


%% TWEAK THE BEHAVIOR LABELS IF YOU WANT
if collapseBhv
    % IF YOU WANT, COLLAPSE MULTIPLE BEHAVIOR LABELS INTO A SINGLE LABEL
    % bhvID(bhvID == 2) = 15;

    % Remove all investigate/locomotion/orients
    rmvIndices = find(svmID == 0 | svmID == 1 | svmID == 2 | svmID == 13 | svmID == 14 | svmID == 15);
    svmID(rmvIndices) = [];
    svmInd(rmvIndices) = [];
    % svmInd
    % rmvSvmInd = intersect(svmInd, rmvIndices);
    % svmInd = setdiff(svmInd, rmvSvmInd);
    % svmID = bhvID(svmInd);

    transWithinLabel = [transWithinLabel, ', remove loco-invest-orients'];
end
%% REMOVE ENTIRE BEHAVIORS WITH UNDER A MINIMUM NUMBER OF BOUTS
if minBoutNumber
    nMinBouts = 100;

    % Remove consecutive repetitions
    noRepsVec = svmID([true; diff(svmID) ~= 0]);

    % Count instances of each unique integer (each bout)
    [bhvDataCount, ~] = histcounts(noRepsVec, 'BinMethod', 'integers');

    % bhvBoutCount = histcounts(noRepsVec);
    rmvBehaviors = find(bhvDataCount < nMinBouts) - 2;

    rmvBhvInd = find(ismember(bhvID, rmvBehaviors)); % find bhvID indices to remove
    rmvSvmInd = intersect(svmInd, rmvBhvInd); % find those indices that are also in svmInd
    svmInd = setdiff(svmInd, rmvSvmInd); % the remaining svmInd are the ones to keep (from the original bhvInd)
    svmID = bhvID(svmInd);

    transWithinLabel = [transWithinLabel, ', minBouts ', num2str(nMinBouts)];
    [codes'; bhvDataCount]
end
%% DOWNSAMPLE TO A CERTAIN NUMBER OF BOUTS (THE BEHAVIOR WITH THE MINUMUM NUMBER OF BOUTS)
if downSampleBouts

    % Remove consecutive repetitions
    noRepsVec = svmID([true; diff(svmID) ~= 0]);

    % Count instances of each unique integer (each bout)
    [bhvDataCount, ~] = histcounts(noRepsVec, (min(bhvID)-0.5):(max(bhvID)+0.5));

    % subsampling to match single frame transition number
    downSample = min(bhvDataCount(bhvDataCount > 0));
    for iBhv = 1 : length(bhvDataCount)
        iBhvInd = find(svmID == iBhv - 2);
        if ~isempty(iBhvInd)
            nRemove = length(iBhvInd) - downSample;
            rmvBhvInd = iBhvInd(randperm(length(iBhvInd), nRemove));
            svmID(rmvBhvInd) = [];
            svmInd(rmvBhvInd) = [];
        end
    end
    transWithinLabel = [transWithinLabel, ', downsample to ', num2str(downSample), ' bouts'];
end

%% REMOVE OF ENTIRE BEHAVIORS WITH UNDER A MINIMUM NUMBER OF FRAMES/DATA POINTS
if minFrames
    nMinFrames = 100;

    [uniqueVals, ~, idx] = unique(svmID); % Find unique integers and indices
    bhvDataCount = accumarray(idx, 1); % Count occurrences of each unique integer
    % bhvDataCount = histcounts(svmID, (min(bhvID)-0.5):(max(bhvID)+0.5));
    rmvBehaviors = uniqueVals(bhvDataCount < nMinFrames);

    rmvBhvInd = find(ismember(bhvID, rmvBehaviors));
    rmvSvmInd = intersect(svmInd, rmvBhvInd);
    svmInd = setdiff(svmInd, rmvSvmInd);
    svmID = bhvID(svmInd);

    transWithinLabel = [transWithinLabel, ', minTotalFrames ', num2str(nMinFrames)];
    [uniqueVals'; bhvDataCount']
end

%% DOWNSAMPLE TO A CERTAIN NUMBER OF DATA POINTS
if downSampleFrames
    %     cutoff = 1000;
    % frameCounts = histcounts(svmID);
    % for iBhv = 2 : length(frameCounts)
    %     iBhvInd = find(svmID == iBhv - 2);
    %     if length(iBhvInd) > cutoff
    %         nRemove = length(iBhvInd) - cutoff;
    %         rmvBhvInd = iBhvInd(randperm(length(iBhvInd), nRemove));
    %         svmID(rmvBhvInd) = [];
    %         svmInd(rmvBhvInd) = [];
    %     end
    % end
    % transWithinLabel = ['within-behavior max frames ', num2str(cutoff)];

    % subsampling to match single frame transition number
    [uniqueVals, ~, idx] = unique(svmID); % Find unique integers and indices
    frameCounts = accumarray(idx, 1); % Count occurrences of each unique integer
    downSample = min(frameCounts(frameCounts > 0));
    for iBhv = 1 : length(frameCounts)
        iBhvInd = find(svmID == uniqueVals(iBhv));
        if ~isempty(iBhvInd)
            nRemove = length(iBhvInd) - downSample;
            rmvBhvInd = iBhvInd(randperm(length(iBhvInd), nRemove));
            svmID(rmvBhvInd) = [];
            svmInd(rmvBhvInd) = [];
        end
    end
    transWithinLabel = [transWithinLabel, ', downsample to ', num2str(downSample), ' data points'];

end


