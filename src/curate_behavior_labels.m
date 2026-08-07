function [dataBhv, bhvID] = curate_behavior_labels(dataBhv, opts)

%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

minRemoveDur = .15; % Remove intervening behavior if it is less than this long (sec)
minFlankDur = .15; % Minimum time for behavior before and after

i = 2;
while i < size(dataBhv, 1)

    % skip if it's in_nest_sleeping_or_irrelevant
    if dataBhv.ID(i) == -1
        i = i + 1;
        continue
    end

    % Check if you need to collapse a sequence of behaviors into one
    % behavior
    if dataBhv.ID(i-1) == dataBhv.ID(i+1) && dataBhv.ID(i-1) ~= dataBhv.ID(i) &&...
            dataBhv.Dur(i-1) >= minFlankDur && dataBhv.Dur(i+1) >= minFlankDur &&...
            dataBhv.Dur(i) <= minRemoveDur

        dataBhv.Dur(i-1) = sum(dataBhv.Dur(i-1:i+1));
        % dataBhv.DurFrame(i-1) = floor(dataBhv.Dur(i-1) ./ opts.frameSize);


        % Remove the rows that were collapsed into the first row behavior
        dataBhv(i:i+1,:) = [];

        % Adjust the index to the next row after deleting the relevant rows
        % i = i - 1;
    else
        i = i + 1;
    end
end

% Reclassify Valid behaviors based on new dataBhv
dataBhv.Valid = behavior_selection(dataBhv, opts);

% Recalculate StartFrame in collect-window coordinates (absolute StartTime)
collectStart = 0;
if isfield(opts, 'collectStart') && ~isempty(opts.collectStart)
    collectStart = opts.collectStart;
end
dataBhv.StartFrame = abs_time_to_collect_frame( ...
    dataBhv.StartTime, collectStart, opts.frameSize, 'floor');
nFrame = ceil((opts.collectEnd - collectStart) / opts.frameSize);

% Re-Create bhvIDMat, a vector of ID labels, one element per frame to match the
% neural matrix (see get_standard_data.m)

% Use StartFrame method (clamp to collect window)
bhvID = int8(zeros(nFrame, 1));
if size(dataBhv, 1) < 1 || nFrame < 1
    return;
end
for i = 1 : size(dataBhv, 1) - 1
    iStart = max(1, dataBhv.StartFrame(i));
    iEnd = min(nFrame, dataBhv.StartFrame(i+1) - 1);
    if iStart <= iEnd
        bhvID(iStart:iEnd) = dataBhv.ID(i);
    end
end
iStart = max(1, dataBhv.StartFrame(end));
if ismember('Dur', dataBhv.Properties.VariableNames)
    iEnd = min(nFrame, abs_time_to_collect_frame( ...
        dataBhv.StartTime(end) + dataBhv.Dur(end), collectStart, opts.frameSize, 'floor'));
else
    iEnd = nFrame;
end
if iStart <= iEnd
    bhvID(iStart:iEnd) = dataBhv.ID(end);
end


% Use majority time in frame method:

% % 1. Create a vector of ID labels, one element per behavioral frame
% % rate
% timeWindows = 0 : 1/opts.fsBhv : dataBhv.StartTime(end) + dataBhv.Dur(end);
% idFull = int8(zeros(length(timeWindows), 1));
% for i = 1 : length(timeWindows) - 1
%     % Find behavior active during the current time window
%     idFull(i) = dataBhv.ID(dataBhv.StartTime <= timeWindows(i) & dataBhv.StartTime + dataBhv.Dur > timeWindows(i));
% end
% idFull(end) = dataBhv.ID(end);
%
% % 2. Find majority behavior in each opts.frameSize window
% for i = 1 : size(bhvIDMat, 1)
%     % if i == 100
%     %     disp('here')
%     % end
%     iStartTime = (i-1) * opts.frameSize;
%     iStopTime = i * opts.frameSize;
%     iStartFrame = round(1 + iStartTime * opts.fsBhv);
%     iStopFrame = round(iStopTime * opts.fsBhv);
%     iID = idFull(iStartFrame : iStopFrame);
%     bhvIDMat(i) = mode(iID);
% end

end