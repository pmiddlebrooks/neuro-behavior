function [dataBhv, bhvIDMat] = curate_behavior_labels(dataBhv, opts)

%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

minRemoveDur = .15; % Remove intervening behavior if it is less than this long (sec)
minFlankDur = .15; % Minimum time for behavior before and after

i = 2;
while i < size(dataBhv, 1)

    % skip if it's in_nest_sleeping_or_irrelevant
    if dataBhv.ID == -1
        i = i + 1;
        continue
    end

    % Check if you need to collapse a sequence of behaviors into one
    % behavior
    if dataBhv.ID(i-1) == dataBhv.ID(i+1) && dataBhv.ID(i-1) ~= dataBhv.ID(i) &&...
            dataBhv.Dur(i-1) >= minFlankDur && dataBhv.Dur(i+1) >= minFlankDur &&...
            dataBhv.Dur(i) <= minRemoveDur
        dataBhv.Dur(i-1) = sum(dataBhv.Dur(i-1:i+1));
        dataBhv.DurFrame(i-1) = floor(dataBhv.Dur(i-1) ./ opts.frameSize);


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

% Create a new bhvIDMat based on the new dataBvh (same code as in
% get_standard_data
nFrame = ceil(opts.collectFor / opts.frameSize);
bhvIDMat = int8(zeros(nFrame, 1));
for i = 1 : size(dataBhv, 1)
    iInd = dataBhv.StartFrame(i) : dataBhv.StartFrame(i) + dataBhv.DurFrame(i) - 1;
    bhvIDMat(iInd) = dataBhv.ID(i);
end

end