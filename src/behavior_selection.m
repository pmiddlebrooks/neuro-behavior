function validBhv = behavior_selection(data, opts)
% Get indices of usable behaviors
% Details:
%    None
%
% Syntax:
%     validBhv = behavior_selection(data, opts)
%
% Inputs:
%    data - 
%    opts - 
%
% Outputs:
%    validBhv - [m,n] size,[double] type,Description
%
% Example:
%    None
%
% See also: None


codes = unique(data.ID);
behaviors = {};
for iBhv = 1 : length(codes)
    firstIdx = find(data.ID == codes(iBhv), 1);
    behaviors = [behaviors, data.Name{firstIdx}];
    % fprintf('behavior %d:\t code:%d\t name: %s\n', i, codes(i), dataBhvAlex.Behavior{firstIdx})
end
validBhv = zeros(size(data, 1), 1);

for i = 1 : length(codes) % length(actList)

    iAct = codes(i);

    actIdx = data.ID == iAct; % All instances labeled as this behavior
    allPossible = sum(actIdx);

    longEnough = data.Dur >= opts.minActTime; % Only use if it lasted long enough to count

    actAndLong = actIdx & longEnough;
    andLongEnough = sum(actAndLong);  % for printing sanity check report below

    % iPossible is a list of behavior indices for this behavior that is
    % at least long enough
    % Go through possible instances and discard unusable (repeated) ones
    for iPossible = find(actAndLong)'

        % Was there the same behvaior within the last minNoRepeat sec?
        endTime = [data.StartTime(2:end); data.StartTime(end) + data.Dur(end)];
        % possible repeated behaviors are any behaviors that came
        % before this one that were within the no-repeat minimal time
        iPossRepeat = endTime < data.StartTime(iPossible) & endTime >= (data.StartTime(iPossible) - opts.minNoRepeatTime);

        % sanity checks
        % preBehv = sum(iPossRepeat);


        % If it's within minNoRepeat and any of the behaviors during that time are the same as this one (this behavior is a repeat), get rid of it
        if sum(iPossRepeat) && any(data.ID(iPossRepeat) == iAct)

            % % debug display
            % data.bStart100(iPossible-3:iPossible+3,:)
            % removeTrial = iPossible

            actAndLong(iPossible) = 0;

        end
    end



    andNotRepeated = sum(actAndLong);

    fprintf('%d: %s: Valid: %d\t (%.1f)%%\n', codes(i), behaviors{i}, andNotRepeated, 100 * andNotRepeated / allPossible)

    validBhv(actAndLong) = 1;
end
