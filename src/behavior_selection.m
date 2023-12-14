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



behaviors = opts.behaviors;
codes = opts.bhvCodes;



% bhvEventType = 3 * ones(length(behaviors), 1); % different type of events. these are all peri-event variables.

validBhv = cell(1, length(behaviors));

for i = 1 : length(codes) % length(actList)

    if ismember(codes(i), opts.validCodes)
        iAct = codes(i);

        actIdx = data.bhvID == iAct; % All instances labeled as this behavior
        allPossible = sum(actIdx);

        longEnough = data.bhvDur >= opts.minActTime; % Only use if it lasted long enough to count

        actAndLong = actIdx & longEnough;
        andLongEnough = sum(actAndLong);  % for printing sanity check report below

        % iPossible is a list of behavior indices for this behavior that is
        % at least long enough
        % Go through possible instances and discard unusable (repeated) ones
        for iPossible = find(actAndLong)'

            % Was there the same behvaior within the last minNoRepeat sec?
            endTime = [data.bhvStartTime(2:end); data.bhvStartTime(end) + data.bhvDur(end)];
            % possible repeated behaviors are any behaviors that came
            % before this one that were within the no-repeat minimal time
            iPossRepeat = endTime < data.bhvStartTime(iPossible) & endTime >= (data.bhvStartTime(iPossible) - opts.minNoRepeatTime);

            % sanity checks
            % preBehv = sum(iPossRepeat);


            % If it's within minNoRepeat and any of the behaviors during that time are the same as this one (this behavior is a repeat), get rid of it
            if sum(iPossRepeat) && any(data.bhvID(iPossRepeat) == iAct)

                % % debug display
                % data.bStart100(iPossible-3:iPossible+3,:)
                % removeTrial = iPossible

                actAndLong(iPossible) = 0;

            end
        end



        andNotRepeated = sum(actAndLong);

        fprintf('Behavior %d: %s\n', codes(i), behaviors{i})
        fprintf('%d: allPossible\n', allPossible)
        fprintf('%d: andLongEnough\n', andLongEnough)
        fprintf('%d: andNotRepeated \n', andNotRepeated)
        fprintf('Percent valid: %.1f\n\n', 100* andNotRepeated / allPossible)

        if sum(actAndLong) >= opts.minBhvNum
            validBhv{i} = actAndLong;
        else
            validBhv{i} = false(length(actAndLong), 1);

            fprintf('Not enough %s bouts to analyze (%d of %d needed)\n', behaviors{i}, sum(actAndLong), opts.minBhvNum)

        end
    else
        validBhv{i} = false(size(data, 1), 1);

        fprintf('%s code %d is not a valid behavior for this analysis\n\n', behaviors{i}, codes(i))
    end
end
