function variableSessionTime = convert_to_session_time(variable, trialOnset)
% Convert trial-relative times to session-relative times
% Inputs:
%   variable - Either a vector (one time per trial) or cell array (multiple times per trial)
%   trialOnset - Vector of trial onset times relative to session start
% Output:
%   variableSessionTime - Single double vector of all times in session-time

if iscell(variable)
    % Case 2: Cell array - multiple times per trial
    variableSessionTime = [];
    for i = 1:length(variable)
        if ~isempty(variable{i})
            % Add trial onset to each time in this trial
            trialTimes = trialOnset(i) + variable{i}(:);  % Ensure column vector
            variableSessionTime = [variableSessionTime; trialTimes];
        end
    end
else
    % Case 1: Vector - one time per trial
    if length(variable) ~= length(trialOnset)
        error('Variable length (%d) must match trialOnset length (%d)', length(variable), length(trialOnset));
    end
    % Add trial onset to each time
    variableSessionTime = trialOnset(:) + variable(:);
end

% Ensure output is a column vector
variableSessionTime = variableSessionTime(:);
end