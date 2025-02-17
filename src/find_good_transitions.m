function goodTransitions = find_good_transitions(bhvID, opts)
% % FIND_GOOD_TRANSITIONS - Identify valid transitions into a specific behavior
% %
% % This function detects instances where bhvID transitions into a target behavior
% % (opts.transTo) while ensuring that the previous behavior is one of the allowed behaviors
% % (opts.transFrom) and that the bout lasts at least opts.minBoutDur seconds. Additionally,
% % the preceding behavior must have lasted at least opts.minTransFromDur seconds.
% %
% % INPUTS:
% %   bhvID  - Vector of categorical behavior labels per time bin
% %   opts   - Structure containing the following fields:
% %       .transTo         - Target behavior to transition into
% %       .transFrom       - List of allowed preceding behaviors (vector of behavior labels)
% %       .minBoutDur      - Minimum duration (in seconds) that the new behavior must last
% %       .minTransFromDur - Minimum duration (in seconds) that the preceding behavior must last
% %       .frameSize       - Duration of each frame (time bin size in seconds)
% %
% % OUTPUT:
% %   goodTransitions - Indices of time bins where a "good" transition occurs
% 
% % Convert duration criteria from seconds to frames
% minBoutFrames = round(opts.minBoutDur / opts.frameSize);
% minTransFromFrames = round(opts.minTransFromDur / opts.frameSize);
% 
% % Find where transitions into opts.transTo occur
% transitionPoints = find(diff(bhvID) ~= 0) + 1;
% 
% % Preallocate list of good transitions
% goodTransitions = [];
% 
% % Loop through detected transitions
% for i = 1:length(transitionPoints)
%     idx = transitionPoints(i);
% 
%     % Check if transition is into opts.transTo
%     if bhvID(idx) == opts.transTo
%         % Ensure the preceding behavior is in opts.transFrom
%         if ismember(bhvID(idx-1), opts.transFrom)
%             % Check if the bout of transFrom lasted at least minTransFromFrames
%             boutStart = find(bhvID(1:idx-1) ~= bhvID(idx-1), 1, 'last') + 1;
%             if isempty(boutStart)
%                 boutStart = 1;
%             end
%             transFromDurFrames = idx - boutStart;
% 
%             if transFromDurFrames >= minTransFromFrames
%                 % Check if the bout of transTo lasts at least minBoutFrames
%                 boutEnd = find(bhvID(idx:end) ~= opts.transTo, 1, 'first'); % Find when the behavior ends
% 
%                 % If boutEnd is empty, the behavior lasts until the end of bhvID
%                 if isempty(boutEnd)
%                     boutLengthFrames = length(bhvID) - idx + 1;
%                 else
%                     boutLengthFrames = boutEnd - 1;
%                 end
% 
%                 % Store transition index if the bout duration is sufficient
%                 if boutLengthFrames >= minBoutFrames
%                     goodTransitions = [goodTransitions; idx];
%                 end
%             end
%         end
%     end
% end
% end



% FIND_GOOD_TRANSITIONS - Identify valid transitions into specific behaviors
%
% INPUTS:
%   bhvID  - Vector of categorical behavior labels per time bin
%   opts   - Structure containing the following fields:
%       .transTo         - Vector of target behaviors to transition into
%       .transFrom       - List of allowed preceding behaviors (vector of behavior labels)
%       .minBoutDur      - Minimum duration (in seconds) that the new behavior must last
%       .minTransFromDur - Minimum duration (in seconds) that the preceding behavior must last
%       .frameSize       - Duration of each frame (time bin size in seconds)
%
% OUTPUT:
%   goodTransitions - Indices of time bins where a "good" transition occurs

% Convert duration criteria from seconds to frames
minBoutFrames = round(opts.minBoutDur / opts.frameSize);
minTransFromFrames = round(opts.minTransFromDur / opts.frameSize);

% Find where transitions occur
transitionPoints = find(diff(bhvID) ~= 0) + 1;

% Preallocate list of good transitions
goodTransitions = [];

% Loop through detected transitions
for i = 1:length(transitionPoints)
    idx = transitionPoints(i);

    % Check if transition is into any of the behaviors in opts.transTo
    if ismember(bhvID(idx), opts.transTo)
        % Ensure the preceding behavior is in opts.transFrom
        if ismember(bhvID(idx-1), opts.transFrom)
            % Check if the bout of transFrom lasted at least minTransFromFrames
            boutStart = find(bhvID(1:idx-1) ~= bhvID(idx-1), 1, 'last') + 1;
            if isempty(boutStart)
                boutStart = 1;
            end
            transFromDurFrames = idx - boutStart;
            
            if transFromDurFrames >= minTransFromFrames
                % Check if the bout of transTo lasts at least minBoutFrames
                boutEnd = find(bhvID(idx:end) ~= bhvID(idx), 1, 'first'); % Find when the behavior ends
                
                % If boutEnd is empty, the behavior lasts until the end of bhvID
                if isempty(boutEnd)
                    boutLengthFrames = length(bhvID) - idx + 1;
                else
                    boutLengthFrames = boutEnd - 1;
                end
                
                % Store transition index if the bout duration is sufficient
                if boutLengthFrames >= minBoutFrames
                    goodTransitions = [goodTransitions; idx];
                end
            end
        end
    end
end
end