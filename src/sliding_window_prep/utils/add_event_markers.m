function add_event_markers(dataStruct, startS, varargin)
% ADD_EVENT_MARKERS Add event markers to current axes for sliding window plots
%
% Variables:
%   dataStruct - Data structure from load_sliding_window_data() with fields:
%     .sessionType - 'reach', 'hong', 'schall', 'naturalistic'
%     .reachStart - Vector of reach onset times (for 'reach' sessions)
%     .startBlock2 - Block 2 start time (for 'reach' sessions)
%     .T - Table with startTime_oe and trialType fields (for 'hong' sessions)
%     .responseOnset - Vector of response onset times (for 'schall' sessions)
%   startS - Cell array of time vectors (one per area), or single time vector
%   varargin - Optional name-value pairs:
%     'firstNonEmptyArea' - Index of first non-empty area (default: find first non-empty)
%     'areaIdx' - Index of specific area to use (optional, for single-area plots)
%     'dataSource' - 'spikes' or 'lfp' (optional, for filtering)
%     'numAreas' - Number of areas (optional, for validation)
%
% Goal:
%   Add appropriate event markers (reach onsets, block 2, trial types, response onsets)
%   to the current axes based on session type. Uses the time range from startS
%   to filter events to only those within the plot range.
%
% Returns:
%   None (modifies current axes)

    % Parse optional arguments
    p = inputParser;
    addParameter(p, 'firstNonEmptyArea', [], @(x) isnumeric(x) || isempty(x));
    addParameter(p, 'areaIdx', [], @(x) isnumeric(x) || isempty(x));
    addParameter(p, 'dataSource', '', @ischar);
    addParameter(p, 'numAreas', [], @(x) isnumeric(x) || isempty(x));
    parse(p, varargin{:});
    
    firstNonEmptyArea = p.Results.firstNonEmptyArea;
    areaIdx = p.Results.areaIdx;
    dataSource = p.Results.dataSource;
    numAreas = p.Results.numAreas;
    
    % Determine which startS to use for plotTimeRange
    if ~isempty(areaIdx) && isnumeric(areaIdx)
        % Use specific area index
        if iscell(startS) && areaIdx <= length(startS) && ~isempty(startS{areaIdx})
            plotTimeRange = [startS{areaIdx}(1), startS{areaIdx}(end)];
        else
            return;  % No valid time range for this area
        end
    else
        % Use first non-empty area
        if isempty(firstNonEmptyArea)
            % Find first non-empty area
            if iscell(startS)
                firstNonEmptyArea = find(~cellfun(@isempty, startS), 1);
            else
                firstNonEmptyArea = 1;
            end
        end
        
        if ~isempty(firstNonEmptyArea)
            if iscell(startS) && firstNonEmptyArea <= length(startS) && ~isempty(startS{firstNonEmptyArea})
                plotTimeRange = [startS{firstNonEmptyArea}(1), startS{firstNonEmptyArea}(end)];
            elseif ~iscell(startS) && ~isempty(startS)
                plotTimeRange = [startS(1), startS(end)];
            else
                return;  % No valid time range
            end
        else
            return;  % No valid time range
        end
    end
    
    % Get session type
    if ~isfield(dataStruct, 'sessionType')
        return;  % Cannot determine session type
    end
    sessionType = dataStruct.sessionType;
    
    % Add reach onsets and block 2 if applicable
    if strcmp(sessionType, 'reach') && isfield(dataStruct, 'reachStart')
        % Optional: filter by dataSource if provided
        if ~isempty(dataSource) && strcmp(dataSource, 'spikes')
            % Only add for spike data (as in RQA version)
            if isempty(numAreas) || numAreas > 0
                if ~isempty(dataStruct.reachStart)
                    reachOnsetsInRange = dataStruct.reachStart(...
                        dataStruct.reachStart >= plotTimeRange(1) & dataStruct.reachStart <= plotTimeRange(2));
                    
                    if ~isempty(reachOnsetsInRange)
                        for i = 1:length(reachOnsetsInRange)
                            xline(reachOnsetsInRange(i), 'Color', [0.5 0.5 0.5], 'LineWidth', 0.8, ...
                                'LineStyle', '--', 'Alpha', 0.7, 'HandleVisibility', 'off');
                        end
                        if isfield(dataStruct, 'startBlock2') && ~isempty(dataStruct.startBlock2)
                            xline(dataStruct.startBlock2, 'Color', [1 0 0], 'LineWidth', 3, ...
                                'HandleVisibility', 'off');
                        end
                    end
                end
            end
        else
            % Add for all data sources (for backward compatibility)
            if ~isempty(dataStruct.reachStart)
                reachOnsetsInRange = dataStruct.reachStart(...
                    dataStruct.reachStart >= plotTimeRange(1) & dataStruct.reachStart <= plotTimeRange(2));
                
                if ~isempty(reachOnsetsInRange)
                    for i = 1:length(reachOnsetsInRange)
                        xline(reachOnsetsInRange(i), 'Color', [0.5 0.5 0.5], 'LineWidth', 0.8, ...
                            'LineStyle', '--', 'Alpha', 0.7, 'HandleVisibility', 'off');
                    end
                    if isfield(dataStruct, 'startBlock2') && ~isempty(dataStruct.startBlock2)
                        xline(dataStruct.startBlock2, 'Color', [1 0 0], 'LineWidth', 3, ...
                            'HandleVisibility', 'off');
                    end
                end
            end
        end
    end
    
    % Add hong trial type fills if applicable
    if strcmp(sessionType, 'hong')
        if isfield(dataStruct, 'T') && ~isempty(dataStruct.T.startTime_oe)            % Get unique trial types and assign colors
            uniqueTrialTypes = unique(dataStruct.T.trialType);
            nTypes = length(uniqueTrialTypes);
            
            % Create color map for different trial types
            % Use distinct colors for up to 8 types, then cycle
            baseColors = [
                1 1 0;      % Yellow
                1 0 1;      % Magenta
                0 1 1;      % Cyan
                0.5 0 0.5;  % Purple
            ];
            
            % Plot horizontal fills for each trial type
            yLevel = 1.03;
            startTimes = dataStruct.T.startTime_oe;
            trialTypes = dataStruct.T.trialType;
            
            % Filter to plot time range
            inRange = startTimes >= plotTimeRange(1) & startTimes <= plotTimeRange(2);
            if any(inRange)
                filteredStartTimes = startTimes(inRange);
                filteredTrialTypes = trialTypes(inRange);
                
                % Sort by start time
                [sortedStartTimes, sortIdx] = sort(filteredStartTimes);
                sortedTrialTypes = filteredTrialTypes(sortIdx);
                
                % Plot fills for each trial type segment
                for i = 1:length(sortedStartTimes)
                    trialType = sortedTrialTypes(i);
                    trialStart = sortedStartTimes(i);
                    
                    % Determine trial end (next start time or end of plot range)
                    if i < length(sortedStartTimes)
                        trialEnd = sortedStartTimes(i + 1);
                    else
                        trialEnd = plotTimeRange(2);
                    end
                    
                    % Get color for this trial type
                    typeIdx = find(uniqueTrialTypes == trialType, 1);
                    if ~isempty(typeIdx)
                        colorIdx = mod(typeIdx - 1, size(baseColors, 1)) + 1;
                        trialColor = baseColors(colorIdx, :);
                        
                        % Plot horizontal fill
                        fill([trialStart, trialEnd, trialEnd, trialStart], ...
                            [yLevel, yLevel, yLevel + 0.01, yLevel + 0.01], ...
                            trialColor, 'EdgeColor', 'none', 'FaceAlpha', 0.4, ...
                            'HandleVisibility', 'off');
                    end
                end
            end
        end
    end
    
    % Add schall response onsets if applicable
    if strcmp(sessionType, 'schall') && isfield(dataStruct, 'responseOnset')
        % Optional: filter by dataSource if provided
        if ~isempty(dataSource) && strcmp(dataSource, 'spikes')
            % Only add for spike data (as in RQA version)
            if isempty(numAreas) || numAreas > 0
                if ~isempty(dataStruct.responseOnset)
                    responseOnsetsInRange = dataStruct.responseOnset(...
                        dataStruct.responseOnset >= plotTimeRange(1) & dataStruct.responseOnset <= plotTimeRange(2));
                    
                    if ~isempty(responseOnsetsInRange)
                        for i = 1:length(responseOnsetsInRange)
                            xline(responseOnsetsInRange(i), 'Color', [0.5 0.5 0.5], 'LineWidth', 0.8, ...
                                'LineStyle', '--', 'Alpha', 0.7, 'HandleVisibility', 'off');
                        end
                    end
                end
            end
        else
            % Add for all data sources (for backward compatibility)
            if ~isempty(dataStruct.responseOnset)
                responseOnsetsInRange = dataStruct.responseOnset(...
                    dataStruct.responseOnset >= plotTimeRange(1) & dataStruct.responseOnset <= plotTimeRange(2));
                
                if ~isempty(responseOnsetsInRange)
                    for i = 1:length(responseOnsetsInRange)
                        xline(responseOnsetsInRange(i), 'Color', [0.5 0.5 0.5], 'LineWidth', 0.8, ...
                            'LineStyle', '--', 'Alpha', 0.7, 'HandleVisibility', 'off');
                    end
                end
            end
        end
    end
end

