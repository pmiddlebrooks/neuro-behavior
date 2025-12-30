%%
% Complexity Sliding Window Analysis - Batch Script
% Loops through multiple sessions and runs complexity analysis for each
%
% This script can process sessions from:
%   - reach_session_list() for reach task sessions
%   - schall_session_list() for Schall choice countermanding sessions
%   - open_field_session_list() for open field sessions

% ===== CONFIGURATION =====
% Select which session type to process: 'reach', 'schall', or 'open_field'
batchSessionType = 'open_field';  % Change this to process different session types
dataSource = 'spikes';  % 'spikes' or 'lfp'
% =========================

% Add paths for session list functions
basePath = fileparts(mfilename('fullpath'));  % complexity/scripts
srcPath = fullfile(basePath, '..', '..');     % src

reachPath = fullfile(srcPath, 'reach_task');
schallPath = fullfile(srcPath, 'schall');
openFieldPath = fullfile(srcPath, 'open_field');

if exist(reachPath, 'dir')
    addpath(reachPath);
end
if exist(schallPath, 'dir')
    addpath(schallPath);
end
if exist(openFieldPath, 'dir')
    addpath(openFieldPath);
end

% Get list of sessions based on selected type
switch lower(batchSessionType)
    case 'reach'
        sessions = reach_session_list();
        sessionType = 'reach';
    case 'schall'
        sessions = schall_session_list();
        sessionType = 'schall';
    case 'open_field'
        sessions = open_field_session_list();
        sessionType = 'naturalistic';
    otherwise
        error('Invalid batchSessionType: %s. Must be ''reach'', ''schall'', or ''open_field''.', batchSessionType);
end

numSessions = length(sessions);

fprintf('\n=== Complexity Analysis Batch Processing ===\n');
fprintf('Session type: %s\n', sessionType);
fprintf('Data source: %s\n', dataSource);
fprintf('Number of sessions: %d\n', numSessions);
fprintf('\n');

% Loop through each session
for s = 3:numSessions
    sessionName = sessions{s};
    
    fprintf('\n');
    fprintf('%s\n', repmat('=', 1, 80));
    fprintf('Processing session %d/%d: %s\n', s, numSessions, sessionName);
    fprintf('%s\n', repmat('=', 1, 80));
    fprintf('\n');
    
    try
        % Clear previous session's workspace variables to avoid conflicts
        % Keep only loop variables and configuration
        varsToKeep = {'sessions', 'numSessions', 's', 'sessionType', 'dataSource', 'sessionName'};
        allVars = who;
        varsToClear = setdiff(allVars, varsToKeep);
        if ~isempty(varsToClear)
            clear(varsToClear{:});
        end
        
        % sessionName, sessionType, and dataSource are already set
        % These will be picked up by run_complexity.m
        
        % Run complexity analysis for this session
        run_complexity;
        
        fprintf('\n✓ Session %d/%d completed successfully: %s\n', s, numSessions, sessionName);
        
    catch ME
        fprintf('\n✗ Error processing session %d/%d (%s): %s\n', s, numSessions, sessionName, ME.message);
        fprintf('  Stack trace:\n');
        for i = 1:length(ME.stack)
            fprintf('    %s (line %d)\n', ME.stack(i).name, ME.stack(i).line);
        end
        fprintf('  Continuing with next session...\n');
    end
end

fprintf('\n');
fprintf('%s\n', repmat('=', 1, 80));
fprintf('Batch processing complete: %d sessions processed\n', numSessions);
fprintf('%s\n', repmat('=', 1, 80));
fprintf('\n');

