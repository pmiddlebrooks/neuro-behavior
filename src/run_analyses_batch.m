%%
% Sliding Window Analyses - Batch Script
% Loops through multiple sessions and runs multiple sliding window analyses for each
%
% This script can process sessions from:
%   - reach_session_list() for reach task sessions
%   - schall_session_list() for Schall choice countermanding sessions
%   - spontaneous_session_list() for open field sessions 
%
% To add more analyses, add them to the analysesToRun cell array below

% ===== CONFIGURATION =====
% Select which session type to process: 'reach', 'schall', or 'spontaneous'
batchSessionType = 'schall';  % Change this to process different session types
dataSource = 'spikes';  % 'spikes' or 'lfp'
paths = get_paths;

% Define which analyses to run (set to true to run, false to skip)
% Add new analyses here as needed
analysesToRun = struct();
analysesToRun.lzc = false;  % Run lzc analysis
analysesToRun.rqa = false;         % Run RQA analysis
analysesToRun.criticality_ar = true;  % Run criticality AR (d2/mrBr) analysis
analysesToRun.criticality_av = false;  % Run criticality AV (avalanche) analysis
analysesToRun.criticality_lfp = false; % Run criticality LFP analysis
% =========================

% Add paths for session list functions
basePath = fileparts(mfilename('fullpath'));  % src
srcPath = basePath;

reachPath = fullfile(srcPath, 'reach_task');
schallPath = fullfile(srcPath, 'schall');
openFieldPath = fullfile(srcPath, 'spontaneous');

if exist(reachPath, 'dir')
    addpath(reachPath);
end
if exist(schallPath, 'dir')
    addpath(schallPath);
end
if exist(openFieldPath, 'dir')
    addpath(openFieldPath);
end

% Add paths for analysis scripts
complexityScriptPath = fullfile(srcPath, 'complexity', 'scripts');
rqaScriptPath = fullfile(srcPath, 'metastability', 'scripts');
criticalityScriptPath = fullfile(srcPath, 'criticality', 'scripts');

if exist(complexityScriptPath, 'dir')
    addpath(complexityScriptPath);
end
if exist(rqaScriptPath, 'dir')
    addpath(rqaScriptPath);
end
if exist(criticalityScriptPath, 'dir')
    addpath(criticalityScriptPath);
end

% Get list of sessions based on selected type
switch lower(batchSessionType)
    case 'reach'
        sessions = reach_session_list();
        sessionType = 'reach';
    case 'schall'
        % sessions = schall_session_list();
        sessionType = 'schall';
        subjectID = 'broca';  % joule  broca
        searchDir = fullfile(paths.dropPath, 'schall/data', subjectID);  % Set this+ to the directory path containing .mat files
        loadPath = fullfile(searchDir, 'goodSessionsCCM.mat');
        load(loadPath);
        sessions = goodSessionsCCM;
    case 'spontaneous'
        sessions = spontaneous_session_list();
        sessionType = 'spontaneous';
    otherwise
        error('Invalid batchSessionType: %s. Must be ''reach'', ''schall'', or ''spontaneous''.', batchSessionType);
end

numSessions = length(sessions);

% Count how many analyses will be run
analysisNames = fieldnames(analysesToRun);
numAnalyses = sum(cellfun(@(x) analysesToRun.(x), analysisNames));

fprintf('\n=== Sliding Window Analyses Batch Processing ===\n');
fprintf('Session type: %s\n', sessionType);
fprintf('Data source: %s\n', dataSource);
fprintf('Number of sessions: %d\n', numSessions);
fprintf('Analyses to run: ');
activeAnalyses = {};
for i = 1:length(analysisNames)
    if analysesToRun.(analysisNames{i})
        activeAnalyses{end+1} = analysisNames{i};
        fprintf('%s ', analysisNames{i});
    end
end
fprintf('\n');
fprintf('Total analyses per session: %d\n', numAnalyses);
fprintf('\n');

% Initialize results struct array for parfor compatibility
% First create a template struct with all analysis fields
templateStruct = struct();
for a = 1:length(activeAnalyses)
    analysisName = activeAnalyses{a};
    templateStruct.(analysisName) = false;
end
% Replicate the template to create the array
sessionResults = repmat(templateStruct, numSessions, 1);

% % Ensure parallel pool is available
% if isempty(gcp('nocreate'))
%     parpool('local', 4);
% end

% Loop through each session (parallel)
% parfor s = 1:numSessions
for s = 1:numSessions
    sessionName = sessions{s};
    
    fprintf('\n');
    fprintf('%s\n', repmat('=', 1, 80));
    fprintf('Processing session %d/%d: %s\n', s, numSessions, sessionName);
    fprintf('%s\n', repmat('=', 1, 80));
    fprintf('\n');
    
    % Run each analysis
    for a = 1:length(activeAnalyses)
        analysisName = activeAnalyses{a};
        
        fprintf('\n--- Running %s analysis (session %d/%d) ---\n', analysisName, s, numSessions);
        
        try
            % sessionName, sessionType, and dataSource are already set
            % These will be picked up by the analysis scripts
            
            % Run the appropriate analysis script
            switch lower(analysisName)
                case 'lzc'
                    run_lzc_sliding;
                case 'rqa'
                    run_rqa_sliding;
                case 'criticality_ar'
                    run_criticality_ar;
                case 'criticality_av'
                    run_criticality_av;
                case 'criticality_lfp'
                    run_criticality_lfp;
                otherwise
                    error('Unknown analysis: %s', analysisName);
            end
            
            sessionResults(s).(analysisName) = true;
            fprintf('\n✓ %s analysis completed successfully for session %d/%d: %s\n', ...
                analysisName, s, numSessions, sessionName);
            
        catch ME
            sessionResults(s).(analysisName) = false;
            fprintf('\n✗ Error running %s analysis for session %d/%d (%s): %s\n', ...
                analysisName, s, numSessions, sessionName, ME.message);
            fprintf('  Stack trace:\n');
            for i = 1:length(ME.stack)
                fprintf('    %s (line %d)\n', ME.stack(i).name, ME.stack(i).line);
            end
            fprintf('  Continuing with next analysis...\n');
        end
    end
end

% Print summaries for all sessions (after loop completes)
fprintf('\n');
fprintf('%s\n', repmat('=', 1, 80));
fprintf('Batch processing complete: %d sessions processed\n', numSessions);
fprintf('%s\n', repmat('=', 1, 80));
fprintf('\n');

% Print summary for each session
for s = 1:numSessions
    sessionName = sessions{s};
    fprintf('\n--- Session %d/%d Summary: %s ---\n', s, numSessions, sessionName);
    for a = 1:length(activeAnalyses)
        analysisName = activeAnalyses{a};
        if sessionResults(s).(analysisName)
            fprintf('  ✓ %s: Success\n', analysisName);
        else
            fprintf('  ✗ %s: Failed\n', analysisName);
        end
    end
end

fprintf('\n');

