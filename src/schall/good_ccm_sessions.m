%%
% GOOD_CCM_SESSIONS - Find or load CCM sessions that meet quality criteria
%
% This script searches a directory for .mat files and identifies sessions that:
% 1. Have taskID == 'ccm'
% 2. Have at least nNeurons variables starting with "spikeUnit"
% 3. Have trialOnset(end) >= minSessDur seconds
%
% Output: goodSessionsCCM - cell array of session names (filenames without extension)

% =============================    Configuration    =============================
% User-provided directory to search
subjectID = 'joule';  % joule  broca
searchDir = fullfile(paths.dropPath, 'schall/data', subjectID);  % Set this to the directory path containing .mat files

%% Find good sessions
% Quality criteria
nNeurons = 30;      % Minimum number of neurons (variables starting with "spikeUnit")
minSessDur = 60*60;   % Minimum session duration in seconds

% =============================    Validation    =============================
if isempty(searchDir)
    error('Please set searchDir to the directory path containing .mat files');
end

if ~exist(searchDir, 'dir')
    error('Directory not found: %s', searchDir);
end

% =============================    Search for .mat files    =============================
fprintf('\n=== Searching for CCM sessions in: %s ===\n', searchDir);

% Get all .mat files in the directory
matFiles = dir(fullfile(searchDir, '*.mat'));
nFiles = length(matFiles);

if nFiles == 0
    warning('No .mat files found in directory: %s', searchDir);
    goodSessionsCCM = {};
    return;
end

fprintf('Found %d .mat file(s)\n', nFiles);

% =============================    Evaluate each file    =============================
goodSessionsCCM = {};
nGood = 0;

for i = 1:nFiles
    fileName = matFiles(i).name;
    filePath = fullfile(searchDir, fileName);
    
    % Get session name (filename without extension)
    [~, sessionName, ~] = fileparts(fileName);
    
    try
        % Load file to check variables (without loading all data)
        fileInfo = whos('-file', filePath);
        varNames = {fileInfo.name};
        
        % Check 1: SessionData.taskID == 'ccm'
        if ~ismember('SessionData', varNames)
            continue;  % Skip if SessionData doesn't exist
        end
        
        % Load only SessionData to check taskID
        temp = load(filePath, 'SessionData');
        if ~isfield(temp.SessionData, 'taskID') || ~strcmp(temp.SessionData.taskID, 'ccm')
            continue;  % Not a CCM session
        end
        
        % Check 2: Count variables starting with "spikeUnit"
        spikeUnitVars = varNames(startsWith(varNames, 'spikeUnit'));
        nSpikeUnits = length(spikeUnitVars);
        
        if nSpikeUnits < nNeurons
            continue;  % Not enough neurons
        end
        
        % Check 3: trialOnset(end) >= minSessDur
        if ~ismember('trialOnset', varNames)
            continue;  % Skip if trialOnset doesn't exist
        end
        
        % Load only trialOnset to check duration
        temp = load(filePath, 'trialOnset');
        if isempty(temp.trialOnset) || temp.trialOnset(end) < minSessDur * 1000
            continue;  % Session too short
        end
        
        % All criteria met - add to good sessions
        nGood = nGood + 1;
        goodSessionsCCM{nGood} = sessionName;
        fprintf('  ✓ %s (nNeurons=%d, duration=%.1fm)\n', sessionName, nSpikeUnits, temp.trialOnset(end)/60/1000);
        
    catch ME
        % Skip files that can't be loaded or don't have required structure
        fprintf('  ✗ %s (error: %s)\n', sessionName, ME.message);
        continue;
    end
end

% =============================    Summary    =============================
fprintf('\n=== Summary ===\n');
fprintf('Total files checked: %d\n', nFiles);
fprintf('Good CCM sessions found: %d\n', nGood);

if nGood > 0
    fprintf('\nGood sessions:\n');
    for i = 1:length(goodSessionsCCM)
        fprintf('  %d. %s\n', i, goodSessionsCCM{i});
    end
    
    % Save the list to the same directory
    savePath = fullfile(searchDir, 'goodSessionsCCM.mat');
    save(savePath, 'goodSessionsCCM', 'nNeurons', 'minSessDur');
    fprintf('\nSaved good sessions list to: %s\n', savePath);
else
    fprintf('No sessions met all criteria.\n');
    % Still save an empty list
    savePath = fullfile(searchDir, 'goodSessionsCCM.mat');
    goodSessionsCCM = {};
    save(savePath, 'goodSessionsCCM', 'nNeurons', 'minSessDur');
    fprintf('Saved empty list to: %s\n', savePath);
end

fprintf('\n=== Done ===\n');


%% Load good sessions
subjectID = 'joule';  % joule  broca
searchDir = fullfile(paths.dropPath, 'schall/data', subjectID);  % Set this to the directory path containing .mat files
loadPath = fullfile(searchDir, 'goodSessionsCCM.mat');
load(loadPath);
disp(goodSessionsCCM')

%% Check for LFP session files
fprintf('\n=== Checking for LFP session files ===\n');

% Get all .mat files in the directory
matFiles = dir(fullfile(searchDir, '*.mat'));
allFileNames = {matFiles.name};

% Check each good session for corresponding LFP file
sessionsWithLfp = {};
sessionsWithoutLfp = {};

for i = 1:length(goodSessionsCCM)
    sessionName = goodSessionsCCM{i};
    
    % Look for files that contain the session name followed by "_lfp"
    % Pattern: sessionName_lfp or sessionName_lfp_*
    lfpPattern = [sessionName, '_lfp'];
    
    % Check if any file matches the pattern
    hasLfp = false;
    lfpFileName = '';
    
    for j = 1:length(allFileNames)
        fileName = allFileNames{j};
        % Check if filename contains the pattern (case-insensitive)
        if contains(fileName, lfpPattern, 'IgnoreCase', true)
            hasLfp = true;
            lfpFileName = fileName;
            break;
        end
    end
    
    if hasLfp
        sessionsWithLfp{end+1} = lfpFileName;
        fprintf('  ✓ %s -> Found LFP file: %s\n', sessionName, lfpFileName);
    else
        sessionsWithoutLfp{end+1} = lfpFileName;
        fprintf('  ✗ %s -> No LFP file found\n', sessionName);
    end
end

% Summary
fprintf('\n=== LFP File Check Summary ===\n');
fprintf('Total good sessions: %d\n', length(goodSessionsCCM));
fprintf('Sessions with LFP files: %d\n', length(sessionsWithLfp));
fprintf('Sessions without LFP files: %d\n', length(sessionsWithoutLfp));

if ~isempty(sessionsWithLfp)
    fprintf('\nSessions with LFP files:\n');
    for i = 1:length(sessionsWithLfp)
        fprintf('  %d. %s\n', i, sessionsWithLfp{i});
    end
end

% if ~isempty(sessionsWithoutLfp)
%     fprintf('\nSessions without LFP files:\n');
%     for i = 1:length(sessionsWithoutLfp)
%         fprintf('  %d. %s\n', i, sessionsWithoutLfp{i});
%     end
% end

% % Store results in workspace
% fprintf('\nVariables created:\n');
% fprintf('  sessionsWithLfp - cell array of sessions with LFP files\n');
% fprintf('  sessionsWithoutLfp - cell array of sessions without LFP files\n');
