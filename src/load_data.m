function data = load_data(opts, dataType)

% Load a specific type of data (input: dataType) for analysis

% dataType: behavior:
%
% Make a data structure with absolute start times and durations of each
% behavior bout within opts.collectStart:opts.collectEnd. StartTime is in
% absolute session seconds (not rezeroed to the collect window).

% dataType: neuron:
%


% sessionFolder = fullfile(opts.dataPath, opts.sessionName(1:2), opts.sessionName);
sessionFolder = fullfile(opts.dataPath, opts.sessionName);

switch dataType
    case 'behavior'
        % Frame-wise behavior_labels*.csv (session root, or session/behavior).
        % Interval ethograms are bout tables: session/behavior/bouts.csv.
        [searchPath, behaviorFileKind] = resolve_behavior_file(sessionFolder);
        if strcmp(behaviorFileKind, 'bouts')
            data = load_behavior_bouts(fullfile(searchPath, 'bouts.csv'), opts);
        else
            csvFiles = dir(fullfile(searchPath, 'behavior_labels*.csv'));

            if isempty(csvFiles)
                error('No CSV file starting with "behavior_labels" found in %s', searchPath);
            elseif length(csvFiles) > 1
                warning('Multiple CSV files starting with "behavior_labels" found. Using first: %s', csvFiles(1).name);
            end

            fileName = csvFiles(1).name;
            dataFull = read_delimited_table(fullfile(searchPath, fileName), ',');
            if isempty(opts.collectEnd)
                opts.collectEnd = dataFull.Time(end);
            end

            % Use a time window of recorded data (integer frame indices)
            nFrames = height(dataFull);
            collectStart = 0;
            if isfield(opts, 'collectStart') && ~isempty(opts.collectStart)
                collectStart = opts.collectStart;
            end
            startFrame = max(1, 1 + round(opts.fsBhv * collectStart));
            endFrame = min(nFrames, max(startFrame, round(opts.fsBhv * opts.collectEnd)));
            getWindow = startFrame:endFrame;
            dataWindow = dataFull(getWindow,:);
            % Keep absolute session times (do not rezero to collectStart)
            tAbs = dataWindow.Time;
            bhvID = dataWindow.Code;

            changeBhv = [0; diff(bhvID)]; % nonzeros at all the indices when a new behavior begins
            changeBhvIdx = find(changeBhv);

            data = table();
            if isempty(changeBhvIdx)
                data.StartTime = tAbs(1);
                data.Dur = max(tAbs(end) - tAbs(1), 0);
                data.ID = bhvID(1);
                data.Name = dataWindow.Behavior(1);
            else
                startTimes = [tAbs(1); tAbs(changeBhvIdx)];
                data.StartTime = startTimes;
                data.Dur = [diff(startTimes); max(tAbs(end) - startTimes(end), 0)];
                data.ID = [bhvID(1); bhvID(changeBhvIdx)];
                data.Name = [dataWindow.Behavior(1); dataWindow.Behavior(changeBhvIdx)];
            end

            data.Valid = behavior_selection(data, opts);
        end

    case 'kinematics'
        warning('Adjust kinematics loading in load_data.m to get the path/filename correct')
        % kinFileName = '2021-11-23_13-19-58DLC_resnet50_bottomup_clearSep21shuffle1_700000_kinematics.npy';
        % kinFileName = 'AdenKinematicsAligned.csv';
        % Define the path to your CSV file
        csvFilePath = [opts.dataPath, opts.fileName];
        % kinData = readmatrix(csvFilePath);
        kinData = readNPY(csvFilePath)';

       %          kinFileName = 'AdenKinematicsAligned.csv';
       %   csvFilePath = [opts.bhvDataPath, kinFileName];
       % kData = readmatrix(csvFilePath);


        getWindow = (1 + opts.fsBhv * opts.collectStart : opts.fsBhv * (opts.collectEnd));
        data = kinData(getWindow, :);







    case 'spikes'
        % Find spike data files in session folder
        searchPath = sessionFolder;

        % cluster_rf.tsv if present (quality source of truth), else cluster_info.tsv;
        % depth/area are copied from cluster_info when cluster_rf lacks them
        ci = load_session_cluster_info(searchPath, opts.sessionName);
        if ~ismember('area', ci.Properties.VariableNames)
            error(['Cluster table has no depth/area in %s. ', ...
                'cluster_info.tsv is required for area assignment; ', ...
                'cluster_rf.tsv supplies quality labels only.'], searchPath);
        end

        % Keep good / mua / real units (single mask: cluster_quality_mask.m)
        allGood = cluster_quality_mask(ci, opts);
        ci = ci(allGood, :);

        % Check for spike_times.npy
        spikeTimesPath = fullfile(searchPath, 'spike_times.npy');
        if ~exist(spikeTimesPath, 'file')
            error('spike_times.npy not found in %s', searchPath);
        end
        spikeTimes = readNPY(spikeTimesPath);
        spikeTimes = double(spikeTimes) / opts.fsSpike;
        
        % Check for spike_clusters.npy
        spikeClustersPath = fullfile(searchPath, 'spike_clusters.npy');
        if ~exist(spikeClustersPath, 'file')
            error('spike_clusters.npy not found in %s', searchPath);
        end
        spikeClusters = readNPY(spikeClustersPath);

        if ismember('id', ci.Properties.VariableNames)
            acceptedIds = ci.id;
        else
            acceptedIds = ci.cluster_id;
        end
        keepSpikes = ismember(spikeClusters, acceptedIds);
        spikeTimes = spikeTimes(keepSpikes);
        spikeClusters = spikeClusters(keepSpikes);

        % Drop extra spikes within 1.5 ms on the same unit (good, mua, or real)
        [spikeTimes, spikeClusters] = filter_isi_violations(spikeTimes, spikeClusters);

        if ~isfield(opts, 'collectStart') || isempty(opts.collectStart)
            opts.collectStart = 0;
        end
        if isempty(spikeTimes)
            error('No spikes from accepted units (good/mua/real) in %s', searchPath);
        end
        if ~isfield(opts, 'collectEnd') || isempty(opts.collectEnd)
            opts.collectEnd = max(spikeTimes);
        end

        % Return the requested window of data, formatted  so start time is zero,
        dataWindow = spikeTimes >= opts.collectStart & spikeTimes <= (opts.collectEnd);
        spikeTimes = spikeTimes(dataWindow);
        warning('You changed load_data.m so spikes are not shifted to zero (they load at their actual time). This might affect many analyses that use get_standard_data.m')
        % spikeTimes = spikeTimes - opts.collectStart;
        spikeClusters = spikeClusters(dataWindow);

        data.ci = ci;
        data.spikeTimes = spikeTimes;
        data.spikeClusters = spikeClusters;

    case 'lfp'
        data = readmatrix(fullfile(sessionFolder, 'lfp.txt'));

        data = data(1 + (opts.collectStart * opts.fsLfp) : (opts.collectEnd) * opts.fsLfp, :);

end




%
%
% function validBhv = behavior_selection(data, opts)
% % Get indices of usable behaviors
%
% codes = unique(data.ID);
% behaviors = {};
% for iBhv = 1 : length(codes)
%     firstIdx = find(data.ID == codes(iBhv), 1);
%     behaviors = [behaviors, data.Name{firstIdx}];
%     % fprintf('behavior %d:\t code:%d\t name: %s\n', i, codes(i), dataBhvAlex.Behavior{firstIdx})
% end
% validBhv = zeros(size(data, 1), 1);
%
% for i = 1 : length(codes) % length(actList)
%
%     iAct = codes(i);
%
%     actIdx = data.ID == iAct; % All instances labeled as this behavior
%     allPossible = sum(actIdx);
%
%     longEnough = data.Dur >= opts.minActTime; % Only use if it lasted long enough to count
%
%     actAndLong = actIdx & longEnough;
%     andLongEnough = sum(actAndLong);  % for printing sanity check report below
%
%     % iPossible is a list of behavior indices for this behavior that is
%     % at least long enough
%     % Go through possible instances and discard unusable (repeated) ones
%     for iPossible = find(actAndLong)'
%
%         % Was there the same behvaior within the last minNoRepeat sec?
%         endTime = [data.StartTime(2:end); data.StartTime(end) + data.Dur(end)];
%         % possible repeated behaviors are any behaviors that came
%         % before this one that were within the no-repeat minimal time
%         iPossRepeat = endTime < data.StartTime(iPossible) & endTime >= (data.StartTime(iPossible) - opts.minNoRepeatTime);
%
%         % sanity checks
%         % preBehv = sum(iPossRepeat);
%
%
%         % If it's within minNoRepeat and any of the behaviors during that time are the same as this one (this behavior is a repeat), get rid of it
%         if sum(iPossRepeat) && any(data.ID(iPossRepeat) == iAct)
%
%             % % debug display
%             % data.bStart100(iPossible-3:iPossible+3,:)
%             % removeTrial = iPossible
%
%             actAndLong(iPossible) = 0;
%
%         end
%     end
%
%
%
%     andNotRepeated = sum(actAndLong);
%
%     fprintf('%d: %s: Valid: %d\t (%.1f)%%\n', codes(i), behaviors{i}, andNotRepeated, 100 * andNotRepeated / allPossible)
%
%     validBhv(actAndLong) = 1;
% end
%

end

function [searchPath, behaviorFileKind] = resolve_behavior_file(sessionFolder)
% RESOLVE_BEHAVIOR_FILE - Locate ethogram files for a session
%
% Variables:
%   sessionFolder - Session directory (parent of an optional behavior folder)
%
% Goal:
%   Frame-wise behavior_labels*.csv in the session root stay where they are
%   (spontaneous). Interval task ethograms live in sessionFolder/behavior
%   (behavior_labels*.csv or bouts.csv).
%
% Returns:
%   searchPath       - Folder that contains the behavior file
%   behaviorFileKind - 'labels' or 'bouts'

labelHere = dir(fullfile(sessionFolder, 'behavior_labels*.csv'));
if ~isempty(labelHere)
    searchPath = sessionFolder;
    behaviorFileKind = 'labels';
    return;
end

behaviorDir = fullfile(sessionFolder, 'behavior');
if isfolder(behaviorDir)
    labelThere = dir(fullfile(behaviorDir, 'behavior_labels*.csv'));
    if ~isempty(labelThere)
        searchPath = behaviorDir;
        behaviorFileKind = 'labels';
        return;
    end
    if isfile(fullfile(behaviorDir, 'bouts.csv'))
        searchPath = behaviorDir;
        behaviorFileKind = 'bouts';
        return;
    end
end

if isfile(fullfile(sessionFolder, 'bouts.csv'))
    searchPath = sessionFolder;
    behaviorFileKind = 'bouts';
    return;
end

error('No behavior_labels*.csv or bouts.csv found in %s', sessionFolder);
end

function data = load_behavior_bouts(boutFile, opts)
% LOAD_BEHAVIOR_BOUTS - Bout table from behavior/bouts.csv
%
% Variables:
%   boutFile - Path to bouts.csv (start_s, end_s, behavior)
%   opts     - collectStart, collectEnd; minActTime / minNoRepeatTime for Valid
%
% Goal:
%   Return the same bout table as the frame-wise behavior_labels path:
%   absolute StartTime (s), Dur (s), ID, Name, Valid. IDs are 1..n in order
%   of first appearance in the file.

fprintf('Loading behavior bouts: %s\n', boutFile);
boutTable = read_delimited_table(boutFile, ',');
startS = boutTable.start_s(:);
endS = boutTable.end_s(:);
behaviorNames = boutTable.behavior;
if isstring(behaviorNames)
    behaviorNames = cellstr(behaviorNames);
elseif ischar(behaviorNames)
    behaviorNames = cellstr(behaviorNames);
end
behaviorNames = behaviorNames(:);

[~, ~, nameIds] = unique(behaviorNames, 'stable');

collectStart = 0;
if isfield(opts, 'collectStart') && ~isempty(opts.collectStart)
    collectStart = opts.collectStart;
end
collectEnd = [];
if isfield(opts, 'collectEnd')
    collectEnd = opts.collectEnd;
end
if isempty(collectEnd)
    collectEnd = max(endS);
end

keepBout = endS > collectStart & startS <= collectEnd;
startS = max(startS(keepBout), collectStart);
endS = min(endS(keepBout), collectEnd);
behaviorNames = behaviorNames(keepBout);
nameIds = nameIds(keepBout);
positiveDur = endS > startS;
startS = startS(positiveDur);
endS = endS(positiveDur);
behaviorNames = behaviorNames(positiveDur);
nameIds = nameIds(positiveDur);

data = table();
data.StartTime = startS;
data.Dur = endS - startS;
data.ID = nameIds;
data.Name = behaviorNames;
if isempty(data) || ~isfield(opts, 'minActTime') || ~isfield(opts, 'minNoRepeatTime')
    data.Valid = ones(height(data), 1);
else
    data.Valid = behavior_selection(data, opts);
end
end
