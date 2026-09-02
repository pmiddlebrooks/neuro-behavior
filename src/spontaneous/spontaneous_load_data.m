function data = spontaneous_load_data(opts, dataType)

% Load a specific type of data (input: dataType) for analysis

% dataType: behavior:
%
% Make a data structure with absolute start times and durations of each
% behavior bout within opts.collectStart:opts.collectEnd.

% dataType: neuron:
%
sessionFolder = fullfile(opts.dataPath, opts.sessionName);

switch dataType
    case 'behavior'
        % Behavioral data is stored with an asigned B-SOiD label every frame.
        % Find any CSV file that begins with "behavior_labels"
        searchPath = sessionFolder;
        csvFiles = dir(fullfile(searchPath, 'behavior_labels*.csv'));
        
        if isempty(csvFiles)
            error('No CSV file starting with "behavior_labels" found in %s', searchPath);
        elseif length(csvFiles) > 1
            warning('Multiple CSV files starting with "behavior_labels" found. Using first: %s', csvFiles(1).name);
        end
        
        fileName = csvFiles(1).name;
        dataFull = readtable(fullfile(searchPath, fileName));

% if isempty(opts.collectEnd), opts.collectEnd = 
        % Use a time window of recorded data (keep absolute session times)
        collectStart = 0;
        if isfield(opts, 'collectStart') && ~isempty(opts.collectStart)
            collectStart = opts.collectStart;
        end
        getWindow = (1 + opts.fsBhv * collectStart : opts.fsBhv * (opts.collectEnd));
        dataWindow = dataFull(getWindow,:);
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
        % data.StartFrame = bhvStartFrame;

        data.Valid = behavior_selection(data, opts);





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
        % Shared kilosort path: cluster_info/cluster_rf, good/mua/real mask
        data = load_data(opts, 'spikes');

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
%
