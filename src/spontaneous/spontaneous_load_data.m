function data = spontaneous_load_data(opts, dataType)

% Load a specific type of data (input: dataType) for analysis

% dataType: behavior:
%
% make a data structure with the start times and durations of each behavior

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
        % Use a time window of recorded data
        getWindow = (1 + opts.fsBhv * opts.collectStart : opts.fsBhv * (opts.collectEnd));
        dataWindow = dataFull(getWindow,:);
        dataWindow.Time = dataWindow.Time - dataWindow.Time(1);
        bhvID = dataWindow.Code;




        changeBhv = [0; diff(bhvID)]; % nonzeros at all the indices when a new behavior begins
        changeBhvIdx = find(changeBhv);



        data = table();
        data.Dur = [diff([0; dataWindow.Time(changeBhvIdx)]); opts.collectEnd - dataWindow.Time(changeBhvIdx(end))];
        data.ID = [bhvID(1); bhvID(changeBhvIdx)];
        % data.Name = bhvName;
        data.Name = [dataWindow.Behavior(1); dataWindow.Behavior(changeBhvIdx)];
        data.StartTime = [0; dataWindow.Time(changeBhvIdx)];
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
        % Find spike data files in session folder
        searchPath = sessionFolder;
        
        % Check for cluster_info.tsv
        clusterInfoPath = fullfile(searchPath, 'cluster_info.tsv');
        if ~exist(clusterInfoPath, 'file')
            error('cluster_info.tsv not found in %s', searchPath);
        end
        ci = readtable(clusterInfoPath, "FileType","text",'Delimiter', '\t');

        % some of the depth values aren't in order, so re-sort the data by
        % depth
        ci = sortrows(ci, 'depth');

        % Reassign depth so 0 is most superficial (M23) and 3840 is deepest
        % (VS)
        ci.depth = 3840 - ci.depth;

        % Flip the ci table so the "top" is M23 and "bottom" is VS
        ci = flipud(ci);

        % Brain area regions as function of depth from surface
        % 0 - 500  motor l2/3
        % 500 - 1240 motor l5/6
        % 1240 - 1540 corpus callosum, where little neural activity expected
        % 1540 - 2700 dorsal striatum
        % 2700 - 3840 ventral striatum

        m23 = [0 500];
        m56 = [501 1240];
        cc = [1241 1540];
        ds = [1541 2700];
        vs = [2701 3840];


        area = cell(size(ci, 1), 1);
        area(ci.depth >= m23(1) & ci.depth <= m23(2)) = {'M23'};
        area(ci.depth >= m56(1) & ci.depth <= m56(2)) = {'M56'};
        area(ci.depth >= cc(1) & ci.depth <= cc(2)) = {'CC'};
        area(ci.depth >= ds(1) & ci.depth <= ds(2)) = {'DS'};
        area(ci.depth >= vs(1) & ci.depth <= vs(2)) = {'VS'};

        ci.area = area;

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

        % Return the requested window of data, formatted  so start time is zero,
        dataWindow = spikeTimes >= opts.collectStart & spikeTimes < (opts.collectEnd);
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
%
