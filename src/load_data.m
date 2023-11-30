function data = load_data(opts, dataType)

% Load a specific type of data (input: dataType) for analysis

% dataType: behavior:
% 
% make a data structure with the start times and durations of each behavior

% dataType: neuron:
% 

if strcmp(dataType, 'behavior')
    % Behavioral data is stored with an asigned B-SOiD label every frame.
    % dataBhv = load([opts.dataPath ,opts.fileName]);

    dataFull = readtable([opts.dataPath, opts.fileName]);


    % Use a time window of recorded data
    getWindow = (1 + opts.fsBhv * opts.collectStart : opts.fsBhv * (opts.collectStart + opts.collectFor));
    dataWindow = dataFull(getWindow,:);
    dataWindow.Time = dataWindow.Time - dataWindow.Time(1);
    bhvID = dataWindow.Code;




    changeBhv = [0; diff(bhvID)]; % nonzeros at all the indices when a new behavior begins
    changeBhvIdx = find(changeBhv ~= 0);


    % % Convert the data to frame time as per opts.frameSize
    % 
    % bhvStartFrame = [1; ceil(changeBhvIdx / opts.fsBhv / opts.frameSize)];
    % bhvStartTime = (bhvStartFrame - 1) * opts.frameSize;
    % 
    % bhvDur = [diff(bhvStartFrame); opts.collectFor / opts.frameSize - bhvStartFrame(end) + 1]; % How many frames did behavior last (frame rate / duration in sec)
    % bhvID = [bhvID(1); bhvID(changeBhvIdx)]; % Which behavior it was
    % bhvName = [dataWindow.Behavior(1); dataWindow.Behavior(changeBhvIdx)]; % Which behavior it was

    data = table();
    data.bhvDur = [diff([0; dataWindow.Time(changeBhvIdx)]); opts.collectFor - dataWindow.Time(changeBhvIdx(end))];
    data.bhvID = [bhvID(1); bhvID(changeBhvIdx)];
    % data.bhvName = bhvName;
    data.bhvName = [dataWindow.Behavior(1); dataWindow.Behavior(changeBhvIdx)];
    data.bhvStartTime = [0; dataWindow.Time(changeBhvIdx)];
    % data.bhvStartFrame = bhvStartFrame;

elseif strcmp(dataType, 'neuron')
    fileName = 'cluster_info.tsv';
    ci = readtable([opts.dataPath, fileName], "FileType","text",'Delimiter', '\t');

    spikeTimes = readNPY([opts.dataPath, 'spike_times.npy']);
    spikeTimes = double(spikeTimes) / opts.fsSpike;
    spikeClusters = readNPY([opts.dataPath, 'spike_clusters.npy']);

    % Return the requested window of data, formatted  so start time is zero,
    dataWindow = spikeTimes >= opts.collectStart & spikeTimes < (opts.collectStart + opts.collectFor);
    spikeTimes = spikeTimes(dataWindow);
    spikeTimes = spikeTimes - opts.collectStart;
    spikeClusters = spikeClusters(dataWindow);

    data.ci = ci;
    data.spikeTimes = spikeTimes;
    data.spikeClusters = spikeClusters;

elseif strcmp(dataType, 'lfp')
    % dataNeuro =

end