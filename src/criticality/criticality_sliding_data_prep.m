%%
% Criticality Sliding Window Data Preparation Script
% Extracts and organizes data loading and preparation code for different data types
% Prepares neural data matrices (spikes) or LFP binned envelopes, brain area indices, and related variables
% Then calls criticality_sliding_window_ar.m (for spikes) or criticality_sliding_ar_lfp.m (for LFP) to perform the analysis

% =============================    Configuration    =============================
% Data type selection
dataType = 'naturalistic';  % 'reach' , 'naturalistic' , 'schall' , 'hong'

% Data source selection
dataSource = 'lfp';  % 'spikes' or 'lfp'

% Initialize paths
paths = get_paths;

% Initialize options structure
opts = neuro_behavior_options;
opts.frameSize = .001;
opts.firingRateCheckTime = 5 * 60;
opts.collectStart = 0;
opts.minFiringRate = .05;
opts.maxFiringRate = 200;

% LFP-specific options (used when dataSource == 'lfp')
if strcmp(dataSource, 'lfp')
    % Frequency bands for LFP analysis
    bands = {'alpha', [8 13]; ...
        'beta', [13 30]; ...
        'lowGamma', [30 50]; ...
        'highGamma', [50 80]};
    
    
    % LFP cleaning parameters
    lfpCleanParams = struct();
    lfpCleanParams.spikeThresh = 4;
    lfpCleanParams.spikeWinSize = 50;
    lfpCleanParams.notchFreqs = [60 120 180];
    lfpCleanParams.lowpassFreq = 300;
    lfpCleanParams.useHampel = true;
    lfpCleanParams.hampelK = 5;
    lfpCleanParams.hampelNsigma = 3;
    lfpCleanParams.detrendOrder = 'linear';
end






%% =============================    Data Loading    =============================
fprintf('\n=== Loading %s %s data ===\n', dataType, dataSource);

% Create filename suffix based on PCA flag (will be set in analysis script)
% This is a placeholder - actual value will be set in criticality_sliding_window_ar.m
filenameSuffix = '';  % Will be updated based on pcaFlag in analysis script

    % =============================    Naturalistic Data Loading    =============================
if strcmp(dataType, 'naturalistic')
    opts.collectEnd = 45 * 60; % seconds
    opts.collectEnd = 5 * 60; % seconds
    
    if strcmp(dataSource, 'spikes')
        % Load naturalistic spike data
        getDataType = 'spikes';
        get_standard_data

        areas = {'M23', 'M56', 'DS', 'VS'};
        idMatIdx = {idM23, idM56, idDS, idVS};
        idLabel = {idLabels(idM23), idLabels(idM56), idLabels(idDS), idLabels(idVS)};
        
        % Initialize spikeData (will be loaded in analysis script if analyzeModulation is true)
        spikeData = [];
        
        % Print summary
        fprintf('%d M23\n%d M56\n%d DS\n%d VS\n', length(idM23), length(idM56), length(idDS), length(idVS));
        
    elseif strcmp(dataSource, 'lfp')
        % Load naturalistic LFP data
        getDataType = 'lfp';
        opts.fsLfp = 1250;
        get_standard_data
        
        % Lowpass filter LFP at 300 Hz
        lfpPerArea = lowpass(lfpPerArea, 300, opts.fsLfp);
        
        % Clean LFP artifacts
        lfpPerArea = clean_lfp_artifacts(lfpPerArea, opts.fsLfp, ...
            'spikeThresh', lfpCleanParams.spikeThresh, ...
            'spikeWinSize', lfpCleanParams.spikeWinSize, ...
            'notchFreqs', lfpCleanParams.notchFreqs, ...
            'lowpassFreq', lfpCleanParams.lowpassFreq, ...
            'useHampel', lfpCleanParams.useHampel, ...
            'hampelK', lfpCleanParams.hampelK, ...
            'hampelNsigma', lfpCleanParams.hampelNsigma, ...
            'detrendOrder', lfpCleanParams.detrendOrder, ...
'visualize', true, ...
'visualizeChannel', 2, ...
'visualizeSamples', 1:5000);

        areas = {'M23', 'M56', 'DS', 'VS'};
    end

    % Create save directory for naturalistic data
    saveDir = fullfile(paths.dropPath, 'criticality/results');
    if ~exist(saveDir, 'dir'); mkdir(saveDir); end

    % Initialize reach-specific variables as empty for naturalistic data
    dataR = [];
    startBlock2 = [];
    reachStart = [];
    reachClass = [];
    sessionName = '';  % Not used for naturalistic data

    % Areas to analyze
    areasToTest = 1:4;

    % =============================    Reach Data Loading    =============================
elseif strcmp(dataType, 'reach')

    % For reach data: specify session name (uncomment and set one)
    % sessionName =  'AB2_28-Apr-2023 17_50_02_NeuroBeh.mat';
    % sessionName =  'AB2_01-May-2023 15_34_59_NeuroBeh.mat';
    % sessionName =  'AB2_11-May-2023 17_31_00_NeuroBeh.mat';
    % sessionName =  'AB2_30-May-2023 12_49_52_NeuroBeh.mat';
    sessionName =  'AB6_27-Mar-2025 14_04_12_NeuroBeh.mat';
    sessionName =  'AB6_29-Mar-2025 15_21_05_NeuroBeh.mat';
    sessionName =  'AB6_02-Apr-2025 14_18_54_NeuroBeh.mat';
    % sessionName =  'AB6_03-Apr-2025 13_34_09_NeuroBeh.mat';
    % sessionName =  'Y4_06-Oct-2023 14_14_53_NeuroBeh.mat';
    % sessionName =  'Y15_26-Aug-2025 12_24_22_NeuroBeh.mat';
    % sessionName =  'Y15_27-Aug-2025 14_02_21_NeuroBeh.mat';
    % sessionName =  'Y15_28-Aug-2025 19_47_07_NeuroBeh.mat';
    sessionName =  'Y17_20-Aug-2025 17_34_48_NeuroBeh.mat';

    % Validate sessionName is provided
    if ~exist('sessionName', 'var') || isempty(sessionName)
        error('sessionName must be defined for reach data. Uncomment and set one of the sessionName lines above.');
    end

    % Load reach data
    reachDataFile = fullfile(paths.reachDataPath, sessionName);

    [~, dataBaseName, ~] = fileparts(reachDataFile);
    saveDir = fullfile(paths.dropPath, 'reach_task/results', dataBaseName);
    if ~exist(saveDir, 'dir'); mkdir(saveDir); end

    dataR = load(reachDataFile);
    reachClass = dataR.Block(:,3);
    reachStart = dataR.R(:,1) / 1000; % Convert from ms to seconds
    startBlock2 = reachStart(find(ismember(reachClass, [3 4]), 1));
    opts.collectEnd = round(min(dataR.R(end,1) + 5000, max(dataR.CSV(:,1)*1000)) / 1000);

    if strcmp(dataSource, 'spikes')
        % Load reach spike data
        [dataMat, idLabels, areaLabels] = neural_matrix_mark_data(dataR, opts);
        areas = {'M23', 'M56', 'DS', 'VS'};
        idM23 = find(strcmp(areaLabels, 'M23'));
        idM56 = find(strcmp(areaLabels, 'M56'));
        idDS = find(strcmp(areaLabels, 'DS'));
        idVS = find(strcmp(areaLabels, 'VS'));
        idMatIdx = {idM23, idM56, idDS, idVS};
        idLabel = {idLabels(idM23), idLabels(idM56), idLabels(idDS), idLabels(idVS)};

        % Initialize spikeData (will be loaded in analysis script if analyzeModulation is true)
        spikeData = [];
        
    elseif strcmp(dataSource, 'lfp')
        % Load reach LFP data
        % Extract base name for LFP file (assumes LFP file has same base name)
        [~, lfpBaseName, ~] = fileparts(sessionName);
        lfpBaseName = strrep(lfpBaseName, '_NeuroBeh', '');  % Remove _NeuroBeh suffix if present
        
        % Try to find LFP file (common patterns)
        reachPath = paths.reachDataPath;
        lfpFile = fullfile(reachPath, [lfpBaseName, '_g0_t0.imec0.lf.bin']);
        metaFile = fullfile(reachPath, [lfpBaseName, '_g0_t0.imec0.lf.meta']);
        
        if ~exist(lfpFile, 'file') || ~exist(metaFile, 'file')
            error('LFP files not found. Expected: %s and %s', lfpFile, metaFile);
        end
        
        % Read metadata (ReadMeta is from SpikeGLX toolbox)
        if ~exist('ReadMeta', 'file')
            error('ReadMeta function not found. Please ensure SpikeGLX toolbox is in path.');
        end
        m = ReadMeta(metaFile);
        opts.fsLfp = round(str2double(m.imSampRate));
        nChan = str2double(m.nSavedChans);
        nSamps = str2double(m.fileSizeBytes)/2/nChan;
        
        % Read LFP data
        mmf = memmapfile(lfpFile, 'Format', {'int16', [nChan nSamps], 'x'});
        lfpPerArea = double(flipud([...
            mmf.Data.x(360,:);...
            mmf.Data.x(280,:);...
            mmf.Data.x(170,:);...
            mmf.Data.x(40,:)...
            ])');
        
        % Clean LFP artifacts
        lfpPerArea = clean_lfp_artifacts(lfpPerArea, opts.fsLfp, ...
            'spikeThresh', lfpCleanParams.spikeThresh, ...
            'spikeWinSize', lfpCleanParams.spikeWinSize, ...
            'notchFreqs', lfpCleanParams.notchFreqs, ...
            'lowpassFreq', lfpCleanParams.lowpassFreq, ...
            'useHampel', lfpCleanParams.useHampel, ...
            'hampelK', lfpCleanParams.hampelK, ...
            'hampelNsigma', lfpCleanParams.hampelNsigma, ...
            'detrendOrder', lfpCleanParams.detrendOrder);
        
        areas = {'M23', 'M56', 'DS', 'VS'};
    end

    % Areas to analyze
    areasToTest = 1:4;




    % =============================    Schall Choice countermanding Data Loading    =============================
elseif strcmp(dataType, 'schall')

    % For schall data: specify session name (uncomment and set one)
    sessionName =  'broca/bp229n02-mm.mat';
    % sessionName =  'broca/bp240n02.mat';
    sessionName =  'joule/jp121n02.mat';
    sessionName =  'joule/jp125n04.mat';
    sessionName = fullfile('joule', goodSessionsCCM{end});

    %     opts.collectStart = 60*60;
    % opts.collectEnd = 105*60;
    % % opts.collectStart = 0*60;
    opts.collectEnd = 45*60;

    % Validate sessionName is provided
    if ~exist('sessionName', 'var') || isempty(sessionName)
        error('sessionName must be defined for schall data. Uncomment and set one of the sessionName lines above.');
    end

    % Load schall data
    schallDataFile = fullfile(paths.schallDataPath, sessionName);

    [~, dataBaseName, ~] = fileparts(schallDataFile);
    saveDir = fullfile(paths.dropPath, 'schall/results', dataBaseName);
    if ~exist(saveDir, 'dir'); mkdir(saveDir); end

    dataS = load(schallDataFile);
    responseOnset = convert_to_session_time(dataS.responseOnset, dataS.trialOnset);
    responseOnset = responseOnset / 1000;
    responseOnset(responseOnset < opts.collectStart) = [];
    responseOnset(responseOnset > opts.collectEnd) = [];

    if strcmp(dataSource, 'spikes')
        % Load schall spike data
        [dataMat, idLabels, areaLabels] = neural_matrix_schall_fef(dataS, opts);
        areas = {'FEF'};
        idFEF = 1:length(idLabels);
        idMatIdx = {idFEF};
        idLabel = {idLabels(idFEF)};

        % Initialize spikeData (will be loaded in analysis script if analyzeModulation is true)
        spikeData = [];

        % Print summary
        fprintf('%d FEF\n', length(idFEF));
        
    elseif strcmp(dataSource, 'lfp')
                        opts.fsLfp = 1000;  % Default fallback

        % Load schall LFP data
     sessionName =  'jp121n02_lfp.mat';
       % Extract subject ID and session name
        [subjectPath, lfpFileName, ~] = fileparts(sessionName);
        lfpDataFile = fullfile(paths.schallDataPath, subjectID, sessionName);

        % Load LFP data
        dataL = load(lfpDataFile);
        
        % Choose which LFP contact to use (typically lfp16)
        if isfield(dataL, 'lfp16')
            lfpPerArea = cell2mat(dataL.lfp16);
        elseif isfield(dataL, 'lfp')
            lfpPerArea = dataL.lfp;
        else
            error('LFP data not found in file. Expected fields: lfp16 or lfp');
        end
        
        % Cut it off based on opts.collectStart/End
        startSample = max(1, round(opts.collectStart * opts.fsLfp) + 1);
        endSample = min(size(lfpPerArea, 1), round(opts.collectEnd * opts.fsLfp));
        lfpPerArea = lfpPerArea(startSample:endSample, :);
        
        % Clean LFP artifacts
        lfpPerArea = clean_lfp_artifacts(lfpPerArea, opts.fsLfp, ...
            'spikeThresh', lfpCleanParams.spikeThresh, ...
            'spikeWinSize', lfpCleanParams.spikeWinSize, ...
            'notchFreqs', lfpCleanParams.notchFreqs, ...
            'lowpassFreq', lfpCleanParams.lowpassFreq, ...
            'useHampel', lfpCleanParams.useHampel, ...
            'hampelK', lfpCleanParams.hampelK, ...
            'hampelNsigma', lfpCleanParams.hampelNsigma, ...
            'detrendOrder', lfpCleanParams.detrendOrder);
        
        areas = {'FEF'};
    end

    % Areas to analyze
    areasToTest = 1;


        % =============================    Kate Hong whisker/lick Data Loading    =============================
elseif strcmp(dataType, 'hong')
paths = get_paths;
% sp: spike depths spikeDepths
load(fullfile(paths.dropPath, 'hong/data', 'spikeData.mat'));

% load T_allUnits: table of single units used
load(fullfile(paths.dropPath, 'hong/data', 'T_allUnits2.mat'));

% Load T: behavior table for the session
load(fullfile(paths.dropPath, 'hong/data', 'behaviorTable.mat'));

opts.collectEnd = min(T.startTime_oe(end)+max(diff(T.startTime_oe)), max(sp.st));

data.sp = sp;
data.ci = T_allUnits;

[dataMat, idLabels, areaLabels, rmvNeurons] = neural_matrix_hong(data, opts);

    areas = {'S1', 'SC'};
    idS1 = find(strcmp(areaLabels, 'S1'));
    idSC = find(strcmp(areaLabels, 'SC'));
    idMatIdx = {idS1, idSC};
    idLabel = {idLabels(idS1), idLabels(idSC)};
    % Print summary
    fprintf('%d S1\n', length(idS1));
    fprintf('%d SC\n', length(idSC));

    saveDir = fullfile(paths.dropPath, 'hong/results');
    if ~exist(saveDir, 'dir'); mkdir(saveDir); end

    % Areas to analyze
    areasToTest = 1:2;


else
    error('Invalid dataType. Must be ''reach'' or ''naturalistic''');


end


%% =============================    Call Analysis Script    =============================
% All data is now loaded into the workspace. Call the appropriate analysis script.

%% ===========================     SPIKING DATA SCRIPTS
%%  D2 sliding window
%  Sliding window size (seconds)
    binSize = .01;
    slidingWindowSize = 2;

    criticality_sliding_window_ar
    %% Avalanche analyses
    % Sliding window and step size (seconds)
    slidingWindowSize = 180;  % seconds - user specified
    avStepSize = 20;          % seconds - user specified

    criticality_sliding_window_av
    
%% ===========================     LFP SCRIPTS
    binSize = .005;
    
    % Compute binned envelopes for each area (store as cell structure)
    % This is done here to avoid redundancy across data types
    numBands = size(bands, 1);
    numAreas = size(lfpPerArea, 2);
    
    if strcmp(dataType, 'schall')
        % Schall: FEF is single area, but may have multiple channels
        if numAreas == 1
            % Single channel - use it directly
            [iBinnedPower, iBinnedEnvelopes, timePoints] = lfp_bin_bandpower(lfpPerArea, opts.fsLfp, bands, binSize, 'cwt');
            binnedEnvelopes = {iBinnedEnvelopes'};  % Cell with single area [nFrames x numBands]
            binnedPower = {iBinnedPower'};  % [nFrames x numBands]
        else
            % Multiple channels - average them
            lfpMean = mean(lfpPerArea, 2);
            [iBinnedPower, iBinnedEnvelopes, timePoints] = lfp_bin_bandpower(lfpMean, opts.fsLfp, bands, binSize, 'cwt');
            binnedEnvelopes = {iBinnedEnvelopes'};  % Cell with single area [nFrames x numBands]
            binnedPower = {iBinnedPower}';  % [nFrames x numBands]
        end
        fprintf('LFP data loaded: %d frames, %d bands\n', size(binnedEnvelopes{1}, 1), numBands);
    else
        % Naturalistic and Reach: multiple areas
        binnedEnvelopes = cell(1, numAreas);
        binnedPower = cell(1, numAreas);
        for iArea = 1:numAreas
            [iBinnedPower, iBinnedEnvelopes, timePoints] = lfp_bin_bandpower(lfpPerArea(:, iArea), opts.fsLfp, bands, binSize, 'cwt');
            binnedEnvelopes{iArea} = iBinnedEnvelopes';  % [nFrames x numBands]
            binnedPower{iArea} = iBinnedPower';  % [nFrames x numBands]
        end
        fprintf('LFP data loaded: %d frames, %d bands/area, %d areas\n', size(binnedEnvelopes{1}, 1), numBands, numAreas);
    end
%%
    % D2 sliding window
    % Sliding window size (seconds)
    binnedEnvelopes = binnedPower;
    slidingWindowSize = 20;     % Window size in seconds
    criticality_sliding_ar_lfp

    %% Avalanche analyses
    % Sliding window and step size (seconds)

