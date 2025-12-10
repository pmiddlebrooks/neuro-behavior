%%
% Criticality Sliding Window Data Preparation Script
% Extracts and organizes data loading and preparation code for different data types
% Prepares neural data matrices, brain area indices, and related variables
% Then calls criticality_sliding_window_ar.m to perform the analysis

% =============================    Configuration    =============================
% Data type selection
dataType = 'schall';  % 'reach' , 'naturalistic' , 'schall' , 'hong'

% Initialize paths
paths = get_paths;

% Initialize options structure
opts = neuro_behavior_options;
opts.frameSize = .001;
opts.firingRateCheckTime = 5 * 60;
opts.collectStart = 0;
opts.minFiringRate = .05;
opts.maxFiringRate = 200;





%% =============================    Data Loading    =============================
fprintf('\n=== Loading %s data ===\n', dataType);

% Create filename suffix based on PCA flag (will be set in analysis script)
% This is a placeholder - actual value will be set in criticality_sliding_window_ar.m
filenameSuffix = '';  % Will be updated based on pcaFlag in analysis script

    % =============================    Naturalistic Data Loading    =============================
if strcmp(dataType, 'naturalistic')
    % Load naturalistic data
    getDataType = 'spikes';
    opts.collectEnd = 45 * 60; % seconds
    get_standard_data

    areas = {'M23', 'M56', 'DS', 'VS'};
    idMatIdx = {idM23, idM56, idDS, idVS};
    idLabel = {idLabels(idM23), idLabels(idM56), idLabels(idDS), idLabels(idVS)};

    % Create save directory for naturalistic data
    saveDir = fullfile(paths.dropPath, 'criticality/results');
    if ~exist(saveDir, 'dir'); mkdir(saveDir); end

    % Initialize reach-specific variables as empty for naturalistic data
    dataR = [];
    startBlock2 = [];
    reachStart = [];
    reachClass = [];
    sessionName = '';  % Not used for naturalistic data

    % Initialize spikeData (will be loaded in analysis script if analyzeModulation is true)
    spikeData = [];

    % Print summary
    fprintf('%d M23\n%d M56\n%d DS\n%d VS\n', length(idM23), length(idM56), length(idDS), length(idVS));


    saveDir = fullfile(paths.dropPath, 'criticality/results');
    if ~exist(saveDir, 'dir'); mkdir(saveDir); end


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
   % Get reach onset times
 
    opts.collectEnd = round(min(dataR.R(end,1) + 5000, max(dataR.CSV(:,1)*1000)) / 1000);

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

    [~, dataBaseName, ~] = fileparts(reachDataFile);
    saveDir = fullfile(paths.dropPath, 'reach_task/results', dataBaseName);
    if ~exist(saveDir, 'dir'); mkdir(saveDir); end

    % Areas to analyze
    areasToTest = 1:4;




    % =============================    Schall Choice countermanding Data Loading    =============================
elseif strcmp(dataType, 'schall')

    % For reach data: specify session name (uncomment and set one)
    sessionName =  'broca/bp229n02-mm.mat';
    % sessionName =  'broca/bp240n02.mat';
    sessionName =  'joule/jp121n02.mat';
    sessionName =  'joule/jp125n04.mat';
    sessionName = fullfile('joule', goodSessionsCCM{end})

    %     opts.collectStart = 60*60;
    % opts.collectEnd = 105*60;
    % % opts.collectStart = 0*60;
    opts.collectEnd = 45*60;

    % Validate sessionName is provided
    if ~exist('sessionName', 'var') || isempty(sessionName)
        error('sessionName must be defined for reach data. Uncomment and set one of the sessionName lines above.');
    end

    % Load reach data
    schallDataFile = fullfile(paths.schallDataPath, sessionName);

    [~, dataBaseName, ~] = fileparts(schallDataFile);
    saveDir = fullfile(paths.dropPath, 'schall/results', dataBaseName);
    if ~exist(saveDir, 'dir'); mkdir(saveDir); end

    dataS = load(schallDataFile);
responseOnset = convert_to_session_time(dataS.responseOnset, dataS.trialOnset);
responseOnset = responseOnset / 1000;
responseOnset(responseOnset < opts.collectStart) = [];
responseOnset(responseOnset > opts.collectEnd) = [];



    [dataMat, idLabels, areaLabels] = neural_matrix_schall_fef(dataS, opts);
    areas = {'FEF'};
    idFEF = 1:length(idLabels);
    idMatIdx = {idFEF};
    idLabel = {idLabels(idFEF)};

    % Initialize spikeData (will be loaded in analysis script if analyzeModulation is true)
    spikeData = [];

    % Print summary
    fprintf('%d FEF\n', length(idFEF));


    [~, dataBaseName, ~] = fileparts(schallDataFile);
    saveDir = fullfile(paths.dropPath, 'schall/results', dataBaseName);
    if ~exist(saveDir, 'dir'); mkdir(saveDir); end

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
% All data is now loaded into the workspace. Call the analysis script.

% Sliding window size (seconds)
slidingWindowSize = 2;

criticality_sliding_window_ar
%%
% Sliding window and step size (seconds)
slidingWindowSize = 180;  % seconds - user specified
avStepSize = 20;          % seconds - user specified

criticality_sliding_window_av

