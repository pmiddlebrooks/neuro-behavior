%%
% Criticality Sliding Window Data Preparation Script
% Extracts and organizes data loading and preparation code for different data types
% Prepares neural data matrices, brain area indices, and related variables
% Then calls criticality_sliding_window_ar.m to perform the analysis

% =============================    Configuration    =============================
% Data type selection
dataType = 'reach';  % 'reach' or 'naturalistic'

% Initialize paths
paths = get_paths;

% Initialize options structure
opts = neuro_behavior_options;
opts.frameSize = .001;
opts.firingRateCheckTime = 5 * 60;
opts.collectStart = 0;
opts.minFiringRate = .05;
opts.maxFiringRate = 100;





%% =============================    Data Loading    =============================
fprintf('\n=== Loading %s data ===\n', dataType);

% Create filename suffix based on PCA flag (will be set in analysis script)
% This is a placeholder - actual value will be set in criticality_sliding_window_ar.m
filenameSuffix = '';  % Will be updated based on pcaFlag in analysis script

% =============================    Reach Data Loading    =============================
if strcmp(dataType, 'reach')

    % For reach data: specify session name (uncomment and set one)
sessionName =  'AB2_28-Apr-2023 17_50_02_NeuroBeh.mat';
% sessionName =  'AB2_01-May-2023 15_34_59_NeuroBeh.mat';
% sessionName =  'AB2_11-May-2023 17_31_00_NeuroBeh.mat';
% sessionName =  'AB2_30-May-2023 12_49_52_NeuroBeh.mat';
% sessionName =  'AB6_27-Mar-2025 14_04_12_NeuroBeh.mat';
% sessionName =  'AB6_29-Mar-2025 15_21_05_NeuroBeh.mat';
% sessionName =  'AB6_02-Apr-2025 14_18_54_NeuroBeh.mat';
% sessionName =  'AB6_03-Apr-2025 13_34_09_NeuroBeh.mat';
% sessionName =  'Y4_06-Oct-2023 14_14_53_NeuroBeh.mat';
% sessionName =  'Y15_26-Aug-2025 12_24_22_NeuroBeh.mat';
% sessionName =  'Y15_27-Aug-2025 14_02_21_NeuroBeh.mat';
% sessionName =  'Y15_28-Aug-2025 19_47_07_NeuroBeh.mat';
% sessionName =  'Y17_20-Aug-2025 17_34_48_NeuroBeh.mat';

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
    
% =============================    Naturalistic Data Loading    =============================
elseif strcmp(dataType, 'naturalistic')
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
    
else
    error('Invalid dataType. Must be ''reach'' or ''naturalistic''');
end

% Print summary
fprintf('%d M23\n%d M56\n%d DS\n%d VS\n', length(idM23), length(idM56), length(idDS), length(idVS));

% =============================    Call Analysis Script    =============================
% All data is now loaded into the workspace. Call the analysis script.
criticality_sliding_window_ar

