function dataStruct = load_sliding_window_data(sessionType, dataSource, varargin)
% LOAD_SLIDING_WINDOW_DATA Load and prepare data for sliding window analyses
%
% Variables:
%   sessionType - Type of data: 'reach', 'naturalistic', 'schall', 'hong'
%   dataSource - Source of data: 'spikes' or 'lfp'
%   varargin - Optional name-value pairs:
%       'sessionName' - Session name (required for reach/schall data)
%       'opts' - Options structure (if not provided, creates default)
%       'lfpCleanParams' - LFP cleaning parameters structure
%       'bands' - Frequency bands for LFP (default: alpha, beta, lowGamma, highGamma)
%       'minBinSize' - Minimum bin size for LFP bands (default: 0.005)
%
% Goal:
%   Load and prepare data for sliding window analyses. Returns a structure
%   containing all necessary data instead of setting workspace variables.
%   This makes the code more reproducible and testable.
%
% Returns:
%   dataStruct - Structure with fields:
%       .sessionType, .dataSource, .areas, .saveDir, .opts
%       For spikes: .dataMat, .idMatIdx, .idLabel, .spikeData
%       For LFP: .lfpPerArea, .binnedEnvelopes, .bands, .bandBinSizes, etc.
%       Data-type specific: .dataR, .reachStart, .sessionName, etc.

    % Parse optional arguments
    p = inputParser;
    addParameter(p, 'sessionName', '', @(x) ischar(x) || isempty(x));
    addParameter(p, 'opts', [], @(x) isstruct(x) || isempty(x));
    addParameter(p, 'lfpCleanParams', [], @(x) isstruct(x) || isempty(x));
    addParameter(p, 'bands', [], @(x) iscell(x) || isempty(x));
    addParameter(p, 'minBinSize', 0.005, @isnumeric);
    parse(p, varargin{:});
    
    sessionName = p.Results.sessionName;
    opts = p.Results.opts;
    lfpCleanParams = p.Results.lfpCleanParams;
    bands = p.Results.bands;
    minBinSize = p.Results.minBinSize;
    
    % Initialize paths
    paths = get_paths;
    
    % Initialize options structure if not provided
    if isempty(opts)
        opts = neuro_behavior_options;
        opts.frameSize = .001;
        opts.firingRateCheckTime = 5 * 60;
        opts.collectStart = 0;
        opts.minFiringRate = .05;
        opts.maxFiringRate = 200;
    end
    
    % Initialize LFP parameters if needed
    if strcmp(dataSource, 'lfp')
        if isempty(bands)
            bands = {'alpha', [8 13]; ...
                'beta', [13 30]; ...
                'lowGamma', [30 50]; ...
                'highGamma', [50 80]};
        end
        
        if isempty(lfpCleanParams)
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
    end
    
    % Initialize output structure
    dataStruct = struct();
    dataStruct.sessionType = sessionType;
    dataStruct.dataSource = dataSource;
    dataStruct.opts = opts;
    
    fprintf('\n=== Loading %s %s data ===\n', sessionType, dataSource);
    
    % Load data based on data type
    switch sessionType
        case 'naturalistic'
            if isempty(sessionName)
                error('sessionName must be provided for naturalistic data');
            end
            dataStruct.sessionName = sessionName;
            dataStruct = load_naturalistic_data(dataStruct, dataSource, paths, opts, sessionName, lfpCleanParams, bands);
            
        case 'reach'
            if isempty(sessionName)
                error('sessionName must be provided for reach data');
            end
            dataStruct.sessionName = sessionName;
            dataStruct = load_reach_data(dataStruct, dataSource, paths, sessionName, opts, lfpCleanParams, bands);
            
        case 'schall'
            if isempty(sessionName)
                error('sessionName must be provided for schall data');
            end
            dataStruct.sessionName = sessionName;
            dataStruct = load_schall_data(dataStruct, dataSource, paths, sessionName, opts, lfpCleanParams, bands);
            
        case 'hong'
            dataStruct = load_hong_data(dataStruct, dataSource, paths, opts, lfpCleanParams, bands);
            dataStruct.sessionName = sessionName;            
        otherwise
            error('Invalid sessionType: %s. Must be ''reach'', ''naturalistic'', ''schall'', or ''hong''', sessionType);
    end
    
    % Set default areasToTest if not set
    if ~isfield(dataStruct, 'areasToTest') || isempty(dataStruct.areasToTest)
        dataStruct.areasToTest = 1:length(dataStruct.areas);
    end
    
    fprintf('Data loading complete. %d areas loaded.\n', length(dataStruct.areas));
end
