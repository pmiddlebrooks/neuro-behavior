function results = criticality_lfp_analysis(dataStruct, config)
% CRITICALITY_LFP_ANALYSIS Perform d2 and DFA criticality analysis on LFP data
%
% Variables:
%   dataStruct - Data structure from load_sliding_window_data() with LFP fields:
%     .binnedEnvelopes, .bands, .bandBinSizes, .lfpPerArea, .lfpBinSize, .opts
%   config - Configuration structure with fields:
%     .slidingWindowSize - Window size for d2 analysis (seconds)
%     .analyzeD2 - Compute d2 (default: true)
%     .analyzeDFA - Compute DFA alpha (default: true)
%     .enablePermutations - Perform circular permutations (default: false)
%     .nShuffles - Number of permutations (default: 3)
%     .makePlots - Create plots (default: true)
%     .plotBinnedEnvelopes - Plot binned envelope results (default: true)
%     .plotRawLfp - Plot raw LFP results (default: true)
%     .saveDir - Save directory (optional, uses dataStruct.saveDir)
%
% Goal:
%   Compute d2 and/or DFA criticality measures on LFP binned envelopes and raw LFP.
%   Uses common time points across all analyses with different window sizes.
%
% Returns:
%   results - Structure with d2, dfa, startS, and params

    % Add paths
    addpath(fullfile(fileparts(mfilename('fullpath')), '..', '..', 'sliding_window_prep', 'utils'));
    
    % Validate inputs
    validate_workspace_vars({'binnedEnvelopes', 'bands', 'bandBinSizes'}, dataStruct, ...
        'errorMsg', 'Required field', 'source', 'load_sliding_window_data');
    
    % Set defaults
    config = set_config_defaults_lfp(config);
    
    areas = dataStruct.areas;
    numBands = size(dataStruct.bands, 1);
    numAreas = length(dataStruct.binnedEnvelopes);
    
    if isfield(dataStruct, 'areasToTest')
        areasToTest = dataStruct.areasToTest;
    else
        areasToTest = 1:numAreas;
    end
    
    % Check for raw LFP data
    hasRawLfp = isfield(dataStruct, 'lfpPerArea') && ~isempty(dataStruct.lfpPerArea) && ...
        isfield(dataStruct, 'lfpBinSize') && ~isempty(dataStruct.lfpBinSize);
    
    if hasRawLfp
        if ~isfield(dataStruct.opts, 'fsLfp')
            error('opts.fsLfp must be defined for raw LFP analysis.');
        end
        numLfpBins = length(dataStruct.lfpBinSize);
    else
        numLfpBins = 0;
    end
    
    fprintf('\n=== LFP Criticality Analysis Setup ===\n');
    fprintf('Number of bands: %d\n', numBands);
    fprintf('Number of areas: %d\n', numAreas);
    fprintf('Has raw LFP: %d\n', hasRawLfp);
    
    % Calculate common step size
    maxBandBinSize = max(dataStruct.bandBinSizes);
    stepSize = 20 * maxBandBinSize;
    fprintf('Step size: %.3f s\n', stepSize);
    
    % Determine total session duration
    durations = [];
    if hasRawLfp
        durations(end+1) = size(dataStruct.lfpPerArea, 1) / dataStruct.opts.fsLfp;
    end
    for a = 1:numAreas
        if length(dataStruct.binnedEnvelopes) >= a && ~isempty(dataStruct.binnedEnvelopes{a})
            for b = 1:numBands
                if length(dataStruct.binnedEnvelopes{a}) >= b && ~isempty(dataStruct.binnedEnvelopes{a}{b})
                    durations(end+1) = length(dataStruct.binnedEnvelopes{a}{b}) * dataStruct.bandBinSizes(b);
                end
            end
        end
    end
    
    if isempty(durations)
        error('Cannot determine session duration from available data.');
    end
    
    totalSessionDuration = min(durations);
    fprintf('Total session duration: %.1f s\n', totalSessionDuration);
    
    % Calculate window sizes
    d2WindowSize = config.slidingWindowSize;
    dfaEnvWinSamples_min = 1000;
    dfaEnvWinSize = max(dfaEnvWinSamples_min * max(dataStruct.bandBinSizes), 30);
    dfaLfpWinSamples_min = 2000;
    
    % Calculate maximum window size
    maxWindowSize = max(d2WindowSize, dfaEnvWinSize);
    if hasRawLfp
        for lb = 1:numLfpBins
            dfaLfpWinSize = max(dfaLfpWinSamples_min * dataStruct.lfpBinSize(lb), 30);
            maxWindowSize = max(maxWindowSize, dfaLfpWinSize);
        end
    end
    
    % Calculate common time points (startS)
    lastCenterTime = totalSessionDuration - maxWindowSize/2;
    startS = (stepSize/2) : stepSize : lastCenterTime;
    numWindows = length(startS);
    fprintf('Common time points: %d windows\n', numWindows);
    
    % Initialize results
    d2 = cell(1, numAreas);
    dfa = cell(1, numAreas);
    
    for a = 1:numAreas
        d2{a} = cell(1, numBands);
        dfa{a} = cell(1, numBands);
    end
    
    if hasRawLfp
        d2Lfp = cell(1, numAreas);
        dfaLfp = cell(1, numAreas);
        for a = 1:numAreas
            d2Lfp{a} = cell(1, numLfpBins);
            dfaLfp{a} = cell(1, numLfpBins);
        end
    end
    
    % Main analysis loop
    fprintf('\n=== Processing Areas ===\n');
    
    for a = areasToTest
        fprintf('\nProcessing area %s...\n', areas{a});
        
        % Process binned envelopes
        if length(dataStruct.binnedEnvelopes) >= a && ~isempty(dataStruct.binnedEnvelopes{a})
            areaEnvelopes = dataStruct.binnedEnvelopes{a};
            
            for b = 1:numBands
                fprintf('  Processing band %d (%s)...\n', b, dataStruct.bands{b, 1});
                tic;
                
                bandSignal = areaEnvelopes{b};
                bandBinSize = dataStruct.bandBinSizes(b);
                numFrames_b = length(bandSignal);
                
                % Calculate window sizes in samples
                d2WinSamples = round(d2WindowSize / bandBinSize);
                if config.analyzeDFA
                    dfaWinSamples = round(dfaEnvWinSize / bandBinSize);
                end
                
                if d2WinSamples < config.minSegmentLength
                    fprintf('    Skipping: Not enough samples for d2\n');
                    continue;
                end
                
                % Initialize arrays
                d2{a}{b} = nan(1, numWindows);
                if config.analyzeDFA
                    dfa{a}{b} = nan(1, numWindows);
                end
                
                % Process each time point
                for w = 1:numWindows
                    centerTime = startS(w);
                    
                    % d2 analysis
                    if config.analyzeD2
                        d2StartTime = centerTime - d2WindowSize/2;
                        d2EndTime = centerTime + d2WindowSize/2;
                        d2StartBin = max(1, round(d2StartTime / bandBinSize) + 1);
                        d2EndBin = min(numFrames_b, round(d2EndTime / bandBinSize));
                        
                        if d2EndBin > d2StartBin
                            d2Segment = bandSignal(d2StartBin:d2EndBin);
                            [varphi, ~] = myYuleWalker3(d2Segment, config.pOrder);
                            d2{a}{b}(w) = getFixedPointDistance2(config.pOrder, config.critType, varphi);
                        end
                    end
                    
                    % DFA analysis
                    if config.analyzeDFA
                        dfaStartTime = centerTime - dfaEnvWinSize/2;
                        dfaEndTime = centerTime + dfaEnvWinSize/2;
                        dfaStartBin = max(1, round(dfaStartTime / bandBinSize) + 1);
                        dfaEndBin = min(numFrames_b, round(dfaEndTime / bandBinSize));
                        
                        if dfaEndBin > dfaStartBin
                            dfaSegment = bandSignal(dfaStartBin:dfaEndBin);
                            dfa{a}{b}(w) = dfa_alpha(dfaSegment);
                        end
                    end
                end
                
                fprintf('    Band %d completed in %.1f minutes\n', b, toc/60);
            end
        end
        
        % Process raw LFP if available
        if hasRawLfp
            rawSignal = dataStruct.lfpPerArea(:, a);
            fsRaw = dataStruct.opts.fsLfp;
            
            for lb = 1:numLfpBins
                fprintf('  Processing raw LFP bin size %.3f s...\n', dataStruct.lfpBinSize(lb));
                tic;
                
                % Bin raw LFP
                binnedLfp = neural_matrix_ms_to_frames(rawSignal, dataStruct.lfpBinSize(lb));
                numFrames_lfp = size(binnedLfp, 1);
                
                % Calculate window sizes
                d2WinSamples = round(d2WindowSize / dataStruct.lfpBinSize(lb));
                dfaLfpWinSize = max(dfaLfpWinSamples_min * dataStruct.lfpBinSize(lb), 30);
                dfaWinSamples = round(dfaLfpWinSize / dataStruct.lfpBinSize(lb));
                
                if d2WinSamples < config.minSegmentLength
                    fprintf('    Skipping: Not enough samples for d2\n');
                    continue;
                end
                
                % Initialize arrays
                d2Lfp{a}{lb} = nan(1, numWindows);
                if config.analyzeDFA
                    dfaLfp{a}{lb} = nan(1, numWindows);
                end
                
                % Process each time point
                for w = 1:numWindows
                    centerTime = startS(w);
                    
                    % d2 analysis
                    if config.analyzeD2
                        d2StartTime = centerTime - d2WindowSize/2;
                        d2EndTime = centerTime + d2WindowSize/2;
                        d2StartBin = max(1, round(d2StartTime / dataStruct.lfpBinSize(lb)) + 1);
                        d2EndBin = min(numFrames_lfp, round(d2EndTime / dataStruct.lfpBinSize(lb)));
                        
                        if d2EndBin > d2StartBin
                            d2Segment = binnedLfp(d2StartBin:d2EndBin);
                            [varphi, ~] = myYuleWalker3(d2Segment, config.pOrder);
                            d2Lfp{a}{lb}(w) = getFixedPointDistance2(config.pOrder, config.critType, varphi);
                        end
                    end
                    
                    % DFA analysis
                    if config.analyzeDFA
                        dfaStartTime = centerTime - dfaLfpWinSize/2;
                        dfaEndTime = centerTime + dfaLfpWinSize/2;
                        dfaStartBin = max(1, round(dfaStartTime / dataStruct.lfpBinSize(lb)) + 1);
                        dfaEndBin = min(numFrames_lfp, round(dfaEndTime / dataStruct.lfpBinSize(lb)));
                        
                        if dfaEndBin > dfaStartBin
                            dfaSegment = binnedLfp(dfaStartBin:dfaEndBin);
                            dfaLfp{a}{lb}(w) = dfa_alpha(dfaSegment);
                        end
                    end
                end
                
                fprintf('    Raw LFP bin size %.3f s completed in %.1f minutes\n', dataStruct.lfpBinSize(lb), toc/60);
            end
        end
    end
    
    % Build results structure
    results = build_results_structure_lfp(dataStruct, config, areas, areasToTest, ...
        d2, dfa, startS, stepSize, d2WindowSize, dfaEnvWinSize, dfaLfpWinSamples_min, ...
        hasRawLfp, d2Lfp, dfaLfp, dataStruct.lfpBinSize);
    
    % Setup results path
    if ~isfield(config, 'saveDir') || isempty(config.saveDir)
        config.saveDir = dataStruct.saveDir;
    end
    
    resultsPath = create_results_path('criticality_lfp', dataStruct.dataType, config.slidingWindowSize, ...
        dataStruct.sessionName, config.saveDir);
    
    % Save results
    save(resultsPath, 'results');
    fprintf('\nSaved results to: %s\n', resultsPath);
    
    % Plotting
    if config.makePlots
        plotConfig = setup_plotting(config.saveDir, 'sessionName', dataStruct.sessionName, ...
            'dataBaseName', dataStruct.dataBaseName);
        plot_criticality_lfp_results(results, plotConfig, config, dataStruct);
    end
end

function config = set_config_defaults_lfp(config)
% SET_CONFIG_DEFAULTS_LFP Set default values for LFP configuration structure
    
    defaults = struct();
    defaults.analyzeD2 = true;
    defaults.analyzeDFA = true;
    defaults.enablePermutations = false;
    defaults.nShuffles = 3;
    defaults.makePlots = true;
    defaults.plotBinnedEnvelopes = true;
    defaults.plotRawLfp = true;
    defaults.plotD2 = true;
    defaults.plotDFA = true;
    defaults.minSegmentLength = 50;
    defaults.pOrder = 10;
    defaults.critType = 2;
    
    % Apply defaults
    fields = fieldnames(defaults);
    for i = 1:length(fields)
        if ~isfield(config, fields{i})
            config.(fields{i}) = defaults.(fields{i});
        end
    end
end

function results = build_results_structure_lfp(dataStruct, config, areas, areasToTest, ...
    d2, dfa, startS, stepSize, d2WindowSize, dfaEnvWinSize, dfaLfpWinSamples_min, ...
    hasRawLfp, d2Lfp, dfaLfp, lfpBinSize)
% BUILD_RESULTS_STRUCTURE_LFP Build results structure for LFP analysis
    
    results = struct();
    results.areas = areas;
    results.bands = dataStruct.bands;
    results.startS = startS;
    results.stepSize = stepSize;
    results.d2WindowSize = d2WindowSize;
    results.dfaEnvWinSize = dfaEnvWinSize;
    results.dfaEnvWinSamples_min = 1000;
    results.dfaLfpWinSamples_min = dfaLfpWinSamples_min;
    results.bandBinSizes = dataStruct.bandBinSizes;
    results.d2 = d2;
    results.dfa = dfa;
    
    if hasRawLfp
        results.d2Lfp = d2Lfp;
        results.dfaLfp = dfaLfp;
        results.lfpBinSize = lfpBinSize;
    end
    
    results.params.slidingWindowSize = config.slidingWindowSize;
    results.params.analyzeD2 = config.analyzeD2;
    results.params.analyzeDFA = config.analyzeDFA;
    results.params.pOrder = config.pOrder;
    results.params.critType = config.critType;
end

function plot_criticality_lfp_results(results, plotConfig, config, dataStruct)
% PLOT_CRITICALITY_LFP_RESULTS Create plots for criticality LFP analysis
    
    % This is a placeholder - the full plotting code from the original script
    % would need to be extracted and adapted here
    fprintf('Plotting functionality to be implemented...\n');
    % TODO: Extract plotting code from criticality_sliding_lfp.m
end

