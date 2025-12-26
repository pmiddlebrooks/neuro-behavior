function [proportions, spikeCounts] = neural_pct_spike_count(dataMat, binSizes, nSpikeMax, varargin)
% NEURAL_PCT_SPIKE_COUNT Analyze spike count distributions across different bin sizes
%
% Variables:
%   dataMat - Neural matrix [nTimePoints x nNeurons] sampled at 1000 Hz
%   binSizes - Array of bin sizes to test (in seconds)
%   nSpikeMax - Maximum number of spikes to count up to (0 to nSpikeMax)
%   varargin - Optional name-value pairs:
%       'plotFlag' - Whether to create plots (default: true)
%       'figureNum' - Figure number for plotting (default: 913)
%
% Goal:
%   For each bin size, resample the data using neural_matrix_ms_to_frames,
%   and calculate the proportion of all individual neuron-time bins
%   containing 0, 1, 2, ..., nSpikeMax spikes (without summing across
%   neurons). Plot the proportion distributions for comparison across bin sizes.
%
% Returns:
%   proportions - Cell array {binSizeIdx} = [1 x (nSpikeMax+1)] of proportions
%   spikeCounts - Cell array {binSizeIdx} = [nBins x 1] of spike counts per bin

    % Parse optional arguments
    p = inputParser;
    addParameter(p, 'plotFlag', true, @islogical);
    addParameter(p, 'figureNum', 913, @isnumeric);
    parse(p, varargin{:});
    
    plotFlag = p.Results.plotFlag;
    figureNum = p.Results.figureNum;
    
    % Validate inputs
    if nargin < 3
        error('At least 3 arguments required: dataMat, binSizes, nSpikeMax');
    end
    
    if isempty(binSizes)
        error('binSizes cannot be empty');
    end
    
    if nSpikeMax < 0
        error('nSpikeMax must be >= 0');
    end
    
    % Initialize output arrays
    numBinSizes = length(binSizes);
    proportions = cell(1, numBinSizes);
    spikeCounts = cell(1, numBinSizes);
    
    fprintf('\n=== Analyzing Spike Count Distributions ===\n');
    fprintf('Data matrix size: [%d x %d] (time x neurons)\n', size(dataMat, 1), size(dataMat, 2));
    fprintf('Number of bin sizes to test: %d\n', numBinSizes);
    fprintf('Spike count range: 0 to %d\n', nSpikeMax);
    
    % Process each bin size
    for b = 1:numBinSizes
        binSize = binSizes(b);
        fprintf('\nProcessing bin size %.4f s...\n', binSize);
        
        % Resample data using neural_matrix_ms_to_frames
        binnedData = neural_matrix_ms_to_frames(dataMat, binSize);
        % binnedData is [nBins x nNeurons]
        
        % Flatten all neuron-time bin combinations (don't sum across neurons)
        % This gives spike counts for each individual neuron in each time bin
        spikesPerBin = binnedData(:);  % [nBins * nNeurons x 1]
        spikeCounts{b} = spikesPerBin;
        
        numTotalBins = length(spikesPerBin);  % Total number of neuron-time bins
        fprintf('  Number of time bins: %d\n', size(binnedData, 1));
        fprintf('  Number of neurons: %d\n', size(binnedData, 2));
        fprintf('  Total neuron-time bins: %d\n', numTotalBins);
        
        % Calculate proportions for each spike count (0 to nSpikeMax)
        % Proportions are across all individual neuron-time bins
        propArray = zeros(1, nSpikeMax + 1);
        for spikeCount = 0:nSpikeMax
            propArray(spikeCount + 1) = sum(spikesPerBin == spikeCount) / numTotalBins;
        end
        
        proportions{b} = propArray;
        
        % Print summary statistics
        fprintf('  Proportion of neuron-time bins with 0 spikes: %.4f\n', propArray(1));
        fprintf('  Proportion of neuron-time bins with 1 spike: %.4f\n', propArray(2));
        if nSpikeMax >= 2
            fprintf('  Proportion of neuron-time bins with 2+ spikes: %.4f\n', sum(propArray(3:end)));
        end
        fprintf('  Mean spikes per neuron-time bin: %.4f\n', mean(spikesPerBin));
    end
    
    % Create plots
    if plotFlag
        fprintf('\nCreating plots...\n');
        
        % Detect monitors
        monitorPositions = get(0, 'MonitorPositions');
        if size(monitorPositions, 1) >= 2
            targetPos = monitorPositions(size(monitorPositions, 1), :);
        else
            targetPos = monitorPositions(1, :);
        end
        
        figure(figureNum); clf;
        set(gcf, 'Position', targetPos);
        
        % Create subplot layout: one row, multiple columns (one per bin size)
        numCols = min(numBinSizes, 4);  % Max 4 columns
        numRows = ceil(numBinSizes / numCols);
        
        % Use tight_subplot if available, otherwise use subplot
        if exist('tight_subplot', 'file')
            ha = tight_subplot(numRows, numCols, [0.08 0.05], [0.1 0.08], [0.08 0.05]);
        else
            ha = zeros(numBinSizes, 1);
        end
        
        % Define colors for different bin sizes
        colors = lines(numBinSizes);
        
        % Plot each bin size
        for b = 1:numBinSizes
            if exist('tight_subplot', 'file')
                axes(ha(b));
            else
                subplot(numRows, numCols, b);
            end
            hold on;
            
            propArray = proportions{b};
            spikeCountRange = 0:nSpikeMax;
            
            % Create bar plot
            bar(spikeCountRange, propArray, 'FaceColor', colors(b, :), 'EdgeColor', 'none', 'FaceAlpha', 0.7);
            
            % Formatting
            xlabel('Number of Spikes per Neuron-Time Bin');
            ylabel('Proportion of Neuron-Time Bins');
            title(sprintf('Bin Size: %.4f s', binSizes(b)));
            grid on;
            xlim([-0.5, nSpikeMax + 0.5]);
            ylim([0, max(propArray) * 1.1]);
            set(gca, 'XTick', 0:max(1, floor(nSpikeMax/10)):nSpikeMax);
        end
        
        % Overall title
        if exist('tight_subplot', 'file')
            sgtitle(sprintf('Spike Count Distributions (0 to %d spikes)', nSpikeMax), 'FontSize', 14);
        else
            suptitle(sprintf('Spike Count Distributions (0 to %d spikes)', nSpikeMax));
        end
        
        % Also create a comparison plot showing all bin sizes together
        figure(figureNum + 1); clf;
        set(gcf, 'Position', targetPos);
        hold on;
        
        spikeCountRange = 0:nSpikeMax;
        for b = 1:numBinSizes
            propArray = proportions{b};
            plot(spikeCountRange, propArray, '-o', 'Color', colors(b, :), 'LineWidth', 2, ...
                'MarkerSize', 6, 'MarkerFaceColor', colors(b, :), ...
                'DisplayName', sprintf('%.4f s', binSizes(b)));
        end
        
        xlabel('Number of Spikes per Neuron-Time Bin');
        ylabel('Proportion of Neuron-Time Bins');
        title('Spike Count Distributions: Comparison Across Bin Sizes');
        legend('Location', 'best');
        grid on;
        xlim([-0.5, nSpikeMax + 0.5]);
        
        fprintf('Plots created in figures %d and %d\n', figureNum, figureNum + 1);
    end
    
    fprintf('\n=== Analysis Complete ===\n');
end

