function [binStats] = spikes_per_bin(spikeData, binSize, varargin)
% SPIKES_PER_BIN - Analyzes spike distribution across bins for each neuron
%
% This function takes spike data and analyzes what proportion of bins contain
% zero, one, and more than one spike for each neuron. It also provides
% visualization of these distributions.
%
% INPUTS:
%   spikeData - Matrix with columns:
%       Column 1: Spike times (timestamps)
%       Column 2: Neuron IDs corresponding to each spike
%       Column 3: Brain area labels (numeric: 1=M23, 2=M56, 3=DS, 4=VS)
%   binSize   - Size of time bins in seconds
%
% OPTIONAL INPUTS (name-value pairs):
%   'plotResults'     - Logical flag to create plots (default: true)
%   'plotTheme'        - 'light' or 'dark' theme (default: 'light')
%   'savePlotFlag'     - Logical flag to save plots (default: false)
%   'savePath'         - Path to save plots (default: current directory)
%   'plotType'         - 'individual' or 'summary' plots (default: 'summary')
%   'maxNeurons'       - Maximum number of neurons to plot individually (default: 20)
%
% OUTPUTS:
%   binStats - Structure containing:
%       .neuronIds     - List of unique neuron IDs
%       .areaLabels    - Area labels for each neuron
%       .zeroSpikes    - Proportion of bins with 0 spikes for each neuron
%       .oneSpike      - Proportion of bins with 1 spike for each neuron
%       .multipleSpikes- Proportion of bins with >1 spike for each neuron
%       .meanSpikesPerBin - Mean spikes per bin for each neuron
%       .totalBins     - Total number of bins for each neuron
%       .binSize       - Bin size used
%       .areaNames     - Area name mapping

%% Parse inputs
p = inputParser;
addRequired(p, 'spikeData', @isnumeric);
addRequired(p, 'binSize', @isnumeric);
addParameter(p, 'plotResults', true, @islogical);
addParameter(p, 'plotTheme', 'light', @(x) ismember(x, {'light', 'dark'}));
addParameter(p, 'savePlotFlag', false, @islogical);
addParameter(p, 'savePath', pwd, @ischar);
addParameter(p, 'plotType', 'summary', @(x) ismember(x, {'individual', 'summary'}));
addParameter(p, 'maxNeurons', 20, @isnumeric);

parse(p, spikeData, binSize, varargin{:});

plotResults = p.Results.plotResults;
plotTheme = p.Results.plotTheme;
savePlotFlag = p.Results.savePlotFlag;
savePath = p.Results.savePath;
plotType = p.Results.plotType;
maxNeurons = p.Results.maxNeurons;

%% Extract data
spikeTimes = spikeData(:, 1);
neuronIds = spikeData(:, 2);
areaLabels = spikeData(:, 3);

% Get unique neurons and their areas
uniqueNeurons = unique(neuronIds);
nNeurons = length(uniqueNeurons);

% Area name mapping
areaNames = {'M23', 'M56', 'DS', 'VS'};

% Determine time range
timeStart = min(spikeTimes);
timeEnd = max(spikeTimes);
totalTime = timeEnd - timeStart;
nBins = ceil(totalTime / binSize);

fprintf('Analyzing %d neurons across %.2f seconds with %.2f second bins\n', ...
    nNeurons, totalTime, binSize);
fprintf('Total bins: %d\n', nBins);

%% Initialize storage
zeroSpikes = zeros(nNeurons, 1);
oneSpike = zeros(nNeurons, 1);
multipleSpikes = zeros(nNeurons, 1);
meanSpikesPerBin = zeros(nNeurons, 1);
totalBins = zeros(nNeurons, 1);
neuronAreaLabels = zeros(nNeurons, 1);

%% Analyze each neuron
for i = 1:nNeurons
    neuronId = uniqueNeurons(i);
    
    % Get spikes for this neuron
    neuronSpikes = spikeTimes(neuronIds == neuronId);
    
    % Get area label for this neuron
    neuronAreaLabels(i) = areaLabels(find(neuronIds == neuronId, 1));
    
    % Create bins for this neuron
    binEdges = timeStart:binSize:timeEnd;
    if binEdges(end) < timeEnd
        binEdges = [binEdges, timeEnd];
    end
    
    % Count spikes in each bin
    spikeCounts = histcounts(neuronSpikes, binEdges);
    actualBins = length(spikeCounts);
    totalBins(i) = actualBins;
    
    % Calculate proportions
    zeroCount = sum(spikeCounts == 0);
    oneCount = sum(spikeCounts == 1);
    multipleCount = sum(spikeCounts > 1);
    binsWithSpikes = sum(spikeCounts > 0);  % Total bins with at least one spike
    
    zeroSpikes(i) = zeroCount / actualBins;
    oneSpike(i) = oneCount / actualBins;
    
    % Multi-spike proportion: out of bins that have spikes, how many have multiple spikes
    if binsWithSpikes > 0
        multipleSpikes(i) = multipleCount / binsWithSpikes;
    else
        multipleSpikes(i) = 0;  % No spikes at all
    end
    
    % Calculate mean spikes per bin
    meanSpikesPerBin(i) = mean(spikeCounts);
    
    % Progress indicator
    if mod(i, 50) == 0 || i == nNeurons
        fprintf('Processed %d/%d neurons\n', i, nNeurons);
    end
end

%% Store results
binStats = struct();
binStats.neuronIds = uniqueNeurons;
binStats.areaLabels = neuronAreaLabels;
binStats.zeroSpikes = zeroSpikes;
binStats.oneSpike = oneSpike;
binStats.multipleSpikes = multipleSpikes;
binStats.meanSpikesPerBin = meanSpikesPerBin;
binStats.totalBins = totalBins;
binStats.binSize = binSize;
binStats.areaNames = areaNames;

%% Create plots if requested
if plotResults
    fprintf('\nCreating plots...\n');
    
    % Theme setup
    if strcmp(plotTheme, 'dark')
        backgroundColor = [0.1, 0.1, 0.1];
        textColor = [1.0, 1.0, 1.0];
        gridColor = [0.5, 0.5, 0.5];
        areaColors = [
            0.0, 0.5, 1.0;      % M23 - Blue
            1.0, 0.3, 0.0;       % M56 - Red
            1.0, 0.8, 0.0;       % DS - Yellow
            0.0, 1.0, 0.0;       % VS - Green
        ];
    else
        backgroundColor = [1.0, 1.0, 1.0];
        textColor = [0.0, 0.0, 0.0];
        gridColor = [0.8, 0.8, 0.8];
        areaColors = [
            0.0, 0.4470, 0.7410;  % M23 - Blue
            0.8500, 0.3250, 0.0980;  % M56 - Red
            0.9290, 0.6940, 0.1250;  % DS - Yellow
            0.4660, 0.6740, 0.1880;  % VS - Green
        ];
    end
    
    if strcmp(plotType, 'summary')
        % Create summary plots
        createSummaryPlots(binStats, plotTheme, backgroundColor, textColor, gridColor, areaColors);
        
        if savePlotFlag
            plotFilename = sprintf('spikes_per_bin_summary_bin%.2fs.png', binSize);
            plotPath = fullfile(savePath, plotFilename);
            exportgraphics(gcf, plotPath, 'Resolution', 300);
            fprintf('Saved summary plot: %s\n', plotFilename);
        end
        
    elseif strcmp(plotType, 'individual')
        % Create individual neuron plots
        nToPlot = min(maxNeurons, nNeurons);
        createIndividualPlots(binStats, nToPlot, plotTheme, backgroundColor, textColor, gridColor, areaColors);
        
        if savePlotFlag
            for i = 1:nToPlot
                plotFilename = sprintf('spikes_per_bin_neuron%d_bin%.2fs.png', ...
                    binStats.neuronIds(i), binSize);
                plotPath = fullfile(savePath, plotFilename);
                exportgraphics(gcf, plotPath, 'Resolution', 300);
            end
            fprintf('Saved %d individual plots\n', nToPlot);
        end
    end
end

%% Print summary statistics
fprintf('\n=== SUMMARY STATISTICS ===\n');
fprintf('Bin size: %.2f seconds\n', binSize);
fprintf('Total neurons: %d\n', nNeurons);
fprintf('Total time: %.2f seconds\n', totalTime);
fprintf('Total bins: %d\n', nBins);

for areaIdx = 1:4
    areaMask = neuronAreaLabels == areaIdx;
    if any(areaMask)
        areaName = areaNames{areaIdx};
        nAreaNeurons = sum(areaMask);
        
        fprintf('\n%s (%d neurons):\n', areaName, nAreaNeurons);
        fprintf('  Zero spikes: %.3f ± %.3f\n', mean(zeroSpikes(areaMask)), std(zeroSpikes(areaMask)));
        fprintf('  One spike: %.3f ± %.3f\n', mean(oneSpike(areaMask)), std(oneSpike(areaMask)));
        fprintf('  Multiple spikes (of bins with spikes): %.3f ± %.3f\n', mean(multipleSpikes(areaMask)), std(multipleSpikes(areaMask)));
        fprintf('  Mean spikes/bin: %.3f ± %.3f\n', mean(meanSpikesPerBin(areaMask)), std(meanSpikesPerBin(areaMask)));
    end
end

end

%% Helper function to create summary plots
function createSummaryPlots(binStats, plotTheme, backgroundColor, textColor, gridColor, areaColors)
    areaNames = binStats.areaNames;
    neuronAreaLabels = binStats.areaLabels;
    
    % Create figure
    fig = figure('Position', [100, 100, 1200, 800]);
    set(fig, 'Color', backgroundColor);
    
    % Subplot 1: Zero spikes proportion
    subplot(2, 2, 1);
    hold on;
    for areaIdx = 1:4
        areaMask = neuronAreaLabels == areaIdx;
        if any(areaMask)
            histogram(binStats.zeroSpikes(areaMask), 'FaceColor', areaColors(areaIdx, :), ...
                'FaceAlpha', 0.7, 'EdgeColor', 'none');
        end
    end
    xlabel('Proportion of bins with 0 spikes', 'Color', textColor);
    ylabel('Number of neurons', 'Color', textColor);
    title('Distribution of Zero-Spike Bins', 'Color', textColor);
    legend(areaNames, 'TextColor', textColor, 'Color', backgroundColor);
    set(gca, 'Color', backgroundColor, 'XColor', textColor, 'YColor', textColor);
    grid on;
    set(gca, 'GridColor', gridColor);
    
    % Subplot 2: One spike proportion
    subplot(2, 2, 2);
    hold on;
    for areaIdx = 1:4
        areaMask = neuronAreaLabels == areaIdx;
        if any(areaMask)
            histogram(binStats.oneSpike(areaMask), 'FaceColor', areaColors(areaIdx, :), ...
                'FaceAlpha', 0.7, 'EdgeColor', 'none');
        end
    end
    xlabel('Proportion of bins with 1 spike', 'Color', textColor);
    ylabel('Number of neurons', 'Color', textColor);
    title('Distribution of Single-Spike Bins', 'Color', textColor);
    legend(areaNames, 'TextColor', textColor, 'Color', backgroundColor);
    set(gca, 'Color', backgroundColor, 'XColor', textColor, 'YColor', textColor);
    grid on;
    set(gca, 'GridColor', gridColor);
    
    % Subplot 3: Multiple spikes proportion
    subplot(2, 2, 3);
    hold on;
    for areaIdx = 1:4
        areaMask = neuronAreaLabels == areaIdx;
        if any(areaMask)
            histogram(binStats.multipleSpikes(areaMask), 'FaceColor', areaColors(areaIdx, :), ...
                'FaceAlpha', 0.7, 'EdgeColor', 'none');
        end
    end
    xlabel('Proportion of multi-spike bins (out of bins with spikes)', 'Color', textColor);
    ylabel('Number of neurons', 'Color', textColor);
    title('Distribution of Multi-Spike Bins', 'Color', textColor);
    legend(areaNames, 'TextColor', textColor, 'Color', backgroundColor);
    set(gca, 'Color', backgroundColor, 'XColor', textColor, 'YColor', textColor);
    grid on;
    set(gca, 'GridColor', gridColor);
    
    % Subplot 4: Mean spikes per bin
    subplot(2, 2, 4);
    hold on;
    for areaIdx = 1:4
        areaMask = neuronAreaLabels == areaIdx;
        if any(areaMask)
            histogram(binStats.meanSpikesPerBin(areaMask), 'FaceColor', areaColors(areaIdx, :), ...
                'FaceAlpha', 0.7, 'EdgeColor', 'none');
        end
    end
    xlabel('Mean spikes per bin', 'Color', textColor);
    ylabel('Number of neurons', 'Color', textColor);
    title('Distribution of Mean Spikes per Bin', 'Color', textColor);
    legend(areaNames, 'TextColor', textColor, 'Color', backgroundColor);
    set(gca, 'Color', backgroundColor, 'XColor', textColor, 'YColor', textColor);
    grid on;
    set(gca, 'GridColor', gridColor);
    
    % Overall title
    sgtitle(sprintf('Spike Distribution Analysis (Bin Size: %.2f s)', binStats.binSize), ...
        'FontSize', 16, 'Color', textColor);
end

%% Helper function to create individual neuron plots
function createIndividualPlots(binStats, nToPlot, plotTheme, backgroundColor, textColor, gridColor, areaColors)
    areaNames = binStats.areaNames;
    
    for i = 1:nToPlot
        neuronId = binStats.neuronIds(i);
        areaIdx = binStats.areaLabels(i);
        areaName = areaNames{areaIdx};
        
        % Create figure for this neuron
        fig = figure('Position', [100 + i*50, 100 + i*50, 600, 400]);
        set(fig, 'Color', backgroundColor);
        
        % Create pie chart
        pieData = [binStats.zeroSpikes(i), binStats.oneSpike(i), binStats.multipleSpikes(i)];
        pieLabels = {'0 spikes', '1 spike', '>1 spike (of bins with spikes)'};
        
        pie(pieData, pieLabels);
        colormap([areaColors(areaIdx, :); areaColors(areaIdx, :)*0.7; areaColors(areaIdx, :)*0.4]);
        
        title(sprintf('Neuron %d (%s)\nBin Size: %.2f s', neuronId, areaName, binStats.binSize), ...
            'Color', textColor, 'FontSize', 14);
        
        % Add text with statistics
        statsText = sprintf('Mean spikes/bin: %.3f\nTotal bins: %d', ...
            binStats.meanSpikesPerBin(i), binStats.totalBins(i));
        text(1.3, 0, statsText, 'Color', textColor, 'FontSize', 10, ...
            'VerticalAlignment', 'middle');
    end
end
