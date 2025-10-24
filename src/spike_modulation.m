function [modulationResults] = spike_modulation(spikeData, opts)
% SPIKE_MODULATION - Analyzes spike modulation between baseline and event windows
%
% This function compares spike activity between a baseline window and an event window
% for each neuron, identifying neurons that show significant modulation above or below
% a specified threshold (in standard deviations).
%
% INPUTS:
%   spikeData - Matrix with columns:
%       Column 1: Spike times (timestamps in seconds)
%       Column 2: Neuron IDs corresponding to each spike
%   opts - Structure containing the following options:
%       .binSize      - Time bin size in seconds for spike counting
%       .baseWindow   - Duration of baseline window in seconds (before event window)
%       .eventWindow  - Duration of event window in seconds (centered on align time)
%       .alignTimes   - Vector of alignment times around which to place windows
%       .threshold    - Number of standard deviations above/below mean for modulation
%       .plotFlag     - Optional: Set to true to generate modulation analysis plots (default: true)
%
% OUTPUTS:
%   modulationResults - Structure containing:
%       .neuronIds        - List of unique neuron IDs
%       .baseMean         - Mean spikes per second in baseline window
%       .eventMean        - Mean spikes per second in event window
%       .modulationIndex  - (eventMean - baseMean) / baseMean
%       .zScore          - Z-score of modulation (standardized difference)
%       .isModulated     - Logical array indicating modulated neurons
%       .modulationType   - 'positive', 'negative', or 'none' for each neuron
%       .baseStd         - Standard deviation of baseline spikes per second
%       .eventStd        - Standard deviation of event spikes per second
%       .baseBins        - Number of bins in baseline window
%       .eventBins       - Number of bins in event window
%       .opts            - Copy of input options
%
% EXAMPLE:
%   opts.binSize = 0.1;           % 100ms bins
%   opts.baseWindow = 2.0;         % Baseline: 2 seconds before event window
%   opts.eventWindow = 1.0;        % Event: 1 second centered on align time
%   opts.alignTimes = [10, 20, 30]; % Align windows around these times
%   opts.threshold = 2;            % 2 standard deviations
%   opts.plotFlag = true;          % Generate plots (optional, default: true)
%   results = spike_modulation(spikeData, opts);

%% Extract data
spikeTimes = spikeData(:, 1);
neuronIds = spikeData(:, 2);

% Get unique neurons
uniqueNeurons = unique(neuronIds);
nNeurons = length(uniqueNeurons);

% Extract options
binSize = opts.binSize;
baseWindow = opts.baseWindow;
eventWindow = opts.eventWindow;
alignTimes = opts.alignTimes;
threshold = opts.threshold;

% Extract plotFlag with default value
if isfield(opts, 'plotFlag')
    plotFlag = opts.plotFlag;
else
    plotFlag = true; % Default to true for backward compatibility
end

% Validate inputs
if ~isscalar(baseWindow) || ~isscalar(eventWindow)
    error('baseWindow and eventWindow must be scalar values (window durations in seconds)');
end

if baseWindow <= 0 || eventWindow <= 0
    error('Window durations must be positive');
end

if threshold <= 0
    error('Threshold must be positive');
end

if isempty(alignTimes)
    error('alignTimes cannot be empty');
end

nAlignTimes = length(alignTimes);

fprintf('Analyzing %d neurons for spike modulation\n', nNeurons);
fprintf('Bin size: %.3f seconds\n', binSize);
fprintf('Baseline window: %.1f seconds before event window\n', baseWindow);
fprintf('Event window: %.1f seconds centered on align time\n', eventWindow);
fprintf('Number of alignment times: %d\n', nAlignTimes);
fprintf('Modulation threshold: %.1f standard deviations\n', threshold);

%% Initialize storage
baseMean = zeros(nNeurons, 1);
eventMean = zeros(nNeurons, 1);
modulationIndex = zeros(nNeurons, 1);
zScore = zeros(nNeurons, 1);
isModulated = false(nNeurons, 1);
modulationType = cell(nNeurons, 1);
baseStd = zeros(nNeurons, 1);
eventStd = zeros(nNeurons, 1);
baseBins = zeros(nNeurons, 1);
eventBins = zeros(nNeurons, 1);

%% Analyze each neuron
for i = 1:nNeurons
    neuronId = uniqueNeurons(i);
    
    % Get spikes for this neuron
    neuronSpikes = spikeTimes(neuronIds == neuronId);
    
    % Initialize arrays to collect spikes per second across all alignment times
    allBaseSpikesPerSecond = [];
    allEventSpikesPerSecond = [];
    
    % Process each alignment time
    for alignIdx = 1:nAlignTimes
        alignTime = alignTimes(alignIdx);
        
        % Calculate absolute window times
        % Event window is centered on align time
        eventHalfWidth = eventWindow / 2;
        absEventStart = alignTime - eventHalfWidth;
        absEventEnd = alignTime + eventHalfWidth;
        
        % Baseline window ends 0.001s before event window starts
        absBaseEnd = absEventStart - 0.001;
        absBaseStart = absBaseEnd - baseWindow;
        
        
        % Count spikes in baseline window for this alignment
        baseSpikes = sum(neuronSpikes >= absBaseStart & neuronSpikes <= absBaseEnd);
        baseSpikesPerSecond = baseSpikes / baseWindow;
        
        % Count spikes in event window for this alignment
        eventSpikes = sum(neuronSpikes >= absEventStart & neuronSpikes <= absEventEnd);
        eventSpikesPerSecond = eventSpikes / eventWindow;
        
        % Collect spikes per second across all alignments
        allBaseSpikesPerSecond = [allBaseSpikesPerSecond, baseSpikesPerSecond];
        allEventSpikesPerSecond = [allEventSpikesPerSecond, eventSpikesPerSecond];
    end
    
    % Store total alignment counts
    baseBins(i) = length(allBaseSpikesPerSecond);
    eventBins(i) = length(allEventSpikesPerSecond);
    
    % Calculate means and standard deviations across all alignment times
    if baseBins(i) > 0
        baseMean(i) = mean(allBaseSpikesPerSecond);
        baseStd(i) = std(allBaseSpikesPerSecond);
    else
        baseMean(i) = 0;
        baseStd(i) = 0;
    end
    
    if eventBins(i) > 0
        eventMean(i) = mean(allEventSpikesPerSecond);
        eventStd(i) = std(allEventSpikesPerSecond);
    else
        eventMean(i) = 0;
        eventStd(i) = 0;
    end
    
    % Calculate modulation index (relative change)
    if baseMean(i) > 0
        modulationIndex(i) = (eventMean(i) - baseMean(i)) / baseMean(i);
    else
        % If baseline is zero, use absolute difference
        modulationIndex(i) = eventMean(i);
    end
    
    % Calculate z-score (standardized difference)
    if baseStd(i) > 0
        zScore(i) = (eventMean(i) - baseMean(i)) / baseStd(i);
    else
        % If no variability in baseline, use raw difference
        zScore(i) = eventMean(i) - baseMean(i);
    end
    
    % Determine modulation type
    if abs(zScore(i)) >= threshold
        isModulated(i) = true;
        if zScore(i) > 0
            modulationType{i} = 'positive';
        else
            modulationType{i} = 'negative';
        end
    else
        modulationType{i} = 'none';
    end
    
    % Progress indicator
    if mod(i, 100) == 0 || i == nNeurons
        fprintf('Processed %d/%d neurons\n', i, nNeurons);
    end
end

%% Store results
modulationResults = struct();
modulationResults.neuronIds = uniqueNeurons;
modulationResults.baseMean = baseMean;
modulationResults.eventMean = eventMean;
modulationResults.modulationIndex = modulationIndex;
modulationResults.zScore = zScore;
modulationResults.isModulated = isModulated;
modulationResults.modulationType = modulationType;
modulationResults.baseStd = baseStd;
modulationResults.eventStd = eventStd;
modulationResults.baseBins = baseBins;
modulationResults.eventBins = eventBins;
modulationResults.opts = opts;

%% Print summary statistics
fprintf('\n=== MODULATION ANALYSIS RESULTS ===\n');
fprintf('Total neurons analyzed: %d\n', nNeurons);
fprintf('Modulated neurons: %d (%.1f%%)\n', sum(isModulated), 100*sum(isModulated)/nNeurons);

positiveMod = strcmp(modulationType, 'positive');
negativeMod = strcmp(modulationType, 'negative');

fprintf('Positive modulation: %d (%.1f%%)\n', sum(positiveMod), 100*sum(positiveMod)/nNeurons);
fprintf('Negative modulation: %d (%.1f%%)\n', sum(negativeMod), 100*sum(negativeMod)/nNeurons);

fprintf('\nModulation statistics:\n');
fprintf('Mean modulation index: %.3f ± %.3f\n', mean(modulationIndex), std(modulationIndex));
fprintf('Mean z-score: %.3f ± %.3f\n', mean(zScore), std(zScore));

fprintf('\nBaseline vs Event activity:\n');
fprintf('Baseline mean spikes/second: %.3f ± %.3f\n', mean(baseMean), std(baseMean));
fprintf('Event mean spikes/second: %.3f ± %.3f\n', mean(eventMean), std(eventMean));
fprintf('Total alignments analyzed per neuron: %d baseline + %d event\n', round(mean(baseBins)), round(mean(eventBins)));

%% Create visualization
if plotFlag
    createModulationPlot(modulationResults);
end

end

%% Helper function to create modulation visualization
function createModulationPlot(results)
    % Create figure
    fig = figure('Position', [100, 100, 1200, 800]);
    
    % Subplot 1: Modulation index distribution
    subplot(2, 2, 1);
    histogram(results.modulationIndex, 50, 'FaceColor', [0.3, 0.3, 0.3], 'EdgeColor', 'none');
    xlabel('Modulation Index');
    ylabel('Number of neurons');
    title('Distribution of Modulation Indices');
    grid on;
    
    % Add threshold lines
    hold on;
    ylim_vals = ylim;
    plot([results.opts.threshold, results.opts.threshold], ylim_vals, 'r--', 'LineWidth', 2);
    plot([-results.opts.threshold, -results.opts.threshold], ylim_vals, 'r--', 'LineWidth', 2);
    legend('Modulation Index', 'Threshold', 'Location', 'best');
    
    % Subplot 2: Z-score distribution
    subplot(2, 2, 2);
    histogram(results.zScore, 50, 'FaceColor', [0.3, 0.3, 0.3], 'EdgeColor', 'none');
    xlabel('Z-Score');
    ylabel('Number of neurons');
    title('Distribution of Z-Scores');
    grid on;
    
    % Add threshold lines
    hold on;
    ylim_vals = ylim;
    plot([results.opts.threshold, results.opts.threshold], ylim_vals, 'r--', 'LineWidth', 2);
    plot([-results.opts.threshold, -results.opts.threshold], ylim_vals, 'r--', 'LineWidth', 2);
    legend('Z-Score', 'Threshold', 'Location', 'best');
    
    % Subplot 3: Baseline vs Event scatter plot
    subplot(2, 2, 3);
    scatter(results.baseMean, results.eventMean, 30, 'filled', 'MarkerFaceAlpha', 0.6);
    xlabel('Baseline Mean Spikes/Second');
    ylabel('Event Mean Spikes/Second');
    title('Baseline vs Event Activity');
    grid on;
    
    % Add diagonal line (no change)
    hold on;
    max_val = max(max(results.baseMean), max(results.eventMean));
    plot([0, max_val], [0, max_val], 'k--', 'LineWidth', 1);
    legend('Neurons', 'No Change', 'Location', 'best');
    
    % Subplot 4: Modulation type pie chart
    subplot(2, 2, 4);
    positiveCount = sum(strcmp(results.modulationType, 'positive'));
    negativeCount = sum(strcmp(results.modulationType, 'negative'));
    noneCount = sum(strcmp(results.modulationType, 'none'));
    
    pieData = [positiveCount, negativeCount, noneCount];
    pieLabels = {'Positive', 'Negative', 'None'};
    pie(pieData, pieLabels);
    title('Modulation Types');
    
    % Overall title
    sgtitle(sprintf('Spike Modulation Analysis (Threshold: %.1f SD)', results.opts.threshold), ...
        'FontSize', 16);
    
    % Save plot
    plotFilename = sprintf('spike_modulation_analysis_thresh%.1f.png', results.opts.threshold);
    savePath = pwd;
    plotPath = fullfile(savePath, plotFilename);
    exportgraphics(fig, plotPath, 'Resolution', 300);
    fprintf('Saved modulation plot: %s\n', plotFilename);
end
