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
%       .baseWindow   - Baseline time range [min max] in seconds relative to align time
%                       Example: [-3, -2] for 1 second window from -3 to -2 sec
%       .eventWindow  - Event time range [min max] in seconds relative to align time
%                       Example: [-0.2, 0.6] for 0.8 second window from -0.2 to 0.6 sec
%       .alignTimes   - Vector of alignment times around which to place windows
%       .threshold    - Number of standard deviations above/below mean for modulation (legacy)
%       .plotFlag     - Optional: Set to true to generate modulation analysis plots (default: true)
%
% OUTPUTS:
%   modulationResults - Structure containing:
%       .neuronIds        - List of unique neuron IDs
%       .baseMean         - Mean spikes per second in baseline window
%       .eventMean        - Mean spikes per second in event window
%       .modulationIndex  - (eventMean - baseMean) / baseMean
%       .zScore          - Z-score of modulation (standardized difference)
%       .pValue          - p-value from paired t-test
%       .isModulated     - Logical array indicating modulated neurons
%       .modulationType   - 'positive', 'negative', or 'none' for each neuron
%       .baseStd         - Standard deviation of baseline spikes per second
%       .eventStd        - Standard deviation of event spikes per second
%       .baseBins        - Number of bins in baseline window
%       .eventBins       - Number of bins in event window
%       .opts            - Copy of input options
%
% EXAMPLE:
%   opts.baseWindow = [-3, -2];    % Baseline: -3 to -2 seconds (relative to reach)
%   opts.eventWindow = [-0.2, 0.6]; % Event: -0.2 to 0.6 seconds (relative to reach)
%   opts.alignTimes = [10, 20, 30]; % Align windows around these times
%   opts.threshold = 2;            % 2 standard deviations (legacy fallback)
%   opts.plotFlag = true;          % Generate plots (optional, default: true)
%   results = spike_modulation(spikeData, opts);

%% Extract data
spikeTimes = spikeData(:, 1);
neuronIds = spikeData(:, 2);

% Get unique neurons
uniqueNeurons = unique(neuronIds);
nNeurons = length(uniqueNeurons);

% Extract options
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

% Validate inputs - baseWindow and eventWindow should be [min max] ranges
if ~isnumeric(baseWindow) || ~isnumeric(eventWindow)
    error('baseWindow and eventWindow must be numeric arrays');
end

if length(baseWindow) ~= 2 || length(eventWindow) ~= 2
    error('baseWindow and eventWindow must be 2-element arrays [min max]');
end

if baseWindow(1) >= baseWindow(2) || eventWindow(1) >= eventWindow(2)
    error('Window ranges must have min < max (e.g., [-3, -2])');
end

% Extract window boundaries
baseStartOffset = baseWindow(1);   % Relative to align time
baseEndOffset = baseWindow(2);     % Relative to align time
eventStartOffset = eventWindow(1); % Relative to align time
eventEndOffset = eventWindow(2);   % Relative to align time

% Calculate window durations for reporting
baseDuration = baseEndOffset - baseStartOffset;
eventDuration = eventEndOffset - eventStartOffset;

if threshold <= 0
    error('Threshold must be positive');
end

if isempty(alignTimes)
    error('alignTimes cannot be empty');
end

nAlignTimes = length(alignTimes);

fprintf('Analyzing %d neurons for spike modulation\n', nNeurons);
fprintf('Baseline window: [%.2f, %.2f] seconds (%.2f sec duration)\n', baseStartOffset, baseEndOffset, baseDuration);
fprintf('Event window: [%.2f, %.2f] seconds (%.2f sec duration)\n', eventStartOffset, eventEndOffset, eventDuration);
fprintf('Number of alignment times: %d\n', nAlignTimes);
fprintf('Modulation test: Paired t-test (p < 0.05 for significance)\n');

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
pValue = zeros(nNeurons, 1);  % p-value from paired t-test

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
        
        % Calculate absolute window times using input ranges
        % Baseline window: relative to align time
        absBaseStart = alignTime + baseStartOffset;
        absBaseEnd = alignTime + baseEndOffset;
        
        % Event window: relative to align time
        absEventStart = alignTime + eventStartOffset;
        absEventEnd = alignTime + eventEndOffset;
        
        
        % Count spikes in baseline window for this alignment
        baseSpikes = sum(neuronSpikes >= absBaseStart & neuronSpikes <= absBaseEnd);
        baseSpikesPerSecond = baseSpikes / baseDuration;
        
        % Count spikes in event window for this alignment
        eventSpikes = sum(neuronSpikes >= absEventStart & neuronSpikes <= absEventEnd);
        eventSpikesPerSecond = eventSpikes / eventDuration;
        
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
    
    % Calculate z-score (standardized difference) for backward compatibility
    if baseStd(i) > 0
        zScore(i) = (eventMean(i) - baseMean(i)) / baseStd(i);
    else
        % If no variability in baseline, use raw difference
        zScore(i) = eventMean(i) - baseMean(i);
    end
    
    % Determine modulation using paired t-test instead of z-score threshold
    % Perform paired t-test between baseline and event windows
    if length(allBaseSpikesPerSecond) >= 3 && length(allEventSpikesPerSecond) >= 3
        % Perform paired t-test (event - baseline)
        try
            [h, pVal] = ttest(allEventSpikesPerSecond - allBaseSpikesPerSecond);
            pValue(i) = pVal;
            
            % Determine if significant modulation exists
            if h == 1 && pVal < 0.05  % p < 0.05 for significance
                isModulated(i) = true;
                % Determine positive or negative modulation
                if mean(allEventSpikesPerSecond) > mean(allBaseSpikesPerSecond)
                    modulationType{i} = 'positive';
                else
                    modulationType{i} = 'negative';
                end
            else
                isModulated(i) = false;
                modulationType{i} = 'none';
            end
        catch
            % If t-test fails, use z-score method as fallback
            pValue(i) = 1.0;
            isModulated(i) = false;
            modulationType{i} = 'none';
        end
    else
        % Not enough samples for t-test, use z-score method
        pValue(i) = 1.0;
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
modulationResults.pValue = pValue;
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
