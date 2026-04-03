function hmm_mazz_plot(hmmRes, config)
% HMM_MAZZ_PLOT Create basic plots for Mazzucato-style HMM results.
%
% Variables:
%   hmmRes - Either:
%       (1) Top-level results struct with fields:
%             .areas       - Cell array of area names
%             .hmm_results - Cell array of per-area HMM result structs
%         as saved by hmm_mazz_analysis / hmm_load_saved_model when all
%         areas are loaded.
%       (2) Single-area HMM result struct for one brain area, with fields:
%             .HmmParam, .best_model, .hmm_results, .continuous_results
%
%   config - Optional struct for future plot options (currently unused).
%
% Goal:
%   Plot transition and emission matrices, and a continuous state sequence
%   with posterior probabilities for a single representative area.

if nargin < 2
    config = struct(); %#ok<NASGU>
end
if ~isfield(config, 'plotStateDominance') || isempty(config.plotStateDominance)
    config.plotStateDominance = true;
end

% Handle multi-area vs single-area input
if isfield(hmmRes, 'hmm_results') && iscell(hmmRes.hmm_results)
    % All areas loaded - use first available area or config.brainArea
    if isfield(config, 'brainArea') && ~isempty(config.brainArea)
        areaMap = containers.Map({'M23', 'M56', 'DS', 'VS'}, {1, 2, 3, 4});
        if ~isKey(areaMap, config.brainArea)
            error('Unknown brain area "%s" in config.brainArea.', config.brainArea);
        end
        areaIdx = areaMap(config.brainArea);
        hmmResSingle = hmmRes.hmm_results{areaIdx};
        areaLabel = hmmRes.areas{areaIdx};
        fprintf('Using area %s for plots\n', areaLabel);
    else
        areaIdx = 1;
        while areaIdx <= numel(hmmRes.hmm_results) && isempty(hmmRes.hmm_results{areaIdx})
            areaIdx = areaIdx + 1;
        end
        if areaIdx > numel(hmmRes.hmm_results)
            error('No HMM results found in provided structure.');
        end
        hmmResSingle = hmmRes.hmm_results{areaIdx};
        areaLabel = hmmRes.areas{areaIdx};
        fprintf('Using area %s for plots\n', areaLabel);
    end
else
    % Single-area struct
    hmmResSingle = hmmRes;
    if isfield(hmmResSingle, 'metadata') && isfield(hmmResSingle.metadata, 'brain_area')
        areaLabel = hmmResSingle.metadata.brain_area;
    else
        areaLabel = 'UnknownArea';
    end
    fprintf('Using single-area HMM results for plots (%s)\n', areaLabel);
end

% Extract HMM parameters
HmmParam = hmmResSingle.HmmParam;
if isfield(HmmParam, 'VarStates')
    if isfield(hmmResSingle, 'best_model') && ...
            isfield(hmmResSingle.best_model, 'best_state_index')
        bestStateIdx = hmmResSingle.best_model.best_state_index;
        if isscalar(HmmParam.VarStates)
            HmmParam.VarStates = HmmParam.VarStates;
        else
            HmmParam.VarStates = HmmParam.VarStates(bestStateIdx);
        end
    elseif isfield(hmmResSingle, 'best_model') && ...
            isfield(hmmResSingle.best_model, 'num_states')
        HmmParam.VarStates = hmmResSingle.best_model.num_states;
    elseif isscalar(HmmParam.VarStates)
        HmmParam.VarStates = HmmParam.VarStates;
    else
        HmmParam.VarStates = HmmParam.VarStates(1);
    end
elseif isfield(hmmResSingle, 'best_model') && ...
        isfield(hmmResSingle.best_model, 'num_states')
    HmmParam.VarStates = hmmResSingle.best_model.num_states;
end

% Extract HMM results
hmmResults = hmmResSingle.hmm_results;
hmmPostfit = hmmResSingle.hmm_postfit; %#ok<NASGU>

% Colors for states
numStates = HmmParam.VarStates;
colors = distinguishable_colors(max(numStates, 4));

% Plot transition and emission matrices
figure;
if exist('tight_subplot', 'file')
    [axesArray, ~] = tight_subplot(1, 2, 0.05, [0.08 0.06], [0.06 0.03]);
    ax1 = axesArray(1);
    ax2 = axesArray(2);
else
    ax1 = subplot(1, 2, 1);
    ax2 = subplot(1, 2, 2);
end

axes(ax1);
imagesc(hmmResSingle.best_model.transition_matrix);
colorbar;
title(sprintf('Transition Matrix (%s)', areaLabel), 'Interpreter', 'none');
xlabel('State');
ylabel('State');

axes(ax2);
imagesc(hmmResSingle.best_model.emission_matrix(:, 1:end-1));
colorbar;
title(sprintf('Emission Matrix (%s)', areaLabel), 'Interpreter', 'none');
xlabel('Neuron');
ylabel('State');

% Plot full sequence with probabilities
figure;
clf;
hold on;

sequence = hmmResSingle.continuous_results.sequence;
probabilities = hmmResSingle.continuous_results.pStates;

timeAxis = (1:numel(sequence)) * HmmParam.BinSize;
numStatesProb = size(probabilities, 2);

colors = distinguishable_colors(numStatesProb);

% Draw translucent rectangles for state segments
for stateIdx = 1:numStatesProb
    stateMask = sequence == stateIdx;
    if any(stateMask)
        diffMask = diff([0; stateMask; 0]);
        startIdx = find(diffMask == 1);
        endIdx = find(diffMask == -1) - 1;
        for segmentIdx = 1:numel(startIdx)
            xStart = timeAxis(startIdx(segmentIdx));
            xEnd = timeAxis(endIdx(segmentIdx));
            patch([xStart xEnd xEnd xStart], [0 0 1 1], colors(stateIdx, :), ...
                'FaceAlpha', 0.2, 'EdgeColor', 'none');
        end
    end
end

% Plot probability traces
for stateIdx = 1:numStatesProb
    plot(timeAxis, probabilities(:, stateIdx), ...
        'Color', colors(stateIdx, :), 'LineWidth', 2);
end

xlabel('Time (s)');
ylabel('State probability');
ylim([0 1]);
title(sprintf('HMM State Probabilities (%s)', areaLabel), 'Interpreter', 'none');
hold off;

fprintf('Basic HMM plots completed for %s.\n', areaLabel);

if config.plotStateDominance
    plot_state_dominance_diagnostics(hmmResSingle, areaLabel);
end

end

function plot_state_dominance_diagnostics(hmmResSingle, areaLabel)
% PLOT_STATE_DOMINANCE_DIAGNOSTICS Plot state dominance diagnostics.
%
% Variables:
%   hmmResSingle - Single-area HMM result struct with:
%       .continuous_results.pStates  - [timeBins x numStates] posteriors
%       .continuous_results.sequence - [timeBins x 1] thresholded states
%       .HmmParam.MinP               - Threshold for state assignment
%       .HmmParam.BinSize            - Bin size in seconds
%   areaLabel   - Area name for titles.
%
% Goal:
%   Compare soft state occupancy (posteriors) with hard occupancy
%   (thresholded sequence), and visualize confidence/dwell statistics.

if ~isfield(hmmResSingle, 'continuous_results') || ...
        ~isfield(hmmResSingle.continuous_results, 'pStates') || ...
        ~isfield(hmmResSingle.continuous_results, 'sequence')
    warning('Skipping state dominance diagnostics: missing continuous results.');
    return;
end

pStates = hmmResSingle.continuous_results.pStates;
sequence = hmmResSingle.continuous_results.sequence(:);

if isempty(pStates) || isempty(sequence)
    warning('Skipping state dominance diagnostics: empty pStates or sequence.');
    return;
end

[numTimeBins, numStates] = size(pStates);
if numel(sequence) ~= numTimeBins
    minLen = min(numel(sequence), numTimeBins);
    warning(['Length mismatch between sequence (%d) and pStates (%d). ' ...
        'Truncating to %d bins.'], numel(sequence), numTimeBins, minLen);
    sequence = sequence(1:minLen);
    pStates = pStates(1:minLen, :);
    numTimeBins = minLen;
end

minP = 0.8;
if isfield(hmmResSingle, 'HmmParam') && isfield(hmmResSingle.HmmParam, 'MinP')
    minP = hmmResSingle.HmmParam.MinP;
end

binSize = 1;
if isfield(hmmResSingle, 'HmmParam') && isfield(hmmResSingle.HmmParam, 'BinSize')
    binSize = hmmResSingle.HmmParam.BinSize;
end

[maxProb, maxState] = max(pStates, [], 2);
assignedMask = sequence > 0;

% Soft occupancy: expected fraction of time spent in each state.
occSoft = mean(pStates, 1);
occMap = arrayfun(@(stateIdx) mean(maxState == stateIdx), 1:numStates);
occHard = arrayfun(@(stateIdx) mean(sequence == stateIdx), 1:numStates);
occHardCond = zeros(1, numStates);
if any(assignedMask)
    occHardCond = arrayfun(@(stateIdx) mean(sequence(assignedMask) == stateIdx), 1:numStates);
end
fracUnassigned = mean(~assignedMask);

% Entropy diagnostic for posterior concentration.
epsVal = 1e-12;
stateEntropy = -sum(pStates .* log(pStates + epsVal), 2);
if numStates > 1
    entropyNorm = stateEntropy / log(numStates);
else
    entropyNorm = zeros(size(stateEntropy));
end

figure('Name', sprintf('HMM state dominance: %s', areaLabel), 'Color', 'w');
tiledlayout(2, 3, 'Padding', 'compact', 'TileSpacing', 'compact');

nexttile;
bar((1:numStates)', [occSoft(:), occMap(:), occHard(:)], 'grouped');
xlabel('State');
ylabel('Occupancy fraction');
title('Occupancy: soft vs MAP vs hard');
legend({'Soft mean(P)', 'MAP winner', 'Thresholded sequence'}, ...
    'Location', 'best');
ylim([0, 1]);
grid on;

nexttile;
histogram(maxProb, 30, 'FaceColor', [0.2, 0.4, 0.8], 'EdgeColor', 'none');
hold on;
xline(minP, '--r', sprintf('MinP=%.2f', minP), 'LineWidth', 1.5);
hold off;
xlabel('max_k P(state=k | t)');
ylabel('Count');
title(sprintf('Confidence (unassigned=%.2f)', fracUnassigned));
grid on;

nexttile;
timeAxis = (1:numTimeBins) * binSize;
plot(timeAxis, entropyNorm, 'k-', 'LineWidth', 1);
xlabel('Time (s)');
ylabel('Normalized entropy');
title('Posterior entropy over time');
ylim([0, 1]);
grid on;

nexttile;
maxPlotBins = min(numTimeBins, 3000);
plotTimeAxis = (1:maxPlotBins) * binSize;
area(plotTimeAxis, pStates(1:maxPlotBins, :), 'LineStyle', 'none');
xlabel('Time (s)');
ylabel('Probability mass');
title(sprintf('Stacked pStates (first %d bins)', maxPlotBins));
ylim([0, 1]);
grid on;

nexttile;
[runState, runDurSec] = local_compute_run_durations(sequence, binSize);
if isempty(runState)
    text(0.1, 0.5, 'No assigned state runs (all bins unassigned).', 'FontSize', 10);
    axis off;
else
    boxplot(runDurSec, runState, 'Symbol', '.');
    xlabel('State');
    ylabel('Run duration (s)');
    title('Dwell-time distribution (hard sequence)');
    grid on;
end

nexttile;
edges = linspace(0, 1, 21);
probHist = zeros(numStates, numel(edges) - 1);
for stateIdx = 1:numStates
    probHist(stateIdx, :) = histcounts(pStates(:, stateIdx), edges, ...
        'Normalization', 'probability');
end
imagesc(edges(1:end-1), 1:numStates, probHist);
axis xy;
colorbar;
xlabel('Posterior probability');
ylabel('State');
title('Per-state posterior distribution');

sgtitle(sprintf(['%s | softOcc sum=%.2f | assigned=%.2f | ' ...
    'hardCond max=%.2f'], ...
    areaLabel, sum(occSoft), mean(assignedMask), max(occHardCond)), ...
    'Interpreter', 'none');

fprintf('State dominance diagnostics completed for %s.\n', areaLabel);

end

function [runState, runDurSec] = local_compute_run_durations(sequence, binSize)
% LOCAL_COMPUTE_RUN_DURATIONS Compute run durations by state from sequence.

if isempty(sequence)
    runState = [];
    runDurSec = [];
    return;
end

changeMask = [true; diff(sequence) ~= 0];
runStart = find(changeMask);
runEnd = [runStart(2:end) - 1; numel(sequence)];
runState = sequence(runStart);
runDurBins = runEnd - runStart + 1;

validMask = runState > 0;
runState = runState(validMask);
runDurSec = runDurBins(validMask) * binSize;

end

