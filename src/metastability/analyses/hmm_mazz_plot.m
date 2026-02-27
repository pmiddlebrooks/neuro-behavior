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

end

