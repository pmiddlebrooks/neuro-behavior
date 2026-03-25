function hmm_mazz_debug_model_selection_plot(results, config)
% HMM_MAZZ_DEBUG_MODEL_SELECTION_PLOT Debug model-selection diagnostics.
%
% Variables:
%   results - Top-level results struct from hmm_mazz_analysis with fields:
%       .areas, .hmm_results
%
%   config - Optional struct:
%       .areasToPlot - Area indices or area labels to include (default: all valid)
%       .showDiffElbow - True to compute elbow on diff(LL) for comparison (default: true)
%
% Goal:
%   Plot diagnostics to verify whether selected number of states matches:
%   (1) raw model-selection curve in LLtot.m2LL and
%   (2) elbow computed on diff(LL), which mirrors toolbox logic.

if nargin < 2 || isempty(config)
    config = struct();
end

if ~isfield(config, 'showDiffElbow') || isempty(config.showDiffElbow)
    config.showDiffElbow = true;
end

if ~isfield(results, 'hmm_results') || ~iscell(results.hmm_results)
    error('Expected top-level results struct with cell field results.hmm_results.');
end

if ~isfield(results, 'areas') || isempty(results.areas)
    error('Expected results.areas to be present.');
end

numAreas = numel(results.areas);
areasToPlot = 1:numAreas;

if isfield(config, 'areasToPlot') && ~isempty(config.areasToPlot)
    if isnumeric(config.areasToPlot)
        areasToPlot = config.areasToPlot(:)';
    elseif iscell(config.areasToPlot)
        areaMap = containers.Map(results.areas, 1:numAreas);
        resolvedAreas = [];
        for idx = 1:numel(config.areasToPlot)
            areaLabel = config.areasToPlot{idx};
            if isKey(areaMap, areaLabel)
                resolvedAreas(end + 1) = areaMap(areaLabel); %#ok<AGROW>
            else
                warning('Skipping unknown area label: %s', areaLabel);
            end
        end
        areasToPlot = resolvedAreas;
    else
        error('config.areasToPlot must be numeric indices or cell array of labels.');
    end
end

for areaIdx = areasToPlot
    if areaIdx < 1 || areaIdx > numAreas
        warning('Skipping invalid area index: %d', areaIdx);
        continue;
    end

    areaRes = results.hmm_results{areaIdx};
    if isempty(areaRes)
        warning('Skipping area %s: empty result.', results.areas{areaIdx});
        continue;
    end

    if ~isfield(areaRes, 'LLtot') || ~isfield(areaRes.LLtot, 'm2LL')
        warning('Skipping area %s: missing LLtot.m2LL.', results.areas{areaIdx});
        continue;
    end
    if ~isfield(areaRes, 'HmmParam') || ~isfield(areaRes.HmmParam, 'VarStates')
        warning('Skipping area %s: missing HmmParam.VarStates.', results.areas{areaIdx});
        continue;
    end
    if ~isfield(areaRes, 'best_model') || ~isfield(areaRes.best_model, 'best_state_index')
        warning('Skipping area %s: missing best_model.best_state_index.', results.areas{areaIdx});
        continue;
    end

    hiddenTotal = areaRes.HmmParam.VarStates(:)';
    llCurve = areaRes.LLtot.m2LL(:)';
    bestStateIdx = areaRes.best_model.best_state_index;

    if numel(hiddenTotal) ~= numel(llCurve)
        minLen = min(numel(hiddenTotal), numel(llCurve));
        warning(['Area %s: Hidden/state axis and LL curve lengths differ (%d vs %d). ' ...
            'Truncating to %d.'], results.areas{areaIdx}, numel(hiddenTotal), numel(llCurve), minLen);
        hiddenTotal = hiddenTotal(1:minLen);
        llCurve = llCurve(1:minLen);
    end

    if isempty(hiddenTotal)
        warning('Skipping area %s: empty hiddenTotal.', results.areas{areaIdx});
        continue;
    end

    if bestStateIdx < 1 || bestStateIdx > numel(hiddenTotal)
        warning('Area %s: best_state_index=%d out of bounds. Clamping.', ...
            results.areas{areaIdx}, bestStateIdx);
        bestStateIdx = max(1, min(bestStateIdx, numel(hiddenTotal)));
    end

    selectedNumStates = hiddenTotal(bestStateIdx);
    llDiff = diff(llCurve);

    diffElbowIdx = NaN;
    diffElbowState = NaN;
    if config.showDiffElbow && numel(llCurve) >= 3
        [~, diffElbowIdx] = hmm.funCriterion(llCurve, 'elbow');
        diffElbowIdx = max(1, min(diffElbowIdx, numel(hiddenTotal)));
        diffElbowState = hiddenTotal(diffElbowIdx);
    end

    figure('Name', sprintf('HMM model-selection debug: %s', results.areas{areaIdx}), ...
        'Color', 'w');

    subplot(2, 1, 1);
    plot(hiddenTotal, llCurve, '-ok', 'LineWidth', 1.5, 'MarkerSize', 5); hold on;
    plot(hiddenTotal(bestStateIdx), llCurve(bestStateIdx), 'or', ...
        'MarkerFaceColor', 'r', 'MarkerSize', 8);
    if ~isnan(diffElbowIdx)
        plot(hiddenTotal(diffElbowIdx), llCurve(diffElbowIdx), 'sb', ...
            'MarkerFaceColor', 'b', 'MarkerSize', 7);
        legend({'LL curve', 'Selected by pipeline', 'Elbow on diff(LL)'}, ...
            'Location', 'best');
    else
        legend({'LL curve', 'Selected by pipeline'}, 'Location', 'best');
    end
    xlabel('Number of states');
    ylabel('-2 log-likelihood / criterion');
    title(sprintf('%s: raw LL curve', results.areas{areaIdx}), 'Interpreter', 'none');
    grid on;
    hold off;

    subplot(2, 1, 2);
    if isempty(llDiff)
        text(0.1, 0.5, 'Not enough points for diff(LL)', 'FontSize', 11);
        axis off;
    else
        plot(hiddenTotal(2:end), llDiff, '-^', 'LineWidth', 1.5, 'MarkerSize', 5);
        xlabel('Number of states (aligned to LL index 2:end)');
        ylabel('diff(LL)');
        title('First difference used by elbow criterion');
        grid on;
    end

    sgtitle(sprintf(['Area %s | selectedIdx=%d selectedK=%d' ...
        ' | diffElbowIdx=%s diffElbowK=%s'], ...
        results.areas{areaIdx}, bestStateIdx, selectedNumStates, ...
        local_num_to_str(diffElbowIdx), local_num_to_str(diffElbowState)), ...
        'Interpreter', 'none');

    fprintf(['[HMM DEBUG] Area=%s | selectedIdx=%d selectedK=%d | ' ...
        'diffElbowIdx=%s diffElbowK=%s\n'], ...
        results.areas{areaIdx}, bestStateIdx, selectedNumStates, ...
        local_num_to_str(diffElbowIdx), local_num_to_str(diffElbowState));
end

end

function numStr = local_num_to_str(numValue)
% LOCAL_NUM_TO_STR Convert scalar numeric to printable string.
if isnan(numValue)
    numStr = 'NaN';
elseif isinf(numValue)
    numStr = 'Inf';
else
    numStr = sprintf('%g', numValue);
end
end
