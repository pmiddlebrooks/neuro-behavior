function hmm_mazz_plot_penalized_ll_curve(results, config)
% HMM_MAZZ_PLOT_PENALIZED_LL_CURVE Penalized -2LL / criterion curve vs scanned K states.
%
% Variables:
%   results - Struct from hmm_mazz_analysis containing .areas, .hmm_results,
%             and optionally .llCriterionAcrossStates for prepacked curves.
%
%   config - Optional struct:
%       .areasToPlot - Cell array of brain area strings (empty = all nonempty results)
%
% Goal:
%   Match hmm.funHMM figure style: shaded SEM from hmm_data(:, K).LLtrain when
%   available, black criterion trace, red circle on the pipeline-selected column.

if nargin < 2 || isempty(config)
    config = struct();
end
if ~isfield(config, 'areasToPlot')
    config.areasToPlot = [];
end

if ~isfield(results, 'hmm_results') || ~iscell(results.hmm_results)
    warning('Expected results struct with hmm_results cell; skipping penalized LL plot.');
    return;
end

numAreasTotal = numel(results.areas);
plotAreaIndices = resolve_plot_area_indices(results.areas, numAreasTotal, config.areasToPlot);
if isempty(plotAreaIndices)
    warning('No areas to plot for penalized LL curve.');
    return;
end

figure('Name', 'HMM penalized LL / criterion', 'Color', 'w');
tileRows = max(1, ceil(sqrt(numel(plotAreaIndices))));
tileCols = max(1, ceil(numel(plotAreaIndices) / tileRows));
tileIdxLocal = 0;

for listIdx = 1:numel(plotAreaIndices)
    areaIdx = plotAreaIndices(listIdx);
    if areaIdx < 1 || areaIdx > numel(results.hmm_results)
        continue;
    end

    hmmR = results.hmm_results{areaIdx};
    if isempty(hmmR)
        continue;
    end

    [hiddenTotalSeq, penalizedLlRow] = local_curve_from_results(results, areaIdx, hmmR);
    if isempty(hiddenTotalSeq) || isempty(penalizedLlRow)
        fprintf('Skipping %s: missing LL criterion data.\n', results.areas{areaIdx});
        continue;
    end

    curveLenLocal = numel(hiddenTotalSeq);
    if curveLenLocal ~= numel(penalizedLlRow)
        shrinkLenLocal = min(curveLenLocal, numel(penalizedLlRow));
        hiddenTotalSeq = hiddenTotalSeq(1:shrinkLenLocal);
        penalizedLlRow = penalizedLlRow(1:shrinkLenLocal);
        curveLenLocal = shrinkLenLocal;
    end

    if isempty(hiddenTotalSeq)
        continue;
    end

    bestColIdxLocal = hmmR.best_model.best_state_index;
    if bestColIdxLocal < 1 || bestColIdxLocal > numel(hiddenTotalSeq)
        bestColIdxLocal = max(1, min(bestColIdxLocal, numel(hiddenTotalSeq)));
    end

    methodStrLocal = '';
    if isfield(hmmR, 'metadata') && isfield(hmmR.metadata, 'model_selection_method')
        methodStrLocal = hmmR.metadata.model_selection_method;
    end

    llSemAcrossStates = compute_ll_sem_across_training_runs(hmmR.hmm_data, numel(hiddenTotalSeq));

    tileIdxLocal = tileIdxLocal + 1;
    subplot(tileRows, tileCols, tileIdxLocal);
    hold on;

    semMaskLocal = ~isempty(llSemAcrossStates) && numel(llSemAcrossStates) == numel(hiddenTotalSeq) ...
        && any(isfinite(llSemAcrossStates)) && any(llSemAcrossStates(:) > 0);
    if semMaskLocal
        upperLlLocal = penalizedLlRow(:)' + reshape(llSemAcrossStates, 1, []);
        lowerLlLocal = penalizedLlRow(:)' - reshape(llSemAcrossStates, 1, []);
        if exist('aux.jbfill', 'file') == 2
            aux.jbfill(hiddenTotalSeq, upperLlLocal, lowerLlLocal, [0.8, 0.8, 0.8], 0, 0.1);
        else
            xRibbonLocal = [hiddenTotalSeq, fliplr(hiddenTotalSeq)];
            yRibbonLocal = [upperLlLocal, fliplr(lowerLlLocal)];
            fill(xRibbonLocal, yRibbonLocal, [0.8, 0.8, 0.8], 'EdgeColor', 'none', ...
                'FaceAlpha', 0.15); %#ok<NASGU>
        end
    end

    hLineLocal = plot(hiddenTotalSeq, penalizedLlRow(:)', 'k-', 'LineWidth', 2);
    hMarkLocal = plot(hiddenTotalSeq(bestColIdxLocal), penalizedLlRow(bestColIdxLocal), 'ro', ...
        'MarkerFaceColor', 'r', 'MarkerSize', 10);

    xlabel('# hidden states');
    ylblLocal = sprintf('Criterion (LLtot.m2LL; %s)', methodStrLocal);
    ylabel(ylblLocal);
    title(sprintf('%s (selected K=%d)', results.areas{areaIdx}, ...
        hmmR.best_model.num_states), 'Interpreter', 'none');
    legend([hLineLocal, hMarkLocal], {'Criterion (penalized / -2LL)', ...
        sprintf('Chosen (K=%d, column %d)', hmmR.best_model.num_states, bestColIdxLocal)}, ...
        'Location', 'best');
    grid on;
    hold off;
end

if tileIdxLocal == 0
    close(gcf);
    warning('No valid criterion curves plotted.');
end

if exist('aux.figset', 'file') == 2 && tileIdxLocal > 0 && numel(plotAreaIndices) == 1
    methodStrFoot = '';
    hmmRef = results.hmm_results{plotAreaIndices(1)};
    if ~isempty(hmmRef) && isfield(hmmRef, 'metadata')
        methodStrFoot = hmmRef.metadata.model_selection_method;
    end
    aux.figset(gca, 'hidden states', sprintf('%s', methodStrFoot), '', 15);
end

end

function plotAreaIndices = resolve_plot_area_indices(areas, numAreasTotal, labelsOrEmpty)
% RESOLVE_PLOT_AREA_INDICES Map optional area-name list to canonical indices.

if isempty(labelsOrEmpty)
    plotAreaIndices = [];
    for scanIdxLocal = 1:numAreasTotal
        plotAreaIndices(end + 1) = scanIdxLocal; %#ok<AGROW>
    end
    return;
end

if ~iscell(labelsOrEmpty)
    error('config.areasToPlot must be a cell array of area name strings.');
end

areaLookup = containers.Map(areas(:), num2cell(1:numel(areas)));
plotAreaIndices = [];
for lblIdxLocal = 1:numel(labelsOrEmpty)
    nameStrLocal = labelsOrEmpty{lblIdxLocal};
    if ~ischar(nameStrLocal) && ~(isstring(nameStrLocal) && isscalar(nameStrLocal))
        warning('Skipping non-character area entry at index %d.', lblIdxLocal);
        continue;
    end
    if isstring(nameStrLocal)
        nameStrLocal = char(nameStrLocal);
    end
    if ~isKey(areaLookup, nameStrLocal)
        warning('Unknown area label "%s" in config.areasToPlot.', nameStrLocal);
        continue;
    end
    plotAreaIndices(end + 1) = areaLookup(nameStrLocal); %#ok<AGROW>
end

plotAreaIndices = unique(plotAreaIndices, 'stable');

end

function [hiddenTotalSeq, penalizedLlRow] = local_curve_from_results(results, areaIdxLocal, hmmR)
% LOCAL_CURVE_FROM_RESULTS Prefer top-level bundle; fallback to hmmR.LLtot.

hiddenTotalSeq = [];
penalizedLlRow = [];

if isfield(results, 'llCriterionAcrossStates') ...
        && numel(results.llCriterionAcrossStates) >= areaIdxLocal ...
        && ~isempty(results.llCriterionAcrossStates{areaIdxLocal})
    bundleLocal = results.llCriterionAcrossStates{areaIdxLocal};
    if isfield(bundleLocal, 'hidden_total') && isfield(bundleLocal, 'penalized_minus2_ll')
        hiddenTotalSeq = bundleLocal.hidden_total(:)';
        penalizedLlRow = bundleLocal.penalized_minus2_ll(:)';
        return;
    end
end

if ~isfield(hmmR, 'HmmParam') || ~isfield(hmmR, 'LLtot')
    return;
end
hiddenTotalSeq = hmmR.HmmParam.VarStates(:)';
penalizedLlRow = hmmR.LLtot.m2LL(:, 1)';
end

function llSemAcrossStates = compute_ll_sem_across_training_runs(hmmData, numHiddenStatesCols)
% COMPUTE_LL_SEM_ACROSS_TRAINING_RUNS Mirrors hmm.funHMM shading (nanstd / sqrt(n)).

llSemAcrossStates = [];

if nargin < 2 || isempty(hmmData) || ~isstruct(hmmData) || ~isfield(hmmData, 'LLtrain')
    return;
end

maxColLocal = min(numHiddenStatesCols, size(hmmData, 2));
if maxColLocal < 1
    return;
end

llSemAcrossStates = NaN(1, maxColLocal);
for hiddenColIdxLocal = 1:maxColLocal
    llTrainVals = [hmmData(:, hiddenColIdxLocal).LLtrain]';
    llTrainVals = llTrainVals(:);
    if isempty(llTrainVals)
        continue;
    end
    llSemAcrossStates(hiddenColIdxLocal) = nanstd(llTrainVals) ./ sqrt(max(numel(llTrainVals), 1));
end

end
