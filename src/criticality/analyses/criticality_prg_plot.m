function criticality_prg_plot(results, plotConfig, config, dataStruct)
% CRITICALITY_PRG_PLOT Plot PRG kurtosis time series and real vs surrogate distributions
%
% Kurtosis axes are shared across all area tiles: Fig 1 uses one y-range for kappa vs
% time; Fig 2 (when surrogates exist) uses one x-range and identical histogram bin edges.
% Upper kurtosis limit is config.kappaAxisMax / results.params.kappaAxisMax (default 20; Inf = no cap).
%
% Variables:
%   results - Output of criticality_prg_analysis()
%   plotConfig - From setup_plotting()
%   config - Analysis configuration (optional field kappaAxisMax; default from results.params or 20)
%   dataStruct - Loaded session data

    areas = results.areas;
    numAreas = numel(areas);

    kappaAxisMax = prg_get_kappa_axis_max(results, config);

    % Shared kurtosis (kappa) vertical scale across all area tiles (Fig 1)
    allKappaTimeSeries = collect_prg_kappa_for_axis_limits(results, true);
    [kappaYLimLo, kappaYLimHi] = prg_padded_kurtosis_limits(allKappaTimeSeries);
    [kappaYLimLo, kappaYLimHi] = prg_apply_kappa_axis_max(kappaYLimLo, kappaYLimHi, kappaAxisMax);
    hasSharedKappaY = isfinite(kappaYLimLo) && isfinite(kappaYLimHi);

    %% Figure 1: kappa along session (one trace per area)
    fig = figure('Color', 'w', 'Position', [80 80 1100 280 * numAreas]);
    tiledLayoutObj = tiledlayout(numAreas, 1, 'TileSpacing', 'compact', 'Padding', 'compact');

    for a = 1:numAreas
        nexttile(tiledLayoutObj);
        hold on;

        if isempty(results.windowStartS{a}) || isempty(results.kappa{a})
            title(sprintf('%s (no data)', areas{a}));
            if hasSharedKappaY
                ylim([kappaYLimLo, kappaYLimHi]);
            end
            continue;
        end

        tVec = results.windowStartS{a};
        kappaVec = results.kappa{a};
        validMask = isfinite(kappaVec) & ~results.windowExcluded{a};

        plot(tVec(validMask), kappaVec(validMask), '-o', ...
            'Color', [0.15 0.45 0.75], 'LineWidth', 1.1, 'MarkerSize', 4);

        excludedMask = results.windowExcluded{a};
        if any(excludedMask)
            plot(tVec(excludedMask), nanmean(kappaVec(validMask)) * ones(1, sum(excludedMask)), ...
                'x', 'Color', [0.7 0.2 0.2], 'MarkerSize', 6);
        end

        yline(3, '--', 'Color', [0.4 0.4 0.4], 'LineWidth', 1);
        grid on;
        xlabel('Window start (s)');
        ylabel(sprintf('\\kappa (N/%d)', results.params.finalCutoffDivisor));
        title(areas{a});

        if isfield(results, 'kappaSurrogate') && ~isempty(results.kappaSurrogate{a})
            surrMean = nanmean(results.kappaSurrogate{a}, 2);
            plot(tVec, surrMean, '--', 'Color', [0.55 0.55 0.55], 'LineWidth', 1);
        end

        if hasSharedKappaY
            ylim([kappaYLimLo, kappaYLimHi]);
        end
    end

    sessionLabel = '';
    if isfield(dataStruct, 'sessionName') && ~isempty(dataStruct.sessionName)
        sessionLabel = dataStruct.sessionName;
    end
    sgtitle(tiledLayoutObj, sprintf('PRG kurtosis | %s | %.0f s blocks, %.0f ms bins%s', ...
        results.sessionType, results.params.blockWindowSize, ...
        results.params.binSize * 1000, prg_title_suffix(sessionLabel)), ...
        'FontSize', 12, 'Interpreter', 'none');

    savePlots = false;
    saveDir = '';
    if isfield(plotConfig, 'savePlots') && plotConfig.savePlots
        savePlots = true;
    end
    if isfield(plotConfig, 'saveDir') && ~isempty(plotConfig.saveDir)
        saveDir = plotConfig.saveDir;
    elseif isfield(config, 'saveDir') && ~isempty(config.saveDir)
        saveDir = config.saveDir;
    end

    if savePlots && ~isempty(saveDir)
        pngPath = fullfile(saveDir, sprintf('criticality_prg_%s.png', prg_safe_name(sessionLabel)));
        exportgraphics(fig, pngPath, 'Resolution', 200);
        fprintf('Saved PRG plot: %s\n', pngPath);
    end

    %% Figure 2: overlapping distributions of kappa (real vs surrogate), Cambrainha-style
    % Paper Fig. 3 compares real vs null; here we show probability density of
    % window-wise kappa for data vs ISI-shuffled surrogates on the same axes per area.
    if has_prg_surrogate_kappa(results)
        figDist = figure('Color', 'w', 'Position', [120 120 900 260 * numAreas]);
        tlDist = tiledlayout(numAreas, 1, 'TileSpacing', 'compact', 'Padding', 'compact');

        % Same kurtosis horizontal range and bin edges for every area (Fig 2)
        allKappaDist = collect_prg_kappa_for_axis_limits(results, true);
        [distKappaLo, distKappaHi] = prg_padded_kurtosis_limits(allKappaDist);
        [distKappaLo, distKappaHi] = prg_apply_kappa_axis_max(distKappaLo, distKappaHi, kappaAxisMax);
        nBinsDist = 28;
        if isfinite(distKappaLo) && isfinite(distKappaHi)
            globalDistBinEdges = linspace(distKappaLo, distKappaHi, max(8, round(nBinsDist)) + 1);
        else
            globalDistBinEdges = [];
        end
        hasSharedKappaX = ~isempty(globalDistBinEdges);

        for a = 1:numAreas
            nexttile(tlDist);
            hold on;

            if isempty(results.kappa{a}) || isempty(results.windowExcluded{a})
                title(sprintf('%s (no data)', areas{a}));
                if hasSharedKappaX
                    xlim([globalDistBinEdges(1), globalDistBinEdges(end)]);
                end
                continue;
            end

            validMask = isfinite(results.kappa{a}) & ~results.windowExcluded{a};
            realKappa = results.kappa{a}(validMask);
            surrMat = [];
            if isfield(results, 'kappaSurrogate') && ~isempty(results.kappaSurrogate{a})
                surrMat = results.kappaSurrogate{a}(validMask, :);
            end

            if isempty(realKappa)
                title(sprintf('%s (no valid \\kappa)', areas{a}));
                grid on;
                xlabel('Kurtosis \kappa');
                ylabel('Probability density');
                if hasSharedKappaX
                    xlim([globalDistBinEdges(1), globalDistBinEdges(end)]);
                end
                continue;
            end

            surrKappa = [];
            if ~isempty(surrMat)
                surrKappa = surrMat(isfinite(surrMat));
            end

            % Global bin edges so tiles share the same kappa axis and binning
            if hasSharedKappaX
                binEdges = globalDistBinEdges;
            else
                [binEdges, ~] = prg_kurtosis_bin_edges(realKappa, surrKappa, nBinsDist);
            end

            histogram(realKappa, binEdges, 'Normalization', 'pdf', ...
                'FaceColor', [0.15 0.45 0.75], 'FaceAlpha', 0.45, 'EdgeColor', 'none', ...
                'DisplayName', 'Data');

            if ~isempty(surrKappa)
                histogram(surrKappa, binEdges, 'Normalization', 'pdf', ...
                    'FaceColor', [0.55 0.55 0.55], 'FaceAlpha', 0.4, 'EdgeColor', 'none', ...
                    'DisplayName', 'Surrogate');
            end

            xline(3, '--', 'Color', [0.35 0.35 0.35], 'LineWidth', 1);
            yLimAxis = ylim;
            text(3, yLimAxis(1) + 0.92 * (yLimAxis(2) - yLimAxis(1)), ...
                ' \kappa=3 (Gaussian)', 'Color', [0.25 0.25 0.25], ...
                'FontSize', 9, 'Interpreter', 'tex', 'VerticalAlignment', 'top');
            grid on;
            xlabel(sprintf('Kurtosis \\kappa (N/%d)', results.params.finalCutoffDivisor));
            ylabel('Probability density');
            title(areas{a}, 'Interpreter', 'none');
            legend('Location', 'northeast');

            if hasSharedKappaX
                xlim([binEdges(1), binEdges(end)]);
            end
        end

        sgtitle(tlDist, sprintf( ...
            'Distribution of PRG kurtosis | real vs surrogate | %s%s', ...
            results.sessionType, prg_title_suffix(sessionLabel)), ...
            'FontSize', 12, 'Interpreter', 'none');

        if savePlots && ~isempty(saveDir)
            distPath = fullfile(saveDir, sprintf('criticality_prg_kappa_distribution_%s.png', prg_safe_name(sessionLabel)));
            exportgraphics(figDist, distPath, 'Resolution', 200);
            fprintf('Saved PRG kappa distribution plot: %s\n', distPath);
        end
    end
end

function tf = has_prg_surrogate_kappa(results)
% HAS_PRG_SURROGATE_KAPPA True if any area has finite surrogate kappa values
    tf = false;
    if ~isfield(results, 'kappaSurrogate')
        return;
    end
    for a = 1:numel(results.kappaSurrogate)
        s = results.kappaSurrogate{a};
        if isempty(s)
            continue;
        end
        if any(isfinite(s(:)))
            tf = true;
            return;
        end
    end
end

function kappaAxisMax = prg_get_kappa_axis_max(results, config)
% PRG_GET_KAPPA_AXIS_MAX Upper cap for kurtosis axes in PRG plots
%
% Variables:
%   results - Saved or fresh analysis struct (optional: results.params.kappaAxisMax)
%   config  - Run config (optional: config.kappaAxisMax); overrides results.params when set
%
% Goal:
%   Return scalar limit; Inf means no cap. Default 20 when unset.

    kappaAxisMax = 20;
    if nargin >= 2 && isstruct(config) && isfield(config, 'kappaAxisMax')
        v = config.kappaAxisMax;
        if prg_is_valid_kappa_axis_max(v)
            kappaAxisMax = v;
            return;
        end
    end
    if isfield(results, 'params') && isfield(results.params, 'kappaAxisMax')
        v = results.params.kappaAxisMax;
        if prg_is_valid_kappa_axis_max(v)
            kappaAxisMax = v;
        end
    end
end

function tf = prg_is_valid_kappa_axis_max(v)
% PRG_IS_VALID_KAPPA_AXIS_MAX True for positive scalar kurtosis cap (including Inf)
    tf = isnumeric(v) && isscalar(v) && v > 0;
end

function [loOut, hiOut] = prg_apply_kappa_axis_max(lo, hi, kappaAxisMax)
% PRG_APPLY_KAPPA_AXIS_MAX Cap upper kurtosis axis; ensure lo < hi
%
% Variables:
%   lo, hi          - Axis limits from padded data range
%   kappaAxisMax    - Upper cap; ignored if not finite (e.g. Inf)
%
% Goal:
%   Match plot kurtosis max to config; if all data sit above the cap, widen downward slightly.

    loOut = lo;
    hiOut = hi;
    if ~isfinite(loOut) || ~isfinite(hiOut)
        return;
    end
    if isfinite(kappaAxisMax)
        hiOut = min(hiOut, kappaAxisMax);
    end
    if hiOut <= loOut
        spanBelow = max(0.5, 0.1 * abs(hiOut));
        loOut = hiOut - spanBelow;
    end
end

function allVals = collect_prg_kappa_for_axis_limits(results, includeSurrogate)
% COLLECT_PRG_KAPPA_FOR_AXIS_LIMITS Pool finite kappa across areas (valid windows only)
%
% Variables:
%   results          - Output of criticality_prg_analysis()
%   includeSurrogate - If true, append finite surrogate kappa for the same valid windows
%
% Goal:
%   Build one value list for shared kurtosis axis limits across all area tiles in a figure.

    allVals = [];
    for a = 1:numel(results.areas)
        if isempty(results.kappa{a}) || isempty(results.windowExcluded{a})
            continue;
        end
        validMask = isfinite(results.kappa{a}) & ~results.windowExcluded{a};
        kappaVec = results.kappa{a}(validMask);
        allVals = [allVals; kappaVec(:)]; %#ok<AGROW>
        if includeSurrogate && isfield(results, 'kappaSurrogate') && ~isempty(results.kappaSurrogate{a})
            surrMat = results.kappaSurrogate{a}(validMask, :);
            allVals = [allVals; surrMat(isfinite(surrMat))]; %#ok<AGROW>
        end
    end
end

function [lo, hi] = prg_padded_kurtosis_limits(allVals)
% PRG_PADDED_KURTOSIS_LIMITS Padded [min, max] for kurtosis axes
%
% Variables:
%   allVals - Vector of kappa samples (may be empty)
%
% Goal:
%   Match the padding rule in prg_kurtosis_bin_edges so time-series y and histogram x align.

    allVals = allVals(isfinite(allVals(:)));
    if isempty(allVals)
        lo = NaN;
        hi = NaN;
        return;
    end
    lo = min(allVals);
    hi = max(allVals);
    span = hi - lo;
    if span <= 0 || ~isfinite(span)
        pad = max(0.5, abs(lo) * 0.05 + eps);
        lo = lo - pad;
        hi = hi + pad;
    else
        pad = 0.03 * span;
        lo = lo - pad;
        hi = hi + pad;
    end
end

function [binEdges, nBinsOut] = prg_kurtosis_bin_edges(realKappa, surrKappa, nBinsTarget)
% PRG_KURTOSIS_BIN_EDGES Common histogram edges for real and surrogate kappa
%
% Variables:
%   realKappa   - vector of finite real kappa values
%   surrKappa   - vector of finite surrogate kappa (can be empty)
%   nBinsTarget - approximate number of bins
%
% Goal:
%   Avoid degenerate range when values are identical; pad slightly for PDF display.

    allVals = realKappa(:);
    if ~isempty(surrKappa)
        allVals = [allVals; surrKappa(:)];
    end
    allVals = allVals(isfinite(allVals));

    lo = min(allVals);
    hi = max(allVals);
    span = hi - lo;
    if span <= 0 || ~isfinite(span)
        pad = max(0.5, abs(lo) * 0.05 + eps);
        lo = lo - pad;
        hi = hi + pad;
        span = hi - lo;
    else
        pad = 0.03 * span;
        lo = lo - pad;
        hi = hi + pad;
    end

    nBinsOut = max(8, round(nBinsTarget));
    binEdges = linspace(lo, hi, nBinsOut + 1);
end

function nameStr = prg_safe_name(sessionLabel)
% PRG_SAFE_NAME Filename fragment from session label
    if isempty(sessionLabel)
        nameStr = 'session';
    else
        nameStr = strrep(sessionLabel, filesep, '_');
        nameStr = regexprep(nameStr, '[<>:\"/\\|?*]', '_');
        if numel(nameStr) > 80
            nameStr = nameStr(1:80);
        end
    end
end

function suffixStr = prg_title_suffix(sessionLabel)
    if isempty(sessionLabel)
        suffixStr = '';
    else
        suffixStr = [' | ' sessionLabel];
    end
end
