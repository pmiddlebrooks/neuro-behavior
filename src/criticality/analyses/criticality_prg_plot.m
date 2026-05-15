function criticality_prg_plot(results, plotConfig, config, dataStruct)
% CRITICALITY_PRG_PLOT Plot PRG kurtosis time series and real vs surrogate distributions
%
% Variables:
%   results - Output of criticality_prg_analysis()
%   plotConfig - From setup_plotting()
%   config - Analysis configuration
%   dataStruct - Loaded session data

    areas = results.areas;
    numAreas = numel(areas);

    %% Figure 1: kappa along session (one trace per area)
    fig = figure('Color', 'w', 'Position', [80 80 1100 280 * numAreas]);
    tiledLayoutObj = tiledlayout(numAreas, 1, 'TileSpacing', 'compact', 'Padding', 'compact');

    for a = 1:numAreas
        nexttile(tiledLayoutObj);
        hold on;

        if isempty(results.windowStartS{a}) || isempty(results.kappa{a})
            title(sprintf('%s (no data)', areas{a}));
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

        for a = 1:numAreas
            nexttile(tlDist);
            hold on;

            if isempty(results.kappa{a}) || isempty(results.windowExcluded{a})
                title(sprintf('%s (no data)', areas{a}));
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
                continue;
            end

            surrKappa = [];
            if ~isempty(surrMat)
                surrKappa = surrMat(isfinite(surrMat));
            end

            % Shared bin edges so both histograms are directly comparable (normalized PDF)
            [binEdges, ~] = prg_kurtosis_bin_edges(realKappa, surrKappa, 28);

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
