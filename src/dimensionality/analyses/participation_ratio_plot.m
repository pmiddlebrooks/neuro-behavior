function participation_ratio_plot(results, plotConfig, config, dataStruct)
% PARTICIPATION_RATIO_PLOT Plot participation ratio and pop activity time series
%
% Variables:
%   results    - Results from participation_ratio_analysis()
%   plotConfig - From setup_plotting()
%   config     - Configuration struct
%   dataStruct - From load_sliding_window_data()
%
% Goal:
%   Create time series plots: left axis = raw or normalized PR (config.plotNormalizedPR);
%   right axis = pop activity normalized to max, and PR/n, with ylim [0 1].

    utilsPath = fullfile(fileparts(mfilename('fullpath')), '..', '..', 'sliding_window_prep', 'utils');
    if exist(utilsPath, 'dir')
        addpath(utilsPath);
    end

    % Left axis: raw or normalized PR based on config
    plotNormalizedPR = false;
    if exist('config', 'var') && ~isempty(config) && isfield(config, 'plotNormalizedPR') && config.plotNormalizedPR
        hasNormalized = isfield(results, 'participationRatioNormalized') && ...
            any(cellfun(@(x) ~isempty(x) && any(~isnan(x)), results.participationRatioNormalized));
        if hasNormalized
            plotNormalizedPR = true;
        end
    end

    areas = results.areas;
    participationRatio = results.participationRatio;
    if plotNormalizedPR
        plotPR = results.participationRatioNormalized;
        prYLabel = 'PR (normalized)';
    else
        plotPR = participationRatio;
        prYLabel = 'Participation ratio';
    end
    startS = results.startS;
    if isfield(results, 'popActivityWindows')
        popActivityWindows = results.popActivityWindows;
    else
        popActivityWindows = cell(1, length(areas));
    end

    areasToTest = 1:length(areas);
    sessionType = results.sessionType;
    % Window size for title: use max per-area window (s) or "per area (multiple=N)"
    if isfield(results.params, 'windowSizeMax') && ~isempty(results.params.windowSizeMax)
        winLabel = sprintf('win~%gs', results.params.windowSizeMax);
    elseif isfield(results.params, 'windowSizeNeuronMultiple') && ~isempty(results.params.windowSizeNeuronMultiple)
        winLabel = sprintf('win per area (multiple=%d)', results.params.windowSizeNeuronMultiple);
    elseif isfield(results.params, 'slidingWindowSize') && ~isempty(results.params.slidingWindowSize)
        winLabel = sprintf('win=%gs', results.params.slidingWindowSize);  % backward compat
    else
        winLabel = 'win per area';
    end

    figure(919); clf;
    set(gcf, 'Units', 'pixels');
    set(gcf, 'Position', plotConfig.targetPos);

    numRows = length(areasToTest) + 1;
    useTightSubplot = exist('tight_subplot', 'file');
    if useTightSubplot
        ha = tight_subplot(numRows, 1, [0.035 0.04], [0.03 0.08], [0.08 0.04]);
    else
        ha = zeros(numRows, 1);
        for i = 1:numRows
            ha(i) = subplot(numRows, 1, i);
        end
    end

    % Colors: M23 (pink), M56 (green), DS (blue), VS (magenta), M2356 (orange), ...
    areaColors = {[1 0.6 0.6], [0 .8 0], [0 0 1], [1 .4 1], [1 0.5 0], [1 0.6 0]};
    % Right axis: dark brown-yellow for ticks, tick labels, ylabel
    rightAxisColor = [0.55 0.4 0.1];

    % Use saved PR/nNeurons if available; otherwise compute from participationRatio and idMatIdx
    prOverN = cell(1, length(areas));
    if isfield(results, 'participationRatioOverNeurons') && ...
            any(cellfun(@(x) ~isempty(x) && any(~isnan(x)), results.participationRatioOverNeurons))
        prOverN = results.participationRatioOverNeurons;
    else
        for a = areasToTest
            nNeurons = 0;
            if isfield(dataStruct, 'idMatIdx') && a <= length(dataStruct.idMatIdx) && ~isempty(dataStruct.idMatIdx{a})
                nNeurons = length(dataStruct.idMatIdx{a});
            end
            if ~isempty(participationRatio{a}) && nNeurons > 0
                prOverN{a} = participationRatio{a} / nNeurons;
            else
                prOverN{a} = [];
            end
        end
    end

    % Global y-limits for left axis (raw PR only; right axis is [0 1])
    allPRForYLim = [];
    for a = areasToTest
        if ~isempty(plotPR{a})
            allPRForYLim = [allPRForYLim; plotPR{a}(:)];
        end
    end
    if ~isempty(allPRForYLim)
        validPR = allPRForYLim(~isnan(allPRForYLim));
        if ~isempty(validPR)
            yMinPR = min(validPR);
            yMaxPR = max(validPR);
            yRangePR = yMaxPR - yMinPR;
            if yRangePR <= 0
                yRangePR = 1;
            end
            yLimitsPR = [yMinPR - 0.05 * yRangePR, yMaxPR + 0.05 * yRangePR];
        else
            yLimitsPR = [0, 1];
        end
    else
        yLimitsPR = [0, 1];
    end

    firstNonEmptyArea = [];
    for a = areasToTest
        if ~isempty(startS{a})
            firstNonEmptyArea = a;
            break;
        end
    end
    if isempty(firstNonEmptyArea)
        warning('No areas with valid startS. Skipping plot.');
        return;
    end

    if ~strcmp(dataStruct.sessionType, 'schall')
        add_event_markers(dataStruct, startS, 'firstNonEmptyArea', firstNonEmptyArea);
    end

    % Top row: all areas participation ratio
    axes(ha(1));
    hold on;
    allStartS = [];
    for a = areasToTest
        if ~isempty(startS{a})
            allStartS = [allStartS, startS{a}];
        end
    end
    if ~isempty(allStartS)
        xLimitsCombined = [min(allStartS), max(allStartS)];
        allPR = [];
        for idx = 1:length(areasToTest)
            a = areasToTest(idx);
            if ~isempty(plotPR{a})
                allPR = [allPR; plotPR{a}(:)];
            end
        end
        for idx = 1:length(areasToTest)
            a = areasToTest(idx);
            areaColor = areaColors{min(a, length(areaColors))};
            if ~isempty(plotPR{a}) && ~isempty(startS{a})
                validIdx = ~isnan(plotPR{a});
                if any(validIdx)
                    plot(startS{a}(validIdx), plotPR{a}(validIdx), '-', ...
                        'Color', areaColor, 'LineWidth', 3, 'DisplayName', areas{a});
                end
            end
        end
        xlim(xLimitsCombined);
        ylim(yLimitsPR);
        if plotNormalizedPR
            yline(1, 'k--', 'LineWidth', 1, 'Alpha', 0.5, 'HandleVisibility', 'off');
        end
        ylabel(prYLabel);
        title(['All Areas - ', prYLabel, ' (left), Pop act norm, PR/n (right)']);
        if length(areasToTest) > 1
            legend('Location', 'best');
        end
        grid on;
        set(gca, 'YTickLabelMode', 'auto', 'YTickMode', 'auto');

        % Right axis: popActivityWindows normalized to max, and PR/n
        yyaxis right;
        for idx = 1:length(areasToTest)
            a = areasToTest(idx);
            areaColor = areaColors{min(a, length(areaColors))};
            if ~isempty(popActivityWindows{a}) && ~isempty(startS{a})
                validIdx = ~isnan(popActivityWindows{a});
                if any(validIdx)
                    vals = popActivityWindows{a}(validIdx);
                    maxVal = max(vals);
                    if maxVal > 0
                        plot(startS{a}(validIdx), vals / maxVal, '-', ...
                            'Color', areaColor, 'LineWidth', 1.5, 'HandleVisibility', 'off');
                    end
                end
            end
            if ~isempty(prOverN{a}) && ~isempty(startS{a})
                validIdx = ~isnan(prOverN{a});
                if any(validIdx)
                    plot(startS{a}(validIdx), prOverN{a}(validIdx), '--', ...
                        'Color', areaColor, 'LineWidth', 1.5, 'HandleVisibility', 'off');
                end
            end
        end
        ylim([0 1]);
        ylabel('Pop act (norm), PR/n', 'Color', rightAxisColor);
        set(gca, 'YColor', rightAxisColor, 'YTickLabelMode', 'auto', 'YTickMode', 'auto');
        yyaxis left;
    end

    % Per-area rows
    for idx = 1:length(areasToTest)
        a = areasToTest(idx);
        if useTightSubplot
            axes(ha(idx + 1));
        else
            subplot(numRows, 1, idx + 1);
        end
        hold on;
        add_event_markers(dataStruct, startS, 'areaIdx', a);

        yyaxis left;
        areaColor = areaColors{min(a, length(areaColors))};
        if ~isempty(plotPR{a}) && ~isempty(startS{a})
            validIdx = ~isnan(plotPR{a});
            if any(validIdx)
                plot(startS{a}(validIdx), plotPR{a}(validIdx), '-', ...
                    'Color', areaColor, 'LineWidth', 3, 'DisplayName', 'PR');
            end
        end
        ylabel(prYLabel, 'Color', areaColor);
        ylim(yLimitsPR);
        if plotNormalizedPR
            yline(1, 'k--', 'LineWidth', 1, 'Alpha', 0.5, 'HandleVisibility', 'off');
        end
        set(gca, 'YTickLabelMode', 'auto', 'YTickMode', 'auto');

        % Right axis: popActivityWindows normalized to max, and PR/n; ylim [0 1]
        yyaxis right;
        if ~isempty(popActivityWindows{a}) && ~isempty(startS{a})
            validIdx = ~isnan(popActivityWindows{a});
            if any(validIdx)
                vals = popActivityWindows{a}(validIdx);
                maxVal = max(vals);
                if maxVal > 0
                    plot(startS{a}(validIdx), vals / maxVal, '-', ...
                        'Color', [.7 .7 .7], 'LineWidth', 2, 'DisplayName', 'Pop act (norm)');
                end
            end
        end
        if ~isempty(prOverN{a}) && ~isempty(startS{a})
            validIdx = ~isnan(prOverN{a});
            if any(validIdx)
                plot(startS{a}(validIdx), prOverN{a}(validIdx), '--', ...
                    'Color', areaColor, 'LineWidth', 2, 'DisplayName', 'PR/n');
            end
        end
        ylim([0 1]);
        ylabel('Pop act (norm), PR/n', 'Color', rightAxisColor);
        set(gca, 'YColor', rightAxisColor, 'YTickLabelMode', 'auto', 'YTickMode', 'auto');

        if ~isempty(startS{a})
            xlim([startS{a}(1), startS{a}(end)]);
        end
        nNeurons = 0;
        if isfield(dataStruct, 'idMatIdx') && a <= length(dataStruct.idMatIdx) && ~isempty(dataStruct.idMatIdx{a})
            nNeurons = length(dataStruct.idMatIdx{a});
        end
        title(sprintf('%s (n=%d) - PR (left), Pop act norm, PR/n (right)', areas{a}, nNeurons));
        if idx == length(areasToTest)
            xlabel('Time (s)');
        end
        grid on;
        set(gca, 'XTickLabelMode', 'auto');
    end

    if ~isempty(plotConfig.filePrefix)
        sgtitle(sprintf('[%s] %s PR (left), Pop act norm, PR/n (right) - %s', ...
            plotConfig.filePrefix, sessionType, winLabel));
    else
        sgtitle(sprintf('%s PR (left), Pop act norm, PR/n (right) - %s', ...
            sessionType, winLabel));
    end

    if ~exist('config', 'var') || isempty(config)
        config = struct();
    end
    if ~isfield(config, 'saveDir') || isempty(config.saveDir)
        config.saveDir = pwd;
    end
    if ~exist(config.saveDir, 'dir')
        mkdir(config.saveDir);
    end

    if ~isempty(plotConfig.filePrefix)
        plotPath = fullfile(config.saveDir, sprintf('%s_participation_ratio_%s.png', plotConfig.filePrefix, sessionType));
    else
        plotPath = fullfile(config.saveDir, sprintf('participation_ratio_%s.png', sessionType));
    end
    drawnow
    pause(5);
    exportgraphics(gcf, plotPath, 'Resolution', 300);
    fprintf('Saved plot to: %s\n', plotPath);

    if ~isempty(plotConfig.filePrefix)
        plotPathEps = fullfile(config.saveDir, sprintf('%s_participation_ratio_%s.eps', plotConfig.filePrefix, sessionType));
    else
        plotPathEps = fullfile(config.saveDir, sprintf('participation_ratio_%s.eps', sessionType));
    end
    exportgraphics(gcf, plotPathEps, 'ContentType', 'vector');
    fprintf('Saved plot to: %s\n', plotPathEps);
end
