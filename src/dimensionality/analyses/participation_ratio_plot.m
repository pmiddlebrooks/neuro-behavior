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
%   Create time series plots of participation ratio (left axis) and mean
%   population activity per window (right axis), with optional event markers.

    utilsPath = fullfile(fileparts(mfilename('fullpath')), '..', '..', 'sliding_window_prep', 'utils');
    if exist(utilsPath, 'dir')
        addpath(utilsPath);
    end

    areas = results.areas;
    participationRatio = results.participationRatio;
    startS = results.startS;
    if isfield(results, 'popActivityWindows')
        popActivityWindows = results.popActivityWindows;
    else
        popActivityWindows = cell(1, length(areas));
    end

    areasToTest = 1:length(areas);
    sessionType = results.sessionType;
    slidingWindowSize = results.params.slidingWindowSize;

    figure(909); clf;
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

    areaColors = {[1 0.6 0.6], [0 .8 0], [0 0 1], [1 .4 1], [1 0.5 0]};

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
            if ~isempty(participationRatio{a})
                allPR = [allPR; participationRatio{a}(:)];
            end
        end
        for idx = 1:length(areasToTest)
            a = areasToTest(idx);
            if ~isempty(participationRatio{a}) && ~isempty(startS{a})
                validIdx = ~isnan(participationRatio{a});
                if any(validIdx)
                    plot(startS{a}(validIdx), participationRatio{a}(validIdx), '-', ...
                        'Color', areaColors{min(a, length(areaColors))}, ...
                        'LineWidth', 3, 'DisplayName', areas{a});
                end
            end
        end
        xlim(xLimitsCombined);
        if ~isempty(allPR)
            validPR = allPR(~isnan(allPR));
            if ~isempty(validPR)
                ylim([min(validPR), max(validPR)]);
            end
        end
        ylabel('Participation ratio');
        title('All Areas - Participation ratio');
        if length(areasToTest) > 1
            legend('Location', 'best');
        end
        grid on;
        set(gca, 'YTickLabelMode', 'auto', 'YTickMode', 'auto');
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

        if isfield(results, 'popActivityWindows') && ~isempty(popActivityWindows{a}) && ...
                ~isempty(startS{a}) && any(~isnan(popActivityWindows{a}))
            yyaxis right;
            validIdx = ~isnan(popActivityWindows{a});
            if any(validIdx)
                plot(startS{a}(validIdx), popActivityWindows{a}(validIdx), '-', ...
                    'Color', [.7 .7 .7], 'LineWidth', 2, 'HandleVisibility', 'off');
            end
            ylabel('Pop Activity', 'Color', [0.5 0.5 0.5]);
            ylim('auto');
            set(gca, 'YTickLabelMode', 'auto', 'YTickMode', 'auto');
            yyaxis left;
        end

        yyaxis left;
        if ~isempty(participationRatio{a}) && ~isempty(startS{a})
            validIdx = ~isnan(participationRatio{a});
            if any(validIdx)
                areaColor = areaColors{min(a, length(areaColors))};
                plot(startS{a}(validIdx), participationRatio{a}(validIdx), '-', ...
                    'Color', areaColor, 'LineWidth', 3, 'DisplayName', 'PR');
            end
        end
        ylabel('Participation ratio', 'Color', areaColors{min(a, length(areaColors))});
        if ~isempty(participationRatio{a})
            validPR = participationRatio{a}(~isnan(participationRatio{a}));
            if ~isempty(validPR)
                ylim([min(validPR), max(validPR)]);
            end
        end
        set(gca, 'YTickLabelMode', 'auto', 'YTickMode', 'auto');

        if ~isempty(startS{a})
            xlim([startS{a}(1), startS{a}(end)]);
        end
        nNeurons = 0;
        if isfield(dataStruct, 'idMatIdx') && a <= length(dataStruct.idMatIdx) && ~isempty(dataStruct.idMatIdx{a})
            nNeurons = length(dataStruct.idMatIdx{a});
        end
        title(sprintf('%s (n=%d) - Participation ratio (left), Pop activity (right)', areas{a}, nNeurons));
        if idx == length(areasToTest)
            xlabel('Time (s)');
        end
        grid on;
        set(gca, 'XTickLabelMode', 'auto');
        yyaxis left;
        set(gca, 'YTickLabelMode', 'auto', 'YTickMode', 'auto');
        if isfield(results, 'popActivityWindows') && ~isempty(popActivityWindows{a}) && any(~isnan(popActivityWindows{a}))
            yyaxis right;
            set(gca, 'YTickLabelMode', 'auto', 'YTickMode', 'auto');
            yyaxis left;
        end
    end

    if ~isempty(plotConfig.filePrefix)
        sgtitle(sprintf('[%s] %s participation ratio (left), Pop activity (right) - win=%gs', ...
            plotConfig.filePrefix, sessionType, slidingWindowSize));
    else
        sgtitle(sprintf('%s participation ratio (left), Pop activity (right) - win=%gs', ...
            sessionType, slidingWindowSize));
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
