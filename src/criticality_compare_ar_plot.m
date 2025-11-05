%%
% Criticality D2/MrBr Plotting Script
% Plots and correlates d2 (distance to criticality) and mrBr (MR branching ratio) measures
% Uses results from criticality_compare.m (d2/mrBr analysis)
%
% Update controls:
%   loadExistingResults - load existing saved results to update selectively
%   makePlots          - generate plots if true
%   runCorrelation     - compute correlation matrices if true

%%
opts = neuro_behavior_options;
opts.minActTime = .16;
opts.collectStart = 0 * 60 * 60; % seconds
opts.minFiringRate = .05;
opts.frameSize = .001;

paths = get_paths;

% =============================    Update/Run Flags    =============================
loadExistingResults = true;   % load existing results file to preserve untouched fields
makePlots = true;            % create comparison plots
runCorrelation = false;       % compute correlation matrices

% Areas to analyze
areas = {'M23', 'M56', 'DS', 'VS'};
areasToTest = 1:4;

% Data collection parameters
opts.collectStart = 0 * 60; % seconds
opts.collectEnd = 45 * 60; % seconds

%% Load existing results if requested
slidingWindowSize = 30;        % For d2, use a small window to try to optimize temporal resolution
slidingWindowSize = 60;        % For d2, use a small window to try to optimize temporal resolution

resultsPathD2MrBr = fullfile(paths.dropPath, sprintf('criticality/criticality_compare_ar_win%d.mat', slidingWindowSize));
results = struct();
if loadExistingResults
    if exist(resultsPathD2MrBr, 'file')
        loaded = load(resultsPathD2MrBr);
    else
        loaded = struct();
    end
    if isfield(loaded, 'results')
        results = loaded.results;
    end
end

% Criticality parameter ranges for reference
tauRange = [1.2 2.5];
alphaRange = [1.5 2.2];
paramSDRange = [1.3 1.7];

% Monitor setup
monitorPositions = get(0, 'MonitorPositions');
monitorOne = monitorPositions(1, :);
monitorTwo = monitorPositions(size(monitorPositions, 1), :);

%% ==============================================     Plotting Results     ==============================================

% Create comparison plots for each area with d2/mrBr measures
measures = {'mrBr', 'd2'};
measureNames = {'MR Branching Ratio', 'Distance to Criticality (d2)'};

% Colors for different measures
measureColors = {'k', 'b'};

% Create one figure per area
if makePlots
    for a = areasToTest
        figure(100 + a); clf;
        set(gcf, 'Position', monitorTwo);

        % Use tight_subplot for 2x1 layout
        ha = tight_subplot(2, 1, [0.05 0.02], [0.1 0.05], [0.1 0.05]);

        % Plot d2
        axes(ha(1));
        hold on;
        plot(results.naturalistic.startS{a}/60, results.naturalistic.d2{a}, '-', 'Color', 'b', 'LineWidth', 2, 'MarkerSize', 4);
        plot(results.reach.startS{a}/60, results.reach.d2{a}, '--', 'Color', 'b', 'LineWidth', 2, 'MarkerSize', 4);
        ylabel('Distance to Criticality (d2)', 'FontSize', 14);
        title(sprintf('%s - Distance to Criticality (d2)', areas{a}), 'FontSize', 14);
        legend({'Naturalistic', 'Reach'}, 'Location', 'best', 'FontSize', 14);
        grid on;
        set(gca, 'XTickLabel', [], 'FontSize', 14); % Remove x-axis labels for all but bottom subplot
        set(gca, 'YTickLabelMode', 'auto');  % Enable Y-axis labels
        xlim([opts.collectStart/60 (opts.collectStart+opts.collectEnd)/60]);

        % Plot mrBr
        axes(ha(2));
        hold on;
        plot(results.naturalistic.startS{a}/60, results.naturalistic.mrBr{a}, '-', 'Color', 'k', 'LineWidth', 2, 'MarkerSize', 4);
        plot(results.reach.startS{a}/60, results.reach.mrBr{a}, '--', 'Color', 'k', 'LineWidth', 2, 'MarkerSize', 4);
        ylabel('MR Branching Ratio', 'FontSize', 14);
        title(sprintf('%s - MR Branching Ratio', areas{a}), 'FontSize', 14);
        legend({'Naturalistic', 'Reach'}, 'Location', 'best', 'FontSize', 14);
        grid on;
        xlabel('Minutes', 'FontSize', 14); % Only add xlabel to bottom subplot
        set(gca, 'YTickLabelMode', 'auto', 'FontSize', 14);  % Enable Y-axis labels
        set(gca, 'XTickLabelMode', 'auto');  % Enable X-axis labels
        xlim([opts.collectStart/60 (opts.collectStart+opts.collectEnd)/60]);

        sgtitle(sprintf('D2/MrBr Criticality Measures - %s', areas{a}), 'FontSize', 14);

        % Save PNG using exportgraphics
        filename = fullfile(paths.dropPath, sprintf('criticality/d2mrbr_comparison_%s_win%gs.png', areas{a}, slidingWindowSize));
        exportgraphics(gcf, filename, 'Resolution', 300);
        fprintf('Saved d2/mrBr area plot to: %s\n', filename);

    end
end
% ==============================================     Correlation Analysis     ==============================================

if runCorrelation

    % Calculate correlations between d2 and mrBr (original sliding window timebase)
    fprintf('\n=== Correlation Analysis: d2 and mrBr ===\n');
    corrMatNat_d2mrBr = cell(1, length(areas));
    corrMatRea_d2mrBr = cell(1, length(areas));

    for a = areasToTest
        % Naturalistic data
        Xnat = [results.naturalistic.d2{a}(:), results.naturalistic.mrBr{a}(:)];
        corrMatNat_d2mrBr{a} = corr(Xnat, 'Rows', 'pairwise');

        % Reach data
        Xrea = [results.reach.d2{a}(:), results.reach.mrBr{a}(:)];
        corrMatRea_d2mrBr{a} = corr(Xrea, 'Rows', 'pairwise');
    end

    % Set colorbar scale for all correlation matrices
    cmin = -1;
    cmax = 1;

    % Plot correlation matrices for each area
    figure(201); clf;
    set(gcf, 'Position', monitorTwo);
    for a = areasToTest
        % d2/mrBr correlations
        subplot(2, length(areasToTest), a - areasToTest(1) + 1);
        imagesc(corrMatNat_d2mrBr{a});
        colorbar;
        title(sprintf('%s (Nat d2/mrBr)', areas{a}));
        xticks(1:2); yticks(1:2);
        xticklabels({'d2','mrBr'}); yticklabels({'d2','mrBr'});
        axis square;
        caxis([cmin cmax]); % Set consistent color axis

        subplot(2, length(areasToTest), length(areasToTest) + (a - areasToTest(1) + 1));
        imagesc(corrMatRea_d2mrBr{a});
        colorbar;
        title(sprintf('%s (Rea d2/mrBr)', areas{a}));
        xticks(1:2); yticks(1:2);
        xticklabels({'d2','mrBr'}); yticklabels({'d2','mrBr'});
        axis square;
        caxis([cmin cmax]); % Set consistent color axis
    end
    sgtitle('Correlation Matrices: d2 and mrBr');

    % Save PNG using exportgraphics
    filename = fullfile(paths.dropPath, sprintf('criticality/correlation_matrices_d2_mrbr_win%gs.png', slidingWindowSize));
    exportgraphics(gcf, filename, 'Resolution', 300);
    fprintf('Saved d2/mrBr correlation plot to: %s\n', filename);
end % runCorrelation
