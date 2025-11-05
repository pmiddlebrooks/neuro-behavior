%%
% Criticality Avalanche Plotting Script
% Plots and correlates dcc (distance to criticality from avalanche analysis) and kappa measures
% Uses results from criticality_compare_av.m (avalanche analysis)
%
% Update controls:
%   loadExistingResults - load existing saved results to update selectively
%   makePlots          - generate plots if true
%   runCorrelation     - compute correlation matrices if true

%%
% Sliding window and step size
slidingWindowSize = 2 * 60; % seconds - user specified
avStepSize = 20; %round(.5 * 60); % seconds - user specified

%%
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
resultsPathAvalanche = fullfile(paths.dropPath, sprintf('criticality/criticality_compare_av_win%d_step%d.mat', slidingWindowSize, avStepSize));
results = struct();
if loadExistingResults
    if exist(resultsPathAvalanche, 'file')
        loaded = load(resultsPathAvalanche);
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

% Create comparison plots for each area with dcc/kappa measures
measures = {'dcc', 'kappa', 'decades'};
measureNames = {'Distance to Criticality (dcc)', 'Kappa', 'Decades'};

% Colors for different measures
measureColors = {'r', [0 0.75 0], [0.5 0 0.5]};

% Create one figure per area
if makePlots
    for a = areasToTest
        figure(200 + a); clf;
        set(gcf, 'Position', monitorTwo);

        % Use tight_subplot for 3x1 layout
        ha = tight_subplot(3, 1, [0.05 0.02], [0.1 0.05], [0.1 0.05]);

        % Plot dcc
        axes(ha(1));
        hold on;
        plot(results.naturalistic.startS_dcc{a}/60, results.naturalistic.dcc{a}, '-o', 'Color', 'r', 'LineWidth', 2, 'MarkerSize', 4);
        plot(results.reach.startS_dcc{a}/60, results.reach.dcc{a}, '--s', 'Color', 'r', 'LineWidth', 2, 'MarkerSize', 4);
        ylabel('Distance to Criticality (dcc)', 'FontSize', 14);
        title(sprintf('%s - Distance to Criticality (dcc)', areas{a}), 'FontSize', 14);
        legend({'Naturalistic', 'Reach'}, 'Location', 'best', 'FontSize', 14);
        grid on;
        set(gca, 'XTickLabel', [], 'FontSize', 14); % Remove x-axis labels for all but bottom subplot
        set(gca, 'YTickLabelMode', 'auto');  % Enable Y-axis labels
        xlim([opts.collectStart/60 (opts.collectStart+opts.collectEnd)/60]);

        % Plot kappa
        axes(ha(2));
        hold on;
        plot(results.naturalistic.startS_dcc{a}/60, results.naturalistic.kappa{a}, '-o', 'Color', [0 0.75 0], 'LineWidth', 2, 'MarkerSize', 4);
        plot(results.reach.startS_dcc{a}/60, results.reach.kappa{a}, '--s', 'Color', [0 0.75 0], 'LineWidth', 2, 'MarkerSize', 4);
        ylabel('Kappa', 'FontSize', 14);
        title(sprintf('%s - Kappa', areas{a}), 'FontSize', 14);
        legend({'Naturalistic', 'Reach'}, 'Location', 'best', 'FontSize', 14);
        grid on;
        set(gca, 'XTickLabel', [], 'FontSize', 14); % Remove x-axis labels for all but bottom subplot
        set(gca, 'YTickLabelMode', 'auto');  % Enable Y-axis labels
        xlim([opts.collectStart/60 (opts.collectStart+opts.collectEnd)/60]);

        % Plot decades
        axes(ha(3));
        hold on;
        plot(results.naturalistic.startS_dcc{a}/60, results.naturalistic.decades{a}, '-o', 'Color', [0.5 0 0.5], 'LineWidth', 2, 'MarkerSize', 4);
        plot(results.reach.startS_dcc{a}/60, results.reach.decades{a}, '--s', 'Color', [0.5 0 0.5], 'LineWidth', 2, 'MarkerSize', 4);
        ylabel('Decades', 'FontSize', 14);
        title(sprintf('%s - Decades', areas{a}), 'FontSize', 14);
        legend({'Naturalistic', 'Reach'}, 'Location', 'best', 'FontSize', 14);
        grid on;
        xlabel('Minutes', 'FontSize', 14); % Only add xlabel to bottom subplot
        set(gca, 'YTickLabelMode', 'auto', 'FontSize', 14);  % Enable Y-axis labels
        set(gca, 'XTickLabelMode', 'auto');  % Enable X-axis labels
        xlim([opts.collectStart/60 (opts.collectStart+opts.collectEnd)/60]);

        sgtitle(sprintf('Avalanche Criticality Measures - %s', areas{a}), 'FontSize', 14);

        % Save PNG using exportgraphics
        filename = fullfile(paths.dropPath, sprintf('criticality/avalanche_comparison_%s.png', areas{a}));
        exportgraphics(gcf, filename, 'Resolution', 300);
        fprintf('Saved avalanche area plot to: %s\n', filename);

    end
end

%% ==============================================     Correlation Analysis     ==============================================

if runCorrelation

    % Calculate correlations between dcc, kappa, and decades (dcc sliding window timebase)
    fprintf('\n=== Correlation Analysis: dcc, kappa, and decades ===\n');
    corrMatNat_dccKappaDecades = cell(1, length(areas));
    corrMatRea_dccKappaDecades = cell(1, length(areas));

    for a = areasToTest
        % Naturalistic data
        Xnat = [results.naturalistic.dcc{a}(:), results.naturalistic.kappa{a}(:), results.naturalistic.decades{a}(:)];
        corrMatNat_dccKappaDecades{a} = corr(Xnat, 'Rows', 'pairwise');

        % Reach data
        Xrea = [results.reach.dcc{a}(:), results.reach.kappa{a}(:), results.reach.decades{a}(:)];
        corrMatRea_dccKappaDecades{a} = corr(Xrea, 'Rows', 'pairwise');
    end

    % Set colorbar scale for all correlation matrices
    cmin = -1;
    cmax = 1;

    % Plot correlation matrices for each area
    figure(300); clf;
    set(gcf, 'Position', monitorTwo);
    for a = areasToTest
        % dcc/kappa/decades correlations
        subplot(2, length(areasToTest), a - areasToTest(1) + 1);
        imagesc(corrMatNat_dccKappaDecades{a});
        colorbar;
        title(sprintf('%s (Nat dcc/kappa/decades)', areas{a}));
        xticks(1:3); yticks(1:3);
        xticklabels({'dcc','kappa','decades'}); yticklabels({'dcc','kappa','decades'});
        axis square;
        caxis([cmin cmax]); % Set consistent color axis

        subplot(2, length(areasToTest), length(areasToTest) + (a - areasToTest(1) + 1));
        imagesc(corrMatRea_dccKappaDecades{a});
        colorbar;
        title(sprintf('%s (Rea dcc/kappa/decades)', areas{a}));
        xticks(1:3); yticks(1:3);
        xticklabels({'dcc','kappa','decades'}); yticklabels({'dcc','kappa','decades'});
        axis square;
        caxis([cmin cmax]); % Set consistent color axis
    end
    sgtitle('Correlation Matrices: dcc, kappa, and decades');

    % Save PNG using exportgraphics
    filename = fullfile(paths.dropPath, 'criticality/correlation_matrices_dcc_kappa_decades.png');
    exportgraphics(gcf, filename, 'Resolution', 300);
    fprintf('Saved dcc/kappa/decades correlation plot to: %s\n', filename);
end % runCorrelation
