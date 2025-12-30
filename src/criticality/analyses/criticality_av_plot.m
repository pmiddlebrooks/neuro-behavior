function criticality_av_plot(results, plotConfig, config, dataStruct, filenameSuffix)
% criticality_av_plot - Create plots for criticality avalanche analysis
%
% Variables:
%   results - Results structure from criticality_av_analysis()
%   plotConfig - Plotting configuration from setup_plotting()
%   config - Configuration structure
%   dataStruct - Data structure from load_sliding_window_data()
%   filenameSuffix - Filename suffix (e.g., '_pca')
%
% Goal:
%   Create time series plots of avalanche metrics (dcc, tau, alpha, paramSD, decades)
%   with optional permutation shading.

    % Extract data from results
    areas = results.areas;
    dcc = results.dcc;
    kappa = results.kappa;
    decades = results.decades;
    tau = results.tau;
    alpha = results.alpha;
    paramSD = results.paramSD;
    startS = results.startS;
    
    % Get areas to plot
    if isfield(dataStruct, 'areasToTest')
        areasToTest = dataStruct.areasToTest;
    else
        areasToTest = 1:length(areas);
    end
    
    dataType = results.dataType;
    slidingWindowSize = results.params.slidingWindowSize;
    avStepSize = results.params.avStepSize;
    if isfield(results, 'enablePermutations')
        enablePermutations = results.enablePermutations;
    else
        enablePermutations = false;
    end
    
    % Collect data for axis limits
    allStartS = [];
    allDcc = [];
    allTau = [];
    allAlpha = [];
    allParamSD = [];
    allDecades = [];
    
    for a = areasToTest
        if ~isempty(startS{a})
            allStartS = [allStartS, startS{a}];
        end
        if ~isempty(dcc{a})
            allDcc = [allDcc, dcc{a}(~isnan(dcc{a}))];
        end
        if ~isempty(tau{a})
            allTau = [allTau, tau{a}(~isnan(tau{a}))];
        end
        if ~isempty(alpha{a})
            allAlpha = [allAlpha, alpha{a}(~isnan(alpha{a}))];
        end
        if ~isempty(paramSD{a})
            allParamSD = [allParamSD, paramSD{a}(~isnan(paramSD{a}))];
        end
        if ~isempty(decades{a})
            allDecades = [allDecades, decades{a}(~isnan(decades{a}))];
        end
        
        % Include permuted data in axis limits if available
        if enablePermutations
            % dcc permuted data
            if isfield(results, 'dccPermuted') && ~isempty(results.dccPermuted{a})
                permutedMean = nanmean(results.dccPermuted{a}, 2);
                permutedStd = nanstd(results.dccPermuted{a}, 0, 2);
                validIdx = ~isnan(permutedMean) & ~isnan(permutedStd);
                if any(validIdx)
                    permutedVals = [permutedMean(validIdx) + permutedStd(validIdx); ...
                        permutedMean(validIdx) - permutedStd(validIdx)];
                    allDcc = [allDcc(:); permutedVals(:)];
                end
            end
            
            % tau permuted data
            if isfield(results, 'tauPermuted') && ~isempty(results.tauPermuted{a})
                permutedMean = nanmean(results.tauPermuted{a}, 2);
                permutedStd = nanstd(results.tauPermuted{a}, 0, 2);
                validIdx = ~isnan(permutedMean) & ~isnan(permutedStd);
                if any(validIdx)
                    permutedVals = [permutedMean(validIdx) + permutedStd(validIdx); ...
                        permutedMean(validIdx) - permutedStd(validIdx)];
                    allTau = [allTau(:); permutedVals(:)];
                end
            end
            
            % alpha permuted data
            if isfield(results, 'alphaPermuted') && ~isempty(results.alphaPermuted{a})
                permutedMean = nanmean(results.alphaPermuted{a}, 2);
                permutedStd = nanstd(results.alphaPermuted{a}, 0, 2);
                validIdx = ~isnan(permutedMean) & ~isnan(permutedStd);
                if any(validIdx)
                    permutedVals = [permutedMean(validIdx) + permutedStd(validIdx); ...
                        permutedMean(validIdx) - permutedStd(validIdx)];
                    allAlpha = [allAlpha(:); permutedVals(:)];
                end
            end
            
            % paramSD permuted data
            if isfield(results, 'paramSDPermuted') && ~isempty(results.paramSDPermuted{a})
                permutedMean = nanmean(results.paramSDPermuted{a}, 2);
                permutedStd = nanstd(results.paramSDPermuted{a}, 0, 2);
                validIdx = ~isnan(permutedMean) & ~isnan(permutedStd);
                if any(validIdx)
                    permutedVals = [permutedMean(validIdx) + permutedStd(validIdx); ...
                        permutedMean(validIdx) - permutedStd(validIdx)];
                    allParamSD = [allParamSD(:); permutedVals(:)];
                end
            end
            
            % decades permuted data
            if isfield(results, 'decadesPermuted') && ~isempty(results.decadesPermuted{a})
                permutedMean = nanmean(results.decadesPermuted{a}, 2);
                permutedStd = nanstd(results.decadesPermuted{a}, 0, 2);
                validIdx = ~isnan(permutedMean) & ~isnan(permutedStd);
                if any(validIdx)
                    permutedVals = [permutedMean(validIdx) + permutedStd(validIdx); ...
                        permutedMean(validIdx) - permutedStd(validIdx)];
                    allDecades = [allDecades(:); permutedVals(:)];
                end
            end
        end
    end
    
    % Determine axis limits
    if ~isempty(allStartS)
        xMin = min(allStartS);
        xMax = max(allStartS);
    else
        xMin = 0;
        xMax = 1;
    end
    
    % Set dcc y-limit max
    dccMaxPlot = 1;
    if ~isempty(allDcc)
        yMinDcc = 0;
        yMaxDcc = min(max(allDcc), dccMaxPlot);
        % Add small padding
        yRangeDcc = yMaxDcc - yMinDcc;
        yMaxDcc = min(yMaxDcc + 0.05 * yRangeDcc, dccMaxPlot);
    else
        yMinDcc = 0;
        yMaxDcc = dccMaxPlot;
    end
    
    % Determine y-limits for tau, alpha, paramSD (all on same plot)
    allParams = [allTau, allAlpha, allParamSD];
    if ~isempty(allParams)
        yMinParams = min(allParams(:));
        yMaxParams = max(allParams(:));
        % Add small padding
        yRangeParams = yMaxParams - yMinParams;
        yMinParams = yMinParams - 0.05 * yRangeParams;
        yMaxParams = yMaxParams + 0.05 * yRangeParams;
    else
        yMinParams = 0;
        yMaxParams = 1;
    end
    
    if ~isempty(allDecades)
        yMinDecades = min(allDecades);
        yMaxDecades = max(allDecades);
        % Add small padding
        yRangeDecades = yMaxDecades - yMinDecades;
        yMinDecades = yMinDecades - 0.05 * yRangeDecades;
        yMaxDecades = yMaxDecades + 0.05 * yRangeDecades;
    else
        yMinDecades = 0;
        yMaxDecades = 1;
    end
    
    % Create figure
    figure(903); clf;
    set(gcf, 'Units', 'pixels');
    set(gcf, 'Position', plotConfig.targetPos);
    
    % Avalanche analysis plots: 3 rows (dcc, tau/alpha/paramSD, decades) x num areas (columns)
    for idx = 1:length(areasToTest)
        a = areasToTest(idx);
        
        % dcc (top row)
        subplot(3, length(areasToTest), idx);
        hold on;
        if ~isempty(dcc{a}) && ~isempty(startS{a})
            validIdx = ~isnan(dcc{a});
            if any(validIdx)
                plot(startS{a}(validIdx), dcc{a}(validIdx), '-', 'Color', [1 0 0], 'LineWidth', 2);
            end
        end
        
        % Plot permutation mean ± std per window if available
        if enablePermutations && isfield(results, 'dccPermuted') && ~isempty(results.dccPermuted{a})
            % Calculate mean and std for each window across shuffles
            permutedMean = nanmean(results.dccPermuted{a}, 2);
            permutedStd = nanstd(results.dccPermuted{a}, 0, 2);
            
            % Find valid indices
            validIdx = ~isnan(permutedMean) & ~isnan(permutedStd) & ~isnan(startS{a}(:));
            if any(validIdx)
                xFill = startS{a}(validIdx);
                yMean = permutedMean(validIdx);
                yStd = permutedStd(validIdx);
                
                % Ensure row vectors for fill
                if iscolumn(xFill); xFill = xFill'; end
                if iscolumn(yMean); yMean = yMean'; end
                if iscolumn(yStd); yStd = yStd'; end
                
                % Plot shaded region (mean ± std)
                fill([xFill, fliplr(xFill)], ...
                     [yMean + yStd, fliplr(yMean - yStd)], ...
                     [0.7 0.7 1], 'FaceAlpha', 0.3, 'EdgeColor', 'none', ...
                     'DisplayName', 'Permuted mean ± std');
                
                % Plot mean line
                plot(startS{a}(validIdx), permutedMean(validIdx), '-', ...
                    'Color', [0.5 0.5 1], 'LineWidth', 1.5, 'LineStyle', '--', ...
                    'DisplayName', 'Permuted mean');
            end
        end
        
        title(sprintf('%s - dcc', areas{a}));
        xlabel('Time (s)');
        ylabel('dcc');
        grid on;
        xlim([xMin, xMax]);
        ylim([yMinDcc, dccMaxPlot]);
        set(gca, 'XTickLabelMode', 'auto');
        set(gca, 'YTickLabelMode', 'auto');
        
        % tau, alpha, paramSD (middle row) - all on same subplot
        subplot(3, length(areasToTest), length(areasToTest) + idx);
        hold on;
        
        % Plot tau
        if ~isempty(tau{a}) && ~isempty(startS{a})
            validIdx = ~isnan(tau{a});
            if any(validIdx)
                plot(startS{a}(validIdx), tau{a}(validIdx), '-', ...
                    'Color', [1 0.5 0], 'LineWidth', 2, 'DisplayName', 'tau');
            end
        end
        
        % Plot alpha
        if ~isempty(alpha{a}) && ~isempty(startS{a})
            validIdx = ~isnan(alpha{a});
            if any(validIdx)
                plot(startS{a}(validIdx), alpha{a}(validIdx), '-', ...
                    'Color', [0 0.8 0], 'LineWidth', 2, 'DisplayName', 'alpha');
            end
        end
        
        % Plot paramSD
        if ~isempty(paramSD{a}) && ~isempty(startS{a})
            validIdx = ~isnan(paramSD{a});
            if any(validIdx)
                plot(startS{a}(validIdx), paramSD{a}(validIdx), '-', ...
                    'Color', [0 0 1], 'LineWidth', 2, 'DisplayName', 'paramSD');
            end
        end
        
        % Plot permutation mean ± std per window for tau if available
        if enablePermutations && isfield(results, 'tauPermuted') && ~isempty(results.tauPermuted{a})
            permutedMean = nanmean(results.tauPermuted{a}, 2);
            permutedStd = nanstd(results.tauPermuted{a}, 0, 2);
            validIdx = ~isnan(permutedMean) & ~isnan(permutedStd) & ~isnan(startS{a}(:));
            if any(validIdx)
                xFill = startS{a}(validIdx);
                yMean = permutedMean(validIdx);
                yStd = permutedStd(validIdx);
                if iscolumn(xFill); xFill = xFill'; end
                if iscolumn(yMean); yMean = yMean'; end
                if iscolumn(yStd); yStd = yStd'; end
                fill([xFill, fliplr(xFill)], ...
                     [yMean + yStd, fliplr(yMean - yStd)], ...
                     [1 0.8 0.6], 'FaceAlpha', 0.2, 'EdgeColor', 'none', ...
                     'DisplayName', 'tau Permuted mean ± std');
                plot(startS{a}(validIdx), permutedMean(validIdx), '-', ...
                    'Color', [1 0.7 0.3], 'LineWidth', 1, 'LineStyle', '--', ...
                    'DisplayName', 'tau Permuted mean');
            end
        end
        
        % Plot permutation mean ± std per window for alpha if available
        if enablePermutations && isfield(results, 'alphaPermuted') && ~isempty(results.alphaPermuted{a})
            permutedMean = nanmean(results.alphaPermuted{a}, 2);
            permutedStd = nanstd(results.alphaPermuted{a}, 0, 2);
            validIdx = ~isnan(permutedMean) & ~isnan(permutedStd) & ~isnan(startS{a}(:));
            if any(validIdx)
                xFill = startS{a}(validIdx);
                yMean = permutedMean(validIdx);
                yStd = permutedStd(validIdx);
                if iscolumn(xFill); xFill = xFill'; end
                if iscolumn(yMean); yMean = yMean'; end
                if iscolumn(yStd); yStd = yStd'; end
                fill([xFill, fliplr(xFill)], ...
                     [yMean + yStd, fliplr(yMean - yStd)], ...
                     [0.6 0.9 0.6], 'FaceAlpha', 0.2, 'EdgeColor', 'none', ...
                     'DisplayName', 'alpha Permuted mean ± std');
                plot(startS{a}(validIdx), permutedMean(validIdx), '-', ...
                    'Color', [0.3 0.7 0.3], 'LineWidth', 1, 'LineStyle', '--', ...
                    'DisplayName', 'alpha Permuted mean');
            end
        end
        
        % Plot permutation mean ± std per window for paramSD if available
        if enablePermutations && isfield(results, 'paramSDPermuted') && ~isempty(results.paramSDPermuted{a})
            permutedMean = nanmean(results.paramSDPermuted{a}, 2);
            permutedStd = nanstd(results.paramSDPermuted{a}, 0, 2);
            validIdx = ~isnan(permutedMean) & ~isnan(permutedStd) & ~isnan(startS{a}(:));
            if any(validIdx)
                xFill = startS{a}(validIdx);
                yMean = permutedMean(validIdx);
                yStd = permutedStd(validIdx);
                if iscolumn(xFill); xFill = xFill'; end
                if iscolumn(yMean); yMean = yMean'; end
                if iscolumn(yStd); yStd = yStd'; end
                fill([xFill, fliplr(xFill)], ...
                     [yMean + yStd, fliplr(yMean - yStd)], ...
                     [0.6 0.6 0.9], 'FaceAlpha', 0.2, 'EdgeColor', 'none', ...
                     'DisplayName', 'paramSD Permuted mean ± std');
                plot(startS{a}(validIdx), permutedMean(validIdx), '-', ...
                    'Color', [0.3 0.3 0.7], 'LineWidth', 1, 'LineStyle', '--', ...
                    'DisplayName', 'paramSD Permuted mean');
            end
        end
        
        title(sprintf('%s - tau (orange), alpha (green), paramSD (blue)', areas{a}));
        xlabel('Time (s)');
        ylabel('Value');
        grid on;
        xlim([xMin, xMax]);
        ylim([yMinParams, yMaxParams]);
        legend('Location', 'best');
        set(gca, 'XTickLabelMode', 'auto');
        set(gca, 'YTickLabelMode', 'auto');
        
        % decades (bottom row)
        subplot(3, length(areasToTest), 2*length(areasToTest) + idx);
        hold on;
        if ~isempty(decades{a}) && ~isempty(startS{a})
            validIdx = ~isnan(decades{a});
            if any(validIdx)
                plot(startS{a}(validIdx), decades{a}(validIdx), '-', ...
                    'Color', [0.6 0 0.6], 'LineWidth', 2);
            end
        end
        
        % Plot permutation mean ± std per window if available
        if enablePermutations && isfield(results, 'decadesPermuted') && ~isempty(results.decadesPermuted{a})
            % Calculate mean and std for each window across shuffles
            permutedMean = nanmean(results.decadesPermuted{a}, 2);
            permutedStd = nanstd(results.decadesPermuted{a}, 0, 2);
            
            % Find valid indices
            validIdx = ~isnan(permutedMean) & ~isnan(permutedStd) & ~isnan(startS{a}(:));
            if any(validIdx)
                xFill = startS{a}(validIdx);
                yMean = permutedMean(validIdx);
                yStd = permutedStd(validIdx);
                
                % Ensure row vectors for fill
                if iscolumn(xFill); xFill = xFill'; end
                if iscolumn(yMean); yMean = yMean'; end
                if iscolumn(yStd); yStd = yStd'; end
                
                % Plot shaded region (mean ± std)
                fill([xFill, fliplr(xFill)], ...
                     [yMean + yStd, fliplr(yMean - yStd)], ...
                     [0.7 0.7 1], 'FaceAlpha', 0.3, 'EdgeColor', 'none', ...
                     'DisplayName', 'Permuted mean ± std');
                
                % Plot mean line
                plot(startS{a}(validIdx), permutedMean(validIdx), '-', ...
                    'Color', [0.5 0.5 1], 'LineWidth', 1.5, 'LineStyle', '--', ...
                    'DisplayName', 'Permuted mean');
            end
        end
        
        title(sprintf('%s - decades', areas{a}));
        xlabel('Time (s)');
        ylabel('decades');
        grid on;
        xlim([xMin, xMax]);
        ylim([yMinDecades, yMaxDecades]);
        set(gca, 'XTickLabelMode', 'auto');
        set(gca, 'YTickLabelMode', 'auto');
    end
    
    % Create super title
    if ~isempty(plotConfig.filePrefix)
        sgtitle(sprintf('[%s] %s dcc (top), tau/alpha/paramSD (mid), decades (bottom) - win=%gs, step=%gs', ...
            plotConfig.filePrefix, dataType, slidingWindowSize, avStepSize), 'interpreter', 'none');
    else
        sgtitle(sprintf('%s dcc (top), tau/alpha/paramSD (mid), decades (bottom) - win=%gs, step=%gs', ...
            dataType, slidingWindowSize, avStepSize), 'interpreter', 'none');
    end
    
    % Ensure save directory exists (including any subdirectories)
    if ~exist(config.saveDir, 'dir')
        mkdir(config.saveDir);
    end
    
    % Save figure
    if ~isempty(plotConfig.filePrefix)
        plotPath = fullfile(config.saveDir, ...
            sprintf('%s_criticality_%s_av%s_win%d_step%d.png', ...
            plotConfig.filePrefix, dataType, filenameSuffix, slidingWindowSize, avStepSize));
        exportgraphics(gcf, plotPath, 'Resolution', 300);
        fprintf('Saved plot to: %s\n', plotPath);
    else
        plotPath = fullfile(config.saveDir, ...
            sprintf('criticality_%s_av%s_win%d_step%d.png', ...
            dataType, filenameSuffix, slidingWindowSize, avStepSize));
        exportgraphics(gcf, plotPath, 'Resolution', 300);
        fprintf('Saved plot to: %s\n', plotPath);
    end
    
    fprintf('Saved criticality AV plot to: %s\n', config.saveDir);
end
