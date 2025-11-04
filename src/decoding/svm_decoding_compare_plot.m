% Plots data from svm_decoding_compare.m, either from the analyses
% (svm_decoding_compare.m) or by loading specific data
% (svm_decoding_compare_load.m)

% =============================================================================
% --------    PLOT FULL DATA FOR EACH METHOD
% =============================================================================
for areaIdx = areasToTest
    areaName = areas{areaIdx};

    figHFull = 270;
    if plotFullMap
        fprintf('Plotting full data for each method...\n');

        fieldNames = fieldnames(allResults.latents{areaIdx});
        for i = 1:length(fieldNames)
            methodName = fieldNames{i};
            latentData = allResults.latents{areaIdx}.(methodName);

            % Plot full time of all behaviors
            colorsForPlot = arrayfun(@(x) colors(x,:), bhvID + colorsAdjust, 'UniformOutput', false);
            colorsForPlot = vertcat(colorsForPlot{:});

            figure(figHFull + i);
            % Set figure position using monitor dimensions (3/4 height, 1/2 width)
            figWidth = targetMonitor(3) * 0.5;  % Half width
            figHeight = targetMonitor(4) * 0.75; % Three-quarters height
            figX = targetMonitor(1) + (targetMonitor(3) - figWidth) / 2;  % Center horizontally
            figY = targetMonitor(2) + (targetMonitor(4) - figHeight) / 2; % Center vertically
            set(figHFull + i, 'Position', [figX, figY, figWidth, figHeight]);
            clf; hold on;

            if nDim > 2
                scatter3(latentData(:, 1), latentData(:, 2), latentData(:, 3), 20, colorsForPlot, 'filled', 'MarkerFaceAlpha', 0.6);
                % Set view angle to show all three axes with depth
                view(45, 30);
            else
                scatter(latentData(:, 1), latentData(:, 2), 20, colorsForPlot, 'filled', 'MarkerFaceAlpha', 0.6);
            end

            title(sprintf('%s %s %dD - All Behaviors (bin=%.2f)', areaName, upper(methodName), nDim, opts.frameSize), 'Interpreter','none');
            xlabel('D1'); ylabel('D2');
            if nDim > 2
                zlabel('D3');
            end
            grid on;

            % Add legend for behaviors
            uniqueBhv = unique(bhvID);
            uniqueBhv = uniqueBhv(uniqueBhv >= 0);
            for j = 1:length(uniqueBhv)
                bhvIdx = uniqueBhv(j);
                if bhvIdx < length(behaviors)
                    scatter(NaN, NaN, 20, colors(bhvIdx+colorsAdjust,:), 'filled', 'DisplayName', behaviors{bhvIdx+colorsAdjust});
                end
            end
            legend('Location', 'best', 'Interpreter', 'none');

            % Save plot if flag is set
            if savePlotFlag
                plotFilename = sprintf('full_data_%s_%s_%s_%dD_bin%.2f.png', ...
                    dataType, areaName, methodName, nDim, opts.frameSize);
                plotPath = fullfile(savePath, plotFilename);
                exportgraphics(figure(figHFull + i), plotPath, 'Resolution', 300);
                fprintf('Saved plot: %s\n', plotFilename);
            end
        end
    end

    % =============================================================================
    % --------    PLOT MODELING DATA FOR EACH METHOD
    % =============================================================================
    figHModel = 280;
    if plotModelData
        fprintf('Plotting modeling data for each method...\n');

        fieldNames = fieldnames(allResults.latents{areaIdx});
        for i = 1:length(fieldNames)
            methodName = fieldNames{i};
            latentData = allResults.latents{areaIdx}.(methodName);

            % Get colors for modeled behaviors
            colorsForPlot = arrayfun(@(x) bhv2ModelColors(x,:), allResults.svmID{areaIdx} + colorsAdjust, 'UniformOutput', false);
            colorsForPlot = vertcat(colorsForPlot{:});

            figure(figHModel + i);
            % Set figure position using monitor dimensions (3/4 height, 1/2 width)
            figWidth = targetMonitor(3) * 0.5;  % Half width
            figHeight = targetMonitor(4) * 0.75; % Three-quarters height
            figX = targetMonitor(1) + (targetMonitor(3) - figWidth) / 2;  % Center horizontally
            figY = targetMonitor(2) + (targetMonitor(4) - figHeight) / 2; % Center vertically
            set(figHModel + i, 'Position', [figX, figY, figWidth, figHeight]);
            clf; hold on;

 if nDim > 2
    scatter3(latentData(allResults.svmInd{areaIdx}, 1), latentData(allResults.svmInd{areaIdx}, 2), latentData(allResults.svmInd{areaIdx}, 3), 40, colorsForPlot, 'filled', 'HandleVisibility', 'off');
    % Set view angle to show all three axes with depth
    view(45, 30);
else
    scatter(latentData(allResults.svmInd{areaIdx}, 1), latentData(allResults.svmInd{areaIdx}, 2), 40, colorsForPlot, 'filled', 'HandleVisibility', 'off');
end
            title(sprintf('%s %s %dD - %s (bin=%.2f)', areaName, upper(methodName), nDim, dataSubsetLabel, opts.frameSize), 'Interpreter','none');
            xlabel('D1'); ylabel('D2');
            if nDim > 2
                zlabel('D3');
            end
            grid on;

            % Add legend for modeled behaviors
            for j = 1:length(allResults.bhv2ModelCodes{areaIdx})
                bhvIdx = allResults.bhv2ModelCodes{areaIdx}(j);
                % if bhvIdx < length(behaviors)
                    scatter(NaN, NaN, 40, colors(bhvIdx+colorsAdjust,:), 'filled', 'DisplayName', behaviors{bhvIdx+colorsAdjust});
                % end
            end
            legend('Location', 'best');

            % Save plot if flag is set
            if savePlotFlag
                plotFilename = sprintf('modeling_%s_%s_%s_%s_%dD_bin%.2f.png', ...
                    dataType, areaName, methodName, dataSubset, nDim, opts.frameSize);
                plotPath = fullfile(savePath, plotFilename);
                exportgraphics(figure(figHModel + i), plotPath, 'Resolution', 300);
                fprintf('Saved plot: %s\n', plotFilename);
            end
        end
    end
end

% =============================================================================
% --------    PLOT RESULTS COMPARISON FOR ALL AREAS
% =============================================================================

if plotComparisons
    fprintf('Plotting results comparison for all areas...\n');

    % Create comparison figure for all areas
    fig = figure(112);
    set(fig, 'Position', [targetMonitor(1), targetMonitor(2), targetMonitor(3), targetMonitor(4)]);
    clf;

    % Plot all areas regardless of which ones are being analyzed
    allAreasToPlot = 1:4;  % M23, M56, DS, VS
    nAreas = length(allAreasToPlot);

    % Find the first area that has data to determine number of methods
    nMethods = 0;
    for checkArea = allAreasToPlot
        if ~isempty(allResults.methods{checkArea})
            nMethods = length(allResults.methods{checkArea});
            break;
        end
    end

    if nMethods == 0
        fprintf('No data found for any area. Skipping plot.\n');
        % continue;
    end

    % Create subplots for each area
    for a = 1:nAreas
        areaIdx = allAreasToPlot(a);
        areaName = areas{areaIdx};

        subplot(1, nAreas, a);
        hold on;

        % Check if this area has data
        if isempty(allResults.methods{areaIdx}) || isempty(allResults.accuracy{areaIdx})
            % No data for this area - show empty plot
            text(0.5, 0.5, 'No Data', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', ...
                'FontSize', 14, 'Color', 'red');
            title(sprintf('%s (%s) - No Data', areaName, dataSubsetLabel), 'Interpreter','none');
            ylim([0, 1]);
            xlim([0, 1]);
            continue;
        end

        % Get data for this area
        accuracy = allResults.accuracy{areaIdx};
        accuracyPermuted = allResults.accuracyPermuted{areaIdx};
        methods = allResults.methods{areaIdx};

        % Use the actual number of methods for this area
        areaMethods = length(methods);

        % Bar plot of accuracies
        x = 1:areaMethods;
        barWidth = 0.35;

        % Real data bars
        b1 = bar(x, accuracy(:), barWidth, 'FaceColor', 'blue', 'DisplayName', 'Real Data');

        % Permuted data bars
        b2 = bar(x + barWidth/2, mean(accuracyPermuted, 2), barWidth, 'FaceColor', 'red', 'DisplayName', 'Permuted (mean)');

        % Add error bars for permuted data
        errorbar(x + barWidth/2, mean(accuracyPermuted, 2), std(accuracyPermuted, [], 2), 'k.', 'LineWidth', 1.5, 'DisplayName', 'Permuted (std)');

        % Customize plot
        set(gca, 'XTick', x, 'XTickLabel', upper(methods), 'TickLabelInterpreter', 'none');
        ylabel('Accuracy');
        title(sprintf('%s (%s)', areaName, dataSubsetLabel), 'Interpreter','none');
        legend('Location', 'best');
        grid on;
        ylim([0, 1]);

        % Add significance indicators
        for m = 1:areaMethods
            % Simple significance test: if real accuracy > 95th percentile of permuted
            permSorted = sort(accuracyPermuted(m, :));
            threshold95 = permSorted(ceil(0.95 * nShuffles));

            if accuracy(m) > threshold95
                text(m, accuracy(m) + 0.02, '*', 'HorizontalAlignment', 'center', 'FontSize', 16, 'FontWeight', 'bold');
            end
        end

        % Add accuracy values as text
        for m = 1:areaMethods
            text(m - barWidth/2, accuracy(m) + 0.01, sprintf('%.3f', accuracy(m)), ...
                'HorizontalAlignment', 'center', 'FontSize', 10);
            text(m + barWidth/2, mean(accuracyPermuted(m, :)) + 0.01, sprintf('%.3f', mean(accuracyPermuted(m, :))), ...
                'HorizontalAlignment', 'center', 'FontSize', 10);
        end
    end

    sgtitle('SVM Decoding Accuracy Comparison - All Areas', 'FontSize', 16);

    % Save comparison plot if flag is set
    if savePlotFlag
        plotFilename = sprintf('accuracy_svm_%s_%s_%s_%dD_bin%.2f_nShuffles%d.png', ...
            dataType, kernelFunction, dataSubset, nDim, opts.frameSize, nShuffles);
        plotPath = fullfile(savePath, plotFilename);
        exportgraphics(fig, plotPath, 'Resolution', 300);
        fprintf('Saved comparison plot: %s\n', plotFilename);
    end

    % Summary statistics for all areas
    fprintf('\n=== SUMMARY FOR ALL AREAS ===\n');
    fprintf('Area\t\tMethod\t\tReal\t\tPermuted (mean ± std)\t\tDifference\n');
    fprintf('----\t\t-----\t\t----\t\t----------------------\t\t----------\n');

    for a = 1:nAreas
        areaIdx = allAreasToPlot(a);
        areaName = areas{areaIdx};

        % Check if this area has data
        if isempty(allResults.methods{areaIdx}) || isempty(allResults.accuracy{areaIdx})
            fprintf('%s\t\tNo Data Available\n', areaName);
            continue;
        end

        accuracy = allResults.accuracy{areaIdx};
        accuracyPermuted = allResults.accuracyPermuted{areaIdx};
        methods = allResults.methods{areaIdx};

        for m = 1:length(methods)
            diffAcc = accuracy(m) - mean(accuracyPermuted(m, :));
            fprintf('%s\t\t%s\t\t%.3f\t\t%.3f ± %.3f\t\t%.3f\n', ...
                areaName, upper(methods{m}), accuracy(m), mean(accuracyPermuted(m, :)), ...
                std(accuracyPermuted(m, :)), diffAcc);
        end

        % Find best method for this area
        [bestAcc, bestIdx] = max(accuracy);
        fprintf('Best method for %s: %s (accuracy: %.3f)\n', areaName, upper(methods{bestIdx}), bestAcc);
        fprintf('\n');
    end
end
