% Plot SVM decoding results from svm_decoding_compare.m or
% svm_decoding_compare_joint_area.m — run after analysis or via
% svm_decoding_compare_load.m (supports per-area and conjoint bundles).

%%
if ~exist('paths', 'var') || isempty(paths)
    paths = get_paths;
end

if ~exist('allResults', 'var')
    error('Missing allResults. Run svm_decoding_compare_load or keep results in workspace.');
end

layoutJoint = isfield(allResults, 'jointLatents');
if layoutJoint && isfield(allResults, 'parameters')
    plotDimPerArea = allResults.parameters.nDim;
    jointFeatDim = allResults.parameters.jointFeatureDim;
    nComponentsJoint = allResults.parameters.nComponentsPerArea;
elseif exist('nDim', 'var')
    plotDimPerArea = nDim;
    jointFeatDim = [];
    nComponentsJoint = [];
else
    plotDimPerArea = 8;
    jointFeatDim = [];
    nComponentsJoint = [];
end

if ~exist('jointLabelStr', 'var')
    jointLabelStr = ''; %#ok<NASGU>
end
if layoutJoint && isempty(strtrim(char(jointLabelStr)))
    if isfield(allResults, 'areasToInclude')
        jointLabelStr = strjoin(allResults.areas(allResults.areasToInclude(:)'), '+');
    end
end

if layoutJoint && isfield(allResults, 'areasToInclude')
    areasPlotList = allResults.areasToInclude(:)';
elseif ~exist('areasToTest', 'var') || isempty(areasToTest)
    if isfield(allResults, 'areasToTest')
        areasPlotList = allResults.areasToTest(:)';
    else
        areasPlotList = 1:4;
    end
else
    areasPlotList = areasToTest(:)';
end

if layoutJoint && isfield(allResults, 'svmID')
    svmID_vec = allResults.svmID(:);
    svmInd_vec = allResults.svmInd(:);
else
    svmID_vec = [];
    svmInd_vec = [];
end

if ~exist('nShuffles', 'var') || isempty(nShuffles)
    if layoutJoint && isfield(allResults, 'accuracyPermuted')
        nShuffles = size(allResults.accuracyPermuted, 2);
    elseif isfield(allResults, 'accuracyPermuted')
        cp = allResults.accuracyPermuted(~cellfun(@isempty, allResults.accuracyPermuted));
        if ~isempty(cp)
            nShuffles = size(cp{1}, 2);
        else
            nShuffles = 1;
        end
    else
        nShuffles = 1;
    end
end

%%
savePath = fullfile(paths.dropPath, 'sfn2025/decoding');
if ~exist(savePath, 'dir')
    mkdir(savePath);
end

% Monitor setup for plotting
monitorPositions = get(0, 'MonitorPositions');
monitorOne = monitorPositions(1, :);
monitorTwo = monitorPositions(size(monitorPositions, 1), :);

% Choose target monitor (use second/last monitor if connected)
if size(monitorPositions, 1) > 1
    targetMonitor = monitorTwo;
else
    targetMonitor = monitorOne;
end

%%
% set(0, 'DefaultTextColor', '#efdb7b');
% set(0, 'DefaultAxesXColor', '#efdb7b');
% set(0, 'DefaultAxesYColor', '#efdb7b');
% set(0, 'DefaultAxesZColor', '#efdb7b');
set(0, 'DefaultTextColor', '#D5A570');
set(0, 'DefaultAxesXColor', '#D5A570');
set(0, 'DefaultAxesYColor', '#D5A570');
set(0, 'DefaultAxesZColor', '#D5A570');
set(0, 'DefaultFigureColor', 'remove');
set(0, 'DefaultAxesColor', 'remove');
set(0, 'DefaultAxesFontSize', 24);
% set(0, 'DefaultAxesTickLength', [0.02 0.04]); % Default tick mark length
set(0, 'DefaultAxesLineWidth', 3); % Default tick mark linewidth = 4

%% Reset — defaults aligned with svm_decoding_compare_load
if ~exist('plotFullMap', 'var'); plotFullMap = false; end %#ok<*NEXIST>
if ~exist('plotModelData', 'var'); plotModelData = true; end
if ~exist('plotComparisons', 'var'); plotComparisons = true; end
if ~exist('savePlotFlag', 'var'); savePlotFlag = true; end
if ~exist('dataType', 'var'); dataType = 'spontaneous'; end
if ~exist('kernelFunction', 'var'); kernelFunction = 'polynomial'; end
if ~exist('dataSubset', 'var'); dataSubset = 'all'; end %#ok<*NODEF>
if ~exist('dataSubsetLabel', 'var'); dataSubsetLabel = 'all'; end
if ~exist('opts', 'var') || ~isfield(opts, 'frameSize')
    opts = struct('frameSize', 0.1);
end

if ~exist('colors', 'var') || isempty(colors)
    colors = lines(max(40, plotDimPerArea + 16));
end
if ~exist('colorsAdjust', 'var'); colorsAdjust = 0; end
if ~exist('bhv2ModelColors', 'var') || isempty(bhv2ModelColors)
    bhv2ModelColors = colors;
end
if ~exist('behaviors', 'var')
    error('Missing behaviors colors context — run svm_decoding_compare_load or parent analysis first.');
end

if plotFullMap && (~exist('bhvID', 'var') || isempty(bhvID))
    warning('svm_decoding_compare_plot:MissingBhvID', 'bhvID not in workspace; disabling plotFullMap.');
    plotFullMap = false;
end

%% =============================================================================
% --------    PLOT FULL DATA FOR EACH METHOD
% =============================================================================
for areaIdx = areasPlotList
    areaName = areas{areaIdx};

    figHFull = 270;
    if isempty(allResults.latents{areaIdx})
        fprintf('Skipping area %s: no stored latents.\n', areaName);
        continue;
    end

    if plotFullMap
        fprintf('Plotting full data for each method...\n');

        fieldNames = fieldnames(allResults.latents{areaIdx});
        for i = 1:length(fieldNames)
            methodName = fieldNames{i};
            latentData = allResults.latents{areaIdx}.(methodName);

            % Plot full time of all behaviors
            colorsForPlot = arrayfun(@(rowIdx) colors(rowIdx, :), bhvID + colorsAdjust, 'UniformOutput', false);
            colorsForPlot = vertcat(colorsForPlot{:});

            figHd = figure(figHFull + i);
            figWidth = targetMonitor(3) * 0.5;  % Half width
            figHeight = targetMonitor(4) * 0.75; % Three-quarters height
            figX = targetMonitor(1) + (targetMonitor(3) - figWidth) / 2;  % Center horizontally
            figY = targetMonitor(2) + (targetMonitor(4) - figHeight) / 2; % Center vertically
            set(figHd, 'Position', [figX, figY, figWidth, figHeight]);
            clf; hold on;

            nColsLat = size(latentData, 2);
            if nColsLat >= 3
                scatter3(latentData(:, 1), latentData(:, 2), latentData(:, 3), 20, colorsForPlot, 'filled', 'MarkerFaceAlpha', 0.6);
                view(45, 30);
                zlabel('D3');
            elseif nColsLat >= 2
                scatter(latentData(:, 1), latentData(:, 2), 20, colorsForPlot, 'filled', 'MarkerFaceAlpha', 0.6);
            else
                clf;
                fprintf('Skipping %s %s: latent has one column.\n', areaName, methodName);
                continue;
            end

            ttlExtra = '';
            if layoutJoint && ~isempty(nComponentsJoint)
                ttlExtra = sprintf(' [%d/embed; concat %d', plotDimPerArea, nComponentsJoint);
                ttlExtra = strcat(ttlExtra, '] ');
            end
            title(sprintf('%s %s%s- All behaviors (embed %dD, bin=%.2f)', ...
                areaName, upper(methodName), ttlExtra, plotDimPerArea, opts.frameSize), 'Interpreter', 'none');
            xlabel('D1'); ylabel('D2');
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
                    dataType, areaName, methodName, plotDimPerArea, opts.frameSize);
                plotPath = fullfile(savePath, plotFilename);
                print(figHd, '-depsc', '-vector', plotPath);
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
        if layoutJoint
            svmLabs = svmID_vec;
            svmPick = svmInd_vec;
            bhvLegCodes = allResults.bhv2ModelCodes;
        else
            svmLabs = allResults.svmID{areaIdx};
            svmPick = allResults.svmInd{areaIdx};
            bhvLegCodes = allResults.bhv2ModelCodes{areaIdx};
        end

        for i = 1:length(fieldNames)
            methodName = fieldNames{i};
            latentData = allResults.latents{areaIdx}.(methodName);

            colorsForPlot = arrayfun(@(code) bhv2ModelColors(code + colorsAdjust, :), svmLabs, 'UniformOutput', false);
            colorsForPlot = vertcat(colorsForPlot{:});
            samp = latentData(svmPick(:), :);
            nColsLat = size(samp, 2);

            figMod = figure(figHModel + i);
            figWidth = targetMonitor(3) * 0.5;  % Half width
            figHeight = targetMonitor(4) * 0.75; % Three-quarters height
            figX = targetMonitor(1) + (targetMonitor(3) - figWidth) / 2;
            figY = targetMonitor(2) + (targetMonitor(4) - figHeight) / 2;
            set(figMod, 'Position', [figX, figY, figWidth, figHeight]);
            clf; hold on;

            if nColsLat >= 3
                scatter3(samp(:, 1), samp(:, 2), samp(:, 3), 40, colorsForPlot, 'filled', 'HandleVisibility', 'off');
                view(45, 30);
                zlabel('D3');
            elseif nColsLat >= 2
                scatter(samp(:, 1), samp(:, 2), 40, colorsForPlot, 'filled', 'HandleVisibility', 'off');
            else
                clf;
                fprintf('Skipping modeling %s %s: latent width < 2.\n', areaName, methodName);
                continue;
            end

            ttlJoint = '';
            if layoutJoint && ~isempty(jointFeatDim)
                ttlJoint = sprintf(' | concat → joint %dD', jointFeatDim);
            end
            title(sprintf('%s %s %dD%s - %s (bin=%.2f)', areaName, upper(methodName), plotDimPerArea, ...
                ttlJoint, dataSubsetLabel, opts.frameSize), 'Interpreter', 'none');
            xlabel('D1'); ylabel('D2');
            grid on;

            for j = 1:length(bhvLegCodes)
                bhvIdx = bhvLegCodes(j);
                scatter(NaN, NaN, 40, colors(bhvIdx+colorsAdjust, :), 'filled', ...
                    'DisplayName', behaviors{bhvIdx+colorsAdjust});
            end
            legend('Location', 'best');

            if savePlotFlag
                plotFilename = sprintf('modeling_%s_%s_%s_%s_%dD_bin%.2f.png', ...
                    dataType, areaName, methodName, dataSubset, plotDimPerArea, opts.frameSize);
                plotPath = fullfile(savePath, plotFilename);
                print(figMod, '-depsc', '-vector', plotPath);
                fprintf('Saved plot: %s\n', plotFilename);
            end
        end
    end
end


%% Joint concatenated latent space (SVM inputs) — conjoint bundles only

figHJoint = 470;
if layoutJoint && plotModelData && isfield(allResults, 'jointLatents')
    svmLabsJ = svmID_vec;
    svmPickJ = svmInd_vec;
    jFields = fieldnames(allResults.jointLatents);
    for ij = 1:length(jFields)
        methodNameJ = jFields{ij};
        latentJoint = allResults.jointLatents.(methodNameJ);
        if isempty(latentJoint)
            continue;
        end
        sampJ = latentJoint(svmPickJ(:), :);
        colorsJ = arrayfun(@(code) bhv2ModelColors(code + colorsAdjust, :), svmLabsJ, 'UniformOutput', false);
        colorsJ = vertcat(colorsJ{:});

        fj = figure(figHJoint + ij);
        figWidth = targetMonitor(3) * 0.55;
        figHeight = targetMonitor(4) * 0.75;
        figX = targetMonitor(1) + (targetMonitor(3) - figWidth) / 2;
        figY = targetMonitor(2) + (targetMonitor(4) - figHeight) / 2;
        set(fj, 'Position', [figX, figY, figWidth, figHeight]);
        clf; hold on;

        if size(sampJ, 2) >= 3
            scatter3(sampJ(:, 1), sampJ(:, 2), sampJ(:, 3), 40, colorsJ, 'filled');
            view(45, 30); zlabel('Joint D3');
        elseif size(sampJ, 2) >= 2
            scatter(sampJ(:, 1), sampJ(:, 2), 40, colorsJ, 'filled');
        else
            continue;
        end
        title(sprintf('JOINT %s %s (%d-D) — %s', jointLabelStr, upper(methodNameJ), jointFeatDim, dataSubsetLabel), ...
            'Interpreter', 'none');
        xlabel('Joint D1'); ylabel('Joint D2'); grid on;
        for jb = 1:length(allResults.bhv2ModelCodes)
            bhvIdx = allResults.bhv2ModelCodes(jb);
            scatter(NaN, NaN, 40, colors(bhvIdx+colorsAdjust, :), 'filled', ...
                'DisplayName', behaviors{bhvIdx+colorsAdjust});
        end
        legend('Location', 'best');

        if savePlotFlag
            plotFilename = sprintf('modeling_joint_%s_%s_%s_%dD_bin%.2f.png', ...
                jointLabelStr, methodNameJ, dataSubset, jointFeatDim, opts.frameSize);
            plotPath = fullfile(savePath, plotFilename);
            print(fj, '-depsc', '-vector', plotPath);
            fprintf('Saved plot: %s\n', plotFilename);
        end
    end
end

%% =============================================================================
% --------    BAR CHART ACCURACY (per-area bundles vs conjoint)
% =============================================================================

if plotComparisons
    fig = figure(112);
    set(fig, 'Position', [targetMonitor(1), targetMonitor(2), targetMonitor(3)/2, targetMonitor(4)/1.5]);
    clf;

    if layoutJoint
        fprintf('Plotting joint decoding accuracy comparison...\n');
        methodsJ = allResults.methods;
        accuracyJ = allResults.accuracy(:);
        accuracyPermJ = allResults.accuracyPermuted;
        nShuffleUse = size(accuracyPermJ, 2);

        subplot(1, 1, 1);
        hold on;
        areaMethods = length(methodsJ);
        if areaMethods == 0
            text(0.5, 0.5, 'No Data', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', ...
                'FontSize', 14, 'Color', 'red');
        else
            x = 1:areaMethods;
            barWidth = 0.35;
            bar(x, accuracyJ(:), barWidth, 'FaceColor', 'blue', 'DisplayName', 'Real Data');
            bar(x + barWidth/2, mean(accuracyPermJ, 2), barWidth, 'FaceColor', 'red', 'DisplayName', 'Permuted (mean)');
            errorbar(x + barWidth/2, mean(accuracyPermJ, 2), std(accuracyPermJ, [], 2), 'k.', 'LineWidth', 1.5);
            set(gca, 'XTick', x, 'XTickLabel', upper(methodsJ), 'TickLabelInterpreter', 'none');
            ylabel('Accuracy');
            title(sprintf('Joint %s (%s)', jointLabelStr, dataSubsetLabel), 'Interpreter', 'none');
            ylim([0, 1]);
            for m = 1:areaMethods
                permSorted = sort(accuracyPermJ(m, :));
                q = min(length(permSorted), max(1, ceil(0.95 * nShuffleUse)));
                threshold95 = permSorted(q);
                if accuracyJ(m) > threshold95
                    text(m, accuracyJ(m) + 0.02, '*', 'HorizontalAlignment', 'center', 'FontSize', 16, 'FontWeight', 'bold');
                end
            end
        end
        sgtitle(sprintf('SVM joint decoding (%d-D input)', jointFeatDim), 'FontSize', 16);

        if savePlotFlag
            jointSlug = strrep(jointLabelStr, '+', '_');
            nCompFileTag = nComponentsJoint;
            if isempty(nCompFileTag)
                nCompFileTag = 0;
            end
            plotFilename = sprintf('accuracy_svm_joint_%s_%s_%s_embed%d_nComp%d_bin%.2f_nShuffles%d.eps', ...
                jointSlug, kernelFunction, dataSubset, plotDimPerArea, nCompFileTag, opts.frameSize, nShuffleUse);
            plotPath = fullfile(savePath, plotFilename);
            print(fig, '-depsc', '-vector', plotPath);
            fprintf('Saved comparison plot: %s\n', plotFilename);
        end

        fprintf('\n=== SUMMARY (JOINT) ===\n');
        fprintf('Region\tMethod\tReal\tPermuted (mean ± std)\tDifference\n');
        for m = 1:areaMethods
            diffAcc = accuracyJ(m) - mean(accuracyPermJ(m, :));
            fprintf('%s\t%s\t%.3f\t%.3f ± %.3f\t%.3f\n', jointLabelStr, upper(methodsJ{m}), ...
                accuracyJ(m), mean(accuracyPermJ(m, :)), std(accuracyPermJ(m, :)), diffAcc);
        end
        if areaMethods > 0
            [bestAcc, bestIdx] = max(accuracyJ);
            fprintf('Best method (joint): %s (accuracy: %.3f)\n', upper(methodsJ{bestIdx}), bestAcc);
        end

    else
        fprintf('Plotting per-area decoding accuracy comparison...\n');

        if isfield(allResults, 'areasToTest') && ~isempty(allResults.areasToTest)
            allAreasToPlot = allResults.areasToTest(:)';
        else
            allAreasToPlot = 2:3;
        end
        nAreas = length(allAreasToPlot);

        nMethodsFound = 0;
        for checkArea = allAreasToPlot
            if ~isempty(allResults.methods{checkArea})
                nMethodsFound = length(allResults.methods{checkArea});
                break;
            end
        end

        if nMethodsFound == 0
            fprintf('No data found for any area. Skipping plot.\n');
        end

        for a = 1:nAreas
            areaIdx = allAreasToPlot(a);
            areaName = areas{areaIdx};

            subplot(1, nAreas, a);
            hold on;

            if isempty(allResults.methods{areaIdx}) || isempty(allResults.accuracy{areaIdx})
                text(0.5, 0.5, 'No Data', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', ...
                    'FontSize', 14, 'Color', 'red');
                title(sprintf('%s (%s) - No Data', areaName, dataSubsetLabel), 'Interpreter', 'none');
                ylim([0, 1]);
                xlim([0, 1]);
                continue;
            end

            accuracy = allResults.accuracy{areaIdx};
            accuracyPermuted = allResults.accuracyPermuted{areaIdx};
            methods = allResults.methods{areaIdx};
            areaMethods = length(methods);
            nShuffleUse = size(accuracyPermuted, 2);

            x = 1:areaMethods;
            barWidth = 0.35;
            bar(x, accuracy(:), barWidth, 'FaceColor', 'blue', 'DisplayName', 'Real Data');
            bar(x + barWidth/2, mean(accuracyPermuted, 2), barWidth, 'FaceColor', 'red', 'DisplayName', 'Permuted (mean)');
            errorbar(x + barWidth/2, mean(accuracyPermuted, 2), std(accuracyPermuted, [], 2), 'k.', 'LineWidth', 1.5);
            set(gca, 'XTick', x, 'XTickLabel', upper(methods), 'TickLabelInterpreter', 'none');
            ylabel('Accuracy');
            title(sprintf('%s', areaName), 'Interpreter', 'none');
            ylim([0, 1]);

            for m = 1:areaMethods
                permSorted = sort(accuracyPermuted(m, :));
                q = min(length(permSorted), max(1, ceil(0.95 * nShuffleUse)));
                threshold95 = permSorted(q);
                if accuracy(m) > threshold95
                    text(m, accuracy(m) + 0.02, '*', 'HorizontalAlignment', 'center', 'FontSize', 16, 'FontWeight', 'bold');
                end
            end
        end

        sgtitle('SVM Decoding Accuracy Comparison - All Areas', 'FontSize', 16);

        if savePlotFlag
            plotFilename = sprintf('accuracy_svm_%s_%s_%s_%dD_bin%.2f_nShuffles%d.eps', ...
                dataType, kernelFunction, dataSubset, plotDimPerArea, opts.frameSize, nShuffles);
            plotPath = fullfile(savePath, plotFilename);
            print(fig, '-depsc', '-vector', plotPath);
            fprintf('Saved comparison plot: %s\n', plotFilename);
        end

        fprintf('\n=== SUMMARY FOR ALL AREAS ===\n');
        fprintf('Area\t\tMethod\t\tReal\t\tPermuted (mean ± std)\t\tDifference\n');
        fprintf('----\t\t-----\t\t----\t\t----------------------\t\t----------\n');

        for a = 1:nAreas
            areaIdx = allAreasToPlot(a);
            areaName = areas{areaIdx};
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
            [bestAcc, bestIdx] = max(accuracy);
            fprintf('Best method for %s: %s (accuracy: %.3f)\n', areaName, upper(methods{bestIdx}), bestAcc);
            fprintf('\n');
        end
    end
end
