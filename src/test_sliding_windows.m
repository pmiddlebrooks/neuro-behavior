%%
opts.method = 'gaussian';
opts.frameSize = 1/60;
opts.gaussWidth = 10; % ms
getDataType = 'spikes';
get_standard_data
%%
fileName = sprintf('dataMat_%d_min_frame%.3f_gaussian%d.mat', opts.collectFor/60, opts.frameSize, opts.gaussWidth);
save(fullfile(paths.dropPath, 'dataMat', fileName), 'dataMat', '-mat');
%%
fileName = sprintf('dataMat_%d_min_frame%.3f_gaussian%d.mat', opts.collectFor/60, opts.frameSize, opts.gaussWidth);
load(fullfile(paths.dropPath, 'dataMat', fileName));

%%
minCount = 30;
% Extract unique behaviors
uniqueBhv = unique(bhvID);
numBhv = length(uniqueBhv);

% Create transition matrix (zero-initialized)
transitionMatrix = zeros(numBhv, numBhv);

% Find transition points (ignoring consecutive repetitions)
transitionPoints = find(diff(bhvID) ~= 0);

% Compute transitions only when behavior changes
for i = 1:length(transitionPoints)
    fromIdx = find(uniqueBhv == bhvID(transitionPoints(i)));
    toIdx = find(uniqueBhv == bhvID(transitionPoints(i) + 1));
    transitionMatrix(fromIdx, toIdx) = transitionMatrix(fromIdx, toIdx) + 1;
end

% Normalize to obtain probabilities
% transitionMatrix = transitionMatrix ./ sum(transitionMatrix, 2);

% Replace NaNs (from rows with no transitions) with zeros
transitionMatrix(isnan(transitionMatrix)) = 0;

% Plot the transition matrix
figure(65); clf
subplot(1,2,1);
colormap(parula); % Use a visually effective colormap
imagesc(transitionMatrix);
colorbar;
% Overlay numbers on the matrix
num_rows = size(transitionMatrix, 1);
num_cols = size(transitionMatrix, 2);
for i = 1:num_rows
    for j = 1:num_cols
        text(j, i, num2str(transitionMatrix(i, j)), 'HorizontalAlignment', 'center', ...
            'Color', 'w', 'FontSize', 12, 'FontWeight', 'bold');
    end
end


% Set axis labels
xticks(1:numBhv);
yticks(1:numBhv);
xticklabels(string(uniqueBhv));
yticklabels(string(uniqueBhv));
xlabel('To Behavior');
ylabel('From Behavior');
title('Behavior Transition Matrix (Only Changes)');

% Improve readability
set(gca, 'FontSize', 12, 'XTickLabelRotation', 45);

% Create binary transition matrix based on minCount threshold
binaryMatrix = transitionMatrix >= minCount;

% Plot the binary transition matrix
subplot(1,2,2);
imagesc(binaryMatrix);
% colormap([1 1 1; 0 0 0]); % White for below minCount, Black for above

% Set axis labels
xticks(1:numBhv);
yticks(1:numBhv);
xticklabels(string(uniqueBhv));
yticklabels(string(uniqueBhv));
xlabel('To Behavior');
ylabel('From Behavior');
title(['Transitions with Count ≥ ', num2str(minCount)]);
set(gca, 'FontSize', 12, 'XTickLabelRotation', 45);






%% Get a dataMat for bhvID to find good transitions below

monitorPositions = get(0, 'MonitorPositions');
monitorOne = monitorPositions(1, :); % Just use single monitor if you don't have second one
monitorTwo = monitorPositions(size(monitorPositions, 1), :); % Just use single monitor if you don't have second one



opts = neuro_behavior_options;
opts.minActTime = .16;
opts.collectStart = 0 * 60 * 60; % seconds
opts.collectFor = 60 * 60; % seconds
opts.frameSize = .1;

opts.frameSize = 1/60;

opts.method = 'standard';
opts.windowSize = .2;
opts.stepSize = opts.frameSize;

getDataType = 'behavior';


get_standard_data
%% Save some dataMats to load them quicker than forming them each time
% for f = 2:length(frameSizes)
%     for w = 1 : length(windowSizes)
%         for a = 1:length(windowAligns)
%             tic
%             fprintf('\n%s %s f %s w %s a %s\n', selectFrom, lowDModel, num2str(frameSizes(f)), num2str(windowSizes(w)), windowAligns{a});
%
%             opts.frameSize = frameSizes(f);
%             opts.stepSize = frameSizes(f);
%             opts.windowSize = windowSizes(w);
%             opts.windowAlign = windowAligns{a};
%
%             get_standard_data
%             fileName = sprintf('dataMat_%d_min_step%.3f_window%.2f_align_%s.mat', opts.collectFor/60, opts.stepSize, opts.windowSize, opts.windowAlign);
%             save(fullfile(paths.dropPath, 'dataMat', fileName), 'dataMat', '-mat');
%             toc
%         end
%     end
% end

%%

% opts.useOverlappingBins = 1;
% opts.frameSize = 1/60;
% opts.windowSize = .2;
% opts.stepSize = opts.frameSize;
% getDataType = 'behavior';
% get_standard_data




%% Set the various step sizes and windows to test

nDim = 6;

frameSizes = [1/60 2/60];
% frameSizes = 1/60;
windowSizes = [.05 .1 .15 .2 .25];
% windowSizes = [.15];
windowAligns = {'center', 'left', 'right'};
% windowSizes = [.2];


%% How much around transitions do you want to decode?
opts.mPreTime = .14;
opts.mPostTime = .14;
transWindow = ceil(-opts.mPreTime/opts.frameSize : opts.mPostTime/opts.frameSize - 1);


% fromBehavior = 'itch';
fromBehavior = 'locomotion';
% fromBehavior = 'locomotion';
% toBehavior = 'locomotion';
toBehavior = 'itch';
% fromCodes = codes(contains(behaviors, 'groom'));
fromCodes = codes(contains(behaviors, fromBehavior));
toCode = codes(contains(behaviors, toBehavior));
opts.transTo = toCode;
opts.transFrom = fromCodes;
% opts.minBoutDur = .25;
% opts.minTransFromDur = .25;
opts.minBoutDur = .15;
opts.minTransFromDur = .15;
goodTransitions = find_good_transitions(bhvID, opts);
length(goodTransitions)
svmInd = [];

for i = 1 : length(goodTransitions)
    svmInd = [svmInd; goodTransitions(i) + transWindow'];
end



kFolds = 3; % How many folds for cross-validation?
nPermutations = 1;
accuracyPermuted = zeros(length(frameSizes), length(windowSizes), length(windowAligns), nPermutations);
accuracy = zeros(length(frameSizes), length(windowSizes), length(windowAligns), kFolds);

lowDModel = 'umap';
% lowDModel = 'tsne';
lowDModel = 'pca';

selectFrom = 'M56';
% selectFrom = 'DS';
% selectFrom = 'Both';
% selectFrom = 'VS';
% selectFrom = 'All';
switch selectFrom
    case 'M56'
        % projSelect = projectionsM56;
        % projProject = projectionsDS;
        idSelect = idM56;
        figHFull = 260;
        figHModel = 270;
        figHFullModel = 280;
    case 'DS'
        % projSelect = projectionsDS;
        % projProject = projectionsM56;
        idSelect = idDS;
        figHFull = 261;
        figHModel = 271;
        figHFullModel = 281;
    case 'Both'
        % projSelect = [projectionsM56; projectionsDS];
        idSelect = [idM56, idDS];
        figHFull = 262;
        figHModel = 272;
        figHFullModel = 282;
    case 'VS'
        % projSelect = projectionsVS;
        % projProject = projectionsDS;
        idSelect = idVS;
        figHFull = 263;
        figHModel = 273;
        figHFullModel = 283;
    case 'All'
        % projSelect = projectionsAll;
        % projProject = projectionsDS;
        idSelect = cell2mat(idAll);
        figHFull = 264;
        figHModel = 274;
        figHFullModel = 284;
end











lowDs = {'pca', 'umap'};
% lowDs = {'umap'};
% lowDs = {'pca'};


%% Loop over the parameters to test them
for m = 1 :length(lowDs)
    lowDModel = lowDs{m};
    for f = 1:length(frameSizes)
        for w = 1 : length(windowSizes)
            for a = 1:length(windowAligns)
                disp('======================================================================')
                fprintf('\n\n%s %s Predicting %s to %s... f %.3f w %.2f a %s\n', selectFrom, lowDModel, fromBehavior, toBehavior, ...
                    frameSizes(f), windowSizes(w), windowAligns{a});
                % fprintf('\n\n%s Predicting %s to %s... %s %s\n', selectFrom, fromBehavior, toBehavior, ...
                % lowDModel, opts.method);

                opts.frameSize = frameSizes(f);
                opts.stepSize = frameSizes(f);
                opts.windowSize = windowSizes(w);
                opts.windowAlign = windowAligns{a};
                transWindow = round(-opts.mPreTime/opts.frameSize : opts.mPostTime/opts.frameSize - 1);

                getDataType = 'behavior';
                get_standard_data
                fileName = sprintf('dataMat_%d_min_step%.3f_window%.2f_align_%s.mat', opts.collectFor/60, opts.stepSize, opts.windowSize, opts.windowAlign);
                % fileName = sprintf('dataMat_%d_min_frame%.3f_gaussian%d.mat', opts.collectFor/60, opts.frameSize, opts.gaussWidth);
                load(fullfile(paths.dropPath, 'dataMat', fileName), 'dataMat');

                % tic
                % fileName = sprintf('dataMat_step%.3f_window%.2f_align_%s.mat', opts.stepSize, opts.windowSize, opts.windowAlign);
                % save(fullfile(paths.dropPath, fileName));
                % fprintf('%.2f sec to make dataMat\n', toc)

                % Get the indices of data you want to model
                goodTransitions = find_good_transitions(bhvID, opts);
                fprintf('Fitting %d transitions from %s to %s\n', length(goodTransitions), fromBehavior, toBehavior)
                svmInd = [];
                for i = 1 : length(goodTransitions)
                    svmInd = [svmInd; goodTransitions(i) + transWindow'];
                end

                colors = colors_for_behaviors(codes);

                % Project into low-D. Load the files if there are saved
                % versions. Otherwise, project here
                switch lowDModel
                    case 'umap'

                        fileName = sprintf('umap_%ddim_%d_min_step%.3f_window%.2f_align_%s.mat', nDim, opts.collectFor/60, opts.stepSize, opts.windowSize, opts.windowAlign);
                        saveP = fullfile(paths.dropPath, 'dataMat', fileName);
                        if exist(saveP, 'file')
                            load(saveP);
                        else
                            % fprintf('\n%s %s min_dist=%.2f spread=%.1f n_n=%d\n\n', selectFrom, fitType, min_dist(x), spread(y), n_neighbors(z));
                            umapFrameSize = opts.frameSize;
                            rng(1);
                            % [projSelect, ~, ~, ~] = run_umap(zscore(dataMat(:, idSelect)), 'n_components', nDim, 'randomize', false, 'verbose', 'none', ...
                            %     'min_dist', min_dist(x), 'spread', spread(y), 'n_neighbors', n_neighbors(z));

                            [projSelect, ~, ~, ~] = run_umap(zscore(dataMat(:, idSelect)), 'n_components', nDim, 'randomize', true, 'verbose', 'none');
                            % [projSelect, ~, ~, ~] = run_umap(zscore(dataMat(svmInd, idSelect)), 'n_components', nDim, 'randomize', true, 'verbose', 'none');

                            % minFrames = 2;
                            %                     [stackedActivity, stackedLabels] = datamat_stacked_means(dataMat, bhvID, minFrames);
                            %                     [~, umap, ~, ~] = run_umap(zscore(stackedActivity(:, idSelect)), 'n_components', iDim, 'randomize', false, 'verbose', 'none', ...
                            %                         'min_dist', min_dist(x), 'spread', spread(y), 'n_neighbors', n_neighbors(z));
                            %                     projSelect = umap.transform(zscore(dataMat(:, idSelect)));
                            save(saveP, 'projSelect');
                        end
                    case 'pca'
                        % fileName = sprintf('pca_%d_min_step%.3f_window%.2f_align_%s.mat', opts.collectFor/60, opts.stepSize, opts.windowSize, opts.windowAlign);
                        % saveP = fullfile(paths.dropPath, 'dataMat', fileName);
                        % if exist(saveP, 'file')
                        %     load(saveP);
                        % else
                        [coeff, projSelect, ~, ~, explained] = pca(zscore(dataMat(:, idSelect)));

                        % expVarThresh = 25;
                        % dimExplained = find(cumsum(explained) > expVarThresh, 1);
                        % disp(['PCA: Exp Var = ', num2str(expVar(k)), ' nComponents = ', num2str(pcaDim(k))])
                        % [projSelect, ~, ~, ~] = run_umap(score(:, 1:pcaDim(k)), 'n_components', iDim, 'randomize', false, 'verbose', 'none', ...
                        %     'min_dist', min_dist(x), 'spread', spread(y), 'n_neighbors', n_neighbors(z));
                        save(saveP, 'projSelect', 'coeff', 'explained');
                        % end
                    case 'tsne'
                        fileName = sprintf('tsne_%ddim_%d_min_step%.3f_window%.2f_align_%s.mat', nDim, opts.collectFor/60, opts.stepSize, opts.windowSize, opts.windowAlign);
                        saveP = fullfile(paths.dropPath, 'dataMat', fileName);
                        if exist(saveP, 'file')
                            load(saveP);
                        else
                            % fprintf('\n%s %s exagg=%d perplx=%d \n\n', selectFrom, fitType, exaggeration(x), perplexity(y));
                            % projSelect = tsne(zscore(dataMat(:, idSelect)),'Exaggeration', exaggeration(x), 'Perplexity', perplexity(y), 'NumDimensions',iDim);
                            projSelect = tsne(zscore(dataMat(:, idSelect)), 'NumDimensions',nDim);
                            save(saveP, 'projSelect');
                        end
                end


                % Create a neural matrix and labels vector for the decoding
                svmProj = projSelect(svmInd,1:nDim);
                % svmProj = projSelect;
                svmID = bhvID(svmInd);


                %% Get the neural frames and labels for decoding

                svmID(ismember(svmID, fromCodes)) = -4;
                svmID(ismember(svmID, toCode)) = -3;
                colors = [0 0 0; 0 .6 .2];

                % Plot data to model
                % colorsForPlot = arrayfun(@(x) colors(x,:), svmID + 2, 'UniformOutput', false);
                colorsForPlot = arrayfun(@(x) colors(x,:), svmID+5, 'UniformOutput', false);
                colorsForPlot = vertcat(colorsForPlot{:}); % Convert cell array to a matrix

                figH = figHModel;
                iDim = nDim;
                % Plot on second monitor, half-width
                plotPos = [monitorTwo(1) + monitorTwo(3)/2, 1, monitorTwo(3)/2, monitorTwo(4)];
                % titleM = [selectFrom, ' ', fitType, ' ', transWithinLabel, ' bin = ', num2str(opts.frameSize), ' shift = ', num2str(shiftSec)];
                % titleM = [selectFrom, ' ', lowDModel, ' ', fromBehavior, ' to ', toBehavior, ' step=', num2str(opts.stepSize), ' win=', num2str(opts.windowSize), '  align=', opts.windowAlign];
                titleM = sprintf('%s %s %s to %s step=%.3f win=%.2f align=%s', selectFrom, lowDModel, fromBehavior, toBehavior, opts.stepSize, opts.windowSize, opts.windowAlign);
                % titleM = sprintf('%s %s %s to %s %s', selectFrom, lowDModel, fromBehavior, toBehavior, opts.method);

                plotFrames = svmInd;
                % plotFrames = 1:length(svmID);
                plot_3d_scatter






                %%
                % Use SVM to decode the behaviors
                % Split data into training (80%) and testing (20%) sets
                % cv = cvpartition(svmID, 'HoldOut', 0.2);
                % Perform 3-fold cross-validation instead of hold-out
                cv = cvpartition(svmID, 'KFold', kFolds);

                disp('===========================================')

                % low dimension version
                fprintf('%s DIM %d %.2f Window\n', selectFrom, nDim, windowSizes(w))  %
                % fprintf('%s DIM %d %s\n', selectFrom, nDim, opts.method)  %
                tic
                for k = 1:kFolds
                    % Choose which data to model
                    trainData = svmProj(training(cv, k), :);  % pca Dimensions
                    testData = svmProj(test(cv, k), :); % pca Dimensions



                    trainLabels = svmID(training(cv, k));
                    testLabels = svmID(test(cv, k));


                    % Define different kernel functions to try
                    % kernelFunctions = {'linear', 'gaussian', 'polynomial', 'rbf'};
                    kernelFunction = 'polynomial';

                    % Train model
                    % Set SVM template with the current kernel
                    t = templateSVM('Standardize', true, 'KernelFunction', kernelFunction);

                    % Train the SVM model using cross-validation
                    svmModel = fitcecoc(trainData, trainLabels, 'Learners', t);

                    predictedLabels = predict(svmModel, testData);

                    % Calculate and display the overall accuracy
                    accuracy(f, w, a, k) = sum(predictedLabels == testLabels) / length(testLabels);
                    fprintf('%s Fold Accuracy: %.4f\n', selectFrom, accuracy(f, w, a, k));
                end
                meanAccuracy = mean(accuracy(f, w, a,:));
                fprintf('%s Overall Accuracy: %.4f\n', selectFrom, meanAccuracy);

                fprintf('Model fit took %.2f min\n', toc/60)

                tic


                % Randomize labels and Train model on single hold-out set
                % tic
                for iPerm = 1:nPermutations

                    % Split data into training (80%) and testing (20%) sets
                    cv = cvpartition(svmID, 'HoldOut', 0.2);

                    trainData = svmProj(training(cv), :);  % pca Dimensions
                    testData = svmProj(test(cv), :); % pca Dimensions

                    trainLabels = svmID(training(cv));
                    testLabels = svmID(test(cv));

                    shuffleInd = zeros(length(trainLabels), nPermutations);
                    iRandom = 1 : length(trainLabels);

                    randShift = randi([1 length(trainLabels)]);
                    % Shuffle the data by moving the last randShift elements to the front
                    lastNElements = iRandom(end - randShift + 1:end);  % Extract the last randShift elements
                    iRandom(randShift+1:end) = iRandom(1:end-randShift); % Shift the remaining elements to the end
                    iRandom(1:randShift) = lastNElements; % Place the last n elements at the beginning
                    shuffleInd(:, iPerm) = iRandom;
                    shuffledLabels = trainLabels(shuffleInd(:, iPerm));


                    % Set SVM template with the current kernel
                    t = templateSVM('Standardize', true, 'KernelFunction', kernelFunction);

                    % Train the SVM model on shuffled training data
                    svmModelPermuted = fitcecoc(trainData, shuffledLabels, 'Learners', t);

                    % Predict the labels using observed test data
                    predictedLabelsPermuted = predict(svmModelPermuted, testData);

                    % Calculate the permuted accuracy
                    accuracyPermuted(f, w, a, iPerm) = sum(predictedLabelsPermuted == testLabels) / length(testLabels);
                    fprintf('Permuted %s Accuracy permutation %d: %.4f\n', selectFrom, iPerm, accuracyPermuted(f, w, a, iPerm));

                end

                fprintf('Mean Permuted = %.4f\n', mean(accuracyPermuted(f, w, a, :)));

                % Get the elapsed time
                fprintf('Permutation model fit(s) took %.2f min\n', toc/60)







            end
        end
    end

    % % Reassign accuracy variables
    % accName = ['acc_', selectFrom, '_', fromBehavior, '_', toBehavior]
    % eval([accName, ' = accuracy;']);
    %
    % accNameP = ['accPerm_', selectFrom, '_', fromBehavior, '_', toBehavior]
    % eval([accNameP, ' = accuracyPermuted;']);



    % slack_code_done





    % Compare the accuracy between the various dataMats

    % f, w, a
    figure(883); clf;
    for f = 1 : length(frameSizes)
        subplot(1,2,f)
        hold on;
        for a = 1 : 3
            plot(mean(accuracy(f,:,a,:), 4), '-o', 'linewidth', 2)
        end
        ylim([.4 round(max(accuracy(:)))])
        xticks(1:length(windowSizes))
        xlim([.5 .5 + length(windowSizes)])
        xticklabels(windowSizes)
        legend({'Center', 'Left', 'Right'}, 'Location','southeast')
        title(sprintf('Step size %.3f', frameSizes(f)))
        ylabel('Accuracy')
        xlabel('Window size')
    end
    titleS = sprintf('%s %s %s to %s n=%d', selectFrom, lowDModel, fromBehavior, toBehavior, length(goodTransitions));
    sgtitle(titleS, 'interpreter', 'none')

    % print('-dpdf', fullfile(paths.dropPath, [titleS, '.pdf']), '-bestfit')
    print('-dpng', fullfile(paths.dropPath, [titleS, '.png']))
    copy_figure_to_clipboard
end
%%
selectFrom = 'M56';
fromBehavior = 'locomotion';
toBehavior = 'investigate_2';
% toBehavior = 'investigate_2';
accPlot2 = acc_M56_locomotion_investigate_2;
accPlot = acc_M56_investigate_2_locomotion;
figure(883); clf;
for f = 1 : length(frameSizes)
    subplot(1,2,f)
    hold on;
    for a = 1 : 3
        plot(mean(accPlot(f,:,a,:), 4), 'linewidth', 2)
        plot(mean(accPlot2(f,:,a,:), 4), 'linewidth', 2)
    end
    ylim([.4 round(max(accuracy(:)))])
    xticks(1:length(windowSizes))
    xlim([.5 .5 + length(windowSizes)])
    xticklabels(windowSizes)
    legend({'Center', 'Left', 'Right'})
    title(['Step size ', num2str(frameSizes(f))])
    ylabel('Accuracy')
    xlabel('Window size')
end
titleS = sprintf('%s %s %s to %s', selectFrom, lowDModel, fromBehavior, toBehavior);
sgtitle(titleS, 'interpreter', 'none')
copy_figure_to_clipboard