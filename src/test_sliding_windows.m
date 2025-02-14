%% Testing different sliding window parameters

monitorPositions = get(0, 'MonitorPositions');
monitorOne = monitorPositions(1, :); % Just use single monitor if you don't have second one
monitorTwo = monitorPositions(size(monitorPositions, 1), :); % Just use single monitor if you don't have second one



opts = neuro_behavior_options;
opts.minActTime = .16;
opts.collectStart = 0 * 60 * 60; % seconds
opts.collectFor = 45 * 60; % seconds
opts.frameSize = .1;

opts.frameSize = 1/60;

opts.useOverlappingBins = 1;
opts.windowSize = .2;
opts.stepSize = opts.frameSize;

getDataType = 'spikes';

nDim = 6;

frameSizes = [.01 1/60 2/60];
frameSizes = 1/60;
windowSizes = [.05 .1 .15 .2 .25];
% windowSizes = [.2];

% get_standard_data

%%
fromBehavior = 'locomotion';
toBehavior = 'investigate_2';
fromCodes = codes(contains(behaviors, 'groom'));
fromCodes = codes(contains(behaviors, fromBehavior));
toCode = codes(contains(behaviors, toBehavior));
opts.transTo = toCode;
opts.transFrom = fromCodes;
opts.minBoutDur = .25;
opts.minTransFromDur = .25;
goodTransitions = find_good_transitions(bhvID, opts);
length(goodTransitions)
svmInd = [];
for i = 1 : length(goodTransitions)
    svmInd = [svmInd; goodTransitions(i) + transWindow'];
end


%% How much around transitions do you want to decode?
opts.mPreTime = .2;
opts.mPostTime = .2;
transWindow = round(-opts.mPreTime/opts.frameSize : opts.mPostTime/opts.frameSize - 1);

kFolds = 3; % How many folds for cross-validation?
nPermutations = 10;
accuracyPermuted = zeros(length(windowSizes), nPermutations);
accuracy = zeros(length(windowSizes), kFolds);

lowDModel = 'umap';

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

%%

for f = 1:length(frameSizes)
    for w = 1 : length(windowSizes)
        fprintf('\n%s %s \n', selectFrom, lowDModel);

        opts.frameSize = frameSizes(f);
        opts.stepSize = frameSizes(f);
        opts.windowSize = windowSizes(w);
        get_standard_data

        % Get the indices of data you want to model
        goodTransitions = find_good_transitions(bhvID, opts);
        fprintf('Fitting %d transitions from %s to %s\n', length(goodTransitions), fromBehavior, toBehavior)
        svmInd = [];
        for i = 1 : length(goodTransitions)
            svmInd = [svmInd; goodTransitions(i) + transWindow'];
        end

        colors = colors_for_behaviors(codes);

        % Project into low-D
        switch lowDModel
            case 'umap'
                % fprintf('\n%s %s min_dist=%.2f spread=%.1f n_n=%d\n\n', selectFrom, fitType, min_dist(x), spread(y), n_neighbors(z));
                umapFrameSize = opts.frameSize;
                rng(1);
                % [projSelect, ~, ~, ~] = run_umap(zscore(dataMat(:, idSelect)), 'n_components', nDim, 'randomize', false, 'verbose', 'none', ...
                %     'min_dist', min_dist(x), 'spread', spread(y), 'n_neighbors', n_neighbors(z));

                [projSelect, ~, ~, ~] = run_umap(zscore(dataMat(:, idSelect)), 'n_components', nDim, 'randomize', false, 'verbose', 'none');

                % minFrames = 2;
                %                     [stackedActivity, stackedLabels] = datamat_stacked_means(dataMat, bhvID, minFrames);
                %                     [~, umap, ~, ~] = run_umap(zscore(stackedActivity(:, idSelect)), 'n_components', iDim, 'randomize', false, 'verbose', 'none', ...
                %                         'min_dist', min_dist(x), 'spread', spread(y), 'n_neighbors', n_neighbors(z));
                %                     projSelect = umap.transform(zscore(dataMat(:, idSelect)));
                pause(2); close
            case 'pca'
                % [coeff, score, ~, ~, explained] = pca(zscore(dataMat(:, idSelect)));
                % expVarThresh = 25;
                % dimExplained = find(cumsum(explained) > expVarThresh, 1);
                % disp(['PCA: Exp Var = ', num2str(expVar(k)), ' nComponents = ', num2str(pcaDim(k))])
                % [projSelect, ~, ~, ~] = run_umap(score(:, 1:pcaDim(k)), 'n_components', iDim, 'randomize', false, 'verbose', 'none', ...
                %     'min_dist', min_dist(x), 'spread', spread(y), 'n_neighbors', n_neighbors(z));

            case 'tsne'
                % fprintf('\n%s %s exagg=%d perplx=%d \n\n', selectFrom, fitType, exaggeration(x), perplexity(y));
                % projSelect = tsne(zscore(dataMat(:, idSelect)),'Exaggeration', exaggeration(x), 'Perplexity', perplexity(y), 'NumDimensions',iDim);
                projSelect = tsne(zscore(dataMat(:, idSelect)),'Exaggeration', 'NumDimensions',nDim);
        end



        %% Get the neural frames and labels for decoding
        svmProj = projSelect(svmInd,:);
        svmID = bhvID(svmInd);



        % Plot data to model
        colorsForPlot = arrayfun(@(x) colors(x,:), svmID + 2, 'UniformOutput', false);
        colorsForPlot = vertcat(colorsForPlot{:}); % Convert cell array to a matrix

        figH = figHModel;
        % Plot on second monitor, half-width
        plotPos = [monitorTwo(1) + monitorTwo(3)/2, 1, monitorTwo(3)/2, monitorTwo(4)];
        titleM = [selectFrom, ' ', fitType, ' ', transWithinLabel, ' bin = ', num2str(opts.frameSize), ' shift = ', num2str(shiftSec)];

        plotFrames = svmInd;
        plot_3d_scatter







        % Use SVM to decode the behaviors
        % Split data into training (80%) and testing (20%) sets
        % cv = cvpartition(svmID, 'HoldOut', 0.2);
        % Perform 3-fold cross-validation instead of hold-out
        cv = cvpartition(svmID, 'KFold', kFolds);

        disp('=================================================================')

        % pca dimension version
        fprintf('\n\n%s DIM %d %.2f Window\n', selectFrom, nDim, windowSizes(w))  % pca Dimensions

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
            modelName = ['svmModel', appendModelName];
            % Reassign the value of modelName to the new variable name using eval
            eval([modelName, ' = svmModel;']);


            predictedLabels = predict(svmModel, testData);

            % Calculate and display the overall accuracy
            accuracy(w, k) = sum(predictedLabels == testLabels) / length(testLabels);
            fprintf('%s Fold Accuracy: %.4f\n', selectFrom, accuracy(w, k));
        end
        meanAccuracy = mean(accuracy(w,:));
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
            accuracyPermuted(w, iPerm) = sum(predictedLabelsPermuted == testLabels) / length(testLabels);
            fprintf('Permuted %s Accuracy permutation %d: %.4f\n', selectFrom, iPerm, accuracyPermuted(w, iPerm));

        end

        fprintf('Mean Permuted = %.4f\n', mean(accuracyPermuted(w, :)));

        % Get the elapsed time
        fprintf('Permutation model fit(s) took %.2f min\n', toc/60)

        





    end
end
slack_code_done
