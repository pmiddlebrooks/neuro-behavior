%% Go to spiking_script and get behavioral and neural data

% Alter this to try out different populations
idInd = [idM56 idDS];
idInd = [idM56]

%% Remove the first and last bouts from the dataBhv table
dataBhvTrunc = dataBhv(2:end-1, :);
validBhvTrunc = validBhv(2:end-1,:);
%% If you want to subsample bouts (match the number of bouts for each behavior...

nBout = zeros(length(analyzeCodes), 1);
for i = 1 : length(analyzeCodes)
    nBout(i) = sum(dataBhvTrunc.ID == analyzeCodes(i) & validBhvTrunc(:, codes == analyzeCodes(i)));
end
nSample = min(nBout);

%% Do you want to subsample to match the number of bouts?
matchBouts = 1;
%% Make a neural matrix with concatenated peri-onset bins for each behavior

% Create a neural matrix. Each column is a neuron. Each row are spike
% counts peri-onset of each behavior.
behaviorID = [];
neuralMatrix = [];
periEventTime = -.1 : opts.frameSize : .1; % seconds around onset
dataWindow = round(periEventTime(1:end-1) / opts.frameSize); % frames around onset (remove last frame)

for iBhv = 1 : length(analyzeCodes)
    dataBhvInd = dataBhvTrunc.ID == analyzeCodes(iBhv) & validBhvTrunc(:, codes == analyzeCodes(iBhv));
    iStartFrames = 1 + floor(dataBhvTrunc.StartTime(dataBhvInd) ./ opts.frameSize);
    if matchBouts
        iRand = randperm(length(iStartFrames));
        iStartFrames = iStartFrames(iRand(1:nSample));
    end
    behaviorID = [behaviorID; analyzeCodes(iBhv) * ones(length(iStartFrames) * length(dataWindow), 1)];
    for jStart = 1 : length(iStartFrames)
        % neuralMatrix = [neuralMatrix; sum(dataMatZ(iStartFrames(jStart) + dataWindow, :))];
        neuralMatrix = [neuralMatrix; dataMatZ(iStartFrames(jStart) + dataWindow, :)];
    end
end

%% Instead, train it with a matrix that goes from -.1 before onset, through the duration of the behavior, to .1 s before next behavior onset
behaviorID = [];
neuralMatrix = [];
preCurrFrames = -.1 / opts.frameSize;
preNextFrames = -.1 / opts.frameSize;
dataBhvTrunc.DurFrame = floor(dataBhvTrunc.Dur ./ opts.frameSize);
dataBhvTrunc.DurFrame(dataBhvTrunc.DurFrame == 0) = 1;
for iBhv = 1 : length(analyzeCodes)
    dataBhvInd = dataBhvTrunc.ID == analyzeCodes(iBhv) & validBhvTrunc(:, codes == analyzeCodes(iBhv));
    iStartFrames = 1 + floor(dataBhvTrunc.StartTime(dataBhvInd) ./ opts.frameSize);
    iDurFrames = dataBhvTrunc.DurFrame(dataBhvInd);
    if matchBouts
        iRand = randperm(length(iStartFrames));
        iStartFrames = iStartFrames(iRand(1:nSample));
        iDurFrames = iDurFrames(iRand(1:nSample));
    end
    for jStart = 1 : length(iStartFrames)
        % Get the duration of this behavior (from -.1 to .1 before next
        % behavior
        jWindowDur = iDurFrames(jStart) + preNextFrames;
        dataWindow = preCurrFrames : jWindowDur - 1;
        behaviorID = [behaviorID; analyzeCodes(iBhv) * ones(length(dataWindow), 1)];
        neuralMatrix = [neuralMatrix; dataMatZ(iStartFrames(jStart) + dataWindow, :)];
    end
end

% Samples still won't be equal since different bouts typically last shorter
% or longer than others
nDataPoints = zeros(length(analyzeCodes), 1);
for i = 1 : length(analyzeCodes)
    nDataPoints(i) = sum(behaviorID == analyzeCodes(i));
end
fprintf('Warning: Using this method you still sample un-matched data points, range: %d to %d\n', min(nDataPoints), max(nDataPoints));
fprintf('... Because your subsampling for bout number, but not durations\n')
%% Instead, train it with a matrix that goes from -.1 before onset, through the the minimum of .5 sec after onset or .1 sec before next behavior onset
behaviorID = [];
neuralMatrix = [];
preCurrFrames = -.1 / opts.frameSize;
maxDur = .5 / opts.frameSize;
preNextFrames = -.1 / opts.frameSize;
dataBhvTrunc.DurFrame = floor(dataBhvTrunc.Dur ./ opts.frameSize);
dataBhvTrunc.DurFrame(dataBhvTrunc.DurFrame == 0) = 1;
for iBhv = 1 : length(analyzeCodes)
    dataBhvInd = dataBhvTrunc.ID == analyzeCodes(iBhv) & validBhvTrunc(:, codes == analyzeCodes(iBhv));
    iStartFrames = 1 + floor(dataBhvTrunc.StartTime(dataBhvInd) ./ opts.frameSize);
    iDurFrames = dataBhvTrunc.DurFrame(dataBhvInd);
    if matchBouts
        iRand = randperm(length(iStartFrames));
        iStartFrames = iStartFrames(iRand(1:nSample));
        iDurFrames = iDurFrames(iRand(1:nSample));
    end
    for jStart = 1 : length(iStartFrames)
        % Get the duration of this behavior (from -.1 to .1 before next
        % behavior
        jWindowDur = iDurFrames(jStart) + preNextFrames;
        dataWindow = preCurrFrames : min(maxDur, jWindowDur - 1);
        behaviorID = [behaviorID; analyzeCodes(iBhv) * ones(length(dataWindow), 1)];
        neuralMatrix = [neuralMatrix; dataMatZ(iStartFrames(jStart) + dataWindow, :)];
    end
end
nDataPoints = zeros(length(analyzeCodes), 1);
for i = 1 : length(analyzeCodes)
    nDataPoints(i) = sum(behaviorID == analyzeCodes(i));
end
fprintf('Warning: Using this method you still sample un-matched data points, range: %d to %d\n', min(nDataPoints), max(nDataPoints));
fprintf('... Because your subsampling for bout number, but not durations\n')





%%

% MdlLinear = fitcdiscr(neuralMatrix, behaviorID);

averageAccuracy = Perform5FoldCVLDA(neuralMatrix(:, idInd), behaviorID);
%%

trainedModel = train_full_lda_model(neuralMatrix(:, idInd), behaviorID)


% %%
% neuronIndices = [4 38];
% selectedBehaviors = [2 11];
% VisualizeProjectionVectorsLDA(neuralMatrix(:, idM56), behaviorID, neuronIndices, selectedBehaviors)
%
%
% %%
% ldaProjectionPlot(neuralMatrix(:, idM56), behaviorID)


%%   FIT and project LDA model
fitMatrix = neuralMatrix(:, idInd);
% Fit the LDA model
ldaModel = fitcdiscr(fitMatrix, behaviorID);

% Eigenvalue decomposition for the lda components
[eigenvectors, eigenvalues] = eig(ldaModel.BetweenSigma, ldaModel.Sigma);

% Sort the eigenvalues and associated eigenvectors
[eigenvalues, sortedIndices] = sort(diag(eigenvalues), 'descend');
eigenvectors = eigenvectors(:, sortedIndices);

% Project the data onto the LDA components
projectedData = fitMatrix * eigenvectors;

% Project full data matrix to plot trajectories
projectedTraj = dataMatZ(:, idInd) * eigenvectors;

dataBhv.StartFrame = 1 + floor(dataBhv.StartTime / opts.frameSize);






%% Plot behavior onsets/transitions in LDA space

nComponents = 5;
[ax] = lda_plot_behaviors(projectedData, behaviorID, nComponents);





%% Plot ongoing behavioral sequences
bhv2Plot = 0;
lda_plot_trajectories(projectedTraj, dataBhv, behaviorID, nComponents, ax, bhv2Plot);


%%   Plot specific behavior sequences
seq1 = 15;
seq2 = 6;
seqCode = [seq1 seq2];

% Get all sequence start times and IDs
[seqStartTimes, seqCodes, seqNames] = behavior_sequences(dataBhv, analyzeCodes, analyzeBhv);

% Make a dataBhv table from a desired sequence
% which index in seqStartTimes is it?
seqIdx = seqCodes(:,1) == seq1 & seqCodes(:,2) == seq2;
% dataBhvIdx = dataBhv.StartTime == seqStartTimes{seqIdx}(:,2);
dataBhvIdx = ismember(dataBhv.StartTime, seqStartTimes{seqIdx}(:,2));
dataBhvSeq = dataBhv(dataBhvIdx, :);
dataBhvSeq.bhvDurFrame = floor(dataBhvSeq.bhvDur ./ opts.frameSize);


lda_plot_seq_trajectories(projectedTraj, dataBhvSeq, seqCode, nComponents, ax)


%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Plot trajectories of behavioral sequences
function lda_plot_seq_trajectories(projectedTraj, dataBhv, seqCode, nComponents, ax)

% bhvList = unique(behaviorID);
colors = colors_for_behaviors(seqCode);

% Plot trajectories
jTraj = 1;
% for iStart = 2 : size(dataBhv, 1)
for iStart = 1 : size(dataBhv, 1)
    iStartFrame = dataBhv.StartFrame(iStart);
    iEndFrame = iStartFrame + dataBhv.DurFrame(iStart);
    iStartColor = colors(1,:);
    iEndColor = colors(2,:);

    % Every time there are 3 trajectories, erase the first one
    if jTraj == 4
        jTraj = 1;
        delete(plt)
        delete(sct1)
        delete(sctS)
        delete(sctE)
    end
    for iComp = 1 : nComponents-1
        axes(ax(iComp))
        hold on

        plt(iComp, jTraj) = plot(projectedTraj(iStartFrame : iEndFrame, iComp), projectedTraj(iStartFrame : iEndFrame, iComp+1), 'Color', iStartColor, 'lineWidth', 2);
        sct1(iComp, jTraj) = scatter(projectedTraj(iStartFrame : iEndFrame, iComp), projectedTraj(iStartFrame : iEndFrame, iComp+1), 30, 'k', 'filled');
        sctS(iComp, jTraj) = scatter(projectedTraj(iStartFrame, iComp), projectedTraj(iStartFrame, iComp+1), 150, iStartColor, 'filled', 'MarkerEdgeColor', 'k');
        sctE(iComp, jTraj) = scatter(projectedTraj(iEndFrame, iComp), projectedTraj(iEndFrame, iComp+1), 150, iEndColor, 'd', 'filled', 'MarkerEdgeColor', 'k');
    end
    jTraj = jTraj + 1;
end

end % function


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function lda_plot_trajectories(projectedTraj, dataBhv, behaviorID, nComponents, ax, bhv2Plot)
bhvList = unique(behaviorID);

bhvStartIdx = find(dataBhv.ID == bhv2Plot);

% colors = distinguishable_colors(length(bhvList));
colors = colors_for_behaviors(bhvList);

% Plot trajectories
jTraj = 1;
for iStart = 2 : size(dataBhv, 1)
% for iStart = 1 : length(bhvStartIdx)
    % Skip trajectories that involve in_nest_sleeping_or_irrelevant
    % if dataBhv.ID(bhvStartIdx(iStart)) == -1 || dataBhv.ID(bhvStartIdx(iStart)+1) == -1
    if dataBhv.ID(iStart) == -1 || dataBhv.ID(iStart+1) == -1
        continue
    end
    % iStartFrame = dataBhv.StartFrame(bhvStartIdx(iStart));
    % iEndFrame = dataBhv.StartFrame(bhvStartIdx(iStart) + 1);
    % iStartColor = colors(bhvList == dataBhv.ID(bhvStartIdx(iStart)), :);
    % iEndColor = colors(bhvList == dataBhv.ID(bhvStartIdx(iStart) + 1), :);
    iStartFrame = dataBhv.StartFrame(iStart);
    iEndFrame = dataBhv.StartFrame(iStart + 1);
    iStartColor = colors(bhvList == dataBhv.ID(iStart), :);
    iEndColor = colors(bhvList == dataBhv.ID(iStart + 1), :);

    % Every time there are 3 trajectories, erase the first one
    if jTraj == 4
        jTraj = 1;
        delete(plt)
        delete(sct1)
        delete(sctS)
        delete(sctE)
    end
    for iComp = 1 : nComponents-1
        axes(ax(iComp))
        hold on

        plt(iComp, jTraj) = plot(projectedTraj(iStartFrame : iEndFrame, iComp), projectedTraj(iStartFrame : iEndFrame, iComp+1), 'Color', iStartColor, 'lineWidth', 2);
        sct1(iComp, jTraj) = scatter(projectedTraj(iStartFrame : iEndFrame, iComp), projectedTraj(iStartFrame : iEndFrame, iComp+1), 30, 'k', 'filled');
        sctS(iComp, jTraj) = scatter(projectedTraj(iStartFrame, iComp), projectedTraj(iStartFrame, iComp+1), 150, iStartColor, 'filled', 'MarkerEdgeColor', 'k');
        sctE(iComp, jTraj) = scatter(projectedTraj(iEndFrame, iComp), projectedTraj(iEndFrame, iComp+1), 150, iEndColor, 'd', 'filled', 'MarkerEdgeColor', 'k');
    end
    jTraj = jTraj + 1;
end

end % function




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function ax = lda_plot_behaviors(projectedData, behaviorID, nComponents)
% Generates an LDA space based on spike counts peri-onset w.r.t. behavior
% onsets, across all behaviors
% Projects the peri-onset spike times into the LDA space for each behvaior.
% For the first 10 LDA components, Plots the trial-wise examples from each behavior, or the means + std (as
% ellipses, as a function of component x and x + 1

bhvList = unique(behaviorID);
% colors = distinguishable_colors(length(bhvList));
colors = colors_for_behaviors(bhvList);

% Create 3x3 grid of subplots
% Get monitor positions and size
monitorPositions = get(0, 'MonitorPositions');
if size(monitorPositions, 1) < 2
    error('Second monitor not detected');
end
secondMonitorPosition = monitorPositions(2, :);
% Create a maximized figure on the second monitor
fig = figure(81);
clf
set(fig, 'Position', secondMonitorPosition);
nPlot = nComponents - 1;
[ax, pos] = tight_subplot(ceil(nPlot/2), ceil(nPlot/2));


% Plot all variances as ellipses
for bhv = 1 : length(bhvList)
    bhvIdx = behaviorID == bhvList(bhv);
    bhvColor = colors(bhv,:);

    for i = 1 : nComponents-1
        axes(ax(i))
        hold on

        % Extract the successive pairs of components
        componentX = projectedData(bhvIdx, i);
        componentY = projectedData(bhvIdx, i+1);

        % Create a scatter plot of the projected data for each pair of components
        plotDataOrMean = 0;
        if plotDataOrMean
            scatter(componentX, componentY, 30, bhvColor);
        else
            x = mean(componentX);
            y = mean(componentY);
            w = std(componentX);
            h = std(componentY);

            % Generate points on the ellipse
            theta = linspace(0, 2 * pi, 100); % Generate 100 points
            ellipseX = (w / 2) * cos(theta) + x; % X coordinates
            ellipseY = (h / 2) * sin(theta) + y; % Y coordinates

            % Plot the ellipse
            fill(ellipseX, ellipseY, bhvColor, 'FaceAlpha', 0.15, 'EdgeColor', 'none'); % Fill the ellipse with semi-transparency
        end
        if bhv == 1
            title(['LDA Components ', num2str(i), ' and ', num2str(i+1)]);
            % xlabel(['Component ', num2str(i)]);
            % ylabel(['Component ', num2str(i+1)]);
            xlabel([]);
            ylabel([]);
            grid on;
        end
    end
end
% Plot all means on top of their variance ellipses
for bhv = 1 : length(bhvList)
    bhvIdx = behaviorID == bhvList(bhv);
    bhvColor = colors(bhv,:);

    for i = 1 : nComponents-1
        axes(ax(i))
        hold on

        % Extract the successive pairs of components
        componentX = projectedData(bhvIdx, i);
        componentY = projectedData(bhvIdx, i+1);

        % Create a scatter plot of the projected data for each pair of components
        plotDataOrMean = 0;
        if plotDataOrMean
        else
            x = mean(componentX);
            y = mean(componentY);
            s = scatter(x, y, 130, bhvColor, 'filled');
        end
    end
end
% Adjust subplot layout to minimize overlap
sgtitle('LDA Component Projections'); % Add a centered title if desired
% set(gcf, 'Units', 'normalized', 'OuterPosition', [0 0 1 1]); % Maximize figure within monitor

end % function








function trainedModel = train_full_lda_model(neuralMatrix, behaviorID)
% train_full_lda_model: Performs Linear Discriminant Analysis on
% spiking data
% Model is based on all the data. No cross-validation is performed.
% neuralMatrix is a matrix where each column is a neuron, each row is a time point.
% behaviorID is a vector of categories corresponding to time points in neuralMatrix.

% Validate inputs
if size(neuralMatrix, 1) ~= length(behaviorID)
    error('Number of rows in neuralMatrix must equal length of behaviorID');
end

% Train LDA model
trainedModel = fitcdiscr(neuralMatrix, behaviorID);

end




function averageAccuracy = Perform5FoldCVLDA(neuralMatrix, behaviorID)
% Perform5FoldCVLDA: Performs 5-fold cross-validation for Linear Discriminant Analysis on spiking data.
% neuralMatrix is a matrix where each column is a neuron, each row is a time point.
% behaviorID is a vector of categories corresponding to time points in neuralMatrix.

% Validate inputs
if size(neuralMatrix, 1) ~= length(behaviorID)
    error('Number of rows in neuralMatrix must equal length of behaviorID');
end

% Define number of folds for cross-validation
numFolds = 5;
cvPartition = cvpartition(behaviorID, 'KFold', numFolds);

% Initialize accuracy for each fold
accuracies = zeros(numFolds, 1);

% Perform cross-validation
for i = 1:numFolds
    % Indices for training and testing sets
    trainingIndices = training(cvPartition, i);
    testingIndices = test(cvPartition, i);

    % Split the data
    trainingSet = neuralMatrix(trainingIndices, :);
    trainingLabels = behaviorID(trainingIndices);
    testingSet = neuralMatrix(testingIndices, :);
    testingLabels = behaviorID(testingIndices);

    % Train LDA model
    trainedModel = fitcdiscr(trainingSet, trainingLabels);

    % Test the model
    predictions = predict(trainedModel, testingSet);

    % Calculate accuracy for this fold
    accuracies(i) = sum(predictions == testingLabels) / length(testingLabels);
end

% Calculate the average accuracy across all folds
averageAccuracy = mean(accuracies);
fprintf('Average accuracy across 5 folds: %.2f%%\n', averageAccuracy * 100);
end






function VisualizeProjectionVectorsLDA(neuralMatrix, behaviorID, featureIndices, selectedBehaviors)
% VisualizeProjectionVectorsLDA: Trains LDA on full data set, and plots projection vectors for selected features and behaviors.
% neuralMatrix is the matrix of neural data where rows are observations and columns are features.
% behaviorID is the vector of behavior labels.
% featureIndices is a vector containing two indices of the features to be used for plotting.
% selectedBehaviors is a vector containing behaviorID values to be visualized.

% Validate inputs
if length(featureIndices) ~= 2 || length(selectedBehaviors) ~= 2
    error('Exactly two feature indices and two behaviorIDs must be provided for plotting');
end

% Train LDA model on the full data
ldaModel = fitcdiscr(neuralMatrix, behaviorID);

% Extract the coefficients for the two selected features
coeffs = ldaModel.Coeffs(1,2).Linear(featureIndices);

% Filter data for the selected behaviors
behaviorFilter = ismember(behaviorID, selectedBehaviors);
filteredData = neuralMatrix(behaviorFilter, featureIndices);
filteredBehaviorID = behaviorID(behaviorFilter);
b1Ind = filteredBehaviorID == selectedBehaviors(1);
b2Ind = filteredBehaviorID == selectedBehaviors(2);


% Plotting
colors = distinguishable_colors(16);
colors = [colors(selectedBehaviors(1),:); colors(selectedBehaviors(2),:)];
figure(92);

% First subplot: 2D scatter plot with projection line
subplot(1, 2, 1);
cla
hold on;

% Scatter plot of the filtered data
h = gscatter(filteredData(:,1), filteredData(:,2), filteredBehaviorID, 'br', '^v', 10);
% Assign custom colors to each group
for k = 1:length(h)
    set(h(k), 'Color', colors(k, :));
    set(h(k), 'LineWidth', 2);
end

% Plot the projection vectors
% Calculate the mean of the filtered data for the constant term of the line equation
meanData = mean(filteredData);
c = -coeffs(1) * meanData(1) - coeffs(2) * meanData(2);

% Define the line equation for fimplicit
lineEq = @(x, y) coeffs(1) * x + coeffs(2) * y + c;

% scale = max(std(filteredData)) * 10;
% quiver(mean(filteredData(:,1)), mean(filteredData(:,2)), coeffs(1) * scale, coeffs(2) * scale, 'k', 'LineWidth', 2, 'MaxHeadSize', 2);
% quiver(meanData(1), meanData(2), coeffs(featureIndices(1)) * scale, coeffs(featureIndices(2)) * scale, 0, 'k', 'LineWidth', 2, 'MaxHeadSize', 1);

% Plot the projection vector line using fimplicit
fimplicit(lineEq, [min(filteredData(:,1)) max(filteredData(:,1)) min(filteredData(:,2)) max(filteredData(:,2))], 'k', 'LineWidth', 2);

% Labels and title
xlabel(['Neuron ' num2str(featureIndices(1))]);
ylabel(['Neuron ' num2str(featureIndices(2))]);
title('LDA Projection Vectors for Selected Features and Behaviors');
legend(arrayfun(@(x) sprintf('Behavior %d', x), selectedBehaviors, 'UniformOutput', false));
hold off;


% Second subplot: Data projection along the LDA vector
subplot(1, 2, 2);
cla
hold on;

% Calculate projections along the LDA vector
projection = filteredData * coeffs;

% Scatter plot of the projections
scatter(projection(b1Ind), -.05 * ones(size(projection(b1Ind))), 100, filteredBehaviorID(b1Ind), 'MarkerEdgeColor', colors(1,:), 'LineWidth',2);
scatter(projection(b2Ind), .05 * ones(size(projection(b2Ind))), 100, filteredBehaviorID(b2Ind), 'MarkerEdgeColor', colors(2,:), 'LineWidth',2);

% Labels and title for the second subplot
xlabel('Projection along LDA Vector');
title('Data Projection Along LDA Vector');
yticks([]);
hold off;

end





function ldaProjectionPlot(neuralMatrix, behaviorID)
% ldaProjectionPlot: Performs LDA and plots data points in the first two LDA components.
% neuralMatrix: Matrix where each column is a neuron, each row is spike counts at different time points.
% behaviorID: Vector of categories corresponding to the time points in neuralMatrix.

% Check if the number of observations matches the number of labels
if size(neuralMatrix, 1) ~= length(behaviorID)
    error('Number of rows in neuralMatrix must match the length of behaviorID');
end

% Train LDA model
ldaModel = fitcdiscr(neuralMatrix, behaviorID);

% Get the coefficients (loadings) for the LDA components
ldaCoeff = ldaModel.Coeffs(1,2).Linear;

% Project the data onto the first two LDA components
% Note: In MATLAB, LDA typically provides one less component than the number of classes,
% so for two classes, there will be only one component.
projectedData = neuralMatrix * ldaCoeff;

% Plotting the projected data
figure(65);
scatter(projectedData, zeros(size(projectedData)), 10, behaviorID, 'filled');
xlabel('First LDA Component');
ylabel('Second LDA Component (Not Applicable for Two Classes)');
title('Projection of Neural Data onto First LDA Component');
end




