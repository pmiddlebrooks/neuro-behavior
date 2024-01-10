%% Go to spiking_script and get behavioral and neural data

%%

% Create a neural matrix. Each column is a neuron. Each row are spike
% counts peri-onset of each behavior.
behaviorID = [];
neuralMatrix = [];
periEventTime = -.2 : opts.frameSize : .2; % seconds around onset
dataWindow = round(periEventTime(1:end-1) / opts.frameSize); % frames around onset (remove last frame)

for iBhv = 1 : length(analyzeCodes)
    iStartFrames = floor(dataBhv.bhvStartTime(dataBhv.bhvID == analyzeCodes(iBhv)) ./ opts.frameSize);
    iStartFrames = iStartFrames(3:end-3);
    behaviorID = [behaviorID; analyzeCodes(iBhv) * ones(length(iStartFrames), 1)];
    for jStart = 1 : length(iStartFrames)
        neuralMatrix = [neuralMatrix; sum(dataMatZ(iStartFrames(jStart) + dataWindow, :))];
    end
end
%%

% MdlLinear = fitcdiscr(neuralMatrix, behaviorID);

averageAccuracy = Perform5FoldCVLDA(neuralMatrix(:, idM56), behaviorID)
%%

trainedModel = train_full_lda_model(neuralMatrix(:, idM56), behaviorID)


%%
neuronIndices = [4 38];
selectedBehaviors = [2 11];
VisualizeProjectionVectorsLDA(neuralMatrix(:, idM56), behaviorID, neuronIndices, selectedBehaviors)


%%
ldaProjectionPlot(neuralMatrix(:, idM56), behaviorID)


%%
lda_project_and_plot(neuralMatrix(:, idM56), behaviorID)
%%


function lda_project_and_plot(neuralMatrix, behaviorID)
bhvList = unique(behaviorID);
colors = distinguishable_colors(length(bhvList));

% Fit the LDA model
ldaModel = fitcdiscr(neuralMatrix, behaviorID);

% Eigenvalue decomposition for the lda components
[eigenvectors, eigenvalues] = eig(ldaModel.BetweenSigma, ldaModel.Sigma);

% Sort the eigenvalues and associated eigenvectors
[eigenvalues, sortedIndices] = sort(diag(eigenvalues), 'descend');
eigenvectors = eigenvectors(:, sortedIndices);

% Project the data onto the LDA components
projectedData = neuralMatrix * eigenvectors;

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
[ha, pos] = tight_subplot(3, 3);

for bhv = 1 : length(bhvList)
    bhvIdx = behaviorID == bhvList(bhv);
    bhvColor = colors(bhv,:);
    for i = 1:9
        axes(ha(i))
        % subplot(3, 3, i);
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
            % w = std(componentX) / sqrt(length(componentX));
            % h = std(componentY) / sqrt(length(componentY));
            w = std(componentX);
            h = std(componentY);

        s = scatter(x, y, 100, bhvColor, 'filled');

         % Generate points on the ellipse
    theta = linspace(0, 2 * pi, 100); % Generate 100 points
    ellipseX = (w / 2) * cos(theta) + x; % X coordinates
    ellipseY = (h / 2) * sin(theta) + y; % Y coordinates
    
    % Plot the ellipse
    fill(ellipseX, ellipseY, bhvColor, 'FaceAlpha', 0.2); % Fill the ellipse with semi-transparency

        end
        title(['LDA Components ', num2str(i), ' and ', num2str(i+1)]);
        % xlabel(['Component ', num2str(i)]);
        % ylabel(['Component ', num2str(i+1)]);
        xlabel([]);
        ylabel([]);
        grid on;
    end
end
% Adjust subplot layout to minimize overlap
sgtitle('LDA Component Projections'); % Add a centered title if desired
% set(gcf, 'Units', 'normalized', 'OuterPosition', [0 0 1 1]); % Maximize figure within monitor

end



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




