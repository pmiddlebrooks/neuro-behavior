%% Use trained model to predict labels

% If needed, run a model to get svmProj, svmInd, and svmID (the data, the
% indices to use from the original umap projections, and the
% original/observed behavior labels.
    predictedLabels = predict(svmModel, svmProj); 



%% Error Analyses

checkBhvCode = 15;

obsCodeInd = svmID == checkBhvCode;
corrCodeInd = obsCodeInd & (predictedLabels == checkBhvCode);
errCodeInd = obsCodeInd & (predictedLabels ~= checkBhvCode);

obsCodeFrame = nan(length(svmID), 1);
obsCodeFrame(obsCodeInd) = find(obsCodeInd);
corrCodeFrame = nan(length(svmID), 1);
corrCodeFrame(corrCodeInd) = find(corrCodeInd);
errCodeFrame = nan(length(svmID), 1);
errCodeFrame(errCodeInd) = find(errCodeInd);

% sum(corrCodeInd) / (sum(obsCodeInd))
% sum(errCodeInd) / (sum(obsCodeInd))

figure(85); clf; hold on;
scatter(obsCodeFrame, ones(length(svmID), 1), '.k', 'sizeData', 100)
scatter(corrCodeFrame, 1.1 * ones(length(svmID), 1), '.b', 'sizeData', 100)
scatter(errCodeFrame, 1.2 * ones(length(svmID), 1), '.r', 'sizeData', 100)
ylim([0 3])
xlabel('Frame')
legend({'Observed', 'CorrectPred', 'ErrPred'})



%%
% Initialize variables to store the results
predValues = {};
propCorr = [];


% Find the start and end indices of consecutive non-NaN stretches
diffIndices = [false; diff(obsCodeInd) ~= 0]; % Marks the changes between NaN and non-NaN
startIndices = find(diffIndices & obsCodeInd); % Find the start of non-NaN stretches
endIndices = find(diffIndices & ~obsCodeInd) - 1; % Find the end of non-NaN stretches

% Handle case where the last stretch runs until the end of the vector
if isempty(endIndices) || (endIndices(end) < numel(data) && obsCodeInd(end))
    endIndices = [endIndices, numel(data)];
end

% Loop through each stretch and store the length and values
for i = 1:length(startIndices)
    % Get the current stretch
    currentPred = predictedLabels(startIndices(i):endIndices(i));

    % Store the values of the current stretch
    predValues{i} = currentPred;

    propCorr(i) = sum(currentPred == checkBhvCode) / length(currentPred);
end


%%
% Get bout length distribution
[boutLength, ~, idx] = unique(cellfun(@length, predValues));
counts = accumarray(idx, 1);
% Plot the distribution of counts
figure(86);
bar(boutLength, counts);
xlabel('Bout Length');
ylabel('Count');
title('Bout Length Distribution');
grid on;

%%
% monitorPositions = get(0, 'MonitorPositions');
% secondMonitorPosition = monitorPositions(size(monitorPositions, 1), :); % Just use single monitor if you don't have second one
% set(fig, 'Position', secondMonitorPosition);

figure(87); clf; hold on;
figure(88); clf; hold on;
figure(89); clf; hold on;
% [ax, pos] = tight_subplot(3, 1, [.04 .02], .1, .1);

nLengths = 8;
meanPropCorr = zeros(nLengths, 1);
for i = 1 : nLengths - 1
    % Get the distribution of proportion correct for all bout lengths of i
    iInd = cellfun(@length, predValues) == i;

    [iPropCorr, ~, idx] = unique(propCorr(iInd));
    iPropCorrDist = accumarray(idx, 1) ./ sum(accumarray(idx, 1));
    meanPropCorr(i) = mean(propCorr(iInd));
    figure(87)
    plot(iPropCorr, iPropCorrDist, '.-', 'lineWidth', 2, 'markerSize', 20)
    figure(88)
    plot(iPropCorr, cumsum(iPropCorrDist), '.-', 'lineWidth', 2, 'markerSize', 20)
end
iInd = cellfun(@length, predValues) >= nLengths;
[iPropCorr, ~, idx] = unique(propCorr(iInd));
iPropCorrDist = accumarray(idx, 1) ./ sum(accumarray(idx, 1));
meanPropCorr(nLengths) = mean(propCorr(iInd));

figure(87)
plot(iPropCorr, iPropCorrDist, '.-', 'lineWidth', 2, 'markerSize', 20)
xlabel('Proportion Correct')
ylabel('Distribution')
title('Distribution of Accuracies per Bout Length')

figure(88)
plot(iPropCorr, cumsum(iPropCorrDist), '.-', 'lineWidth', 2, 'markerSize', 20)
xlabel('Proportion Correct')
ylabel('Cumulative Distribution')
title('Cumulative Distribution of Accuracies')

figure(89)
plot(1:nLengths, meanPropCorr, 'lineWidth', 2);
xlabel('Bout Length')
ylabel('Mean Proportion Correct Frames')
title('Accuracy Per Bout Length')













