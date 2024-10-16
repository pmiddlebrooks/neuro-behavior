    

%% Confusion matrix

% Generate predictions
predictedLabels = predict(svmModel, testData);

% Confusion matrix
confusionMat = confusionmat(testLabels, predictedLabels);
% disp('Confusion Matrix:');
% disp(confusionMat);

% Plot the confusion matrix
figure(342)
confusionchart(testLabels, predictedLabels);


%% Analzyze the predictions vs observed


    % Create a maximized figure on the second monitor
    fig = figure(54); clf
    set(fig, 'Position', monitorTwo);
    nPlot = length(bhv2ModelCodes);
    [ax, pos] = tight_subplot(2, ceil(nPlot/2), [.08 .02], .1);

    edges = -.5 : 1 : bhv2ModelCodes(end)+1;
    iAccuracy = zeros(length(bhv2ModelCodes), 1);
    for iBhv = 1 : length(bhv2ModelCodes)
        iObsInd = testLabels == bhv2ModelCodes(iBhv);
        iObs = testLabels(iObsInd);
        iPred = predictedLabels(iObsInd);
        iAccuracy(iBhv) = sum(iObs == iPred) / length(iObs);
        iPredWrong = iPred(iObs ~= iPred);
        % N = histcounts(iPredWrong, edges, 'Normalization', 'pdf');
        bhv2ModelErrProp = zeros(length(bhv2ModelCodes), 1);
        for kBhv = 1 : length(bhv2ModelCodes)
        bhv2ModelErrProp(kBhv) = sum(iPredWrong == bhv2ModelCodes(kBhv)) / length(iPredWrong);
        end
        % barCodes = 0:bhv2ModelCodes(end);

        axes(ax(iBhv))
        % barH = bar(barCodes, N);
        barH = bar(bhv2ModelCodes, bhv2ModelErrProp);
        for kBar = 1:length(bhv2ModelCodes)
            barH.FaceColor = 'flat';        % Enable face color for each bar
            barH.CData(kBar,:) = bhv2ModelColors(kBar,:);  % Assign color to each bar
        end

        barH.BarWidth = 1;
        % set(findall(fig,'-property','FontSize'),'FontSize',allFontSize) % adjust fontsize to your document
        % set(findall(fig,'-property','Box'),'Box','off') % optional
        titlePos = get(ax(iBhv).Title, 'Position');  % Get the position of the title

% Add the text object at the title position with a background color

        iTitle = sprintf('%s (%d): %.2f', bhv2ModelNames{iBhv}, bhv2ModelCodes(iBhv), iAccuracy(iBhv));
        % title(iTitle, 'Color', bhv2ModelColors(iBhv,:), 'interpreter', 'none');
        % title(iTitle, 'interpreter', 'none');
text(titlePos(1), titlePos(2), iTitle, ...
    'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', ...
    'FontSize', 14, 'Color', [1 1 1], 'BackgroundColor', bhv2ModelColors(iBhv,:), 'interpreter', 'none'); % Specify the background color here
    end
    sTitle = sprintf('%s %s %s Errors due to other behaviors', selectFrom, fitType, transWithinLabel);
figure_pretty_things    
    sgtitle(sTitle)
%  set(gcf, 'PaperOrientation', 'landscape');   % Set orientation to landscape
% set(gcf, 'PaperPositionMode', 'auto');       % Adjust the paper size automatically
   print('-dpdf', fullfile(paths.figurePath, [sTitle, '.pdf']), '-fillpage')

    %% Accuracies and F-scores
    accuracyPosition = [monitorOne(1:2), monitorOne(3)/3, monitorOne(4)/2.2];
    fig = figure(55); clf;
    set(fig, 'Position', accuracyPosition);
    % barH = bar(bhv2ModelCodes, iAccuracy);
    barH = bar(bhv2ModelCodes, iAccuracy);
    for kBar = 1:length(iAccuracy)
        barH.FaceColor = 'flat';        % Enable face color for each bar
        barH.CData(kBar,:) = bhv2ModelColors(kBar,:);  % Assign color to each bar
    end
    barH.BarWidth = 1;
    xticks(bhv2ModelCodes)
    ylim([0 1])
    set(findall(fig,'-property','FontSize'),'FontSize',allFontSize) % adjust fontsize to your document
    set(findall(fig,'-property','Box'),'Box','off') % optional
    xlabel('Behavior')
    ylabel('Proportion Accurate')
    % bvhLabels = behaviors(2:end)
    % xticklabels(bhvLabels)
    % xtickangle(45);
    titleE = [selectFrom, ' ', fitType, ' ', transWithinLabel, ' Bhv Accuracies'];
    title(titleE)
    print('-dpdf', fullfile(paths.figurePath, [titleE, '.pdf']), '-bestfit')

    % F1 score
    % Initialize arrays to store precision, recall, and F1 scores for each label
    % observedLabels = svmID;
    uniqueLabels = unique(testLabels);
    precision = zeros(size(uniqueLabels));
    recall = zeros(size(uniqueLabels));
    f1Score = zeros(size(uniqueLabels));

    % Calculate F1 score for each unique label
    for iBhv = 1:length(uniqueLabels)
        label = uniqueLabels(iBhv);

        % True positives (TP)
        tp = sum((predictedLabels == label) & (testLabels == label));

        % False positives (FP)
        fp = sum((predictedLabels == label) & (testLabels ~= label));

        % False negatives (FN)
        fn = sum((predictedLabels ~= label) & (testLabels == label));

        % Precision
        if (tp + fp) > 0
            precision(iBhv) = tp / (tp + fp);
        else
            precision(iBhv) = 0;
        end

        % Recall
        if (tp + fn) > 0
            recall(iBhv) = tp / (tp + fn);
        else
            recall(iBhv) = 0;
        end

        % F1 Score
        if (precision(iBhv) + recall(iBhv)) > 0
            f1Score(iBhv) = 2 * (precision(iBhv) * recall(iBhv)) / (precision(iBhv) + recall(iBhv));
        else
            f1Score(iBhv) = 0;
        end
    end

    % % Display the results
    % resultsTable = table(uniqueLabels', precision', recall', f1Score', ...
    %                      'VariableNames', {'Label', 'Precision', 'Recall', 'F1_Score'});
    % disp(resultsTable);

    f1Position = [monitorOne(1), monitorOne(4)/2, monitorOne(3)/3, monitorOne(4)/2.2];
    fig = figure(56); clf;
    set(fig, 'Position', f1Position);
    % bar(codes(2:end), f1Score);
    % xticks(0:codes(end))
    barH = bar(uniqueLabels, f1Score);
    for kBar = 1:length(f1Score)
        barH.FaceColor = 'flat';        % Enable face color for each bar
        barH.CData(kBar,:) = bhv2ModelColors(kBar,:);  % Assign color to each bar
    end
    barH.BarWidth = 1;
    set(findall(fig,'-property','FontSize'),'FontSize',allFontSize) % adjust fontsize to your document
    set(findall(fig,'-property','Box'),'Box','off') % optional
    xticks(uniqueLabels)
    ylim([0 1])
    xlabel('Behavior')
    ylabel('F-score')
    titleE = [selectFrom, ' ', fitType, ' ', transWithinLabel, ' Bhv F1 scores'];
    title(titleE)
    print('-dpdf', fullfile(paths.figurePath, [titleE, '.pdf']), '-bestfit')
