%% Get data from get_standard_data

opts = neuro_behavior_options;
opts.minActTime = .16;
opts.collectStart = 0 * 60 * 60; % seconds
opts.collectFor = 60 * 60; % seconds
opts.frameSize = .1;
% opts.shiftAlignFactor = .05; % I want spike counts xxx ms peri-behavior label

opts.minFiringRate = .5;

getDataType = 'spikes';
get_standard_data
% getDataType = 'behavior';
% get_standard_data

[dataBhv, bhvID] = curate_behavior_labels(dataBhv, opts);

if opts.frameSize > .1
    error('Ridge Regression on this data doesn"t like frames larger than .1 s')
end

% for plotting consistency
%
monitorPositions = get(0, 'MonitorPositions');
if exist('/Users/paulmiddlebrooks/Projects/', 'dir')
    monitorPositions = flipud(monitorPositions);
end
monitorOne = monitorPositions(1, :); % Just use single monitor if you don't have second one
monitorTwo = monitorPositions(size(monitorPositions, 1), :); % Just use single monitor if you don't have second one




%% =============================================================
%                  Behavior: Design Matrix
% =============================================================
% What window around onset times do you want to use as regressors?
opts.mPreTime = .2;
opts.mPostTime = .5;
regWindow = int8(-opts.mPreTime/opts.frameSize : opts.mPostTime/opts.frameSize - 1);
regWindowMat = repmat(regWindow, 1, length(analyzeBhv));
%%

% Get design matrix for behaviors
[bhvDesign, regressorLabels, regressorBhv] = rrm_design_matrix_behavior(dataBhv, opts);


frameIdx = 1 : size(bhvDesign, 1);


% If you want, Get one-back design matrix
% vector of one-back behaviors
dataBhv.prevBhvID = [nan; dataBhv.ID(1:end-1)];
[oneBackDesign, oneBackLabels] = rrm_design_matrix_one_back(dataBhv, opts);


% Remove columns without valid regressors (and remove their labels)
% Columns get removed if:
%   - a behavior is not listed in the behaviors we want to analyze (e.g. nest/irrelevant)
%   - there aren't enough behavior bouts to analyze for that behavior, as set in opt.minBhvNum

% Get rid of in-nest/sleeping:
bhvDesign(:, regressorBhv == -1) = [];
regressorLabels(regressorBhv == -1) = [];
regressorBhv(regressorBhv == -1) = [];


rmBhvDesignCol = sum(bhvDesign, 1) == 0;
bhvDesign(:, rmBhvDesignCol) = [];
regressorLabels(rmBhvDesignCol) = [];


% Remove rows without any regressors
% % (happens a lot, e.g. when a behavior lasts longer than the time
% window we're interested in analyzing)
removeRows = sum(bhvDesign, 2) == 0;

frameIdx(removeRows) = [];
bhvDesign(removeRows,:) = [];
oneBackDesign(removeRows,:) = [];



% After removing any rows for which bhvDesign didn't have regressors, need to
% again remove any zero-regressor columns from oneBackDesign (b/c some of
% the columns only had one-back regressors when there is no regressor data
% to analyze in bhvDesign)
rmOneBackCol = sum(oneBackDesign, 1) == 0;
oneBackLabels(rmOneBackCol) = [];
oneBackDesign(:, rmOneBackCol) = [];

figure();
imagesc([bhvDesign])
ylabel('Time Bins')
xlabel('Regressors')
title('Regression Design Matrix')
figure_pretty_things

% Finally, if removing rows also removed too many one-back regressors, get rid of those

oneBackBelowMin = sum(oneBackDesign, 1) < opts.minOneBackNum;
oneBackDesign(:, oneBackBelowMin) = [];
oneBackLabels(oneBackBelowMin)= [];






% Make a full Regression Design matrix for the model (choose what to include);

% fullR = [bhvDesign oneBackDesign];
% regLabels = [regressorLabels, oneBackLabels];
fullR = [bhvDesign];
regLabels = [regressorLabels];
% fullR = [bhvDesign(:,1:24)];
% regLabels = [regressorLabels(1:24)];



%% run QR and check for rank-defficiency. This will show whether a given regressor is highly collinear with other regressors in the design matrix.

% From tutorial_linearModel.m

% The resulting plot ranges from 0 to 1 for each regressor, with 1 being
% fully orthogonal to all preceeding regressors in the matrix and 0 being
% fully redundant. Having fully redundant regressors in the matrix will
% break the model, so in this example those regressors are removed. In
% practice, you should understand where the redundancy is coming from and
% change your model design to avoid it in the first place!

rejIdx = false(1,size(fullR,2));
[~, fullQRR] = qr(bsxfun(@rdivide,fullR,sqrt(sum(fullR.^2))),0); %orthogonalize normalized design matrix
figure; plot(abs(diag(fullQRR)),'linewidth',2); ylim([0 1.1]); title('Regressor orthogonality'); drawnow; %this shows how orthogonal individual regressors are to the rest of the matrix
axis square; ylabel('Norm. vector angle'); xlabel('Regressors');
if sum(abs(diag(fullQRR)) > max(size(fullR)) * eps(fullQRR(1))) < size(fullR,2) %check if design matrix is full rank
    temp = ~(abs(diag(fullQRR)) > max(size(fullR)) * eps(fullQRR(1)));
    fprintf('Design matrix is rank-defficient. Removing %d/%d additional regressors.\n', sum(temp), sum(~rejIdx));
    rejIdx(~rejIdx) = temp; %reject regressors that cause rank-defficint matrix
end







%% Neural
% Which data to model:


selectFrom = 'M23';
selectFrom = 'M56';
% selectFrom = 'DS';
% selectFrom = 'VS';
% selectFrom = 'M56 DS';
% selectFrom = 'DS VS';
% selectFrom = 'All';
switch selectFrom
    case 'M23'
        idSelect = idM23;
        figHFull = 260;
        figHModel = 270;
        figHFullModel = 280;
    case 'M56'
        idSelect = idM56;
    case 'DS'
        idSelect = idDS;
    case 'M56 DS'
        idSelect = [idM56, idDS];
    case 'DS VS'
        idSelect = [idDS, idVS];
    case 'VS'
        idSelect = idVS;
    case 'All'
        idSelect = cell2mat(idAll);
end
% idSelect = [idM23 idM56];


% =============================================================
%                  Neural: Neural Matrix
% =============================================================


% remove the frames you removed from the design matrix
dataMatZ = zscore(dataMat(~removeRows,:), 0, 1);

% Establish the datamat you want to model (which area(s))?
dataMatModel = dataMatZ(:,idSelect);
% dataMatModel = dataMatZ(:,idSelect(end-13:end-9));






% =============================================================
%                  Run the regression and get standard stats
% =============================================================

%{

% Run the regression and get the neural predictions
tic
[ridgeVals, dimBeta] = ridgeMML(dataMatModel, fullR, true); %get ridge penalties and beta weights.
toc

% yModel = (fullR * dimBeta);
yModel = (fullR - mean(fullR, 1)) * dimBeta;

% %% Explained variance
% % R squared
% % = (SSE / SSE + SSRes)
%
SSRes = sum((dataMatModel(:) - yModel(:)) .^2);
SSE = sum((dataMatModel(:) - mean(dataMatModel(:))) .^2);
rSquaredFull = 1 - (SSRes / SSE);
rSquaredFull

% %%
% % F statistic
% % = (SSRes / numRegressors) / (SSE / (numDataPoints - numRegressors - 1))
%
% dfNum = size(fullR, 2);
% dfDenom = (size(dataMat, 1) - size(fullR,2) - 1);
%
% Fstat = (SSRes / dfNum) / (SSE / dfDenom)
% % dfModel = size(fullR, 2);
% % dfTotal = size(dataMat, 1) - 1;
% % dfRedisdual = dfTotal - dfModel;
%
% % Calculate the p-value
% pVal = 1 - fcdf(Fstat, dfNum, dfDenom)


%
% R squared per neuron
SSMod = sum((yModel - mean(dataMatModel, 1)) .^2, 1);
SSRes = sum((dataMatModel - yModel) .^2, 1);
SSTot = sum((dataMatModel - mean(dataMatModel, 1)) .^2, 1);

% degFM = size(fullR, 2) - 1; % degrees of freedom of the model
% msm = SSMod ./ degFM;
% degFE = size(dataMatModel, 1) - size(fullR, 2);
% mse = SSRes ./ degFE;
%
% Fstat = msm ./ mse; % Model error / Residuals error
% % Calculate the p-value
% pVal = 1 - fcdf(Fstat, dfNum, dfDenom);


% Calculate Explained Variance
explained_variance = SSTot - SSRes;

% Calculate R-squared
rSquaredFull = explained_variance ./ SSTot;

% Calculate Degrees of Freedom (Residual)
df_residual = size(dataMatModel, 1) - size(fullR, 2);

% Calculate Degrees of Freedom (Explained)
df_explained = size(fullR, 2);

% Calculate F-statistic (mean square error of model / mean square error or
% residuals)
% fStat = (explained_variance ./ df_explained) ./ (SSRes ./ df_residual);
fStat = (SSMod ./ df_explained)  ./ (SSRes ./ (df_residual));
pVal = 1 - fcdf(fStat, df_explained, df_residual);

%}




% ridge_cross_validation - Perform n-fold cross-validation to calculate R-squared, F-statistics, and p-values.
%
% Inputs:
% dataMatModel - Neural data matrix (nSamples x pNeurons)
% fullR        - Full design matrix (nSamples x rRegressors)
% nFolds       - Number of cross-validation folds
%
% Outputs:
% R2       - R-squared values for each neuron (1 x pNeurons)
% Fstat    - F-statistic values for each neuron (1 x pNeurons)
% pValues  - p-values for the F-statistics for each neuron (1 x pNeurons)

nFolds = 3;

% Dimensions
[nSamples, pNeurons] = size(dataMatModel);

% Preallocate storage for metrics
R2 = zeros(1, pNeurons);
Fstat = zeros(1, pNeurons);
pValues = zeros(1, pNeurons);

% Create cross-validation indices
cv = cvpartition(nSamples, 'KFold', nFolds);

% Initialize arrays to store metrics for each fold
R2_folds = zeros(nFolds, pNeurons);
Fstat_folds = zeros(nFolds, pNeurons);
dimBeta = zeros(size(fullR, 2), pNeurons, nFolds);

% Loop through each fold
for foldIdx = 1:nFolds
    % Get train and test indices
    trainIdx = training(cv, foldIdx);
    testIdx = test(cv, foldIdx);

    % Split data into training and testing sets
    trainR = fullR(trainIdx, :);
    testR = fullR(testIdx, :);
    trainData = dataMatModel(trainIdx, :);
    testData = dataMatModel(testIdx, :);

    % Train ridge regression on training data
    [~, dimBeta(:,:,foldIdx)] = ridgeMML(trainData, trainR, true);

    % Predict on test data
    yPred = (testR - mean(testR, 1)) * dimBeta(:,:,foldIdx);

    % Compute residuals and variance for test data
    residuals = testData - yPred;
    SSR = sum(residuals .^ 2, 1); % Residual sum of squares error w.r.t. model)
    SST = sum((testData - mean(testData, 1)) .^ 2, 1); % Total sum of squares

    % Calculate R-squared for each neuron
    R2_folds(foldIdx, :) = 1 - (SSR ./ SST);

    % Compute F-statistic for each neuron
    dfFull = size(trainR, 2) - 1; % Degrees of freedom for the full model
    dfRes = size(testData, 1) - dfFull - 1; % Residual degrees of freedom
    Fstat_folds(foldIdx, :) = ((SST - SSR) ./ dfFull) ./ (SSR ./ dfRes);
end

% Average weights across folds
betas = mean(dimBeta, 3);

% Average metrics across folds
rSquaredFull = mean(R2_folds, 1);
fStat = mean(Fstat_folds, 1);
pVal = 1 - fcdf(fStat, dfFull, dfRes);

% Neurons that have significant F-stats for the whole model:
sigNeuronIdx = pVal < .05;



plotFlag = 1;
savePlot = 1;
if plotFlag
    fig = figure(23); clf
    [ax, pos] = tight_subplot(3, 1, [.08 .02], .1);

    axes(ax(1))
    scatter(1:size(dataMatModel, 2), rSquaredFull, 100, 'k', 'filled')
    % scatter(ax(1), 1:size(dataMatModel, 2), rSquaredFull, 100, 'k', 'filled')

    % xline(.5 + length(idM23), 'LineWidth', 3)
    % xline(.5 + length(idM23) + length(idM56), 'LineWidth', 3)
    % xline(.5 + length(idM23) + length(idM56) + length(idDS), 'LineWidth', 3)
    % title(ax(1), 'R-squared for each neuron')
    % xlabel(ax(1), 'M23 -> M56 -> DS -> VS')
    title(['R-squared for each neuron ', selectFrom])
    % xlabel('M23 -> M56 -> DS -> VS')
    xlabel(['Neurons ', selectFrom])
    ylabel('R-squared')

    axes(ax(2))
    scatter(1:size(dataMatModel, 2), fStat, 100, 'k', 'filled')
    ylabel('Fstat')
    axes(ax(3))
    scatter(1:size(dataMatModel, 2), pVal, 100, 'k', 'filled')
    ylabel('P-val')

    figure_pretty_things
    if savePlot
        % saveas(figureHandle,fullfile(figurePath, 'r-Squared per neuron'), 'pdf')
        print('-dpdf', fullfile(paths.figurePath, ['r-Squared per neuron ', selectFrom, '.pdf']), '-bestfit')
    end
end






%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%               Shuffle F-Stat to determine which neurons are significant
%               across all regressors (behaviors)
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
optsShuffle.nIter = 1000;
[fStat, fStatShuffle, fStatZ] = rrm_fstat_shuffled(fullR, dataMatModel, optsShuffle);

%
figure(28); clf; hold on
scatter(1:length(fStatZ), fStatZ, 'filled')
yline(2)
ylabel('Fstat Z-score wrt Shuffled')
xlabel('Neurons')
% title('Fstats per neuron across all behaviors')
% xline(.5 + length(idM23), 'LineWidth', 3)
% xline(.5 + length(idM23) + length(idM56), 'LineWidth', 3)
% xline(.5 + length(idM23) + length(idM56) + length(idDS), 'LineWidth', 3)

titleM = ['Fstat across behaviors ', selectFrom];
title(titleM)
% saveas(gcf, fullfile(paths.figurePath, [titleM, '.png']), 'png')
figure_pretty_things
print('-dpdf', fullfile(paths.figurePath, [titleM, '.pdf']), '-bestfit')


propSig = sum(fStatZ > 2) / length(fStatZ);
fprintf('%s %.3f significant neurons\n', selectFrom, propSig)



% copy_figure_to_clipboard

















%% For each neuron:
% - examine which regressors are significant (F-stat on partial models). Thus, the pattern of which
% behaviors are important in general, and which time bins relative to
% behavior onset are important.
% - examine how much each regressor contributes relative to the other
% regerssors (relative R-squared)



%%
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 1. Whether each regressor contributes (is important)?
% Use F-statistic of partial models
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% For each regressor, remove it
% Run regression without it
% Run 1000 shuffled regressions without it (shuffle the start times of each behavior).
% Ask whether F-statisic of model is different than distribution of F-stats of shuffled model
% For each area, across behaviors, when does F-statistic differ? Is this time different within different areas?

%% Test: Nested F-stats for partial models


optsNest.nIter = 10;
tic
fStatZPartial = zeros(length(regLabels), size(dataMatModel, 2));
for iReg = 1 % : 2 %length(regLabels)

    fStatZPartial(iReg, :) = rrm_nested_fstat_shuffled(fullR, dataMatModel, iReg, optsNest);
    % fStatZPartial(iReg, :) = rrm_nested_fstat_shuffled(fullR, dataMatModel(:,1:10), iReg, optsNest);
    toc
end % iReg

%% Nested F-stats for partial models

optsNest.nIter = 300;





fStatZPartial = zeros(length(regLabels), size(dataMatModel, 2));
nReg = length(regLabels);
% nReg = 60

poolID = parpool(4, 'IdleTimeout', Inf);
tic
parfor iReg = 1 : length(regLabels)
% tic
regLabels{iReg}
% for iReg = 1 : 1

    % optsNest.removeReg = iReg;

    fStatZPartial(iReg, :) = rrm_nested_fstat_shuffled(fullR, dataMatModel, iReg, optsNest);
    % toc
end % iReg
toc
delete(poolID)

%% Relative R-Squared
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  2. How much each regressor contributes
%     Relative explained variance of time regressors for each behavior.
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%  For a given behavior, calculate relative explained variance of each time
%  regressor relative to all other time regressors Within That Behavior.
%  End up with, for each time regressor, a distribution of relative explained variance for each brain
%  area.
%  Ask at what point (for which time regressor) do the distributions
%  of relative explained variances differ (and in which direction).

tic
% dataMatModel
opts.withinOrAcrossBhv = 'within';
[rSquaredPartial, relativeExplained] = rrm_relative_r_squared(fullR, regLabels, dataMatModel, analyzeBhv, opts);
toc



fileName = [selectFrom, ' partial ridge.mat'];
save(fullfile(paths.saveDataPath , fileName), 'fStatZPartial', 'rSquaredPartial', 'relativeExplained', 'regLabels', '-mat')
copyfile(fullfile(paths.saveDataPath , fileName), fullfile(paths.dropPath, fileName))
copyfile(fullfile(paths.dropPath , fileName), fullfile(paths.saveDataPath, fileName))









%%      ====================      ANALYSES WITH PARTIAL F-STAT AND RELATIVE R-SQUARED    ====================
selectFrom = 'DS';
fileName = [selectFrom, ' partial ridge.mat'];
load(fullfile(paths.saveDataPath , fileName))

colors = colors_for_behaviors(analyzeCodes);


%% For each behvior:
% Proportion of neurons where -1 is peak nested F-stat
% Ratio of pre- to post- nested F-stat

fig = figure(78); clf
set(fig, 'Position', monitorOne);
nPlot = length(analyzeBhv);
[ax, pos] = tight_subplot(2, ceil(nPlot/2), [.08 .02], .1);
bhvFrameRange = - opts.mPreTime/opts.frameSize : opts.mPostTime/opts.frameSize - 1;

for iBhv = 1 : length(analyzeBhv)
    iReg = find(strcmp(['0 ',analyzeBhv{iBhv}], regLabels)); % find regressor at start of behavior
    iRange = iReg + bhvFrameRange;

    [~, maxIndices] = max(fStatZPartial(iRange,:), [], 1);
    pMaxFrame = zeros(length(iRange), 1);
    for j = 1 : length(iRange)
        pMaxFrame(j) = sum(maxIndices == j) / size(fStatZPartial, 2);
    end
    axes(ax(iBhv))
    plot(bhvFrameRange, pMaxFrame, '--.', 'color', colors(iBhv,:), 'lineWidth', 2, 'markersize', 50)
    xline(-.5, 'color', [.4 .4 .4], 'lineWidth', 4)
    title(analyzeBhv{iBhv}, 'Interpreter','none')
    xlim([-2.5 4.5])
    ylim([0 .5])
    if iBhv == 1
        ylabel('Prop. Neurons')
        xlabel('Frames w.r.t. Onset')
    end
end
% sgtitle('Peak Partial fStat in each Behavior')
sgtitle([selectFrom, ' Relative Importance of Each Time Bin Around Onset'])
figure_pretty_things
% copy_figure_to_clipboard

%%
        print('-dpdf', fullfile(paths.figurePath, ['testSave.pdf']), '-bestfit')

        %%
        disp('hwer')

%% Compare nested partial f-stats at different frames relative to behavior onset

fig = figure(79); clf
set(fig, 'Position', monitorTwo);
nPlot = length(analyzeBhv);
[ax, pos] = tight_subplot(2, ceil(nPlot/2), [.08 .02], .1);



prePostCompare = cell(length(analyzeBhv), size(fStatZPartial, 2));
prePostRatio = cell(length(analyzeBhv), size(fStatZPartial, 2));
for iBhv = 1 : length(analyzeBhv)
       iReg = find(strcmp(['0 ',analyzeBhv{iBhv}], regLabels)); % find regressor at start of behavior
       preF = fStatZPartial(iReg-1, :);
       postF = mean(fStatZPartial(iReg:iReg+1,:), 1);
       % preF = betas(iReg-1, :);
       % postF = mean(betas(iReg:iReg+1,:), 1);
       useNeurons = preF > 1 | postF > 1;
       prePostCompare{iBhv} = preF(useNeurons) - postF(useNeurons);
       preF(preF<0) = 0;
       postF(postF<0) = 0;
       prePostRatio{iBhv} = preF(useNeurons) ./ (preF(useNeurons) + postF(useNeurons));

           axes(ax(iBhv))
       edges = linspace(0, 1, 11);
histogram(prePostRatio{iBhv}, edges, 'Normalization','probability', 'FaceColor', colors(iBhv,:))
xline(.5, '--', 'lineWidth', 2)
ylim([0 .5])
    title(analyzeBhv{iBhv}, 'Interpreter','none')
if(iBhv == 1)
    ylabel('Proportion neurons')
    xlabel('Pre / (Pre+Post)')
end
end

sgtitle([selectFrom, ' Partial F-stat Pre- vs Post- onset (pre / (pre + post)'])
% sgtitle([selectFrom, ' Betas Pre- vs Post- onset (pre / (pre + post)'])
% copy_figure_to_clipboard
% for iBhv = 1 : length(analyzeBhv)
% [N,edges] = histcounts(prePostCompare(iBhv,:));
% 
%     axes(ax(iBhv))
% histogram(prePostCompare{iBhv}, edges, 'Normalization','probability')
% xline(0, 'lineWidth', 2)
% end

























%%
figurePath = strcat(paths.figurePath, animal, '/', sessionSave, '/', string(datetime('today', 'format', 'yyyy-MM-dd')), '/figures/', ['start ' num2str(opts.collectStart), ' for ', num2str(opts.collectFor)]);
if ~exist(figurePath, 'dir')
    mkdir(figurePath);
    % addpath(figurePath)
end

for iBhv = 1 : length(analyzeBhv)
    for iReg = 1 : 20
        fStatZInd = 20 * (iBhv-1) + iReg;  % which regressors belong to this behavior?
        % clf


        % plot(fStatZ(iReg,:), 'LineWidth', 2)
        head = scatter((1:size(dataMatModel, 2)), fStatZ(fStatZInd,:), 100, 'k', 'filled');
        hold on
        bhvTitle = [analyzeBhv{iBhv}, ' ', num2str(iReg)];
        title(bhvTitle, 'Interpreter', 'none')
        xline(m56IdxStart, 'LineWidth', 2)
        xline(dsIdxStart, 'LineWidth', 2)
        xline(vsIdxStart, 'LineWidth', 2)
        plot(xlim, [-3 -3])

        m23Prop = sum(fStatZ(fStatZInd, idM23) <= -3) / length(idM23);
        m56Prop = sum(fStatZ(fStatZInd, idM56) <= -3) / length(idM56);
        dsProp = sum(fStatZ(fStatZInd, idDS) <= -3) / length(idDS);
        vsProp = sum(fStatZ(fStatZInd, idVS) <= -3) / length(idVS);
        fprintf('\nM23: %.2f\nM56: %.2f\nDS: %.2f\nVS: %.2f\n', m23Prop, m56Prop, dsProp, vsProp)

        drawnow;

        % Capture the frame
        frames(iReg) = getframe(gcf);


        % Pause to control the speed (optional)
        pause(.5);
        delete(head)

        % pause(10)
    end

    %     % Define the filename for the output video
    % outputVideoFile = fullfile(figurePath, ['f-stat per behavior time regressors ', analyzeBhv{iBhv}, '.mp4']);
    %
    % % Create a VideoWriter object
    % writerObj = VideoWriter(outputVideoFile, 'MPEG-4');
    % writerObj.FrameRate = 1;  % Set the frame rate (frames per second)
    %
    % % Open the video file for writing
    % open(writerObj);
    % % Write the frames to the video file
    % writeVideo(writerObj, frames);
    %
    % % Close the video file
    % close(writerObj);
end
% toc

%%
% Which neurons were significant throughout each behavior?
pctSigCriterion = .5; % how many time frames within each behavior is the cutoff for a "significant" neuron?
nSigCriterion = ceil(pctSigCriterion * opts.windowSize / opts.frameSize);

allSigNrnInd = zeros(length(analyzeBhv), size(fStatZ, 2));
sumSigNrnPerBhv = zeros(length(analyzeBhv), size(fStatZ, 2));
for iBhv = 1 : length(analyzeBhv)


    iFirstRegInd = 20 * (iBhv-1) + 1; % which regressors belong to this behavior?
    iLastRegInd = iFirstRegInd + 19;

    % For how many time regressors in each behavior was each neuron significant
    sumSigNrnPerBhv(iBhv, :) = sum(fStatZ(iFirstRegInd : iLastRegInd, :) <= -3, 1);
    % Get indices of significant neurons
    allSigNrnInd(iBhv, :) = sumSigNrnPerBhv(iBhv, :) >= nSigCriterion;


end

sigNeurons = sum(allSigNrnInd, 1) > 0; % B/c all neurons are either significan across all behaviors or no behaviors
sigM23 = sigNeurons & strcmp(areaLabels, 'M23');
sigM56 = sigNeurons & strcmp(areaLabels, 'M56');
sigDS = sigNeurons & strcmp(areaLabels, 'DS');
sigVS = sigNeurons & strcmp(areaLabels, 'VS');
% m23SigNrnInd = fStatZ(fStatZInd, idM23) <= -3;
% m56SigNrnInd = fStatZ(fStatZInd, idM56) <= -3;
% dsSigNrnInd = fStatZ(fStatZInd, idDS) <= -3;
% vsSigNrnInd = fStatZ(fStatZInd, idVS) <= -3;









%% Nested F-stats for partial models: Use only M56 neural data and only 3 regressors per behavior
tic
useReg = find(contains(regLabels, '-5 ') | (contains(regLabels, '0 ') & ~contains(regLabels, '10 ')) | contains(regLabels, '4 '));
useReg = find(contains(regLabels, '-10 ') | (contains(regLabels, '0 ') & ~contains(regLabels, '10 ')));
m56Design = bhvDesign(:, [1: 40]);

keepRows = sum(m56Design, 2) > 0;
m56Design = m56Design(keepRows, :);

m56Data = dataMatZ(keepRows, strcmp(areaLabels, 'M56'));% | strcmp(areaLabels, 'DS'));
m56Data = zscore(dataMat(keepRows, strcmp(areaLabels, 'M56')), 0, 1);% | strcmp(areaLabels, 'DS'));


[ridgeVals, dimBeta] = ridgeMML(m56Data, m56Design, true); %get ridge penalties and beta weights.

% yModel = (fullR * dimBeta);
yModel = (m56Design - mean(m56Design, 1)) * dimBeta;

% R squared per neuron
SSMod = sum((yModel - mean(m56Data, 1)) .^2, 1);
SSRes = sum((m56Data - yModel) .^2, 1);
SSTot = sum((m56Data - mean(m56Data, 1)) .^2, 1);
toc
%%
tic
optsNest.nIter = 1000;



fStatZ = zeros(size(m56Design, 2), size(m56Data, 2));

% poolID = parpool(4, 'IdleTimeout', Inf);
parfor iReg = 1 : size(m56Design, 2)
    % for iReg = 1 : 1

    % optsNest.removeReg = iReg;

    fStatZ(iReg, :) = rrm_nested_fstat_shuffled(m56Design, m56Data, SSRes, iReg, optsNest);

end % iReg
delete(poolID)

toc / 60






























%%
relativeExplained = relativeExplainedM56;
rSquaredPartial = rSquaredPartialM56;
fStatZPartial = fStatZPartialM56;

bhvFrameRange = - opts.mPreTime/opts.frameSize : opts.mPostTime/opts.frameSize - 1;
iNrn = 1;
figure(61); clf; hold on;
for iBhv = 1 : length(analyzeBhv)
    iReg = find(strcmp(['0 ',analyzeBhv{iBhv}], regLabels)); % find regressor at start of behavior
    iRange = iReg + bhvFrameRange;
    subplot(2,1,1)
    % plot(fStatZPartial(iRange, iNrn));
    plot(mean(fStatZPartial(iRange, :), 2), '-o');
    yline(2, '--')
    xline(2.5, 'linewidth', 2)
    xticklabels(bhvFrameRange)
    title(analyzeBhv{iBhv}, 'Interpreter','none')
    subplot(2,1,2)
    % plot(relativeExplained(iRange, iNrn));
    plot(mean(relativeExplained(iRange, :), 2), '-o');
    xline(2.5, 'linewidth', 2)
    xticklabels(bhvFrameRange)

end

































%%
% For each behavior, Plot the mean relative contributions of neurons in
% each brain area, along the time course of the behavior

bhvReg = (0 : 20 : 300);
zBeta = cell(length(bhvReg), 1);
figureHandle = figure(55);
clf
firstIdx = nan(size(zBeta, 1), size(dimBeta, 2));

hold on;
for iBhv = 1 : length(bhvReg)
    iBhv
    iRegSection = relativeExplained(1 + bhvReg(iBhv) : 20 + bhvReg(iBhv), :);
    % iRegSection = abs(iRegSection);


    if plotFlag
        meanM23Rel = mean(iRegSection(:,idM23), 2);
        plot(meanM23Rel, 'r', 'lineWidth', 3)
        meanM56Rel = mean(iRegSection(:,idM56), 2);
        plot(meanM56Rel, 'm', 'lineWidth', 3)
        meanDSRel = mean(iRegSection(:,idDS), 2);
        plot(meanDSRel, 'b', 'lineWidth', 3)
        meanVSRel = mean(iRegSection(:,idVS), 2);
        plot(meanVSRel, 'c', 'lineWidth', 3)

        legend('M23','M56','DS','VS')
        title(analyzeBhv{iBhv}, 'Interpreter', 'none')
        % plot(xlim, [0 0], '--k')
        if savePlot
            saveas(figureHandle,fullfile(figurePath, ['relative contributions ', analyzeBhv{iBhv}]), 'pdf')
            pause(15)
        end
        clf
        hold on
    end
end























%% Relative contributions
% For each neuron, how much did each regressor contribute to the explained
% variance?

% relContribution = zeros(size(fullR, 2), size(dataMat, 2)); % matrix with dimensions #regressors X #neurons
normDiff = zeros(size(fullR, 2), size(dataMat, 2));
for iReg = 1 : size(fullR, 2)

    % Remove a regressor
    removeReg = zeros(size(fullR, 2), 1);
    removeReg(iReg) = 1;

    % Run regression with partial model
    partialR = fullR(:, ~removeReg);
    [ridgeVals, dimBetaPartial] = ridgeMML(dataMatModel, partialR, true); %get ridge penalties and beta weights.

    % Predict neural data with partial model
    yModelPartial = (partialR * dimBetaPartial);

    % Get explained variance of partial model
    SSRes = sum((dataMatModel - yModelPartial) .^2, 1);
    SSE = sum((dataMatModel - mean(dataMatModel)) .^2, 1);
    rSquaredPartial = 1 - SSRes ./ SSE;



    %    cur_R2_diff = (full_R2_vec(cellctr,1) - partial_R2_vec(cellctr,:))/full_R2_vec(cellctr,1);

    % Normalized difference between full model and partial model for this
    % regressor: How many "full model units" better or worse is full model relative to partial
    % model
    iNormDiff = (rSquaredFull - rSquaredPartial) ./ rSquaredFull;
    iNormDiff(iNormDiff < 0) = 0;
    iNormDiff(iNormDiff > 1) = 1;
    %
    normDiff(iReg, :) = iNormDiff;
end


relContribution = normDiff ./ sum(normDiff, 1);


%%
% Plot relative contributions of regressors for each neuron
% hold on
bhvAlign = (10 : 20 : 300);
for i = 1 : size(dataMat, 2)
    plot(relContribution(:,i), 'lineWidth', 2)
    hold on
    ylim([0 .16])
    title(rSquared(i))
    for j = 1 : length(bhvAlign)
        plot([bhvAlign(j), bhvAlign(j)], ylim, '-k');  % Vertical line at x_value
    end
    pause
    clf
end
%%
clf
hold on

meanRel = mean(relContribution, 2);
plot(meanRel)

for j = 1 : length(bhvAlign)
    plot([bhvAlign(j), bhvAlign(j)], ylim, '-k');  % Vertical line at x_value
end
































%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% To compare when different brain areas become important:

% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 0. Compare betas over time for each behavior, comparing across brain
% areas (when do betas in each brain increase/decrease first, last, etc?)
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%% For each behavior, plot the beta values for each neuron

bhvReg = (1 : opts.framesPerTrial : size(bhvDesign, 2));
zBeta = cell(length(bhvReg), 1);
firstIdx = nan(length(bhvReg), size(dimBeta, 2));

figureHandle = figure(55);


hold on;
plotThis = 'mean'; % 'mean' 'single' 'zscore'

for iBhv = 1 : length(bhvReg)
    iBhv
    iRegSection = dimBeta(bhvReg(iBhv) : 19 + bhvReg(iBhv), :);
    maxBeta = max(iRegSection(:));
    minBeta = min(iRegSection(:));
    % iRegSection = abs(iRegSection);

    switch plotThis

        case 'mean'
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % mean beta plots
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

            if plotFlag


                meanM23Beta = mean(iRegSection(:,idM23), 2);
                % meanM23Beta = mean(iRegSection(:,sigM23), 2);
                % meanM23Beta = mean(iRegSection(:, idM23(logical(relExpSig(iBhv,idM23)))), 2);
                plot(meanM23Beta, 'r', 'lineWidth', 3)

                meanM56Beta = mean(iRegSection(:,idM56), 2);
                % meanM56Beta = mean(iRegSection(:,sigM56), 2);
                % meanM56Beta = mean(iRegSection(:, idM56(logical(relExpSig(iBhv,idM56)))), 2);
                plot(meanM56Beta, 'm', 'lineWidth', 3)

                meanDSBeta = mean(iRegSection(:,idDS), 2);
                % meanDSBeta = mean(iRegSection(:,sigDS), 2);
                % meanDSBeta = mean(iRegSection(:, idDS(logical(relExpSig(iBhv,idDS)))), 2);
                plot(meanDSBeta, 'b', 'lineWidth', 3)

                meanVSBeta = mean(iRegSection(:,idVS), 2);
                % meanVSBeta = mean(iRegSection(:,sigVS), 2);
                % meanVSBeta = mean(iRegSection(:, idVS(logical(relExpSig(iBhv,idVS)))), 2);
                plot(meanVSBeta, 'c', 'lineWidth', 3)

                legend('M23','M56','DS','VS')
                title(analyzeBhv{iBhv}, 'Interpreter', 'none')
                plot(xlim, [0 0], '--k')
                if savePlot
                    % saveas(figureHandle,fullfile(figurePath, ['meanBetas neurons with high rel expl ', analyzeBhv{iBhv}]), 'pdf')
                    saveas(figureHandle,fullfile(figurePath, ['meanBetas all neurons ', analyzeBhv{iBhv}]), 'pdf')
                end
                clf
                hold on
            end

        case 'single'
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % Single neuron beta plots
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            for jNrn = 1 : size(iRegSection, 2)
                plot(iRegSection(:,jNrn), 'linewidth', 2)
                hold on
                plot([0; diff(iRegSection(:,jNrn))], '--k','linewidth', 2)
                ylim([minBeta maxBeta])
                plot(xlim, [0 0], '--k')
                iLabel = areaLabels{jNrn};
                title([analyzeBhv{iBhv} ' ' iLabel], 'interpreter', 'none')

                if savePlot
                    saveas(figureHandle,fullfile(figurePath, ['betas ', analyzeBhv{iBhv}, ' ', iLabel, ' ',num2str(jNrn)]), 'pdf')
                end

                iRegSection(:,jNrn)
                % pause(10)
                clf
            end


        case 'zscore'
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % Z-scored single neuron beta plots
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

            % zBeta{iBhv} = (iRegSection - mean(dimBeta, 1)) ./ std(dimBeta, 1);
            zBeta{iBhv} = (iRegSection - mean(iRegSection, 1)) ./ std(iRegSection, 1);

            beyondStd1 = zBeta{iBhv} >= 1 | zBeta{iBhv} <= -1;

            for jNrn = 1 : size(iRegSection, 2)

                nonzeroRows = find(beyondStd1(:, jNrn) ~= 0);
                if ~isempty(nonzeroRows)
                    % Find the first two consecutive numbers
                    for k = 1:(length(nonzeroRows) - 1)
                        if nonzeroRows(k) == nonzeroRows(k + 1) - 1
                            firstConsecutiveNumber = nonzeroRows(k);
                            firstIdx(iBhv, jNrn) = firstConsecutiveNumber;

                            break;  % Exit the loop when a pair is found
                        end
                    end

                end

                plot(zBeta{iBhv}(:,jNrn))
                hold on
                if ~isnan(firstIdx(iBhv, jNrn))
                    plot([firstIdx(iBhv, jNrn) firstIdx(iBhv, jNrn)], [-1 1], 'k')
                end
                plot(xlim, [0 0], '--k')
                iLabel = areaLabels{jNrn};
                title([analyzeBhv{iBhv} ' ' iLabel], 'interpreter', 'none')
                zBeta{iBhv}(:,jNrn)
                clf
            end
    end


end

%% For each behavior, make a distribution across neurons of when the beta value for the time regressor first inflects
figHandle = figure(22);
hold on;
firstIdx = nan(size(zBeta, 1), size(dimBeta, 2));
% Loop through behaviors to get distribution of neurons start times for
% each behavior
for iBhv = 1 : size(zBeta, 1)

    % are there any beta values above/below 2 standard deviations
    beyondStd2 = zBeta{iBhv} >= 2 | zBeta{iBhv} <= -2;
    beyondStd1 = zBeta{iBhv} >= 1 | zBeta{iBhv} <= -1;
    % beyondStd2 = zBeta{iBhv} >= 2 | zBeta{iBhv} <= -2;

    % Loop through neurons and find first index above/below 1std
    for jNrn = 1:size(beyondStd2, 2)

        % if plotFlag
        %     plot(zBeta{iBhv}(:, jNrn))
        % end

        % nonzeroRows = find(beyondStd2(:, jNrn) ~= 0);
        nonzeroRows = find(beyondStd1(:, jNrn) ~= 0);
        if ~isempty(nonzeroRows)
            % Find the first two consecutive numbers
            for i = 1:(length(nonzeroRows) - 1)
                if nonzeroRows(i) == nonzeroRows(i + 1) - 1
                    firstConsecutiveNumber = nonzeroRows(i);
                    firstIdx(iBhv, jNrn) = firstConsecutiveNumber;

                    break;  % Exit the loop when a pair is found
                end
            end

        end

        % if plotFlag
        %     clf
        % end

    end
    if plotFlag
        % iRegSection = dimBeta(1 + bhvReg(iBhv) : 20 + bhvReg(iBhv), :);
        % meanM23Beta = mean(iRegSection(:,idM23), 2);
        % plot(meanM23Beta, 'r', 'lineWidth', 3)
        hold on
        plot([0 20], [0 0], '--k')

        nonNanM23 = ~isnan(firstIdx(iBhv,idM23));
        % plot(mean(zBeta{iBhv}(:, idM23), 2), 'r', 'linewidth', 2)
        plot(mean(zBeta{iBhv}(:, idM23(nonNanM23)), 2), 'r', 'linewidth', 2)
        firstM23 = mean(firstIdx(iBhv,idM23), 'omitnan');
        plot([firstM23 firstM23], ylim, 'r', 'linewidth', 2)

        nonNanM56 = ~isnan(firstIdx(iBhv,idM56));
        % plot(mean(zBeta{iBhv}(:, idM56), 2), 'm', 'linewidth', 2)
        plot(mean(zBeta{iBhv}(:, idM56(nonNanM56)), 2), 'm', 'linewidth', 2)
        firstM56 = mean(firstIdx(iBhv,idM56), 'omitnan');
        plot([firstM56 firstM56], ylim, 'm', 'linewidth', 2)

        nonNanDS = ~isnan(firstIdx(iBhv,idDS));
        % plot(mean(zBeta{iBhv}(:, idDS), 2), 'b', 'linewidth', 2)
        plot(mean(zBeta{iBhv}(:, idDS(nonNanDS)), 2), 'b', 'linewidth', 2)
        firstDS = mean(firstIdx(iBhv,idDS), 'omitnan');
        plot([firstDS firstDS], ylim, 'b', 'linewidth', 2)

        nonNanVS = ~isnan(firstIdx(iBhv,idVS));
        % plot(mean(zBeta{iBhv}(:, idVS), 2), 'c', 'linewidth', 2)
        plot(mean(zBeta{iBhv}(:, idVS(nonNanVS)), 2), 'c', 'linewidth', 2)
        firstVS = mean(firstIdx(iBhv,idVS), 'omitnan');
        plot([firstVS firstVS], ylim, 'c', 'linewidth', 2)

        clf
    end
    fprintf('\n%s\n',analyzeBhv{iBhv})
    fprintf('M23: %.2f\nM56: %.2f\nDS: %.2f\nVS: %.2f\n', firstM23, mean(firstIdx(iBhv,idM56), 'omitnan'), mean(firstIdx(iBhv,idDS), 'omitnan'), mean(firstIdx(iBhv,idVS), 'omitnan'))

end



%% Plot alignments of first index when betas "turn on" for each behavior and brain area
for iBhv = 1 : length(bhvReg)
    iBhv
    iRegSection = dimBeta(1 + bhvReg(iBhv) : 20 + bhvReg(iBhv), :);
    % iRegSection = abs(iRegSection);

    if plotFlag
        meanM23Beta = mean(iRegSection(:,idM23), 2);
        plot(meanM23Beta, 'r', 'lineWidth', 3)

        meanM56Beta = mean(iRegSection(:,idM56), 2);
        plot(meanM56Beta, 'm', 'lineWidth', 3)

        meanDSBeta = mean(iRegSection(:,idDS), 2);
        plot(meanDSBeta, 'b', 'lineWidth', 3)

        meanVSBeta = mean(iRegSection(:,idVS), 2);
        plot(meanVSBeta, 'c', 'lineWidth', 3)

        legend('M23','M56','DS','VS')
        title(analyzeBhv{iBhv}, 'Interpreter', 'none')
        plot(xlim, [0 0], '--k')


        firstM23 = mean(firstIdx(iBhv,idM23), 'omitnan');
        plot([firstM23 firstM23], ylim, 'r')

        firstM56 = mean(firstIdx(iBhv,idM56), 'omitnan');
        plot([firstM56 firstM56], ylim, 'm')

        firstDS = mean(firstIdx(iBhv,idDS), 'omitnan');
        plot([firstDS firstDS], ylim, 'b')

        firstVS = mean(firstIdx(iBhv,idVS), 'omitnan');
        plot([firstVS firstVS], ylim, 'c')

        if savePlot
            saveas(figureHandle,fullfile(figurePath, ['meanBetas ', analyzeBhv{iBhv}]), 'pdf')
        end
        clf
        hold on
    end

end


%%
figure();
hold on;
for iBhv = 1 : size(zBeta, 1)
    for jNrn = 1 : size(zBeta{1}, 2)
        plot(zBeta{iBhv}(:,jNrn))
        pause
    end
end
close







































%%  Neural activity (z-scored for each neuron) for each behavior
warning('You are using mean and std within each behavioral time window to z-score - what might be better?')
spikeCounts = cell(length(analyzeBhv), size(dataMat, 2)); % collects the spike counts surrounding each behavior start
spikeZ = cell(length(analyzeBhv), size(dataMat, 2)); % calculates z-score of the spike counts

for iBhv = 1 : length(analyzeBhv)
    iReg = strcmp(['0 ',analyzeBhv{iBhv}], regressorLabels); % find regressor at start of behavior
    iStartFrames = find(bhvDesign(:,iReg)); % every frame where the behavior starts

    for jNeur = 1 : size(dataMat, 2)
        for k = 1 : length(iStartFrames)
            iRange = iStartFrames(k) - opts.mPreTime/opts.frameSize : iStartFrames(k) + opts.mPostTime/opts.frameSize;
            if iStartFrames(k) - opts.mPreTime/opts.frameSize > 0 && iStartFrames(k) + opts.mPostTime/opts.frameSize < size(dataMat, 1)
                spikeCounts{iBhv, jNeur} = [spikeCounts{iBhv, jNeur}; dataMat(iRange, jNeur)'];
                if sum(dataMat(iRange, jNeur))
                    spikeZ{iBhv, jNeur} = [spikeZ{iBhv, jNeur}; (dataMat(iRange, jNeur)' - mean(dataMat(iRange, jNeur))') / std(dataMat(iRange, jNeur))];
                else
                    spikeZ{iBhv, jNeur} = [spikeZ{iBhv, jNeur}; zeros(1,length(iRange))];
                end
            end
        end
    end
end


%% Plot some betas and psths

% set up figure
nRow = 4;
nColumn = 2;
orientation = 'portrait';
figureHandle = 34;
[axisWidth, axisHeight, xAxesPosition, yAxesPosition] = standard_figure(nRow, nColumn, orientation, figureHandle);
for col = 1 : nColumn
    for row = 1 : nRow
        ax(row,col) = axes('units', 'centimeters', 'position', [xAxesPosition(row, col) yAxesPosition(row, col) axisWidth axisHeight]);
        hold on;
    end
end
% colormap(bluewhitered)
colormap(jet)


% Indices of the neurons in each area
neuronsIdx = [goodM23, goodM56, goodDS, goodVS];
plotTitles = {'M23', 'M56', 'DS', 'VS'};


% Loop through all the behaviors you want to see
for iBhv = 1 : length(analyzeBhv)

    % Initialize a logical array to find relevant regressors
    containsReg = false(size(regressorLabels));

    % Loop through the regressors to check for the set corresponding to
    % that behavior
    for rIdx = 1:numel(regressorLabels)
        if ischar(regressorLabels{rIdx}) && contains(regressorLabels{rIdx}, analyzeBhv{iBhv})
            containsReg(rIdx) = true;
        end
    end

    % Specify the custom title position (in normalized figure coordinates)
    titleText = analyzeBhv{iBhv};
    annotation('textbox', [.5, 1, 0, 0], 'String', titleText, 'Interpreter', 'none', 'FontSize', 14, 'FontWeight', 'bold', 'EdgeColor', 'none', 'HorizontalAlignment', 'center');
    annotation('textbox', [.2, 1, 0, 0], 'String', 'Betas', 'Interpreter', 'none', 'FontSize', 14, 'FontWeight', 'bold', 'EdgeColor', 'none', 'HorizontalAlignment', 'center');
    annotation('textbox', [.8, 1, 0, 0], 'String', 'Spikes', 'Interpreter', 'none', 'FontSize', 14, 'FontWeight', 'bold', 'EdgeColor', 'none', 'HorizontalAlignment', 'center');

    nrnIdx = 0;
    for row = 1 : nRow
        % Example plotting in the first subplot (ax1)
        neuronsPlot = intersect(idLabels, data.ci.cluster_id(neuronsIdx(:,row))); % Plot the neurons within this brain area in this row
        nrnIdx = 1 + nrnIdx(end) : nrnIdx(end) + length(neuronsPlot);

        % Beta value plot
        betaPlot = dimBeta(containsReg, nrnIdx);
        axes(ax(row, 1));
        if row < nRow
            set(ax(row, 1), 'xticklabel',{[]})
        end
        imagesc('CData', betaPlot');
        plot(ax(row, 1), [10.5 10.5], [0 length(neuronsPlot)], 'k', 'linewidth', 3)
        title(plotTitles{row});
        xlim([0.5 20.5])
        ylim([0.5 length(neuronsPlot)+.5])


        % PSTH plot
        axes(ax(row, 2))
        if row < nRow
            set(ax(row, 2), 'xticklabel',{[]})
        end
        imagesc(cell2mat(cellfun(@mean, (spikeZ(iBhv,nrnIdx)), 'UniformOutput', false)'))
        plot(ax(row, 2), [10.5 10.5], [0 length(neuronsPlot)], 'k', 'linewidth', 3)
        title(plotTitles{row});
        xlim([0.5 20.5])
        ylim([0.5 length(neuronsPlot)+.5])

    end



    c = colorbar;
    c.Position = [0.93, 0.1, 0.02, 0.5];

    if savePlot
        saveas(figureHandle,fullfile(figurePath, ['betas_and_spikes ', analyzeBhv{iBhv}]), 'pdf')
    end
    pause(10)
    delete(findall(gcf,'type','annotation'))
    % clf
end

close(figureHandle)






%%
iBhv = 14;
% imagesc(cell2mat(cellfun(@sum, (spikeCounts(1,:)), 'UniformOutput', false)'))
imagesc(cell2mat(cellfun(@mean, (spikeZ(iBhv,:)), 'UniformOutput', false)'))
colormap(bluewhitered), colorbar



%%










%% Cross-validate
%full model - this will take a moment
opts.folds = 10;
% [Vfull, fullBeta, ~, fullIdx, fullRidge, fullLabels] = crossValModel(fullR, dataMat, regLabels, regIdx, regLabels, opts.folds);
% [Vfull, fullBeta, ~, fullIdx, fullRidge, fullLabels] = crossValModel(fullR, Vc, regLabels, regIdx, regLabels, opts.folds);
% save([fPath 'cvFull.mat'], 'Vfull', 'fullBeta', 'fullR', 'fullIdx', 'fullRidge', 'fullLabels'); %save some results

randIdx = randperm(size(dataMat, 1));
foldCnt = floor(size(dataMat, 1) / opts.folds);
cBeta = cell(1,opts.folds);
yCVModel = zeros(size(dataMat));

for iFold = 1 : opts.folds
    dataIdx = true(size(dataMat, 1), 1);

    dataIdx(randIdx((iFold - 1) * foldCnt + (1:foldCnt))) = false;

    if iFold == 1
        [cRidge, cBeta{iFold}] = ridgeMML(dataMatModel(dataIdx,:), fullR(dataIdx,:), true); %get beta weights and ridge penalty for first fold
    else
        [~, cBeta{iFold}] = ridgeMML(dataMatModel(dataIdx,:), fullR(dataIdx,:), true, cRidge); %get beta weights and ridge penalty for remaining folds, using same ridge penalties
    end
    yCVModel(~dataIdx, :) = fullR(~dataIdx,:) * cBeta{iFold};
end


%%
numerator = sum((dataMatModel - yCVModel) .^2, 1);
denominator = sum((dataMatModel - mean(dataMatModel)) .^2, 1);
rSquared = 1 - numerator ./ denominator;






