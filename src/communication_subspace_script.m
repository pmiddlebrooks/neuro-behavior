cd('E:/Projects/toolboxes/communication-subspace/')
addpath regress_methods
addpath regress_util
addpath fa_util
% startup

%% Run this, then go to spiking_script and get the behavior and data matrix
% opts.frameSize = .05; % 50 ms framesize for now
opts.collectFor = 60*60; % Get an hour of data



%%




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%     Communication Subspace (Semedo et al 2019)            %%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%     adapted from communication-subsapce/example.m         %%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%
% from Semedo et al 2019:
% To study how neuronal activity in the two areas is related, we reasoned that any fluctuations in the V1 responses, whether due to changes in the
% visual stimulus or not, could relate to fluctuations in V2. We therefore subtracted the appropriate peri-stimulus time histogram (PSTH)
% from each single-trial response, and then analyzed the residuals for each orientation (termed datasets) separately

bhv = 'investigate_2';
% bhv = 'locomotion';
bhv = 'face_groom_1';
% bhv = 'contra_orient';
% bhv = 'investigate_2';
% bhv = 'head_groom';

bhvCode = analyzeCodes(strcmp(analyzeBhv, bhv));

bhvStartFrames = 1 + floor(dataBhv.StartTime(dataBhv.ID == bhvCode) ./ opts.frameSize);
bhvStartFrames(bhvStartFrames < 10) = [];
bhvStartFrames(bhvStartFrames > size(dataMat, 1) - 10) = [];
nTrial = length(bhvStartFrames);

periEventTime = -.5 : opts.frameSize : .5; % seconds around onset
dataWindow = periEventTime(1:end-1) / opts.frameSize; % frames around onset (remove last frame)

% a dataMat for a particular behavior
dataMatBhv = zeros(length(dataWindow), size(dataMat, 2), nTrial); % peri-event time X neurons X nTrial
nTrial = length(bhvStartFrames);
for j = 1 : nTrial
    dataMatBhv(:,:,j) = dataMat(bhvStartFrames(j) + dataWindow ,:);
end


meanPsthM56 = mean(dataMatBhv(:,idM56,:), 3);
meanRatesM56 = mean(meanPsthM56, 1) ./ opts.frameSize;
residualPsthM56 = dataMatBhv(:,idM56,:) - meanPsthM56;
shuffleIdx = randperm(nTrial);
residualPsthM56Shuffle = residualPsthM56(:,:,shuffleIdx);

meanPsthDS = mean(dataMatBhv(:,idDS,:), 3);
meanRatesDS = mean(meanPsthDS, 1) ./ opts.frameSize;
residualPsthDS = dataMatBhv(:,idDS,:) - meanPsthDS;
shuffleIdx = randperm(nTrial);
residualPsthDSShuffle = residualPsthDS(:,:,shuffleIdx);

%% Mean-match firing rates among the sub-populations
rateBins = 0.5 : 0.5 : 40;


%%% Stopped here: Nneed to test subspace
%%% dimensions among neurons that are "tuned" for a give behavior
%%% transition. That means selecting sub-neuron populations based on:
% tuning preference (positive and/or negative)
% mean-matched avg firing rate
%
% That means there will be very few neurons in any analysis

[histM56, ~, binsM56] = histcounts(meanRatesM56, rateBins);
[histDS, ~, binsDS] = histcounts(meanRatesDS, rateBins);

M56Targets = [];
DSTargets = [];
for i = 1 : length(histDS)
    iNSample = min(histM56(i), histDS(i)); % which brain area had minimum number of units with that spike rate?
    if iNSample
        allM56Targets = find(binsM56 == i);
        allDSTargets = find(binsDS == i);
        randM56Targets = allM56Targets(randperm(length(allM56Targets)));
        randDSTargets = allDSTargets(randperm(length(allDSTargets)));
        jM56Targets = [];
        jDSTargets = [];
        for j = 1 : iNSample % Take random sa
            jM56Targets = [jM56Targets; randM56Targets(j)];
            jDSTargets = [jDSTargets; randDSTargets(j)];
        end

        M56Targets = [M56Targets; jM56Targets];
        DSTargets = [DSTargets; jDSTargets];
    end
end

% Use up to half of the M56 (or DS) population as target population (take a random
% sample of the rate-matched subset
% nTargetNeurons = min(length(DSTargets), floor(length(idDS)/2));
nTargetNeurons = min(length(M56Targets), floor(length(idM56)/2));
neuronsIdx = randperm(nTargetNeurons);
M56Targets = M56Targets(neuronsIdx);
DSTargets = DSTargets(neuronsIdx);
% DSSources = setdiff(1:length(idDS), DSTargets);
M56Sources = setdiff(1:length(idM56), M56Targets);

%% put residualPsth in format for communication subspace code
resDSSourceShuffle = [];
% resDSSource = [];
resM56Source = [];
resM56Target = [];
resDSTarget = [];
for i = 1 : nTrial
    % resDSSource = [resDSSource; residualPsthDS(:,DSSources,i)];
    resM56Source = [resM56Source; residualPsthM56(:,M56Sources,i)];
    resDSSourceShuffle = [resDSSourceShuffle; residualPsthDSShuffle(:,DSSources,i)];
    resM56Target = [resM56Target; residualPsthM56(:,M56Targets,i)];
    resDSTarget = [resDSTarget; residualPsthDS(:,DSTargets,i)];
end


%%
% X = fullMat(:,strcmp(areaLabels, 'M56'));
% Y_V2 = fullMat(:, strcmp(areaLabels, 'DS'));

% X = resDSSource;
X = resM56Source;
% X = resDSSourceShuffle;
% Y_V2 = resM56Target;
Y_V2 = resDSTarget;





%%

SET_CONSTS
%%
load('mat_sample/sample_data.mat')

%%
% ========================================================================
% 1) Identifying dimensions in the source activity space
% ========================================================================

% The process of identifying dimensions in the source activity space
% depends on the method used. Here, we provide a few examples.

% Sample data used here corresponds to V1 and V2 residuals (i.e., PSTHs
% have been subtracted from the full responses), in response to a drifting
% grating. sample_data.mat contains two variables, X and Y_V2. X contains
% the activity in the source population. It's a N-by-p matrix, where N is
% the number of datapoints and p is the number of source neurons. Y
% contains the activity in the target population. It's a N-by-K matrix,
% where K is the number of target neurons. For the sample data, N = 4000,
% p = 79 and K = 31. The N datapoints can come from different time points
% or different trials. As an example, the 4000 datapoints used here come
% from 400 trials that contained 10 time points each.



%% Reduced Rank Regression

[~, B_] = ReducedRankRegress(Y_V2, X);
% The columns of B_ contain the predictive dimensions of source activity
% X. Predictive dimensions are ordered by target variance explained. As a
% result, the top d predictive dimension are given by B_(:,1:d). The
% columns of B_ are not orthonormal, i.e., they do not form an orthonormal
% basis for a subspace of the source activity X. They are, however,
% guaranteed to be independent. A suitable basis for the predictive
% subspace can be found via QR decomposition: [Q,R] = qr( B_(:,1:q) ). In
% this instance, Q provides an orthonormal basis for a q-dimensional
% predictive subspace. The correct dimensionality for the Reduced Rank
% Regression model (i.e., the optimal number of predictive dimensions) is
% found via cross-validation (see section 2), below).

%% Factor Analysis

q = 30;

[Z, U, Q] = ExtractFaLatents(X, q);
% The columns of U contain the dominant dimensions of the source activity
% X. Dominant dimensions are ordered by shared variance explained. As a
% result, the top d dominant dimensions are given by U(:,1:d). Note that
% the latent variables Z under the Factor Analysis model are not obtained
% by projecting the data onto the dominant dimensions. This is due to
% Factor Analysis' noise model. Rather, Z = (X - M)*Q, where Q is the
% Factor Analysis "decoding" matrix, and M is the sample mean. The
% reconstructed data is given by Z*U' + M. The correct dimensionality for
% the Factor Analysis model (i.e., the optimal number of dominant
% dimensions) is found via cross-validation (see section 2), below).

%%
% ========================================================================
% 2) Cross-validation
% ========================================================================

% - Cross-validating any of the included regression methods follows the
% same general from. First, define the auxiliary cross-validation function
% based on the chosen regression method:
%
% cvFun = @(Ytrain, Xtrain, Ytest, Xtest) RegressFitAndPredict...
%	(regressMethod, Ytrain, Xtrain, Ytest, Xtest, cvParameter, ...
%	'LossMeasure', lossMeasure, ...
%	'RegressMethodExtraArg1', regressMethodExtraArg1, ...);
%
% When using Ridge regression, for example, we have:
%
% regressMethod = @RidgeRegress;
% cvParameter = lambda;
% lossMeasure = 'NSE'; % NSE stands for Normalized Squared Error
%
% Ridge Regression has no extra arguments, so the auxiliary
% cross-validation function becomes:
%
% cvFun = @(Ytrain, Xtrain, Ytest, Xtest) RegressFitAndPredict...
%	(@RidgeRegress, Ytrain, Xtrain, Ytest, Xtest, lambda, ...
%	'LossMeasure', 'NSE');
%
% Whenever the regression function has extra (potentially optional)
% arguments, they are passed to the auxiliary cross-validation function as
% name argument pairs.
%
% For Ridge regression, the correct range for lambda can be determined
% using:
%
% dMaxShrink = .5:.01:1;
% lambda = GetRidgeLambda(dMaxShrink, X);
%
% (See Elements of Statistical Learning, by Hastie, Tibshirani and
% Friedman for more information.)



%% Regression cross-validation examples +++++++++++++++++++++++++++++++++++
% ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

%% Cross-validate Reduced Rank Regression

% Vector containing the interaction dimensionalities to use when fitting
% RRR. 0 predictive dimensions results in using the mean for prediction.
numDimsUsedForPrediction = 1:10;

% Number of cross validation folds.
cvNumFolds = 10;

% Initialize default options for cross-validation.
cvOptions = statset('crossval');

% If the MATLAB parallel toolbox is available, uncomment this line to
% enable parallel cross-validation.
% cvOptions.UseParallel = true;

% Regression method to be used.
regressMethod = @ReducedRankRegress;

% Auxiliary function to be used within the cross-validation routine (type
% 'help crossval' for more information). Briefly, it takes as input the
% the train and test sets, fits the model to the train set and uses it to
% predict the test set, reporting the model's test performance. Here we
% use NSE (Normalized Squared Error) as the performance metric. MSE (Mean
% Squared Error) is also available.
cvFun = @(Ytrain, Xtrain, Ytest, Xtest) RegressFitAndPredict...
    (regressMethod, Ytrain, Xtrain, Ytest, Xtest, ...
    numDimsUsedForPrediction, 'LossMeasure', 'NSE');

% Cross-validation routine.
cvl = crossval(cvFun, Y_V2, X, ...
    'KFold', cvNumFolds, ...
    'Options', cvOptions);

% Stores cross-validation results: mean loss and standard error of the
% mean across folds.
cvLoss = [ mean(cvl); std(cvl)/sqrt(cvNumFolds) ];

% To compute the optimal dimensionality for the regression model, call
% ModelSelect:
optDimReducedRankRegress = ModelSelect...
    (cvLoss, numDimsUsedForPrediction);

% Plot Reduced Rank Regression cross-validation results
x = numDimsUsedForPrediction;
y = 1-cvLoss(1,:);
e = cvLoss(2,:);

errorbar(x, y, e, 'o--', 'Color', COLOR(V2,:), ...
    'MarkerFaceColor', COLOR(V2,:), 'MarkerSize', 10)

xlabel('Number of predictive dimensions')
ylabel('Predictive performance')



%% Cross-validate Factor Regression

numDimsUsedForPrediction = 1:10;

cvNumFolds = 10;

cvOptions = statset('crossval');
% cvOptions.UseParallel = true;

regressMethod = @FactorRegress;

% In order to apply Factor Regression, we must first determine the optimal
% dimensionality for the Factor Analysis Model
p = size(X, 2);
q = 0:28;
qOpt = FactorAnalysisModelSelect( ...
    CrossValFa(X, q, cvNumFolds, cvOptions), ...
    q);

cvFun = @(Ytrain, Xtrain, Ytest, Xtest) RegressFitAndPredict...
    (regressMethod, Ytrain, Xtrain, Ytest, Xtest, ...
    numDimsUsedForPrediction, ...
    'LossMeasure', 'NSE', 'qOpt', qOpt);
% qOpt is an extra argument for FactorRegress. Extra arguments for the
% regression function are passed as name/value pairs after the
% cross-validation parameter (in this case numDimsUsedForPrediction).
% qOpt, the optimal factor analysis dimensionality for the source activity
% X, must be provided when cross-validating Factor Regression. When
% absent, Factor Regression will automatically determine qOpt via
% cross-validation (which will generate an error if Factor Regression is
% itself used within a cross-validation procedure).

cvl = crossval(cvFun, Y_V2, X, ...
    'KFold', cvNumFolds, ...
    'Options', cvOptions);

cvLoss = ...
    [ mean(cvl); std(cvl)/sqrt(cvNumFolds) ];

optDimFactorRegress = ModelSelect...
    (cvLoss, numDimsUsedForPrediction);

% Plot Reduced Rank Regression cross-validation results
x = numDimsUsedForPrediction;
x(x > qOpt) = [];
y = 1-cvLoss(1,:);
e = cvLoss(2,:);

hold on
errorbar(x, y, e, 'o--', 'Color', COLOR(V2,:), ...
    'MarkerFaceColor', 'w', 'MarkerSize', 10)
hold off



%%
legend('Reduced Rank Regression', ...
    'Factor Regression', ...
    'Location', 'SouthEast')



%% Factor analysis cross-validation example +++++++++++++++++++++++++++++++
% ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

%%
q = 0:30;

cvNumFolds = 10;

cvOptions = statset('crossval');
% cvOptions.UseParallel = true;

cvLoss= CrossValFa(X, q, cvNumFolds, cvOptions);


% CrossValFa returns the cumulative shared variance explained. To compute
% the optimal Factor Analysis dimensionality, call
% FactorAnalysisModelSelect:
qOpt = FactorAnalysisModelSelect(cvLoss, q);








%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%          Feedforward - feedback (Semedo et al 2022)       %%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%        adapted from adapted from canonical-correlation-maps/example.m     %%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

opts.frameSize = .01; % Need 10ms binned spike matrix



%% Get data ready for CCA canonical-correlation-maps
areas = {'M56', 'VS'};
dataBhv = dataBhv(8:end-8,:);
bhvStartFrames = 1 + floor(dataBhv.StartTime ./ opts.frameSize);
expCond = dataBhv.ID;

nTrial = length(bhvStartFrames);
% how many frames is 1 sec peri-event?
dataWindow = -2 / opts.frameSize : 2 / opts.frameSize; % Frames

psths = zeros(sum(nrnInd), length(dataWindow), nTrial);
spikes = cell(1, 2);

dataMatBhv = zeros(size(dataMat, 2), length(dataWindow), nTrial); % neurons X peri-event time X nTrial
% bhvMat = [];
for j = 1 : nTrial
    % bhvMat = [bhvMat; dataMat(bhvStartFrames(j) + dataWindow ,:)];
    dataMatBhv(:,:,j) = dataMat(bhvStartFrames(j) + dataWindow, :)';
end

spikes{1} = dataMatBhv(:, strcmp(areaLabels, 'M56'), :);
spikes{2} = dataMatBhv(:, strcmp(areaLabels, 'DS'), :);




