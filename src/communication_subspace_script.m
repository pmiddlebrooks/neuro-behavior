cd('E:/Projects/toolboxes/communication-subspace/')
addpath regress_methods
addpath regress_util
addpath fa_util
% startup

%% get desired file paths
computerDriveName = 'ROSETTA'; %'ROSETTA'; % 'Z' or 'home'
paths = get_paths(computerDriveName);


opts = neuro_behavior_options;

animal = 'ag25290';
sessionBhv = '112321_1';
sessionNrn = '112321';
if strcmp(sessionBhv, '112321_1')
    sessionSave = '112321';
end


%% Run this, then go to spiking_script and get the behavior and data matrix
% opts.frameSize = .05; % 50 ms framesize for now
opts.collectFor = 60*60; % Get an hour of data



%%








%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%     Communication Subspace (Semedo et al 2019)           %%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%
% from Semedo et al 2019:
% To study how neuronal activity in the two areas is related, we reasoned that any fluctuations in the V1 responses, whether due to changes in the
% visual stimulus or not, could relate to fluctuations in V2. We therefore subtracted the appropriate peri-stimulus time histogram (PSTH)
% from each single-trial response, and then analyzed the residuals for each orientation (termed datasets) separately

bhv = 'investigate_1';
% bhv = 'locomotion';
bhv = 'face_groom_1';
% bhv = 'contra_orient';
% bhv = 'investigate_2';
% bhv = 'head_groom';

bhvCode = analyzeCodes(strcmp(analyzeBhv, bhv));

bhvStartFrames = floor(dataBhv.bhvStartTime(dataBhv.bhvID == bhvCode) ./ opts.frameSize);
bhvStartFrames(bhvStartFrames < 10) = [];
bhvStartFrames(bhvStartFrames > size(dataMat, 1) - 10) = [];

periEventTime = -.5 : opts.frameSize : .5; % seconds around onset
dataWindow = periEventTime(1:end-1) / opts.frameSize; % frames around onset (remove last frame)

% a dataMat for a particular behavior
dataMatBhv = zeros(length(dataWindow), size(dataMat, 2), nTrial); % peri-event time X neurons X nTrial
nTrial = length(bhvStartFrames);
for j = 1 : nTrial
    dataMatBhv(:,:,j) = dataMat(bhvStartFrames(j) + dataWindow ,:);
end

% meanPsthM56 = mean(dataMatBhv(:, strcmp(areaLabels, 'M56'), :), 3);
% meanPsthDS = mean(dataMatBhv(:, strcmp(areaLabels, 'DS'), :), 3);
% residualM56 = dataMatBhv(:, strcmp(areaLabels, 'M56'), :) - meanPsthM56;
% residualDS = dataMatBhv(:, strcmp(areaLabels, 'DS'), :) - meanPsthDS;

meanPsth = mean(dataMatBhv, 3);
residualPsth = dataMatBhv - meanPsth;


%% Mean-match firing rates among the sub-populations
rateBins = 0.5 : 0.5 : 40;
meanRates = mean(meanPsth, 1) ./ opts.frameSize;



%%% Stopped here: Nneed to test subspace
%%% dimensions among neurons that are "tuned" for a give behavior
%%% transition. That means selecting sub-neuron populations based on:
% tuning preference (positive and/or negative)
% mean-matched avg firing rate
%
% That means there will be very few neurons in any analysis


[histM56, ~, binsM56] = histcounts(meanRates(strcmp(areaLabels, 'M56')), rateBins);
[histDS, ~, binsDS] = histcounts(meanRates(strcmp(areaLabels, 'DS')), rateBins);


% meanM56Rates = mean(meanPsth(:, strcmp(areaLabels, 'M56')), 1) ./ opts.frameSize;
% meanDSRates = mean(meanPsth(:, strcmp(areaLabels, 'DS')), 1) ./ opts.frameSize;
% [histM56, ~, binsM56] = histcounts(meanM56Rates, rateBins);
% [histDS, ~, binsDS] = histcounts(meanDSRates, rateBins);

M56Targets = [];
DSTargets = [];
for i = 1 : length(histM56)
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

 % Use up to half of the M56 population as target population
nTargetNeurons = min(length(M56Targets), floor(sum(strcmp(areaLabels, 'M56'))/2));
M56Targets = M56Targets(1 : nTargetNeurons);
DSTargets = DSTargets(1 : nTargetNeurons);
M56Sources = setdiff(1:length(meanM56Rates), M56Targets);

%% put residualPsth in format for communication subspace code
resM56Source = [];
resM56Target = [];
resDSTarget = [];
for i = 1 : nTrial
    resM56Source = [resM56Source; residualM56(:,M56Sources,i)];
    resM56Target = [resM56Target; residualM56(:,M56Targets,i)];
    resDSTarget = [resDSTarget; residualDS(:,DSTargets,i)];
end


%%
% X = fullMat(:,strcmp(areaLabels, 'M56'));
% Y_V2 = fullMat(:, strcmp(areaLabels, 'DS'));
X = resM56Source;
Y_V2 = resDSTarget;





%%

SET_CONSTS

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
%%%%%%%%%%%%%%%%%%%%%%%          Feedforward - feedback (Semedo et al 2022)       %%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

opts.frameSize = .01; % Need 10ms binned spike matrix

%%                   Get behavior data
bhvDataPath = strcat(paths.bhvDataPath, 'animal_',animal,'/');
bhvFileName = ['behavior_labels_', animal, '_', sessionBhv, '.csv'];


opts.dataPath = bhvDataPath;
opts.fileName = bhvFileName;

dataBhv = load_data(opts, 'behavior');



codes = unique(dataBhv.bhvID);
% codes(codes == -1) = []; % Get rid of the nest/irrelevant behaviors
behaviors = {};
for iBhv = 1 : length(codes)
    firstIdx = find(dataBhv.bhvID == codes(iBhv), 1);
    behaviors = [behaviors, dataBhv.bhvName{firstIdx}];
    % fprintf('behavior %d:\t code:%d\t name: %s\n', i, codes(i), dataBhvAlex.Behavior{firstIdx})
end

opts.behaviors = behaviors;
opts.bhvCodes = codes;
opts.validCodes = codes(codes ~= -1);

%             Get Neural matrix

nrnDataPath = strcat(paths.nrnDataPath, 'animal_',animal,'/', sessionNrn, '/');
nrnDataPath = [nrnDataPath, 'recording1/'];
opts.dataPath = nrnDataPath;

data = load_data(opts, 'neuron');
data.bhvDur = dataBhv.bhvDur;
clusterInfo = data.ci;
spikeTimes = data.spikeTimes;
spikeClusters = data.spikeClusters;

% Find the neuron clusters (ids) in each brain region

allGood = strcmp(data.ci.group, 'good') & strcmp(data.ci.KSLabel, 'good');

goodM23 = allGood & strcmp(data.ci.area, 'M23');
goodM56= allGood & strcmp(data.ci.area, 'M56');
goodDS = allGood & strcmp(data.ci.area, 'DS');
goodVS = allGood & strcmp(data.ci.area, 'VS');

% Make or load neural matrix

% which neurons to use in the neural matrix
opts.useNeurons = find(goodM23 | goodM56 | goodDS | goodVS);

[dataMat, idLabels, areaLabels, removedNeurons] = neural_matrix(data, opts); % Change rrm_neural_matrix

idM23 = find(strcmp(areaLabels, 'M23'));
idM56 = find(strcmp(areaLabels, 'M56'));
idDS = find(strcmp(areaLabels, 'DS'));
idVS = find(strcmp(areaLabels, 'VS'));

fprintf('%d M23\n%d M56\n%d DS\n%d VS\n', length(idM23), length(idM56), length(idDS), length(idVS))

%% Get data ready for CCA canonical-correlation-maps
areas = {'M56', 'VS'};
dataBhv = dataBhv(8:end-8,:);
bhvStartFrames = floor(dataBhv.bhvStartTime ./ opts.frameSize);
expCond = dataBhv.bhvID;

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

spikes{1} = dataMatBhv(strcmp(areaLabels, 'M56'),:,:);
spikes{2} = dataMatBhv(strcmp(areaLabels, 'DS'),:,:);




