%%%%%%%%%%%%%%%%%%%%
%   Design Matrix
%%%%%%%%%%%%%%%%%%%%

%% get desired file paths
computerDriveName = 'ROSETTA'; % 'Z' or 'home'
paths = rrm_get_paths(computerDriveName);

%%

% Model Features
%   Actions
%   Last Action
%   LFP powers as a proxy of animal state (alpha, beta, gamma)
%   Physical location in box
%   Distance from nest
%   Orientation w.r.t. nest




%% load options structure
opts = neuro_behavior_options;
% addpath('E:/Projects/neuro-behavior/src')
% savepath

% Define different frame sizes if desired

opts.windowSize = .2; % sec
opts.frameSize = .2; % sec
opts.framesPerTrial = floor(opts.windowSize / opts.frameSize);
opts.mPreTime = 0;  % precede motor events by 1000 ms to capture preparatory activity (used for eventType 3)
opts.mPostTime = 0;   % follow motor eventS for 1000 ms for mPostStim (used for eventType 3). Remove one frame since behavior starts at time zero.
opts.shiftAlignFactor = -.5;

%%
% opts = neuro_behavior_options;
animal = 'ag25290';
sessionBhv = '112321_1';
sessionNrn = '112321';

if strcmp(sessionBhv, '112321_1')
    sessionSave = '112321';
end


%%
figurePath = strcat(paths.figurePath, animal, '/', sessionSave, '/figures/', ['start ' num2str(opts.collectStart), ' for ', num2str(opts.collectFor)]);
if ~exist(figurePath, 'dir')
    mkdir(figurePath);
    % addpath(figurePath)
end




%% Test for stretches without much in-nest/irrelavent behaviot (behavior = -1)

% bhvDataPath = ['E:/Projects/ridgeRegress/data/neuro-behavior/processed_behavior/animal_',animal,'/'];
% bhvFileName = ['behavior_labels_', animal, '_', sessionBhv, '.csv'];
%
% opts.dataPath = bhvDataPath;
% opts.fileName = bhvFileName;
%
%
% increment = 10 * 60; % 10 minute increments
% iterate = floor(4*60*60 / increment);
% opts.collectStart = -5*60;
% for i = 1 : iterate - 1
%     opts.collectStart = opts.collectStart + increment;
%     dataBhv = load_data(opts, 'behavior');
%     framesNest = sum(dataBhv.Dur(dataBhv.ID == -1)) / (opts.collectFor / opts.frameSize);
%     fprintf("MinuteStart: %d\tPctNest: %.2f\n", opts.collectStart/60, framesNest*100)
% end

%%
bhvDataPath = strcat(paths.bhvDataPath, 'animal_',animal,'/');
bhvFileName = ['behavior_labels_', animal, '_', sessionBhv, '.csv'];


opts.dataPath = bhvDataPath;
opts.fileName = bhvFileName;

dataBhv = load_data(opts, 'behavior');



%% Create list of behaviors to use and select valid ones:
codes = unique(dataBhv.ID);
% codes(codes == -1) = []; % Get rid of the nest/irrelevant behaviors
behaviors = {};
for iBhv = 1 : length(codes)
    firstIdx = find(dataBhv.ID == codes(iBhv), 1);
    behaviors = [behaviors, dataBhv.Name{firstIdx}];
    % fprintf('behavior %d:\t code:%d\t name: %s\n', i, codes(i), dataBhvAlex.Behavior{firstIdx})
end

opts.behaviors = behaviors;
opts.bhvCodes = codes;
opts.validCodes = codes(codes ~= -1);


%% Select valid behaviors
rmvBhv = zeros(1, length(behaviors));
for i = 1 : length(behaviors)
    if sum(dataBhv.ID == codes(i) & dataBhv.Valid) < 20
        rmvBhv(i) = 1;
    end
end

analyzeBhv = behaviors(~rmvBhv);
analyzeCodes = codes(~rmvBhv);




%% Get design matrix for behaviors
[bhvDesign, regressorLabels] = rrm_design_matrix_behavior(dataBhv, opts);
frameIdx = 1 : size(bhvDesign, 1);


% %% Get one-back design matrix
% % vector of one-back behaviors
% dataBhv.prevBhvID = [nan; dataBhv.ID(1:end-1)];
% [oneBackDesign, oneBackLabels] = rrm_design_matrix_one_back(dataBhv, opts);


%% Remove columns without valid regressors (and remove their labels)
% Columns get removed if:
%   - a behavior is not listed in the behaviors we want to analyze (e.g. nest/irrelevant)
%   - there aren't enough behavior bouts to analyze for that behavior, as set in opt.minBhvNum

rmBhvDesignCol = sum(bhvDesign, 1) == 0;
bhvDesign(:, rmBhvDesignCol) = [];
regressorLabels(rmBhvDesignCol) = [];


% Remove rows without any regressors
% % (happens a lot, e.g. when a behavior lasts longer than the time
% window we're interested in analyzing)


removeRows = sum(bhvDesign, 2) == 0;

frameIdx(removeRows) = [];
bhvDesign(removeRows,:) = [];
% oneBackDesign(removeRows,:) = [];



% After removing any rows for which bhvDesign didn't have regressors, need to
% again remove any zero-regressor columns from oneBackDesign (b/c some of
% the columns only had one-back regressors when there is no regressor data
% to analyze in bhvDesign)
% rmOneBackCol = sum(oneBackDesign, 1) == 0;
% oneBackLabels(rmOneBackCol) = [];
% oneBackDesign(:, rmOneBackCol) = [];


% imagesc([bhvDesign oneBackDesign])
imagesc([bhvDesign])


% % Finally, if removing rows also removed too many one-back regressors, get rid of those
%
% oneBackBelowMin = sum(oneBackDesign, 1) < opts.minOneBackNum;
% oneBackDesign(:, oneBackBelowMin) = [];
% oneBackLabels(oneBackBelowMin)= [];
%





%%
fullR = bhvDesign;
regLabels = regressorLabels;
% fullR = [bhvDesign oneBackDesign];
% regLabels = [regressorLabels, oneBackLabels];



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
























%% Get Neural matrix
nrnDataPath = strcat(paths.nrnDataPath, 'animal_',animal,'/', sessionNrn, '/');
nrnDataPath = [nrnDataPath, 'recording1/'];
opts.dataPath = nrnDataPath;

data = load_data(opts, 'neuron');
data.bhvDur = dataBhv.Dur;
clusterInfo = data.ci;
spikeTimes = data.spikeTimes;
spikeClusters = data.spikeClusters;


%% Find the neuron clusters (ids) in each brain region

allGood = strcmp(data.ci.group, 'good') & strcmp(data.ci.KSLabel, 'good');

goodM23 = allGood & strcmp(data.ci.area, 'M23');
goodM56= allGood & strcmp(data.ci.area, 'M56');
goodDS = allGood & strcmp(data.ci.area, 'DS');
goodVS = allGood & strcmp(data.ci.area, 'VS');

opts.useNeurons = find(goodM23 | goodM56 | goodDS | goodVS);

%% Make or load neural matrix

% which neurons to use in the neural matrix
opts.useNeurons = find(goodM23 | goodM56 | goodDS | goodVS);

tic
[dataMat, idLabels, areaLabels, removedNeurons] = neural_matrix(data, opts); % Change rrm_neural_matrix
toc

idVS = find(strcmp(areaLabels, 'VS'));
idDS = find(strcmp(areaLabels, 'DS'));
idM56 = find(strcmp(areaLabels, 'M56'));
idM23 = find(strcmp(areaLabels, 'M23'));

fprintf('%d M23\n%d M56\n%d DS\n%d VS\n', length(idM23), length(idM56), length(idDS), length(idVS))
%%
saveDataPath = fullfile(paths.saveDataPath, animal, sessionSave, ['start ' num2str(opts.collectStart), ' for ', num2str(opts.collectFor)]);

% saveDataPath = strcat(paths.saveDataPath, animal,'/', sessionNrn, '/');
save(fullfile(saveDataPath,'neural_matrix.mat'), 'dataMat', 'idLabels', 'areaLabels', 'removedNeurons')
%%
load(fullfile(saveDataPath,'neural_matrix.mat'), 'dataMat', 'idLabels', 'areaLabels', 'removedNeurons')


%%

% remove the frames you removed from the design matrix
dataMat(removeRows,:) = [];

imagesc(dataMat')
hold on;
line([0, size(dataMat, 1)], [idM23(end)+.5, idM23(end)+.5], 'Color', 'r');
line([0, size(dataMat, 1)], [idM56(end)+.5, idM56(end)+.5], 'Color', 'r');
line([0, size(dataMat, 1)], [idDS(end)+.5, idDS(end)+.5], 'Color', 'r');


% %% Normalize and zero-center the neural data matrix for the regression
%
% % Normalize (z-score)
dataMatStd = std(dataMat, 0, 1);
dataMatMean = mean(dataMat, 1);
dataMatZ = zscore(dataMat, 0, 1);
%















%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%            make some synthetic data to test regression code
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Make a dataset with awesome neuron-behavior alignment
dataMatTest = zeros(size(dataMat));

nNeuronsPerBhv = ceil(size(dataMatTest, 2) / 3); % ceil(size(dataMatTest, 2) / length(analyzeBhv));
nRandomSpikes = ceil(.05 * size(dataMatTest, 1));
for iBhv = 1 : length(analyzeBhv)
    iStarts = find(bhvDesign(:,iBhv));

    iRandomNeurons = randperm(size(dataMatTest, 2), nNeuronsPerBhv);
    dataMatTest(iStarts, iRandomNeurons) = 2;
    for j = iRandomNeurons
           jRandomSpikeInd = randperm(size(dataMatTest, 1), nRandomSpikes);
           dataMatTest(jRandomSpikeInd, j) = 1;
    end

end


%%
dataMatTestZ = zscore(dataMatTest, 0, 1);
tic
% addpath('E:/Projects/sandbox/ridgeModel/');
[ridgeVals, dimBeta] = ridgeMML(dataMatTestZ, fullR, true); %get ridge penalties and beta weights.
toc
yModel = (fullR - mean(fullR, 1)) * dimBeta;

% R squared per neuron
plotFlag = 1;
savePlot = 0;

SSMod = sum((yModel - mean(dataMatTestZ, 1)) .^2, 1);
SSRes = sum((dataMatTestZ - yModel) .^2, 1);
SSTot = sum((dataMatTestZ - mean(dataMatTestZ, 1)) .^2, 1);

degFM = size(fullR, 2) - 1; % degrees of freedom of the model
msm = SSMod ./ degFM;
degFE = size(dataMatTestZ, 1) - size(fullR, 2);
mse = SSRes ./ degFE;
Fstat = msm ./ mse; % Model error / Residuals error


rSquaredFull = SSMod ./ SSTot
% rSquaredFull = 1 - SSRes ./ SSTot;

if plotFlag
    nRow = 2;
    nColumn = 1;
    orientation = 'portrait';
    figureHandle = 34;
    [axisWidth, axisHeight, xAxesPosition, yAxesPosition] = standard_figure(nRow, nColumn, orientation, figureHandle);
    ax(1) = axes('units', 'centimeters', 'position', [xAxesPosition(1, 1) yAxesPosition(1, 1) axisWidth axisHeight]);
    % rFig = figure('Position', get(0, 'ScreenSize'));
    hold on;
    scatter(ax(1), 1:size(dataMatZ, 2), rSquaredFull, 100, 'k', 'filled')

    xline(.5 + length(idM23), 'LineWidth', 3)
    xline(.5 + length(idM23) + length(idM56), 'LineWidth', 3)
    xline(.5 + length(idM23) + length(idM56) + length(idDS), 'LineWidth', 3)
    title(ax(1), 'R-squared for each neuron')
    xlabel(ax(1), 'M23 -> M56 -> DS -> VS')
    if savePlot
        saveas(figureHandle,fullfile(figurePath, 'r-Squared per neuron'), 'pdf')
    end
end


Fstat = (SSRes ./ degFM) ./ (SSTot ./ degFE)

% Calculate the p-value
pVal = 1 - fcdf(Fstat, dfNum, dfDenom);













%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                          Run the regression
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


tic
% addpath('E:/Projects/sandbox/ridgeModel/');
[ridgeVals, dimBeta] = ridgeMML(dataMatZ, fullR, true); %get ridge penalties and beta weights.
toc

% Run model to get neural predictions
%
%  reconstruct data and compute R^2
% yModel = (fullR * dimBeta);
% fullRStd = std(fullR, 0, 1);
% fullR = bsxfun(@rdivide, fullR, fullRStd);

yModel = (fullR - mean(fullR, 1)) * dimBeta;



%% Try built-in matlab functions for lasso and ridge



%% Explained variance
% total
% R squared
% SSR: sum(data - prediction)^2
% SST: 

SSMod = sum((yModel - mean(dataMatZ, 1)) .^2, 1);
SSRes = sum((dataMatZ - yModel) .^2, 1);
SSTot = sum((dataMatZ - mean(dataMatZ, 1)) .^2, 1);

degFM = size(fullR, 2) - 1; % degrees of freedom of the model
msm = SSMod ./ degFM;
degFE = size(dataMatZ, 1) - size(fullR, 2);
mse = SSRes ./ degFE;

Fstat = msm ./ mse; % Model error / Residuals error
% Calculate the p-value
dfNum = size(fullR, 2);
dfDenom = (size(dataMatZ, 1) - size(fullR,2) - 1);


rSquaredFull = SSMod ./ SSTot;

%
% F statistic
% = (SSR / numRegressors) / (SSE / (numDataPoints - numRegressors - 1))


Fstat = (SSRes / dfNum) / (SSTot / dfDenom)
% dfModel = size(fullR, 2);
% dfTotal = size(dataMatZ, 1) - 1;
% dfRedisdual = dfTotal - dfModel;

% Calculate the p-value
pVal = 1 - fcdf(Fstat, dfNum, dfDenom)


%%
% R squared per neuron
plotFlag = 1;
savePlot = 0;

SSMod = sum((yModel - mean(dataMatZ, 1)) .^2, 1);
SSRes = sum((dataMatZ - yModel) .^2, 1);
SSTot = sum((dataMatZ - mean(dataMatZ, 1)) .^2, 1);

dfNum = size(fullR, 2);
dfDenom = (size(dataMatZ, 1) - size(fullR,2) - 1);


rSquaredFull = SSMod ./ SSTot;

if plotFlag
    nRow = 2;
    nColumn = 1;
    orientation = 'portrait';
    figureHandle = 34;
    [axisWidth, axisHeight, xAxesPosition, yAxesPosition] = standard_figure(nRow, nColumn, orientation, figureHandle);
    ax(1) = axes('units', 'centimeters', 'position', [xAxesPosition(1, 1) yAxesPosition(1, 1) axisWidth axisHeight]);
    % rFig = figure('Position', get(0, 'ScreenSize'));
    hold on;
    scatter(ax(1), 1:size(dataMatZ, 2), rSquaredFull, 100, 'k', 'filled')

    xline(idM23(end)+.5, 'LineWidth', 3)
    xline(idM56(end)+.5, 'LineWidth', 3)
    xline(idDS(end)+.5, 'LineWidth', 3)
    title(ax(1), 'R-squared for each neuron')
    xlabel(ax(1), 'M23 -> M56 -> DS -> VS')
    if savePlot
        saveas(figureHandle,fullfile(figurePath, 'r-Squared per neuron'), 'pdf')
    end
end


Fstat = (SSRes ./ dfNum) ./ (SSTot ./ dfDenom);

% Calculate the p-value
pVal = 1 - fcdf(Fstat, dfNum, dfDenom);









%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% To compare when different brain areas become important:

% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 0. Compare betas over time for each behavior, comparing across brain
% areas (when do betas in each brain increase/decrease first, last, etc?)
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%


%%

[sum(relExpSig(:,idM23), 2), ...
    sum(relExpSig(:,idM56), 2), ...
    sum(relExpSig(:,idDS), 2), ...
    sum(relExpSig(:,idVS), 2)]

%% For each behavior, plot the beta values for each neuron

bhvReg = (0 : 20 : 300);
zBeta = cell(length(bhvReg), 1);
figureHandle = figure(55);
firstIdx = nan(size(zBeta, 1), size(dimBeta, 2));

hold on;
for iBhv = 1 : length(bhvReg)
    iBhv
    iRegSection = dimBeta(iBhv, :);
    % iRegSection = abs(iRegSection);


    if plotFlag
        meanM23Beta = mean(iRegSection(:,idM23), 2);
        scatter(1, meanM23Beta, 'r', 'lineWidth', 3)
        meanM56Beta = mean(iRegSection(:,idM56), 2);
        scatter(2, meanM56Beta, 'm', 'lineWidth', 3)
        meanDSBeta = mean(iRegSection(:,idDS), 2);
        scatter(3, meanDSBeta, 'b', 'lineWidth', 3)
        meanVSBeta = mean(iRegSection(:,idVS), 2);
        scatter(4, meanVSBeta, 'c', 'lineWidth', 3)

        legend('M23','M56','DS','VS')
        title(analyzeBhv{iBhv}, 'Interpreter', 'none')
        plot(xlim, [0 0], '--k')
        if savePlot
            saveas(figureHandle,fullfile(figurePath, ['single regressor meanBetas ', analyzeBhv{iBhv}]), 'pdf')
            pause(15)
        end
        clf
        hold on
    end


end














%%
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 1. Use F-statistic of partial models
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% For each regressor, remove it
% Run regression without it
% Run 1000 shuffled regressions without it (shuffle the start times of each
% behavior).
% Ask whether F-statisic of unshuffled model is different than distribution of F-stats of shuffled model
% For each area, across behaviors, when does F-statistic differ? Is this time different within different areas?

%% Nested F-stats for partial models
tic

optsNest.nIter = 1000;
fStatZ = zeros(length(regLabels), size(dataMatZ, 2));
nReg = length(regLabels);

poolID = parpool(4, 'IdleTimeout', Inf);
parfor iReg = 1 : length(regLabels)
    % for iReg = 10 : length(regLabels)
    regLabels{iReg}
    % optsNest.removeReg = iReg;

    fStatZ(iReg, :) = rrm_nested_fstat_shuffled(fullR, dataMatZ, SSRes, iReg, optsNest);

end % iReg
delete(poolID)
toc
%% Save copies to local computer and Z drive
save(fullfile(saveDataPath, 'single regressor nested fstat shuffled.mat'), 'fStatZ', 'idLabels', 'areaLabels', 'regLabels', '-mat')

pathZ = rrm_get_paths('Z');
saveZ = fullfile(pathZ.saveDataPath, animal, sessionSave, ['start ' num2str(opts.collectStart), ' for ', num2str(opts.collectFor)]);
if ~exist(saveZ, 'dir')
    mkdir(saveZ);
end
save(fullfile(saveZ , 'single regressor nested fstat shuffled.mat'), 'fStatZ', 'idLabels', 'areaLabels', 'regLabels', '-mat')

%%
fStat = load(fullfile(saveDataPath, 'single regressor nested fstat shuffled.mat'));

%%
figurePath = strcat(paths.figurePath, animal, '/', sessionSave, '/figures/', ['start ' num2str(opts.collectStart), ' for ', num2str(opts.collectFor)]);
if ~exist(figurePath, 'dir')
    mkdir(figurePath);
    % addpath(figurePath)
end


% Define the filename for the output video
outputVideoFile = fullfile(figurePath, 'F-stat shuffled single regressor per neuron.mp4');

% Create a VideoWriter object
writerObj = VideoWriter(outputVideoFile, 'MPEG-4');
writerObj.FrameRate = .5;  % Set the frame rate (frames per second)

% Open the video file for writing
open(writerObj);

fig = figure(98);
set(fig, 'PaperOrientation', 'landscape','Position', [50, 50, 1000, 800])

for iBhv = 1 : length(analyzeBhv)
    fStatZInd = iBhv;  % which regressors belong to this behavior?
    clf


    % plot(fStatZ(iReg,:), 'LineWidth', 2)
    head = scatter((1:size(dataMatZ, 2)), fStatZ(fStatZInd,:), 100, 'k', 'filled');
    hold on
    bhvTitle = [analyzeBhv{iBhv}];
    title(bhvTitle, 'Interpreter', 'none')
    xline(idM23(end)+.5, 'LineWidth', 2)
    xline(idM56(end)+.5, 'LineWidth', 2)
    xline(idDS(end)+.5, 'LineWidth', 2)
    plot(xlim, [-3 -3], '--k')

    m23Prop = sum(fStatZ(fStatZInd, idM23) <= -3) / length(idM23);
    m56Prop = sum(fStatZ(fStatZInd, idM56) <= -3) / length(idM56);
    dsProp = sum(fStatZ(fStatZInd, idDS) <= -3) / length(idDS);
    vsProp = sum(fStatZ(fStatZInd, idVS) <= -3) / length(idVS);
    fprintf('\nM23: %.2f\nM56: %.2f\nDS: %.2f\nVS: %.2f\n', m23Prop, m56Prop, dsProp, vsProp)

    % Capture the figure as a frame
    frame = getframe(gcf);
    % Write the frames to the video file
    writeVideo(writerObj, frame);

    % saveas(head, fullfile(figurePath, ['single regressor nested fstat shuffled ', analyzeBhv{iBhv}]), 'pdf')
    % Pause to control the speed (optional)
    pause(2);
    delete(head)

end
% toc
% Close the video file
close(writerObj);


%%
% Which neurons were significant throughout each behavior?

fStatSig = zeros(length(analyzeBhv), size(fStatZ, 2));
for iBhv = 1 : length(analyzeBhv)

    fStatSig(iBhv, :) = fStatZ(iBhv, :) <= -3;


    % m23SigNrnInd = fStatZ(fStatZInd, idM23) <= -3;
    % m56SigNrnInd = fStatZ(fStatZInd, idM56) <= -3;
    % dsSigNrnInd = fStatZ(fStatZInd, idDS) <= -3;
    % vsSigNrnInd = fStatZ(fStatZInd, idVS) <= -3;
end

sigNeurons = sum(fStatSig, 1) > 0; % B/c all neurons are either significan across all behaviors or no behaviors
sigM23 = sigNeurons & strcmp(areaLabels, 'M23');
sigM56 = sigNeurons & strcmp(areaLabels, 'M56');
sigDS = sigNeurons & strcmp(areaLabels, 'DS');
sigVS = sigNeurons & strcmp(areaLabels, 'VS');













%%
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  2. use relative explained variance of time regressors for each behavior.
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%  For a given behavior, calculate relative explained variance of each time
%  regressor relative to all other time regressors Within That Behavior.
%  End up with, for each time regressor, a distribution of relative explained variance for each brain
%  area.
%  Ask at what point (for which time regressor) do the distributions
%  of relative explained variances differ (and in which direction).
tic
opts.withinOrAcrossBhv = 'across';
relativeExplained = rrm_relative_r_squared(fullR, regLabels, dataMatZ, analyzeBhv, opts);
toc

%% Save copies to local computer and Z drive
save(fullfile(saveDataPath, 'single regressor relative explained.mat'), 'relativeExplained', 'idLabels', 'areaLabels', 'regLabels', '-mat')

pathZ = rrm_get_paths('Z');
saveZ = fullfile(pathZ.saveDataPath, animal, sessionSave, ['start ' num2str(opts.collectStart), ' for ', num2str(opts.collectFor)]);
if ~exist(saveZ, 'dir')
    mkdir(saveZ);
end
save(fullfile(saveZ , 'single regressor relative explained.mat'), 'relativeExplained', 'idLabels', 'areaLabels', 'regLabels', '-mat')

%%
rel = load(fullfile(saveDataPath, 'single regressor relative explained.mat'));


%%
if makeVideo
    % Define the filename for the output video
    outputVideoFile = fullfile(figurePath, 'relative Rsquared per neuron.mp4');

    % Create a VideoWriter object
    writerObj = VideoWriter(outputVideoFile, 'MPEG-4');
    writerObj.FrameRate = 1;  % Set the frame rate (frames per second)

    % Open the video file for writing
    open(writerObj);
end

fig = figure(99);

% clf
set(fig, 'PaperOrientation', 'landscape','Position', [50, 50, 1000, 800])
% hold on
nDeviations = 2;

relExpSig = zeros(size(relativeExplained));
for iNrn = 1 : size(dataMatZ, 2)
    iRelExp = relativeExplained(:, iNrn)'

    % Use a metric to determine which relative contributions are
    % significant
    var = std(iRelExp) / sqrt(length(analyzeBhv));
    % sigRel = mean(iRelExp) + nDeviations * var;
    sigRel = .05;
    sigRel = 1/size(relativeExplained, 1);  % Use criterion that a relative explained needs to be > (1 / number of behaviors)
    analyzeBhv(iRelExp > sigRel)
    relExpSig(:, iNrn) = iRelExp > sigRel; % For each neuron, which behaviors have a "significant" relative explained variance?

    head = bar(iRelExp);
    % hold on;
    yline(sigRel, '--k')

    xLabels = analyzeBhv;
    ylim([0 1])
    % Set x-axis ticks and labels
    set(gca, 'XTick', 1:length(analyzeBhv), 'XTickLabel', xLabels);
    title(['Relatiave explained variance neuron ', areaLabels{iNrn}, ' ', num2str(idLabels(iNrn))])
    if savePlot
        saveas(fig, fullfile(figurePath, ['relative contributions ', areaLabels{iNrn}, ' ', num2str(idLabels(iNrn))]), 'pdf')
    end


    if makeVideo
        % Capture the figure as a frame
        frame = getframe(gcf);
        % Write the frames to the video file
        writeVideo(writerObj, frame);
    end

    pause(1)
    delete(head)

end
if makeVideo
    % Close the video file
    close(writerObj);
end





%% Determine which neurons are "good" to use to analyze beta values over time regressors within each behavior
% Accept a neuron for a given behavior if it 1) has high relative explained
% variance (rrm_relative_r_squared.m), and 2) has significant nested F-statistic
% (rrm_nested_fstat_suffled.m)

SigAndRelExp = fStatSig & relExpSig;

fprintf('\nFstat\n')
[sum(fStatSig(:,idM23), 2), ...
    sum(fStatSig(:,idM56), 2), ...
    sum(fStatSig(:,idDS), 2), ...
    sum(fStatSig(:,idVS), 2)]

fprintf('\nrelativeExplained\n')
[sum(relExpSig(:,idM23), 2), ...
    sum(relExpSig(:,idM56), 2), ...
    sum(relExpSig(:,idDS), 2), ...
    sum(relExpSig(:,idVS), 2)]

fprintf('\nboth\n')
[sum(SigAndRelExp(:,idM23), 2), ...
    sum(SigAndRelExp(:,idM56), 2), ...
    sum(SigAndRelExp(:,idDS), 2), ...
    sum(SigAndRelExp(:,idVS), 2)]

%%
useNeurons = relExpSig;

useidM23 = relExpSig


idM23(logical(relExpSig(1,idM23)))




























%%
% Which behaviors to plot
analyzeBhv = behaviors(2:end);



%%  Neural activity (z-scored for each neuron) for each behavior
warning('You are using mean and std within each behavioral time window to z-score - what might be better?')
spikeCounts = cell(length(analyzeBhv), size(dataMatZ, 2)); % collects the spike counts surrounding each behavior start
spikeZ = cell(length(analyzeBhv), size(dataMatZ, 2)); % calculates z-score of the spike counts

for iBhv = 1 : length(analyzeBhv)
    iReg = strcmp(['0 ',analyzeBhv{iBhv}], regressorLabels); % find regressor at start of behavior
    iStartFrames = find(bhvDesign(:,iReg)); % every frame where the behavior starts

    for jNeur = 1 : size(dataMatZ, 2)
        for k = 1 : length(iStartFrames)
            iRange = iStartFrames(k) - opts.mPreTime/opts.frameSize : iStartFrames(k) + opts.mPostTime/opts.frameSize;
            if iStartFrames(k) - opts.mPreTime/opts.frameSize > 0 && iStartFrames(k) + opts.mPostTime/opts.frameSize < size(dataMatZ, 1)
                spikeCounts{iBhv, jNeur} = [spikeCounts{iBhv, jNeur}; dataMatZ(iRange, jNeur)'];
                if sum(dataMatZ(iRange, jNeur))
                    spikeZ{iBhv, jNeur} = [spikeZ{iBhv, jNeur}; (dataMatZ(iRange, jNeur)' - mean(dataMatZ(iRange, jNeur))') / std(dataMatZ(iRange, jNeur))];
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
% [Vfull, fullBeta, ~, fullIdx, fullRidge, fullLabels] = crossValModel(fullR, dataMatZ, regLabels, regIdx, regLabels, opts.folds);
% [Vfull, fullBeta, ~, fullIdx, fullRidge, fullLabels] = crossValModel(fullR, Vc, regLabels, regIdx, regLabels, opts.folds);
% save([fPath 'cvFull.mat'], 'Vfull', 'fullBeta', 'fullR', 'fullIdx', 'fullRidge', 'fullLabels'); %save some results

randIdx = randperm(size(dataMatZ, 1));
foldCnt = floor(size(dataMatZ, 1) / opts.folds);
cBeta = cell(1,opts.folds);
yCVModel = zeros(size(dataMatZ));

for iFold = 1 : opts.folds
    dataIdx = true(size(dataMatZ, 1), 1);

    dataIdx(randIdx((iFold - 1) * foldCnt + (1:foldCnt))) = false;

    if iFold == 1
        [cRidge, cBeta{iFold}] = ridgeMML(dataMatZ(dataIdx,:), fullR(dataIdx,:), true); %get beta weights and ridge penalty for first fold
    else
        [~, cBeta{iFold}] = ridgeMML(dataMatZ(dataIdx,:), fullR(dataIdx,:), true, cRidge); %get beta weights and ridge penalty for remaining folds, using same ridge penalties
    end
    yCVModel(~dataIdx, :) = fullR(~dataIdx,:) * cBeta{iFold};
end








