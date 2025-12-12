%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  Perform separate regressions for each of the behaviors, including time
%  regressors
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



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


%%
% opts = neuro_behavior_options;
animal = 'ag25290';
sessionBhv = '112321_1';
sessionNrn = '112321';

if strcmp(sessionBhv, '112321_1')
    sessionSave = '112321';
end





%%
bhvDataPath = strcat(paths.bhvDataPath, 'animal_',animal,'/');
bhvFileName = ['behavior_labels_', animal, '_', sessionBhv, '.csv'];


opts.dataPath = bhvDataPath;
opts.fileName = bhvFileName;

dataBhv = rrm_load_data(opts, 'behavior');
frameIdx = 1 : sum(dataBhv.Dur);



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

bhvView = behaviors(~rmvBhv);




%% Get design matrix for behaviors
[bhvDesign, regressorLabels] = rrm_design_matrix_behavior(dataBhv, opts);


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

data = rrm_load_data(opts, 'neuron');
data.bhvDur = dataBhv.Dur;
clusterInfo = data.ci;
spikeTimes = data.spikeTimes;
spikeClusters = data.spikeClusters;


%% Find the neuron clusters (ids) in each brain region

% 0 - 500  motor l2/3
% 500 - 1240 l5/6
% 1240 - 1540 corpus callosum, where little neural activity expected
% 1540 - 2700 dorsal striatum
% 2700 - 3840 ventral striatum

m23 = [0 499];
m56 = [500 1239];
cc = [1240 1539];
ds = [1540 2699];
vs = [2700 3839];

allGood = strcmp(data.ci.group, 'good') & strcmp(data.ci.KSLabel, 'good');
allM23 = data.ci.depth >= m23(1) & data.ci.depth <= m23(2);
allM56 = data.ci.depth >= m56(1) & data.ci.depth <= m56(2);
allCC = data.ci.depth >= cc(1) & data.ci.depth <= cc(2);
allDS = data.ci.depth >= ds(1) & data.ci.depth <= ds(2);
allVS = data.ci.depth >= vs(1) & data.ci.depth <= vs(2);

goodM23 = allGood & allM23;
goodM56= allGood & allM56;
goodDS = allGood & allDS;
goodVS = allGood & allVS;

opts.useNeurons = find(allGood & ~allCC);
% opts.useNeurons = find(allGood & allDS);

%%
% Make neural matrix or load design matrix
[dataMat, idLabels, rmvNeurons] = rrm_neural_matrix(data, opts);
%%

saveDataPath = strcat(paths.saveDataPath, animal,'/', sessionNrn, '/');
save(fullfile(savePath,'neural_matrix.mat'), 'dataMat', 'idLabels', 'rmvNeurons')
%%
load(fullfile(savePath,'neural_matrix.mat'), 'dataMat', 'idLabels', 'rmvNeurons')

%%
idM23 = intersect(idLabels, data.ci.cluster_id(goodM23));
idM56 = intersect(idLabels, data.ci.cluster_id(goodM56));
idDS = intersect(idLabels, data.ci.cluster_id(goodDS));
idVS = intersect(idLabels, data.ci.cluster_id(goodVS));

% What are the indices of neurons in each brain area within the dataMat
matIdxM23 = 1 : length(idM23);
matIdxM56 = 1 + length(matIdxM23) : length(matIdxM23) + length(idM56);
matIdxDS = 1 + length(matIdxM23) + length(matIdxM56) : length(matIdxM23) + length(matIdxM56) +  length(idDS);
matIdxVS = 1 +  length(matIdxM23) + length(matIdxM56) +  length(matIdxDS) : size(dataMat, 2);

% What index does each area begin
m23IdxStart = 1;
m56IdxStart = 1 + length(idM23);
dsIdxStart = m56IdxStart + length(idM56);
vsIdxStart = dsIdxStart + length(idDS);

areaLabels = [repmat({'M23'}, 1, length(idM23)), repmat({'M23'}, 1, length(idM56)), repmat({'M23'}, 1, length(idDS)), repmat({'M23'}, 1, length(idVS))];

fprintf('%d M23\n%d M56\n%d DS\n%d VS\n', length(idM23), length(idM56), length(idDS), length(idVS))
%%

% remove the frames you removed from the design matrix
dataMat(removeRows,:) = [];

imagesc(dataMat')
hold on;
line([0, size(dataMat, 1)], [length(idM23),(length(idM23))], 'Color', 'r');
line([0, size(dataMat, 1)], [length(idM23)+length(idM56),(length(idM23)+length(idM56))], 'Color', 'r');
line([0, size(dataMat, 1)], [length(idM23)+length(idM56)+length(idDS),(length(idM23)+length(idM56)+length(idDS))], 'Color', 'r');


% %% Normalize and zero-center the neural data matrix for the regression
%
% % Normalize (z-score)
dataMatStd = std(dataMat, 0, 1);
dataMatMean = mean(dataMat, 1);
dataMatZ = zscore(dataMat, 0, 1);
%



























%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                          Run the regressions for the single behaviors
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
tic
for iBhv = 9 : length(bhvView)
iBhv
bhvView{iBhv}

    % Initialize a logical array to find relevant regressors
    containsReg = false(size(regressorLabels));

    % Loop through the regressors to check for the set corresponding to
    % that behavior
    for rIdx = 1:numel(regressorLabels)
        if ischar(regressorLabels{rIdx}) && contains(regressorLabels{rIdx}, bhvView{iBhv})
            containsReg(rIdx) = true;
        end
    end

    iDesignMat = fullR(:, containsReg);
    [ridgeVals, dimBeta] = ridgeMML(dataMatZ, iDesignMat, true); %get ridge penalties and beta weights.

    yModel = (iDesignMat - mean(iDesignMat, 1)) * dimBeta;

    % Explained variance per neuron
    plotFlag = 1;
    savePlot = 0;

    SSR = sum((dataMatZ - yModel) .^2, 1);
    SSE = sum((dataMatZ - mean(dataMatZ)) .^2, 1);
    rSquaredFull = 1 - SSR ./ SSE;

    if plotFlag
        nRow = 2;
        nColumn = 1;
        orientation = 'portrait';
        figureHandle = 33;
        [axisWidth, axisHeight, xAxesPosition, yAxesPosition] = standard_figure(nRow, nColumn, orientation, figureHandle);
        ax(1) = axes('units', 'centimeters', 'position', [xAxesPosition(1, 1) yAxesPosition(1, 1) axisWidth axisHeight]);
        % rFig = figure('Position', get(0, 'ScreenSize'));
        hold on;
        % plot(ax(1), rSquaredFull)
        bar(ax(1), rSquaredFull)
        
        ylim([0 inf])
        xline(length(idM23), 'LineWidth', 3)
        xline(length(idM23) + length(idM56), 'LineWidth', 3)
        xline(length(idM23) + length(idM56) + length(idDS), 'LineWidth', 3)
        title(ax(1), [bhvView{iBhv}, ' R-squared for each neuron'], 'Interpreter', 'none')
        xlabel(ax(1), 'M23 -> M56 -> DS -> VS')
        if savePlot
            saveas(figureHandle,fullfile(figurePath, 'r-Squared per neuron ',bhvView{iBhv}), 'pdf')
        end
    end
pause(3)

    Fstat = (SSR ./ dfNum) ./ (SSE ./ dfDenom);

    % Calculate the p-value
    pVal = 1 - fcdf(Fstat, dfNum, dfDenom);

end

toc
%% Try matlab's built-in regression functions

iBhv = 13;
    % Initialize a logical array to find relevant regressors
    containsReg = false(size(regressorLabels));

    % Loop through the regressors to check for the set corresponding to
    % that behavior
    for rIdx = 1:numel(regressorLabels)
        if ischar(regressorLabels{rIdx}) && contains(regressorLabels{rIdx}, bhvView{iBhv})
            containsReg(rIdx) = true;
        end
    end

    iDesignMat = fullR(:, containsReg);

% % Lasso
%     iNrn = 100;
% [B,FitInfo] = lasso(iDesignMat, dataMatZ(:,iNrn),'Intercept',false);
% 
% yModel = (iDesignMat - mean(iDesignMat, 1)) * B;


% Specify a range of ridge parameters
ridgeParams = logspace(-6, 6, 13);

bestLambda = zeros(size(dataMatZ, 2));
for iNrn = 45%1 : size(dataMatZ, 2)
% Initialize variables to store cross-validated performance
cvMSE = zeros(size(ridgeParams));

% Perform cross-validated ridge regression
for i = 1:length(ridgeParams)
    k = ridgeParams(i);

    % Ridge regression with k
    beta = ridge(dataMatZ(:,iNrn), iDesignMat, k);

    % Predict using the current model
    y_pred = (iDesignMat - mean(iDesignMat, 1)) * beta;
    
    % Calculate mean squared error (MSE)
    cvMSE(i) = mean((dataMatZ(:,iNrn) - y_pred).^2);
end

% Find the optimal ridge parameter
[~, idxOptimal] = min(cvMSE);
bestLambda(iNrn) = ridgeParams(idxOptimal);

% Plot cross-validated mean squared error (MSE) against ridge parameters
figure(10);
semilogx(ridgeParams, cvMSE, 'o-');
xlabel('Ridge Parameter (\lambda)');
ylabel('Cross-Validated MSE');
title('Cross-Validated Ridge Regression');
grid on;

% Display the optimal ridge parameter
fprintf('Optimal Ridge Parameter: %e\n', bestLambda(iNrn));

end


%% Explained variance

figurePath = strcat('E:/Projects/ridgeRegress/docs/', animal, '/', sessionSave, '/', string(datetime('today', 'format', 'yyyy-MM-dd')), '/figures/', ['start ' num2str(opts.collectStart), ' for ', num2str(opts.collectFor)]);
figurePath = strcat(paths.figurePath, animal, '/', sessionSave, '/figures/', ['start ' num2str(opts.collectStart), ' for ', num2str(opts.collectFor)]);
if ~exist(figurePath, 'dir')
    mkdir(figurePath);
    % addpath(figurePath)
end


% total
%reconstruct data and compute R^2
% yModel = (fullR * dimBeta);
% fullRStd = std(fullR, 0, 1);
% fullR = bsxfun(@rdivide, fullR, fullRStd);

yModel = (fullR - mean(fullR, 1)) * dimBeta;

%
% R squared
% = (SSE / SSE + SSR)

SSR = sum((dataMatZ(:) - yModel(:)) .^2);
SSE = sum((dataMatZ(:) - mean(dataMatZ(:))) .^2);
rSquaredFull = 1 - (SSR / SSE);
rSquaredFull

%
% F statistic
% = (SSR / numRegressors) / (SSE / (numDataPoints - numRegressors - 1))

dfNum = size(fullR, 2);
dfDenom = (size(dataMatZ, 1) - size(fullR,2) - 1);

Fstat = (SSR / dfNum) / (SSE / dfDenom)
% dfModel = size(fullR, 2);
% dfTotal = size(dataMatZ, 1) - 1;
% dfRedisdual = dfTotal - dfModel;

% Calculate the p-value
pVal = 1 - fcdf(Fstat, dfNum, dfDenom)











%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% To compare when different brain areas become important:

% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 0. Compare betas over time for each behavior, comparing across brain
% areas (when do betas in each brain increase/decrease first, last, etc?)
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Which behaviors to plot

% Which behaviors are in this analysis?

%%
% bhvView = behaviors(2:end);


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
        meanM23Beta = mean(iRegSection(:,matIdxM23), 2);
        scatter(1, meanM23Beta, 'r', 'lineWidth', 3)
        meanM56Beta = mean(iRegSection(:,matIdxM56), 2);
        scatter(2, meanM56Beta, 'm', 'lineWidth', 3)
        meanDSBeta = mean(iRegSection(:,matIdxDS), 2);
        scatter(3, meanDSBeta, 'b', 'lineWidth', 3)
        meanVSBeta = mean(iRegSection(:,matIdxVS), 2);
        scatter(4, meanVSBeta, 'c', 'lineWidth', 3)

        legend('M23','M56','DS','VS')
        title(bhvView{iBhv}, 'Interpreter', 'none')
        plot(xlim, [0 0], '--k')
        if savePlot
            saveas(figureHandle,fullfile(figurePath, ['single regressor meanBetas ', bhvView{iBhv}]), 'pdf')
            pause(15)
        end
        clf
        hold on
    end


    % for jNrn = 1 : size(iRegSection, 2)
    %     plot(iRegSection(:,jNrn), 'linewidth', 2)
    %     hold on
    %     plot(xlim, [0 0], '--k')
    %     if savePlot
    %         if jNrn <= length(matIdxM23)
    %             areaLabel = 'M23';
    %         elseif jNrn > length(matIdxM23) && jNrn <= length(matIdxM23) + length(matIdxM56)
    %             areaLabel = 'M56';
    %         elseif jNrn > length(matIdxM23) + length(matIdxM56) && jNrn <= length(matIdxM23) + length(matIdxM56) + length(matIdxDS)
    %             areaLabel = 'DS';
    %         else
    %             areaLabel = 'VS';
    %         end
    %     pause(15)
    %
    %         saveas(figureHandle,fullfile(figurePath, ['betas ', bhvView{iBhv}, ' ', areaLabel, ' ',num2str(jNrn)]), 'pdf')
    %     end
    %
    %     iRegSection(:,jNrn)
    %     % pause(10)
    %     clf
    % end


    %
    % zBeta{iBhv} = (iRegSection - mean(dimBeta, 1)) ./ std(dimBeta, 1);
    %
    % beyondStd1 = zBeta{iBhv} >= 1 | zBeta{iBhv} <= -1;
    %
    % for jNrn = 1 : size(iRegSection, 2)
    %
    %     nonzeroRows = find(beyondStd1(:, jNrn) ~= 0);
    %     if ~isempty(nonzeroRows)
    %         % Find the first two consecutive numbers
    %         for k = 1:(length(nonzeroRows) - 1)
    %             if nonzeroRows(k) == nonzeroRows(k + 1) - 1
    %                 firstConsecutiveNumber = nonzeroRows(k);
    %                             firstIdx(iBhv, jNrn) = firstConsecutiveNumber;
    %
    %                 break;  % Exit the loop when a pair is found
    %             end
    %         end
    %
    %     end
    %
    %     plot(zBeta{iBhv}(:,jNrn))
    %     hold on
    %     if ~isnan(firstIdx(iBhv, jNrn))
    %         plot([firstIdx(iBhv, jNrn) firstIdx(iBhv, jNrn)], [-1 1], 'k')
    %     end
    %     plot(xlim, [0 0], '--k')
    %     zBeta{iBhv}(:,jNrn)
    %     clf
    % end



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
        % meanM23Beta = mean(iRegSection(:,matIdxM23), 2);
        % plot(meanM23Beta, 'r', 'lineWidth', 3)
        hold on
        plot([0 20], [0 0], '--k')

        nonNanM23 = ~isnan(firstIdx(iBhv,matIdxM23));
        % plot(mean(zBeta{iBhv}(:, matIdxM23), 2), 'r', 'linewidth', 2)
        plot(mean(zBeta{iBhv}(:, matIdxM23(nonNanM23)), 2), 'r', 'linewidth', 2)
        firstM23 = mean(firstIdx(iBhv,matIdxM23), 'omitnan');
        plot([firstM23 firstM23], ylim, 'r', 'linewidth', 2)

        nonNanM56 = ~isnan(firstIdx(iBhv,matIdxM56));
        % plot(mean(zBeta{iBhv}(:, matIdxM56), 2), 'm', 'linewidth', 2)
        plot(mean(zBeta{iBhv}(:, matIdxM56(nonNanM56)), 2), 'm', 'linewidth', 2)
        firstM56 = mean(firstIdx(iBhv,matIdxM56), 'omitnan');
        plot([firstM56 firstM56], ylim, 'm', 'linewidth', 2)

        nonNanDS = ~isnan(firstIdx(iBhv,matIdxDS));
        % plot(mean(zBeta{iBhv}(:, matIdxDS), 2), 'b', 'linewidth', 2)
        plot(mean(zBeta{iBhv}(:, matIdxDS(nonNanDS)), 2), 'b', 'linewidth', 2)
        firstDS = mean(firstIdx(iBhv,matIdxDS), 'omitnan');
        plot([firstDS firstDS], ylim, 'b', 'linewidth', 2)

        nonNanVS = ~isnan(firstIdx(iBhv,matIdxVS));
        % plot(mean(zBeta{iBhv}(:, matIdxVS), 2), 'c', 'linewidth', 2)
        plot(mean(zBeta{iBhv}(:, matIdxVS(nonNanVS)), 2), 'c', 'linewidth', 2)
        firstVS = mean(firstIdx(iBhv,matIdxVS), 'omitnan');
        plot([firstVS firstVS], ylim, 'c', 'linewidth', 2)

        clf
    end
    fprintf('\n%s\n',bhvView{iBhv})
    fprintf('M23: %.2f\nM56: %.2f\nDS: %.2f\nVS: %.2f\n', firstM23, mean(firstIdx(iBhv,matIdxM56), 'omitnan'), mean(firstIdx(iBhv,matIdxDS), 'omitnan'), mean(firstIdx(iBhv,matIdxVS), 'omitnan'))

end



%% Plot alignments of first index when betas "turn on" for each behavior and brain area
for iBhv = 1 : length(bhvReg)
    iBhv
    iRegSection = dimBeta(1 + bhvReg(iBhv) : 20 + bhvReg(iBhv), :);
    % iRegSection = abs(iRegSection);

    if plotFlag
        meanM23Beta = mean(iRegSection(:,matIdxM23), 2);
        plot(meanM23Beta, 'r', 'lineWidth', 3)

        meanM56Beta = mean(iRegSection(:,matIdxM56), 2);
        plot(meanM56Beta, 'm', 'lineWidth', 3)

        meanDSBeta = mean(iRegSection(:,matIdxDS), 2);
        plot(meanDSBeta, 'b', 'lineWidth', 3)

        meanVSBeta = mean(iRegSection(:,matIdxVS), 2);
        plot(meanVSBeta, 'c', 'lineWidth', 3)

        legend('M23','M56','DS','VS')
        title(bhvView{iBhv}, 'Interpreter', 'none')
        plot(xlim, [0 0], '--k')


        firstM23 = mean(firstIdx(iBhv,matIdxM23), 'omitnan');
        plot([firstM23 firstM23], ylim, 'r')

        firstM56 = mean(firstIdx(iBhv,matIdxM56), 'omitnan');
        plot([firstM56 firstM56], ylim, 'm')

        firstDS = mean(firstIdx(iBhv,matIdxDS), 'omitnan');
        plot([firstDS firstDS], ylim, 'b')

        firstVS = mean(firstIdx(iBhv,matIdxVS), 'omitnan');
        plot([firstVS firstVS], ylim, 'c')

        if savePlot
            saveas(figureHandle,fullfile(figurePath, ['meanBetas ', bhvView{iBhv}]), 'pdf')
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
% optsNest.randShift = floor(size(dataMatZ, 1) / optsNest.nIter); % How many time frames to shift with each randomization step
optsNest.randShift = 7; % How many time frames to shift with each randomization step
fStatZ = zeros(length(regLabels), size(dataMatZ, 2));
nReg = length(regLabels);

% poolID = parpool(4, 'IdleTimeout', Inf);
% parfor iReg = 1 : 4%length(regLabels)
for iReg = 10 : 10
    regLabels{iReg}
    % optsNest.removeReg = iReg;

    fStatZ(iReg, :) = rrm_nested_fstat_shuffled(fullR, dataMatZ, SSR, iReg, optsNest);

end % iReg
% delete(poolID)
find(fStatZ(10,:) > 330)
toc
%%

save(fullfile(saveDataPath, 'single regressor nested fstat shuffled.mat'), 'fStatZ')
%%
load(fullfile(saveDataPath, 'single regressor nested fstat shuffled.mat'))

%%
figurePath = strcat(paths.figurePath, animal, '/', sessionSave, '/', string(datetime('today', 'format', 'yyyy-MM-dd')), '/figures/', ['start ' num2str(opts.collectStart), ' for ', num2str(opts.collectFor)]);
if ~exist(figurePath, 'dir')
    mkdir(figurePath);
    % addpath(figurePath)
end

figure(66);
for iBhv = 1 : length(bhvView)
    fStatZInd = iBhv;  % which regressors belong to this behavior?
    clf


    % plot(fStatZ(iReg,:), 'LineWidth', 2)
    head = scatter((1:size(dataMatZ, 2)), fStatZ(fStatZInd,:), 100, 'k', 'filled');
    hold on
    bhvTitle = [bhvView{iBhv}];
    title(bhvTitle)
    xline(m56IdxStart, 'LineWidth', 2)
    xline(dsIdxStart, 'LineWidth', 2)
    xline(vsIdxStart, 'LineWidth', 2)
    plot(xlim, [-3 -3])

    m23Prop = sum(fStatZ(fStatZInd, matIdxM23) <= -3) / length(matIdxM23);
    m56Prop = sum(fStatZ(fStatZInd, matIdxM56) <= -3) / length(matIdxM56);
    dsProp = sum(fStatZ(fStatZInd, matIdxDS) <= -3) / length(matIdxDS);
    vsProp = sum(fStatZ(fStatZInd, matIdxVS) <= -3) / length(matIdxVS);
    fprintf('\nM23: %.2f\nM56: %.2f\nDS: %.2f\nVS: %.2f\n', m23Prop, m56Prop, dsProp, vsProp)

    % drawnow;
    %
    % % Capture the frame
    % frames(iReg) = getframe(gcf);


    saveas(head, fullfile(figurePath, ['single regressor nested fstat shuffled ', bhvView{iBhv}]), 'pdf')
    % Pause to control the speed (optional)
    pause(1);
    delete(head)

    %     % Define the filename for the output video
    % outputVideoFile = fullfile(figurePath, ['f-stat per behavior time regressors ', bhvView{iBhv}, '.mp4']);
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

allSigNrnInd = zeros(length(bhvView), size(fStatZ, 2));
for iBhv = 1 : 3%length(bhvView)


    iFirstRegInd = 20 * (iBhv-1) + 1;
    iLastRegInd = iFirstRegInd + 19;

    % fStatZInd = 20 * (iBhv-1) + iReg; % which regressors belong to this behavior?

    allSigNrnInd(iBhv, :) = sum(fStatZ(iFirstRegInd : iLastRegInd, :) <= -3, 1) >= nSigCriterion;


    % m23SigNrnInd = fStatZ(fStatZInd, matIdxM23) <= -3;
    % m56SigNrnInd = fStatZ(fStatZInd, matIdxM56) <= -3;
    % dsSigNrnInd = fStatZ(fStatZInd, matIdxDS) <= -3;
    % vsSigNrnInd = fStatZ(fStatZInd, matIdxVS) <= -3;
end














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
relativeExplained = rrm_relative_r_squared(fullR, regLabels, dataMatZ, bhvView, 'across');
toc

%%
fig = figure;
set(fig, 'PaperOrientation', 'landscape','Position', [50, 50, 1000, 800])
hold on
xLabels = bhvView;
% Set x-axis ticks and labels
set(gca, 'XTick', 1:length(bhvView), 'XTickLabel', xLabels);

for iNrn = 1 : size(dataMatZ, 2)

    bar(relativeExplained(:, iNrn));
    title(['Relatiave explained variance neuron ', areaLabels{iNrn}, ' ', num2str(idLabels(iNrn))])
    saveas(fig, fullfile(figurePath, ['relative contributions ', areaLabels{iNrn}, ' ', num2str(idLabels(iNrn))]), 'pdf')
    pause(3)
    clf
end

% % Loop through all the regressors to get explained variance without each
% % regressor
% % relContribution = zeros(size(fullR, 2), size(dataMatZ, 2)); % matrix with dimensions #regressors X #neurons
% rSquaredPartial = zeros(size(fullR, 2), size(dataMatZ, 2));
% for iReg = 1 : size(fullR, 2)
%
%     % Remove a regressor
%     removeReg = zeros(size(fullR, 2), 1);
%     removeReg(iReg) = 1;
%
%     % Run regression with partial model
%     partialR = fullR(:, ~removeReg);
%     [ridgeVals, dimBetaPartial] = ridgeMML(dataMatZ, partialR, true); %get ridge penalties and beta weights.
%
%     % Predict neural data with partial model
%     yModelPartial = (partialR * dimBetaPartial);
%
%     % Get explained variance of partial model
%     SSR = sum((dataMatZ - yModelPartial) .^2, 1);
%     SSE = sum((dataMatZ - mean(dataMatZ)) .^2, 1);
%     rSquaredPartial(iReg, :) = 1 - SSR ./ SSE;
%
% end
%
% %%
% % For each behavior, get explained variance of each regressor relative to all other time
% % regressors in that behavior
% relativeExplained = zeros(size(bhvDesign, 2), size(dataMatZ, 2));
% for iBhv = 1 : length(bhvView)
%     iReg = strcmp(['0 ',bhvView{iBhv}], regLabels); % find regressor at start of behavior
%     iRange = iReg - opts.mPreTime/opts.frameSize : iReg + opts.mPostTime/opts.frameSize;
%
%     % Loop through each regressor in that behavior to calculate its
%     % relative contribution
%     for jRel = iRange
%         jNum = (rSquaredFull - rSquaredPartial(jRel, :)) ./ rSquaredFull;
%         jDen = (rSquaredFull - sum(rSquaredPartial(iRange, :), 1)) ./ rSquaredFull;
% relativeExplained(jRel, :) = jNum ./ jDen;
%
%     end %jRel
%
%     % Make sure each relative contribution sums to 1 for each behavior
% plot(sum(relativeExplained(iRange, :), 1))
%
% end % iBhv

%%
% For each behavior, Plot the mean relative contributions of neurons in
% each brain area, along the time course of the behavior

bhvReg = (0 : 20 : 300);
zBeta = cell(length(bhvReg), 1);
figureHandle = figure(55);
firstIdx = nan(size(zBeta, 1), size(dimBeta, 2));

hold on;
for iBhv = 1 : length(bhvReg)
    iBhv
    iRegSection = relativeExplained(1 + bhvReg(iBhv) : 20 + bhvReg(iBhv), :);
    % iRegSection = abs(iRegSection);


    if plotFlag
        meanM23Rel = mean(iRegSection(:,matIdxM23), 2);
        plot(meanM23Rel, 'r', 'lineWidth', 3)
        meanM56Rel = mean(iRegSection(:,matIdxM56), 2);
        plot(meanM56Rel, 'm', 'lineWidth', 3)
        meanDSRel = mean(iRegSection(:,matIdxDS), 2);
        plot(meanDSRel, 'b', 'lineWidth', 3)
        meanVSRel = mean(iRegSection(:,matIdxVS), 2);
        plot(meanVSRel, 'c', 'lineWidth', 3)

        legend('M23','M56','DS','VS')
        title(bhvView{iBhv}, 'Interpreter', 'none')
        plot(xlim, [0 0], '--k')
        if savePlot
            saveas(figureHandle,fullfile(figurePath, ['relative contributions ', bhvView{iBhv}]), 'pdf')
            pause(15)
        end
        clf
        hold on
    end
end























%% Relative contributions
% For each neuron, how much did each regressor contribute to the explained
% variance?

% relContribution = zeros(size(fullR, 2), size(dataMatZ, 2)); % matrix with dimensions #regressors X #neurons
normDiff = zeros(size(fullR, 2), size(dataMatZ, 2));
for iReg = 1 : size(fullR, 2)

    % Remove a regressor
    removeReg = zeros(size(fullR, 2), 1);
    removeReg(iReg) = 1;

    % Run regression with partial model
    partialR = fullR(:, ~removeReg);
    [ridgeVals, dimBetaPartial] = ridgeMML(dataMatZ, partialR, true); %get ridge penalties and beta weights.

    % Predict neural data with partial model
    yModelPartial = (partialR * dimBetaPartial);

    % Get explained variance of partial model
    SSR = sum((dataMatZ - yModelPartial) .^2, 1);
    SSE = sum((dataMatZ - mean(dataMatZ)) .^2, 1);
    rSquaredPartial = 1 - SSR ./ SSE;



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
for i = 1 : size(dataMatZ, 2)
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













%%
% Which behaviors to plot
bhvView = behaviors(2:end);



%%  Neural activity (z-scored for each neuron) for each behavior
warning('You are using mean and std within each behavioral time window to z-score - what might be better?')
spikeCounts = cell(length(bhvView), size(dataMatZ, 2)); % collects the spike counts surrounding each behavior start
spikeZ = cell(length(bhvView), size(dataMatZ, 2)); % calculates z-score of the spike counts

for iBhv = 1 : length(bhvView)
    iReg = strcmp(['0 ',bhvView{iBhv}], regressorLabels); % find regressor at start of behavior
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
for iBhv = 1 : length(bhvView)

    % Initialize a logical array to find relevant regressors
    containsReg = false(size(regressorLabels));

    % Loop through the regressors to check for the set corresponding to
    % that behavior
    for rIdx = 1:numel(regressorLabels)
        if ischar(regressorLabels{rIdx}) && contains(regressorLabels{rIdx}, bhvView{iBhv})
            containsReg(rIdx) = true;
        end
    end

    % Specify the custom title position (in normalized figure coordinates)
    titleText = bhvView{iBhv};
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
        saveas(figureHandle,fullfile(figurePath, ['betas_and_spikes ', bhvView{iBhv}]), 'pdf')
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








