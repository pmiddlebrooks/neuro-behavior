function [fStatZPartial] = rrm_nested_fstat_shuffled(fullR, dataMat, iRmv, optsNest)

%Nested F-stats for partial models

% F = ((RSSreduced – RSSfull)/p)  /  (RSSfull/n-k)
% where:
% RSSreduced: The residual sum of squares of the reduced (i.e. “nested”) model.
% RSSfull: The residual sum of squares of the full model.
% p: The number of predictors removed from the full model.
% n: The total observations in the dataset.
% k: The number of coefficients (including the intercept) in the full model.

% Input:
% fullR: full design matrix (timeframes X regressors)
% dataMat: zscored data matrix (timeframes X neurons)
% SSR: Residual sum of squares for the full model

nIter = optsNest.nIter;
% randShift = optsNest.randShift;

% Remove a regressor
removeReg = zeros(size(fullR, 2), 1);
% removeReg(optsNest.removeReg) = 1;
removeReg(iRmv) = 1;


% Get partial model design matrtix
partialR = fullR(:, ~removeReg);



% =====================================================================
% 1.   Get nested F-stat of partial model for actual data
% =====================================================================

% Run once with full model (with real neural data)
% --------------------------------------------------------
[~, dimBetaFull] = ridgeMML(dataMat, fullR, true); %get ridge penalties and beta weights.
% Predict neural data with partial model
yModelFull = (fullR - mean(fullR, 1)) * dimBetaFull;
SSRFull = sum((dataMat - yModelFull) .^2, 1);

% Run once with regressor removed (with real neural data)
% --------------------------------------------------------
[~, dimBetaPartial] = ridgeMML(dataMat, partialR, true); %get ridge penalties and beta weights.
% Predict neural data with partial model
% yModelPartial = (partialR * dimBetaPartial);
yModelPartial = (partialR - mean(partialR, 1)) * dimBetaPartial;
SSRPartial = sum((dataMat - yModelPartial) .^2, 1);

nRemoved = size(fullR, 2) - size(partialR, 2);

% High fStatNest means the removed regressor is important for fitting a
% better model (removing it increases the residual errors in the
% regression)
fStatNest = ((SSRPartial - SSRFull) ./ nRemoved) ./ (SSRFull ./ (size(dataMat, 1) - size(fullR, 2)));







% =====================================================================
% 2.  Get distribution of nested F-stat of partial model for shuffled data
% =====================================================================

rng('default')
% Run nIter times with regressor removed, wth shuffled neural data
% --------------------------------------------------------
fStatShuffle = zeros(nIter, size(dataMat, 2));
iRandom = 1 : size(dataMat, 1);

for iShuffle = 1 : nIter
    % tic% iShuffle
    randShift = randi([1 size(dataMat, 1)], 1);
    % iRandom = randperm(size(dataMat, 1));
    % for col = 1:size(dataMat, 2)
    %     dataMatShuffle(:, col) = dataMat(iRandom, col);
    % end

    % Shuffle the data by moving the last randShift elements to the front
    lastNElements = iRandom(end - randShift + 1:end);  % Extract the last randShift elements      
    iRandom(randShift+1:end) = iRandom(1:end-randShift); % Shift the remaining elements to the end       
    iRandom(1:randShift) = lastNElements; % Place the last n elements at the beginning
    dataMatShuffle = dataMat(iRandom, :);

        % Perform the shuffled data regression of the full model
    [~, dimBetaShuffleFull] = ridgeMML(dataMatShuffle, fullR, true); %get ridge penalties and beta weights.
    % Predict neural data with partial model
    % yShufflePartial = (partialR * dimBetaShuffle);
    yShuffleFull = (fullR - mean(fullR, 1)) * dimBetaShuffleFull;
    SSRShuffleFull = sum((dataMatShuffle - yShuffleFull) .^2, 1);
    % SSRShuffle = sum((dataMat - yShufflePartial) .^2, 1);

    % Perform the shuffled data regression of the partial
    [~, dimBetaShuffle] = ridgeMML(dataMatShuffle, partialR, true); %get ridge penalties and beta weights.
    % Predict neural data with partial model
    % yShufflePartial = (partialR * dimBetaShuffle);
    yShufflePartial = (partialR - mean(partialR, 1)) * dimBetaShuffle;
    SSRShufflePartial = sum((dataMatShuffle - yShufflePartial) .^2, 1);
    % SSRShuffle = sum((dataMat - yShufflePartial) .^2, 1);


    fStatShuffle(iShuffle, :) = ((SSRShufflePartial - SSRShuffleFull) ./ nRemoved) ./ (SSRShuffleFull ./ (size(dataMat, 1) - size(fullR, 2)));
% toc
 end %iShuffle




% =====================================================================
% 3.  How many stds worse (greater) is the partial model than the partial shuffled model? 
% =====================================================================


% For this regressor, is the F statistic different from randomly shuffled
% data?
fStatShuffleStd = std(fStatShuffle, 1);
fStatZPartial = (fStatNest - mean(fStatShuffle, 1)) ./ fStatShuffleStd; % Lower fStatZ values mean that regressor is more important than randomly shuffled data.



end