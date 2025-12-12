function [fStat, fStatShuffle, fStatZ] = rrm_fstat_shuffled(fullR, dataMat, opts)

% Compares a ridge regression F-stat with F-stats from 1000 randomly shuffled data matrices, for each neuron/signal

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

nIter = opts.nIter;

% Run once with real neural data
% --------------------------------------------------------
[~, dimBeta] = ridgeMML(dataMat, fullR, true); %get ridge penalties and beta weights.

% Predict neural data
yModel = (fullR - mean(fullR, 1)) * dimBeta;

% Sum of squared errors
SSRes = sum((dataMat - yModel) .^2, 1);
SSTot = sum((dataMat - mean(dataMat, 1)) .^2, 1);

% Explained Variance
explained_variance = SSTot - SSRes;

% Degrees of Freedom (Residual & Explained)
df_residual = size(dataMat, 1) - (size(fullR, 2) + 1);
df_explained = size(fullR, 2);

% Calculate F-statistic
fStat = (explained_variance ./ df_explained) ./ (SSRes ./ df_residual);



% rng('default')
rng(12)
% Run 1000 times wth shuffled neural data
% --------------------------------------------------------
fStatShuffle = zeros(nIter, size(dataMat, 2));
iRandom = 1 : size(dataMat, 1);

shuffleInd = zeros(size(dataMat, 1), nIter);
% Pre-allocate a matrix of shuffled indices
for iShuffle = 1 : nIter
    randShift = randi([1 size(dataMat, 1)], 1);

    % Shuffle the data by moving the last randShift elements to the front
    lastNElements = iRandom(end - randShift + 1:end);  % Extract the last randShift elements
    iRandom(randShift+1:end) = iRandom(1:end-randShift); % Shift the remaining elements to the end
    iRandom(1:randShift) = lastNElements; % Place the last n elements at the beginning
    shuffleInd(:, iShuffle) = iRandom;
end

% tic
poolID = parpool(4, 'IdleTimeout', Inf);
% toc
tic
parfor iShuffle = 1 : nIter
    % for iShuffle = 1 : nIter
    % disp(['shuffle ', num2str(iShuffle)])
    % tic
    dataMatShuffle = dataMat(shuffleInd(:, iShuffle), :);

    % sum(dataMatShuffle, 1)
    % Perform the shuffled data regression
    [~, dimBetaShuffle] = ridgeMML(dataMatShuffle, fullR, true); %get ridge penalties and beta weights.

    % Predict neural data with partial model
    yModelShuffle = (fullR - mean(fullR, 1)) * dimBetaShuffle;

    % Sum of squared errors
    SSRes = sum((dataMatShuffle - yModelShuffle) .^2, 1);
    SSTot = sum((dataMatShuffle - mean(dataMatShuffle, 1)) .^2, 1);

    % Explained Variance
    explained_variance = SSTot - SSRes;

    % Degrees of Freedom (Residual & Explained)
    df_residual = size(dataMatShuffle, 1) - (size(fullR, 2) + 1);
    df_explained = size(fullR, 2);

    % Calculate F-statistic
    fStatShuffle(iShuffle, :) = (explained_variance ./ df_explained) ./ (SSRes ./ df_residual);
    if toc > 30*60
        disp(['shuffle ', num2str(iShuffle)])
        delete(poolID)
        error('Must be stuck!')
    end
    % toc
end %iShuffle
delete(poolID)
toc

% Is the F statistic different from randomly shuffled data?
fStatShuffleStd = std(fStatShuffle, 1);
fStatZ = (fStat - mean(fStatShuffle, 1)) ./ fStatShuffleStd; % How many std above random shuffled data is the real fStat?



end