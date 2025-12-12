function [rSquaredPartial, relativeExplained] = rrm_relative_r_squared(designMat, regLabels, dataMatZ, behaviors, opts)
% Return the relative r-squared distributions for time regressors of a list of behaviors

%  For each given behavior, calculate relative explained variance of each time
%  regressor relative to all other time regressors within That Behavior.
%  End up with, for each time regressor, a distribution of relative explained variance for each brain
%  area.
%  Ask at what point (for which time regressor) do the distributions
%  of relative explained variances differ (and in which direction).
%
% See Engelhard et al 2019 Nat Neuro

% input:
% designMat: design matrix for the regression (timeframes X regressors)
% dataMatZ: z-scored matrix of neural data (timeframes X neurons)
% behaviors: list of behaviors to loop through
% withinOrAcrossBhv: return relativeExplained within each beahvior or across all behaviors

% output:
% rSquaredPartial: explained variance of each regressor relative to other regressors in that same behavior: numRegressors X numNeurons
% relativeExplained: explained variance of each regressor relative to other regressors in that same behavior: numRegressors X numNeurons


% Get full model fit to compare for relative contributions
[~, dimBeta] = ridgeMML(dataMatZ, designMat, true); %get ridge penalties and beta weights.
% Predict neural data with full model
yModel = (designMat - mean(designMat, 1)) * dimBeta;

% Get explained variance of full model
SSRes = sum((dataMatZ - yModel) .^2, 1);
SSTot = sum((dataMatZ - mean(dataMatZ)) .^2, 1);
rSquaredFull = 1 - SSRes ./ SSTot;






% Loop through all the regressors to get explained variance without each
% regressor
% relContribution = zeros(size(fullR, 2), size(dataMat, 2)); % matrix with dimensions #regressors X #neurons
rSquaredPartial = zeros(size(designMat, 2), size(dataMatZ, 2));
for iReg = 1 : size(designMat, 2)

    % Remove a regressor
    removeReg = zeros(size(designMat, 2), 1);
    removeReg(iReg) = 1;

    % Run regression with partial model
    partialR = designMat(:, ~removeReg);
    [~, dimBetaPartial] = ridgeMML(dataMatZ, partialR, true); %get ridge penalties and beta weights.

    % Predict neural data with partial model
    yModelPartial = (partialR - mean(partialR, 1)) * dimBetaPartial;

    % Get explained variance of partial model
    SSRes = sum((dataMatZ - yModelPartial) .^2, 1);
    SSTot = sum((dataMatZ - mean(dataMatZ)) .^2, 1);
    rSquaredPartial(iReg, :) = 1 - SSRes ./ SSTot;

end



if strcmp(opts.withinOrAcrossBhv, 'within')
    % If asking for within-behavior relative contribution for each neuron:
    % For each behavior, get explained variance of each regressor relative to all other time
    % regressors in that behavior
    relativeExplained = zeros(size(designMat, 2), size(dataMatZ, 2));
    for iBhv = 1 : length(behaviors)
        iReg = find(strcmp(['0 ',behaviors{iBhv}], regLabels)); % find regressor at start of behavior
        iRange = iReg - opts.mPreTime/opts.frameSize : iReg + opts.mPostTime/opts.frameSize - 1;

        % Loop through each regressor in that behavior to calculate its
        % relative contribution relative to that behavior
        for jRel = iRange
            jNum = 1 - rSquaredPartial(jRel, :) ./ rSquaredFull;
            jNum(jNum < 0) = 0;
            % jDen = (rSquaredFull - sum(rSquaredPartial(iRange, :), 1)) ./ rSquaredFull;
            jDen = sum(1 - rSquaredPartial(iRange, :) ./ rSquaredFull, 1);
            relativeExplained(jRel, :) = jNum ./ jDen;

        end %jRel

        % Make sure each relative contribution sums to 1 for each behavior
        % plot(sum(relativeExplained(iRange, :), 1))

    end % iBhv
    relativeExplained(relativeExplained < 0) = 0;
end


% If asking for across-behavior relative contribution for each neuron:
% For each behavior, get explained variance of all regressors within that
% behavior relative to total explained variance

if strcmp(opts.withinOrAcrossBhv, 'across')
    num = zeros(length(behaviors), size(dataMatZ, 2));
    for iBhv = 1 : length(behaviors)
        iReg = contains(regLabels, behaviors{iBhv});
        iNum = (rSquaredFull - sum(rSquaredPartial(iReg,:), 1)) ./ rSquaredFull;
        iNum(iNum < 0) = 0;
        num(iBhv,:) = iNum;
        % iDen = (rSquaredFull - sum(rSquaredPartial, 1)) ./ rSquaredFull;
        % relativeExplained(iBhv,:) = iNum ./ iDen;
    end
    relativeExplained = num ./ sum(num, 1);
end

end % function