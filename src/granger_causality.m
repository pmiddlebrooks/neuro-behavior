%% Choose some lfp data to test (get lfp data from lfp_state_script.m
lfpIdx = [1:4];
samples = floor(size(lfpPerArea, 1)/60);
samples = floor(size(lfpPerArea, 1));

%%
% Script to detrend, normalize, and check stationarity of LFP signals

% Detrend the signals (remove linear trend)
detrendedLfp = detrend(lfpPerArea(1:samples, lfpIdx), 'linear');

% Normalize the signals (z-score normalization)
normalizedLfp = zscore(detrendedLfp);

% Stationarity check using Augmented Dickey-Fuller (ADF) test
% Ensure you have the Econometrics Toolbox for the adftest function
[numSamples, numAreas] = size(normalizedLfp);
stationarityResults = zeros(1, numAreas); % Store p-values

for area = 1:numAreas
    [h, pValue] = adftest(normalizedLfp(:, area)); % ADF test
    stationarityResults(area) = pValue; % Save p-value

    % Display result
    if h == 1
        fprintf('Signal %d is stationary (p = %.4f).\n', area, pValue);
    else
        fprintf('Signal %d is not stationary (p = %.4f).', area, pValue);
    end
end

% Plot the signals for visualization
% figure(91); clf;
% for area = 1:numAreas
%     subplot(numAreas, 1, area);
%     plot(normalizedLfp(:, area));
%     title(sprintf('Normalized LFP Signal (Area %d)', area));
%     xlabel('Time');
%     ylabel('Amplitude');
% end
% sgtitle('Detrended and Normalized LFP Signals');








%% My granger function
maxLag = .1 * opts.fsLfp;
[grangerCausalityResults, bestLag] = compute_granger_causality(normalizedLfp, opts.fsLfp);







%% MVGC1 toolbox (see mvgc_demo_statespace.m for the template)
cd E:\Projects\toolboxes\MVGC1
startup

%%
% Get data, then run mvgc from mvgc_demo_script
% X = reshape(normalizedLfp', size(normalizedLfp, 2), samples/60, 60);
X = reshape(lfpPerArea(1:samples, lfpIdx)', size(normalizedLfp, 2), samples/60, 60);

%% MVGC1 toolbox: peri-behavior transitions

% Find all time bins preceding all behavior transitions:
                preInd = find(diff(bhvIDMat) ~= 0); % 1 frame prior to all behavior transitions
                        id = bhvIDMat(preInd + 1);  % behavior ID being transitioned into
preIndLfp = preInd * opts.frameSize * opts.fsLfp;
fullTime = -.2 : 1/opts.fsLfp : .2; % seconds around onset
fullWindow = round(fullTime(1:end-1) * opts.fsLfp); % frames around onset w.r.t. zWindow (remove last frame)
windowCenter = find(fullTime == 0);

% Build a LFP matrix for the areas of interest, for a given behavior
% transition
idTest = 15;

idTestIdx = preInd(id == idTest);
idTestIdxLfp = floor(idTestIdx * opts.frameSize * opts.fsLfp);

lfpTest = zeros(size(lfpPerArea(:,lfpIdx), 2), length(fullWindow), length(idTestIdxLfp));
for i = 1: length(idTestIdxLfp)-1
% lfpTest(:,:,i) = lfpPerArea(idTestIdxLfp(i) + fullWindow,lfpIdx)';
lfpTest(:,:,i) = normalizedLfp(idTestIdxLfp(i) + fullWindow,lfpIdx)';
end

X = lfpTest;



%% MVGC1 toolbox: peri-reach from Mark's data: Get his data from lfp_reach.m
fullTime = -.2 : 1/fs : .2; % seconds around onset
fullWindow = round(fullTime(1:end-1) * fs); % frames around onset w.r.t. zWindow (remove last frame)

nTrial = size(rData.R, 1);
lfpTest = zeros(size(lfpReach, 2), length(fullWindow), nTrial);
for i = 1 : nTrial
    lfpTest(:,:,i) = lfpReach(rData.R(i,1) + fullWindow, lfpIdx)';
end
X = lfpTest;














function [grangerCausalityResults, bestLag] = compute_granger_causality(lfpSignals, samplingFreq)
% COMPUTE_GRANGER_CAUSALITY Computes Granger causality between LFP signals.
% Automatically determines the best maxLag using AIC and evaluates significance using F-statistics.
% Inputs:
%   lfpSignals   - Matrix of LFP signals (time x areas).
%   samplingFreq - Sampling frequency of the LFP signals (in Hz).
%
% Outputs:
%   grangerCausalityResults - Struct containing:
%       .values - Matrix of Granger causality values (areas x areas).
%       .significance - Matrix of p-values indicating significance of Granger causality.

    % Check input dimensions
    [numSamples, numAreas] = size(lfpSignals);
    if numAreas < 2
        error('At least two LFP signals are required for Granger causality analysis.');
    end

    % Determine the best maxLag using AIC
    % maxPossibleLag = round(numSamples / 10); % Set a reasonable upper limit for lag
    maxPossibleLag = 625; % Set a reasonable upper limit for lag
    bestLag = determine_best_lag(lfpSignals, maxPossibleLag);
    disp('bestLag:');
    disp(bestLag);

    % Initialize results matrices
    grangerValues = zeros(numAreas, numAreas);
    pValues = zeros(numAreas, numAreas);

    % Loop over all pairs of areas
    for i = 1:numAreas
        for j = 1:numAreas
            if i ~= j
                % Extract signals
                x = lfpSignals(:, i); % Target signal
                y = lfpSignals(:, j); % Source signal

                % Combine signals for the full model
                combinedSignals = [x, y];

                % Fit full model (multivariate AR)
                [coeffsFull, sigmaFull, residualsFull] = fit_ar_model(combinedSignals, bestLag);

                % Fit reduced model (AR on target signal only)
                [~, sigmaReduced, residualsReduced] = fit_ar_model(x, bestLag);

                % Compute Granger causality
                grangerValue = log(det(sigmaReduced) / det(sigmaFull));
                grangerValues(i, j) = grangerValue;

                % Perform F-test for significance
                numParamsFull = bestLag * size(combinedSignals, 2)^2;
                numParamsReduced = bestLag * size(x, 2)^2;
                fStat = ((det(sigmaReduced) - det(sigmaFull)) / (numParamsFull - numParamsReduced)) / (det(sigmaFull) / (numSamples - numParamsFull));
                pValues(i, j) = 1 - fcdf(fStat, numParamsFull - numParamsReduced, numSamples - numParamsFull);
            end
        end
    end

    % Package results
    grangerCausalityResults.values = grangerValues;
    grangerCausalityResults.significance = pValues;

    % Optional: Display results
    disp('Granger Causality Values:');
    disp(grangerValues);
    disp('Significance Matrix (p-values):');
    disp(pValues);
end

function bestLag = determine_best_lag(data, maxPossibleLag)
% DETERMINE_BEST_LAG Determines the optimal lag for AR modeling using AIC.
% Inputs:
%   data           - Matrix of signals (time x variables).
%   maxPossibleLag - Maximum lag to evaluate.
%
% Outputs:
%   bestLag - Optimal lag based on BIC/AIC.

    [numSamples, ~] = size(data);
    aicValues = zeros(maxPossibleLag, 1);
    bicValues = zeros(maxPossibleLag, 1);

    for lag = 1:maxPossibleLag
        [~, sigma] = fit_ar_model(data, lag);
        logLikelihood = -0.5 * numSamples * log(det(sigma));
        numParams = lag * size(data, 2)^2; % Number of parameters in the AR model
        % aicValues(lag) = -2 * logLikelihood + 2 * numParams; % AIC formula
        bicValues(lag) = -2 * logLikelihood + numParams * log(numSamples); % BIC formula
    end

    % Find the lag with the minimum AIC
    % [~, bestLagA] = min(aicValues);
    [~, bestLag] = min(bicValues);
end

function [coeffs, sigma, residuals] = fit_ar_model(data, maxLag)
% FIT_AR_MODEL Fits an autoregressive model to the input data.
% Inputs:
%   data   - Matrix of signals (time x variables).
%   maxLag - Maximum lag to consider.
%
% Outputs:
%   coeffs    - Coefficients of the fitted AR model.
%   sigma     - Covariance matrix of the residuals.
%   residuals - Residuals of the AR model.

    [numSamples, numVars] = size(data);
    X = []; % Design matrix
    Y = data(maxLag+1:end, :); % Target values

    % Build the design matrix using lags
    for lag = 1:maxLag
        X = [X, data(maxLag-lag+1:end-lag, :)];
    end

    % Solve for AR coefficients using least squares
    coeffs = (X' * X) \ (X' * Y);

    % Compute residuals
    residuals = Y - X * coeffs;

    % Compute covariance of residuals
    sigma = cov(residuals);
end
