function sufficient = avalanches_bin_sufficiency(dataMat)
    % Estimate whether the dataset has enough bins for criticality analysis
    % INPUT:
    %   dataMat - Binary spike matrix (time bins X neurons, or time bins X
    %   1)
    %
    % OUTPUT:
    %   Prints estimated number of avalanches, unique sizes, and fit quality.

    sufficient = 0; % Initialize to false
    numCriterion = 10000; % How many number of avalanches are required?
    sizesCriterion = 50; % How many unique sizes of avalanches are required?

    % Step 1: Identify Avalanches
    if size(dataMat, 2) > 1
    timeSeries = sum(dataMat, 1); % Sum spike activity per bin
    else
        timeSeries = dataMat;
    end
    threshold = 1; % Define threshold for avalanche detection
    avalancheLengths = []; % Store avalanche sizes

    inAvalanche = false;
    currentAvalancheSize = 0;

    for t = 1:length(timeSeries)
        if timeSeries(t) >= threshold
            currentAvalancheSize = currentAvalancheSize + 1;
            inAvalanche = true;
        elseif inAvalanche
            avalancheLengths = [avalancheLengths, currentAvalancheSize];
            currentAvalancheSize = 0;
            inAvalanche = false;
        end
    end

    % Step 2: Check Minimum Avalanche Count and Unique Sizes
    numAvalanches = length(avalancheLengths);
    uniqueSizes = length(unique(avalancheLengths));

    fprintf('Total Avalanches Detected: %d\n', numAvalanches);
    fprintf('Unique Avalanche Sizes: %d\n', uniqueSizes);

    if numAvalanches < 10000
        warning('Too few avalanches detected (<10,000). Consider more bins.');
    end
    if uniqueSizes < 50
        warning('Too few unique avalanche sizes (<50). Power-law fitting may be unreliable.');
    end
    if numAvalanches >= numCriterion && uniqueSizes >= sizesCriterion
        sufficient = 1;
    end
    % Step 3: Fit Power Law (using MLE)
    if numAvalanches > 1000
% Fit power-law distribution using Clauset's plfit (ensure you have plfit.m)
[alpha, xmin, L] = plfit(avalancheLengths);
        % xMin = min(avalancheLengths);
        % params = mle(avalancheLengths, 'distribution', 'pareto', 'start', xMin);
        % alpha = params(1);

        fprintf('Estimated Power-Law Exponent: %.3f\n', alpha);
        if alpha < 1.2 || alpha > 3.0
            warning('Exponent out of critical range (1.2–3.0). Consider more bins.');
        end
    else
        warning('Not enough data points for reliable power-law fitting.');
    end
end
