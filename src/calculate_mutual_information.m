% Function to calculate mutual information between two vectors
function mi = calculate_mutual_information(x, y, nBins)
    % Remove NaN values
    validIdx = ~isnan(x) & ~isnan(y);
    x = x(validIdx);
    y = y(validIdx);
    
    if length(x) < 10  % Need sufficient data points
        mi = nan;
        return;
    end
    
    % Create histograms
    [~, edgesX] = histcounts(x, nBins);
    [~, edgesY] = histcounts(y, nBins);
    
    % Calculate joint and marginal distributions
    [jointHist, ~, ~] = histcounts2(x, y, edgesX, edgesY);
    
    % Normalize to get probabilities
    jointProb = jointHist / sum(jointHist(:));
    marginalX = sum(jointProb, 2);
    marginalY = sum(jointProb, 1);
    
    % Calculate mutual information
    mi = 0;
    for i = 1:size(jointProb, 1)
        for j = 1:size(jointProb, 2)
            if jointProb(i, j) > 0 && marginalX(i) > 0 && marginalY(j) > 0
                mi = mi + jointProb(i, j) * log2(jointProb(i, j) / (marginalX(i) * marginalY(j)));
            end
        end
    end
end
