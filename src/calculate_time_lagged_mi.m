% Function to calculate time-lagged mutual information
function tlmi = calculate_time_lagged_mi(x, y, nBins, lagRange)
    tlmi = nan(1, length(lagRange));
    for l = 1:length(lagRange)
        lag = lagRange(l);
        if lag < 0
            xLag = x(1:end+lag);
            yLag = y(1-lag:end);
        elseif lag > 0
            xLag = x(1+lag:end);
            yLag = y(1:end-lag);
        else
            xLag = x;
            yLag = y;
        end
        tlmi(l) = calculate_mutual_information(xLag, yLag, nBins);
    end
end
