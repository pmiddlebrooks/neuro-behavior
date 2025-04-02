function alpha = compute_DFA(dataMat)
    % Compute Long-Range Temporal Correlations using Detrended Fluctuation Analysis (DFA)
    %
    % INPUT:
    %   dataMat - A matrix of spike counts (neurons x time bins)
    % OUTPUT:
    %   alpha - DFA exponent indicating long-range temporal correlations

%     Interpretation of DFA Exponent 
% α ≈ 0.5  → No long-range correlations (random process).
% 
% 0.5<α<1.0 → Long-range temporal correlations (e.g., power-law behavior).
% 
% α≈1.5 → Brownian noise (strong correlations).
% 
% α>1.5 → Highly persistent signals.
% 
    % Summing across neurons to create a single time series
    timeSeries = sum(dataMat, 1);  % Sum spike activity across neurons

    % Normalize time series (zero mean)
    timeSeries = timeSeries - mean(timeSeries);
    
    % Compute integrated signal (cumulative sum)
    Y = cumsum(timeSeries);
    
    % Define window sizes (logarithmically spaced)
    minWin = 10; % Minimum window size
    maxWin = length(Y) / 4; % Maximum window size (ensuring enough segments)
    scales = round(logspace(log10(minWin), log10(maxWin), 20)); % 20 scales

    % Initialize fluctuation values
    F = zeros(length(scales), 1);

    % Loop over each window size
    for i = 1:length(scales)
        s = scales(i); % Current window size
        N = floor(length(Y) / s); % Number of segments

        % Divide the signal into non-overlapping segments
        Y_segments = reshape(Y(1:N*s), s, N); 

        % Fit a linear trend to each segment and remove it
        for j = 1:N
            x = (1:s)';
            p = polyfit(x, Y_segments(:, j), 1); % Linear detrending
            Y_segments(:, j) = Y_segments(:, j) - polyval(p, x); % Remove trend
        end

        % Compute the root mean square fluctuation
        F(i) = sqrt(mean(Y_segments(:).^2));
    end

    % Fit a line in log-log space to obtain the DFA exponent
    logScales = log(scales);
    logF = log(F);
    p = polyfit(logScales, logF, 1);
    alpha = p(1); % DFA exponent (slope)

    % Plot DFA results
    figure;
    loglog(scales, F, 'o-', 'LineWidth', 1.5);
    hold on;
    loglog(scales, exp(polyval(p, logScales)), '--', 'LineWidth', 2);
    xlabel('Window Size (s)');
    ylabel('Fluctuation F(s)');
    title(['DFA Analysis: \alpha = ', num2str(alpha)]);
    legend('Data', 'Fit', 'Location', 'best');
    grid on;

    % Display result
    disp(['DFA Exponent (Alpha): ', num2str(alpha)]);
end
