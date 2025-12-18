function alpha = compute_DFA(signal, plotFlag)
    % compute_DFA - Compute Long-Range Temporal Correlations using Detrended Fluctuation Analysis (DFA)
    %
    % DFA measures the self-similarity of a signal by calculating the relationship 
    % between the average fluctuation of the signal and the window size.
    %
    % INPUT:
    %   signal - A 1D time series vector (e.g., LFP, summed spike counts, etc.)
    %            If a matrix is provided, it will be summed across the second dimension.
    %   plotFlag - Boolean to enable/disable log-log plotting
    %
    % OUTPUT:
    %   alpha - DFA exponent (slope of the log-log plot)
    %
    % Interpretation of DFA Exponent:
    %   α ≈ 0.5   → White noise (random process, no correlations).
    %   0.5 < α < 1.0 → Long-range temporal correlations (persistent, power-law).
    %   α ≈ 1.0   → 1/f noise (pink noise, strong long-range correlations).
    %   α ≈ 1.5   → Brownian noise (random walk).
    %   α > 1.5   → Highly persistent, non-stationary signals.

    % Ensure signal is a column vector (time bins as rows)
    if size(signal, 1) < size(signal, 2)
        signal = signal';
    end

    % If signal is still a matrix, sum across columns to get a single time series
    if size(signal, 2) > 1
        timeSeries = sum(signal, 2);
    else
        timeSeries = signal;
    end

    % Normalize time series (zero mean) to remove DC offset
    timeSeries = timeSeries - mean(timeSeries);
    
    % Compute integrated signal (cumulative sum) - standard DFA step
    Y = cumsum(timeSeries);
    
    % Define window sizes (scales) logarithmically spaced
    % Using a range that covers approximately 10 bins up to 1/4 of total length
    minWin = 10; 
    maxWin = floor(length(Y) / 4); 
    
    if maxWin <= minWin
        warning('Signal too short for reliable DFA analysis. Returning NaN.');
        alpha = NaN;
        return;
    end
    
    scales = round(logspace(log10(minWin), log10(maxWin), 20)); 
    scales = unique(scales); % Ensure unique scales if signal is short

    % Initialize fluctuation values
    F = zeros(length(scales), 1);

    % Loop over each window size (scale)
    for i = 1:length(scales)
        s = scales(i); 
        numSegments = floor(length(Y) / s); 

        % Reshape Y into non-overlapping segments of length s
        % Each column is a segment
        Y_segments = reshape(Y(1 : numSegments * s), s, numSegments); 

        % Linear detrending within each segment
        x = (1:s)';
        for j = 1:numSegments
            p = polyfit(x, Y_segments(:, j), 1); 
            Y_segments(:, j) = Y_segments(:, j) - polyval(p, x); 
        end

        % Compute root-mean-square (RMS) fluctuation for this scale
        F(i) = sqrt(mean(Y_segments(:).^2));
    end

    % Fit a line in log-log space to find the DFA exponent (alpha)
    validIdx = F > 0; % Ensure F is positive for log
    if sum(validIdx) < 2
        alpha = NaN;
        return;
    end
    
    logScales = log10(scales(validIdx));
    logF = log10(F(validIdx));
    p = polyfit(logScales, logF, 1);
    alpha = p(1); 

    if plotFlag
        figure;
        loglog(scales, F, 'o', 'MarkerSize', 8, 'LineWidth', 1.5);
        hold on;
        loglog(scales, 10.^polyval(p, logScales), 'r--', 'LineWidth', 2);
        xlabel('Scale (bins)');
        ylabel('Fluctuation F(s)');
        title(sprintf('DFA Analysis: \\alpha = %.3f', alpha));
        legend('Empirical F(s)', sprintf('Power-law fit (\\alpha=%.3f)', alpha), 'Location', 'northwest');
        grid on;
    end
    
    % fprintf('DFA Exponent (Alpha): %.3f\n', alpha);
end
