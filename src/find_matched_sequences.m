function [matchFrames, matchCorr] = find_matched_sequences(data, opts)
% FIND_REPEATING_KINEMATICS - Identifies repeating kinematic sequences by 
%                              correlating a template window with the rest of the data.
%
% INPUTS:
%   data   - Matrix of data (time bins x features)
%   opts      - Structure containing the following fields:
%       .frameSize      - The size of time bins in data
%       .window         - The number of bins over which to sample
%       .matchFrame     - The time bin marking the first frame of the template window
%       .thresholdCorr  - Minimum correlation required to count as a match
%
% OUTPUTS:
%   matchFrames - Indices of the first frame of all matching windows
%   matchCorr   - Correlation values of each of the matches

% Ensure matchFrame is within bounds
if opts.matchFrame + opts.window - 1 > size(data, 1)
    error('Template window extends beyond the dataset.');
end

% Extract the template window
templateWindow = data(opts.matchFrame : opts.matchFrame + opts.window - 1, :);

% Get the number of time bins
numBins = size(data, 1);

% Initialize output variables
matchFrames = [];
matchCorr = [];

% clf

% Slide over the data and compute correlation
for startIdx = 1 : (numBins - opts.window + 1)

    % skip if the data is from the template
    if startIdx >= opts.matchFrame && startIdx < opts.matchFrame + opts.window
        continue
    end
    % Extract the current window
    currWindow = data(startIdx : startIdx + opts.window - 1, :);
    
    % Compute correlation between the template and the current window
    corrValue = corr(templateWindow(:), currWindow(:));
%     figure(22); plot(templateWindow(:,1)); hold on; 
%     plot(currWindow(:,1))
% plot(templateWindow(:,7)); 
% plot(currWindow(:,7))
    % If correlation meets or exceeds threshold, store the match
    if corrValue >= opts.thresholdCorr
        matchFrames = [matchFrames; startIdx];
        matchCorr = [matchCorr; corrValue];
    end
end

end
