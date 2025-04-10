function br = stepwise_branching_ratio(activityVec)
% Computes stepwise branching ratio from an activity vector
% INPUT:
%   activityVec - 1D vector with zeros separating avalanches
% OUTPUT:
%   m_stepwise - average of all within-avalanche A(t+1)/A(t) ratios

    stepwise_ratios = [];

    inAvalanche = false;
    startIdx = 0;

    for t = 1:length(activityVec)
        if activityVec(t) > 0
            if ~inAvalanche
                startIdx = t;
                inAvalanche = true;
            end
        elseif inAvalanche
            % End of avalanche
            endIdx = t - 1;
            A = activityVec(startIdx:endIdx);
            if length(A) > 1
                mt = A(2:end) ./ A(1:end-1);  % Stepwise ratios
                stepwise_ratios = [stepwise_ratios; mt(:)];
            end
            inAvalanche = false;
        end
    end

    % Handle avalanche at end of vector
    if inAvalanche
        A = activityVec(startIdx:end);
        if length(A) > 1
            mt = A(2:end) ./ A(1:end-1);
            stepwise_ratios = [stepwise_ratios; mt(:)];
        end
    end

    % Compute mean stepwise branching ratio
    if isempty(stepwise_ratios)
        warning('No valid avalanches of duration >= 2 found.');
        br = NaN;
    else
        br = mean(stepwise_ratios);
    end
end
