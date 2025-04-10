function br = weighted_branching_ratio(activityVec)
% Computes the weighted branching ratio from a 1D activity vector
% INPUT:
%   activityVec - vector of activity values with zeros separating avalanches
% OUTPUT:
%   m_weighted  - weighted global branching ratio

    numerator = 0;
    denominator = 0;

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
                numerator = numerator + sum(A(2:end));
                denominator = denominator + sum(A(1:end-1));
            end
            inAvalanche = false;
        end
    end

    % If signal ends during an avalanche
    if inAvalanche
        A = activityVec(startIdx:end);
        if length(A) > 1
            numerator = numerator + sum(A(2:end));
            denominator = denominator + sum(A(1:end-1));
        end
    end

    % Compute global weighted branching ratio
    if denominator == 0
        warning('No valid avalanches of duration >= 2 found.');
        br = NaN;
    else
        br = numerator / denominator;
    end
end
