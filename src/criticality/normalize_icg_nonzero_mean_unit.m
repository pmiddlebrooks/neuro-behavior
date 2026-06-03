function normAct = normalize_icg_nonzero_mean_unit(activityVec)
% NORMALIZE_ICG_NONZERO_MEAN_UNIT Morales et al. 2023 RG normalization (Eq. 6)
%
% Variables:
%   activityVec - 1 x T or T x 1 activity time series for one coarse-grained unit
%
% Goal:
%   Scale activity so the mean of nonzero entries equals 1, as in real-space
%   phenomenological RG (Morales et al. 2023 PNAS Methods; Meshulam et al. 2019).
%
% Returns:
%   normAct - Same-size vector; unchanged when there are no nonzero samples

    normAct = activityVec(:).';
    nonzeroMask = normAct ~= 0;
    if ~any(nonzeroMask)
        return;
    end

    meanNonzero = mean(normAct(nonzeroMask));
    if meanNonzero > 0 && isfinite(meanNonzero)
        normAct = normAct ./ meanNonzero;
    end
end
