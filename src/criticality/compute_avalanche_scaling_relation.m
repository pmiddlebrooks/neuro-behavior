function scalingRelation = compute_avalanche_scaling_relation(tau, alpha)
% COMPUTE_AVALANCHE_SCALING_RELATION - Crackling prediction (alpha-1)/(tau-1)

scalingRelation = nan;
if isfinite(tau) && isfinite(alpha) && tau > 1
  scalingRelation = (alpha - 1) / (tau - 1);
end
end
