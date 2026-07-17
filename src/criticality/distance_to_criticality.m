function dcc = distance_to_criticality(tauFitted, alphaFitted, gammaFitted)
    % DISTANCE_TO_CRITICALITY - |γ_pred - γ_meas| (Ma et al. 2019 dcc)
    %
    % INPUTS:
    %   tauFitted    - avalanche size exponent τ
    %   alphaFitted  - avalanche duration exponent α
    %   gammaFitted  - measured crackling exponent 1/σνz (paramSD from ⟨S⟩~T^γ)
    % OUTPUT:
    %   dcc - | (α-1)/(τ-1) - gammaFitted |

    % Check for valid input to avoid division by zero
    if tauFitted <= 1
        warning('tau_fitted must be > 1 to compute gamma_predicted - returning nan');
        dcc = nan;
        return
    end

    % Compute predicted gamma from exponent relation
    gammaPredicted = (alphaFitted - 1) ./ (tauFitted - 1);

    % Compute distance to criticality
    dcc = abs(gammaPredicted - gammaFitted);
end
