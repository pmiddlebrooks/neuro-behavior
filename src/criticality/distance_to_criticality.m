function dcc = distance_to_criticality(tauFitted, alphaFitted, gammaFitted)
    % computeDCC calculates the Distance to Criticality Coefficient
    % from Ma et al 2019
    %
    % INPUTS:
    %   alpha_fitted  - avalanche duration exponent
    %   tau_fitted    - avalanche size exponent
    %   gamma_fitted  - exponent from size-duration scaling
    % OUTPUT:
    %   dcc - distance to criticality coefficient

    % Check for valid input to avoid division by zero
    if tauFitted <= 1
        error('tau_fitted must be > 1 to compute gamma_predicted');
    end

    % Compute predicted gamma from exponent relation
    gammaPredicted = (alphaFitted - 1) ./ (tauFitted - 1);

    % Compute distance to criticality
    dcc = abs(gammaPredicted - gammaFitted);
end
