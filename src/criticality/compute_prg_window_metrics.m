function prgOut = compute_prg_window_metrics(dataMat, config)
% COMPUTE_PRG_WINDOW_METRICS PRG kurtosis and D_JS for one binned window
%
% Variables:
%   dataMat - [nTimeBins x nNeurons] binned spike counts
%   config  - PRG config with fields prgMethod, cutoffDivisors, finalCutoffDivisor
%
% Goal:
%   Dispatch to momentum-space PCA PRG or real-space ICG PRG (Morales 2023).
%
% Returns:
%   prgOut - Output struct from compute_prg_momentum_kurtosis or compute_prg_icg_kurtosis

    prgMethod = 'pca';
    if isfield(config, 'prgMethod') && ~isempty(config.prgMethod)
        prgMethod = lower(strtrim(config.prgMethod));
    end

    metricArgs = {};
    if isfield(config, 'cutoffDivisors') && ~isempty(config.cutoffDivisors)
        metricArgs = [metricArgs, {'cutoffDivisors', config.cutoffDivisors}]; %#ok<AGROW>
    end
    if isfield(config, 'finalCutoffDivisor') && ~isempty(config.finalCutoffDivisor)
        metricArgs = [metricArgs, {'finalCutoffDivisor', config.finalCutoffDivisor}]; %#ok<AGROW>
    end

    switch prgMethod
        case 'pca'
            prgOut = compute_prg_momentum_kurtosis(dataMat, metricArgs{:});
        case 'icg'
            prgOut = compute_prg_icg_kurtosis(dataMat, metricArgs{:});
        otherwise
            error('config.prgMethod must be ''pca'' or ''icg'', got "%s".', prgMethod);
    end

    prgOut.prgMethod = prgMethod;
end
