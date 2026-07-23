function [sigmaNuZInv, sigmaNuZInvStd, logCoeff] = size_given_duration(sizes, durations, varargin)
% SIZE_GIVEN_DURATION - size given duration weighted least squares
% Computes the average size given duration for avalanche data and performs
% the weighted least squares fit to determine the scaling parameter and
% its standard deviation.
%
% Syntax: [sigmaNuZInv, sigmaNuZInvStd, logCoeff] = 
%           size_given_duration(sizes, durations, varargin)
%
% Inputs:
%   sizes (vector double) - avalanche sizes
%   durations (vector double) - avalanche durations
%
% Variable Inputs:
%   (..., 'durmin', durMin) - sets duration minimum at durMin (scalar double)
%   (..., 'durmax', durMax) - sets duration maximum at durMax (scalar double)
%
% Outputs:
%   sigmaNuZInv (scalar double) - Crackling / size-duration exponent 1/(σ ν z)
%     from ⟨S⟩(T) ~ T^{1/σνz} (stored as paramSD elsewhere in this codebase)
%   sigmaNuZInvStd (scalar double) - standard deviation on 1/(σ ν z)
%     estimate from WLS fit
%   logCoeff (scalar double) - logarithm of scaling coefficient for the fit
%
% Example:
%   [sigmaNuZInv, sigmaNuZInvStd, logCoeff] = size_given_duration(sizes, durations);
%   [sigmaNuZInv, sigmaNuZInvStd, logCoeff] = size_given_duration(sizes, durations, 'durmin', 2, 'durmax', 10);
%
% See also: AVPROPS, AVPROPVALS, SIZEGIVDURWLS

%% Parse command line for variable inputs
durMin = min(durations);
durMax = max(durations);

iVarArg = 1;
while iVarArg <= length(varargin)
    argOkay = true;
    if ischar(varargin{iVarArg})
        switch varargin{iVarArg}
            case 'durmin'
                if iVarArg + 1 <= length(varargin)
                    durMin = max([durMin, varargin{iVarArg+1}]);
                    iVarArg = iVarArg + 1;  % Skip next argument as it's the value
                else
                    argOkay = false;
                end
            case 'durmax'
                if iVarArg + 1 <= length(varargin)
                    durMax = min([durMax, varargin{iVarArg+1}]);
                    iVarArg = iVarArg + 1;  % Skip next argument as it's the value
                else
                    argOkay = false;
                end
            otherwise
                argOkay = false;
        end
    else
        argOkay = false;
    end
    if ~argOkay
        warning('(SIZE_GIVEN_DURATION) Ignoring invalid argument #%d', iVarArg);
    end
    iVarArg = iVarArg + 1;
end

%% Process data

% find unique durations
unqDurations = unique(durations);
nUnqDurs = length(unqDurations);

% make histogram of duration values
durHist = histc(durations, unqDurations);

% calculate average size given durations
sizeGivDur = zeros(1, nUnqDurs);

for iUnqDur = 1:nUnqDurs
    matchingIndices = durations == unqDurations(iUnqDur);
    if any(matchingIndices)
        sizeGivDur(iUnqDur) = mean(sizes(matchingIndices));
    else
        sizeGivDur(iUnqDur) = NaN;
    end
end

% remove NaN values and corresponding entries from other arrays
validIndices = isfinite(sizeGivDur);
sizeGivDur = sizeGivDur(validIndices);
unqDurations = unqDurations(validIndices);
durHist = durHist(validIndices);

%% Prepare data

% Logarithmically transform unique durations and average sizes
logUnqDurs = log10(unqDurations)'; 
logSizeGivDur = log10(sizeGivDur)';

% Restrict WLS to duration bins inside [durMin, durMax] (power-law fit range).
% Do not fall back to the full duration span if the requested range is empty.
durMinInd = find(unqDurations >= durMin, 1, 'first');
durMaxInd = find(unqDurations <= durMax, 1, 'last');

if durMin > durMax || isempty(durMinInd) || isempty(durMaxInd) || durMinInd > durMaxInd
  sigmaNuZInv = nan;
  sigmaNuZInvStd = nan;
  logCoeff = nan;
  return;
end

nFit = durMaxInd - durMinInd + 1;
if nFit < 2
  sigmaNuZInv = nan;
  sigmaNuZInvStd = nan;
  logCoeff = nan;
  return;
end
logDurSlice = logUnqDurs(durMinInd:durMaxInd);
logSizeSlice = logSizeGivDur(durMinInd:durMaxInd);
weightSlice = durHist(durMinInd:durMaxInd);

% create the design matrix for least squares
X = [logDurSlice(:), ones(nFit, 1)];

%% Perform the weighted least squares fit
[B, S] = lscov(X, logSizeSlice(:), weightSlice(:));

% extract parameter values
sigmaNuZInv = B(1);
logCoeff = B(2);
sigmaNuZInvStd = S(1);

end

