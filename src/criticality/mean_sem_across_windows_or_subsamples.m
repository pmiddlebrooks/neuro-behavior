function [meanVal, semVal] = mean_sem_across_windows_or_subsamples(windowVec, subsampleMat, useSubsampling)
% MEAN_SEM_ACROSS_WINDOWS_OR_SUBSAMPLES - Session mean/SEM for windowed metrics
%
% Variables:
%   windowVec      - Metric per window (already mean across neuron subsamples)
%   subsampleMat   - Optional [nWindows x nSubsamples] raw subsample values
%   useSubsampling - If true and one valid window, SEM across subsample columns
%
% Goal:
%   Full-session analyses (one window) with neuron subsampling report SEM
%   across subsamples. Multi-window analyses keep SEM across window means.

meanVal = nan;
semVal = nan;
if nargin < 2
  subsampleMat = [];
end
if nargin < 3 || isempty(useSubsampling)
  useSubsampling = false;
end

windowVec = windowVec(:);
validWinMask = isfinite(windowVec);
winIdx = find(validWinMask);
nWin = numel(winIdx);
if nWin < 1
  return;
end

useSubsampleSem = logical(useSubsampling) && nWin == 1 && ~isempty(subsampleMat) ...
  && size(subsampleMat, 1) >= winIdx(1);
if useSubsampleSem
  subRow = subsampleMat(winIdx(1), :);
  subRow = subRow(isfinite(subRow));
  if ~isempty(subRow)
    meanVal = mean(subRow);
    if numel(subRow) > 1
      semVal = std(subRow) / sqrt(numel(subRow));
    else
      semVal = 0;
    end
    return;
  end
end

vals = windowVec(validWinMask);
meanVal = mean(vals);
if numel(vals) > 1
  semVal = std(vals) / sqrt(numel(vals));
else
  semVal = 0;
end
end
