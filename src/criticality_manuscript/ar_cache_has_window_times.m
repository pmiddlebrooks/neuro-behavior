function tf = ar_cache_has_window_times(results)
% AR_CACHE_HAS_WINDOW_TIMES - True if cached AR results can be split by engagement

tf = false;
if ~isstruct(results) || ~isfield(results, 'startS') || isempty(results.startS)
  return;
end
if ~iscell(results.startS)
  tf = ~isempty(results.startS);
  return;
end
tf = any(~cellfun(@isempty, results.startS));
end
