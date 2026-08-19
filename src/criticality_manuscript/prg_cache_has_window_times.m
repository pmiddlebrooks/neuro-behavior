function tf = prg_cache_has_window_times(results)
% PRG_CACHE_HAS_WINDOW_TIMES - True if cached PRG results can be split by engagement

tf = false;
if ~isstruct(results) || ~isfield(results, 'windowStartS') || isempty(results.windowStartS)
  return;
end
if ~iscell(results.windowStartS)
  tf = ~isempty(results.windowStartS);
  return;
end
tf = any(~cellfun(@isempty, results.windowStartS));
end
