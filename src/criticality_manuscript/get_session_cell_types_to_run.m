function cellTypes = get_session_cell_types_to_run(splitExcitatoryInhibitory)
% GET_SESSION_CELL_TYPES_TO_RUN - Populations to analyze when E/I split is enabled
%
% Variables:
%   splitExcitatoryInhibitory - If true, run combined (all), excitatory, and inhibitory
%
% Returns:
%   cellTypes - {'all','excitatory','inhibitory'} or {''} when split is off

if splitExcitatoryInhibitory
  cellTypes = {'all', 'excitatory', 'inhibitory'};
else
  cellTypes = {''};
end
end
