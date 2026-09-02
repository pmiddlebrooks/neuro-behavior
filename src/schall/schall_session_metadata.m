function metadata = schall_session_metadata(subjectName, sessionName)
% SCHALL_SESSION_METADATA - Per-session loading-option overrides
%
% Variables:
%   subjectName - Subject folder (e.g. 'broca', 'joule'); inferred if sessionName
%                 is 'subject/session'
%   sessionName - Session identifier (e.g. 'bp229n02-mm' or 'broca/bp229n02-mm')
%
% Goal:
%   Return metadata that apply_session_load_metadata merges into load opts.
%   Empty / omitted fields leave the caller's options unchanged.
%
% Returns:
%   metadata - Struct of overrides. Recognized fields (all optional):
%     .collectStartMin - Floor for opts.collectStart (seconds, not minutes):
%                        collectStart = max(requested, collectStartMin)
%     .collectEndMax   - Cap for opts.collectEnd (seconds); ignored if collectEnd is []
%     .collectStart, .collectEnd - Seconds, same as neuro_behavior_options
%     .minFiringRate, .maxFiringRate, ...
%                      - Any neuro_behavior_options field; nonempty values override
%     .notes           - Comment only; not copied onto opts

metadata = struct();
subjectName = normalize_metadata_string(subjectName);
sessionName = normalize_metadata_string(sessionName);
[subjectName, sessionName] = split_schall_session_id(subjectName, sessionName);

switch subjectName
  case 'broca'
    switch sessionName
      case 'bp229n02-mm'
        % no overrides

      case 'bp200n02'
        % no overrides

      case 'bp240n02'
        % no overrides

      otherwise
        % no overrides
    end

  case 'joule'
    switch sessionName
      case 'jp121n02'
        % no overrides

      case 'jp124n04'
        % no overrides

      case 'jp125n04'
        % no overrides

      otherwise
        % no overrides
    end

  otherwise
    % no overrides
end
end

function value = normalize_metadata_string(value)
% NORMALIZE_METADATA_STRING - Char vector for switch matching

if nargin < 1 || isempty(value)
  value = '';
else
  value = char(value);
end
end

function [subjectName, sessionName] = split_schall_session_id(subjectName, sessionName)
% SPLIT_SCHALL_SESSION_ID - Accept 'subject/session' or separate arguments

slashIdx = find(sessionName == '/', 1);
if isempty(slashIdx)
  return;
end
pathSubject = sessionName(1:slashIdx-1);
pathSession = sessionName(slashIdx+1:end);
if isempty(subjectName)
  subjectName = pathSubject;
end
sessionName = pathSession;
end
