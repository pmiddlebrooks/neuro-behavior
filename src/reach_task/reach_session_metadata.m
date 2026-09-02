function metadata = reach_session_metadata(subjectName, sessionName)
% REACH_SESSION_METADATA - Per-session loading-option overrides
%
% Variables:
%   subjectName - Subject id (e.g. 'AB6'); inferred from sessionName if empty
%   sessionName - Session identifier (e.g. 'AB6_27-Mar-2025 14_04_12_NeuroBeh')
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
sessionName = normalize_metadata_string(sessionName);
subjectName = normalize_metadata_string(subjectName);
if isempty(subjectName)
  subjectName = reach_subject_from_session(sessionName);
end

switch subjectName
  case 'AB2'
    switch sessionName
      case 'AB2_28-Apr-2023 17_50_02_NeuroBeh'
        % no overrides

      case 'AB2_01-May-2023 15_34_59_NeuroBeh'
        % no overrides

      case 'AB2_11-May-2023 17_31_00_NeuroBeh'
        % no overrides

      case 'AB2_30-May-2023 12_49_52_NeuroBeh'
        % no overrides

      otherwise
        % no overrides
    end

  case 'AB6'
    switch sessionName
      case 'AB6_27-Mar-2025 14_04_12_NeuroBeh'
        % no overrides

      case 'AB6_29-Mar-2025 15_21_05_NeuroBeh'
        % no overrides

      case 'AB6_02-Apr-2025 14_18_54_NeuroBeh'
        % no overrides

      case 'AB6_03-Apr-2025 13_34_09_NeuroBeh'
        % no overrides

      otherwise
        % no overrides
    end

  case 'AB19'
    switch sessionName
      case 'AB19_09-Apr-2026 14_28_19_NeuroBeh'
        % no overrides

      case 'AB19_31-Mar-2026 15_46_45_NeuroBeh'
        % no overrides

      otherwise
        % no overrides
    end

  case 'AB21'
    switch sessionName
      case 'AB21_06-Apr-2026 18_07_42_NeuroBeh'
        % no overrides

      otherwise
        % no overrides
    end

  case 'Y4'
    switch sessionName
      case 'Y4_06-Oct-2023 14_14_53_NeuroBeh'
        % no overrides

      otherwise
        % no overrides
    end

  case 'Y12'
    switch sessionName
      case 'Y12_20-Jan-2026 16_16_42_NeuroBeh'
        % no overrides

      otherwise
        % no overrides
    end

  case 'Y15'
    switch sessionName
      case 'Y15_26-Aug-2025 12_24_22_NeuroBeh'
        % no overrides

      case 'Y15_27-Aug-2025 14_02_21_NeuroBeh'
        % no overrides

      case 'Y15_28-Aug-2025 19_47_07_NeuroBeh'
        % no overrides

      otherwise
        % no overrides
    end

  case 'Y16'
    switch sessionName
      case 'Y16_23-Dec-2025 16_07_49_NeuroBeh'
        % no overrides

      case 'Y16_31-Dec-2025 13_50_49_NeuroBeh'
        % no overrides

      otherwise
        % no overrides
    end

  case 'Y17'
    switch sessionName
      case 'Y17_20-Aug-2025 17_34_48_NeuroBeh'
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

function subjectName = reach_subject_from_session(sessionName)
% REACH_SUBJECT_FROM_SESSION - Subject token before the first underscore

usIdx = find(sessionName == '_', 1);
if isempty(usIdx)
  subjectName = sessionName;
else
  subjectName = sessionName(1:usIdx-1);
end
end
