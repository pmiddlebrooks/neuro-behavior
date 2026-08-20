function out = semicircle_criticality_metrics_engagement(subjectName, sessionName, opts)
% SEMICIRCLE_CRITICALITY_METRICS_ENGAGEMENT - Criticality by semicircle engagement
%
% Variables:
%   subjectName - Subject folder (e.g. 'AS1')
%   sessionName - Session identifier (e.g. 'AS1_0618_WellLearned')
%   opts        - Options struct (same fields as interval_criticality_metrics_engagement,
%                 including eventBufferBefore / eventBufferAfter)
%
% Goal:
%   Same engagement analyses as the interval-timing module, but engagement
%   events are TaskMatrix timestamps: trialStartTime, choicePort poke time,
%   leave-home first, enter-home for trial start, and leave-home last.
%   Failed trials still count when those times are finite. Isolated single
%   events may be absorbed into non-engaged gaps (absorbSingleEvents).
%
% Returns:
%   With no inputs: default options struct.
%   Otherwise: engagement results struct from the shared beam-break pipeline.

if nargin == 0
  out = interval_criticality_metrics_engagement();
  out.runD2TrialRateCorrelation = false;
  out.beamBreakTask = 'semicircle';
  return;
end

if nargin < 2 || isempty(subjectName) || isempty(sessionName)
  error('semicircle_criticality_metrics_engagement:MissingSession', ...
    'subjectName and sessionName are required.');
end
if nargin < 3 || isempty(opts)
  opts = struct();
end

opts.beamBreakTask = 'semicircle';
if ~isfield(opts, 'runD2TrialRateCorrelation') || isempty(opts.runD2TrialRateCorrelation)
  opts.runD2TrialRateCorrelation = false;
end

out = interval_criticality_metrics_engagement(subjectName, sessionName, opts);
end
