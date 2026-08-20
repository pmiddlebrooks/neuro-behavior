function [eventTimes, eventTypes, trials] = load_semicircle_beam_break_events( ...
    paths, subjectName, sessionName)
% LOAD_SEMICIRCLE_BEAM_BREAK_EVENTS - TaskMatrix times used as engagement events
%
% Variables:
%   paths       - Paths struct from get_paths
%   subjectName - Subject folder (e.g. 'AS1')
%   sessionName - Session .mat basename (e.g. 'AS1_0618_WellLearned')
%
% Goal:
%   Collect finite TaskMatrix timestamps that count as engaged (non-engaged
%   segments cannot contain these times, except absorbed isolated singles):
%     col 2  trialStartTime
%     col 6  choicePort poke time
%     col 9  leave home first after trial start
%     col 10 enter home for trial start
%     col 11 leave home last after trial start
%   choicePort chosen (col 4) is a port ID, not a time, so it is omitted.
%   trialEnd (col 8) is not an engagement event.
%   Failed trials (outcome -1) are included when the timestamp columns above
%   are finite. Duplicate timestamps keep one label (correct/error preferred).
%
% Returns:
%   eventTimes - Absolute event times (s), sorted unique
%   eventTypes - String array aligned with eventTimes:
%                "correct" / "error" / "choicePoke" for pokes,
%                "trialStart", "leaveHomeFirst", "enterHomeStart",
%                "leaveHomeLast" for the other columns
%   trials     - Poke-only table (rewarded/unrewarded) with type, pokeTimeSec,
%                outcomeTimeSec (interval-compatible)

dataFile = fullfile(paths.semicircleDataPath, subjectName, [sessionName, '.mat']);
if ~exist(dataFile, 'file')
  error('load_semicircle_beam_break_events:FileNotFound', ...
    'Semicircle data file not found: %s', dataFile);
end

dataS = load(dataFile, 'TaskMatrix');
if ~isfield(dataS, 'TaskMatrix') || isempty(dataS.TaskMatrix)
  error('load_semicircle_beam_break_events:NoTaskMatrix', ...
    'TaskMatrix missing in %s', dataFile);
end

taskMatrix = dataS.TaskMatrix;
[eventTimes, eventTypes, trials, eventCounts] = ...
  collect_semicircle_engagement_events(taskMatrix);

if isempty(eventTimes)
  error('load_semicircle_beam_break_events:NoEvents', ...
    'No TaskMatrix engagement event times found in %s', sessionName);
end

fprintf(['Semicircle engagement events: %d unique times ', ...
  '(%d trialStart, %d enterHome, %d leaveHomeFirst, %d leaveHomeLast, ', ...
  '%d rewarded pokes, %d unrewarded pokes, %d other pokes; failed trials: %d)\n'], ...
  numel(eventTimes), eventCounts.nTrialStart, eventCounts.nEnterHome, ...
  eventCounts.nLeaveHomeFirst, eventCounts.nLeaveHomeLast, ...
  eventCounts.nRewardPoke, eventCounts.nUnrewardedPoke, eventCounts.nOtherPoke, ...
  eventCounts.nFailedTrials);
end

function [eventTimes, eventTypes, trials, eventCounts] = ...
    collect_semicircle_engagement_events(taskMatrix)
% COLLECT_SEMICIRCLE_ENGAGEMENT_EVENTS - Parse TaskMatrix into engagement times
%
% Variables:
%   taskMatrix - Session TaskMatrix (see data README columns)
%
% Goal:
%   Build unique engagement timestamps, poke-based trials table, and counts.

trialStartSec = taskMatrix(:, 2);
outcome = taskMatrix(:, 3);
pokeTimeSecAbs = taskMatrix(:, 6);
leaveHomeFirstSec = taskMatrix(:, 9);
enterHomeStartSec = taskMatrix(:, 10);
leaveHomeLastSec = taskMatrix(:, 11);

isReward = outcome == 1 & isfinite(pokeTimeSecAbs);
isUnrewarded = outcome == 0 & isfinite(pokeTimeSecAbs);
isFailedPoke = outcome == -1 & isfinite(pokeTimeSecAbs);
keepPokeMask = isReward | isUnrewarded;

nKeepPoke = sum(keepPokeMask);
trialTypes = strings(nKeepPoke, 1);
trialTypes(isReward(keepPokeMask)) = "correct";
trialTypes(isUnrewarded(keepPokeMask)) = "error";
pokeLatencySec = pokeTimeSecAbs(keepPokeMask) - trialStartSec(keepPokeMask);
outcomeTimesSec = pokeTimeSecAbs(keepPokeMask);
trials = table(trialTypes, pokeLatencySec, outcomeTimesSec, ...
  'VariableNames', {'type', 'pokeTimeSec', 'outcomeTimeSec'});
if ~isempty(trials)
  trials = sortrows(trials, 'outcomeTimeSec');
end

pokeTypes = strings(size(pokeTimeSecAbs));
pokeTypes(:) = "choicePoke";
pokeTypes(isReward) = "correct";
pokeTypes(isUnrewarded) = "error";
pokeTypes(isFailedPoke) = "choicePoke";

nTrial = size(taskMatrix, 1);
eventTimesRaw = [ ...
  trialStartSec; ...
  pokeTimeSecAbs; ...
  leaveHomeFirstSec; ...
  enterHomeStartSec; ...
  leaveHomeLastSec];
eventTypesRaw = [ ...
  repmat("trialStart", nTrial, 1); ...
  pokeTypes; ...
  repmat("leaveHomeFirst", nTrial, 1); ...
  repmat("enterHomeStart", nTrial, 1); ...
  repmat("leaveHomeLast", nTrial, 1)];

keepEvent = isfinite(eventTimesRaw) & strlength(eventTypesRaw) > 0;
eventTimesRaw = eventTimesRaw(keepEvent);
eventTypesRaw = eventTypesRaw(keepEvent);

[eventTimes, eventTypes] = unique_engagement_event_times(eventTimesRaw, eventTypesRaw);

eventCounts = struct();
eventCounts.nTrialStart = sum(eventTypes == "trialStart");
eventCounts.nEnterHome = sum(eventTypes == "enterHomeStart");
eventCounts.nLeaveHomeFirst = sum(eventTypes == "leaveHomeFirst");
eventCounts.nLeaveHomeLast = sum(eventTypes == "leaveHomeLast");
eventCounts.nRewardPoke = sum(eventTypes == "correct");
eventCounts.nUnrewardedPoke = sum(eventTypes == "error");
eventCounts.nOtherPoke = sum(eventTypes == "choicePoke");
eventCounts.nFailedTrials = sum(outcome == -1);
end

function [eventTimes, eventTypes] = unique_engagement_event_times(eventTimes, eventTypes)
% UNIQUE_ENGAGEMENT_EVENT_TIMES - One label per timestamp (prefer poke outcomes)
%
% Variables:
%   eventTimes - Event times (s)
%   eventTypes - String labels aligned with eventTimes
%
% Goal:
%   Sort and collapse identical times so absorbSingleEvents counts one event
%   per instant. Prefer "correct", then "error", then "choicePoke".

eventTimes = eventTimes(:);
eventTypes = eventTypes(:);
if isempty(eventTimes)
  return;
end

[eventTimes, sortOrd] = sort(eventTimes);
eventTypes = eventTypes(sortOrd);

priority = zeros(size(eventTypes));
priority(eventTypes == "choicePoke") = 1;
priority(eventTypes == "error") = 2;
priority(eventTypes == "correct") = 3;

keep = true(size(eventTimes));
iEvent = 1;
nEvent = numel(eventTimes);
while iEvent <= nEvent
  jEvent = iEvent;
  while jEvent < nEvent && eventTimes(jEvent + 1) == eventTimes(iEvent)
    jEvent = jEvent + 1;
  end
  if jEvent > iEvent
    [~, bestRel] = max(priority(iEvent:jEvent));
    keep(iEvent:jEvent) = false;
    keep(iEvent + bestRel - 1) = true;
  end
  iEvent = jEvent + 1;
end

eventTimes = eventTimes(keep);
eventTypes = eventTypes(keep);
end
