function [eventTimes, eventTypes, trials] = load_semicircle_beam_break_events( ...
    paths, subjectName, sessionName)
% LOAD_SEMICIRCLE_BEAM_BREAK_EVENTS - Rewarded / unrewarded choice-poke times
%
% Variables:
%   paths       - Paths struct from get_paths
%   subjectName - Subject folder (e.g. 'AS1')
%   sessionName - Session .mat basename (e.g. 'AS1_0618_WellLearned')
%
% Goal:
%   Use TaskMatrix choice-port poke times as engagement events:
%     outcome 1 (rewarded)   -> "correct"
%     outcome 0 (unrewarded) -> "error"
%   Failed trials (outcome -1) are omitted (usually no choice poke).
%
% Returns:
%   eventTimes - Absolute poke times (s), sorted
%   eventTypes - String array "correct" / "error"
%   trials     - Table with type, pokeTimeSec, outcomeTimeSec (interval-compatible)

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
trialStartSec = taskMatrix(:, 2);
outcome = taskMatrix(:, 3);
pokeTimeSecAbs = taskMatrix(:, 6);

isReward = outcome == 1 & isfinite(pokeTimeSecAbs);
isUnrewarded = outcome == 0 & isfinite(pokeTimeSecAbs);
keepMask = isReward | isUnrewarded;

if ~any(keepMask)
  error('load_semicircle_beam_break_events:NoEvents', ...
    'No rewarded/unrewarded choice pokes found in %s', sessionName);
end

nKeep = sum(keepMask);
trialTypes = strings(nKeep, 1);
rewardKeep = isReward(keepMask);
unrewardedKeep = isUnrewarded(keepMask);
trialTypes(rewardKeep) = "correct";
trialTypes(unrewardedKeep) = "error";
pokeLatencySec = pokeTimeSecAbs(keepMask) - trialStartSec(keepMask);
outcomeTimesSec = pokeTimeSecAbs(keepMask);

trials = table(trialTypes, pokeLatencySec, outcomeTimesSec, ...
  'VariableNames', {'type', 'pokeTimeSec', 'outcomeTimeSec'});
trials = sortrows(trials, 'outcomeTimeSec');

eventTimes = trials.outcomeTimeSec(:);
eventTypes = trials.type(:);

fprintf('Semicircle beam breaks: %d rewarded, %d unrewarded (failed omitted: %d)\n', ...
  sum(isReward), sum(isUnrewarded), sum(outcome == -1));
end
