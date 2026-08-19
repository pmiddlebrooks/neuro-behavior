function [eventTimes, eventTypes, trials] = load_interval_beam_break_events( ...
    paths, subjectName, sessionName, minLeaveSec)
% LOAD_INTERVAL_BEAM_BREAK_EVENTS - Correct and error outcome times from session log
%
% Variables:
%   paths, subjectName, sessionName - Session location under intervalDataPath
%   minLeaveSec                     - Min confirmed leave duration (s); default 0.1
%
% Goal:
%   Parse revised_interval_*.csv and return session-relative outcome times for
%   correct (REWARD) and error (ERROR) beam breaks, matching interval_session_performance.
%
% Returns:
%   eventTimes - Outcome times (s), sorted
%   eventTypes - String array "correct" / "error"
%   trials     - Table with type, pokeTimeSec, outcomeTimeSec

if nargin < 4 || isempty(minLeaveSec)
  minLeaveSec = 0.1;
end

sessionDir = fullfile(paths.intervalDataPath, subjectName, sessionName);
if ~exist(sessionDir, 'dir')
  error('load_interval_beam_break_events:SessionNotFound', ...
    'Session directory not found: %s', sessionDir);
end

csvPath = find_interval_csv(sessionDir);
fprintf('Loading interval log: %s\n', csvPath);
logTable = parse_interval_log(csvPath);
trials = extract_interval_trials(logTable, minLeaveSec);

eventTimes = trials.outcomeTimeSec(:);
eventTypes = trials.type(:);
[eventTimes, ord] = sort(eventTimes);
eventTypes = eventTypes(ord);
end

function csvPath = find_interval_csv(sessionDir)
% FIND_INTERVAL_CSV - Most recent revised_interval_*.csv in session folder

csvFiles = dir(fullfile(sessionDir, 'revised_interval_*.csv'));
if isempty(csvFiles)
  error('load_interval_beam_break_events:NoCsv', ...
    'No revised_interval_*.csv found in %s', sessionDir);
end
[~, newestIdx] = max([csvFiles.datenum]);
csvPath = fullfile(sessionDir, csvFiles(newestIdx).name);
end

function logTable = parse_interval_log(csvPath)
% PARSE_INTERVAL_LOG - Read Arduino/Processing interval task CSV

rawTable = readtable(csvPath, 'TextType', 'string');
varNames = lower(string(rawTable.Properties.VariableNames));

timeCol = find(contains(varNames, 'timestamp'), 1);
eventCol = find(strcmp(varNames, 'event') | contains(varNames, 'event'), 1);
valueCol = find(strcmp(varNames, 'value') | contains(varNames, 'value'), 1);

if isempty(timeCol) || isempty(eventCol) || isempty(valueCol)
  error('load_interval_beam_break_events:BadCsv', ...
    'CSV must contain timestamp, event, and value columns: %s', csvPath);
end

timestampMs = rawTable{:, timeCol};
if iscell(timestampMs)
  timestampMs = cellfun(@str2double, timestampMs);
elseif isstring(timestampMs) || ischar(timestampMs)
  timestampMs = str2double(string(timestampMs));
end
timestampMs = double(timestampMs);

eventNames = string(rawTable{:, eventCol});
eventValues = rawTable{:, valueCol};
if iscell(eventValues) || isstring(eventValues)
  eventValues = str2double(string(eventValues));
end
eventValues = double(eventValues);

validRows = ~isnan(timestampMs) & eventNames ~= "" & ~ismissing(eventNames);
logTable = table(timestampMs(validRows), eventNames(validRows), eventValues(validRows), ...
  'VariableNames', {'timestampMs', 'event', 'value'});
logTable = sortrows(logTable, 'timestampMs');
end

function trials = extract_interval_trials(logTable, minLeaveSec)
% EXTRACT_INTERVAL_TRIALS - Segment ERROR/REWARD trials from event log
%
% Goal: For each ERROR or post-training REWARD, record poke time since last leave
% and session-relative outcome time (same logic as interval_session_performance).

minLeaveMs = minLeaveSec * 1000;

leavePending = false;
leaveConfirmStartMs = NaN;
initialExitMs = NaN;
leaveTimeMs = NaN;
timerArmed = false;
beamState = 0;
firstRewardSeen = false;

sessionOriginMs = min(logTable.timestampMs);
trialTypes = strings(0, 1);
pokeTimesSec = [];
outcomeTimesSec = [];

eventNames = logTable.event;
timestampsMs = logTable.timestampMs;
eventValues = logTable.value;
nEvents = height(logTable);

for eventIdx = 1:nEvents
  eventTimeMs = timestampsMs(eventIdx);
  eventName = eventNames(eventIdx);
  eventValue = eventValues(eventIdx);

  if leavePending && beamState == 0 && (eventTimeMs - leaveConfirmStartMs) >= minLeaveMs
    leaveTimeMs = initialExitMs;
    timerArmed = true;
    leavePending = false;
  end

  if eventName == "B"
    beamState = eventValue;
    if eventValue == 0
      leavePending = true;
      initialExitMs = eventTimeMs;
      leaveConfirmStartMs = eventTimeMs;
    else
      leavePending = false;
    end
  elseif eventName == "ERROR"
    if timerArmed && ~isnan(leaveTimeMs)
      trialTypes(end + 1, 1) = "error"; %#ok<AGROW>
      pokeTimesSec(end + 1, 1) = (eventTimeMs - leaveTimeMs) / 1000; %#ok<AGROW>
      outcomeTimesSec(end + 1, 1) = (eventTimeMs - sessionOriginMs) / 1000; %#ok<AGROW>
    end
    timerArmed = false;
    leaveTimeMs = NaN;
  elseif eventName == "REWARD"
    if ~firstRewardSeen
      firstRewardSeen = true;
    elseif timerArmed && ~isnan(leaveTimeMs)
      trialTypes(end + 1, 1) = "correct"; %#ok<AGROW>
      pokeTimesSec(end + 1, 1) = (eventTimeMs - leaveTimeMs) / 1000; %#ok<AGROW>
      outcomeTimesSec(end + 1, 1) = (eventTimeMs - sessionOriginMs) / 1000; %#ok<AGROW>
    end
    timerArmed = false;
    leaveTimeMs = NaN;
  end
end

trials = table(trialTypes, pokeTimesSec, outcomeTimesSec, ...
  'VariableNames', {'type', 'pokeTimeSec', 'outcomeTimeSec'});
end
