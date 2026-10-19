%% Process Neuropixels 2 LFP into session-folder lfp.mat
% Point this at the *_lfp.raw file produced by extract_lfp_np2.sh.
% The date in sessionName must match the date in np2-<probe>-ephys_<date>T*.
% Each area is the mean of two channels near 1/3 and 2/3 of its depth range.
% Traces are low-passed < 300 Hz, sampled at 1250 Hz, saved as single in lfp.mat.

sessionType = 'spontaneous';  % 'spontaneous' or 'interval'
subjectName = '';             % subject folder, e.g. 'ey9166'
sessionName = '';             % e.g. 'ey9166_2026_05_03' for np2-a-ephys_2026-05-03T...
probeId = 'a';                % letter in np2-<probe>-ephys_*_lfp.raw
brainAreas = {'M23', 'M56', 'DS', 'VS'};  % also available: 'CC'

if isempty(subjectName) || isempty(sessionName)
    error('run_process_np2_lfp:MissingSession', ...
        'Set subjectName and sessionName in run_process_np2_lfp.m before running.');
end

thisDir = fileparts(mfilename('fullpath'));
addpath(thisDir);
addpath(fileparts(thisDir));

process_np2_lfp(sessionType, subjectName, sessionName, ...
    'brainAreas', brainAreas, 'probeId', probeId);
