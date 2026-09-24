%% Process Neuropixels 1 LFP into session-folder lfp.mat
% Set sessionType, subjectName, sessionName, and brainAreas, then run.
% Split np1-lfp_*.raw files from the same date are stitched in time order.
% Each area is the mean of two channels near 1/3 and 2/3 of its depth range.
% Traces are low-passed < 300 Hz, sampled at 1000 Hz, saved as single in lfp.mat.

sessionType = 'interval';  % 'spontaneous' or 'interval'
subjectName = 'ey9166';
sessionName = 'ey9166_2026_04_02';
brainAreas = {'M23', 'M56', 'DS', 'VS'};  % also available: 'CC'

sessionType = 'spontaneous';
subjectName = 'ey9166';
sessionName = 'ey9166_2026_03_24';

thisDir = fileparts(mfilename('fullpath'));
addpath(thisDir);
addpath(fileparts(thisDir));

process_np1_lfp(sessionType, subjectName, sessionName, 'brainAreas', brainAreas);
