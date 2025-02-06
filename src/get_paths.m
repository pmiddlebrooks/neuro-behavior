function paths = get_paths

%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here

if exist('E:/Projects', 'dir')
            homePath = 'E:/Projects';
          figurePath = 'E:/Projects/neuro-behavior/docs/';
            bhvDataPath = 'E:/Projects/neuro-behavior/data/processed_behavior/';
            nrnDataPath = 'E:/Projects/neuro-behavior/data/raw_ephys/';
            saveDataPath = 'E:/Projects/neuro-behavior/data/';
            dropPath = 'C:/Users/yttri-lab/Dropbox/Data/';
elseif exist('Z:/middlebrooks/', 'dir')
            figurePath = 'Z:/middlebrooks/Projects/neuro-behavior/docs/';
            bhvDataPath = 'Z:/middlebrooks/Projects/neuro-behavior/data/processed_behavior/';
            nrnDataPath = 'Z:/middlebrooks/Projects/neuro-behavior/data/raw_ephys/';
            saveDataPath = 'Z:/middlebrooks/Projects/neuro-behavior/data/';
elseif exist('/Users/paulmiddlebrooks/Projects/', 'dir')
            homePath = '/Users/paulmiddlebrooks/Projects/';
            figurePath = '/Users/paulmiddlebrooks/Projects/neuro-behavior/docs/';
            bhvDataPath = '/Users/paulmiddlebrooks/Projects/neuro-behavior/data/processed_behavior/';
            nrnDataPath = '/Users/paulmiddlebrooks/Projects/neuro-behavior/data/raw_ephys/';
            saveDataPath = '/Users/paulmiddlebrooks/Projects/neuro-behavior/data/';
            dropPath = '/Users/paulmiddlebrooks/Library/CloudStorage/Dropbox/Data/';
           
end
if ~exist(figurePath, 'dir')
    mkdir(figurePath);
end
if ~exist(bhvDataPath, 'dir')
    mkdir(bhvDataPath);
end
if ~exist(nrnDataPath, 'dir')
    mkdir(nrnDataPath);
end
if ~exist(saveDataPath, 'dir')
    mkdir(saveDataPath);
end


paths.homePath = homePath;
paths.figurePath = figurePath;
paths.bhvDataPath = bhvDataPath;
paths.nrnDataPath = nrnDataPath;
paths.saveDataPath = saveDataPath;
paths.dropPath = dropPath;
