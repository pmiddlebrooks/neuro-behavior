function paths = rrm_get_paths(computerDriveName)

%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here


switch computerDriveName
    case 'ROSETTA'
            figurePath = 'E:/Projects/ridgeRegress/docs/';
            bhvDataPath = 'E:/Projects/neuro-behavior/data/processed_behavior/';
            freeDataPath = 'E:/Projects/neuro-behavior/data/raw_ephys/';
            saveDataPath = 'E:/Projects/ridgeRegress/data/';
    case 'Z'
            figurePath = 'Z:/middlebrooks/Projects/ridgeregress/docs/';
            bhvDataPath = 'Z:/middlebrooks/Projects/neuro-behavior/data/processed_behavior/';
            freeDataPath = 'Z:/middlebrooks/Projects/neuro-behavior/data/raw_ephys/';
            saveDataPath = 'Z:/middlebrooks/Projects/ridgeRegress/data/';
    case 'home'
            figurePath = '/Users/paulmiddlebrooks/Projects/ridgeregress/docs/';
            bhvDataPath = '/Users/paulmiddlebrooks/Projects/neuro-behavior/data/processed_behavior/';
            freeDataPath = '/Users/paulmiddlebrooks/Projects/neuro-behavior/data/raw_ephys/';
            saveDataPath = '/Users/paulmiddlebrooks/Projects/ridgeRegress/data/';
end
if ~exist(figurePath, 'dir')
    mkdir(figurePath);
end
if ~exist(bhvDataPath, 'dir')
    mkdir(bhvDataPath);
end
if ~exist(freeDataPath, 'dir')
    mkdir(freeDataPath);
end
if ~exist(saveDataPath, 'dir')
    mkdir(saveDataPath);
end


paths.figurePath = figurePath;
paths.bhvDataPath = bhvDataPath;
paths.freeDataPath = freeDataPath;
paths.saveDataPath = saveDataPath;
