function filePath = get_ymaze_data(confidence, volatility, controlStim)

paths = get_paths;


if confidence == 1
    cString = 'NoConf';
elseif confidence == 2
    cString = 'LowConf';
elseif confidence == 3
    cString = 'HighConf';
end

if volatility == 1
    vString = 'LowVol';
elseif volatility == 2
    vString = 'HighVol';
end

if controlStim == 1
    stimString = 'Control';
elseif controlStim == 2
    stimString = 'Stim';
end

filePath = [paths.dropPath, 'ymaze/data/', cString, vString, stimString, '_withDTs.mat'];