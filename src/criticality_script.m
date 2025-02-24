%%
opts = neuro_behavior_options;
opts.minActTime = .16;
opts.collectStart = 0 * 60 * 60; % seconds
opts.collectFor = 60 * 60; % seconds

%% Need to find an appropriate bin size for each brain area
idSelect = idDS;
opts.frameSize = .003;

idSelect = idM56;
opts.frameSize = .005;

opts.useOverlappingBins = 0;
opts.windowSize = .2;
opts.stepSize = opts.frameSize;

getDataType = 'spikes';
get_standard_data

%%
sumSpikes = sum(dataMat(:, idSelect), 2);
figure(91);
histogram(sumSpikes)
sum(sumSpikes == 0) / length(sumSpikes)



%% go to demoempdata.m

asdfMat = rastertoasdf2(dataMat(:,idSelect)', opts.frameSize*1000, 'CBModel', 'Spikes', 'DS')


Av = avprops(asdfMat, 'ratio', 'fingerprint');

% Go to demoempdata.m to run rest of codes using Av