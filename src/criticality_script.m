%%
opts = neuro_behavior_options;
opts.minActTime = .16;
opts.collectStart = 0 * 60 * 60; % seconds
opts.collectFor = 30 * 60; % seconds

paths = get_paths;
%% Need to find an appropriate bin size for each brain area
opts.frameSize = .005; % DS
% opts.frameSize = .01; % M56

opts.useOverlappingBins = 0;
opts.windowSize = .2;
opts.stepSize = opts.frameSize;

getDataType = 'spikes';
get_standard_data

idSelect = idDS;
% idSelect = idM56;

%%
sumSpikes = sum(dataMat(:, idSelect), 2);
figure(91);
histogram(sumSpikes)
sum(sumSpikes == 0) / length(sumSpikes)



%% go to demoempdata.m

asdfMat = rastertoasdf2(dataMat(:,idSelect)', opts.frameSize*1000, 'CBModel', 'Spikes', 'DS')


Av = avprops(asdfMat, 'ratio', 'fingerprint');

% Go to demoempdata.m to run rest of codes using Av












%% Mark's reaching data
opts.frameSize = .005; % DS
opts.frameSize = .01; % DS
opts.frameSize = .0075; % M56
opts.minFiringRate = .1;

data = load(fullfile(paths.saveDataPath, 'reach_data/Y4_100623_Spiketimes_idchan.mat'));
[dataMat, idLabels, areaLabels, rmvNeurons] = neural_matrix_mark_data(data, opts);

idM23 = find(strcmp(areaLabels, 'M23'));
idM56 = find(strcmp(areaLabels, 'M56'));
idDS = find(strcmp(areaLabels, 'DS'));
idVS = find(strcmp(areaLabels, 'VS'));
fprintf('%d M23\n%d M56\n%d DS\n%d VS\n', length(idM23), length(idM56), length(idDS), length(idVS))

%%
idSelect = idDS;
idSelect = idM56;

%
sumSpikes = sum(dataMat(:, idSelect), 2);
figure(91);
histogram(sumSpikes, 'Normalization', 'pdf')
% sum(sumSpikes == 0) / length(sumSpikes)


%% go to demoempdata.m

asdfMat = rastertoasdf2(dataMat(:,idSelect)', opts.frameSize*1000, 'CBModel', 'Spikes', 'DS')


Av = avprops(asdfMat, 'ratio', 'fingerprint');

% Go to demoempdata.m to run rest of codes using Av










%% Set variables and get data
opts.minFiringRate = .1;
opts.collectFor = 45 * 60;
getDataType = 'spikes';

nIter = 20;
nSubsample = 20;
binSizes = [.005, .01, .025 .05];
% binSizes = [.005, .05];
% binSizes = [.005];

% reach data
dataR = load(fullfile(paths.saveDataPath, 'reach_data/Y4_100623_Spiketimes_idchan.mat'));

brPeak = zeros(nIter, 2, length(binSizes));
tau = brPeak;
alpha = tau;
sigmaNuZInvSD = tau;
%%
tic
for b = 1 : length(binSizes)
    for iter = 1:nIter
        fprintf('\nIteration: %d\tBinSize: %.3f\t Time Elapsed: %.1f min\n', iter, binSizes(b), toc/60)

        opts.frameSize = binSizes(b);


        % Naturalistich data
        get_standard_data

        % Randomize a subsample of neurons
        idSelect = idM56(randperm(length(idM56), nSubsample));


        asdfMatM23 = rastertoasdf2(dataMat(:,idSelect)', opts.frameSize*1000, 'CBModel', 'Spikes', 'DS');
        Av = avprops(asdfMatM23, 'ratio', 'fingerprint');

        [brPeak(iter, 1, b), tau(iter, 1, b), alpha(iter, 1, b), sigmaNuZInvSD(iter, 1, b)] = avalanche_log(Av);


        % % Mark's data
        % [dataMatR, idLabels, areaLabels, rmvNeurons] = neural_matrix_mark_data(dataR, opts);
        % % Ensure dataMatR is same size as dataMat
        % dataMatR = dataMatR(1:size(dataMat, 1),:);
        %
        %
        % idM23R = find(strcmp(areaLabels, 'M23'));
        % idM56R = find(strcmp(areaLabels, 'M56'));
        % idDSR = find(strcmp(areaLabels, 'DS'));
        % idVSR = find(strcmp(areaLabels, 'VS'));
        %
        % % Randomize a subsample of neurons
        % idSelect = idM56R(randperm(length(idM56R), nSubsample));
        %
        % asdfMatM23R = rastertoasdf2(dataMatR(:,idSelect)', opts.frameSize*1000, 'CBModel', 'Spikes', 'DS');
        % AvR = avprops(asdfMatM23R, 'ratio', 'fingerprint');
        %
        % [brPeak(iter, 2, b), tau(iter, 2, b), alpha(iter, 2, b), sigmaNuZInvSD(iter, 2, b)] = avalanche_log(AvR);



    end
end


function [brPeak, tau, alpha, sigmaNuZInvSD] = avalanche_log(Av)

minBR = min(Av.branchingRatio);
maxBR = max(Av.branchingRatio);

nEdges = 25;
edges = minBR:((maxBR-minBR)/(nEdges - 1)):maxBR;

BRhist = histcounts(Av.branchingRatio, edges);
[m, idx] = max(BRhist(2:end));
brPeak = edges(idx+1);

% size distribution (SZ)
[tau, xminSZ, xmaxSZ, sigmaSZ, pSZ, pCritSZ, ksDR, DataSZ] =...
    avpropvals(Av.size, 'size');
% avpropvals(Av.size, 'size', 'plot');

% size distribution (SZ) with cutoffs
UniqSizes = unique(Av.size);
Occurances = hist(Av.size,UniqSizes);
AllowedSizes = UniqSizes(Occurances >= 20);
AllowedSizes(AllowedSizes < 4) = [];
LimSize = Av.size(ismember(Av.size,AllowedSizes));
[tau, xminSZ, xmaxSZ, sigmaSZ, pSZ, pCritSZ, DataSZ] =...
    avpropvals(LimSize, 'size');
% avpropvals(LimSize, 'size', 'plot');
tau = cell2mat(tau);

% duration distribution (DR)
[alpha, xminDR, xmaxDR, sigmaDR, pDR, pCritDR, ksDR, DataDR] =...
    avpropvals(Av.duration, 'duration');
% avpropvals(Av.duration, 'duration', 'plot');
alpha = cell2mat(alpha);

% size given duration distribution (SD)
[sigmaNuZInvSD, waste, waste, sigmaSD] = avpropvals({Av.size, Av.duration},...
    'sizgivdur', 'durmin', xminDR{1}, 'durmax', xmaxDR{1});
% 'sizgivdur', 'durmin', xminDR{1}, 'durmax', xmaxDR{1}, 'plot');
end

