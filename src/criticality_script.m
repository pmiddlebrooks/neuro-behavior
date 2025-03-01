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
%% Set variables and get data
opts.minFiringRate = .1;
opts.collectFor = 20 * 60;
getDataType = 'spikes';

nIter = 10;
nSubsample = 20;
% binSizes = [.005, .01, .015, .02];
binSizes = [.01, .015, .02 .025];
% binSizes = [.005, .05];
% binSizes = [.005];

% reach data
dataR = load(fullfile(paths.saveDataPath, 'reach_data/Y4_100623_Spiketimes_idchan.mat'));

brPeak = zeros(nIter, 2, length(binSizes));
tau = brPeak;
alpha = tau;
sigmaNuZInvSD = tau;
%%    Is naturalistic data closer to criticality than reach data?
tic
for b = 1 : length(binSizes)
    for iter = 1:nIter
        fprintf('\nIteration: %d\tBinSize: %.3f\t Time Elapsed: %.1f min\n', iter, binSizes(b), toc/60)

        opts.frameSize = binSizes(b);


        % Naturalistich data4
        get_standard_data

        % Randomize a subsample of neurons
        idSelect = idVS(randperm(length(idVS), nSubsample));


        asdfMat = rastertoasdf2(dataMat(:,idSelect)', opts.frameSize*1000, 'CBModel', 'Spikes', 'DS');
        Av = avprops(asdfMat, 'ratio', 'fingerprint');

        [brPeak(iter, 1, b), tau(iter, 1, b), alpha(iter, 1, b), sigmaNuZInvSD(iter, 1, b)] = avalanche_log(Av);


        % Mark's data
        [dataMatR, idLabels, areaLabels, rmvNeurons] = neural_matrix_mark_data(dataR, opts);
        % Ensure dataMatR is same size as dataMat
        dataMatR = dataMatR(1:size(dataMat, 1),:);


        idM23R = find(strcmp(areaLabels, 'M23'));
        idM56R = find(strcmp(areaLabels, 'M56'));
        idDSR = find(strcmp(areaLabels, 'DS'));
        idVSR = find(strcmp(areaLabels, 'VS'));

        % Randomize a subsample of neurons
        idSelect = idVSR(randperm(length(idVSR), nSubsample));

        asdfMatR = rastertoasdf2(dataMatR(:,idSelect)', opts.frameSize*1000, 'CBModel', 'Spikes', 'DS');
        AvR = avprops(asdfMatR, 'ratio', 'fingerprint');

        [brPeak(iter, 2, b), tau(iter, 2, b), alpha(iter, 2, b), sigmaNuZInvSD(iter, 2, b)] = avalanche_log(AvR);



    end
end



%%
brPeakDS = brPeak;
tauDS = tau;
alphaDS = alpha;
sigmaNuZInvSDDS = sigmaNuZInvSD;

fileName = fullfile(paths.saveDataPath, 'criticality_parameters.mat');
save(fileName, 'brPeakDS', 'tauDS', 'alphaDS', 'sigmaNuZInvSDDS', '-append')















%% Test transition vs. within-bout criticality across all behaviors. 
opts.minFiringRate = .1;
opts.collectFor = 45 * 60;
getDataType = 'spikes';

opts.frameSize = .015;
        get_standard_data
%%
        preTime = .15;
postTime = .15;
transWindow = (-preTime/opts.frameSize : postTime/opts.frameSize - 1);

% Transition indices
preInd = find(diff(bhvID) ~= 0); % 1 frame prior to all behavior transitions

for bout = 1 : length(preInd)
% transInd = unique(sort(reshape(preInd + transWindow(:)', [], 1)))

% Collect transition avalanche data
transInd = preInd(bout) + transWindow;


% Collect within-bout avalanche data
withinInd = preInd(bout) + transWindow(end) + 1 : preInd(bout+1) + transWindow(1) - 1;


end
















%% Behavior scale-free?
% Define behaviors to include (input as a vector)
bhvInclude = analyzeCodes           ; % Modify this as needed
someBhvs = 'groom';
bhvInclude = codes(contains(behaviors, someBhvs));

% Load behavior labels (bhvID) - Ensure bhvID is a column vector
bhvID = bhvID(:);

% Filter bhvID to only include specified behaviors
include_mask = ismember(bhvID, bhvInclude);
bhvFiltered = bhvID(include_mask);

% Compute behavior durations (time spent in each behavior state)
change_points = find(diff([NaN; bhvFiltered]) ~= 0); % Find transitions
durations = diff(change_points); % Compute durations

% Fit power-law distribution using Clauset's plfit (ensure you have plfit.m)
[alpha, xmin, L] = plfit(durations);

% Compare with exponential and lognormal distributions
% Exponential fit
lambda = 1/mean(durations);
L_exp = sum(log(lambda * exp(-lambda * durations)));

% Lognormal fit
mu = mean(log(durations));
sigma = std(log(durations));
L_lognorm = sum(log((1./(durations * sigma * sqrt(2*pi))) .* exp(-((log(durations)-mu).^2) / (2*sigma^2))));

% Compare likelihoods
R_exp = 2 * (L - L_exp);
R_lognorm = 2 * (L - L_lognorm);

% Compute p-values
p_exp = 1 - chi2cdf(R_exp, 1);
p_lognorm = 1 - chi2cdf(R_lognorm, 1);

% Display results
fprintf('Power-law exponent (alpha): %.3f\n', alpha);
fprintf('Comparison with exponential: R=%.3f, p=%.3f\n', R_exp, p_exp);
fprintf('Comparison with lognormal: R=%.3f, p=%.3f\n', R_lognorm, p_lognorm);

% Plot the distribution
figure;
% Define logarithmic bins
binEdges = logspace(log10(min(durations)), log10(max(durations)), 20); % Adjust number of bins
% Compute histogram
[counts, edges] = histcounts(durations, binEdges, 'Normalization', 'pdf');
% Compute bin centers
binCenters = sqrt(edges(1:end-1) .* edges(2:end));
bar(binCenters, counts, 'histc');
hold on;
plot(sort(durations), (xmin ./ sort(durations)).^alpha, 'r', 'LineWidth', 2);
set(gca, 'XScale', 'log', 'YScale', 'log');
xlabel('Duration'); ylabel('Probability Density');
title('Scale-Free Test: Behavior Durations');
legend('Empirical Data', 'Power-Law Fit');








%%
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

