%% Get data from get_standard_data

opts = neuro_behavior_options;
opts.collectStart = 0; % seconds
opts.collectFor = 60*60*4; % seconds

get_standard_data


%%

% which behavior to analyze?
behavior = 'face_groom_1';
bhvInd = analyzeCodes(strcmp(behavior, analyzeBhv));

periTime = -.2 : opts.frameSize : .2;
periWindow = periTime(1:end-1) / opts.frameSize; % frames around onset w.r.t. zWindow (remove last frame)

nBout = size(eventMat{bhvInd}, 3);
periSpikeCt = sum(eventMat{bhvInd}(fullStartInd + periWindow, :, :), 1);
periSpikeCt = permute(periSpikeCt, [3 2 1]);
% make spike rates instead of counts?
meanSpikeCt = mean(periSpikeCt, 1);

resSpikeCt = periSpikeCt - meanSpikeCt;

%% Individual neuron slow drift for given behavior
% Which brain area to analyze?
idInd = idM56;

figure(654);
for i = 1:length(idInd)
    clf; hold on
scatter(1:nBout, resSpikeCt(:,idInd(i)), 'filled')
plot(1:nBout, movmean(resSpikeCt(:,idInd(i)), 20), 'linewidth', 3)
end

%% PCA projections of slow drift for given behavior/brain area

% Which brain area to analyze?
idInd = idM56;
% Perform PCA
[coeff, score, ~, ~, explained] = pca(resSpikeCt(:,idInd));
figure(62); clf; 
for iComp = 1:3
    subplot(3, 1, iComp); hold on;
scatter(1:size(score, 1), score(:,iComp))
plot(1:size(score, 1), movmean(score(:,iComp), 20), 'lineWidth', 3)
ylabel(['Component ', num2str(iComp)])
xlabel('Bouts')
end
sgtitle(['PCA projections: ', behavior], 'interpreter', 'none')