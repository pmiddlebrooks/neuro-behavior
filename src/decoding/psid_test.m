%%                     Compare neuro-behavior in PCA spaces
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%

opts = neuro_behavior_options;
opts.minActTime = .16;
opts.collectStart = 0 * 60 * 60; % seconds
opts.collectFor = 10 * 60; % seconds
opts.frameSize = .1;
%
monitorPositions = get(0, 'MonitorPositions');
monitorOne = monitorPositions(1, :); % Just use single monitor if you don't have second one
monitorTwo = monitorPositions(size(monitorPositions, 1), :); % Just use single monitor if you don't have second one


bhvLabels = {'investigate_1', 'investigate_2', 'investigate_3', ...
    'rear', 'dive_scrunch', 'paw_groom', 'face_groom_1', 'face_groom_2', ...
    'head_groom', 'contra_body_groom', 'ipsi_body groom', 'contra_itch', ...
    'ipsi_itch_1', 'contra_orient', 'ipsi_orient', 'locomotion'};



nDim = 4;


%%           ==========================         WHICH DATA DO YOU WANT TO ANALYZE?        =================================

natOrReach = 'Nat'; % 'Nat'  'Reach'
idAreaName = 'M56';

areas = {'M23', 'M56', 'DS', 'VS'};

switch natOrReach
    case 'Nat'
        % Naturalistic data
        getDataType = 'spikes';
        get_standard_data

    %   Load kinematics (in .1 s binSize right now)
kinBinSize = .1;
getDataType = 'kinematics';
get_standard_data

startFrame = 1 + opts.collectStart / kinBinSize;
endFrame = startFrame - 1 + (opts.collectFor / kinBinSize);
kinData = kinData(startFrame:endFrame,:);

[coeff, score, ~, ~, explained] = pca(zscore(kinData));
kinPCA = score(:, 1:6);

    case 'Reach'
        % Mark's reach data
        % dataR = load(fullfile(paths.dropPath, 'reach_data/Y4_100623_Spiketimes_idchan_BEH.mat'));
        dataR = load(fullfile(paths.dropPath, 'reach_data/Copy_of_Y4_100623_Spiketimes_idchan_BEH.mat'));

        [dataMat, idLabels, areaLabels, rmvNeurons] = neural_matrix_mark_data(dataR, opts);

        idM23 = find(strcmp(areaLabels, 'M23'));
        idM56 = find(strcmp(areaLabels, 'M56'));
        idDS = find(strcmp(areaLabels, 'DS'));
        idVS = find(strcmp(areaLabels, 'VS'));
end

idList = {idM23, idM56, idDS, idVS};
idArea = idList{strcmp(areas, idAreaName)};
dataMatMain = dataMat(:, idArea); % This should be your data


%%

y = zscore(dataMatMain);
z = kinPCA;
% z = bhvID;
nx = nDim * 2;
n1 = nDim;
i = 10;

idSys = PSID(y, z, nx, n1, i);

% Predict behavior using the learned model
[zPred, yPred, xPred] = PSIDPredict(idSys, y);  % xPred: first nDim columns are the behavior-related latents. The rest are neural non-behavior.

figure(5); clf; hold on;
scatter(z, zPred)
hL=plot([-6 6],[-6 6], 'k');

