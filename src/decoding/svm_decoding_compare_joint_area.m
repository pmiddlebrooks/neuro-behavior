%%  Conjoint SVM decoding: combine low-D projections from multiple brain areas
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Before running: set sessionType and sessionName via src/data_prep/choose_task_and_session.m
% (same convention as metastability/scripts/run_hmm_mazz.m). Extend path block / opts if you add pipelines.
%
% Mirrors svm_decoding_compare.m loading and projection steps, but:
%   - User selects multiple areas (areasToInclude).
%   - Each area is embedded separately (PCA, UMAP, PSID-Kin, ...) with dimension nDim.
%   - Outputs are concatenated: first nComponentsPerArea columns from each area.
%     Joint feature dimension = nComponentsPerArea * numel(areasToInclude).
%
% Extend combineStrategy later (e.g. CCA fusion); only 'concatenateFirstN' is supported now.


%% =============================================================================
% --------    PATHS — match metastability/scripts/run_hmm_mazz.m
% =============================================================================
scriptDir = fileparts(mfilename('fullpath'));
srcPath = fullfile(scriptDir, '..');
dataPrepPath = fullfile(srcPath, 'data_prep');
swPrepUtilsPath = fullfile(srcPath, 'sliding_window_prep', 'utils');
if exist(srcPath, 'dir')
    addpath(srcPath);
end
if exist(dataPrepPath, 'dir')
    addpath(dataPrepPath);
end
if exist(swPrepUtilsPath, 'dir')
    addpath(swPrepUtilsPath);
end

paths = get_paths;

% Session selection: define sessionType and sessionName before running this script by
% editing and running src/data_prep/choose_task_and_session.m (same workflow as
% run_hmm_mazz.m lines 102–126). sessionType ∈ {'spontaneous','reach',...}. Only those
% two are handled here.

if ~exist('sessionName', 'var') || isempty(sessionName)
    error(['Define sessionName in the workspace — run ', ...
           'src/data_prep/choose_task_and_session.m (appropriate block) before this script.']);
end
if ~exist('sessionType', 'var') || isempty(sessionType)
    error(['Define sessionType in the workspace — same source as sessionName ', ...
           '(e.g. ''spontaneous'' or ''reach'').']);
end

switch lower(strtrim(sessionType))
    case 'spontaneous'
        dataType = 'spontaneous';
    case 'reach'
        dataType = 'reach';
    otherwise
        error(['svm_decoding_compare_joint_area: unsupported sessionType ''%s''. ', ...
               'Only ''spontaneous'' and ''reach'' are implemented.'], sessionType);
end

% Per-area embedding dimension (must be >= nComponentsPerArea). Same role as nDim in
% svm_decoding_compare.m for fitting PCA/UMAP/PSID.
dimToTest = [8];
dimIdx = 1;
nDim = dimToTest(dimIdx);

% First nComponentsPerArea latent dimensions from EACH included area → horizontal concat
nComponentsPerArea = 4;
combineStrategy = 'concatenateFirstN'; % future: e.g. learned fusion across areas

assert(nComponentsPerArea <= nDim, ...
    'nComponentsPerArea (%d) must be <= per-area embedding nDim (%d).', nComponentsPerArea, nDim);

% Analysis type - which subset of data points to include for fitting
dataSubsetToTest = {'all'};
subsetIdx = 1;
dataSubset = dataSubsetToTest{subsetIdx};

frameSize = .05;

% SVM parameters
kernelFunction = 'polynomial'; % 'linear' or 'polynomial'

savePath = fullfile(paths.dropPath, 'decoding');
if ~exist(savePath, 'dir')
    mkdir(savePath);
end

% Persist full decoding results (.mat can be large); .mat and paired _summary.txt are
% both skipped when saveMatFile is false so you can iterate in memory only.
saveMatFile = false;

% Order of areas in areas cell: M23, M56, DS, VS
areas = {'M23', 'M56', 'DS', 'VS'};
% Which areas to merge (indices into areas / idList)
areasToInclude = [1, 2, 3];  % e.g. M56 and DS

methodsToRun = {'pca', 'umap', 'psidKin'};
methodsToRun = {'pca'};
fprintf('Running methods: %s\n', strjoin(methodsToRun, ', '));
fprintf('Joint areas: %s\n', strjoin(areas(areasToInclude), ', '));

cvType = 'holdout';  % 'holdout' or 'kfold'
holdoutRatio = 0.2;
nFolds = 4;

nShuffles = 10;
permuteStrategy = 'circular';  % 'label' or 'circular'

balanceStrategy = 'subsample';  % 'none' or 'subsample'

plotFullMap = 0; %#ok<NASGU>
plotModelData = 1; %#ok<NASGU>
plotComparisons = 1; %#ok<NASGU>
savePlotFlag = 1; %#ok<NASGU>

allFontSize = 12; %#ok<NASGU>


%% =============================================================================
%            DATA LOADING (sessionName / sessionType from choose_task_and_session.m)
% =============================================================================

areas = {'M23', 'M56', 'DS', 'VS'};

if strcmp(dataType, 'reach')
    reachCode = 2;
    [~, ~, reachExt] = fileparts(sessionName);
    reachFileLocal = sessionName;
    if isempty(reachExt)
        reachFileLocal = [sessionName, '.mat'];
    end
    reachDataFile = fullfile(paths.reachDataPath, reachFileLocal);

    dataR = load(reachDataFile);

    opts = neuro_behavior_options;
    opts.firingRateCheckTime = 5 * 60;
    opts.collectStart = 0;
    opts.collectEnd = round(min(dataR.R(end,1) + 5000, max(dataR.CSV(:,1)*1000)) / 1000);
    opts.minFiringRate = .1;
    opts.maxFiringRate = 70;
    opts.frameSize = frameSize;

    svmFirstBin = floor(dataR.R(1,1) / opts.frameSize / 1000);

    [dataMat, idLabels, areaLabels] = reach_neural_matrix(dataR, opts);
    idM23 = find(strcmp(areaLabels, 'M23'));
    idM56 = find(strcmp(areaLabels, 'M56'));
    idDS = find(strcmp(areaLabels, 'DS'));
    idVS = find(strcmp(areaLabels, 'VS'));
    idList = {idM23, idM56, idDS, idVS};

    bhvOpts = struct();
    bhvOpts.frameSize = frameSize;
    bhvOpts.collectStart = opts.collectStart;
    bhvOpts.collectEnd = opts.collectEnd;
    bhvID = define_reach_bhv_labels(reachDataFile, bhvOpts);

    jsX = zscore(dataR.NIDATA(7,2:end));
    jsY = zscore(dataR.NIDATA(8,2:end));
    kinData = [jsX(:), jsY(:)];

    if opts.frameSize > .001
        binsPerFrame = round(opts.frameSize * 1000);
        numFrames = floor(size(kinData,1) / binsPerFrame);
        kinDataDown = zeros(numFrames, size(kinData,2));
        for iFrames = 1:numFrames
            idxStart = (iFrames-1)*binsPerFrame + 1;
            idxEnd = iFrames*binsPerFrame;
            kinDataDown(iFrames,:) = mean(kinData(idxStart:idxEnd,:), 1, 'omitnan');
        end
        kinData = kinDataDown(1:size(dataMat, 1),:);
    end

    fprintf('Loaded reach (%s): %d neurons, %d time bins\n', sessionName, size(dataMat, 2), size(dataMat, 1));

elseif strcmp(dataType, 'spontaneous')
    % Spike loader + cortex tables (aligned with run_hmm_mazz.m)
    opts = neuro_behavior_options;
    opts.minActTime = 0.16;
    opts.minFiringRate = 0.2;
    opts.firingRateCheckTime = 5 * 60;
    opts.maxFiringRate = 100;
    % Collection window: defaults match metastability/scripts/run_hmm_mazz.m (extend for longer runs)
    opts.collectStart = 0;
    opts.collectEnd = 45 * 60;
    opts.removeSome = true;
    opts.frameSize = frameSize;

    fprintf('Loading spikes (spontaneous): %s\n', sessionName);
    spikeDataStruct = load_spike_times('spontaneous', paths, sessionName, opts);

    if isfield(spikeDataStruct, 'collectStart')
        opts.collectStart = spikeDataStruct.collectStart;
    end
    if isfield(spikeDataStruct, 'collectEnd')
        opts.collectEnd = spikeDataStruct.collectEnd;
    end

    spikeTimesCol = spikeDataStruct.spikeTimes(:);
    spikeClustersCol = spikeDataStruct.spikeClusters(:);
    neuronIDsOrdered = spikeDataStruct.neuronIDs(:);

    areaMappingLoc = containers.Map({'M23', 'M56', 'DS', 'VS'}, {1, 2, 3, 4});
    areaIdxPerNeuronLoc = zeros(numel(neuronIDsOrdered), 1);
    for nLoc = 1:numel(neuronIDsOrdered)
        labelLoc = spikeDataStruct.neuronAreas{nLoc};
        if isKey(areaMappingLoc, labelLoc)
            areaIdxPerNeuronLoc(nLoc) = areaMappingLoc(labelLoc);
        end
    end
    [~, locAssign] = ismember(spikeClustersCol, neuronIDsOrdered);
    areaIdxPerSpikeLoc = zeros(size(spikeClustersCol));
    validMaskLoc = locAssign > 0;
    areaIdxPerSpikeLoc(validMaskLoc) = areaIdxPerNeuronLoc(locAssign(validMaskLoc));
    spikeTab = [spikeTimesCol, spikeClustersCol, areaIdxPerSpikeLoc];

    timeRangeBins = [opts.collectStart, opts.collectEnd];
    dataMat = bin_spikes(spikeTimesCol, spikeClustersCol, neuronIDsOrdered', timeRangeBins, frameSize);

    idM23 = find(ismember(neuronIDsOrdered, unique(spikeTab(spikeTab(:, 3) == 1, 2))));
    idM56 = find(ismember(neuronIDsOrdered, unique(spikeTab(spikeTab(:, 3) == 2, 2))));
    idDS = find(ismember(neuronIDsOrdered, unique(spikeTab(spikeTab(:, 3) == 3, 2))));
    idVS = find(ismember(neuronIDsOrdered, unique(spikeTab(spikeTab(:, 3) == 4, 2))));
    idList = {idM23, idM56, idDS, idVS};

    sessionBaseNm = svm_joint_session_base_name(sessionName);
    opts.sessionName = sessionBaseNm;
    subDirS = sessionBaseNm(1:min(2, numel(sessionBaseNm)));
    pathWithSubfolderS = fullfile(paths.spontaneousDataPath, subDirS, sessionBaseNm);
    pathFlatS = fullfile(paths.spontaneousDataPath, sessionBaseNm);
    if isfolder(pathWithSubfolderS)
        opts.dataPath = fullfile(paths.spontaneousDataPath, subDirS);
    elseif isfolder(pathFlatS)
        opts.dataPath = paths.spontaneousDataPath;
    else
        opts.dataPath = fullfile(paths.spontaneousDataPath, subDirS);
    end

    dataBhv = load_data(opts, 'behavior');

    nFrameBhv = ceil(opts.collectEnd / opts.frameSize);
    dataBhv.StartFrame = 1 + round(dataBhv.StartTime / opts.frameSize);
    bhvID = int8(zeros(nFrameBhv, 1));
    if size(dataBhv, 1) >= 2
        for iB = 1:size(dataBhv, 1) - 1
            iIndB = dataBhv.StartFrame(iB) : dataBhv.StartFrame(iB+1) - 1;
            bhvID(iIndB) = dataBhv.ID(iB);
        end
        bhvID(iIndB(end) + 1 : end) = dataBhv.ID(end);
    elseif size(dataBhv, 1) == 1
        bhvID(:) = dataBhv.ID(1);
    end

    [dataBhv, bhvID] = curate_behavior_labels(dataBhv, opts);

    codes = unique(dataBhv.ID);
    behaviors = {};
    for iBh = 1:numel(codes)
        firstIdxBeh = find(dataBhv.ID == codes(iBh), 1);
        behaviors = [behaviors, dataBhv.Name{firstIdxBeh}]; %#ok<AGROW>
    end

    nBinsSp = size(dataMat, 1);
    if numel(bhvID) > nBinsSp
        bhvID = bhvID(1:nBinsSp);
    elseif numel(bhvID) < nBinsSp
        padVal = bhvID(end);
        bhvID = [bhvID; repmat(padVal, nBinsSp - numel(bhvID), 1)];
    end

    % Kinematics for PSID (same entry point as svm_decoding_compare / get_standard_data)
    kinData = [];
    if exist('animal', 'var') && ~isempty(animal)
        try
            getDataType = 'kinematics';
            get_standard_data
        catch ME
            % warning('Kinematics load failed (%s). Using zeros for kinData (check PSID results).', ME.message);
            warning('Kinematics load failed Using zeros for kinData (check PSID results).');
            kinData = zeros(nBinsSp, 2);
        end
    else
        warning(['workspace variable ''animal'' not set — ', ...
                 'get_standard_data kinematics skipped. kinData set to zeros; PSID-kin may be uninformative.']);
        kinData = zeros(nBinsSp, 2);
    end
    if isempty(kinData)
        kinData = zeros(nBinsSp, 2);
    end
    if size(kinData, 1) > nBinsSp
        kinData = kinData(1:nBinsSp, :);
    elseif size(kinData, 1) < nBinsSp && size(kinData, 1) > 0
        lastRow = kinData(end, :);
        kinData = [kinData; repmat(lastRow, nBinsSp - size(kinData, 1), 1)];
    end

    fprintf('Loaded spontaneous (%s): %d neurons, %d bins (%.2f s)\n', ...
        sessionName, size(dataMat, 2), nBinsSp, frameSize);

else
    error('Internal: unexpected dataType ''%s''.', dataType);
end

for k = 1:length(areasToInclude)
    areaIdx = areasToInclude(k);
    idSelect = idList{areaIdx};
    if length(idSelect) <= nDim
        error('Area %s: need more than nDim=%d neurons (have %d).', areas{areaIdx}, nDim, length(idSelect));
    end
end

if strcmp(dataType, 'reach')
    behaviors = {'pre-reach', 'reach', 'pre-reward', 'reward', 'post-reward', 'intertrial'};
    func = @sRGB_to_OKLab;
    cOpts.exc = [0,0,0];
    cOpts.Lmax = .8;
    colors = maxdistcolor(length(behaviors), func, cOpts);
    colors(end,:) = [.85 .8 .75];
    colorsAdjust = 0;
else
    colors = colors_for_behaviors(codes);
    colorsAdjust = 2;
    % behaviors: cell from get_standard_data (aligned with codes)
end


%% =============================================================================
% --------    LATENTS PER AREA
% =============================================================================
ttime = tic;

fprintf('\n=== Joint-area decoding ===\n');
fprintf('Combined feature dimension: %d (areas × %d components)\n', ...
    nComponentsPerArea * length(areasToInclude), nComponentsPerArea);

allResults = struct();
allResults.areas = areas;
allResults.areasToInclude = areasToInclude(:)';
allResults.combineStrategy = combineStrategy;
allResults.parameters = struct();
allResults.parameters.frameSize = opts.frameSize;
allResults.parameters.nShuffles = nShuffles;
allResults.parameters.kernelFunction = kernelFunction;
allResults.parameters.collectStart = opts.collectStart;
allResults.parameters.collectFor = opts.collectEnd;
if strcmp(dataType, 'spontaneous')
    allResults.parameters.minActTime = opts.minActTime;
end
allResults.parameters.dataSubset = dataSubset;
allResults.parameters.sessionType = sessionType;
allResults.parameters.sessionName = sessionName;
allResults.parameters.nDim = nDim;
allResults.parameters.nComponentsPerArea = nComponentsPerArea;
allResults.parameters.jointFeatureDim = nComponentsPerArea * length(areasToInclude);
allResults.parameters.analysisLayout = 'joint';
allResults.parameters.cvType = cvType;
if strcmp(cvType, 'holdout')
    allResults.parameters.holdoutRatio = holdoutRatio;
else
    allResults.parameters.nFolds = nFolds;
end

allResults.latents = cell(1, length(areas));
allResults.idSelect = cell(1, length(areas));

for k = 1:length(areasToInclude)
    areaIdx = areasToInclude(k);
    areaName = areas{areaIdx};
    fprintf('\n--- Latents for area: %s ---\n', areaName);

    idSelect = idList{areaIdx};
    allResults.idSelect{areaIdx} = idSelect;
    fprintf('Selected %d neurons for area %s\n', length(idSelect), areaName);

    latents = struct();

    if ismember('pca', methodsToRun)
        fprintf('Running PCA...\n');
        [~, score, ~, ~, explained] = pca(zscore(dataMat(:, idSelect)));
        latents.pca = score(:, 1:nDim);
        fprintf('PCA explained variance (first %d): %.2f%%\n', nDim, sum(explained(1:nDim)));
    end

    if ismember('umap', methodsToRun)
        fprintf('Running UMAP...\n');
        cd(fullfile(paths.homePath, '/toolboxes/umapFileExchange (4.4)/umap/'))
        switch dataType
            case 'reach'
                switch areaIdx
                    case 1, min_dist = 0.3; spread = 1.2; n_neighbors = 30;
                    case 2, min_dist = 0.3; spread = 1.3; n_neighbors = 40;
                    case 3, min_dist = 0.5; spread = 1.2; n_neighbors = 40;
                    case 4, min_dist = 0.3; spread = 1.2; n_neighbors = 40;
                    otherwise, min_dist = 0.3; spread = 1.2; n_neighbors = 30;
                end
            case 'spontaneous'
                switch areaIdx
                    case 1, min_dist = 0.1; spread = 1; n_neighbors = 15;
                    case 2, min_dist = 0.2; spread = 1.2; n_neighbors = 30;
                    case 3, min_dist = 0.2; spread = 1.2; n_neighbors = 30;
                    case 4, min_dist = 0.2; spread = 1.2; n_neighbors = 30;
                    otherwise, min_dist = 0.3; spread = 1.2; n_neighbors = 30;
                end
        end
        warning('off', 'MATLAB:unrecognizedPragma');
        [latents.umap, ~, ~, ~] = run_umap(zscore(dataMat(:, idSelect)), 'n_components', nDim, ...
            'randomize', true, 'verbose', 'none', 'min_dist', min_dist, ...
            'spread', spread, 'n_neighbors', n_neighbors, 'ask', false);
        warning('on', 'MATLAB:unrecognizedPragma');
        cd(fullfile(paths.homePath, 'neuro-behavior/src/decoding'));
    end

    if ismember('psidKin', methodsToRun) || ismember('psidKin_nonBhv', methodsToRun)
        fprintf('Running PSID with kinematics...\n');
        cd(fullfile(paths.homePath, 'neuro-behavior/src/decoding'));
        y = zscore(dataMat(:, idSelect));
        z = zscore(kinData);
        nx = nDim * 2;
        n1 = nDim;
        i = 10;
        idSys = PSID(y, z, nx, n1, i);
        [~, ~, xPred] = PSIDPredict(idSys, y);
        if ismember('psidKin', methodsToRun)
            latents.psidKin = xPred(:, 1:nDim);
        end
        if ismember('psidKin_nonBhv', methodsToRun)
            remainingDims = (nDim+1):size(xPred, 2);
            if ~isempty(remainingDims)
                latents.psidKin_nonBhv = xPred(:, remainingDims);
            else
                latents.psidKin_nonBhv = [];
            end
        end
    end

    if ismember('psidBhv', methodsToRun) || ismember('psidBhv_nonBhv', methodsToRun)
        fprintf('Running PSID with behavior labels...\n');
        uniqueBhv = unique(bhvID);
        uniqueBhv = uniqueBhv(uniqueBhv >= 0);
        nBehaviors = length(uniqueBhv);
        bhvOneHot = zeros(length(bhvID), nBehaviors);
        for b = 1:nBehaviors
            bhvOneHot(:, b) = (bhvID == uniqueBhv(b));
        end
        bhvMapping = struct();
        bhvMapping.uniqueBhv = uniqueBhv;
        bhvMapping.nBehaviors = nBehaviors;
        if strcmp(dataType, 'reach')
            bhvMapping.behaviorNames = behaviors(uniqueBhv);
        else
            bhvMapping.behaviorNames = behaviors(uniqueBhv + 2);
        end
        y = zscore(dataMat(:, idSelect));
        z = zscore(bhvOneHot);
        nx = nDim * 2;
        n1 = nDim;
        i = 10;
        idSys = PSID(y, z, nx, n1, i);
        [~, ~, xPred] = PSIDPredict(idSys, y);
        if ismember('psidBhv', methodsToRun)
            latents.psidBhv = xPred(:, 1:nDim);
            allResults.bhvMapping{areaIdx} = bhvMapping;
        end
        if ismember('psidBhv_nonBhv', methodsToRun)
            remainingDims = (nDim+1):size(xPred, 2);
            if ~isempty(remainingDims)
                latents.psidBhv_nonBhv = xPred(:, remainingDims);
            else
                latents.psidBhv_nonBhv = [];
            end
        end
    end

    if ismember('icg', methodsToRun)
        fprintf('Running ICG...\n');
        dataICG = dataMat(:, idSelect);
        [activityICG, ~] = ICG(dataICG');
        if areaIdx == 1
            latents.icg = zscore(activityICG{3}(1:3,:)');
        else
            if nDim == 4
                latents.icg = zscore(activityICG{4}(1:nDim, :)');
            elseif nDim >= 6
                warning('Changed ICG population')
                latents.icg = zscore(activityICG{3}(1:9, :)');
            end
        end
    end

    allResults.latents{areaIdx} = latents;
    fprintf('Stored latents for area %s.\n', areaName);
end


%% =============================================================================
% --------    BUILD JOINT LATENT MATRICES (concatenateFirstN)
% =============================================================================
jointLatents = struct();
methods = methodsToRun;
nMethods = length(methods);

switch combineStrategy
    case 'concatenateFirstN'
        for m = 1:nMethods
            methodName = methods{m};
            jointBlocks = [];
            for k = 1:length(areasToInclude)
                areaIdx = areasToInclude(k);
                L = allResults.latents{areaIdx}.(methodName);
                if isempty(L)
                    error('No latent data for method %s in area %s', methodName, areas{areaIdx});
                end
                nColsAvail = size(L, 2);
                if nColsAvail < nComponentsPerArea
                    error('Method %s area %s: need %d columns, have %d.', ...
                        methodName, areas{areaIdx}, nComponentsPerArea, nColsAvail);
                end
                jointBlocks = [jointBlocks, L(:, 1:nComponentsPerArea)]; %#ok<AGROW>
            end
            jointLatents.(methodName) = jointBlocks;
        end
    otherwise
        error('combineStrategy ''%s'' is not implemented. Use ''concatenateFirstN''.', combineStrategy);
end
allResults.jointLatents = jointLatents;


%% =============================================================================
% --------    SVM SAMPLE INDICES (same rule as svm_decoding_compare)
% =============================================================================
shiftSec = 0;
if shiftSec > 0 && strcmp(dataType, 'spontaneous')
    shiftFrame = ceil(shiftSec / opts.frameSize);
    bhvIDShifted = double(bhvID(1+shiftFrame:end));
    for k = 1:length(areasToInclude)
        areaIdx = areasToInclude(k);
        latents = allResults.latents{areaIdx};
        fieldNames = fieldnames(latents);
        for ii = 1:length(fieldNames)
            latents.(fieldNames{ii}) = latents.(fieldNames{ii})(1:end-shiftFrame, :);
        end
        allResults.latents{areaIdx} = latents;
    end
    for m = 1:nMethods
        methodName = methods{m};
        allResults.jointLatents.(methodName) = allResults.jointLatents.(methodName)(1:end-shiftFrame, :);
    end
else
    bhvIDShifted = double(bhvID);
end

switch dataType
    case 'spontaneous'
        preIdx = find(diff(bhvIDShifted) ~= 0);
        switch dataSubset
            case 'all'
                svmInd = 1:length(bhvIDShifted);
                svmID = bhvIDShifted;
            case 'trans'
                svmID = bhvIDShifted(preIdx + 1);
                svmInd = preIdx;
            case 'transPost'
                svmID = bhvIDShifted(preIdx + 1);
                svmInd = preIdx + 1;
            case 'within'
                svmInd = setdiff(1:length(bhvIDShifted), preIdx);
                svmID = bhvIDShifted(svmInd);
        end
        deleteInd = svmID == -1;
        svmID(deleteInd) = [];
        svmInd(deleteInd) = [];

    case 'reach'
        switch dataSubset
            case 'all'
                svmInd = svmFirstBin:length(bhvID);
                svmID = bhvID(svmFirstBin:end);
            case 'trans'
                reachMask = (bhvID == reachCode);
                reachStarts = find([reachMask(1); diff(reachMask) == 1]);
                reachStops = find([reachMask(1); diff(reachMask) == -1]);
                preSec = 1;
                postSec = 0;
                preFrames = round(preSec / opts.frameSize);
                postFrames = round(postSec / opts.frameSize);
                keepMask = false(length(bhvID), 1);
                for r = 1:length(reachStarts)
                    winStart = reachStarts(r) - preFrames;
                    winEnd = max(min(length(bhvID), reachStarts(r) + postFrames), reachStops(r));
                    keepMask(winStart:winEnd) = true;
                end
                svmInd = find(keepMask);
                svmID = bhvID(svmInd);
            case 'no_intertrial'
                svmInd = svmFirstBin:length(bhvID);
                svmID = bhvID(svmFirstBin:end);
                svmInd(svmID == 6) = [];
                svmID(svmID == 6) = [];
        end
end

allResults.svmID = svmID;
allResults.svmInd = svmInd;

switch dataSubset
    case 'all', dataSubsetLabel = 'all';
    case 'trans'
        if strcmp(dataType, 'spontaneous')
            dataSubsetLabel = 'transitions: Pre';
        else
            dataSubsetLabel = 'peri-reach (-2s to +2s)';
        end
    case 'transPost', dataSubsetLabel = 'transitions: Post';
    case 'within', dataSubsetLabel = 'within-behavior';
    case 'no_intertrial', dataSubsetLabel = 'all except intertrial';
end

if strcmp(balanceStrategy, 'subsample')
    counts = arrayfun(@(c) sum(svmID==c), unique(svmID));
    maxSubsampleSize = round(mean(counts));
    fprintf('Sub-sampling training data to maximum %d per category\n', maxSubsampleSize);
end


%% =============================================================================
% --------    JOINT SVM
% =============================================================================
jointLabelStr = strjoin(areas(areasToInclude), '+');
fprintf('\n=======================   Fitting joint SVM: %s   =======================\n', jointLabelStr);

bhv2ModelCodes = unique(svmID);
if strcmp(dataType, 'reach')
    bhv2ModelNames = behaviors(bhv2ModelCodes);
    bhv2ModelColors = colors(bhv2ModelCodes, :);
else
    bhv2ModelNames = behaviors(bhv2ModelCodes+colorsAdjust);
    bhv2ModelColors = colors;
end

nWorkers = min(4, nMethods);
% fprintf('Setting up parallel pool with %d workers...\n', nWorkers);
% if isempty(gcp('nocreate'))
%     parpool('local', nWorkers);
% else
%     fprintf('Parallel pool already exists with %d workers\n', gcp('nocreate').NumWorkers);
% end

accuracy = zeros(nMethods, 1);
accuracyPermuted = zeros(nMethods, nShuffles);
svmModels = cell(1, nMethods);
allPredictions = cell(1, nMethods);
allPredictionIndices = cell(1, nMethods);

if strcmp(cvType, 'holdout')
    cv = cvpartition(svmID, 'HoldOut', holdoutRatio);
    nCVFolds = 1;
elseif strcmp(cvType, 'kfold')
    cv = cvpartition(svmID, 'KFold', nFolds);
    nCVFolds = nFolds;
else
    error('Unknown cvType: %s. Must be ''holdout'' or ''kfold''', cvType);
end

fprintf('Using %s cross-validation (%d fold(s))\n', cvType, nCVFolds);

if nMethods > 0
    tempAccuracy = zeros(nMethods, 1);
    tempAccuracyPermuted = cell(nMethods, 1);
    tempSvmModels = cell(nMethods, 1);
    tempAllPredictions = cell(nMethods, 1);
    tempAllPredictionIndices = cell(nMethods, 1);

    ticSvmBlock = tic;
    % parfor m = 1:nMethods
    for m = 1:nMethods
        methodName = methods{m};
        latentData = allResults.jointLatents.(methodName);
        if isempty(latentData)
            fprintf('Warning: No joint latent data for method %s\n', methodName);
            tempAccuracy(m) = NaN;
            tempAccuracyPermuted{m} = nan(nShuffles, 1);
            tempSvmModels{m} = [];
            tempAllPredictions{m} = [];
            tempAllPredictionIndices{m} = [];
        else
            svmProj = latentData(svmInd, :);
            fprintf('\n--- %s (joint) ---\n', upper(methodName));
            tt = templateSVM('Standardize', true, 'KernelFunction', kernelFunction);
            svmModelFull = fitcecoc(svmProj, svmID, 'Learners', tt);
            tempSvmModels{m} = svmModelFull;
            tempAllPredictions{m} = predict(svmModelFull, svmProj);
            tempAllPredictionIndices{m} = svmInd;

            foldAccuracies = zeros(nCVFolds, 1);
            for fold = 1:nCVFolds
                if strcmp(cvType, 'holdout')
                    trainMask = training(cv);
                    testMask = test(cv);
                else
                    trainMask = training(cv, fold);
                    testMask = test(cv, fold);
                end
                trainData = svmProj(trainMask, :);
                testData = svmProj(testMask, :);
                trainLabels = svmID(trainMask);
                testLabels = svmID(testMask);
                if strcmp(balanceStrategy, 'subsample')
                    balIdx = balance_subsample_indices(trainLabels, maxSubsampleSize);
                    trainDataCV = trainData(balIdx, :);
                    trainLabelsCV = trainLabels(balIdx);
                else
                    trainDataCV = trainData;
                    trainLabelsCV = trainLabels;
                end
                svmModelCV = fitcecoc(trainDataCV, trainLabelsCV, 'Learners', tt);
                predictedLabels = predict(svmModelCV, testData);
                foldAccuracies(fold) = sum(predictedLabels == testLabels) / length(testLabels);
            end
            tempAccuracy(m) = mean(foldAccuracies);

            fprintf('Running %d permutation tests (holdout CV)...\n', nShuffles);
            methodAccuracyPermuted = zeros(nShuffles, 1);
            for s = 1:nShuffles
                rng('shuffle');
                permCv = cvpartition(svmID, 'HoldOut', holdoutRatio);
                trainMask = training(permCv);
                testMask = test(permCv);
                trainData = svmProj(trainMask, :);
                testData = svmProj(testMask, :);
                trainLabels = svmID(trainMask);
                testLabels = svmID(testMask);
                switch permuteStrategy
                    case 'label'
                        permIdx = randperm(length(trainLabels));
                        shuffledLabels = trainLabels(permIdx);
                        if strcmp(balanceStrategy, 'subsample')
                            balIdxPerm = balance_subsample_indices(shuffledLabels, maxSubsampleSize);
                            trainDataPerm = trainData(balIdxPerm, :);
                            shuffledLabelsPerm = shuffledLabels(balIdxPerm);
                        else
                            trainDataPerm = trainData;
                            shuffledLabelsPerm = shuffledLabels;
                        end
                        svmModelPermuted = fitcecoc(trainDataPerm, shuffledLabelsPerm, 'Learners', tt);
                        predictedLabelsPermuted = predict(svmModelPermuted, testData);
                        methodAccuracyPermuted(s) = sum(predictedLabelsPermuted == testLabels) / length(testLabels);

                    case 'circular'
                        shuffledProj = svmProj;
                        nSamplesLocal = size(shuffledProj, 1);
                        nFeaturesLocal = size(shuffledProj, 2);
                        for c = 1:nFeaturesLocal
                            shiftC = randi([1, nSamplesLocal]);
                            shuffledProj(:, c) = circshift(shuffledProj(:, c), shiftC);
                        end
                        shuffledTrainData = shuffledProj(trainMask, :);
                        shuffledTestData = shuffledProj(testMask, :);
                        if strcmp(balanceStrategy, 'subsample')
                            if strcmp(dataType, 'reach')
                                balIdxPerm = balance_subsample_indices(trainLabels);
                            else
                                balIdxPerm = balance_subsample_indices(trainLabels, maxSubsampleSize);
                            end
                            shuffledTrainDataPerm = shuffledTrainData(balIdxPerm, :);
                            trainLabelsPerm2 = trainLabels(balIdxPerm);
                        else
                            shuffledTrainDataPerm = shuffledTrainData;
                            trainLabelsPerm2 = trainLabels;
                        end
                        svmModelPermuted = fitcecoc(shuffledTrainDataPerm, trainLabelsPerm2, 'Learners', tt);
                        predictedLabelsPermuted = predict(svmModelPermuted, shuffledTestData);
                        methodAccuracyPermuted(s) = sum(predictedLabelsPermuted == testLabels) / length(testLabels);
                    otherwise
                        error('Unknown permuteStrategy: %s', permuteStrategy);
                end
            end
            tempAccuracyPermuted{m} = methodAccuracyPermuted;
        end
    end

    fprintf('Joint SVM block finished in %.2f s\n', toc(ticSvmBlock));

    for m = 1:nMethods
        accuracy(m) = tempAccuracy(m);
        accuracyPermuted(m, :) = tempAccuracyPermuted{m}';
        svmModels{m} = tempSvmModels{m};
        allPredictions{m} = tempAllPredictions{m};
        allPredictionIndices{m} = tempAllPredictionIndices{m};
    end
end

% delete(gcp('nocreate'));

allResults.methods = methods;
allResults.accuracy = accuracy;
allResults.accuracyPermuted = accuracyPermuted;
allResults.bhv2ModelCodes = bhv2ModelCodes;
allResults.bhv2ModelNames = bhv2ModelNames;
allResults.bhv2ModelColors = bhv2ModelColors;
allResults.svmModels = svmModels;
allResults.allPredictions = allPredictions;
allResults.allPredictionIndices = allPredictionIndices;

slack_code_done


%% =============================================================================
% --------    SAVE RESULTS
% =============================================================================
areasSlug = strjoin(areas(areasToInclude), '_');
filename = sprintf('svm_%s_joint_area_%s_subset_%s_embed%d_nComp%d_nAreas%d_bin%.2f_nShuffles%d.mat', ...
    kernelFunction, areasSlug, dataSubset, nDim, nComponentsPerArea, length(areasToInclude), ...
    opts.frameSize, nShuffles);

fullFilePath = fullfile(savePath, filename);
if saveMatFile
    save(fullFilePath, 'allResults', '-v7.3');
    fprintf('Saved: %s\n', fullFilePath);

    summaryFilename = strrep(filename, '.mat', '_summary.txt');
    summaryPath = fullfile(savePath, summaryFilename);

    fid = fopen(summaryPath, 'w');
    fprintf(fid, 'Joint-area SVM decoding\n');
    fprintf(fid, 'Areas: %s\n', jointLabelStr);
    fprintf(fid, 'Combine: concatenate first %d components per area\n', nComponentsPerArea);
    fprintf(fid, 'Per-area embedding nDim: %d | Joint dims: %d\n', nDim, allResults.parameters.jointFeatureDim);
    fprintf(fid, 'Data subset: %s (%s)\n', dataSubset, dataSubsetLabel);
    fprintf(fid, 'Frame size: %.2f | Shuffles: %d | Kernel: %s\n', opts.frameSize, nShuffles, kernelFunction);
    fprintf(fid, '\nMethod\tReal\tPermuted mean +/- std\tDelta\n');
    for m = 1:nMethods
        fprintf(fid, '%s\t%.4f\t%.4f +/- %.4f\t%.4f\n', ...
            methods{m}, accuracy(m), mean(accuracyPermuted(m,:)), std(accuracyPermuted(m,:)), ...
            accuracy(m) - mean(accuracyPermuted(m,:)));
    end
    fclose(fid);
    fprintf('Summary: %s\n', summaryPath);
else
    fprintf('saveMatFile=false — skipping .mat and summary export.\n');
end
fprintf('\nTotal wall time (joint script): %.2f min\n', toc(ttime)/60);


%% =============================================================================
% --------    HELPERS
% =============================================================================

function baseNm = svm_joint_session_base_name(sessionNameIn)
% SVM_JOINT_SESSION_BASE_NAME  basename for spontaneous paths (handles subfolders).
[~, baseNm, ~] = fileparts(sessionNameIn);
if isempty(baseNm)
    baseNm = sessionNameIn;
end
end


function idx = balance_subsample_indices(labels, maxSubsampleSize)
% BALANCE_SUBSAMPLE_INDICES  Subsample class indices for balanced SVM training.
%
% Goal: return row indices into labels so each class has at most maxSubsampleSize
%       samples (or, if maxSubsampleSize omitted, down to the global minority count).
%
% Inputs:
%   labels              - training label vector (any type comparable with ==)
%   maxSubsampleSize    - optional cap per class; [] means use min class count
%
% Output:
%   idx                 - column vector of indices (randomized order)

if nargin < 2
    maxSubsampleSize = [];
end

classes = unique(labels);
counts = arrayfun(@(c) sum(labels==c), classes);

idx = [];
for k = 1:length(classes)
    c = classes(k);
    inds = find(labels==c);

    if isempty(maxSubsampleSize)
        target = min(counts);
    else
        target = min(counts(k), maxSubsampleSize);
    end

    if length(inds) > target
        sel = randperm(length(inds), target);
        inds = inds(sel);
    end
    idx = [idx; inds(:)]; %#ok<AGROW>
end
idx = idx(randperm(length(idx)));
end
