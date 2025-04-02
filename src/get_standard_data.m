% Loads multiple variables in work space, including
% dataBhv: behavioral data
% dataMat: 2-D neural activiy matrix (n time bins X p neurons)
% eventMat, eventMatZ: cell array of 3-D matrices, formed from dataMat
%   cell array: (Behaviors X 1)
%   matrices in cells: peri-onset neural data stacked across bouts (n time bins X p neurons X b bouts)

% Clear variables to free up memory
clear dataMat dataBhv eventMat eventMatZ 

%% get desired file paths
paths = get_paths;

animal = 'ag25290';
sessionBhv = '112321_1';
sessionNrn = '112321';
if strcmp(sessionBhv, '112321_1')
    sessionSave = '112321';
end

%%

% Need to get behavior data if you call spikes, too (for making the
% neural_matrix)
if strcmp(getDataType, 'all') || strcmp(getDataType, 'behavior') || strcmp(getDataType, 'spikes')
    % figurePath = strcat(paths.figurePath, animal, '/', sessionSave, '/figures/', ['start ' num2str(opts.collectStart), ' for ', num2str(opts.collectFor)]);
    % if ~exist(figurePath, 'dir')
    %     mkdir(figurePath);
    % end
    % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %                   Get behavior data
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    bhvDataPath = strcat(paths.bhvDataPath, 'animal_',animal,'/');
    bhvFileName = ['behavior_labels_', animal, '_', sessionBhv, '.csv'];


    opts.dataPath = bhvDataPath;
    opts.fileName = bhvFileName;

    dataBhv = load_data(opts, 'behavior');


    codes = unique(dataBhv.ID);
    % behaviors = unique(dataBhv.Name);
    behaviors = {};
    for iBhv = 1 : length(codes)
        firstIdx = find(dataBhv.ID == codes(iBhv), 1);
        behaviors = [behaviors, dataBhv.Name{firstIdx}];
        % fprintf('behavior %d:\t code:%d\t name: %s\n', i, codes(i), dataBhvAlex.Behavior{firstIdx})
    end

    opts.behaviors = behaviors;
    opts.bhvCodes = codes;
    opts.validCodes = codes(codes ~= -1);

    rmvBhv = zeros(1, length(behaviors));
    for i = 1 : length(behaviors)
        if (sum(dataBhv.ID == codes(i) & dataBhv.Valid) < 20) || codes(i) == -1
            rmvBhv(i) = 1;
        end
    end

    analyzeBhv = behaviors(~rmvBhv);
    analyzeCodes = codes(~rmvBhv);



    nFrame = ceil(opts.collectFor / opts.frameSize);

    dataBhv.StartFrame = 1 + round(dataBhv.StartTime / opts.frameSize);
    % dataBhv.DurFrame = [diff(dataBhv.StartFrame); nFrame - sum(dataBhv.StartFrame)];
    % dataBhv.DurFrame = floor(dataBhv.Dur ./ opts.frameSize);




    % Create bhvID, a vector of ID labels, one element per frame to match the
    % neural matrix

    % Use StartFrame and DurFrame method:
    bhvID = int8(zeros(nFrame, 1));
    for i = 1 : size(dataBhv, 1) - 1
        % iInd = dataBhv.StartFrame(i) : dataBhv.StartFrame(i) + dataBhv.DurFrame(i) - 1;
        iInd = dataBhv.StartFrame(i) : dataBhv.StartFrame(i+1) - 1;
        bhvID(iInd) = dataBhv.ID(i);
    end
    bhvID(iInd(end) + 1 : end) = dataBhv.ID(end);


    % Use majority time in frame method:

    % % 1. Create a vector of ID labels, one element per behavioral frame
    % % rate
    % timeWindows = 0 : 1/opts.fsBhv : dataBhv.StartTime(end) + dataBhv.Dur(end);
    % idFull = int8(zeros(length(timeWindows), 1));
    % for i = 1 : length(timeWindows) - 1
    %     % Find behavior active during the current time window
    %     idFull(i) = dataBhv.ID(dataBhv.StartTime <= timeWindows(i) & dataBhv.StartTime + dataBhv.Dur > timeWindows(i));
    % end
    % idFull(end) = dataBhv.ID(end);
    %
    % % 2. Find majority behavior in each opts.frameSize window
    % for i = 1 : size(bhvID, 1)
    %     % if i == 100
    %     %     disp('here')
    %     % end
    %     iStartTime = (i-1) * opts.frameSize;
    %     iStopTime = i * opts.frameSize;
    %     iStartFrame = round(1 + iStartTime * opts.fsBhv);
    %     iStopFrame = round(iStopTime * opts.fsBhv);
    %     iID = idFull(iStartFrame : iStopFrame);
    %     bhvID(i) = mode(iID);
    % end




end


%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if strcmp(getDataType, 'all') || strcmp(getDataType, 'kinematics')
    bhvDataPath = strcat(paths.bhvDataPath, 'animal_',animal,'/');
    kinFileName = 'AdenKinematicsAligned.csv';

    % Define the path to your CSV file
csvFilePath = [bhvDataPath, kinFileName];
kinData = readmatrix(csvFilePath);

% kinFileName = '2021-11-23_13-19-58DLC_resnet50_bottomup_clearSep21shuffle1_700000_kinematics.npy';
% csvFilePath = [bhvDataPath, kinFileName];
% kinData = readNPY(csvFilePath);

% Here if you want to read the original tracking data instead
    % kinFileName = '2021-11-23_13-19-58DLC_resnet50_bottomup_clearSep21shuffle1_700000.csv';
    % kinFileName = 'TrackingAligned.csv';
% csvFilePath = [bhvDataPath, kinFileName];

% % Read the first 3 rows to get headers using readmatrix, assuming the CSV is not too large
% % This step is primarily for obtaining the headers
% kinHeader = readmatrix(csvFilePath, 'OutputType', 'string', 'Range', '1:4');
% 
% % Concatenate the strings from the 2nd and 3rd row for each column to form column names
% % concatenatedHeader = kinHeader(2,:) + "_" + kinHeader(3,:);
% concatenatedHeader = kinHeader(3,:) + "_" + kinHeader(4,:);
% 
% % Now, prepare to read the entire file with custom headers
% % Initialize detectImportOptions again to specify the DataLines and VariableNames
% optsK = detectImportOptions(csvFilePath, 'NumHeaderLines', 3); % Skip the first 3 header lines
% optsK.VariableNames = matlab.lang.makeValidName(concatenatedHeader); % Make valid MATLAB variable names
% 
% % Assuming data starts from the 4th row, adjust VariableTypes if your data differs
% for i = 1:length(optsK.VariableTypes)
%     optsK = setvartype(optsK, repmat({'double'}, 1, length(optsK.VariableTypes))); % Assuming all data are type double
% end

% Read the CSV data with the new settings
% kinData = readtable(csvFilePath, optsK);

% disp(head(kinData));

end


%%  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%           Get LFP Data
if strcmp(getDataType, 'all') || strcmp(getDataType, 'lfp')
    % m23 = [0 500];
    % m56 = [501 1240];
    % cc = [1241 1540];
    % ds = [1541 2700];
    % vs = [2701 3840];

% if strcmp(getDataType, 'all') || strcmp(getDataType, 'lfp')
    nrnDataPath = strcat(paths.nrnDataPath, 'animal_',animal,'/', sessionNrn, '/');
    nrnDataPath = [nrnDataPath, 'recording1/'];
    opts.dataPath = nrnDataPath;

data = load_data(opts, 'lfp');
data = fliplr(data); % flip data so top row is brain surface
%%
% frequency_bands = [
%     8 13;  % Alpha band
%     13 30; % Beta band
%     30 80  % Gamma band
%     ];

channelSpacing = 100;
channelDepth = 1 : channelSpacing : channelSpacing * size(data, 2);
% lfpPerArea = data(:, [3 10 21 34]);
% lfpPerArea = data(:, [mean(data(:,[3 5]), 2) mean(data(:,[9 11]), 2) mean(data(:,[19 23]), 2) mean(data(:,[30 34]), 2)]);
lfpPerArea = [mean(data(:,[3 5]), 2) mean(data(:,[9 11]), 2) mean(data(:,[19 23]), 2) mean(data(:,[30 34]), 2)];
clear data

% %%  Get wavelet time-frequency analysis
% start2Min = 1 + 0 * 60 * opts.fsLfp;
% windowSize = 30 * opts.fsLfp; %30 sec window
% window = 1 : windowSize;
% lfpM56Window = lfpM56(window);
% [wt, f, coi] = cwt(lfpM56Window, opts.fsLfp);
% % cwt(lfpM56Window, opts.fsLfp)
% %%
% % cwt(lfpM56Window, opts.fsLfp)
% 
% wtGamma = wt(f >= frequency_bands(1,1) & f <= frequency_bands(1,2),:);
% 
% plot(mean(abs(wtGamma).^2, 1));
% 
% % end
% 
% 
end






%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                Get Neural matrix
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if strcmp(getDataType, 'all') || strcmp(getDataType, 'spikes')

    nrnDataPath = strcat(paths.nrnDataPath, 'animal_',animal,'/', sessionNrn, '/');
    nrnDataPath = [nrnDataPath, 'recording1/'];
    opts.dataPath = nrnDataPath;

    data = load_data(opts, 'spikes');
    data.bhvDur = dataBhv.Dur;
    % clusterInfo = data.ci;
    % spikeTimes = data.spikeTimes;
    % spikeClusters = data.spikeClusters;


    % Find the neuron clusters (ids) in each brain region

    allGood = strcmp(data.ci.group, 'good') & strcmp(data.ci.KSLabel, 'good');
    allGood = (strcmp(data.ci.group, 'good') & strcmp(data.ci.KSLabel, 'good')) | (strcmp(data.ci.group, 'mua') & strcmp(data.ci.KSLabel, 'mua'));

    goodM23 = allGood & strcmp(data.ci.area, 'M23');
    goodM56= allGood & strcmp(data.ci.area, 'M56');
    goodCC = allGood & strcmp(data.ci.area, 'CC');
    goodDS = allGood & strcmp(data.ci.area, 'DS');
    goodVS = allGood & strcmp(data.ci.area, 'VS');


    % which neurons to use in the neural matrix
    opts.useNeurons = find(goodM23 | goodM56 | goodDS | goodVS | goodCC);
    opts.useNeurons = find(goodM23 | goodM56 | goodDS | goodVS);

    tic
    [dataMat, idLabels, areaLabels, removedNeurons] = neural_matrix(data, opts);
    toc

    % Normalize and zero-center the neural data matrix
    % dataMatZ = zscore(dataMat, 0, 1);


    idM23 = find(strcmp(areaLabels, 'M23'));
    idM56 = find(strcmp(areaLabels, 'M56'));
    idDS = find(strcmp(areaLabels, 'DS'));
    idVS = find(strcmp(areaLabels, 'VS'));
    idAll{1} = idM23; idAll{2} = idM56; idAll{3} = idDS; idAll{4} = idVS;
    areaAll{1} = 'M23'; areaAll{2} = 'M56'; areaAll{3} = 'DS'; areaAll{4} = 'VS';

    fprintf('%d M23\n%d M56\n%d DS\n%d VS\n', length(idM23), length(idM56), length(idDS), length(idVS))




    return
    %%         Standard PSTHS to use for analyses (b/c they have the same trials, etc)
    % Create 3-D psth data matrix of stacked peri-event start time windows (time X neuron X trial)
    % Make one of spike counts and one of zscored spike counts

    % z-Window is a big window used to calculate z-score. Then store a smaller
    % portion of that window in fullWindow.
    zTime = -3 : opts.frameSize : 2;  % zscore on a 5sec window peri-onset
    zWindow = round(zTime(1:end-1) / opts.frameSize);
    zStartInd = find(zTime == 0);
    fullTime = -1 : opts.frameSize : 1; % seconds around onset
    fullWindow = round(fullTime(1:end-1) / opts.frameSize); % frames around onset w.r.t. zWindow (remove last frame)
    fullStartInd = find(fullWindow == 0);
    periTime = -.1 : opts.frameSize : .1; % seconds around onset
    periWindow = periTime(1:end-1) / opts.frameSize; % frames around onset w.r.t. zWindow (remove last frame)

    eventMat = cell(length(analyzeCodes), 1);
    eventMatZ = cell(length(analyzeCodes), 1);
    % periMatZ = cell(length(analyzeCodes), 1);
    for iBhv = 1 : length(analyzeCodes)

        bhvStartFrames = 1 + floor(dataBhv.StartTime(dataBhv.ID == analyzeCodes(iBhv) & dataBhv.Valid) ./ opts.frameSize);
        bhvStartFrames(bhvStartFrames < -zWindow(1) + 1) = [];
        bhvStartFrames(bhvStartFrames > size(dataMat, 1) - zWindow(end)) = [];

        nTrial = length(bhvStartFrames);

        iDataMat = zeros(length(zWindow), size(dataMat, 2), nTrial); % peri-event time X neurons X nTrial
        for j = 1 : nTrial
            iDataMat(:,:,j) = dataMat(bhvStartFrames(j) + zWindow ,:);
            % iZMat(:,:,j) = zscore(dataMat(bhvStartFrames(j) + zWindow ,:), 1);
        end
        meanPsth = mean(iDataMat, 3);
        meanWindow = mean(meanPsth, 1);
        stdWindow = std(meanPsth, 1);

        iDataMatZ = (iDataMat - meanWindow) ./ stdWindow;

        eventMat{iBhv} = iDataMat(zStartInd + fullWindow, :, :);
        eventMatZ{iBhv} = iDataMatZ(zStartInd + fullWindow, :, :);
        % periMatZ{iBhv} = permute(mean(iDataMatZ(zStartInd + periWindow, :, :), 1), [3 2 1]);

    end
end


% clear big variables
clear iDataMat iDataMatZ data...
    goodM23 goodM56 goodDS goodVS
