% This script uses a q learning model to compare model parameters across
% conflict, volatility, stimulation (d1 and d2), across sessions withiin
% subjects and across subjects.

%% Run constants.m to get codes, event markers, subject info


addpath(fileparts(matlab.desktop.editor.getActiveFilename));
constants
paths = get_paths;

% Column indices in sessInfo matrix
subIdx = 1;
confIdx = 2;
volIdx = 3;
stimIdx = 4;
alphaRIdx = 5;
alphaUIdx = 6;
betaIdx = 7;
biasIdx = 8;
stickyIdx = 9;
wslsIdx = 10;



%% Run Q model for collapsing across sessions and for session-wise fits

% === PARAMETER CONFIGURATION ===
% Set which parameters to include in the model (1 = include, 0 = exclude)
% Make parameter configuration global for fit_Qvalue function
global paramConfig activeParams

paramConfig.alphaR = 1;    % Learning rate for rewarded trials
paramConfig.alphaU = 1;    % Learning rate for unrewarded trials  
paramConfig.beta = 1;      % Inverse temperature (choice sensitivity)
paramConfig.bias = 1;      % Choice bias
paramConfig.sticky = 1;    % Stickiness (perseveration)
paramConfig.wsls = 1;      % Win-stay, lose-shift

% Create parameter info structure
paramNames = fieldnames(paramConfig);
activeParams = paramNames([paramConfig.alphaR, paramConfig.alphaU, paramConfig.beta, paramConfig.bias, paramConfig.sticky, paramConfig.wsls] == 1);
nActiveParams = length(activeParams);


fprintf('Active parameters: %s\n', strjoin(activeParams, ', '));

% === INITIAL PARAMETER GUESSES AND BOUNDS ===
% Default values for each parameter
defaultValues = struct();
defaultValues.alphaR = 0.2;
defaultValues.alphaU = 0.2;
defaultValues.beta = 2;
defaultValues.bias = 0;
defaultValues.sticky = 0;
defaultValues.wsls = 0;

% Bounds for each parameter
lowerBounds = struct();
lowerBounds.alphaR = 0;
lowerBounds.alphaU = 0;
lowerBounds.beta = 0;
lowerBounds.bias = -10;
lowerBounds.sticky = -10;
lowerBounds.wsls = -10;

upperBounds = struct();
upperBounds.alphaR = 1;
upperBounds.alphaU = 1;
upperBounds.beta = 10;
upperBounds.bias = 10;
upperBounds.sticky = 10;
upperBounds.wsls = 10;

% Build parameter vectors for active parameters only
initParams = [];
lb = [];
ub = [];

for i = 1:length(activeParams)
    param = activeParams{i};
    initParams = [initParams, defaultValues.(param)];
    lb = [lb, lowerBounds.(param)];
    ub = [ub, upperBounds.(param)];
end

nStarts = 50;

% === OPTIMIZATION OPTIONS ===
optsMod = optimoptions('fmincon', 'Display', 'off', 'Algorithm', 'sqp');


% Constants, initialize variables
cutoffTrials = 500;
nTrialsRemove = 7;  % Number of trials to remove from the beginning of each block
plotFlag = 0;

sessInfoCollapse = [];
sessInfo = [];
maxRewards = 120;
cutoffFlag = 0;

confRange = 1:3;
volRange = 1:2;
contStimRange = 1:2;
plotModelDiagnostics = 0;

% subjectList = [d1Subj d2Subj daSubj];
subjectList = [d1Subj d2Subj];
d1Idx = find(ismember(subjectList, d1Subj));
d2Idx = find(ismember(subjectList, d2Subj));
% subjectList = 9;
% subjectList = d2Subj;

% Initialize parameter storage arrays
paramStorage = struct();
for i = 1:length(activeParams)
    param = activeParams{i};
    paramStorage.(param) = cell(length(subjectList), length(confRange), length(volRange), length(contStimRange), 2);
end

% Initialize mean parameter arrays
paramMeans = struct();
for i = 1:length(activeParams)
    param = activeParams{i};
    paramMeans.(param) = nan(length(subjectList), length(confRange), length(volRange), length(contStimRange));
end




for sub = 1 : length(subjectList)
    for conf = confRange
        for vol = volRange
            for contStim = contStimRange

                fprintf('Sub %d Conf %d Vol %d Stim %d\n', subjectList(sub), conf, vol, contStim-1)

                filePath = get_ymaze_data(conf, vol, contStim);
                data = load(filePath);

                nSess = size(data.overlap, 2);

                % get all sessions for that subject
                sessIdx = find(cell2mat(data.overlap(3,:)) == subjectList(sub));
                if isempty(sessIdx)
                    continue
                end




                % ==============================================
                % Collapse sessions for each subject per condition
                iEventMat = [];

                for iSess = sessIdx(1):sessIdx(end)
                    iData = data.overlap{1,iSess};

                    % Build a matrix of the rows of turn indications
                    emIdx = [];
                    for row = 2:size(iData, 1)
                        if iData(row, 2) ~= iData(row-1,2) && iData(row-1,2) == 0 %when a turn is detected...
                            emIdx = [emIdx; row];
                        end %ending "if" statement looking for a turn detection
                    end %ending "for" loop going through data

                    if cutoffFlag
                        iRew = ismember(iData(emIdx,2), [crl irl crr irr]);
                        cutoffTrial = find(cumsum(iRew) == maxRewards, 1);

                        % iEventMat = iData(emIdx,:);
                        iEventMat = [iEventMat; iData(emIdx(1:cutoffTrial),:)];
                    else
                        % iEventMat = iData(emIdx,:);
                        iEventMat = [iEventMat; iData(emIdx,:)];
                    end


                end
                [choices, rewards] = deal(zeros(size(iEventMat, 1), 1));
                choices(~ismember(iEventMat(:,2), [crl cul irl, iul])) = 1; % right turns are 1
                choices(ismember(iEventMat(:,2), [crl cul irl, iul])) = 2; % left turns are 2
                rewards(ismember(iEventMat(:,2), [cul iul cur iur])) = 0; % unrewarded are 0
                rewards(ismember(iEventMat(:,2), [crl irl crr irr])) = 1; % rewarded are 1

                % === OBJECTIVE FUNCTION ===
                negLLfun = @(params) -sum(fit_Qvalue(choices, rewards, params));  % minimize -log likelihood

                % === RUN fmincon ===
                bestNegLL = Inf;

                for i = 1:nStarts
                    % Generate random initial parameters for active parameters only
                    initParams = [];
                    for j = 1:length(activeParams)
                        param = activeParams{j};
                        switch param
                            case {'alphaR', 'alphaU'}
                                initParams = [initParams, rand];
                            case 'beta'
                                initParams = [initParams, rand*5];
                            case {'bias', 'sticky', 'wsls'}
                                initParams = [initParams, randn];
                        end
                    end
                    [params, negLL] = fmincon(@(p) -sum(fit_Qvalue(choices, rewards, p)), ...
                        initParams, [], [], [], [], lb, ub, [], optsMod);

                    if negLL < bestNegLL
                        bestNegLL = negLL;
                        bestParams = params;
                    end
                end
                % % === RUN fmincon ===
                % [bestParams, negLL] = fmincon(negLLfun, initParams, [], [], [], [], lb, ub, [], optsMod);

                % === OUTPUT ===
                assignFittedParams(bestParams, sub, conf, vol, contStim, paramStorage, activeParams, paramConfig);

                % Get parameter values for sessInfo
                paramValues = getParamValuesForSessInfo(sub, conf, vol, contStim, paramStorage, activeParams, paramConfig);
                sessInfoCollapse = [sessInfoCollapse; [data.overlap{3,iSess}, conf, vol, contStim-1, paramValues]];

                fprintf('Fit results:\n');
                for i = 1:length(activeParams)
                    param = activeParams{i};
                    fprintf('  %s = %.3f\n', param, paramStorage.(param){sub, conf, vol, contStim, 1});
                end
                fprintf('  Log-likelihood = %.3f\n', -negLL);

                % Visualize Q-values and choice probabilities
                % plot_Qmodel_diagnostics(choices, rewards, bestParams)





                % ==============================================
                % Session-wise for each subject per condition
                % Initialize session-wise parameter arrays
                sessionParams = struct();
                for i = 1:length(activeParams)
                    param = activeParams{i};
                    sessionParams.(param) = nan(length(sessIdx), 1);
                end

                for iSess = 1:length(sessIdx)
                    sessNum = sessIdx(iSess);
                    iData = data.overlap{1,sessNum};

                    % Remove first block and data past cutoff trials
                    iBlock2Start = find(iData(:,9) == 2, 1);
                    iData = iData(iBlock2Start:min(iBlock2Start+cutoffTrials-1, end),:);

                    % Remove first nTrialsRemove trials from each block
                    uniqueBlocks = unique(iData(:,9));
                    validData = [];
                    
                    for block = uniqueBlocks'
                        blockData = iData(iData(:,9) == block, :);
                        if size(blockData, 1) > nTrialsRemove
                            % Remove first nTrialsRemove trials from this block
                            validData = [validData; blockData(nTrialsRemove+1:end, :)];
                        end
                    end
                    
                    if isempty(validData)
                        fprintf('Warning: No valid data after removing trials for session %d\n', sessNum);
                        continue;
                    end
                    
                    iData = validData;

                    % Build a matrix of the rows of turn indications
                    emIdx = [];
                    for row = 2:size(iData, 1)
                        if iData(row, 2) ~= iData(row-1,2) && iData(row-1,2) == 0 && ismember(iData(row,2), 1:8) %when a turn is detected...
                            % if iData(row,2) == 11
                            %     disp('asdfasdfasdf')
                            % end
                            emIdx = [emIdx; row];
                        end %ending "if" statement looking for a turn detection
                    end %ending "for" loop going through data

                    if cutoffFlag
                        iRew = ismember(iData(emIdx,2), [crl irl crr irr]);
                        cutoffTrial = find(cumsum(iRew) == maxRewards, 1);

                        iEventMat = iData(emIdx(1:cutoffTrial),:);
                    else
                        iEventMat = iData(emIdx,:);
                    end

                    % goodTurnIdx = ismember(iEventMat(:,2), 1:8);
                    % if sum(goodTurnIdx == 0)
                    %     fprintf('Removed %d bad turns\n', sum(goodTurnIdx == 0))
                    % end
                    % iEventMat = iEventMat(goodTurnIdx,:);

                    [choices, rewards] = deal(nan(size(iEventMat, 1), 1));
                    choices(ismember(iEventMat(:,2), [crr cur irr, iur])) = 1; % right turns are 1
                    choices(ismember(iEventMat(:,2), [crl cul irl, iul])) = 2; % left turns are 2
                    rewards(ismember(iEventMat(:,2), [cul iul cur iur])) = 0; % unrewarded are 0
                    rewards(ismember(iEventMat(:,2), [crl irl crr irr])) = 1; % rewarded are 1

                    % === OBJECTIVE FUNCTION ===
                    negLLfun = @(params) -sum(fit_Qvalue(choices, rewards, params));  % minimize -log likelihood


                    % === RUN fmincon ===
                    bestNegLL = Inf;

                    for i = 1:nStarts
                        % Generate random initial parameters for active parameters only
                        initParams = [];
                        for j = 1:length(activeParams)
                            param = activeParams{j};
                            switch param
                                case {'alphaR', 'alphaU'}
                                    initParams = [initParams, rand];
                                case 'beta'
                                    initParams = [initParams, rand*5];
                                case {'bias', 'sticky', 'wsls'}
                                    initParams = [initParams, randn];
                            end
                        end
                        
                        [params, negLL] = fmincon(@(p) -sum(fit_Qvalue(choices, rewards, p)), ...
                            initParams, [], [], [], [], lb, ub, [], optsMod);

                        if negLL < bestNegLL
                            bestNegLL = negLL;
                            bestParams = params;
                        end
                    end

                    % === OUTPUT ===
                    paramIdx = 1;
                    for j = 1:length(activeParams)
                        param = activeParams{j};
                        if paramConfig.(param) == 1
                            sessionParams.(param)(iSess) = bestParams(paramIdx);
                            paramIdx = paramIdx + 1;
                        else
                            sessionParams.(param)(iSess) = nan;
                        end
                    end

                    % Get parameter values for sessInfo
                    paramValues = [];
                    for j = 1:length(activeParams)
                        param = activeParams{j};
                        if paramConfig.(param) == 1
                            paramValues = [paramValues, sessionParams.(param)(iSess)];
                        else
                            paramValues = [paramValues, nan];
                        end
                    end
                    
                    sessInfo = [sessInfo; [data.overlap{3,sessNum}, conf, vol, contStim-1, paramValues]];



                    % Visualize Q-values and choice probabilities
                    if plotFlag && plotModelDiagnostics
                        plot_Qmodel_diagnostics(choices, rewards, bestParams)
                        tText = sprintf('QLearning Model Subj %d Conf %d Vol %d ContStim %d', subjectList(sub), conf, vol, contStim-1);
                        sgtitle(tText)
                    end


                end
                % sessInfo
                % === OUTPUT ===
                % Store session-wise parameters
                for i = 1:length(activeParams)
                    param = activeParams{i};
                    paramStorage.(param){sub, conf, vol, contStim, 2} = sessionParams.(param);
                end

                % Calculate means for active parameters
                for i = 1:length(activeParams)
                    param = activeParams{i};
                    if paramConfig.(param) == 1
                        if strcmp(param, 'bias')
                            paramMeans.(param)(sub, conf, vol, contStim) = mean(abs(sessionParams.(param)), 'omitnan');
                        else
                            paramMeans.(param)(sub, conf, vol, contStim) = mean(sessionParams.(param), 'omitnan');
                        end
                    else
                        paramMeans.(param)(sub, conf, vol, contStim) = nan;
                    end
                end
                % fprintf('Fit results:\n');
                % fprintf('  alphaR = %.3f\n', alphaR);
                % fprintf('  alphaU = %.3f\n', alphaU);
                % fprintf('  beta   = %.3f\n', beta);
                % fprintf('  bias   = %.3f\n', bias);
                % fprintf('  Log-likelihood = %.3f\n', -negLL);



            end

            if plotFlag
                figure(subjectList(sub)); clf;
                set(gcf, 'Position', [-1600, 166, 734, 777]); % Example position and size

                ha = tight_subplot(2, 2, [0.07 0.05], [0.05 0.1]);  % [gap, lower margin, upper margin]
                % Plotting
                % Get parameter values for plotting
                [paramVals1C, paramVals2C, paramVals1S, paramVals2S, paramNames] = get_plot_params(sub, conf, vol, paramStorage, paramConfig, activeParams);
                
                rJitterC = (rand(length(paramVals2C), 1)) * 0.2;
                rJitterS = (rand(length(paramVals2S), 1)) * 0.2;
                xScatterC = (1:length(paramVals2C)) + rJitterC;
                xScatterS = (1:length(paramVals2S)) + rJitterS;

                yMax = .2 + max(max([paramVals1C; paramVals2C; paramVals1S; paramVals2S]));
                yMin = -.2 + min(min([paramVals1C; paramVals2C; paramVals1S; paramVals2S]));

                if ~isempty(paramVals1C)
                    axes(ha(1))
                    set(ha(1), 'YTickLabelMode', 'auto');  % Enable Y-axis labels
                    scatter(1:length(paramVals1C), paramVals1C, 80, 'LineWidth',2)
                    yline(0, '--')
                    xlim([.5 length(paramVals1C)+.5])
                    ylim([yMin yMax])
                    xticks(1:length(paramVals1C))
                    xticklabels(paramNames)
                    title('Collapsed sessions Control')
                end

                if ~isempty(paramVals1S)
                    axes(ha(2))
                    set(ha(2), 'YTickLabelMode', 'auto');  % Enable Y-axis labels
                    scatter(1:length(paramVals1S), paramVals1S, 80, 'LineWidth',2)
                    yline(0, '--')
                    xlim([.5 length(paramVals1S)+.5])
                    ylim([yMin yMax])
                    xticks(1:length(paramVals1S))
                    xticklabels(paramNames)
                    title('Collapsed sessions Stim')
                end

                if ~isempty(paramVals2C)
                    axes(ha(3))
                    hold on;
                    set(ha(3), 'YTickLabelMode', 'auto');  % Enable Y-axis labels
                    scatter(xScatterC, paramVals2C, 80, 'LineWidth',2)
                    yline(0, '--')
                    xlim([.5 length(paramVals2C)+.5])
                    ylim([yMin yMax])
                    xticks(1:length(paramVals2C))
                    xticklabels(paramNames)
                    title('Per Session Control')
                end

                if ~isempty(paramVals2S)
                    axes(ha(4))
                    hold on;
                    set(ha(4), 'YTickLabelMode', 'auto');  % Enable Y-axis labels
                    scatter(xScatterS, paramVals2S, 80, 'LineWidth',2)
                    yline(0, '--')
                    xlim([.5 length(paramVals2S)+.5])
                    ylim([yMin yMax])
                    xticks(1:length(paramVals2S))
                    xticklabels(paramNames)
                    title('Per Session Stim')
                end

                tText = sprintf('QLearning Model Subj %d Conf %d Vol %d Stim %d Remove %d Trials per Block', subjectList(sub), conf, vol, contStim-1, nTrialsRemove);
                sgtitle(tText)
                print('-dpng', fullfile(paths.dropPath, 'ymaze/', tText))
            end
        end
    end



end

%% Calculate and plot mean across subject parameters (and mean within subject across session)
% Plot control vs stimulation for each volatility and conflict condition
% Separate figures for D1 and D2 subjects

% Plotting constants
xVal = 1:length(activeParams);  % Dynamic based on number of active parameters
xDist = .2;

subjectSets = {d1Subj, d2Subj};
titles = {'D1 subjects', 'D2 subjects'};
volatilities = {'Low Volatility', 'High Volatility'};
singleSubColor = [.5 .5 .5];
d12StimColors = {[0 0 1], [1 0 0]}; % Blue for D1, Red for D2
colorCont = [0 0 0];

% Create separate figures for D1 and D2 subjects
for d = 1:2
    if d == 1
        sIdx = find(ismember(subjectList, d1Subj));
        groupName = 'D1';
    else
        sIdx = find(ismember(subjectList, d2Subj));
        groupName = 'D2';
    end
    
    if isempty(sIdx)
        continue; % Skip if no subjects in this group
    end
    
    figure('Position', [-1600 + (d-1)*200, 166, 1200, 800]);
    ha = tight_subplot(2, 3, [0.07 0.05], [0.05 0.1]); % 2 rows (volatility) x 3 cols (conflict)
    
    for vol = 1:2
        for conf = 1:3
            axes(ha((vol-1)*3 + conf));
            hold on;
            
            % Plot individual subject lines for each parameter
            for s = 1:length(sIdx)
                sub = sIdx(s);
                for p = 1:length(activeParams)
                    param = activeParams{p};
                    if paramConfig.(param) == 1
                        plot([xVal(p)-xDist xVal(p)+xDist], [paramMeans.(param)(sub,conf,vol,1) paramMeans.(param)(sub,conf,vol,2)], 'color', singleSubColor);
                    end
                end
            end
            
            % Plot group means and error bars for each parameter
            for p = 1:length(activeParams)
                param = activeParams{p};
                if paramConfig.(param) == 1
                    % Plot connecting line
                    plot([xVal(p)-xDist xVal(p)+xDist], [mean(paramMeans.(param)(sIdx,conf,vol,1)) mean(paramMeans.(param)(sIdx,conf,vol,2))], 'k', 'linewidth', 2);
                    
                    % Control point
                    scatter(xVal(p)-xDist, mean(paramMeans.(param)(sIdx,conf,vol,1)), 100, 'MarkerFaceColor', colorCont, 'MarkerEdgeColor', 'none');
                    e1 = errorbar(xVal(p)-xDist, mean(paramMeans.(param)(sIdx,conf,vol,1)), std(paramMeans.(param)(sIdx,conf,vol,1))/sqrt(length(paramMeans.(param)(sIdx,conf,vol,1))), std(paramMeans.(param)(sIdx,conf,vol,1))/sqrt(length(paramMeans.(param)(sIdx,conf,vol,1))), 'LineStyle', 'none', 'LineWidth', 2);
                    e1.Color = colorCont;
                    
                    % Stimulation point
                    scatter(xVal(p)+xDist, mean(paramMeans.(param)(sIdx,conf,vol,2)), 100, 'MarkerFaceColor', d12StimColors{d}, 'MarkerEdgeColor', 'none');
                    e2 = errorbar(xVal(p)+xDist, mean(paramMeans.(param)(sIdx,conf,vol,2)), std(paramMeans.(param)(sIdx,conf,vol,2))/sqrt(length(paramMeans.(param)(sIdx,conf,vol,2))), std(paramMeans.(param)(sIdx,conf,vol,2))/sqrt(length(paramMeans.(param)(sIdx,conf,vol,2))), 'LineStyle', 'none', 'LineWidth', 2);
                    e2.Color = d12StimColors{d};
                end
            end
            
            % Formatting
            yline(0, '--');
            ylim([-1 10]);
            xlim([.4 length(activeParams)+.4]);
            XTickLabel = activeParams;
            set(gca,'xtick',1:length(activeParams),'xticklabel',XTickLabel,'box','off','FontSize',14);
            ylabel('Value');
            xlabel('Params');
            set(gca, 'YTickLabelMode', 'auto');
            title(sprintf('Vol %d Conf %d', vol, conf));
            
            if vol == 1 && conf == 1
                sgtitle(sprintf('%s Subjects: Control vs Stimulation', groupName), 'FontSize', 16);
            end
        end
    end
    
    % Save the figure
    tText = sprintf('%s_Subjects_Control_vs_Stimulation', groupName);
    print('-dpng', fullfile(paths.dropPath, 'ymaze/', tText));
end

%% Calculate and plot modulation indices (Normalized Difference by Sum)
% Modulation Index = (Stim - Control) / (Stim + Control)
% This provides a symmetric ratio of stimulation effects

% Initialize modulation index arrays
modIndex = struct();
paramNames = activeParams;  % Use active parameters

% Calculate modulation indices for each parameter, subject, conflict, and volatility
for p = 1:length(paramNames)
    param = paramNames{p};
    modIndex.(param) = nan(length(subjectList), length(confRange), length(volRange));
    
    for sub = 1:length(subjectList)
        for conf = confRange
            for vol = volRange
                % Get control and stimulation values (session-wise means)
                controlVal = paramMeans.(param)(sub, conf, vol, 1); % contStim = 1 (control)
                stimVal = paramMeans.(param)(sub, conf, vol, 2);   % contStim = 2 (stimulation)
                
                % Calculate modulation index (avoid division by zero)
                if abs(controlVal + stimVal) > 1e-10
                    modIndex.(param)(sub, conf, vol) = (stimVal - controlVal) / (abs(stimVal) + abs(controlVal));
                else
                    modIndex.(param)(sub, conf, vol) = 0; % Default to 0 if both values are 0
                end
            end
        end
    end
end

% Plot modulation indices for each cohort
    % Plotting constants
    singleSubColor = [.5 .5 .5];
    d12StimColors = {[0 0 1], [1 0 0]}; % Blue for D1, Red for D2
    colorCont = [0 0 0];
    xDist = 0.2;
    
    % Create separate figures for D1 and D2 subjects
    for d = 1:2
        if d == 1
            sIdx = find(ismember(subjectList, d1Subj));
            groupName = 'D1';
        else
            sIdx = find(ismember(subjectList, d2Subj));
            groupName = 'D2';
        end
        
        if isempty(sIdx)
            continue; % Skip if no subjects in this group
        end
        
        figure('Position', [-1600 + (d-1)*200, 166, 1200, 800]);
        ha = tight_subplot(2, 3, [0.07 0.05], [0.05 0.1]); % 2 rows (volatility) x 3 cols (conflict)
        
        for vol = 1:2
            for conf = 1:3
                axes(ha((vol-1)*3 + conf));
                hold on;
                
                % Plot individual subject modulation indices for each parameter
                xVal = 1:length(paramNames);
                
                for s = 1:length(sIdx)
                    sub = sIdx(s);
                    
                    % Get modulation indices for this subject and condition
                    modVals = zeros(1, length(paramNames));
                    for p = 1:length(paramNames)
                        param = paramNames{p};
                        modVals(p) = modIndex.(param)(sub, conf, vol);
                    end
                    
                    % Plot individual subject data points only (no lines)
                    scatter(xVal, modVals, 50, 'o', 'MarkerFaceColor', singleSubColor, 'MarkerEdgeColor', 'none');
                end
                
                % Plot group mean
                groupModVals = zeros(1, length(paramNames));
                for p = 1:length(paramNames)
                    param = paramNames{p};
                    groupModVals(p) = mean(modIndex.(param)(sIdx, conf, vol), 'omitnan');
                end
                
                % Plot group mean data points
                scatter(xVal, groupModVals, 100, 'o', 'MarkerFaceColor', d12StimColors{d}, 'MarkerEdgeColor', 'none');
                
                % Add error bars (standard error of the mean)
                for p = 1:length(paramNames)
                    param = paramNames{p};
                    sem = std(modIndex.(param)(sIdx, conf, vol), 'omitnan') / sqrt(sum(~isnan(modIndex.(param)(sIdx, conf, vol))));
                    errorbar(xVal(p), groupModVals(p), sem, 'color', d12StimColors{d}, 'LineWidth', 2, 'CapSize', 5);
                end
                
                % Formatting
                yline(0, '--k', 'LineWidth', 1);
                xlim([0.5, length(paramNames)+0.5]);
                xticks(1:length(paramNames));
                xticklabels(paramNames);
                ylabel('Modulation Index');
                title(sprintf('Vol %d Conf %d', vol, conf));
                
                % Set consistent y-axis limits across all subplots
                ylim([-1, 1]);
                
                % Ensure y-axis tick marks are labeled
                set(gca, 'YTickLabelMode', 'auto');
                
                if vol == 1 && conf == 1
                    sgtitle(sprintf('%s Subjects: Modulation Indices', groupName), 'FontSize', 16);
                end
            end
        end
        
        % Save the figure
        tText = sprintf('%s_Subjects_Modulation_Indices', groupName);
        print('-dpng', fullfile(paths.dropPath, 'ymaze/', tText));
    end

%% Linear Mixed Effects Model: Session-wise Analysis Across All Subjects and Sessions
% This block fits a linear mixed effects model for each parameter, using session-wise data
% Each data point is a session from a subject (not averaged within subject)
% Fixed effects: stimControl, conflict, volatility; Random effect: subject

% Prepare arrays to collect data
paramNames = activeParams;  % Use active parameters

allData = struct();
for p = 1:length(paramNames)
    allData.(paramNames{p}) = [];
end

% Loop over subjects, conditions, and sessions to collect data
for sub = 1:length(subjectList)
    % Determine group (d1 or d2)
    if ismember(subjectList(sub), d1Subj)
        group = 'd1';
    elseif ismember(subjectList(sub), d2Subj)
        group = 'd2';
    else
        group = 'other';
    end
    for conf = confRange
        for vol = volRange
            for contStim = contStimRange
                % Get session-wise parameter vectors (may contain NaN for missing sessions)
                % Use any active parameter to get the number of sessions
                nSess = 0;
                for i = 1:length(activeParams)
                    param = activeParams{i};
                    if paramConfig.(param) == 1
                        nSess = length(paramStorage.(param){sub, conf, vol, contStim, 2});
                        break;
                    end
                end
                for iSess = 1:nSess
                    % For each parameter, add a row to the table
                    for p = 1:length(paramNames)
                        param = paramNames{p};
                        if paramConfig.(param) == 1
                            val = paramStorage.(param){sub, conf, vol, contStim, 2}(iSess);
                            if isnan(val), continue; end
                            % Add row to cell array
                            allData.(param) = [allData.(param); {val, subjectList(sub), contStim-1, conf, vol, group}];
                        end
                    end
                end
            end
        end
    end
end

% Convert cell arrays to tables
for p = 1:length(paramNames)
    param = paramNames{p};
    T = allData.(param);
    if isempty(T)
        allData.(param) = table();
        continue;
    end
    % Ensure T is a 2-D cell array
    if size(T, 1) == 1 && size(T, 2) > 6
        T = T';  % Transpose if it's a single row with many columns
    end
    T = cell2table(T, 'VariableNames', {'Value', 'Subject', 'StimControl', 'Conflict', 'Volatility', 'Group'});
    allData.(param) = T;
end

% Fit linear mixed effects model for each parameter and store results
lmeResults = struct();
significantEffects = struct();

for p = 1:length(paramNames)
    param = paramNames{p};
    T = allData.(param);
    if isempty(T) || height(T) < 10
        fprintf('Not enough data for %s\n', param);
        continue;
    end
    % Convert categorical variables
    T.Subject = categorical(T.Subject);
    T.StimControl = categorical(T.StimControl);
    T.Conflict = categorical(T.Conflict);
    T.Volatility = categorical(T.Volatility);
    T.Group = categorical(T.Group);
    
    % Fit model: Value ~ StimControl*Conflict*Volatility + (1|Subject)
    lme = fitlme(T, 'Value ~ StimControl*Conflict*Volatility + (1|Subject)');
    lmeResults.(param) = lme;
    
    % Get ANOVA results
    anovaResults = anova(lme);
    
    % Store significant effects (p < 0.05)
    sigIdx = anovaResults.pValue < 0.05;
    if any(sigIdx)
        significantEffects.(param) = anovaResults(sigIdx, :);
    else
        significantEffects.(param) = [];
    end
    
    fprintf('\nLinear Mixed Effects Model for %s:\n', param);
    disp(lme);
    disp(anovaResults);
    
    % Print significant effects summary
    if any(sigIdx)
                    iSigIdx = find(sigIdx);

        fprintf('\n=== SIGNIFICANT EFFECTS for %s ===\n', param);
        for i = 1:height(anovaResults(sigIdx, :))
            effect = anovaResults.Term{find(sigIdx,i,'first')};
            pval = anovaResults.pValue(iSigIdx(i));
            fprintf('  %s: p = %.4f\n', effect, pval);
        end
    else
        fprintf('\nNo significant effects found for %s\n', param);
    end
end

%% Create comprehensive summary plots and tables

% 1. Summary table of all significant effects
fprintf('\n\n==========================================\n');
fprintf('COMPREHENSIVE SUMMARY OF SIGNIFICANT EFFECTS\n');
fprintf('==========================================\n');

allSigEffects = {};
for p = 1:length(paramNames)
    param = paramNames{p};
    if isfield(significantEffects, param) && ~isempty(significantEffects.(param))
        for i = 1:height(significantEffects.(param))
            effect = significantEffects.(param).Term{i};
            pval = significantEffects.(param).pValue(i);
            fstat = significantEffects.(param).FStat(i);
            allSigEffects{end+1, 1} = param;
            allSigEffects{end, 2} = effect;
            allSigEffects{end, 3} = pval;
            allSigEffects{end, 4} = fstat;
        end
    end
end

if ~isempty(allSigEffects)
    sigTable = cell2table(allSigEffects, 'VariableNames', {'Parameter', 'Effect', 'PValue', 'FStat'});
    disp(sigTable);
else
    fprintf('No significant effects found across all parameters.\n');
end

% 2. Create heatmap of p-values for all effects
figure('Position', [100, 100, 1200, 800]);
pValueMatrix = nan(length(paramNames), 7); % 7 possible effects: Intercept, StimControl, Conflict, Volatility, S*C, S*V, C*V, S*C*V
effectNames = {'Intercept', 'StimControl', 'Conflict', 'Volatility', 'StimControl:Conflict', 'StimControl:Volatility', 'Conflict:Volatility', 'StimControl:Conflict:Volatility'};

for p = 1:length(paramNames)
    param = paramNames{p};
    if isfield(lmeResults, param)
        anovaResults = anova(lmeResults.(param));
        for e = 1:length(effectNames)
            idx = find(strcmp(anovaResults.Term, effectNames{e}));
            if ~isempty(idx)
                pValueMatrix(p, e) = anovaResults.pValue(idx);
            end
        end
    end
end

% Create heatmap
imagesc(pValueMatrix);
colormap(flipud(hot)); % Red = significant, white = not significant
colorbar;
caxis([0 0.1]); % Focus on p < 0.1 range

% Add labels
xticks(1:length(effectNames));
xticklabels(effectNames);
xtickangle(45);
yticks(1:length(paramNames));
yticklabels(paramNames);

% Add p-value text
for p = 1:length(paramNames)
    for e = 1:length(effectNames)
        if ~isnan(pValueMatrix(p, e))
            if pValueMatrix(p, e) < 0.001
                text(e, p, '***', 'HorizontalAlignment', 'center', 'FontWeight', 'bold');
            elseif pValueMatrix(p, e) < 0.01
                text(e, p, '**', 'HorizontalAlignment', 'center', 'FontWeight', 'bold');
            elseif pValueMatrix(p, e) < 0.05
                text(e, p, '*', 'HorizontalAlignment', 'center', 'FontWeight', 'bold');
            else
                text(e, p, sprintf('%.3f', pValueMatrix(p, e)), 'HorizontalAlignment', 'center', 'FontSize', 8);
            end
        end
    end
end

title('P-values for Linear Mixed Effects Model Results');
xlabel('Effects');
ylabel('Parameters');

% 3. Group-specific analysis
fprintf('\n\n==========================================\n');
fprintf('GROUP-SPECIFIC ANALYSIS (D1 vs D2)\n');
fprintf('==========================================\n');

% Fit separate models for each group
groupResults = struct();
groupSigEffects = struct();

for group = {'d1', 'd2'}
    groupName = group{1};
    fprintf('\n--- %s Subjects ---\n', upper(groupName));
    
    % Collect all significant effects for this group
    groupAllSigEffects = {};
    
    for p = 1:length(paramNames)
        param = paramNames{p};
        T = allData.(param);
        if isempty(T) || height(T) < 5
            continue;
        end
        
        % Filter for specific group
        groupT = T(strcmp(T.Group, groupName), :);
        if height(groupT) < 5
            continue;
        end
        
        % Fit model for this group
        groupLME = fitlme(groupT, 'Value ~ StimControl*Conflict*Volatility + (1|Subject)');
        groupResults.(groupName).(param) = groupLME;
        
        % Get significant effects
        groupAnova = anova(groupLME);
        sigGroupIdx = groupAnova.pValue < 0.05;
        
        if any(sigGroupIdx)
            iSigIdx = find(sigGroupIdx);
            fprintf('\n  %s:\n', param);
            for i = 1:height(groupAnova(sigGroupIdx, :))
                effect = groupAnova.Term{find(sigIdx,i,'first')};
                pval = groupAnova.pValue(iSigIdx(i));
                fstat = groupAnova.FStat(iSigIdx(i));
                fprintf('    %s: p = %.4f\n', effect, pval);
                
                % Store for summary table
                groupAllSigEffects{end+1, 1} = param;
                groupAllSigEffects{end, 2} = effect;
                groupAllSigEffects{end, 3} = pval;
                groupAllSigEffects{end, 4} = fstat;
            end
        end
    end
    
    % Store group-specific significant effects
    if ~isempty(groupAllSigEffects)
        groupSigEffects.(groupName) = cell2table(groupAllSigEffects, 'VariableNames', {'Parameter', 'Effect', 'PValue', 'FStat'});
        fprintf('\n=== %s SUBJECTS SIGNIFICANT EFFECTS SUMMARY ===\n', upper(groupName));
        disp(groupSigEffects.(groupName));
    else
        groupSigEffects.(groupName) = [];
        fprintf('\nNo significant effects found for %s subjects\n', upper(groupName));
    end
end

% Create separate p-value heatmaps for D1 and D2
for group = {'d1', 'd2'}
    groupName = group{1};
    
    figure('Position', [100 + (strcmp(groupName, 'd2') * 200), 100, 1200, 800]);
    pValueMatrix = nan(length(paramNames), 7); % 7 possible effects
    
    for p = 1:length(paramNames)
        param = paramNames{p};
        if isfield(groupResults, groupName) && isfield(groupResults.(groupName), param)
            anovaResults = anova(groupResults.(groupName).(param));
            for e = 1:length(effectNames)
                idx = find(strcmp(anovaResults.Term, effectNames{e}));
                if ~isempty(idx)
                    pValueMatrix(p, e) = anovaResults.pValue(idx);
                end
            end
        end
    end
    
    % Create heatmap
    imagesc(pValueMatrix);
    colormap(flipud(hot)); % Red = significant, white = not significant
    colorbar;
    caxis([0 0.1]); % Focus on p < 0.1 range
    
    % Add labels
    xticks(1:length(effectNames));
    xticklabels(effectNames);
    xtickangle(45);
    yticks(1:length(paramNames));
    yticklabels(paramNames);
    
    % Add p-value text
    for p = 1:length(paramNames)
        for e = 1:length(effectNames)
            if ~isnan(pValueMatrix(p, e))
                if pValueMatrix(p, e) < 0.001
                    text(e, p, '***', 'HorizontalAlignment', 'center', 'FontWeight', 'bold');
                elseif pValueMatrix(p, e) < 0.01
                    text(e, p, '**', 'HorizontalAlignment', 'center', 'FontWeight', 'bold');
                elseif pValueMatrix(p, e) < 0.05
                    text(e, p, '*', 'HorizontalAlignment', 'center', 'FontWeight', 'bold');
                else
                    text(e, p, sprintf('%.3f', pValueMatrix(p, e)), 'HorizontalAlignment', 'center', 'FontSize', 8);
                end
            end
        end
    end
    
    title(sprintf('P-values for %s Subjects Linear Mixed Effects Model Results', upper(groupName)));
    xlabel('Effects');
    ylabel('Parameters');
end

% Create group comparison summary table
fprintf('\n\n==========================================\n');
fprintf('D1 vs D2 COMPARISON SUMMARY\n');
fprintf('==========================================\n');

% Compare number of significant effects between groups
for group = {'d1', 'd2'}
    groupName = group{1};
    if isfield(groupSigEffects, groupName) && ~isempty(groupSigEffects.(groupName))
        fprintf('\n%s Subjects: %d significant effects\n', upper(groupName), height(groupSigEffects.(groupName)));
        
        % Count effects by type
        stimEffects = sum(contains(groupSigEffects.(groupName).Effect, 'StimControl'));
        confEffects = sum(contains(groupSigEffects.(groupName).Effect, 'Conflict'));
        volEffects = sum(contains(groupSigEffects.(groupName).Effect, 'Volatility'));
        interactions = sum(contains(groupSigEffects.(groupName).Effect, ':'));
        
        fprintf('  - Stimulation effects: %d\n', stimEffects);
        fprintf('  - Conflict effects: %d\n', confEffects);
        fprintf('  - Volatility effects: %d\n', volEffects);
        fprintf('  - Interactions: %d\n', interactions);
    else
        fprintf('\n%s Subjects: No significant effects\n', upper(groupName));
    end
end

% Create parameter-wise comparison
fprintf('\nParameter-wise comparison:\n');
for p = 1:length(paramNames)
    param = paramNames{p};
    fprintf('\n%s:\n', param);
    
    for group = {'d1', 'd2'}
        groupName = group{1};
        if isfield(groupSigEffects, groupName) && ~isempty(groupSigEffects.(groupName))
            paramEffects = groupSigEffects.(groupName)(strcmp(groupSigEffects.(groupName).Parameter, param), :);
            if ~isempty(paramEffects)
                fprintf('  %s: %d effects (', upper(groupName), height(paramEffects));
                for i = 1:height(paramEffects)
                    fprintf('%s p=%.3f', paramEffects.Effect{i}, paramEffects.PValue(i));
                    if i < height(paramEffects), fprintf(', '); end
                end
                fprintf(')\n');
            else
                fprintf('  %s: No significant effects\n', upper(groupName));
            end
        else
            fprintf('  %s: No significant effects\n', upper(groupName));
        end
    end
end

% 4. Effect size analysis (Cohen's d for significant effects)
fprintf('\n\n==========================================\n');
fprintf('EFFECT SIZE ANALYSIS\n');
fprintf('==========================================\n');

% Overall effect sizes
fprintf('\n--- OVERALL EFFECT SIZES ---\n');
for p = 1:length(paramNames)
    param = paramNames{p};
    T = allData.(param);
    if isempty(T)
        continue;
    end
    
    % Calculate effect sizes for main effects
    fprintf('\n%s:\n', param);
    
    % StimControl effect
    controlVals = T.Value(T.StimControl == 0);
    stimVals = T.Value(T.StimControl == 1);
    if ~isempty(controlVals) && ~isempty(stimVals)
        cohensD = (mean(stimVals) - mean(controlVals)) / sqrt((var(controlVals) + var(stimVals)) / 2);
        fprintf('  StimControl effect size (Cohen''s d): %.3f\n', cohensD);
    end
    
    % Conflict effect
    for conf = 1:3
        confVals = T.Value(T.Conflict == conf);
        if ~isempty(confVals)
            fprintf('  Conflict %d mean: %.3f (n=%d)\n', conf, mean(confVals), length(confVals));
        end
    end
    
    % Volatility effect
    for vol = 1:2
        volVals = T.Value(T.Volatility == vol);
        if ~isempty(volVals)
            fprintf('  Volatility %d mean: %.3f (n=%d)\n', vol, mean(volVals), length(volVals));
        end
    end
end

% Group-specific effect sizes
fprintf('\n--- GROUP-SPECIFIC EFFECT SIZES ---\n');
for group = {'d1', 'd2'}
    groupName = group{1};
    fprintf('\n%s Subjects:\n', upper(groupName));
    
    for p = 1:length(paramNames)
        param = paramNames{p};
        T = allData.(param);
        if isempty(T)
            continue;
        end
        
        % Filter for specific group
        groupT = T(strcmp(T.Group, groupName), :);
        if height(groupT) < 5
            continue;
        end
        
        fprintf('\n  %s:\n', param);
        
        % StimControl effect for this group
        controlVals = groupT.Value(groupT.StimControl == 0);
        stimVals = groupT.Value(groupT.StimControl == 1);
        if ~isempty(controlVals) && ~isempty(stimVals)
            cohensD = (mean(stimVals) - mean(controlVals)) / sqrt((var(controlVals) + var(stimVals)) / 2);
            fprintf('    StimControl effect size (Cohen''s d): %.3f\n', cohensD);
        end
        
        % Conflict effect for this group
        for conf = 1:3
            confVals = groupT.Value(groupT.Conflict == conf);
            if ~isempty(confVals)
                fprintf('    Conflict %d mean: %.3f (n=%d)\n', conf, mean(confVals), length(confVals));
            end
        end
        
        % Volatility effect for this group
        for vol = 1:2
            volVals = groupT.Value(groupT.Volatility == vol);
            if ~isempty(volVals)
                fprintf('    Volatility %d mean: %.3f (n=%d)\n', vol, mean(volVals), length(volVals));
            end
        end
    end
end

% 5. Create interaction plots for significant interactions
% Overall interactions
figure(10); clf;
set(gcf, 'Position', [200, 200, 1400, 1000]);
plotIdx = 1;

for p = 1:length(paramNames)
    param = paramNames{p};
    if isfield(significantEffects, param) && ~isempty(significantEffects.(param))
        % Check for interactions
        hasInteraction = any(contains(significantEffects.(param).Term, ':'));
        if hasInteraction
            T = allData.(param);
            
            % StimControl:Conflict interaction
            if any(contains(significantEffects.(param).Term, 'StimControl:Conflict'))
                subplot(2, 3, plotIdx);
                plotInteraction(T, 'StimControl', 'Conflict', param);
                plotIdx = plotIdx + 1;
            end
            
            % StimControl:Volatility interaction
            if any(contains(significantEffects.(param).Term, 'StimControl:Volatility'))
                subplot(2, 3, plotIdx);
                plotInteraction(T, 'StimControl', 'Volatility', param);
                plotIdx = plotIdx + 1;
            end
            
            % Conflict:Volatility interaction
            if any(contains(significantEffects.(param).Term, 'Conflict:Volatility'))
                subplot(2, 3, plotIdx);
                plotInteraction(T, 'Conflict', 'Volatility', param);
                plotIdx = plotIdx + 1;
            end
        end
    end
end

if plotIdx > 1
    sgtitle('Overall Significant Interaction Effects');
else
                sgtitle(sprintf('%s No Significant Interaction Effects', upper(groupName)));
end

% Group-specific interaction plots
for group = {'d1', 'd2'}
    groupName = group{1};
    
    figure(11); clf;
set(gcf, 'Position', [300 + (strcmp(groupName, 'd2') * 200), 300, 1400, 1000]);
plotIdx = 1;
    
    for p = 1:length(paramNames)
        param = paramNames{p};
        if isfield(groupSigEffects, groupName) && ~isempty(groupSigEffects.(groupName))
            % Check for interactions in this group
            paramEffects = groupSigEffects.(groupName)(strcmp(groupSigEffects.(groupName).Parameter, param), :);
            hasInteraction = any(contains(paramEffects.Effect, ':'));
            
            if hasInteraction
                T = allData.(param);
                groupT = T(strcmp(T.Group, groupName), :);
                
                % StimControl:Conflict interaction
                if any(contains(paramEffects.Effect, 'StimControl:Conflict'))
                    subplot(2, 3, plotIdx);
                    plotInteraction(groupT, 'StimControl', 'Conflict', sprintf('%s (%s)', param, upper(groupName)));
                    plotIdx = plotIdx + 1;
                end
                
                % StimControl:Volatility interaction
                if any(contains(paramEffects.Effect, 'StimControl:Volatility'))
                    subplot(2, 3, plotIdx);
                    plotInteraction(groupT, 'StimControl', 'Volatility', sprintf('%s (%s)', param, upper(groupName)));
                    plotIdx = plotIdx + 1;
                end
                
                % Conflict:Volatility interaction
                if any(contains(paramEffects.Effect, 'Conflict:Volatility'))
                    subplot(2, 3, plotIdx);
                    plotInteraction(groupT, 'Conflict', 'Volatility', sprintf('%s (%s)', param, upper(groupName)));
                    plotIdx = plotIdx + 1;
                end
            end
        end
    end
    
    if plotIdx > 1
        sgtitle(sprintf('%s Subjects: Significant Interaction Effects', upper(groupName)));
    else
                sgtitle(sprintf('%s No Significant Interaction Effects', upper(groupName)));
    end
end

% Save results summary
save(fullfile(paths.dropPath, 'ymaze/', 'LME_Results_Summary.mat'), 'lmeResults', 'significantEffects', 'groupResults', 'groupSigEffects', 'sigTable');
fprintf('\nResults saved to: %s\n', fullfile(paths.dropPath, 'ymaze/', 'LME_Results_Summary.mat'));



%% Individual subject stats: Store signrank test results
% Initialize p-value storage for active parameters
pValues = struct();
for i = 1:length(activeParams)
    param = activeParams{i};
    if paramConfig.(param) == 1
        pValues.(param) = nan(length(subjectList), length(confRange), length(volRange));
    end
end

for iSub = 1:length(subjectList)
    for iConf = 1:3
        for iVol = 1:2
            % Compare contStim 1 vs 2 for each active parameter
            for i = 1:length(activeParams)
                param = activeParams{i};
                if paramConfig.(param) == 1
                    [pValues.(param)(iSub,iConf,iVol), ~] = ranksum(paramStorage.(param){iSub,iConf,iVol,1,2}, paramStorage.(param){iSub,iConf,iVol,2,2});
                end
            end
        end
    end
end

% Create figures for D1 and D2 subjects
d1Idx = find(ismember(subjectList, d1Subj));
d2Idx = find(ismember(subjectList, d2Subj));
subjectGroups = {d1Idx, d2Idx};
groupNames = {'D1 Subjects', 'D2 Subjects'};
jitterAmount = 0.1; % Amount of x-axis jitter

for group = 1:2
    figure(12); clf;
    ha = tight_subplot(2, 3, [0.1 0.1], [0.15 0.1], [0.1 0.05]); % [gap_h gap_w, marg_h, marg_w]

    currIdx = subjectGroups{group};
    for vol = 1:2
        for conf = 1:3
            axes(ha((vol-1)*3 + conf));
            hold on;

            % For each subject in the group
            for s = 1:length(currIdx)
                % Add random jitter to x positions
                xJitter = (rand(1,length(activeParams))-0.5) * jitterAmount;

                % Plot p-values for each active parameter with jitter
                for p = 1:length(activeParams)
                    param = activeParams{p};
                    if paramConfig.(param) == 1
                        scatter(p+xJitter(p), pValues.(param)(currIdx(s),conf,vol), 50, 'o', 'LineWidth', 2)
                    end
                end
            end

            % Add significance line
            yline(0.05, '--r')

            % Formatting
            xlim([0.5 length(activeParams)+0.5])
            ylim([0 1])
            xticks(1:length(activeParams))
            xticklabels(activeParams)
            ylabel('p-value')
            title(sprintf('Vol %d Conf %d', vol, conf))

            if vol == 1 && conf == 1
                sgtitle(groupNames{group})
            end
        end
    end
end



%%  Calculate change between controls and stimulation for all conditions/subjects
% Initialize change arrays for active parameters
dPct = struct();
pVal = struct();
for i = 1:length(activeParams)
    param = activeParams{i};
    if paramConfig.(param) == 1
        dPct.(param) = nan(length(subjectList), length(confRange), length(volRange));
        pVal.(param) = nan(length(confRange), length(volRange));
    end
end

for conf = confRange
    for vol = volRange
        for sub = 1 : length(subjectList)
            for i = 1:length(activeParams)
                param = activeParams{i};
                if paramConfig.(param) == 1
                    dPct.(param)(sub, conf, vol) = (paramMeans.(param)(sub,conf,vol,2) - paramMeans.(param)(sub,conf,vol,1));
                end
            end
        end
        
        % Calculate p-values for D1 vs D2 comparison
        for i = 1:length(activeParams)
            param = activeParams{i};
            if paramConfig.(param) == 1
                pVal.(param)(conf, vol) = ranksum(dPct.(param)(1:4, conf, vol), dPct.(param)(5:8, conf, vol));
            end
        end
        
        % Print results
        fprintf('Conf %d Vol %d: ', conf, vol);
        for i = 1:length(activeParams)
            param = activeParams{i};
            if paramConfig.(param) == 1
                fprintf('%s: %.3f\t', param, pVal.(param)(conf, vol));
            end
        end
        fprintf('\n');
    end
end

%%
% Parameters
params = activeParams;  % Use active parameters
pVals = {};
for i = 1:length(activeParams)
    param = activeParams{i};
    if paramConfig.(param) == 1
        pVals{end+1} = pVal.(param);
    end
end
thresh = 0.05;

% Conflict and Volatility axis
confLevels = 1:3;
volLevels = 1:2;

% Use a lighter colormap
cmap = parula(256);
cmap = cmap .^ 0.7;  % brighten

% Plot settings
figure(2); clf
tiledlayout(1, length(pVals), 'Padding', 'compact', 'TileSpacing', 'compact');

for i = 1:length(pVals)
    pMat = pVals{i};  % 3 (conflict) x 2 (volatility)

    nexttile;
    imagesc(volLevels, confLevels, pMat);
    set(gca, 'YDir', 'normal');
    colormap(cmap);
    if i == length(pVals)
        colorbar; % only on last subplot
    end
    caxis([0 1]);
    % Get the parameter name for this p-value
    paramIdx = 1;
    for j = 1:length(activeParams)
        param = activeParams{j};
        if paramConfig.(param) == 1
            if paramIdx == i
                title(param);
                break;
            end
            paramIdx = paramIdx + 1;
        end
    end
    xlabel('Volatility');
    ylabel('Conflict');
    xticks(volLevels);
    yticks(confLevels);

    % Overlay annotated text
    for row = 1:3
        for col = 1:2
            val = pMat(row, col);
            label = sprintf('%.3f', val);
            if val < thresh
                label = [label '*'];
            end
            text(col, row, label, 'HorizontalAlignment', 'center', ...
                'Color', 'black', 'FontSize', 10, 'FontWeight', 'bold');
        end
    end
end

sgtitle('Ranksum p-values (D1 vs D2 stimulation effects)');




%%


% Test section - can be updated as needed for specific parameter testing
% This section can be customized based on which parameters are active


%%  Test control vs. stim

volValues = [1 2];
confValues = [1 2 3];
subjectTest = d2Subj;

% Initialize control and stimulation arrays for active parameters
contVals = struct();
stimVals = struct();
for i = 1:length(activeParams)
    param = activeParams{i};
    if paramConfig.(param) == 1
        contVals.(param) = nan(length(subjectTest), 1);
        stimVals.(param) = nan(length(subjectTest), 1);
    end
end

for sub = 1:length(subjectTest)
    rowIdxCont = sessInfo(:, subIdx) == subjectTest(sub) & sessInfo(:, stimIdx) == 0;
    rowIdxStim = sessInfo(:, subIdx) == subjectTest(sub) & sessInfo(:, stimIdx) == 1;

    % Get parameter values for active parameters
    for i = 1:length(activeParams)
        param = activeParams{i};
        if paramConfig.(param) == 1
            % Find the column index for this parameter in sessInfo
            paramColIdx = 4 + i; % Assuming parameters start at column 5
            contVals.(param)(sub) = mean(sessInfo(rowIdxCont, paramColIdx));
            stimVals.(param)(sub) = mean(sessInfo(rowIdxStim, paramColIdx));
        end
    end
end

% Perform statistical tests for active parameters
fprintf('Control vs Stimulation tests:\n');
for i = 1:length(activeParams)
    param = activeParams{i};
    if paramConfig.(param) == 1
        [h, p] = kstest2(contVals.(param), stimVals.(param));
        fprintf('%s: h=%d, p=%.4f\n', param, h, p);
    end
end


%%



% Helper function to assign fitted parameters to storage arrays
function assignFittedParams(bestParams, sub, conf, vol, contStim, paramStorage, activeParams, paramConfig)
    paramIdx = 1;
    for i = 1:length(activeParams)
        param = activeParams{i};
        if paramConfig.(param) == 1
            paramStorage.(param){sub, conf, vol, contStim, 1} = bestParams(paramIdx);
            paramIdx = paramIdx + 1;
        else
            paramStorage.(param){sub, conf, vol, contStim, 1} = nan;
        end
    end
end

% Helper function to get parameter values for sessInfo
function paramValues = getParamValuesForSessInfo(sub, conf, vol, contStim, paramStorage, activeParams, paramConfig)
    paramValues = [];
    for i = 1:length(activeParams)
        param = activeParams{i};
        if paramConfig.(param) == 1
            paramValues = [paramValues, paramStorage.(param){sub, conf, vol, contStim, 1}];
        else
            paramValues = [paramValues, nan];
        end
    end
end





function [LL_total, Q_total, p_choice] = fit_Qvalue(choices, rewards, params)
% Q-learning with softmax, bias, optional stickiness, and optional WSLS term
% Inputs:
% - choices: vector (1 = right, 2 = left)
% - rewards: vector (0 or 1)
% - params: parameter vector with active parameters only
% - paramConfig: structure defining which parameters are active (passed via global)
%
% Outputs:
% - LL_total: log-likelihood per trial
% - Q_total: Q-values per trial (nTrials+1 x 2)
% - p_choice: model probability of choosing right per trial

% Get parameter configuration from global workspace
global paramConfig activeParams

% === UNPACK PARAMS ===
paramIdx = 1;
alphaR = 0.2;  % Default values
alphaU = 0.2;
beta = 2;
bias = 0;
stickiness = 0;
wsls = 0;

% Assign parameters based on active configuration
for i = 1:length(activeParams)
    param = activeParams{i};
    if paramConfig.(param) == 1
        switch param
            case 'alphaR'
                alphaR = params(paramIdx);
            case 'alphaU'
                alphaU = params(paramIdx);
            case 'beta'
                beta = params(paramIdx);
            case 'bias'
                bias = params(paramIdx);
            case 'sticky'
                stickiness = params(paramIdx);
            case 'wsls'
                wsls = params(paramIdx);
        end
        paramIdx = paramIdx + 1;
    end
end

nTrials = length(choices);
Q_total = zeros(nTrials + 1, 2);  % Q(:,1)=right, Q(:,2)=left
LL_total = zeros(nTrials, 1);     % log-likelihood per trial
p_choice = zeros(nTrials, 1);     % predicted P(choose right)

prev_choice = 0;  % initialize previous choice (0 = no prior)

% === MAIN TRIAL LOOP ===
for t = 1:nTrials
    Q = Q_total(t, :);  % current Q-values
    QR = Q(1); QL = Q(2);

    % Stickiness term: +1 if previous choice = right, -1 if left, 0 if no history
    stick_term = 0;
    if prev_choice == 1
        stick_term = +1;
    elseif prev_choice == 2
        stick_term = -1;
    end

    % WSLS term: +1 if previous choice was rewarded and we're choosing the same, or if previous choice was unrewarded and we're choosing different
    wsls_term = 0;
    if t > 1  % Need at least one previous trial
        prev_reward = rewards(t-1);
        if (prev_reward == 1 && choices(t) == prev_choice) || (prev_reward == 0 && choices(t) ~= prev_choice)
            wsls_term = +1;
        else
            wsls_term = -1;
        end
    end

    % Compute softmax probability of choosing right
    p_right = 1 / (1 + exp(-beta * (QR - QL) + bias + stickiness * stick_term + wsls * wsls_term));
    p_choice(t) = p_right;

    % Log-likelihood for actual choice
    if choices(t) == 1
        LL_total(t) = log(p_right + eps);
    elseif choices(t) == 2
        LL_total(t) = log(1 - p_right + eps);
    else
        error('Invalid choice at trial %d. Must be 1 (right) or 2 (left).', t);
    end

    % Update Q-values
    chosen = choices(t);
    reward = rewards(t);
    alpha = alphaR * (reward == 1) + alphaU * (reward == 0);
    Q(chosen) = Q(chosen) + alpha * (reward - Q(chosen));
    Q_total(t + 1, :) = Q;

    % Update previous choice
    prev_choice = chosen;
end
end




function plot_Qmodel_diagnostics(choices, rewards, params)
% Q-learning diagnostic visualizer using tight_subplot (3x2 grid)
% Inputs:
% - choices: vector (1 = right, 2 = left)
% - rewards: vector (0 or 1)
% - params: parameter vector with active parameters only
% - paramConfig: structure defining which parameters are active (passed via global)

% Get parameter configuration from global workspace
global paramConfig activeParams

% Unpack parameters
paramIdx = 1;
alphaR = 0.2;  % Default values
alphaU = 0.2;
beta = 2;
bias = 0;
stickiness = 0;
wsls = 0;

% Assign parameters based on active configuration
for i = 1:length(activeParams)
    param = activeParams{i};
    if paramConfig.(param) == 1
        switch param
            case 'alphaR'
                alphaR = params(paramIdx);
            case 'alphaU'
                alphaU = params(paramIdx);
            case 'beta'
                beta = params(paramIdx);
            case 'bias'
                bias = params(paramIdx);
            case 'sticky'
                stickiness = params(paramIdx);
            case 'wsls'
                wsls = params(paramIdx);
        end
        paramIdx = paramIdx + 1;
    end
end

nTrials = length(choices);

Q = zeros(2,1);
Q_total = zeros(nTrials+1, 2);
p_right = zeros(nTrials, 1);

% Run model with all terms
prev_choice = 0;
for t = 1:nTrials
    QR = Q(1); QL = Q(2);
    
    % Stickiness term
    stick_term = 0;
    if prev_choice == 1
        stick_term = +1;
    elseif prev_choice == 2
        stick_term = -1;
    end
    
    % WSLS term
    wsls_term = 0;
    if t > 1
        prev_reward = rewards(t-1);
        if (prev_reward == 1 && choices(t) == prev_choice) || (prev_reward == 0 && choices(t) ~= prev_choice)
            wsls_term = +1;
        else
            wsls_term = -1;
        end
    end
    
    p = 1 / (1 + exp(-beta * (QR - QL) + bias + stickiness * stick_term + wsls * wsls_term));
    p_right(t) = p;

    chosen = choices(t);
    reward = rewards(t);
    alpha = alphaR * (reward == 1) + alphaU * (reward == 0);
    Q(chosen) = Q(chosen) + alpha * (reward - Q(chosen));
    Q_total(t+1, :) = Q;
    
    prev_choice = chosen;
end

% === Compute choice switching after reward/no-reward ===
switchAfterReward = [];
switchAfterNoReward = [];

for t = 2:nTrials
    if choices(t) ~= choices(t-1)
        if rewards(t-1) == 1
            switchAfterReward(end+1) = 1;
        else
            switchAfterNoReward(end+1) = 1;
        end
    else
        if rewards(t-1) == 1
            switchAfterReward(end+1) = 0;
        else
            switchAfterNoReward(end+1) = 0;
        end
    end
end

% Calculate means
meanSwitchR = mean(switchAfterReward);
meanSwitchU = mean(switchAfterNoReward);

% === PLOT ===
figure(88); clf;
set(gcf, 'Position', [-1650, 31, 1668, 960]); % Example position and size
ha = tight_subplot(3, 2, [0.06 0.06], [0.07 0.08], [0.08 0.03]);

% 1. Q-values
axes(ha(1));
plot(Q_total(1:end-1,1), 'r-', 'DisplayName', 'Q-Right'); hold on;
plot(Q_total(1:end-1,2), 'b-', 'DisplayName', 'Q-Left');
ylabel('Q'); title('Q-values'); legend; set(gca, 'XTick', []);

% 2. Choices
axes(ha(2));
plot(choices, 'ko');
ylim([0.5 2.5]); yticks([1 2]); yticklabels({'Right','Left'});
ylabel('Choice'); title('Subject Choices'); set(gca, 'XTick', []);

% 3. Rewards
axes(ha(3));
stem(rewards, 'filled', 'MarkerSize', 3);
ylim([-0.1 1.1]); ylabel('Rwd');
title('Reward Outcomes'); set(gca, 'XTick', []);

% 4. Choice Probabilities
axes(ha(4));
plot(p_right, 'k-', 'LineWidth', 1.2);
ylim([0 1]); ylabel('P(Right)');
yline(0, '--')
title('Model Choice Prob'); set(gca, 'XTick', []);

% 5. Switch rate after reward/no-reward
axes(ha(5));
bar([1 2], [meanSwitchR meanSwitchU], 0.5);
xticks([1 2]); xticklabels({'After Reward', 'After No Reward'});
yline(0, '--')
ylabel('P(Switch)'); ylim([0 1]); title('Choice Switching');

% 6. Fitted Parameters
axes(ha(6));
param_names = {};
param_vals = [];
for i = 1:length(activeParams)
    param = activeParams{i};
    if paramConfig.(param) == 1
        param_names{end+1} = param;
        switch param
            case 'alphaR', param_vals(end+1) = alphaR;
            case 'alphaU', param_vals(end+1) = alphaU;
            case 'beta', param_vals(end+1) = beta;
            case 'bias', param_vals(end+1) = bias;
            case 'sticky', param_vals(end+1) = stickiness;
            case 'wsls', param_vals(end+1) = wsls;
        end
    end
end
scatter(1:length(param_vals), param_vals, 60, 'ko', 'LineWidth', 2);
xlim([0.5, length(param_vals)+0.5]); ylim padded;
xticks(1:length(param_vals)); xticklabels(param_names);
yline(0, '--')
ylabel('Value'); title('Fitted Parameters');

end

function [paramVals1C, paramVals2C, paramVals1S, paramVals2S, paramNames] = get_plot_params(sub, conf, vol, paramStorage, paramConfig, activeParams)
% Helper function to get parameter values for plotting based on which parameters are enabled
% Returns parameter values for collapsed and session-wise data for both control and stimulation conditions

% Initialize parameter arrays and names
paramVals1C = [];  % Collapsed control
paramVals2C = [];  % Session-wise control  
paramVals1S = [];  % Collapsed stim
paramVals2S = [];  % Session-wise stim
paramNames = {};

% Get parameter values for active parameters only
for i = 1:length(activeParams)
    param = activeParams{i};
    if paramConfig.(param) == 1
        paramVals1C = [paramVals1C, paramStorage.(param){sub,conf,vol,1,1}];
        paramVals2C = [paramVals2C, paramStorage.(param){sub,conf,vol,1,2}];
        paramVals1S = [paramVals1S, paramStorage.(param){sub,conf,vol,2,1}];
        paramVals2S = [paramVals2S, paramStorage.(param){sub,conf,vol,2,2}];
        paramNames = [paramNames, {param}];
    end
end

end

function plotInteraction(T, var1, var2, paramName)
% Helper function to plot interaction effects
% T: data table, var1/var2: variable names, paramName: parameter name

% Get unique values for each variable
vals1 = unique(T.(var1));
vals2 = unique(T.(var2));

% Calculate means for each combination
means = zeros(length(vals1), length(vals2));
sems = zeros(length(vals1), length(vals2));

for i = 1:length(vals1)
    for j = 1:length(vals2)
        subset = T(T.(var1) == vals1(i) & T.(var2) == vals2(j), :);
        if ~isempty(subset)
            means(i, j) = mean(subset.Value);
            sems(i, j) = std(subset.Value) / sqrt(length(subset.Value));
        end
    end
end

% Create plot
hold on;
colors = lines(length(vals1));
for i = 1:length(vals1)
    errorbar(vals2, means(i, :), sems(i, :), 'o-', 'Color', colors(i, :), 'LineWidth', 2, 'MarkerSize', 8);
end

xlabel(var2);
ylabel('Mean Value');
title(sprintf('%s: %s × %s Interaction', paramName, var1, var2));
legend(arrayfun(@(x) sprintf('%s = %d', var1, x), vals1, 'UniformOutput', false), 'Location', 'best');
grid on;
end

