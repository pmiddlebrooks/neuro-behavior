% This script analyzes win-stay lose-switch behavior across conflict, 
% volatility, and stimulation (d1 and d2), across sessions within subjects 
% and across subjects.

%% Run constants.m to get codes, event markers, subject info

addpath(fileparts(matlab.desktop.editor.getActiveFilename));
constants
paths = get_paths;

% Column indices in sessInfo matrix
subIdx = 1;
confIdx = 2;
volIdx = 3;
stimIdx = 4;
winStayIdx = 5;
loseSwitchIdx = 6;
winSwitchIdx = 7;
loseStayIdx = 8;

%% Initialize variables and settings

plotFlag = 0;
maxRewards = 120;
cutoffFlag = 0;

confRange = 1:3;
volRange = 1:2;
contStimRange = 1:2;

% subjectList = [d1Subj d2Subj daSubj];
subjectList = [d1Subj d2Subj];
d1Idx = find(ismember(subjectList, d1Subj));
d2Idx = find(ismember(subjectList, d2Subj));

% Initialize arrays for WSLS measures
% Structure: {subject, conflict, volatility, stimControl, 1=collapsed, 2=session-wise}
[winStay, loseSwitch, winSwitch, loseStay] = deal(cell(length(subjectList), length(confRange), length(volRange), length(contStimRange), 2));

% Initialize arrays for subject means
[winStayMean, loseSwitchMean, winSwitchMean, loseStayMean] = ...
    deal(nan(length(subjectList), length(confRange), length(volRange), length(contStimRange)));

sessInfoCollapse = [];
sessInfo = [];

%% Main analysis loop

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
                        iEventMat = [iEventMat; iData(emIdx(1:cutoffTrial),:)];
                    else
                        iEventMat = [iEventMat; iData(emIdx,:)];
                    end
                end

                % Calculate WSLS measures for collapsed data
                [choices, rewards] = deal(zeros(size(iEventMat, 1), 1));
                choices(~ismember(iEventMat(:,2), [crl cul irl, iul])) = 1; % right turns are 1
                choices(ismember(iEventMat(:,2), [crl cul irl, iul])) = 2; % left turns are 2
                rewards(ismember(iEventMat(:,2), [cul iul cur iur])) = 0; % unrewarded are 0
                rewards(ismember(iEventMat(:,2), [crl irl crr irr])) = 1; % rewarded are 1

                % Calculate WSLS measures
                [winStayVal, loseSwitchVal, winSwitchVal, loseStayVal] = calculate_wsls(choices, rewards);

                % Store collapsed results
                winStay{sub, conf, vol, contStim, 1} = winStayVal;
                loseSwitch{sub, conf, vol, contStim, 1} = loseSwitchVal;
                winSwitch{sub, conf, vol, contStim, 1} = winSwitchVal;
                loseStay{sub, conf, vol, contStim, 1} = loseStayVal;

                % Add to collapsed session info
                sessInfoCollapse = [sessInfoCollapse; [data.overlap{3,sessIdx(1)}, conf, vol, contStim-1,...
                    winStayVal, loseSwitchVal, winSwitchVal, loseStayVal]];

                fprintf('Collapsed WSLS results:\n');
                fprintf('  Win-Stay = %.3f\n', winStayVal);
                fprintf('  Lose-Switch = %.3f\n', loseSwitchVal);
                fprintf('  Win-Switch = %.3f\n', winSwitchVal);
                fprintf('  Lose-Stay = %.3f\n', loseStayVal);

                % ==============================================
                % Session-wise for each subject per condition
                [iWinStay, iLoseSwitch, iWinSwitch, iLoseStay] = deal(nan(length(sessIdx), 1));

                for iSess = 1:length(sessIdx)
                    sessNum = sessIdx(iSess);
                    iData = data.overlap{1,sessNum};

                    % Remove first and last blocks
                    iBlock2Start = find(iData(:,9) == 2, 1);
                    iLastBlockStart = find(iData(:,9) == max(iData(:,9)), 1);
                    iData = iData(iBlock2Start:iLastBlockStart-1,:);

                    % Build a matrix of the rows of turn indications
                    emIdx = [];
                    for row = 2:size(iData, 1)
                        if iData(row, 2) ~= iData(row-1,2) && iData(row-1,2) == 0 && ismember(iData(row,2), 1:8) %when a turn is detected...
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

                    % Calculate choices and rewards
                    [choices, rewards] = deal(nan(size(iEventMat, 1), 1));
                    choices(ismember(iEventMat(:,2), [crr cur irr, iur])) = 1; % right turns are 1
                    choices(ismember(iEventMat(:,2), [crl cul irl, iul])) = 2; % left turns are 2
                    rewards(ismember(iEventMat(:,2), [cul iul cur iur])) = 0; % unrewarded are 0
                    rewards(ismember(iEventMat(:,2), [crl irl crr irr])) = 1; % rewarded are 1

                    % Calculate WSLS measures for this session
                    [iWinStay(iSess), iLoseSwitch(iSess), iWinSwitch(iSess), iLoseStay(iSess)] = calculate_wsls(choices, rewards);

                    % Add to session info
                    sessInfo = [sessInfo; [data.overlap{3,sessNum}, conf, vol, contStim-1,...
                        iWinStay(iSess), iLoseSwitch(iSess), iWinSwitch(iSess), iLoseStay(iSess)]];
                end

                % Store session-wise results
                winStay{sub, conf, vol, contStim, 2} = iWinStay;
                loseSwitch{sub, conf, vol, contStim, 2} = iLoseSwitch;
                winSwitch{sub, conf, vol, contStim, 2} = iWinSwitch;
                loseStay{sub, conf, vol, contStim, 2} = iLoseStay;

                % Calculate means across sessions
                winStayMean(sub, conf, vol, contStim) = mean(iWinStay);
                loseSwitchMean(sub, conf, vol, contStim) = mean(iLoseSwitch);
                winSwitchMean(sub, conf, vol, contStim) = mean(iWinSwitch);
                loseStayMean(sub, conf, vol, contStim) = mean(iLoseStay);

                fprintf('Session-wise means:\n');
                fprintf('  Win-Stay = %.3f\n', winStayMean(sub, conf, vol, contStim));
                fprintf('  Lose-Switch = %.3f\n', loseSwitchMean(sub, conf, vol, contStim));
                fprintf('  Win-Switch = %.3f\n', winSwitchMean(sub, conf, vol, contStim));
                fprintf('  Lose-Stay = %.3f\n', loseStayMean(sub, conf, vol, contStim));

            end

            if plotFlag
                figure(subjectList(sub)); clf;
                set(gcf, 'Position', [-1600, 166, 734, 777]);

                ha = tight_subplot(2, 2, [0.07 0.05], [0.05 0.1]);

                % Get WSLS values for plotting
                [wslsVals1C, wslsVals2C, wslsVals1S, wslsVals2S, wslsNames] = get_wsls_plot_params(sub, conf, vol, winStay, loseSwitch, winSwitch, loseStay);

                rJitterC = (rand(length(wslsVals2C), 1)) * 0.2;
                rJitterS = (rand(length(wslsVals2S), 1)) * 0.2;
                xScatterC = (1:size(wslsVals2C, 2)) + rJitterC;
                xScatterS = (1:size(wslsVals2S, 2)) + rJitterS;

                yMax = .2 + max(max([wslsVals1C; wslsVals2C; wslsVals1S; wslsVals2S]));
                yMin = -.2 + min(min([wslsVals1C; wslsVals2C; wslsVals1S; wslsVals2S]));

                if ~isempty(wslsVals1C)
                    axes(ha(1))
                    set(ha(1), 'YTickLabelMode', 'auto');
                    scatter(1:length(wslsVals1C), wslsVals1C, 80, 'LineWidth',2)
                    yline(0, '--')
                    xlim([.5 length(wslsVals1C)+.5])
                    ylim([yMin yMax])
                    xticks(1:length(wslsVals1C))
                    xticklabels(wslsNames)
                    title('Collapsed sessions Control')
                end

                if ~isempty(wslsVals1S)
                    axes(ha(2))
                    set(ha(2), 'YTickLabelMode', 'auto');
                    scatter(1:length(wslsVals1S), wslsVals1S, 80, 'LineWidth',2)
                    yline(0, '--')
                    xlim([.5 length(wslsVals1S)+.5])
                    ylim([yMin yMax])
                    xticks(1:length(wslsVals1S))
                    xticklabels(wslsNames)
                    title('Collapsed sessions Stim')
                end

                if ~isempty(wslsVals2C)
                    axes(ha(3))
                    hold on;
                    set(ha(3), 'YTickLabelMode', 'auto');
                    scatter(xScatterS, wslsVals2C, 80, 'LineWidth',2)
                    yline(0, '--')
                    xlim([.5 size(wslsVals2C, 2)+.5])
                    ylim([yMin yMax])
                    xticks(1:size(wslsVals2C, 2))
                    xticklabels(wslsNames)
                    title('Per Session Control')
                end

                if ~isempty(wslsVals2S)
                    axes(ha(4))
                    hold on;
                    set(ha(4), 'YTickLabelMode', 'auto');
                    scatter(xScatterS, wslsVals2S, 80, 'LineWidth',2)
                    yline(0, '--')
                    xlim([.5 size(wslsVals2S, 2)+.5])
                    ylim([yMin yMax])
                    xticks(1:size(wslsVals2S, 2))
                    xticklabels(wslsNames)
                    title('Per Session Stim')
                end

                tText = sprintf('WSLS Analysis Subj %d Conf %d Vol %d', subjectList(sub), conf, vol);
                sgtitle(tText)
                print('-dpng', fullfile(paths.dropPath, 'ymaze/', tText))
            end
        end
    end
end

%% Linear Mixed Effects Model: Session-wise Analysis Across All Subjects and Sessions

% Prepare arrays to collect data
wslsParamNames = {'winStay', 'loseSwitch', 'winSwitch', 'loseStay'};

allData = struct();
for p = 1:length(wslsParamNames)
    allData.(wslsParamNames{p}) = [];
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
                nSess = length(winStay{sub, conf, vol, contStim, 2});
                for iSess = 1:nSess
                    % For each parameter, add a row to the table
                    for p = 1:length(wslsParamNames)
                        param = wslsParamNames{p};
                        switch param
                            case 'winStay', val = winStay{sub, conf, vol, contStim, 2}(iSess);
                            case 'loseSwitch', val = loseSwitch{sub, conf, vol, contStim, 2}(iSess);
                            case 'winSwitch', val = winSwitch{sub, conf, vol, contStim, 2}(iSess);
                            case 'loseStay', val = loseStay{sub, conf, vol, contStim, 2}(iSess);
                        end
                        if isnan(val), continue; end
                        % Add row to cell array
                        allData.(param) = [allData.(param); {val, subjectList(sub), contStim-1, conf, vol, group}];
                    end
                end
            end
        end
    end
end

% Convert cell arrays to tables
for p = 1:length(wslsParamNames)
    param = wslsParamNames{p};
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

for p = 1:length(wslsParamNames)
    param = wslsParamNames{p};
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
for p = 1:length(wslsParamNames)
    param = wslsParamNames{p};
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
pValueMatrix = nan(length(wslsParamNames), 8); % 8 possible effects
effectNames = {'Intercept', 'StimControl', 'Conflict', 'Volatility', 'StimControl:Conflict', 'StimControl:Volatility', 'Conflict:Volatility', 'StimControl:Conflict:Volatility'};

for p = 1:length(wslsParamNames)
    param = wslsParamNames{p};
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
yticks(1:length(wslsParamNames));
yticklabels(wslsParamNames);

% Add p-value text
for p = 1:length(wslsParamNames)
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

title('P-values for WSLS Linear Mixed Effects Model Results');
xlabel('Effects');
ylabel('Parameters');

% 3. Group-specific analysis
fprintf('\n\n==========================================\n');
fprintf('GROUP-SPECIFIC ANALYSIS (D1 vs D2)\n');
fprintf('==========================================\n');

% Fit separate models for each group
groupResults = struct();
for group = {'d1', 'd2'}
    groupName = group{1};
    fprintf('\n--- %s Subjects ---\n', upper(groupName));
    
    for p = 1:length(wslsParamNames)
        param = wslsParamNames{p};
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
            fprintf('\n  %s:\n', param);
            for i = 1:height(groupAnova(sigGroupIdx, :))
                effect = groupAnova.Term{sigGroupIdx(i)};
                pval = groupAnova.pValue(sigGroupIdx(i));
                fprintf('    %s: p = %.4f\n', effect, pval);
            end
        end
    end
end

% 4. Effect size analysis
fprintf('\n\n==========================================\n');
fprintf('EFFECT SIZE ANALYSIS\n');
fprintf('==========================================\n');

for p = 1:length(wslsParamNames)
    param = wslsParamNames{p};
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

% 5. Create interaction plots for significant interactions
figure('Position', [200, 200, 1400, 1000]);
plotIdx = 1;

for p = 1:length(wslsParamNames)
    param = wslsParamNames{p};
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
    sgtitle('Significant WSLS Interaction Effects');
end

% Save results summary
save(fullfile(paths.dropPath, 'ymaze/', 'WSLS_Results_Summary.mat'), 'lmeResults', 'significantEffects', 'groupResults', 'sigTable');
fprintf('\nResults saved to: %s\n', fullfile(paths.dropPath, 'ymaze/', 'WSLS_Results_Summary.mat'));

%% Individual subject stats: Store signrank test results
for iSub = 1:length(subjectList)
    for iConf = 1:3
        for iVol = 1:2
            % Compare contStim 1 vs 2 for each parameter
            [p_winStay(iSub,iConf,iVol), h_winStay(iSub,iConf,iVol)] = ranksum(winStay{iSub,iConf,iVol,1,2}, winStay{iSub,iConf,iVol,2,2});
            [p_loseSwitch(iSub,iConf,iVol), h_loseSwitch(iSub,iConf,iVol)] = ranksum(loseSwitch{iSub,iConf,iVol,1,2}, loseSwitch{iSub,iConf,iVol,2,2});
            [p_winSwitch(iSub,iConf,iVol), h_winSwitch(iSub,iConf,iVol)] = ranksum(winSwitch{iSub,iConf,iVol,1,2}, winSwitch{iSub,iConf,iVol,2,2});
            [p_loseStay(iSub,iConf,iVol), h_loseStay(iSub,iConf,iVol)] = ranksum(loseStay{iSub,iConf,iVol,1,2}, loseStay{iSub,iConf,iVol,2,2});
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
    figure;
    ha = tight_subplot(2, 3, [0.1 0.1], [0.15 0.1], [0.1 0.05]); % [gap_h gap_w, marg_h, marg_w]

    currIdx = subjectGroups{group};
    for vol = 1:2
        for conf = 1:3
            axes(ha((vol-1)*3 + conf));
            hold on;

            % For each subject in the group
            for s = 1:length(currIdx)
                % Add random jitter to x positions
                xJitter = (rand(1,4)-0.5) * jitterAmount;

                % Plot p-values for each parameter with jitter
                paramIdx = 1;
                scatter(paramIdx+xJitter(paramIdx), p_winStay(currIdx(s),conf,vol), 50, 'o', 'LineWidth', 2)
                paramIdx = paramIdx + 1;
                scatter(paramIdx+xJitter(paramIdx), p_loseSwitch(currIdx(s),conf,vol), 50, 'o', 'LineWidth', 2)
                paramIdx = paramIdx + 1;
                scatter(paramIdx+xJitter(paramIdx), p_winSwitch(currIdx(s),conf,vol), 50, 'o', 'LineWidth', 2)
                paramIdx = paramIdx + 1;
                scatter(paramIdx+xJitter(paramIdx), p_loseStay(currIdx(s),conf,vol), 50, 'o', 'LineWidth', 2)
            end

            % Add significance line
            yline(0.05, '--r')

            % Formatting
            xlim([0.5 4.5])
            ylim([0 1])
            xticks(1:4)
            xticklabels({'winStay', 'loseSwitch', 'winSwitch', 'loseStay'})
            ylabel('p-value')
            title(sprintf('Vol %d Conf %d', vol, conf))

            if vol == 1 && conf == 1
                sgtitle(groupNames{group})
            end
        end
    end
end

%% Calculate change between controls and stimulation for all conditions/subjects
[dPctWinStay, dPctLoseSwitch, dPctWinSwitch, dPctLoseStay] = ...
    deal(nan(length(subjectList), length(confRange), length(volRange)));
[pValWinStay, pValLoseSwitch, pValWinSwitch, pValLoseStay] = ...
    deal(nan(length(confRange), length(volRange)));

for conf = confRange
    for vol = volRange
        for sub = 1 : length(subjectList)
            dPctWinStay(sub, conf, vol) = (winStayMean(sub,conf,vol,2) - winStayMean(sub,conf,vol,1));
            dPctLoseSwitch(sub, conf, vol) = (loseSwitchMean(sub,conf,vol,2) - loseSwitchMean(sub,conf,vol,1));
            dPctWinSwitch(sub, conf, vol) = (winSwitchMean(sub,conf,vol,2) - winSwitchMean(sub,conf,vol,1));
            dPctLoseStay(sub, conf, vol) = (loseStayMean(sub,conf,vol,2) - loseStayMean(sub,conf,vol,1));
        end
        pValWinStay(conf, vol) = ranksum(dPctWinStay(1:4, conf, vol), dPctWinStay(5:8, conf, vol));
        pValLoseSwitch(conf, vol) = ranksum(dPctLoseSwitch(1:4, conf, vol), dPctLoseSwitch(5:8, conf, vol));
        pValWinSwitch(conf, vol) = ranksum(dPctWinSwitch(1:4, conf, vol), dPctWinSwitch(5:8, conf, vol));
        pValLoseStay(conf, vol) = ranksum(dPctLoseStay(1:4, conf, vol), dPctLoseStay(5:8, conf, vol));

        fprintf('Conf %d Vol %d: winStay: %.3f\tloseSwitch: %.3f\twinSwitch: %.3f\tloseStay: %.3f\t\n', ...
            conf, vol, pValWinStay(conf, vol), pValLoseSwitch(conf, vol), pValWinSwitch(conf, vol), pValLoseStay(conf, vol))
    end
end

%%
% Parameters
params = {'WinStay', 'LoseSwitch', 'WinSwitch', 'LoseStay'};
pVals = {pValWinStay, pValLoseSwitch, pValWinSwitch, pValLoseStay};
thresh = 0.05;

% Conflict and Volatility axis
confLevels = 1:3;
volLevels = 1:2;

% Use a lighter colormap
cmap = parula(256);
cmap = cmap .^ 0.7;  % brighten

% Plot settings
figure(2); clf
tiledlayout(1, numel(params), 'Padding', 'compact', 'TileSpacing', 'compact');

for i = 1:numel(params)
    pMat = pVals{i};  % 3 (conflict) x 2 (volatility)

    nexttile;
    imagesc(volLevels, confLevels, pMat);
    set(gca, 'YDir', 'normal');
    colormap(cmap);
    if i == numel(params)
        colorbar; % only on last subplot
    end
    caxis([0 1]);
    title(params{i});
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

%% Helper Functions

function [winStay, loseSwitch, winSwitch, loseStay] = calculate_wsls(choices, rewards)
% Calculate win-stay lose-switch measures from choice and reward sequences
% Inputs:
% - choices: vector (1 = right, 2 = left)
% - rewards: vector (0 or 1)
% Outputs:
% - winStay: probability of staying after a win
% - loseSwitch: probability of switching after a loss
% - winSwitch: probability of switching after a win
% - loseStay: probability of staying after a loss

nTrials = length(choices);
if nTrials < 2
    winStay = nan; loseSwitch = nan; winSwitch = nan; loseStay = nan;
    return;
end

% Initialize counters
winStayCount = 0; loseSwitchCount = 0; winSwitchCount = 0; loseStayCount = 0;
winTotal = 0; loseTotal = 0;

% Loop through trials (starting from trial 2)
for t = 2:nTrials
    prevChoice = choices(t-1);
    prevReward = rewards(t-1);
    currChoice = choices(t);
    
    % Check if previous trial was a win
    if prevReward == 1
        winTotal = winTotal + 1;
        if currChoice == prevChoice
            winStayCount = winStayCount + 1;
        else
            winSwitchCount = winSwitchCount + 1;
        end
    % Check if previous trial was a loss
    elseif prevReward == 0
        loseTotal = loseTotal + 1;
        if currChoice == prevChoice
            loseStayCount = loseStayCount + 1;
        else
            loseSwitchCount = loseSwitchCount + 1;
        end
    end
end

% Calculate probabilities
if winTotal > 0
    winStay = winStayCount / winTotal;
    winSwitch = winSwitchCount / winTotal;
else
    winStay = nan;
    winSwitch = nan;
end

if loseTotal > 0
    loseSwitch = loseSwitchCount / loseTotal;
    loseStay = loseStayCount / loseTotal;
else
    loseSwitch = nan;
    loseStay = nan;
end

end

function [wslsVals1C, wslsVals2C, wslsVals1S, wslsVals2S, wslsNames] = get_wsls_plot_params(sub, conf, vol, winStay, loseSwitch, winSwitch, loseStay)
% Helper function to get WSLS values for plotting
% Returns WSLS values for collapsed and session-wise data for both control and stimulation conditions

% Initialize arrays
wslsVals1C = [];  % Collapsed control
wslsVals2C = [];  % Session-wise control  
wslsVals1S = [];  % Collapsed stim
wslsVals2S = [];  % Session-wise stim
wslsNames = {};

% Add WSLS measures
wslsVals1C = [wslsVals1C, winStay{sub,conf,vol,1,1}];
wslsVals2C = [wslsVals2C, winStay{sub,conf,vol,1,2}];
wslsVals1S = [wslsVals1S, winStay{sub,conf,vol,2,1}];
wslsVals2S = [wslsVals2S, winStay{sub,conf,vol,2,2}];
wslsNames = [wslsNames, {'winStay'}];

wslsVals1C = [wslsVals1C, loseSwitch{sub,conf,vol,1,1}];
wslsVals2C = [wslsVals2C, loseSwitch{sub,conf,vol,1,2}];
wslsVals1S = [wslsVals1S, loseSwitch{sub,conf,vol,2,1}];
wslsVals2S = [wslsVals2S, loseSwitch{sub,conf,vol,2,2}];
wslsNames = [wslsNames, {'loseSwitch'}];

wslsVals1C = [wslsVals1C, winSwitch{sub,conf,vol,1,1}];
wslsVals2C = [wslsVals2C, winSwitch{sub,conf,vol,1,2}];
wslsVals1S = [wslsVals1S, winSwitch{sub,conf,vol,2,1}];
wslsVals2S = [wslsVals2S, winSwitch{sub,conf,vol,2,2}];
wslsNames = [wslsNames, {'winSwitch'}];

wslsVals1C = [wslsVals1C, loseStay{sub,conf,vol,1,1}];
wslsVals2C = [wslsVals2C, loseStay{sub,conf,vol,1,2}];
wslsVals1S = [wslsVals1S, loseStay{sub,conf,vol,2,1}];
wslsVals2S = [wslsVals2S, loseStay{sub,conf,vol,2,2}];
wslsNames = [wslsNames, {'loseStay'}];

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