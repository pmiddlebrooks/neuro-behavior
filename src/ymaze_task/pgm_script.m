%% Run constants.m to get codes, event markers, subject info
constants

%%  Trying to replicate code_for_figures_pgm section starting at line 172
% Load an example file to get subject count
% filename = load([paths.dropPath, 'ymaze/data/NoConfLowVolControl_withDTs.mat']); %load environment
fileCont = load([paths.dropPath, 'ymaze/data/NoConfLowVolStim_withDTs.mat']); %load environment

nSubj = length(unique(cell2mat(fileCont.overlap(3,:))));

%%
fileCont = load([paths.dropPath, 'ymaze/data/NoConfLowVolControl_withDTs.mat']); %load environment

subj = unique(cell2mat(fileCont.overlap(3,:)))

fileCont = load([paths.dropPath, 'ymaze/data/NoConfLowVolStim_withDTs.mat']); %load environment

subj = unique(cell2mat(fileCont.overlap(3,:)))

%% Make a big matrix of all the subjects data

subjectList = [d1Subj d2Subj daSubj];
nSubj = length(subjectList);


[nTrialL, nTrialR, turnsL, turnsR, turnsLCorr, turnsRCorr, recallL, recallR] = ...
    deal(cell(nSubj, 3, 2, 2));  % Subject X Conflict`````````````` X Volatility X Control/Stim
[accPref, accNonPref, propSwitch] = deal(nan(nSubj, 3, 2, 2));

% Define the valid event markers
validMarkers = 1:8;
rightDir = 1;
leftDir = 0;



% Loop through subjects
for sub = 1 : length(subjectList)
subjectList(sub)
    for conf = 1 : 3
        for vol = 1 : 2
            for contStim = 1 : 2
                filePath = get_ymaze_data(conf, vol, contStim);
                data = load(filePath);

                sessIdx = find(cell2mat(data.overlap(3,:)) == subjectList(sub));


                % Loop through the sessions for this subject
                for i = 1 : length(sessIdx)
                    iData = data.overlap{1,sessIdx(i)};

                    % Build a matrix of the rows of turn indications
                    emIdx = [];
                    for r = 2:size(iData, 1)
                        if iData(r, 2) ~= iData(r-1,2) && iData(r-1,2) == 0 %when a turn is detected...
                            emIdx = [emIdx; r];
                        end %ending "if" statement looking for a turn detection
                    end %ending "for" loop going through data

                    iEventMat = iData(emIdx,:);
                    eventCounts = accumarray(iEventMat(:,2), 1);


                    % Count left and right turns to determine mouse's perferred direction
                    turnsL{sub,conf,vol,contStim}(i) = sum(eventCounts(crl:iul));
                    turnsLCorr{sub,conf,vol,contStim}(i) = sum(eventCounts(crl:cul));
                    turnsR{sub,conf,vol,contStim}(i) = sum(eventCounts(crr:iur));
                    turnsRCorr{sub,conf,vol,contStim}(i) = sum(eventCounts(crr:cur));

                    recallL{sub,conf,vol,contStim}(i) = turnsLCorr{sub,conf,vol,contStim}(i) / (turnsLCorr{sub,conf,vol,contStim}(i) + turnsR{sub,conf,vol,contStim}(i) - turnsRCorr{sub,conf,vol,contStim}(i));
                    recallR{sub,conf,vol,contStim}(i) = turnsRCorr{sub,conf,vol,contStim}(i) / (turnsRCorr{sub,conf,vol,contStim}(i) + turnsL{sub,conf,vol,contStim}(i) - turnsLCorr{sub,conf,vol,contStim}(i));


                    % What percentage of trials during the session was the correct
                    % direaction right?
                    % Find start of each block
                    % blockEnds = [find(diff(iData(:,nBlockIdx)) == 1); size(iData, 1)];
                    % blockStarts = [1; 1 + find(diff(iData(:,nBlockIdx)) == 1)];
                    % blockIdxL = iData(blockEnds, corrIdx) == leftDir;
                    % blockIdxR = iData(blockEnds, corrIdx) == rightDir;
                    % nTrialL{sub,conf,vol,contStim}(i) = sum(iData(blockEnds(blockIdxL), nTrialIdx));
                    % nTrialR{sub,conf,vol,contStim}(i) = sum(iData(blockEnds(blockIdxR), nTrialIdx));
                    nTrialL{sub,conf,vol,contStim}(i) = sum([eventCounts([crl cul irr iur])]);
                    nTrialR{sub,conf,vol,contStim}(i) = sum([eventCounts([irl iul crr cur])]);


                    % % Which direction was preferred during this session
                    % if sum(eventCounts(1:4)) > sum(eventCounts(5:8))
                    % sessionPref(i) = sum(eventCounts(1:2)) / sum(eventCounts(1:4));
                    % sessionNonPref(i) = sum(eventCounts(5:6)) / sum(eventCounts(5:8));
                    % else
                    % sessionPref(i) = sum(eventCounts(5:6)) / sum(eventCounts(5:8));
                    % sessionNonPref(i) = sum(eventCounts(1:2)) / sum(eventCounts(1:4));
                    % end
                end
%                     recallL{sub,conf,vol,contStim}
% recallR{sub,conf,vol,contStim}

                % If you want a grand preferred/non-preferred direction for
                % each mouse...
                % if mean(recallL{sub,conf,vol,contStim}) > mean(recallR{sub,conf,vol,contStim})
                %     accPref(sub,conf,vol,contStim) = mean(recallL{sub,conf,vol,contStim});
                %     accNonPref(sub,conf,vol,contStim) = mean(recallR{sub,conf,vol,contStim});
                % else
                %     accPref(sub,conf,vol,contStim) = mean(recallR{sub,conf,vol,contStim});
                %     accNonPref(sub,conf,vol,contStim) = mean(recallL{sub,conf,vol,contStim});
                % end

                % If you want a session-wise preferred/non-preferred
                % direction for each mouse....
                accPref(sub,conf,vol,contStim) = mean(max([recallL{sub,conf,vol,contStim}; recallR{sub,conf,vol,contStim}], [], 1));
                accNonPref(sub,conf,vol,contStim) = mean(min([recallL{sub,conf,vol,contStim}; recallR{sub,conf,vol,contStim}], [], 1));

                % propSwitch(sub,conf,vol,contStim) = sum((recallL{sub,conf,vol,contStim} - recallR{sub,conf,vol,contStim}) > 0) / length(recallR{sub,conf,vol,contStim});
                propGreater = mean(recallL{sub,conf,vol,contStim} > recallR{sub,conf,vol,contStim});
                propSwitch(sub,conf,vol,contStim) = 1 - abs(propGreater - 0.5) * 2;
            end
        end
    end
end

% t-test for conditions
% [h(conf),p(conf)] = ttest(accNonPref(:,conf), accPref(:,conf));
%%
% Which volatility?
vol = 1;
% Some plotting constants
xVal = [1 2 3];
xDist = .13;

subjectSets = {d1Subj, d2Subj};
titles = {'D1 subjects', 'D2 subjects'};
volatilities = {'Low Volatility', 'High Volatility'};
singleSubColor = [.5 .5 .5];
d12StimColors = {[0 0 1], [1 0 0]};
colorPref = [1 0.07 0.65];
colorNonPref = [0.47 0.67 0.19];

figure(37); clf;
ha = tight_subplot(1, 2, [0.05 0.05], [0.15 0.1], [0.07 0.05]);  % [gap, lower margin, upper margin]
%
for d = 1:2
    sIdx = find(ismember(subjectList, subjectSets{d}));
    axes(ha(d)); hold on;

    for conf = 1 : 3
        for s = 1:length(sIdx)
            sub = sIdx(s);
            plot([xVal(conf) - 2*xDist xVal(conf) - xDist/2], [accPref(sub,conf,vol,1) accPref(sub,conf,vol,2)], 'color', singleSubColor);
            plot([xVal(conf) + xDist/2 xVal(conf) + 2*xDist], [accNonPref(sub,conf,vol,1) accNonPref(sub,conf,vol,2)], 'color', singleSubColor);
        end
        plot([xVal(conf) - 2*xDist xVal(conf) - xDist/2], [mean(accPref(sIdx,conf,vol,1)) mean(accPref(sIdx,conf,vol,2))], 'k', 'linewidth', 2)
        scatter(xVal(conf) - 2*xDist, mean(accPref(sIdx,conf,vol,1)), 100, 'MarkerFaceColor', colorPref, 'MarkerEdgeColor', 'none');
        e1 = errorbar(xVal(conf) - 2*xDist, mean(accPref(sIdx,conf,vol,1)),std(accPref(sIdx,conf,vol,1))/sqrt(length(accPref(sIdx,conf,vol,1))),std(accPref(sIdx,conf,vol,1))/sqrt(length(accPref(sIdx,conf,vol,1))),'LineStyle','none','LineWidth',2);  %no conf nonpref
        e1.Color = colorPref;  %color
        scatter(xVal(conf) - xDist/2, mean(accPref(sIdx,conf,vol,2)), 100, 'MarkerFaceColor', d12StimColors{d}, 'MarkerEdgeColor', 'none');
        e2 = errorbar(xVal(conf) - xDist/2, mean(accPref(sIdx,conf,vol,2)),std(accPref(sIdx,conf,vol,2))/sqrt(length(accPref(sIdx,conf,vol,2))),std(accPref(sIdx,conf,vol,2))/sqrt(length(accPref(sIdx,conf,vol,2))),'LineStyle','none','LineWidth',2);  %no conf nonpref
        e2.Color = d12StimColors{d};  %color

        plot([xVal(conf) + xDist/2 xVal(conf) + 2*xDist], [mean(accNonPref(sIdx,conf,vol,1)) mean(accNonPref(sIdx,conf,vol,2))], 'k', 'linewidth', 2)
        scatter(xVal(conf) + xDist/2, mean(accNonPref(sIdx,conf,vol,1)), 120, 'MarkerFaceColor', colorNonPref, 'MarkerEdgeColor', 'none');
        e3 = errorbar(xVal(conf) + xDist/2, mean(accNonPref(sIdx,conf,vol,1)),std(accNonPref(sIdx,conf,vol,1))/sqrt(length(accNonPref(sIdx,conf,vol,1))),std(accNonPref(sIdx,conf,vol,1))/sqrt(length(accNonPref(sIdx,conf,vol,1))),'LineStyle','none','LineWidth',2);  %no conf nonpref
        e3.Color = colorNonPref;  %color
        scatter(xVal(conf) + 2*xDist, mean(accNonPref(sIdx,conf,vol,2)), 120, 'MarkerFaceColor', d12StimColors{d}, 'MarkerEdgeColor', 'none');
        e4 = errorbar(xVal(conf) + 2*xDist, mean(accNonPref(sIdx,conf,vol,2)),std(accNonPref(sIdx,conf,vol,2))/sqrt(length(accNonPref(sIdx,conf,vol,2))),std(accNonPref(sIdx,conf,vol,2))/sqrt(length(accNonPref(sIdx,conf,vol,2))),'LineStyle','none','LineWidth',2);  %no conf nonpref
        e4.Color = d12StimColors{d};  %color
    end

    ylim([.4 .85])
    XTickLabel = ({'No','Low','High'}); %x label
        set(gca,'xtick',xVal,'xticklabel',XTickLabel,'box','off','FontSize',20) %y lim to 0 aesthetics
    if d == 1
    ylabel('True Positive Rate') %y label
    end
    xlabel('Conflict Level')
    set(ha(d), 'YTickLabelMode', 'auto');  % Enable Y-axis labels
    title(titles{d})
end

sgtitle(volatilities{vol})

%%
figure(35); clf; hold on
for conf = 1 : 3
    scatter(conf,propSwitch(:,conf), 80, 'LineWidth',2)
end
xlim([.9 3.1])
ylim([-.1 1.1])
ylabel('Switching perferred/non (norm)')
XTickLabel = ({'No','Low','High'}); %x label
set(gca,'xtick',xVal,'xticklabel',XTickLabel,'box','off','FontSize',14) %y lim to 0 aesthetics
title('Preferred Direction Switch Rates')
%%
figure(25); clf; hold all;
% Some plotting constants
xVal = [1 2 3];
xDist = .3;


for conf = 1 : 3
    for sub = 1:length(allSubj)
        plot([xVal(conf) - xDist xVal(conf) + xDist], [accNonPref(sub, conf) accPref(sub, conf)], 'color', [.7 .7 .7]);
    end
    plot([xVal(conf) - xDist xVal(conf) + xDist], [mean(accNonPref(:,conf)) mean(accPref(:,conf))], 'k', 'linewidth', 2)
    scatter(xVal(conf) - xDist, mean(accNonPref(:,conf)), 100, 'MarkerFaceColor', [0.47 0.67 0.19], 'MarkerEdgeColor', 'none');
    e1 = errorbar(xVal(conf) - xDist, mean(accNonPref(:,conf)),std(accNonPref(:,conf))/sqrt(length(accNonPref(:,conf))),std(accNonPref(:,conf))/sqrt(length(accNonPref(:,conf))),'LineStyle','none','LineWidth',2);  %no conf nonpref
    e1.Color = '[0.47 0.67 0.19]';  %color
    scatter(xVal(conf) + xDist, mean(accPref(:,conf)), 120, 'MarkerFaceColor', [1 0.07 0.65], 'MarkerEdgeColor', 'none');
    e2 = errorbar(xVal(conf) + xDist, mean(accPref(:,conf)),std(accPref(:,conf))/sqrt(length(accPref(:,conf))),std(accPref(:,conf))/sqrt(length(accPref(:,conf))),'LineStyle','none','LineWidth',2);  %no conf nonpref
    e2.Color = '[1 0.07 0.65]';  %color
end

xlim([xVal(1) - 2*xDist, xVal(end) + 2*xDist])
ylim([.45 .85])
XTickLabel = ({'No','Low','High'}); %x label
%set(gca,'xtick',[1 1.1 1.2],'xticklabel',XTickLabel,'ytick',[0.5 0.6 0.7
%0.8],'box','off','FontSize',20) %original aesthetics
set(gca,'xtick',xVal,'xticklabel',XTickLabel,'box','off','FontSize',20) %y lim to 0 aesthetics
ylabel('True Positive Rate') %y label
legend({'Non-Pref','Pref'},'AutoUpdate','off') %legend

%%

if sum(turnsL) > sum(turnsR)
    accPref = sum(turnsLCorr) / sum(turnsL);
    accNonPref = sum(turnsRCorr) / sum(turnsR);
else
    accPref = sum(turnsRCorr) / sum(turnsR);
    accNonPref = sum(turnsLCorr) / sum(turnsL);
end

[nTrialL, nTrialR, turnsL, turnsR, turnsLCorr ./ turnsL, turnsRCorr ./ turnsR, recallL, recallR]
% accPref
% accNonPref

[nTrialL ./ (nTrialL + nTrialR), turnsL ./ (turnsL + turnsR)]

%%
