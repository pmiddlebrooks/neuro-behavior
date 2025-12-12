%% task introduction - example behavior

% filename = load('C:\Users\Buckethead\OneDrive\Documents\MATLAB\example data\NoConfLowVolControl_withDTs.mat'); %load environment
% filename = load('E:\Projects\ymaze\data\NoConfLowVolControl_withDTs.mat'); %load environment
% filename = load('~/Projects/ymaze/data/HighConfHighVolStim_withDTs.mat'); %load environment
filename = load([paths.dropPath, 'ymaze/data/NoConfLowVolControl_withDTs.mat']); %load environment
% filename = load([paths.dropPath, 'ymaze/data/HighConfHighVolStim_withDTs.mat']); %load environment
overlap = filename.overlap; %pull out cells
data = overlap{1,35}; %pull out raw data
Corrects = [1 2 5 6]; %EMs for correct turns
Incorrects = [3 4 7 8 9]; %EMs for incorrect turns
cor = []; %create a vector that will be filled for correct turns
wantedEM = []; %wanted EM
wem=1; %format
for WEM = 2:length(data) %go through data
    if data(WEM,2) ~= data(WEM-1,2) && data(WEM-1,2) == 0 %when a turn is detected...
        wantedEM(wem) = data(WEM,2); %mark the EM
        wem=wem+1; %format
    end %ending "if" statement looking for a turn detection
end %ending "for" loop going through data
baitedDir = data(:,4); %identifying the "correct" direction: 0 means left; 1 means right
baitedDir(:,2) = data(:,2); %filling second column with EMs
correctDir = []; %empty vector that will be filled with correct directions for each trial
for w = 2:length(baitedDir) %setting w to go through baitedDir starting at 2
    if baitedDir(w,2)~=baitedDir(w-1,2) && baitedDir(w,2)~=0 && baitedDir(w,2)~=11 && baitedDir(w,2)~=12 %if the EM for a trial indicates a turn...
        correctDir = [correctDir,baitedDir(w,1)]; %...fill in the correct direction for that trial #
    end %ending "if" statement
end %ending "for" loop
cor = wantedEM; %fill "cor" with the EMs associated with turns
cor(ismember(cor,Corrects)) = 1; %set any correct turns to a 1 value
cor(ismember(cor,Incorrects)) = 0; %set any incorrect turns to a 0 value

f = figure; plot(movmean(cor,[4 0]),'k','LineWidth',2); %plot the moving average for correct
set(gca,'box','off') %makes it look nicer
ylim([-0.05 1.13]) %set y axis to be able to see trial markers/bars
hold on %set plot to add the trial markers

LR = [1 3]; %EMs for left rewarded - for reference
LUR = [2 4]; %EMs for left unrewarded - for reference
RR = [5 7]; %EMs for right rewarded - for reference
RUR = [6 8]; %EMs for right unrewarded - for reference

for j = 1:length(wantedEM) %going through all values in wantedEM
    if (wantedEM(j) == 1) %LR value
        line([j,j],[1,1.05],'Color','[0.47 0.67 0.19]','LineWidth',0.75); %LR line
    elseif (wantedEM(j) == 3) %LR value
        line([j,j],[1,1.05],'Color','[0.47 0.67 0.19]','LineWidth',0.75); %LR line
    elseif (wantedEM(j) == 2) %LUR value
        line([j,j],[1.025,1.05],'Color','[0.47 0.67 0.19]','LineWidth',0.5); %LUR line
    elseif (wantedEM(j) == 4) %LUR value
        line([j,j],[1.025,1.05],'Color','[0.47 0.67 0.19]','LineWidth',0.5); %LUR line
    elseif (wantedEM(j) == 5) %RR value
        line([j,j],[-0.05,0],'Color','m','LineWidth',0.75); %RR line
    elseif (wantedEM(j) == 7) %RR value
        line([j,j],[-0.05,0],'Color','m','LineWidth',0.75); %RR line
    elseif (wantedEM(j) == 6) %RUR value
        line([j,j],[-0.05,-0.025],'Color','m','LineWidth',0.5); %RUR line
    elseif (wantedEM(j) == 8) %RUR value
        line([j,j],[-0.05,-0.025],'Color','m','LineWidth',0.5); %RUR line
    end %ending "if" statements
end %ending "for" loop
for r = 1:length(correctDir) %going through the correct direction vector
    if (correctDir(r) == 0) %if the correct direction is LEFT...
        line([r,r+1],[1.1,1.1],'Color','[0.47 0.67 0.19]','LineWidth',12); %...plot a box on top in blue
    elseif (correctDir(r) == 1) %if the correct direction is RIGHT...
        line([r,r+1],[1.1,1.1],'Color','m','LineWidth',12); %...plot a box on the top in pink
    end %ending "if" statement"
end %ending "for" loop
hold on %holding the plot
for e = 2:length(correctDir) %going through the block direction vector
      if correctDir(e)~=correctDir(e-1) %if the block changes...
           xline(e-0.5,'k--') %...draw a vertical dashed line between those trials
      end %ending "if" statement for block changes
end %ending "for" loop going through block directions
xlabel('Trial') %x label
ylabel('Percent Correct') %y label
set(gca,'FontSize',20) %set font size

%% low vol control percent correct

f = figure; %figure
ctr = []; ydt = []; %space
y = [mean(min_no_c); mean(low_no_c); mean(high_no_c)]; %accuracy
a_vert = [std(min_no_c)/sqrt(length(min_no_c)); std(low_no_c)/sqrt(length(low_no_c)); std(high_no_c)/sqrt(length(high_no_c))]; %SEM
errorplus_vert=a_vert'; %positive SEM
errorminus_vert=errorplus_vert; %negative SEM
x = [1; 1.05; 1.1]; %x positions
a_horz = [0.04;0.04;0.04]/2; %horizontal bars
errorplus_horz=a_horz'; %positive horizontal bars
errorminus_horz=errorplus_horz; %negative horizontal bars
hBar = bar(x,y,'FaceColor',[1 1 1],'EdgeColor',[1 1 1]); %plot accuracy
for k1 = 1:size(y,2) %"for" loop going through accuracies
    ctr(k1,:) = bsxfun(@plus, hBar(k1).XData, hBar(k1).XOffset'); %ctr
    ydt(k1,:) = hBar(k1).YData;                    %ydt
end %ending "for" loop going through accuracies
minnoc = [0.66, 0.67, 0.72, 0.64, 0.74, 0.745, 0.65, 0.675, 0.725, 0.7, 0.63, 0.635, 0.625]; %individual mice
lownoc = [0.66, 0.67, 0.69, 0.62, 0.75, 0.74, 0.65, 0.655, 0.695, 0.645, 0.63, 0.665, 0.635]; %individual mice
highnoc = [0.58, 0.585, 0.605, 0.54, 0.65, 0.67, 0.565, 0.63, 0.595, 0.61, 0.57, 0.615, 0.6]; %individual mice
minnoc(2,:) = ctr(1,1); %center
lownoc(2,:) = ctr(1,2); %center
highnoc(2,:) = ctr(1,3); %center
hold on %hold fig
c = [0.5 0.5 0.5]; %color
scatter(minnoc(2,:),minnoc(1,:),50,c); %scatter individual
scatter(lownoc(2,:),lownoc(1,:),50,c); %scatter individual
scatter(highnoc(2,:),highnoc(1,:),50,c); %scatter individual
e1 = errorbar(x(1), y(1), errorminus_vert(1), errorplus_vert(1), 'LineStyle', 'none', 'LineWidth', 2); %no conf error bar
e1.Color = '[0.3 0.3 0.3]';  %color
hLine = line([x(1) - errorminus_horz(1), x(1) + errorplus_horz(1)], [y(1), y(1)], 'Color', get(e1, 'Color'), 'LineWidth', 2); %horizontal line
e2 = errorbar(x(2), y(2), errorminus_vert(2), errorplus_vert(2), 'LineStyle', 'none', 'LineWidth', 2); %low conf error bar
e2.Color = '[0.3 0.3 0.3]'; %color
hLine = line([x(2) - errorminus_horz(2), x(2) + errorplus_horz(2)], [y(2), y(2)], 'Color', get(e2, 'Color'), 'LineWidth', 2); %horizontal line
e3 = errorbar(x(3), y(3), errorminus_vert(3), errorplus_vert(3), 'LineStyle', 'none', 'LineWidth', 2); %high conf error bar
e3.Color = '[0.3 0.3 0.3]';  %color
hLine = line([x(3) - errorminus_horz(3), x(3) + errorplus_horz(3)], [y(3), y(3)], 'Color', get(e3, 'Color'), 'LineWidth', 2);  %horizontal line
%ylim([0.45 0.87]) %original y lim
ylim([0 0.87]) %ylim axis to 0
xlim([0.97 1.13]) %xlim
XTickLabel = ({'No','Low','High'}); %labels
%set(gca,'xtick',[1 1.05 1.1],'xticklabel',XTickLabel,'ytick',[0.5 0.6 0.7
%0.8],'box','off','FontSize',20) %original aesthetics
set(gca,'xtick',[1 1.05 1.1],'xticklabel',XTickLabel,'box','off','FontSize',20) %axis to 0 aesthetics
ylabel('% Correct') %y label
X(1) = ctr(1,2); X(2) = ctr(1,3); Y(1) = 0.77; Y(2) = 0.77; %sig bar
plot(X(1:2),Y,'LineWidth',1.5,'Color','k') %appearance
X(1) = ctr(1,1); X(2) = ctr(1,3); Y(1) = 0.8; Y(2) = 0.8; %sig bar
plot(X(1:2),Y,'LineWidth',1.5,'Color','k') %appearance

%% low vol control reward rate

f = figure; %figure
ctr = []; ydt = []; %space
y = [mean(RPM_noconflowvolctrl); mean(RPM_lowconflowvolctrl); mean(RPM_highconflowvolctrl)]; %rewards per min
a_vert = [std(RPM_noconflowvolctrl)/sqrt(length(RPM_noconflowvolctrl)); std(RPM_lowconflowvolctrl)/sqrt(length(RPM_lowconflowvolctrl)); std(RPM_highconflowvolctrl)/sqrt(length(RPM_highconflowvolctrl))]; %SEM
errorplus_vert=a_vert'; %positive SEM
errorminus_vert=errorplus_vert; %negative SEM
x = [1; 1.05; 1.1]; %x position
a_horz = [0.04;0.04;0.04]/2; %horizontal line
errorplus_horz=a_horz'; %positive horizontal
errorminus_horz=errorplus_horz; %negative horizontal
hBar = bar(x,y,'FaceColor',[1 1 1],'EdgeColor',[1 1 1]); %plot rewards
for k1 = 1:size(y,2) %"for" loop going through rewards per min
    ctr(k1,:) = bsxfun(@plus, hBar(k1).XData, hBar(k1).XOffset');      %ctr
    ydt(k1,:) = hBar(k1).YData;                     %ydt
end %ending "for" loop going through rewards
minnoc = noconflowvolctrl_individuals(1:12)'; %individuals
lownoc =  lowconflowvolctrl_individuals(1:12)'; %individuals
highnoc =  highconflowvolctrl_individuals(1:12)'; %individuals
minnoc(2,:) = ctr(1,1); %ctr
lownoc(2,:) = ctr(1,2); %ctr
highnoc(2,:) = ctr(1,3); %ctr
hold on %hold fig
c = [0.5 0.5 0.5]; %color
scatter(minnoc(2,:),minnoc(1,:),50,c); %scatter
scatter(lownoc(2,:),lownoc(1,:),50,c); %scatter
scatter(highnoc(2,:),highnoc(1,:),50,c); %scatter
e1 = errorbar(x(1), y(1), errorminus_vert(1), errorplus_vert(1), 'LineStyle', 'none', 'LineWidth', 2); %no conf
e1.Color = '[0.3 0.3 0.3]'; %color
hLine = line([x(1) - errorminus_horz(1), x(1) + errorplus_horz(1)], [y(1), y(1)], 'Color', get(e1, 'Color'), 'LineWidth', 2); %horiz line
e2 = errorbar(x(2), y(2), errorminus_vert(2), errorplus_vert(2), 'LineStyle', 'none', 'LineWidth', 2); %low conf
e2.Color = '[0.3 0.3 0.3]';  %color
hLine = line([x(2) - errorminus_horz(2), x(2) + errorplus_horz(2)], [y(2), y(2)], 'Color', get(e2, 'Color'), 'LineWidth', 2); %horiz line
e3 = errorbar(x(3), y(3), errorminus_vert(3), errorplus_vert(3), 'LineStyle', 'none', 'LineWidth', 2); %high conf
e3.Color = '[0.3 0.3 0.3]'; %color
hLine = line([x(3) - errorminus_horz(3), x(3) + errorplus_horz(3)], [y(3), y(3)], 'Color', get(e3, 'Color'), 'LineWidth', 2);  %horiz line
ylim([0 25]) %ylim
xlim([0.97 1.13]) %xlim
XTickLabel = ({'No','Low','High'}); %x labels
set(gca,'xtick',[1 1.05 1.1],'xticklabel',XTickLabel,'box','off','FontSize',20) %aesthetics
ylabel('Rewards per Min') %y label

%% low vol control performance by block type

f = figure; %figure
ctr = []; ydt = []; %space
y = [mean(min_no_left_percent) mean(min_no_right_percent); mean(low_no_left_percent) mean(low_no_right_percent); mean(high_no_left_percent) mean(high_no_right_percent)]; %performance by block type
a_vert = [std(min_no_left_percent)/sqrt(length(min_no_left_percent)) std(min_no_right_percent)/sqrt(length(min_no_right_percent)); std(low_no_left_percent)/sqrt(length(low_no_left_percent)) std(low_no_right_percent)/sqrt(length(low_no_right_percent)); std(high_no_left_percent)/sqrt(length(high_no_left_percent)) std(high_no_right_percent)/sqrt(length(high_no_right_percent))]; %SEM
errorplus_vert=a_vert'; %positive SEM
errorminus_vert=errorplus_vert; %negative SEM
x = [1; 1.1; 1.2]; %x position
a_horz = [0.01;0.01;0.01]; %horiz lines
errorplus_horz=a_horz'; %positive horiz
errorminus_horz=errorplus_horz; %negative horiz
hBar = bar(x,y,'FaceColor',[1 1 1],'EdgeColor',[1 1 1]); %plot performance by block type
for k1 = 1:size(y,2) %"for" loop going through performance data
    ctr(k1,:) = bsxfun(@plus, hBar(k1).XData, hBar(k1).XOffset');      %ctr
    ydt(k1,:) = hBar(k1).YData;                     %ydt
end %ending "for" loop going through performance data
minnoleft_percent = [0.63 0.6 0.69 0.58 0.7 0.73 0.64 0.64 0.67 0.67 0.63 0.6 0.55]; %no conf non pref indiviudals
minnoright_percent = [0.69 0.75 0.75 0.67 0.79 0.76 0.65 0.7 0.76 0.72 0.64 0.67 0.72]; %no conf pref individuals
lownoleft_percent = [0.66 0.59 0.68 0.6 0.67 0.73 0.64 0.6 0.65 0.64 0.6 0.64 0.55]; %low conf non pref individuals
lownoright_percent = [0.67 0.75 0.69 0.64 0.84 0.75 0.66 0.7 0.73 0.66 0.65 0.69 0.71]; %low conf pref individuals
highnoleft_percent = [0.54 0.52 0.56 0.53 0.56 0.62 0.48 0.53 0.52 0.59 0.56 0.61 0.5]; %high conf non pref invidiuals
highnoright_percent = [0.63 0.66 0.65 0.56 0.75 0.72 0.67 0.74 0.71 0.62 0.58 0.62 0.71]; %high conf pref individuals
minnoleft_percent(2,:) = ctr(1,1); %ctr
minnoright_percent(2,:) = ctr(2,1); %center
lownoleft_percent(2,:) = ctr(1,2); %center
lownoright_percent(2,:) = ctr(2,2); %center
highnoleft_percent(2,:) = ctr(1,3); %center
highnoright_percent(2,:) = ctr(2,3); %center
hold on %hold fig
for ugh = 1:13 %go through each mouse
    X(1) =  minnoleft_percent(2,ugh); %non pref X
    Y(1) =  minnoleft_percent(1,ugh); %nonpref Y
    X(2) =  minnoright_percent(2,ugh); %pref X
    Y(2) =  minnoright_percent(1,ugh); %pref Y
    plot(X(1:2),Y,'Color','[0.7 0.7 0.7]') %plot lines
end %ending "for" loop going through each mouse
for ugh = 1:13 %go through each mouse
    X(1) =  lownoleft_percent(2,ugh); %non pref X
    Y(1) =  lownoleft_percent(1,ugh); %nonpref Y
    X(2) =  lownoright_percent(2,ugh); %pref X
    Y(2) =  lownoright_percent(1,ugh); %pref Y
    plot(X(1:2),Y,'Color','[0.7 0.7 0.7]') %plot lines
end %ending "for" loop going through each mouse
for ugh = 1:13 %go through each mouse
    X(1) =  highnoleft_percent(2,ugh); %non pref X
    Y(1) =  highnoleft_percent(1,ugh); %nonpref Y
    X(2) =  highnoright_percent(2,ugh); %pref X
    Y(2) =  highnoright_percent(1,ugh); %pref Y
    plot(X(1:2),Y,'Color','[0.7 0.7 0.7]') %plot lines
end %ending "for" loop going through each mouse
X(1) = ctr(1,1); X(2) = ctr(2,1); Y(1) = y(1,1); Y(2) = y(1,2); %connect means
plot(X(1:2),Y,'LineWidth',1.5,'Color','k') %plot
X(1) = ctr(1,2); X(2) = ctr(2,2); Y(1) = y(2,1); Y(2) = y(2,2); %connect means
plot(X(1:2),Y,'LineWidth',1.5,'Color','k') %plot
X(1) = ctr(1,3); X(2) = ctr(2,3); Y(1) = y(3,1); Y(2) = y(3,2); %connect means
plot(X(1:2),Y,'LineWidth',1.5,'Color','k') %plot
e1 = errorbar(ctr(1,1), y(1,1),errorminus_vert(1,1),errorplus_vert(1,1),'LineStyle','none','LineWidth',2);  %no conf nonpref
e1.Color = '[0.47 0.67 0.19]';  %color
hLine = line([ctr(1,1) - errorminus_horz(1), ctr(1,1) + errorplus_horz(1)], [y(1,1), y(1,1)], 'Color', get(e1, 'Color'), 'LineWidth', 2); %horiz
e12 = errorbar(ctr(2,1), y(1,2),errorminus_vert(2,1),errorplus_vert(2,1),'LineStyle','none','LineWidth',2);  %no conf pref
e12.Color = '[1 0.07 0.65]';  %color
hLine = line([ctr(2,1) - errorminus_horz(1), ctr(2,1) + errorplus_horz(1)], [y(1,2), y(1,2)], 'Color', get(e12, 'Color'), 'LineWidth', 2); %horiz
e2 = errorbar(ctr(1,2), y(2,1),errorminus_vert(1,2),errorplus_vert(1,2),'LineStyle','none','LineWidth',2);  %low conf nonpref
e2.Color = '[0.47 0.67 0.19]';  %color
hLine = line([ctr(1,2) - errorminus_horz(1), ctr(1,2) + errorplus_horz(1)], [y(2,1), y(2,1)], 'Color', get(e2, 'Color'), 'LineWidth', 2); %horiz
e22 = errorbar(ctr(2,2), y(2,2),errorminus_vert(2,2),errorplus_vert(2,2),'LineStyle','none','LineWidth',2);  %low conf pref
e22.Color = '[1 0.07 0.65]'; %color
hLine = line([ctr(2,2) - errorminus_horz(1), ctr(2,2) + errorplus_horz(1)], [y(2,2), y(2,2)], 'Color', get(e22, 'Color'), 'LineWidth', 2); %horiz
e3 = errorbar(ctr(1,3), y(3,1),errorminus_vert(1,3),errorplus_vert(1,3),'LineStyle','none','LineWidth',2); %high conf nonpref
e3.Color = '[0.47 0.67 0.19]';  %color
hLine = line([ctr(1,3) - errorminus_horz(1), ctr(1,3) + errorplus_horz(1)], [y(3,1), y(3,1)], 'Color', get(e3, 'Color'), 'LineWidth', 2); %horiz
e32 = errorbar(ctr(2,3), y(3,2),errorminus_vert(2,3),errorplus_vert(2,3),'LineStyle','none','LineWidth',2);  %high conf pref
e32.Color = '[1 0.07 0.65]';  %color
hLine = line([ctr(2,3) - errorminus_horz(1), ctr(2,3) + errorplus_horz(1)], [y(3,2), y(3,2)], 'Color', get(e32, 'Color'), 'LineWidth', 2); %horiz
%ylim([0.45 0.87]) %original y lim
ylim([0 0.87]) %ylim to 0
XTickLabel = ({'No','Low','High'}); %x label
%set(gca,'xtick',[1 1.1 1.2],'xticklabel',XTickLabel,'ytick',[0.5 0.6 0.7
%0.8],'box','off','FontSize',20) %original aesthetics
set(gca,'xtick',[1 1.1 1.2],'xticklabel',XTickLabel,'box','off','FontSize',20) %y lim to 0 aesthetics
ylabel('% Correct') %y label
legend({'Non-Pref','Pref'},'AutoUpdate','off') %legend
X(1) = ctr(1,1); X(2) = ctr(2,1); Y(1) = 0.8; Y(2) = 0.8; %sig bar
plot(X(1:2),Y,'LineWidth',1.5,'Color','k') %plot
X(1) = ctr(1,2); X(2) = ctr(2,2); Y(1) = 0.85; Y(2) = 0.85; %sig bar
plot(X(1:2),Y,'LineWidth',1.5,'Color','k') %plot
X(1) = ctr(1,3); X(2) = ctr(2,3); Y(1) = 0.76; Y(2) = 0.76; %sig bar
plot(X(1:2),Y,'LineWidth',1.5,'Color','k') %plot

%% low vol control learning rate

f = figure; %figure
ctr = []; ydt = []; %space
y = [mean(noconflowvolctrl_learning_rate); mean(lowconflowvolctrl_learning_rate); mean(highconflowvolctrl_learning_rate)]; %learning rate
a_vert = [std(noconflowvolctrl_learning_rate)/sqrt(length(noconflowvolctrl_learning_rate)); std(lowconflowvolctrl_learning_rate)/sqrt(length(lowconflowvolctrl_learning_rate)); std(highconflowvolctrl_learning_rate)/sqrt(length(highconflowvolctrl_learning_rate))]; %SEM
errorplus_vert=a_vert'; %positive SEM
errorminus_vert=errorplus_vert; %negative SEM
x = [1; 1.05; 1.1]; %x position
a_horz = [0.04;0.04;0.04]/2; %horizontal line
errorplus_horz=a_horz'; %postive horiz
errorminus_horz=errorplus_horz; %negative horiz
hBar = bar(x,y,'FaceColor',[1 1 1],'EdgeColor',[1 1 1]); %plot learning rate
for k1 = 1:size(y,2) %"for" loop going through LRs
    ctr(k1,:) = bsxfun(@plus, hBar(k1).XData, hBar(k1).XOffset');     %ctr
    ydt(k1,:) = hBar(k1).YData;                     %ydt
end %ending "for" loop going through LRs
minnoc = [1,1,0.71,1,0.6,1,0.96,1,1,1,0.99,1,0.52]; %no conf individuals
lownoc = [0.8,0.71,0.65,1,0.99,1,1,1,1,0.5,0.67,1,0.63]; %low conf individuals
highnoc = [0.54,0.69,0.54,1,1,1,1,0.79,1,0.91,0.94,0.4,0.85]; %high conf individuals
minnoc(2,:) = ctr(1,1); %center
lownoc(2,:) = ctr(1,2); %center
highnoc(2,:) = ctr(1,3); %center
hold on %hold figure
c = [0.5 0.5 0.5]; %color
scatter(minnoc(2,:),minnoc(1,:),50,c); %scatter no conf
scatter(lownoc(2,:),lownoc(1,:),50,c); %scatter low conf
scatter(highnoc(2,:),highnoc(1,:),50,c); %scatter high conf
e1 = errorbar(x(1), y(1), errorminus_vert(1), errorplus_vert(1), 'LineStyle', 'none', 'LineWidth', 2); %no conf LR
e1.Color = '[0.3 0.3 0.3]';  %color
hLine = line([x(1) - errorminus_horz(1), x(1) + errorplus_horz(1)], [y(1), y(1)], 'Color', get(e1, 'Color'), 'LineWidth', 2); %horiz
e2 = errorbar(x(2), y(2), errorminus_vert(2), errorplus_vert(2), 'LineStyle', 'none', 'LineWidth', 2); %low conf LR
e2.Color = '[0.3 0.3 0.3]'; %color
hLine = line([x(2) - errorminus_horz(2), x(2) + errorplus_horz(2)], [y(2), y(2)], 'Color', get(e2, 'Color'), 'LineWidth', 2); %horiz
e3 = errorbar(x(3), y(3), errorminus_vert(3), errorplus_vert(3), 'LineStyle', 'none', 'LineWidth', 2); %high conf LR
e3.Color = '[0.3 0.3 0.3]'; %color
hLine = line([x(3) - errorminus_horz(3), x(3) + errorplus_horz(3)], [y(3), y(3)], 'Color', get(e3, 'Color'), 'LineWidth', 2); %horiz
%ylim([0.2 1.1]) %original ylim
ylim([0 1.1]) %ylim to 0
xlim([0.97 1.13]) %xlim
XTickLabel = ({'No','Low','High'}); %x labels
%set(gca,'xtick',[1 1.05 1.1],'xticklabel',XTickLabel,'ytick',[0.4:0.2:1],'box','off','FontSize',20) %original aesthetics
set(gca,'xtick',[1 1.05 1.1],'xticklabel',XTickLabel,'box','off','FontSize',20) %ylim to 0 aesthetics
ylabel('Learning Rate') %y label
X(1) = ctr(1,1); X(2) = ctr(1,3); Y(1) = 1.03; Y(2) = 1.03; %sig bar
plot(X(1:2),Y,'LineWidth',1.5,'Color','k') %plot

%% low vol control bias

f = figure; %figure
ctr = []; ydt = []; %space
y = [mean(noconflowvolctrl_Q_bias); mean(lowconflowvolctrl_Q_bias); mean(highconflowvolctrl_Q_bias)]; %bias
a_vert = [std(noconflowvolctrl_Q_bias)/sqrt(length(noconflowvolctrl_Q_bias)); std(lowconflowvolctrl_Q_bias)/sqrt(length(lowconflowvolctrl_Q_bias)); std(highconflowvolctrl_Q_bias)/sqrt(length(highconflowvolctrl_Q_bias))]; %SEM
errorplus_vert=a_vert'; %positive SEM
errorminus_vert=errorplus_vert; %negative SEM
x = [1; 1.05; 1.1]; %x position
a_horz = [0.04;0.04;0.04]/2; %horizontal lines
errorplus_horz=a_horz'; %positive horiz
errorminus_horz=errorplus_horz; %negative horiz
hBar = bar(x,y,'FaceColor',[1 1 1],'EdgeColor',[1 1 1]); %plot bias
for k1 = 1:size(y,2) %"for" loop going through bias
    ctr(k1,:) = bsxfun(@plus, hBar(k1).XData, hBar(k1).XOffset');      %ctr
    ydt(k1,:) = hBar(k1).YData;                    %ydt
end %ending "for" loop going through bias
h = hline(0,'w'); %block out 0
h.LineWidth = 2; %line width
minnoc = [0.01,0.67,0.64,0.05,0.08,0.48,0.08,0.07,0.64,0.02,0.31,0.02,0.44]; %no conf ind.
lownoc = [0.56,0.92,0.02,0.07,0.26,0.04,0.06,0.1,0.39,0.45,0.26,0.1,0.26]; %low conf ind.
highnoc = [0.86,1.07,0.86,0.21,0.11,0.37,0.27,0.33,0.37,0.04,0.11,0.45,0.33]; %high conf ind.
minnoc(2,:) = ctr(1,1); %center
lownoc(2,:) = ctr(1,2); %center
highnoc(2,:) = ctr(1,3); %center
hold on %hold figure
c = [0.5 0.5 0.5]; %color
scatter(minnoc(2,:),minnoc(1,:),50,c); %scatter no conf
scatter(lownoc(2,:),lownoc(1,:),50,c); %scatter low conf
scatter(highnoc(2,:),highnoc(1,:),50,c); %scatter high conf
e1 = errorbar(x(1), y(1), errorminus_vert(1), errorplus_vert(1), 'LineStyle', 'none', 'LineWidth', 2); %no conf bias
e1.Color = '[0.3 0.3 0.3]';  %color
hLine = line([x(1) - errorminus_horz(1), x(1) + errorplus_horz(1)], [y(1), y(1)], 'Color', get(e1, 'Color'), 'LineWidth', 2); %horiz
e2 = errorbar(x(2), y(2), errorminus_vert(2), errorplus_vert(2), 'LineStyle', 'none', 'LineWidth', 2); %low conf bias
e2.Color = '[0.3 0.3 0.3]';  %color
hLine = line([x(2) - errorminus_horz(2), x(2) + errorplus_horz(2)], [y(2), y(2)], 'Color', get(e2, 'Color'), 'LineWidth', 2); %horiz
e3 = errorbar(x(3), y(3), errorminus_vert(3), errorplus_vert(3), 'LineStyle', 'none', 'LineWidth', 2); %high conf bias
e3.Color = '[0.3 0.3 0.3]'; %color
hLine = line([x(3) - errorminus_horz(3), x(3) + errorplus_horz(3)], [y(3), y(3)], 'Color', get(e3, 'Color'), 'LineWidth', 2); %horiz
ylim([-0.2 1.2]) %ylim
xlim([0.97 1.13]) %xlim
XTickLabel = ({'No','Low','High'}); %x label
set(gca,'xtick',[1 1.05 1.1],'xticklabel',XTickLabel,'box','off','FontSize',20) %aesthetics
ylabel('Bias') %y label
X(1) = ctr(1,1); X(2) = ctr(1,3); Y(1) = 1.1; Y(2) = 1.1;
plot(X(1:2),Y,'LineWidth',1.5,'Color','k')

%% low vol ctrl regression analysis

f = figure; %figure
ctr = []; ydt = []; %space
y = [mean(min_no_rew1) mean(min_no_unrew1)*-1 mean(min_no_bias); mean(low_no_rew1) mean(low_no_unrew1)*-1 mean(low_no_bias); mean(high_no_rew1) mean(high_no_unrew1)*-1 mean(high_no_bias)]; %regression weights
a_vert = [std(min_no_rew1)/sqrt(length(min_no_rew1)) std(min_no_unrew1)/sqrt(length(min_no_unrew1)) std(min_no_bias)/sqrt(length(min_no_bias)); std(low_no_rew1)/sqrt(length(low_no_rew1)) std(low_no_unrew1)/sqrt(length(low_no_unrew1)) std(low_no_bias)/sqrt(length(low_no_bias)); std(high_no_rew1)/sqrt(length(high_no_rew1)) std(high_no_unrew1)/sqrt(length(high_no_unrew1)) std(high_no_bias)/sqrt(length(high_no_bias))]; %SEM
errorplus_vert=a_vert'; %positive SEM
errorminus_vert=errorplus_vert; %negative SEM
x = [1; 1.1; 1.2]; %x position
a_horz = [0.01;0.01;0.01]/2; %horizontal bars
errorplus_horz=a_horz'; %positive horiz
errorminus_horz=errorplus_horz; %negative horiz
hBar = bar(x,y,'FaceColor',[1 1 1],'EdgeColor',[1 1 1]); %plot regression weights
for k1 = 1:size(y,2) %"for" loop going through regression
    ctr(k1,:) = bsxfun(@plus, hBar(k1).XData, hBar(k1).XOffset');     %ctr
    ydt(k1,:) = hBar(k1).YData;                    %ydt
end %ending "for" loop going through regression
h = hline(0,'w'); %block 0
h.LineWidth = 2; %line wdith
ctr(1,:) = ctr(1,:)+0.01; %extend center
ctr(3,:) = ctr(3,:)-0.01; %extend center
minnobias = [0.09,0.27,0.14,0.14,0.21,0.15,0.04,0.19,0.17,0.09,0.01,0.12,0.36]; %no conf bias ind
minnorew1 = [1.26,1.53,1.62,0.74,1.84,1.99,1.25,1.4,1.55,1.02,1.19,0.81]; %no conf rew ind
minnounrew1 = [0.38,0.19,0.38,0.21,0.4,1.01,0.42,0.01,0.54,0.74,0.29,0.53,0.35]; %no conf unrew ind
lownobias = [0,0.33,0.06,0.06,0.51,0.04,0.02,0.22,0.18,0.06,0.1,0.17,0.27]; %low conf bias ind
lownorew1 = [1.39,1.36,1.62,0.87,2.11,1.8,1.13,1.48,1.48,0.82,1.31,1.15,0.8]; %low conf rew ind
lownounrew1 = [0.18,0,0.23,0.01,0.68,0.6,0.53,0.08,0.95,0.64,0.32,0.71,0.34]; %low conf unrew ind
highnobias = [0.13,0.24,0.19,0.06,0.47,0.36,0.32,0.41,0.32,0.04,0.06,0.01,0.39]; %high conf bias ind
highnorew1 = [1.23,1.49,1.33,0.69,2.12,2.2,1.26,1.61,1.56,0.81,1.35,1.37,0.83]; %high conf rew ind
highnounrew1 = [0.41,0.08,0.44,0.13,0.41,1.03,0.55,0.26,0.62,0.73,0.2,0.83,0.27]; %high conf unrew ind
minnobias(2,:) = ctr(3,1); %center
minnorew1(2,:) = ctr(1,1); %center
minnounrew1(2,:) = ctr(2,1); %center
lownobias(2,:) = ctr(3,2); %center
lownorew1(2,:) = ctr(1,2); %center
lownounrew1(2,:) = ctr(2,2); %center
highnobias(2,:) = ctr(3,3); %center
highnorew1(2,:) = ctr(1,3); %center
highnounrew1(2,:) = ctr(2,3); %center
hold on %hold fig
scatter(minnobias(2,:),minnobias(1,:),30,[0.96 0.65 0.71]); %scatter no conf bias
scatter(lownobias(2,:),lownobias(1,:),30,[0.96 0.65 0.71]); %scatter low conf bias
scatter(highnobias(2,:),highnobias(1,:),30,[0.96 0.65 0.71]); %scatter high conf bias
scatter(minnorew1(2,:),minnorew1(1,:),30,[0.53 0.79 0.89]); %scatter no conf rew
scatter(lownorew1(2,:),lownorew1(1,:),30,[0.53 0.79 0.89]); %scatter low conf rew
scatter(highnorew1(2,:),highnorew1(1,:),30,[0.53 0.79 0.89]); %scatter high conf rew
scatter(minnounrew1(2,:),minnounrew1(1,:),30,[0.7 0.7 0.7]); %scatter no conf unrew
scatter(lownounrew1(2,:),lownounrew1(1,:),30,[0.7 0.7 0.7]); %scatter low conf unrew
scatter(highnounrew1(2,:),highnounrew1(1,:),30,[0.7 0.7 0.7]); %scatter high conf unrew
e1 = errorbar(ctr(1,1), y(1,1),errorminus_vert(1,1),errorplus_vert(1,1),'LineStyle','none','LineWidth',2);  %no conf rew
e1.Color = 'b';  %color
hLine = line([ctr(1,1) - errorminus_horz(1), ctr(1,1) + errorplus_horz(1)], [y(1,1), y(1,1)], 'Color', get(e1, 'Color'), 'LineWidth', 2); %horiz
e12 = errorbar(ctr(2,1), y(1,2),errorminus_vert(2,1),errorplus_vert(2,1),'LineStyle','none','LineWidth',2); %no conf unrew
e12.Color = '[0.5 0.5 0.5]';  %color
hLine = line([ctr(2,1) - errorminus_horz(1), ctr(2,1) + errorplus_horz(1)], [y(1,2), y(1,2)], 'Color', get(e12, 'Color'), 'LineWidth', 2); %horiz
e13 = errorbar(ctr(3,1), y(1,3),errorminus_vert(3,1),errorplus_vert(3,1),'LineStyle','none','LineWidth',2);  %no conf bias
e13.Color = 'r';  %color
hLine = line([ctr(3,1) - errorminus_horz(1), ctr(3,1) + errorplus_horz(1)], [y(1,3), y(1,3)], 'Color', get(e13, 'Color'), 'LineWidth', 2); %horiz
e2 = errorbar(ctr(1,2), y(2,1),errorminus_vert(1,2),errorplus_vert(1,2),'LineStyle','none','LineWidth',2);  %low conf rew
e2.Color = 'b';  %color
hLine = line([ctr(1,2) - errorminus_horz(1), ctr(1,2) + errorplus_horz(1)], [y(2,1), y(2,1)], 'Color', get(e2, 'Color'), 'LineWidth', 2); %horiz
e22 = errorbar(ctr(2,2), y(2,2),errorminus_vert(2,2),errorplus_vert(2,2),'LineStyle','none','LineWidth',2);  %low conf unrew
e22.Color = '[0.5 0.5 0.5]';  %color
hLine = line([ctr(2,2) - errorminus_horz(1), ctr(2,2) + errorplus_horz(1)], [y(2,2), y(2,2)], 'Color', get(e22, 'Color'), 'LineWidth', 2); %horiz
e23 = errorbar(ctr(3,2), y(2,3),errorminus_vert(3,2),errorplus_vert(3,2),'LineStyle','none','LineWidth',2);  %low conf bias
e23.Color = 'r';  %color
hLine = line([ctr(3,2) - errorminus_horz(1), ctr(3,2) + errorplus_horz(1)], [y(2,3), y(2,3)], 'Color', get(e23, 'Color'), 'LineWidth', 2); %horiz
e3 = errorbar(ctr(1,3), y(3,1),errorminus_vert(1,3),errorplus_vert(1,3),'LineStyle','none','LineWidth',2);  %high conf rew
e3.Color = 'b';  %color
hLine = line([ctr(1,3) - errorminus_horz(1), ctr(1,3) + errorplus_horz(1)], [y(3,1), y(3,1)], 'Color', get(e3, 'Color'), 'LineWidth', 2); %horiz
e32 = errorbar(ctr(2,3), y(3,2),errorminus_vert(2,3),errorplus_vert(2,3),'LineStyle','none','LineWidth',2);  %high conf unrew
e32.Color = '[0.5 0.5 0.5]';  %color
hLine = line([ctr(2,3) - errorminus_horz(1), ctr(2,3) + errorplus_horz(1)], [y(3,2), y(3,2)], 'Color', get(e32, 'Color'), 'LineWidth', 2); %horiz
e33 = errorbar(ctr(3,3), y(3,3),errorminus_vert(3,3),errorplus_vert(3,3),'LineStyle','none','LineWidth',2);  %high conf bias
e33.Color = 'r';  %color
hLine = line([ctr(3,3) - errorminus_horz(1), ctr(3,3) + errorplus_horz(1)], [y(3,3), y(3,3)], 'Color', get(e33, 'Color'), 'LineWidth', 2); %horiz
ylim([-0.1 2.7]) %ylim
XTickLabel = ({'No','Low','High'}); %x label
set(gca,'xtick',[1 1.1 1.2],'xticklabel',XTickLabel,'ytick',[0 1 2],'box','off','FontSize',20) %aesthetics
ylabel('Regression Weight') %y label
legend({'Rew','Unrew','Bias'},'AutoUpdate','off') %legend
X(1) = ctr(2,1)+0.001; X(2) = ctr(3,1); Y(1) = 2.05; Y(2) = 2.05; %sig
plot(X(1:2),Y,'LineWidth',1.5,'Color','k') %plot
X(1) = ctr(1,1); X(2) = ctr(2,1)-0.001; Y(1) = 2.05; Y(2) = 2.05; %sig bar
plot(X(1:2),Y,'LineWidth',1.5,'Color','k') %plot
X(1) = ctr(1,1); X(2) = ctr(3,1); Y(1) = 2.1; Y(2) = 2.1; %sig bar
plot(X(1:2),Y,'LineWidth',1.5,'Color','k') %plot
X(1) = ctr(2,2)+0.001; X(2) = ctr(3,2); Y(1) = 2.17; Y(2) = 2.17; %sig bar
plot(X(1:2),Y,'LineWidth',1.5,'Color','k') %plot
X(1) = ctr(1,2); X(2) = ctr(2,2)-0.001; Y(1) = 2.17; Y(2) = 2.17; %sig bar
plot(X(1:2),Y,'LineWidth',1.5,'Color','k') %plot
X(1) = ctr(1,2); X(2) = ctr(3,2); Y(1) = 2.22; Y(2) = 2.22; %sig bar
plot(X(1:2),Y,'LineWidth',1.5,'Color','k') %plot
X(1) = ctr(1,3); X(2) = ctr(2,3); Y(1) = 2.26; Y(2) = 2.26; %sig bar
plot(X(1:2),Y,'LineWidth',1.5,'Color','k') %plot
X(1) = ctr(1,3); X(2) = ctr(3,3); Y(1) = 2.31; Y(2) = 2.31; %sig bar
plot(X(1:2),Y,'LineWidth',1.5,'Color','k') %plot

%% low vol control decision time

f = figure; %figure
ctr = []; ydt = []; %space
y = [mean(min_no_dt); mean(low_no_dt); mean(high_no_dt)]/1000; %DTs
a_vert = [std(min_no_dt)/sqrt(length(min_no_dt)); std(low_no_dt)/sqrt(length(low_no_dt)); std(high_no_dt)/sqrt(length(high_no_dt))]/1000; %SEM
errorplus_vert=a_vert'; %positive SEM
errorminus_vert=errorplus_vert; %negative SEM
x = [1; 1.05; 1.1]; %x positions
a_horz = [0.04;0.04;0.04]/2; %horizontal lines
errorplus_horz=a_horz'; %positive horiz
errorminus_horz=errorplus_horz; %negative horiz
hBar = bar(x,y,'FaceColor',[1 1 1],'EdgeColor',[1 1 1]); %plot decision times
for k1 = 1:size(y,2) %"for" loop going through DTs
    ctr(k1,:) = bsxfun(@plus, hBar(k1).XData, hBar(k1).XOffset');     %ctr
    ydt(k1,:) = hBar(k1).YData;                     %ydt
end %ending "for" loop going through DTs
minnoc = [821,888,568,457,454,1406,720,1012,1010,397,1943,974,1041]/1000; %no conf ind
lownoc = [1115,931,585,444,418,1571,812,1314,950,428,1963,1132,1146]/1000; %low conf ind
highnoc = [777,908,614,781,410,1430,604,852,798,401,1833,907,1198]/1000; %high conf ind
minnoc(2,:) = ctr(1,1); %center
lownoc(2,:) = ctr(1,2); %center
highnoc(2,:) = ctr(1,3); %center
hold on %hold on
c = [0.5 0.5 0.5]; %color
scatter(minnoc(2,:),minnoc(1,:),50,c); %scatter no conf
scatter(lownoc(2,:),lownoc(1,:),50,c); %scatter low conf
scatter(highnoc(2,:),highnoc(1,:),50,c); %scatter high conf
e1 = errorbar(x(1), y(1), errorminus_vert(1), errorplus_vert(1), 'LineStyle', 'none', 'LineWidth', 2); %plot DT
e1.Color = '[0.3 0.3 0.3]';  %color
hLine = line([x(1) - errorminus_horz(1), x(1) + errorplus_horz(1)], [y(1), y(1)], 'Color', get(e1, 'Color'), 'LineWidth', 2); %horiz
e2 = errorbar(x(2), y(2), errorminus_vert(2), errorplus_vert(2), 'LineStyle', 'none', 'LineWidth', 2); %low conf DT
e2.Color = '[0.3 0.3 0.3]';  %color
hLine = line([x(2) - errorminus_horz(2), x(2) + errorplus_horz(2)], [y(2), y(2)], 'Color', get(e2, 'Color'), 'LineWidth', 2); %horiz
e3 = errorbar(x(3), y(3), errorminus_vert(3), errorplus_vert(3), 'LineStyle', 'none', 'LineWidth', 2); %high conf DT
e3.Color = '[0.3 0.3 0.3]';  %color
hLine = line([x(3) - errorminus_horz(3), x(3) + errorplus_horz(3)], [y(3), y(3)], 'Color', get(e3, 'Color'), 'LineWidth', 2);  %horiz
ylim([0 2]) %ylim
xlim([0.97 1.13]) %xlim
XTickLabel = ({'No','Low','High'}); %x label
set(gca,'xtick',[1 1.05 1.1],'xticklabel',XTickLabel,'ytick',[0.5 1 1.5 2],'box','off','FontSize',20) %aesthetics
ylabel('Decision Time (s)') %y label

%% low vol control time in center

f = figure; %figure
ctr = []; ydt = []; %space
y = [mean(min_no_iti); mean(low_no_iti); mean(high_no_iti)]; %time in center
a_vert = [std(min_no_iti)/sqrt(length(min_no_iti)); std(low_no_iti)/sqrt(length(low_no_iti)); std(high_no_iti)/sqrt(length(high_no_iti))]; %SEM
errorplus_vert=a_vert'; %positive SEM
errorminus_vert=errorplus_vert; %negative SEM
x = [1; 1.05; 1.1]; %x positions
a_horz = [0.04;0.04;0.04]/2; %horizontal lines
errorplus_horz=a_horz'; %positive horiz
errorminus_horz=errorplus_horz; %negative horiz
hBar = bar(x,y,'FaceColor',[1 1 1],'EdgeColor',[1 1 1]); %plot time in center
for k1 = 1:size(y,2) %"for" loop going through time in center
    ctr(k1,:) = bsxfun(@plus, hBar(k1).XData, hBar(k1).XOffset');      %ctr
    ydt(k1,:) = hBar(k1).YData;                     %ydt
end %ending "for" loop going through time in center
minnoc = [345,309,343,261,340,347,342,299,335,258,529,366,469]; %no conf ind
lownoc = [320,308,351,273,335,380,348,307,343,268,438,417,431]; %low conf ind
highnoc = [336,296,391,268,341,326,325,272,323,260,444,305,435]; %high conf ind
minnoc(2,:) = ctr(1,1); %center
lownoc(2,:) = ctr(1,2); %center
highnoc(2,:) = ctr(1,3); %center
hold on %hold on
c = [0.5 0.5 0.5]; %color
scatter(minnoc(2,:),minnoc(1,:),50,c); %scatter no conf
scatter(lownoc(2,:),lownoc(1,:),50,c); %scatter low conf
scatter(highnoc(2,:),highnoc(1,:),50,c); %scatter high conf 
e1 = errorbar(x(1), y(1), errorminus_vert(1), errorplus_vert(1), 'LineStyle', 'none', 'LineWidth', 2); %no conf time in center
e1.Color = '[0.3 0.3 0.3]';  %color
hLine = line([x(1) - errorminus_horz(1), x(1) + errorplus_horz(1)], [y(1), y(1)], 'Color', get(e1, 'Color'), 'LineWidth', 2); %horiz
e2 = errorbar(x(2), y(2), errorminus_vert(2), errorplus_vert(2), 'LineStyle', 'none', 'LineWidth', 2); %low conf time in center
e2.Color = '[0.3 0.3 0.3]';  %color
hLine = line([x(2) - errorminus_horz(2), x(2) + errorplus_horz(2)], [y(2), y(2)], 'Color', get(e2, 'Color'), 'LineWidth', 2); %horiz
e3 = errorbar(x(3), y(3), errorminus_vert(3), errorplus_vert(3), 'LineStyle', 'none', 'LineWidth', 2); %high conf time in center
e3.Color = '[0.3 0.3 0.3]';  %color
hLine = line([x(3) - errorminus_horz(3), x(3) + errorplus_horz(3)], [y(3), y(3)], 'Color', get(e3, 'Color'), 'LineWidth', 2);   %horiz
%ylim([200 600]) %original ylim
ylim([0 600]) %ylim axis to 0
xlim([0.97 1.13]) %xlim
XTickLabel = ({'No','Low','High'}); %x label
%set(gca,'xtick',[1 1.05 1.1],'xticklabel',XTickLabel,'ytick',[300 400 500
%600 700],'box','off','FontSize',20) %original aesthetics
set(gca,'xtick',[1 1.05 1.1],'xticklabel',XTickLabel,'box','off','FontSize',20) %axis to 0 aesthetics
ylabel('Time in Center (ms)') %y label

%% low vol control speed at entry

f = figure; %figure
ctr = []; ydt = []; %space
y = [mean(min_no_vel); mean(low_no_vel); mean(high_no_vel)]/3; %speed at execution
a_vert = [std(min_no_vel)/sqrt(length(min_no_vel)); std(low_no_vel)/sqrt(length(low_no_vel)); std(high_no_vel)/sqrt(length(high_no_vel))]/3; %SEM
errorplus_vert=a_vert'; %positive SEM
errorminus_vert=errorplus_vert; %negative SEM
x = [1; 1.05; 1.1]; %x position
a_horz = [0.04;0.04;0.04]/2; %horizontal lines
errorplus_horz=a_horz'; %positive horiz
errorminus_horz=errorplus_horz; %negative horiz
hBar = bar(x,y,'FaceColor',[1 1 1],'EdgeColor',[1 1 1]); %plot speed at execution
for k1 = 1:size(y,2) %"for" loop going through speeds
    ctr(k1,:) = bsxfun(@plus, hBar(k1).XData, hBar(k1).XOffset');      %ctr
    ydt(k1,:) = hBar(k1).YData;                     %ydt
end %ending "for" loop going through speeds
minnoc = [10.8,11.66,12.14,13.98,12.13,10.06,9.86,10.16,9.88,10.86,9.01,9.49,9.21]/3; %no conf ind
lownoc = [10.43,10.78,12.02,13.75,12.17,9.87,9.74,9.95,9.875,10.44,9.12,9.57,9.41]/3; %low conf ind
highnoc = [10.98,11.94,11.58,13.87,12.03,10.2,10.06,10.51,10.1,10.61,8.9,10.21,9.21]/3; %high conf ind
minnoc(2,:) = ctr(1,1); %center
lownoc(2,:) = ctr(1,2); %center
highnoc(2,:) = ctr(1,3); %center
hold on %hold fig
c = [0.5 0.5 0.5]; %color
scatter(minnoc(2,:),minnoc(1,:),50,c); %scatter no conf
scatter(lownoc(2,:),lownoc(1,:),50,c); %scatter low conf
scatter(highnoc(2,:),highnoc(1,:),50,c); %scatter high conf
e1 = errorbar(x(1), y(1), errorminus_vert(1), errorplus_vert(1), 'LineStyle', 'none', 'LineWidth', 2); %no conf speed
e1.Color = '[0.3 0.3 0.3]';  %color
hLine = line([x(1) - errorminus_horz(1), x(1) + errorplus_horz(1)], [y(1), y(1)], 'Color', get(e1, 'Color'), 'LineWidth', 2); %horiz
e2 = errorbar(x(2), y(2), errorminus_vert(2), errorplus_vert(2), 'LineStyle', 'none', 'LineWidth', 2); %low conf speed
e2.Color = '[0.3 0.3 0.3]';  %color
hLine = line([x(2) - errorminus_horz(2), x(2) + errorplus_horz(2)], [y(2), y(2)], 'Color', get(e2, 'Color'), 'LineWidth', 2); %horiz
e3 = errorbar(x(3), y(3), errorminus_vert(3), errorplus_vert(3), 'LineStyle', 'none', 'LineWidth', 2); %high conf speed
e3.Color = '[0.3 0.3 0.3]';  %color
hLine = line([x(3) - errorminus_horz(3), x(3) + errorplus_horz(3)], [y(3), y(3)], 'Color', get(e3, 'Color'), 'LineWidth', 2);   %horiz
%ylim([2.6 5]) %original ylim
ylim([0 5]) %ylim axis ot 0
xlim([0.97 1.13]) %xlim
XTickLabel = ({'No','Low','High'}); %x label
%set(gca,'xtick',[1 1.05 1.1],'xticklabel',XTickLabel,'ytick',[3 4
%5],'box','off','FontSize',20) %original aesthetics
set(gca,'xtick',[1 1.05 1.1],'xticklabel',XTickLabel,'box','off','FontSize',20) %aesthetics axis to 0
ylabel('Speed at Hall Entry (cm/s)') %y label

%% volatility - percent correct

f = figure; %figure
ctr = []; ydt = []; %space
y = [mean(min_no_c) mean(minhigh_no_c); mean(low_no_c) mean(lowhigh_no_c); mean(high_no_c) mean(highhigh_no_c)]; %accuracy
a_vert = [std(min_no_c)/sqrt(length(min_no_c)) std(minhigh_no_c)/sqrt(length(minhigh_no_c)); std(low_no_c)/sqrt(length(low_no_c)) std(lowhigh_no_c)/sqrt(length(lowhigh_no_c)); std(high_no_c)/sqrt(length(high_no_c)) std(highhigh_no_c)/sqrt(length(highhigh_no_c))]; %std
errorplus_vert=a_vert'; %positive SEM
errorminus_vert=errorplus_vert; %negative SEM
x = [1; 1.1; 1.2]; %x position
a_horz = [0.01;0.01;0.01]; %horiz lines
errorplus_horz=a_horz'; %positive horiz
errorminus_horz=errorplus_horz; %negative horiz
hBar = bar(x,y,'FaceColor',[1 1 1],'EdgeColor',[1 1 1]); %plot accuracy
for k1 = 1:size(y,2) %"for" loop going through accuracy
    ctr(k1,:) = bsxfun(@plus, hBar(k1).XData, hBar(k1).XOffset');     %ctr
    ydt(k1,:) = hBar(k1).YData;                     %ydt
end %ending "for" loop going through accuracy
minnoc = [0.66 0.68 0.72 0.64 0.74 0.74 0.65 0.67 0.72 0.7 0.63]; %NL ind
minnohighvol = [0.69 0.65 0.65 0.57 0.71 0.73 0.72 0.7 0.71 0.67 0.6]; %NH ind
lownoc = [0.68 0.7 0.69 0.62 0.75 0.74 0.65 0.65 0.69 0.65 0.63]; %LL ind
lownohighvol = [0.61 0.61 0.66 0.58 0.7 0.74 0.72 0.69 0.69 0.64 0.61]; %LH ind
highnoc = [0.59 0.59 0.61 0.54 0.65 0.67 0.57 0.63 0.6 0.61 0.57]; %HL ind
highnohighvol = [0.55 0.56 0.59 0.56 0.62 0.64 0.6 0.59 0.59 0.59 0.58]; %HH ind
minnoc(2,:) = ctr(1,1); %center
minnohighvol(2,:) = ctr(2,1); %center
lownoc(2,:) = ctr(1,2); %center
lownohighvol(2,:) = ctr(2,2); %center
highnoc(2,:) = ctr(1,3); %center
highnohighvol(2,:) = ctr(2,3); %center
hold on %hold fig
for ugh = 1:11 %go through ind
    X(1) =  minnoc(2,ugh); %low vol acc
    Y(1) =  minnoc(1,ugh); %low vol acc
    X(2) =  minnohighvol(2,ugh); %high vol acc
    Y(2) =  minnohighvol(1,ugh); %high vol acc
    plot(X(1:2),Y,'Color','[0.7 0.7 0.7]') %plot lines
end %ending "for" loop going through ind
for ugh = 1:11 %go through low vol ind
    X(1) =  lownoc(2,ugh); %low vol x
    Y(1) =  lownoc(1,ugh); %low vol acc
    X(2) =  lownohighvol(2,ugh); %high vol x
    Y(2) =  lownohighvol(1,ugh); %high vol acc
    plot(X(1:2),Y,'Color','[0.7 0.7 0.7]') %plot lines
end %ending "for" loop going through low vol
for ugh = 1:11 %"for" loop going through ind
    X(1) =  highnoc(2,ugh); %low vol x
    Y(1) =  highnoc(1,ugh); %low vol acc
    X(2) =  highnohighvol(2,ugh); %high vol x
    Y(2) =  highnohighvol(1,ugh); %high vol acc
    plot(X(1:2),Y,'Color','[0.7 0.7 0.7]') %plot lines
end %ending "for" loop going through ind
X(1) = ctr(1,1); X(2) = ctr(2,1); Y(1) = y(1,1); Y(2) = y(1,2); %no conf avg
plot(X(1:2),Y,'LineWidth',1.5,'Color','k') %plot
X(1) = ctr(1,2); X(2) = ctr(2,2); Y(1) = y(2,1); Y(2) = y(2,2); %low conf avg
plot(X(1:2),Y,'LineWidth',1.5,'Color','k') %plot
X(1) = ctr(1,3); X(2) = ctr(2,3); Y(1) = y(3,1); Y(2) = y(3,2); %high conf avg
plot(X(1:2),Y,'LineWidth',1.5,'Color','k') %plot
e1 = errorbar(ctr(1,1), y(1,1),errorminus_vert(1,1),errorplus_vert(1,1),'LineStyle','none','LineWidth',2); %low vol acc
e1.Color = '[0.4 0.4 0.4]';  %color
hLine = line([ctr(1,1) - errorminus_horz(1), ctr(1,1) + errorplus_horz(1)], [y(1,1), y(1,1)], 'Color', get(e1, 'Color'), 'LineWidth', 2); %horiz
e12 = errorbar(ctr(2,1), y(1,2),errorminus_vert(2,1),errorplus_vert(2,1),'LineStyle','none','LineWidth',2);  %NH acc
e12.Color = 'k';  %color
hLine = line([ctr(2,1) - errorminus_horz(1), ctr(2,1) + errorplus_horz(1)], [y(1,2), y(1,2)], 'Color', get(e12, 'Color'), 'LineWidth', 2); %horiz
e2 = errorbar(ctr(1,2), y(2,1),errorminus_vert(1,2),errorplus_vert(1,2),'LineStyle','none','LineWidth',2);  %LL acc
e2.Color = '[0.4 0.4 0.4]';  %color
hLine = line([ctr(1,2) - errorminus_horz(1), ctr(1,2) + errorplus_horz(1)], [y(2,1), y(2,1)], 'Color', get(e2, 'Color'), 'LineWidth', 2); %horiz
e22 = errorbar(ctr(2,2), y(2,2),errorminus_vert(2,2),errorplus_vert(2,2),'LineStyle','none','LineWidth',2); %LH acc
e22.Color = 'k';  %color
hLine = line([ctr(2,2) - errorminus_horz(1), ctr(2,2) + errorplus_horz(1)], [y(2,2), y(2,2)], 'Color', get(e22, 'Color'), 'LineWidth', 2); %horiz
e3 = errorbar(ctr(1,3), y(3,1),errorminus_vert(1,3),errorplus_vert(1,3),'LineStyle','none','LineWidth',2); %HL acc
e3.Color = '[0.4 0.4 0.4]';  %color
hLine = line([ctr(1,3) - errorminus_horz(1), ctr(1,3) + errorplus_horz(1)], [y(3,1), y(3,1)], 'Color', get(e3, 'Color'), 'LineWidth', 2); %horiz
e32 = errorbar(ctr(2,3), y(3,2),errorminus_vert(2,3),errorplus_vert(2,3),'LineStyle','none','LineWidth',2); %HH acc
e32.Color = 'k'; %color
hLine = line([ctr(2,3) - errorminus_horz(1), ctr(2,3) + errorplus_horz(1)], [y(3,2), y(3,2)], 'Color', get(e32, 'Color'), 'LineWidth', 2); %horiz
%ylim([0.4 1]) %original y lim
ylim([0 1]) %ylim to 0
XTickLabel = ({'No','Low','High'}); %x label
set(gca,'xtick',[1 1.1 1.2],'xticklabel',XTickLabel,'box','off','FontSize',20) %aesthetics
ylabel('% Correct') %y label
legend({'Low Vol','High Vol'},'AutoUpdate','off') %legend

%% volatility - performance by block type

f = figure; %figure
ctr = []; ydt = []; %space
y = [mean(min_no_left_percent) mean(minhigh_no_left_percent) mean(min_no_right_percent) mean(minhigh_no_right_percent); mean(low_no_left_percent) mean(lowhigh_no_left_percent) mean(low_no_right_percent) mean(lowhigh_no_right_percent); mean(high_no_left_percent) mean(highhigh_no_left_percent) mean(high_no_right_percent) mean(highhigh_no_right_percent)]; %performance by block type
a_vert = [std(min_no_left_percent)/sqrt(length(min_no_left_percent)) std(minhigh_no_left_percent)/sqrt(length(minhigh_no_left_percent)) std(min_no_right_percent)/sqrt(length(min_no_right_percent)) std(minhigh_no_right_percent)/sqrt(length(minhigh_no_right_percent)); std(low_no_left_percent)/sqrt(length(low_no_left_percent)) std(lowhigh_no_left_percent)/sqrt(length(lowhigh_no_left_percent)) std(low_no_right_percent)/sqrt(length(low_no_right_percent)) std(lowhigh_no_right_percent)/sqrt(length(lowhigh_no_right_percent)); std(high_no_left_percent)/sqrt(length(high_no_left_percent)) std(highhigh_no_left_percent)/sqrt(length(highhigh_no_left_percent)) std(high_no_right_percent)/sqrt(length(high_no_right_percent)) std(highhigh_no_right_percent)/sqrt(length(highhigh_no_right_percent))]; %SEM
errorplus_vert=a_vert'; %positive SEM
errorminus_vert=errorplus_vert; %negative SEM
x = [1; 1.1; 1.2]; %x position
a_horz = [0.005;0.005;0.005]; %horiz
errorplus_horz=a_horz'; %positive horiz
errorminus_horz=errorplus_horz; %negative horiz
hBar = bar(x,y,'FaceColor',[1 1 1],'EdgeColor',[1 1 1]); %plot performance by block type
for k1 = 1:size(y,2) %"for" loop going through performance
    ctr(k1,:) = bsxfun(@plus, hBar(k1).XData, hBar(k1).XOffset');      %ctr
    ydt(k1,:) = hBar(k1).YData;                    %ydt
end %ending "for" loop going through performance
minnowsL = [0.63 0.6 0.69 0.58 0.7 0.73 0.64 0.64 0.67 0.67 0.63]; %NL non pref
 minhighwsL = [0.67 0.59 0.65 0.61 0.71 0.69 0.74 0.69 0.65 0.69 0.57]; %NH nonpref
 lownowsL = [0.66 0.59 0.68 0.6 0.67 0.73 0.64 0.6 0.65 0.64 0.6]; %LL nonpref
 lowhighwsL = [0.6 0.58 0.67 0.6 0.69 0.67 0.7 0.7 0.63 0.66 0.58]; %LH nonpref
 highnowsL = [0.54 0.52 0.56 0.53 0.56 0.62 0.48 0.53 0.52 0.59 0.56]; %HL nonpref
 highhighwsL = [0.53 0.5 0.58 0.62 0.61 0.65 0.64 0.66 0.51 0.57 0.57]; %HH nonpref
 minnowsR = [0.69 0.75 0.75 0.67 0.79 0.76 0.65 0.7 0.76 0.72 0.64]; %NL pref
 minhighwsR = [0.69 0.7 0.65 0.56 0.7 0.76 0.7 0.71 0.79 0.66 0.63]; %NH pref
 lownowsR = [0.67 0.75 0.69 0.64 0.84 0.75 0.66 0.7 0.73 0.66 0.65]; %LL pref
 lowhighwsR = [0.62 0.65 0.65 0.56 0.71 0.8 0.75 0.69 0.74 0.62 0.64]; %%LH pref
 highnowsR = [0.63 0.66 0.65 0.56 0.75 0.72 0.67 0.74 0.71 0.62 0.58]; %HL pref
 highhighwsR = [0.59 0.65 0.61 0.51 0.63 0.62 0.56 0.54 0.69 0.6 0.6]; %HH pref
 minnowsL(2,:) = ctr(1,1); %center
 minhighwsL(2,:) = ctr(2,1);%center
 minnowsR(2,:) = ctr(3,1);%center
 minhighwsR(2,:) = ctr(4,1);%center
 lownowsL(2,:) = ctr(1,2);%center
 lowhighwsL(2,:) = ctr(2,2);%center
 lownowsR(2,:) = ctr(3,2);%center
 lowhighwsR(2,:) = ctr(4,2);%center
 highnowsL(2,:) = ctr(1,3);%center
 highhighwsL(2,:) = ctr(2,3);%center
 highnowsR(2,:) = ctr(3,3);%center
 highhighwsR(2,:) = ctr(4,3);%center
hold on %hold fig
scatter(minnowsL(2,:),minnowsL(1,:),30,[0.63 0.8 0.38]); %scatter ind
scatter(minhighwsL(2,:),minhighwsL(1,:),30,[0.47 0.67 0.19]);%scatter ind
scatter(lownowsL(2,:),lownowsL(1,:),30,[0.63 0.8 0.38]);%scatter ind
scatter(lowhighwsL(2,:),lowhighwsL(1,:),30,[0.47 0.67 0.19]);%scatter ind
scatter(highnowsL(2,:),highnowsL(1,:),30,[0.63 0.8 0.38]);%scatter ind
scatter(highhighwsL(2,:),highhighwsL(1,:),30,[0.47 0.67 0.19]);%scatter ind
scatter(minnowsR(2,:),minnowsR(1,:),30,[1 0.75 0.91]);%scatter ind
scatter(minhighwsR(2,:),minhighwsR(1,:),30,[0.98 0.51 0.8]);%scatter ind
scatter(lownowsR(2,:),lownowsR(1,:),30,[1 0.75 0.91]);%scatter ind
scatter(lowhighwsR(2,:),lowhighwsR(1,:),30,[0.98 0.51 0.8]);%scatter ind
scatter(highnowsR(2,:),highnowsR(1,:),30,[1 0.75 0.91]);%scatter ind
scatter(highhighwsR(2,:),highhighwsR(1,:),30,[0.98 0.51 0.8]);%scatter ind
X(1) = ctr(1,1); X(2) = ctr(2,1); Y(1) = y(1,1); Y(2) = y(1,2); %avg
plot(X(1:2),Y,'LineWidth',1.5,'Color','k') %plot
X(1) = ctr(3,1); X(2) = ctr(4,1); Y(1) = y(1,3); Y(2) = y(1,4); %avg
plot(X(1:2),Y,'LineWidth',1.5,'Color','k')%plot
X(1) = ctr(1,2); X(2) = ctr(2,2); Y(1) = y(2,1); Y(2) = y(2,2);%avg
plot(X(1:2),Y,'LineWidth',1.5,'Color','k')%plot
X(1) = ctr(3,2); X(2) = ctr(4,2); Y(1) = y(2,3); Y(2) = y(2,4);%avg
plot(X(1:2),Y,'LineWidth',1.5,'Color','k')%plot
X(1) = ctr(1,3); X(2) = ctr(2,3); Y(1) = y(3,1); Y(2) = y(3,2);%avg
plot(X(1:2),Y,'LineWidth',1.5,'Color','k')%plot
X(1) = ctr(3,3); X(2) = ctr(4,3); Y(1) = y(3,3); Y(2) = y(3,4);%avg
plot(X(1:2),Y,'LineWidth',1.5,'Color','k')%plot
e1 = errorbar(ctr(1,1), y(1,1),errorminus_vert(1,1),errorplus_vert(1,1),'LineStyle','none','LineWidth',2); %NL nonpref
e1.Color = '[0.64 0.84 0.48]';  %color
hLine = line([ctr(1,1) - errorminus_horz(1), ctr(1,1) + errorplus_horz(1)], [y(1,1), y(1,1)], 'Color', get(e1, 'Color'), 'LineWidth', 2); %horiz
e12 = errorbar(ctr(2,1), y(1,2),errorminus_vert(2,1),errorplus_vert(2,1),'LineStyle','none','LineWidth',2); %NH nonpref
e12.Color = '[0.29 0.41 0.1]'; %color
hLine = line([ctr(2,1) - errorminus_horz(1), ctr(2,1) + errorplus_horz(1)], [y(1,2), y(1,2)], 'Color', get(e12, 'Color'), 'LineWidth', 2);%horiz
e13 = errorbar(ctr(3,1), y(1,3),errorminus_vert(3,1),errorplus_vert(3,1),'LineStyle','none','LineWidth',2);  %NL pref
e13.Color = '[0.98 0.67 0.86]'; %color
hLine = line([ctr(3,1) - errorminus_horz(1), ctr(3,1) + errorplus_horz(1)], [y(1,3), y(1,3)], 'Color', get(e13, 'Color'), 'LineWidth', 2);%horiz
e14 = errorbar(ctr(4,1), y(1,4),errorminus_vert(4,1),errorplus_vert(4,1),'LineStyle','none','LineWidth',2);  %NH pref
e14.Color = '[1 0.07 0.65]';%color
hLine = line([ctr(4,1) - errorminus_horz(1), ctr(4,1) + errorplus_horz(1)], [y(1,4), y(1,4)], 'Color', get(e14, 'Color'), 'LineWidth', 2);%horiz
e2 = errorbar(ctr(1,2), y(2,1),errorminus_vert(1,2),errorplus_vert(1,2),'LineStyle','none','LineWidth',2); %LL nonpref
e2.Color = '[0.64 0.84 0.48]'; %color
hLine = line([ctr(1,2) - errorminus_horz(1), ctr(1,2) + errorplus_horz(1)], [y(2,1), y(2,1)], 'Color', get(e2, 'Color'), 'LineWidth', 2);%horiz
e22 = errorbar(ctr(2,2), y(2,2),errorminus_vert(2,2),errorplus_vert(2,2),'LineStyle','none','LineWidth',2);  %LH nonpref
e22.Color = '[0.29 0.41 0.1]'; %color
hLine = line([ctr(2,2) - errorminus_horz(1), ctr(2,2) + errorplus_horz(1)], [y(2,2), y(2,2)], 'Color', get(e22, 'Color'), 'LineWidth', 2);%horiz
e23 = errorbar(ctr(3,2), y(2,3),errorminus_vert(3,2),errorplus_vert(3,2),'LineStyle','none','LineWidth',2); %LL pref
e23.Color = '[0.98 0.67 0.86]'; %color
hLine = line([ctr(3,2) - errorminus_horz(1), ctr(3,2) + errorplus_horz(1)], [y(2,3), y(2,3)], 'Color', get(e23, 'Color'), 'LineWidth', 2);%horiz
e24 = errorbar(ctr(4,2), y(2,4),errorminus_vert(4,2),errorplus_vert(4,2),'LineStyle','none','LineWidth',2); %LH pref
e24.Color = '[1 0.07 0.65]';%color
hLine = line([ctr(4,2) - errorminus_horz(1), ctr(4,2) + errorplus_horz(1)], [y(2,4), y(2,4)], 'Color', get(e24, 'Color'), 'LineWidth', 2);%horiz
e3 = errorbar(ctr(1,3), y(3,1),errorminus_vert(1,3),errorplus_vert(1,3),'LineStyle','none','LineWidth',2); %HL nonpref
e3.Color = '[0.64 0.84 0.48]'; %color
hLine = line([ctr(1,3) - errorminus_horz(1), ctr(1,3) + errorplus_horz(1)], [y(3,1), y(3,1)], 'Color', get(e3, 'Color'), 'LineWidth', 2);%horiz
e32 = errorbar(ctr(2,3), y(3,2),errorminus_vert(2,3),errorplus_vert(2,3),'LineStyle','none','LineWidth',2);  %HH nonpref
e32.Color = '[0.29 0.41 0.1]'; %color
hLine = line([ctr(2,3) - errorminus_horz(1), ctr(2,3) + errorplus_horz(1)], [y(3,2), y(3,2)], 'Color', get(e32, 'Color'), 'LineWidth', 2);%horiz
e33 = errorbar(ctr(3,3), y(3,3),errorminus_vert(3,3),errorplus_vert(3,3),'LineStyle','none','LineWidth',2); %HL pref
e33.Color = '[0.98 0.67 0.86]'; %color
hLine = line([ctr(3,3) - errorminus_horz(1), ctr(3,3) + errorplus_horz(1)], [y(3,3), y(3,3)], 'Color', get(e33, 'Color'), 'LineWidth', 2);%horiz
e34 = errorbar(ctr(4,3), y(3,4),errorminus_vert(4,3),errorplus_vert(4,3),'LineStyle','none','LineWidth',2); %HH pref
e34.Color = '[1 0.07 0.65]';%color
hLine = line([ctr(4,3) - errorminus_horz(1), ctr(4,3) + errorplus_horz(1)], [y(3,4), y(3,4)], 'Color', get(e34, 'Color'), 'LineWidth', 2);%horiz
ylim([0.4 1]) %ylim
XTickLabel = ({'No','Low','High'}); %x label
set(gca,'xtick',[1 1.1 1.2],'xticklabel',XTickLabel,'box','off','FontSize',20) %aesthetics
ylabel('% Correct') %y label
legend({'Low Vol Non-Pref','High Vol Non-Pref','Low Vol Pref','High Vol Pref'},'AutoUpdate','off') %legend
X(1) = ctr(3,3); X(2) = ctr(4,3); Y(1) = 0.76; Y(2) = 0.76; %sig bar
plot(X(1:2),Y,'LineWidth',1.5,'Color','k') %plot

%% volatility - beta

f = figure; %figure
ctr = []; ydt = []; %space
y = [mean(noconflowvolctrl_explore_exploit) mean(lowconflowvolctrl_explore_exploit) mean(highconflowvolctrl_explore_exploit); mean(noconfhighvolctrl_explore_exploit) mean(lowconfhighvolctrl_explore_exploit) mean(highconfhighvolctrl_explore_exploit)]; %beta
a_vert = [std(noconflowvolctrl_explore_exploit)/sqrt(length(noconflowvolctrl_explore_exploit)) std(lowconflowvolctrl_explore_exploit)/sqrt(length(lowconflowvolctrl_explore_exploit)) std(highconflowvolctrl_explore_exploit)/sqrt(length(highconflowvolctrl_explore_exploit)); std(noconfhighvolctrl_explore_exploit)/sqrt(length(noconfhighvolctrl_explore_exploit)) std(lowconfhighvolctrl_explore_exploit)/sqrt(length(lowconfhighvolctrl_explore_exploit)) std(highconfhighvolctrl_explore_exploit)/sqrt(length(highconfhighvolctrl_explore_exploit))]; %SEM
errorplus_vert=a_vert'; %positive SEM
errorminus_vert=errorplus_vert; %negative SEM
x = [1; 1.1]; %x position
a_horz = [0.01;0.01]/2; %horiz
errorplus_horz=a_horz'; %positive horiz
errorminus_horz=errorplus_horz; %negative horiz
hBar = bar(x,y,'FaceColor',[1 1 1],'EdgeColor',[1 1 1]); %plot beta
for k1 = 1:size(y,2) %"for" loop going through betas
    ctr(k1,:) = bsxfun(@plus, hBar(k1).XData, hBar(k1).XOffset');     %ctr
    ydt(k1,:) = hBar(k1).YData;                    %ydt
end %ending "for" loop going through betas
ctr(1,:) = ctr(1,:)+0.01; %positive offest
ctr(3,:) = ctr(3,:)-0.01; %negative offset
minnobeta = [1.48 0.77 1.44 0.64 0.7 1.31 1.12 1.23 1.13 0.73 1.15]; %NL ind
lownobeta = [1.06 0.6 1.3 0.7 0.4 1.38 1.36 1.09 1.11 1.59 0.62]; %LL ind
highnobeta = [1.49 0.66 1.49 0.54 0.2 0.93 1.12 1.07 1.75 0.6 0.95]; %HL ind
minhighbeta = [0.73 0.98 1.43 0.45 1.32 1.6 1.1 1.66 0.84 1.27 1.26]; %NH ind
lowhighbeta = [0.4 1.77 0.92 0.28 1.47 1.66 1.9 1.62 1.1 1.03 0.97]; %LH ind
highhighbeta = [0.9 1.77 0.33 0.78 1.05 1.38 1.47 1.51 0.9 1.38 0.55]; %HH ind
minnobeta(2,:) = ctr(1,1); %center
lownobeta(2,:) = ctr(2,1);%center
highnobeta(2,:) = ctr(3,1);%center
minhighbeta(2,:) = ctr(1,2);%center
lowhighbeta(2,:) = ctr(2,2);%center
highhighbeta(2,:) = ctr(3,2);%center
hold on %hold fig
scatter(minnobeta(2,:),minnobeta(1,:),30,[0.64 0.86 1]); %scatter ind
scatter(minhighbeta(2,:),minhighbeta(1,:),30,[0.64 0.86 1]);%scatter ind
scatter(lownobeta(2,:),lownobeta(1,:),30,[0.89 0.75 0.54]);%scatter ind
scatter(lowhighbeta(2,:),lowhighbeta(1,:),30,[0.89 0.75 0.54]);%scatter ind
scatter(highnobeta(2,:),highnobeta(1,:),30,[0.98 0.64 0.5]);%scatter ind
scatter(highhighbeta(2,:),highhighbeta(1,:),30,[0.98 0.64 0.5]);%scatter ind
e1 = errorbar(ctr(1,1), y(1,1),errorminus_vert(1,1),errorplus_vert(1,1),'LineStyle','none','LineWidth',2); %NL beta
e1.Color = '[0 0.45 0.74]'; %color
hLine = line([ctr(1,1) - errorminus_horz(1), ctr(1,1) + errorplus_horz(1)], [y(1,1), y(1,1)], 'Color', get(e1, 'Color'), 'LineWidth', 2); %horiz
e12 = errorbar(ctr(2,1), y(1,2),errorminus_vert(2,1),errorplus_vert(2,1),'LineStyle','none','LineWidth',2); %LL beta
e12.Color = '[0.93 0.69 0.13]'; %color
hLine = line([ctr(2,1) - errorminus_horz(1), ctr(2,1) + errorplus_horz(1)], [y(1,2), y(1,2)], 'Color', get(e12, 'Color'), 'LineWidth', 2);%horiz
e13 = errorbar(ctr(3,1), y(1,3),errorminus_vert(3,1),errorplus_vert(3,1),'LineStyle','none','LineWidth',2); %HL beta
e13.Color = '[0.85 0.33 0.10]'; %color
hLine = line([ctr(3,1) - errorminus_horz(1), ctr(3,1) + errorplus_horz(1)], [y(1,3), y(1,3)], 'Color', get(e13, 'Color'), 'LineWidth', 2);%horiz
e2 = errorbar(ctr(1,2), y(2,1),errorminus_vert(1,2),errorplus_vert(1,2),'LineStyle','none','LineWidth',2); %NH beta
e2.Color = '[0 0.45 0.74]'; %color
hLine = line([ctr(1,2) - errorminus_horz(1), ctr(1,2) + errorplus_horz(1)], [y(2,1), y(2,1)], 'Color', get(e2, 'Color'), 'LineWidth', 2);%horiz
e22 = errorbar(ctr(2,2), y(2,2),errorminus_vert(2,2),errorplus_vert(2,2),'LineStyle','none','LineWidth',2); %LH beta
e22.Color = '[0.93 0.69 0.13]'; %color
hLine = line([ctr(2,2) - errorminus_horz(1), ctr(2,2) + errorplus_horz(1)], [y(2,2), y(2,2)], 'Color', get(e22, 'Color'), 'LineWidth', 2);%horiz
e23 = errorbar(ctr(3,2), y(2,3),errorminus_vert(3,2),errorplus_vert(3,2),'LineStyle','none','LineWidth',2); %HH beta
e23.Color = '[0.85 0.33 0.10]'; %color
hLine = line([ctr(3,2) - errorminus_horz(1), ctr(3,2) + errorplus_horz(1)], [y(2,3), y(2,3)], 'Color', get(e23, 'Color'), 'LineWidth', 2);%horiz
ylim([0 2]) %ylim
XTickLabel = ({'Low Vol','High Vol'}); %x label
set(gca,'xtick',[1 1.1],'xticklabel',XTickLabel,'box','off','FontSize',20) %aesthetics
ylabel('Explore          vs.          Exploit') %y label
legend({'No','Low','High'},'AutoUpdate','off') %legend
X(1) = ctr(1,1); X(2) = ctr(2,1); Y(1) = 1.64; Y(2) = 1.64; %sig bar
plot(X(1:2),Y,'LineWidth',1.5,'Color','k') %plot
X(1) = ctr(1,1); X(2) = ctr(3,1); Y(1) = 1.8; Y(2) = 1.8;%sig bar
plot(X(1:2),Y,'LineWidth',1.5,'Color','k')%plot

%% volatility - rew vs unrew learning rate

f = figure; %fig
ctr = []; ydt = []; %space
y = [mean(min_no_rew_lr) mean(low_no_rew_lr) mean(high_no_rew_lr); mean(minhigh_no_rew_lr) mean(lowhigh_no_rew_lr) mean(highhigh_no_rew_lr); mean(min_no_unrew_lr) mean(low_no_unrew_lr) mean(high_no_unrew_lr);  mean(minhigh_no_unrew_lr) mean(lowhigh_no_unrew_lr) mean(highhigh_no_unrew_lr)]; %LRs
a_vert = [std(min_no_rew_lr)/sqrt(length(min_no_rew_lr)) std(low_no_rew_lr)/sqrt(length(low_no_rew_lr)) std(high_no_rew_lr)/sqrt(length(high_no_rew_lr)); std(minhigh_no_rew_lr)/sqrt(length(minhigh_no_rew_lr)) std(lowhigh_no_rew_lr)/sqrt(length(lowhigh_no_rew_lr)) std(highhigh_no_rew_lr)/sqrt(length(highhigh_no_rew_lr)); std(min_no_unrew_lr)/sqrt(length(min_no_unrew_lr)) std(low_no_unrew_lr)/sqrt(length(low_no_unrew_lr)) std(high_no_unrew_lr)/sqrt(length(high_no_unrew_lr)); std(minhigh_no_unrew_lr)/sqrt(length(minhigh_no_unrew_lr)) std(lowhigh_no_unrew_lr)/sqrt(length(lowhigh_no_unrew_lr)) std(highhigh_no_unrew_lr)/sqrt(length(highhigh_no_unrew_lr))]; %SEM
errorplus_vert=a_vert'; %positive SEM
errorminus_vert=errorplus_vert; %negative SEM
x = [1; 1.1; 1.2; 1.3]; %x positions
a_horz = [0.01;0.01;0.01]/2; %horiz
errorplus_horz=a_horz'; %positive horiz
errorminus_horz=errorplus_horz; %negative horiz
hBar = bar(x,y,'FaceColor',[1 1 1],'EdgeColor',[1 1 1]); %plot LRs
for k1 = 1:size(y,2)%"for" loop going through LRs
    ctr(k1,:) = bsxfun(@plus, hBar(k1).XData, hBar(k1).XOffset');     %ctr
    ydt(k1,:) = hBar(k1).YData;                    %ydt
end %ending "for" loop going through LRs
ctr(1,:) = ctr(1,:)+0.01; %positive offset
ctr(3,:) = ctr(3,:)-0.01; %negative offset
minnorew = [0.7 0.55 1 0.71 0.94 1 0.8 1 0.81 1 0.18]; %NL rew LR
lownorew = [0.87 0.56 1 0.62 0.77 0.71 0.715 1 1 0.79 0.69]; %LL rew LR
highnorew = [0.28 0.51 1 0.85 0.84 1 1 0.96 0.75 1 1]; %HL rew LR
minhighrew = [1 1 1 0.38 0.91 1 1 0.95 1 0.92 0.52]; %NH rew LR
lowhighrew = [1 0.8 0.86 0.805 0.81 1 1 0.7 0.67 0.93 0.815]; %LH rew LR
highhighrew = [0.76 1 1 0.89 0.83 1 1 1 0.68 0.75 0.87]; %HH rew LR
minnounrew = [0.65 1 1 0.62 1 1 1 1 0.98 1 0.66]; %NL unrew LR
lownounrew = [0.83 0.82 0.94 0.85 1 1 0.92 1 0.84 0.45]; %LL unrew LR
highnounrew = [0.45 0.47 0.92 0.455 0.98 0.95 1 0.88 0.75 0.71 0.66]; %HL unrew LR
minhighunrew = [1 0.87 0.68 0.78 1 1 1 0.74 1 0.97 0.44]; %NH unrew LR
lowhighunrew = [0.99 1 0.95 0.84 1 1 1 1 0.8 0.995 0.48]; %LH unrew LR
highhighunrew = [0.72 0.65 0.93 0.48 0.95 1 1 1 0.77 0.93 0.86]; %HH unrew LR
minnorew(2,:) = ctr(1,1); %center
lownorew(2,:) = ctr(2,1);%center
highnorew(2,:) = ctr(3,1);%center
minhighrew(2,:) = ctr(1,2);%center
lowhighrew(2,:) = ctr(2,2);%center
highhighrew(2,:) = ctr(3,2);%center
minnounrew(2,:) = ctr(1,3);%center
lownounrew(2,:) = ctr(2,3);%center
highnounrew(2,:) = ctr(3,3);%center
minhighunrew(2,:) = ctr(1,4);%center
lowhighunrew(2,:) = ctr(2,4);%center
highhighunrew(2,:) = ctr(3,4);%center
hold on %hold fig
scatter(minnorew(2,:),minnorew(1,:),30,[0.64 0.86 1]); %scatter ind
scatter(minhighrew(2,:),minhighrew(1,:),30,[0.64 0.86 1]);%scatter ind
scatter(minnounrew(2,:),minnounrew(1,:),30,[0.64 0.86 1]);%scatter ind
scatter(minhighunrew(2,:),minhighunrew(1,:),30,[0.64 0.86 1]);%scatter ind
scatter(lownorew(2,:),lownorew(1,:),30,[0.89 0.75 0.54]);%scatter ind
scatter(lowhighrew(2,:),lowhighrew(1,:),30,[0.89 0.75 0.54]);%scatter ind
scatter(lownounrew(2,:),lownounrew(1,:),30,[0.89 0.75 0.54]);%scatter ind
scatter(lowhighunrew(2,:),lowhighunrew(1,:),30,[0.89 0.75 0.54]);%scatter ind
scatter(highnorew(2,:),highnorew(1,:),30,[0.98 0.64 0.5]);%scatter ind
scatter(highhighrew(2,:),highhighrew(1,:),30,[0.98 0.64 0.5]);%scatter ind
scatter(highnounrew(2,:),highnounrew(1,:),30,[0.98 0.64 0.5]);%scatter ind
scatter(highhighunrew(2,:),highhighunrew(1,:),30,[0.98 0.64 0.5]);%scatter ind
e1 = errorbar(ctr(1,1), y(1,1),errorminus_vert(1,1),errorplus_vert(1,1),'LineStyle','none','LineWidth',2); %NL rew LR
e1.Color = '[0 0.45 0.74]'; %color
hLine = line([ctr(1,1) - errorminus_horz(1), ctr(1,1) + errorplus_horz(1)], [y(1,1), y(1,1)], 'Color', get(e1, 'Color'), 'LineWidth', 2); %horiz
e12 = errorbar(ctr(2,1), y(1,2),errorminus_vert(2,1),errorplus_vert(2,1),'LineStyle','none','LineWidth',2); %LL rew LR
e12.Color = '[0.93 0.69 0.13]';  %color
hLine = line([ctr(2,1) - errorminus_horz(1), ctr(2,1) + errorplus_horz(1)], [y(1,2), y(1,2)], 'Color', get(e12, 'Color'), 'LineWidth', 2);%horiz
e13 = errorbar(ctr(3,1), y(1,3),errorminus_vert(3,1),errorplus_vert(3,1),'LineStyle','none','LineWidth',2);  %HL rew LR
e13.Color = '[0.85 0.33 0.10]';  %color
hLine = line([ctr(3,1) - errorminus_horz(1), ctr(3,1) + errorplus_horz(1)], [y(1,3), y(1,3)], 'Color', get(e13, 'Color'), 'LineWidth', 2);%horiz
e2 = errorbar(ctr(1,2), y(2,1),errorminus_vert(1,2),errorplus_vert(1,2),'LineStyle','none','LineWidth',2);  %NH rew LR
e2.Color = '[0 0.45 0.74]';  %color
hLine = line([ctr(1,2) - errorminus_horz(1), ctr(1,2) + errorplus_horz(1)], [y(2,1), y(2,1)], 'Color', get(e2, 'Color'), 'LineWidth', 2);%horiz
e22 = errorbar(ctr(2,2), y(2,2),errorminus_vert(2,2),errorplus_vert(2,2),'LineStyle','none','LineWidth',2);  %LH rew LR
e22.Color = '[0.93 0.69 0.13]';  %color
hLine = line([ctr(2,2) - errorminus_horz(1), ctr(2,2) + errorplus_horz(1)], [y(2,2), y(2,2)], 'Color', get(e22, 'Color'), 'LineWidth', 2); %horiz
e23 = errorbar(ctr(3,2), y(2,3),errorminus_vert(3,2),errorplus_vert(3,2),'LineStyle','none','LineWidth',2);  %HH rew LR
e23.Color = '[0.85 0.33 0.10]';  %color
hLine = line([ctr(3,2) - errorminus_horz(1), ctr(3,2) + errorplus_horz(1)], [y(2,3), y(2,3)], 'Color', get(e23, 'Color'), 'LineWidth', 2); %horiz
e3 = errorbar(ctr(1,3), y(3,1),errorminus_vert(1,3),errorplus_vert(1,3),'LineStyle','none','LineWidth',2);  %NL unrew LR
e3.Color = '[0 0.45 0.74]';  %color
hLine = line([ctr(1,3) - errorminus_horz(1), ctr(1,3) + errorplus_horz(1)], [y(3,1), y(3,1)], 'Color', get(e3, 'Color'), 'LineWidth', 2); %horiz
e32 = errorbar(ctr(2,3), y(3,2),errorminus_vert(2,3),errorplus_vert(2,3),'LineStyle','none','LineWidth',2); %LL unrew LR
e32.Color = '[0.93 0.69 0.13]';  %color
hLine = line([ctr(2,3) - errorminus_horz(1), ctr(2,3) + errorplus_horz(1)], [y(3,2), y(3,2)], 'Color', get(e32, 'Color'), 'LineWidth', 2); %horiz
e33 = errorbar(ctr(3,3), y(3,3),errorminus_vert(3,3),errorplus_vert(3,3),'LineStyle','none','LineWidth',2); %HL unrew LR
e33.Color = '[0.85 0.33 0.10]';  %color
hLine = line([ctr(3,3) - errorminus_horz(1), ctr(3,3) + errorplus_horz(1)], [y(3,3), y(3,3)], 'Color', get(e33, 'Color'), 'LineWidth', 2); %horiz
e4 = errorbar(ctr(1,4), y(4,1),errorminus_vert(1,4),errorplus_vert(1,4),'LineStyle','none','LineWidth',2);  %NH unrew LR
e4.Color = '[0 0.45 0.74]';  %color
hLine = line([ctr(1,4) - errorminus_horz(1), ctr(1,4) + errorplus_horz(1)], [y(4,1), y(4,1)], 'Color', get(e4, 'Color'), 'LineWidth', 2); %horiz
e42 = errorbar(ctr(2,4), y(4,2),errorminus_vert(2,4),errorplus_vert(2,4),'LineStyle','none','LineWidth',2); %LH unrew LR
e42.Color = '[0.93 0.69 0.13]';  %color
hLine = line([ctr(2,4) - errorminus_horz(1), ctr(2,4) + errorplus_horz(1)], [y(4,2), y(4,2)], 'Color', get(e42, 'Color'), 'LineWidth', 2); %horiz
e43 = errorbar(ctr(3,4), y(4,3),errorminus_vert(3,4),errorplus_vert(3,4),'LineStyle','none','LineWidth',2); %%HH unrew LR
e43.Color = '[0.85 0.33 0.10]';  %color
hLine = line([ctr(3,4) - errorminus_horz(1), ctr(3,4) + errorplus_horz(1)], [y(4,3), y(4,3)], 'Color', get(e43, 'Color'), 'LineWidth', 2);%horiz
ylim([0 1.2]) %ylim
XTickLabel = {'Low Rew','High Rew','Low Unrew','High Unrew'}; %x label
set(gca, 'xtick', [1 1.1 1.2 1.3],'XTickLabel', XTickLabel, 'ytick',[0.2:0.2:1],'box', 'off', 'FontSize', 15); %aesthetic
ylabel('Learning Rate') %y label
legend({'No Conf','Low Conf','High Conf'},'AutoUpdate','off') %legend
X(1) = ctr(2,3); X(2) = ctr(3,3); Y(1) = 1.03; Y(2) = 1.03; %sig bar
plot(X(1:2),Y,'LineWidth',1.5,'Color','k') %plot
X(1) = ctr(1,3); X(2) = ctr(3,3); Y(1) = 1.13; Y(2) = 1.13;%sig bar
plot(X(1:2),Y,'LineWidth',1.5,'Color','k')%plot

%% volatility - bias

f = figure; %fig
ctr = []; ydt = []; %space
y = [mean(noconflowvolctrl_Q_bias) mean(lowconflowvolctrl_Q_bias) mean(highconflowvolctrl_Q_bias); mean(noconfhighvolctrl_Q_bias) mean(lowconfhighvolctrl_Q_bias) mean(highconfhighvolctrl_Q_bias)]; %bias
a_vert = [std(noconflowvolctrl_Q_bias)/sqrt(length(noconflowvolctrl_Q_bias)) std(lowconflowvolctrl_Q_bias)/sqrt(length(lowconflowvolctrl_Q_bias)) std(highconflowvolctrl_Q_bias)/sqrt(length(highconflowvolctrl_Q_bias)); std(noconfhighvolctrl_Q_bias)/sqrt(length(noconfhighvolctrl_Q_bias)) std(lowconfhighvolctrl_Q_bias)/sqrt(length(lowconfhighvolctrl_Q_bias)) std(highconfhighvolctrl_Q_bias)/sqrt(length(highconfhighvolctrl_Q_bias))]; %SEM
errorplus_vert=a_vert'; %positive SEM
errorminus_vert=errorplus_vert; %negative SEM
x = [1; 1.1]; %x position
a_horz = [0.01;0.01]/2; %horiz
errorplus_horz=a_horz'; %positive horiz
errorminus_horz=errorplus_horz; %negative horiz
hBar = bar(x,y,'FaceColor',[1 1 1],'EdgeColor',[1 1 1]); %plot bias
for k1 = 1:size(y,2) %"for" loop going through bias
    ctr(k1,:) = bsxfun(@plus, hBar(k1).XData, hBar(k1).XOffset');     %ctr
    ydt(k1,:) = hBar(k1).YData;                    %ydt
end %ending "for" loop going through bias
ctr(1,:) = ctr(1,:)+0.01; %positive offest
ctr(3,:) = ctr(3,:)-0.01; %negative offset
minnobias = [0.01,0.67,0.64,0.05,0.08,0.48,0.08,0.07,0.64,0.02,0.31]; %NL ind
lownobias = [0.56,0.92,0.02,0.07,0.26,0.04,0.06,0.1,0.39,0.45,0.26]; %LL ind
highnobias = [0.86,1.07,0.86,0.21,0.11,0.37,0.27,0.33,0.37,0.04,0.11]; %HL ind
minhighbias = [0.15 -0.08 0.77 0.02 -0.22 0.05 0.48 0.48 0.01 0.35 -0.15]; %NH ind
lowhighbias = [0.45 0.45 0.55 -0.08 0.09 0.45 0.08 0.27 -0.24 0.33 0.03]; %LH ind
highhighbias = [0.33 -0.16 0.68 -0.03 0.46 -0.64 -0.66 0.32 -0.06 0.23 0.4]; %HH ind
minnobias(2,:) = ctr(1,1); %center
lownobias(2,:) = ctr(2,1);%center
highnobias(2,:) = ctr(3,1);%center
minhighbias(2,:) = ctr(1,2);%center
lowhighbias(2,:) = ctr(2,2);%center
highhighbias(2,:) = ctr(3,2);%center
hold on %hold fig
h = hline(0,'w'); %block 0 line
h.LineWidth = 2; %line width
scatter(minnobias(2,:),minnobias(1,:),30,[0.64 0.86 1]); %scatter ind
scatter(minhighbias(2,:),minhighbias(1,:),30,[0.64 0.86 1]);%scatter ind
scatter(lownobias(2,:),lownobias(1,:),30,[0.89 0.75 0.54]);%scatter ind
scatter(lowhighbias(2,:),lowhighbias(1,:),30,[0.89 0.75 0.54]);%scatter ind
scatter(highnobias(2,:),highnobias(1,:),30,[0.98 0.64 0.5]);%scatter ind
scatter(highhighbias(2,:),highhighbias(1,:),30,[0.98 0.64 0.5]);%scatter ind
e1 = errorbar(ctr(1,1), y(1,1),errorminus_vert(1,1),errorplus_vert(1,1),'LineStyle','none','LineWidth',2);  %NL bias
e1.Color = '[0 0.45 0.74]'; %color
hLine = line([ctr(1,1) - errorminus_horz(1), ctr(1,1) + errorplus_horz(1)], [y(1,1), y(1,1)], 'Color', get(e1, 'Color'), 'LineWidth', 2); %horiz
e12 = errorbar(ctr(2,1), y(1,2),errorminus_vert(2,1),errorplus_vert(2,1),'LineStyle','none','LineWidth',2); %LL bias
e12.Color = '[0.93 0.69 0.13]'; %color
hLine = line([ctr(2,1) - errorminus_horz(1), ctr(2,1) + errorplus_horz(1)], [y(1,2), y(1,2)], 'Color', get(e12, 'Color'), 'LineWidth', 2);%horiz
e13 = errorbar(ctr(3,1), y(1,3),errorminus_vert(3,1),errorplus_vert(3,1),'LineStyle','none','LineWidth',2);  %HL bias
e13.Color = '[0.85 0.33 0.10]'; %color
hLine = line([ctr(3,1) - errorminus_horz(1), ctr(3,1) + errorplus_horz(1)], [y(1,3), y(1,3)], 'Color', get(e13, 'Color'), 'LineWidth', 2);%horiz
e2 = errorbar(ctr(1,2), y(2,1),errorminus_vert(1,2),errorplus_vert(1,2),'LineStyle','none','LineWidth',2);  %NH bias
e2.Color = '[0 0.45 0.74]'; %color
hLine = line([ctr(1,2) - errorminus_horz(1), ctr(1,2) + errorplus_horz(1)], [y(2,1), y(2,1)], 'Color', get(e2, 'Color'), 'LineWidth', 2);%horiz
e22 = errorbar(ctr(2,2), y(2,2),errorminus_vert(2,2),errorplus_vert(2,2),'LineStyle','none','LineWidth',2); %LH bias
e22.Color = '[0.93 0.69 0.13]'; %color
hLine = line([ctr(2,2) - errorminus_horz(1), ctr(2,2) + errorplus_horz(1)], [y(2,2), y(2,2)], 'Color', get(e22, 'Color'), 'LineWidth', 2);%horiz
e23 = errorbar(ctr(3,2), y(2,3),errorminus_vert(3,2),errorplus_vert(3,2),'LineStyle','none','LineWidth',2);  %HH bias
e23.Color = '[0.85 0.33 0.10]'; %color
hLine = line([ctr(3,2) - errorminus_horz(1), ctr(3,2) + errorplus_horz(1)], [y(2,3), y(2,3)], 'Color', get(e23, 'Color'), 'LineWidth', 2);%horiz
ylim([-0.5 1.5]) %ylim
XTickLabel = ({'Low Vol','High Vol'}); %x label
set(gca,'xtick',[1 1.1],'xticklabel',XTickLabel,'box','off','FontSize',20) %aesthetics
ylabel('Bias') %y label
legend({'No','Low','High'},'AutoUpdate','off') %legend

%% volatility - regression within the vol.

f = figure; %fig
ctr = []; ydt = []; %space
y = [mean(minhigh_no_rew1) mean(minhigh_no_unrew1)*-1 mean(minhigh_no_bias); mean(lowhigh_no_rew1) mean(lowhigh_no_unrew1)*-1 mean(lowhigh_no_bias); mean(highhigh_no_rew1) mean(highhigh_no_unrew1)*-1 mean(highhigh_no_bias)]; %regression
a_vert = [std(minhigh_no_rew1)/sqrt(length(minhigh_no_rew1)) std(minhigh_no_unrew1)/sqrt(length(minhigh_no_unrew1)) std(minhigh_no_bias)/sqrt(length(minhigh_no_bias)); std(lowhigh_no_rew1)/sqrt(length(lowhigh_no_rew1)) std(lowhigh_no_unrew1)/sqrt(length(lowhigh_no_unrew1)) std(lowhigh_no_bias)/sqrt(length(lowhigh_no_bias)); std(highhigh_no_rew1)/sqrt(length(highhigh_no_rew1)) std(highhigh_no_unrew1)/sqrt(length(highhigh_no_unrew1)) std(highhigh_no_bias)/sqrt(length(highhigh_no_bias))]; %SEM
errorplus_vert=a_vert'; %positive SEM
errorminus_vert=errorplus_vert; %negative SEM
x = [1; 1.1; 1.2]; %x position
a_horz = [0.01;0.01;0.01]/2; %horiz lines
errorplus_horz=a_horz'; %positive horiz
errorminus_horz=errorplus_horz; %negative horiz
hBar = bar(x,y,'FaceColor',[1 1 1],'EdgeColor',[1 1 1]); %plot regression
for k1 = 1:size(y,2) %"for" loop going through regression
    ctr(k1,:) = bsxfun(@plus, hBar(k1).XData, hBar(k1).XOffset');      %ctr
    ydt(k1,:) = hBar(k1).YData;                    %ydt
end %ending "for" loop going through regresison
h = hline(0,'w'); %block out 0 line
h.LineWidth = 2; %line width
ctr(1,:) = ctr(1,:)+0.01; %positive offset
ctr(3,:) = ctr(3,:)-0.01; %negative offset
minnobias = [0.05 0.23 0 -0.04 -0.05 0.23 -0.06 0.07 0.37 -0.06]; %no conf bias
minnorew1 = [1.59 1.63 1.14 0.25 1.55 2.11 2.03 1.99 1.79 1.08]; %no conf rew
minnounrew1 = [0.08 0.32 0.22 0.36 0.39 0.53 1.1 0.21 0.65 0.61]; %no conf unrew
lownobias = [0.04 0.09 -0.03 -0.06 0.04 0.48 0.14 -0.01 0.33 -0.09]; %low conf bias
lownorew1 = [1.28 1.48 1.51 0.35 1.93 2.15 2.1 1.54 1.21 0.97]; %low conf rew
lownounrew1 = [0.29 0.48 0.28 0.33 0.08 0.57 0.82 0.03 0.96 0.89]; %low conf unrew
highnobias = [0.09 0.28 0.06 -0.26 0.02 -0.07 -0.16 -0.24 0.4 0.09]; %high conf bias
highnorew1 = [1.33 1.54 1.25 0.44 1.66 1.73 1.75 1.66 1.53 0.78]; %high conf rew
highnounrew1 = [0.25 0.41 0.15 0.47 0.3 0.62 0.32 0.08 0.58 0.78]; %high conf unrew
minnobias(2,:) = ctr(3,1); %center
minnorew1(2,:) = ctr(1,1);%center
minnounrew1(2,:) = ctr(2,1);%center
lownobias(2,:) = ctr(3,2);%center
lownorew1(2,:) = ctr(1,2);%center
lownounrew1(2,:) = ctr(2,2);%center
highnobias(2,:) = ctr(3,3);%center
highnorew1(2,:) = ctr(1,3);%center
highnounrew1(2,:) = ctr(2,3);%center
hold on %hold fig
scatter(minnobias(2,:),minnobias(1,:),30,[0.96 0.65 0.71]); %scatter ind
scatter(lownobias(2,:),lownobias(1,:),30,[0.96 0.65 0.71]);%scatter ind
scatter(highnobias(2,:),highnobias(1,:),30,[0.96 0.65 0.71]);%scatter ind
scatter(minnorew1(2,:),minnorew1(1,:),30,[0.53 0.79 0.89]);%scatter ind
scatter(lownorew1(2,:),lownorew1(1,:),30,[0.53 0.79 0.89]);%scatter ind
scatter(highnorew1(2,:),highnorew1(1,:),30,[0.53 0.79 0.89]);%scatter ind
scatter(minnounrew1(2,:),minnounrew1(1,:),30,[0.7 0.7 0.7]);%scatter ind
scatter(lownounrew1(2,:),lownounrew1(1,:),30,[0.7 0.7 0.7]);%scatter ind
scatter(highnounrew1(2,:),highnounrew1(1,:),30,[0.7 0.7 0.7]);%scatter ind
e1 = errorbar(ctr(1,1), y(1,1),errorminus_vert(1,1),errorplus_vert(1,1),'LineStyle','none','LineWidth',2); %no conf rew
e1.Color = 'b'; %color
hLine = line([ctr(1,1) - errorminus_horz(1), ctr(1,1) + errorplus_horz(1)], [y(1,1), y(1,1)], 'Color', get(e1, 'Color'), 'LineWidth', 2); %horiz
e12 = errorbar(ctr(2,1), y(1,2),errorminus_vert(2,1),errorplus_vert(2,1),'LineStyle','none','LineWidth',2); %no conf unrew
e12.Color = '[0.5 0.5 0.5]'; %color
hLine = line([ctr(2,1) - errorminus_horz(1), ctr(2,1) + errorplus_horz(1)], [y(1,2), y(1,2)], 'Color', get(e12, 'Color'), 'LineWidth', 2); %horiz
e13 = errorbar(ctr(3,1), y(1,3),errorminus_vert(3,1),errorplus_vert(3,1),'LineStyle','none','LineWidth',2); %no conf bias
e13.Color = 'r'; %color
hLine = line([ctr(3,1) - errorminus_horz(1), ctr(3,1) + errorplus_horz(1)], [y(1,3), y(1,3)], 'Color', get(e13, 'Color'), 'LineWidth', 2); %horiz
e2 = errorbar(ctr(1,2), y(2,1),errorminus_vert(1,2),errorplus_vert(1,2),'LineStyle','none','LineWidth',2); %low conf rew
e2.Color = 'b'; %color
hLine = line([ctr(1,2) - errorminus_horz(1), ctr(1,2) + errorplus_horz(1)], [y(2,1), y(2,1)], 'Color', get(e2, 'Color'), 'LineWidth', 2); %horiz
e22 = errorbar(ctr(2,2), y(2,2),errorminus_vert(2,2),errorplus_vert(2,2),'LineStyle','none','LineWidth',2); %low conf unrew
e22.Color = '[0.5 0.5 0.5]'; %color
hLine = line([ctr(2,2) - errorminus_horz(1), ctr(2,2) + errorplus_horz(1)], [y(2,2), y(2,2)], 'Color', get(e22, 'Color'), 'LineWidth', 2); %horiz
e23 = errorbar(ctr(3,2), y(2,3),errorminus_vert(3,2),errorplus_vert(3,2),'LineStyle','none','LineWidth',2); %low conf bias
e23.Color = 'r'; %color
hLine = line([ctr(3,2) - errorminus_horz(1), ctr(3,2) + errorplus_horz(1)], [y(2,3), y(2,3)], 'Color', get(e23, 'Color'), 'LineWidth', 2); %horiz
e3 = errorbar(ctr(1,3), y(3,1),errorminus_vert(1,3),errorplus_vert(1,3),'LineStyle','none','LineWidth',2); %high conf rew
e3.Color = 'b'; %color
hLine = line([ctr(1,3) - errorminus_horz(1), ctr(1,3) + errorplus_horz(1)], [y(3,1), y(3,1)], 'Color', get(e3, 'Color'), 'LineWidth', 2); %horiz
e32 = errorbar(ctr(2,3), y(3,2),errorminus_vert(2,3),errorplus_vert(2,3),'LineStyle','none','LineWidth',2); %high conf unrew
e32.Color = '[0.5 0.5 0.5]'; %color
hLine = line([ctr(2,3) - errorminus_horz(1), ctr(2,3) + errorplus_horz(1)], [y(3,2), y(3,2)], 'Color', get(e32, 'Color'), 'LineWidth', 2); %horiz
e33 = errorbar(ctr(3,3), y(3,3),errorminus_vert(3,3),errorplus_vert(3,3),'LineStyle','none','LineWidth',2); %high conf bias
e33.Color = 'r'; %color
hLine = line([ctr(3,3) - errorminus_horz(1), ctr(3,3) + errorplus_horz(1)], [y(3,3), y(3,3)], 'Color', get(e33, 'Color'), 'LineWidth', 2); %horiz
ylim([-0.5 2.5]) %ylim
XTickLabel = ({'No','Low','High'}); %x label
set(gca,'xtick',[1 1.1 1.2],'xticklabel',XTickLabel,'box','off','FontSize',20) %aesthetics
ylabel('Regression Weight') %y label
legend({'Rew','Unrew','Bias'},'AutoUpdate','off') %legend
X(1) = ctr(2,1)+0.001; X(2) = ctr(3,1); Y(1) = 2.21; Y(2) = 2.21; %sig bar
plot(X(1:2),Y,'LineWidth',1.5,'Color','k') %plot
X(1) = ctr(1,1); X(2) = ctr(2,1)-0.001; Y(1) = 2.21; Y(2) = 2.21;%sig bar
plot(X(1:2),Y,'LineWidth',1.5,'Color','k')%plot
X(1) = ctr(1,1); X(2) = ctr(3,1); Y(1) = 2.26; Y(2) = 2.26;%sig bar
plot(X(1:2),Y,'LineWidth',1.5,'Color','k')%plot
X(1) = ctr(2,2)+0.001; X(2) = ctr(3,2); Y(1) = 2.25; Y(2) = 2.25;%sig bar
plot(X(1:2),Y,'LineWidth',1.5,'Color','k')%plot
X(1) = ctr(1,2); X(2) = ctr(2,2)-0.001; Y(1) = 2.25; Y(2) = 2.25;%sig bar
plot(X(1:2),Y,'LineWidth',1.5,'Color','k')%plot
X(1) = ctr(1,2); X(2) = ctr(3,2); Y(1) = 2.3; Y(2) = 2.3;%sig bar
plot(X(1:2),Y,'LineWidth',1.5,'Color','k')%plot
X(1) = ctr(2,3)+0.001; X(2) = ctr(3,3); Y(1) = 1.85; Y(2) = 1.85;%sig bar
plot(X(1:2),Y,'LineWidth',1.5,'Color','k')%plot
X(1) = ctr(1,3); X(2) = ctr(2,3)-0.001; Y(1) = 1.85; Y(2) = 1.85;%sig bar
plot(X(1:2),Y,'LineWidth',1.5,'Color','k')%plot
X(1) = ctr(1,3); X(2) = ctr(3,3); Y(1) = 1.9; Y(2) = 1.9;%sig bar
plot(X(1:2),Y,'LineWidth',1.5,'Color','k')%plot

%% volatility - decision time by difference

f = figure; %figure
ctr = []; ydt = []; %space
y = [(mean(minhigh_no_dt) - mean(min_no_dt))/mean(min_no_dt); (mean(lowhigh_no_dt) - mean(low_no_dt))/mean(low_no_dt); (mean(highhigh_no_dt) - mean(high_no_dt))/mean(high_no_dt)]; %DT diff
a_vert = [min_dt_diff_std; low_dt_diff_std; high_dt_diff_std]; %SEM
errorplus_vert=a_vert'; %positive SEM
errorminus_vert=errorplus_vert; %negative SEM
x = [1; 1.05; 1.1]; %x label
a_horz = [0.04;0.04;0.04]/2; %horiz lines
errorplus_horz=a_horz'; %positive horiz
errorminus_horz=errorplus_horz; %negative horiz
hBar = bar(x,y,'FaceColor',[1 1 1],'EdgeColor',[1 1 1]); %plot DT diff
for k1 = 1:size(y,2) %"for" loop going through DT diff
    ctr(k1,:) = bsxfun(@plus, hBar(k1).XData, hBar(k1).XOffset');      %ctr
    ydt(k1,:) = hBar(k1).YData;                    %ydt
end %ending "for" loop going through DT diff
h = hline(0,'w'); %block 0 line
h.LineWidth = 2; %line width
minnodt = [-0.36 -0.32 0.78 0.02 -0.26 0.1 0.07 -0.2 -0.08 -0.06 0.07]; %no conf ind
lownodt = [-0.5 -0.35 0.11 0.02 -0.15 -0.06 -0.07 -0.32 0.04 -0.04 -0.01]; %low conf ind
highnodt = [-0.35 -0.46 0.12 -0.5 -0.16 -0.18 -0.1 0.03 0.15 -0.08 -0.11]; %high conf ind
minnodt(2,:) = ctr(1,1); %center
lownodt(2,:) = ctr(1,2);%center
highnodt(2,:) = ctr(1,3);%center
hold on %hold fig
h = hline(0,'--k'); %mark 0%
h.Color = [0.64 0.08 0.18]; %color
c = [0.5 0.5 0.5]; %color
scatter(minnodt(2,:),minnodt(1,:),50,c); %scatter ind
scatter(lownodt(2,:),lownodt(1,:),50,c);%scatter ind
scatter(highnodt(2,:),highnodt(1,:),50,c);%scatter ind
e1 = errorbar(x(1), y(1), errorminus_vert(1), errorplus_vert(1), 'LineStyle', 'none', 'LineWidth', 2); %no conf diff
e1.Color = '[0.3 0.3 0.3]'; %color
hLine = line([x(1) - errorminus_horz(1), x(1) + errorplus_horz(1)], [y(1), y(1)], 'Color', get(e1, 'Color'), 'LineWidth', 2); %horiz
e2 = errorbar(x(2), y(2), errorminus_vert(2), errorplus_vert(2), 'LineStyle', 'none', 'LineWidth', 2); %low conf diff
e2.Color = '[0.3 0.3 0.3]';  %color
hLine = line([x(2) - errorminus_horz(2), x(2) + errorplus_horz(2)], [y(2), y(2)], 'Color', get(e2, 'Color'), 'LineWidth', 2);%horiz
e3 = errorbar(x(3), y(3), errorminus_vert(3), errorplus_vert(3), 'LineStyle', 'none', 'LineWidth', 2); %high conf diff
e3.Color = '[0.3 0.3 0.3]';  %color
hLine = line([x(3) - errorminus_horz(3), x(3) + errorplus_horz(3)], [y(3), y(3)], 'Color', get(e3, 'Color'), 'LineWidth', 2); %horiz
ylim([-0.6 0.8]) %ylim
xlim([0.97 1.13]) %xlim
XTickLabel = ({'No','Low','High'}); %x label
set(gca,'xtick',[1 1.05 1.1],'xticklabel',XTickLabel,'box','off','FontSize',20) %aesthetics
ylabel('Decision Time % Change') %y label

%% volatiltiy - time in center by difference

f = figure; %fig
ctr = []; ydt = []; %space
y = [(mean(minhigh_no_iti) - mean(min_no_iti))/mean(min_no_iti); (mean(lowhigh_no_iti) - mean(low_no_iti))/mean(low_no_iti); (mean(highhigh_no_iti) - mean(high_no_iti))/mean(high_no_iti)]; %time in center diff
a_vert = [min_iti_diff_std; low_iti_diff_std; high_iti_diff_std]; %SEM
errorplus_vert=a_vert'; %positive SEM 
errorminus_vert=errorplus_vert; %negative SEM
x = [1; 1.05; 1.1]; %x position
a_horz = [0.04;0.04;0.04]/2; %horizontal lines
errorplus_horz=a_horz'; %positive horiz
errorminus_horz=errorplus_horz; %negative horiz
hBar = bar(x,y,'FaceColor',[1 1 1],'EdgeColor',[1 1 1]); %plot time in center diff
for k1 = 1:size(y,2) %"for" loop going through time in center diff
    ctr(k1,:) = bsxfun(@plus, hBar(k1).XData, hBar(k1).XOffset');     %ctr
    ydt(k1,:) = hBar(k1).YData;                     %ydt
end %ending "for" loop going through time in center diff
h = hline(0,'w'); %block 0 line
h.LineWidth = 2; %line width
minnoiti = [-0.07 0.03 0.55 0.1 -0.15 0.07 -0.09 -0.06 0.1 -0.04 -0.095]; %no conf ind
lownoiti = [0.08 0.02 0.11 0.06 -0.13 -0.05 -0.12 -0.08 0.15 -0.055 0.05]; %low conf ind
highnoiti = [0.11 0 -0.08 0.04 -0.09 0.01 -0.05 0.115 0.24 -0.01 -0.06]; %high conf ind
minnoiti(2,:) = ctr(1,1); %center
lownoiti(2,:) = ctr(1,2);%center
highnoiti(2,:) = ctr(1,3);%center
hold on %hold fig
h = hline(0,'--k'); %dashed line for 0
h.Color = [0.64 0.08 0.18]; %line color
c = [0.5 0.5 0.5]; %color
scatter(minnoiti(2,:),minnoiti(1,:),50,c); %scatter ind
scatter(lownoiti(2,:),lownoiti(1,:),50,c);%scatter ind
scatter(highnoiti(2,:),highnoiti(1,:),50,c);%scatter ind
e1 = errorbar(x(1), y(1), errorminus_vert(1), errorplus_vert(1), 'LineStyle', 'none', 'LineWidth', 2); %no conf diff
e1.Color = '[0.3 0.3 0.3]'; %color
hLine = line([x(1) - errorminus_horz(1), x(1) + errorplus_horz(1)], [y(1), y(1)], 'Color', get(e1, 'Color'), 'LineWidth', 2); %horiz
e2 = errorbar(x(2), y(2), errorminus_vert(2), errorplus_vert(2), 'LineStyle', 'none', 'LineWidth', 2); %low conf diff
e2.Color = '[0.3 0.3 0.3]'; %color
hLine = line([x(2) - errorminus_horz(2), x(2) + errorplus_horz(2)], [y(2), y(2)], 'Color', get(e2, 'Color'), 'LineWidth', 2);%horiz
e3 = errorbar(x(3), y(3), errorminus_vert(3), errorplus_vert(3), 'LineStyle', 'none', 'LineWidth', 2); %high conf diff
e3.Color = '[0.3 0.3 0.3]'; %color
hLine = line([x(3) - errorminus_horz(3), x(3) + errorplus_horz(3)], [y(3), y(3)], 'Color', get(e3, 'Color'), 'LineWidth', 2); %horiz
ylim([-0.6 0.8]) %ylim
xlim([0.97 1.13]) %xlim
XTickLabel = ({'No','Low','High'}); %x label
set(gca,'xtick',[1 1.05 1.1],'xticklabel',XTickLabel,'box','off','FontSize',20) %aesthetics
ylabel('Time in Center % Change') %y label

%% volatiltiy - velocity as difference

f = figure; %figure
ctr = []; ydt = []; %space
y = [(mean(minhigh_no_vel) - mean(min_no_vel))/mean(min_no_vel); (mean(lowhigh_no_vel) - mean(low_no_vel))/mean(low_no_vel); (mean(highhigh_no_vel) - mean(high_no_vel))/mean(high_no_vel)]; %speed diff
a_vert = [min_vel_diff_std; low_vel_diff_std; high_vel_diff_std]; %SEM
errorplus_vert=a_vert'; %positive SEM
errorminus_vert=errorplus_vert; %negative SEM
x = [1; 1.05; 1.1]; %x position
a_horz = [0.04;0.04;0.04]/2; %horizontal lines
errorplus_horz=a_horz'; %positive horiz
errorminus_horz=errorplus_horz; %negative horiz
hBar = bar(x,y,'FaceColor',[1 1 1],'EdgeColor',[1 1 1]); %plot speed diff
for k1 = 1:size(y,2) %"for" loop going through speed diff
    ctr(k1,:) = bsxfun(@plus, hBar(k1).XData, hBar(k1).XOffset');     %ctr
    ydt(k1,:) = hBar(k1).YData;                    %ydt
end %ending "for" loop going through speed diff
h = hline(0,'w'); %block 0 line
h.LineWidth = 2; %line width
minnovel = [0.15 -0.04 -0.12 -0.05 0.06 -0.02 0.04 0.01 -0.01 0.02 -0.01]; %no conf ind
lownovel = [0.15 0.21 0.11 -0.02 0.06 0.01 0.05 0.04 -0.03 0.04 0]; %low conf ind
highnovel = [0.1 0.11 -0.04 -0.02 0.04 0 0.03 -0.04 -0.05 0.01 0.03]; %high conf ind
minnovel(2,:) = ctr(1,1); %center
lownovel(2,:) = ctr(1,2);%center
highnovel(2,:) = ctr(1,3);%center
hold on %hold fig
h = hline(0,'--k'); %line at 0
h.Color = [0.64 0.08 0.18]; %color for line
c = [0.5 0.5 0.5]; %color
scatter(minnovel(2,:),minnovel(1,:),50,c); %scatter ind
scatter(lownovel(2,:),lownovel(1,:),50,c);%scatter ind
scatter(highnovel(2,:),highnovel(1,:),50,c);%scatter ind
e1 = errorbar(x(1), y(1), errorminus_vert(1), errorplus_vert(1), 'LineStyle', 'none', 'LineWidth', 2); %no conf diff
e1.Color = '[0.3 0.3 0.3]';  %color
hLine = line([x(1) - errorminus_horz(1), x(1) + errorplus_horz(1)], [y(1), y(1)], 'Color', get(e1, 'Color'), 'LineWidth', 2); %horiz
e2 = errorbar(x(2), y(2), errorminus_vert(2), errorplus_vert(2), 'LineStyle', 'none', 'LineWidth', 2); %low conf diff
e2.Color = '[0.3 0.3 0.3]'; %color
hLine = line([x(2) - errorminus_horz(2), x(2) + errorplus_horz(2)], [y(2), y(2)], 'Color', get(e2, 'Color'), 'LineWidth', 2);%horiz
e3 = errorbar(x(3), y(3), errorminus_vert(3), errorplus_vert(3), 'LineStyle', 'none', 'LineWidth', 2); %high conf diff
e3.Color = '[0.3 0.3 0.3]'; %color
hLine = line([x(3) - errorminus_horz(3), x(3) + errorplus_horz(3)], [y(3), y(3)], 'Color', get(e3, 'Color'), 'LineWidth', 2); %horiz
ylim([-0.6 0.8]) %ylim
xlim([0.97 1.13]) %xlim
XTickLabel = ({'No','Low','High'}); %x label
set(gca,'xtick',[1 1.05 1.1],'xticklabel',XTickLabel,'box','off','FontSize',20) %aesthetics
ylabel('Speed at Hall Entry % Change') %y label

%% iSPN - percent correct

f = figure; %figure
ctr = []; ydt = []; %space
y = [mean(a2amin_no_c) mean(min_stim_c); mean(a2alow_no_c) mean(low_stim_c); mean(a2ahigh_no_c) mean(high_stim_c)]; %accuracy
a_vert = [std(a2amin_no_c)/sqrt(length(a2amin_no_c)) std(min_stim_c)/sqrt(length(min_stim_c)); std(a2alow_no_c)/sqrt(length(a2alow_no_c)) std(low_stim_c)/sqrt(length(low_stim_c)); std(a2ahigh_no_c)/sqrt(length(a2ahigh_no_c)) std(high_stim_c)/sqrt(length(high_stim_c))]; %SEM
errorplus_vert=a_vert'; %positive SEM
errorminus_vert=errorplus_vert; %negative SEM
x = [1; 1.1; 1.2]; %x position
a_horz = [0.01;0.01;0.01]; %horizontal line
errorplus_horz=a_horz'; %positive horiz
errorminus_horz=errorplus_horz; %negative horiz
hBar = bar(x,y,'FaceColor',[1 1 1],'EdgeColor',[1 1 1]); %plot percent correct
for k1 = 1:size(y,2) %"For" loop going through data
    ctr(k1,:) = bsxfun(@plus, hBar(k1).XData, hBar(k1).XOffset');     %ctr
    ydt(k1,:) = hBar(k1).YData;                    %ydt
end %ending "for" loop going through data
minnoc = [0.66 0.68 0.72 0.64 0.7 0.68 0.64 0.65 0.57 0.64]; %NC ind
minstim = [0.54 0.46 0.53 0.55 0.63 0.52 0.5 0.52 0.51 0.51]; %NS ind
lownoc = [0.68 0.7 0.69 0.62 0.65 0.61 0.61 0.66 0.58 0.61]; %LC ind
lowstim = [0.55 0.47 0.51 0.53 0.63 0.52 0.5 0.51 0.51 0.51]; %LS ind
highnoc = [0.59 0.59 0.61 0.54 0.61 0.55 0.56 0.59 0.56 0.57]; %HC ind
highstim = [0.52 0.44 0.53 0.50 0.57 0.52 0.52 0.49 0.51 0.51]; %HS ind
minnoc(2,:) = ctr(1,1); %center
minstim(2,:) = ctr(2,1);%center
lownoc(2,:) = ctr(1,2);%center
lowstim(2,:) = ctr(2,2);%center
highnoc(2,:) = ctr(1,3);%center
highstim(2,:) = ctr(2,3);%center
hold on %hold fit
for ugh = 1:10 %go through ind
    X(1) =  minnoc(2,ugh); %X control
    Y(1) =  minnoc(1,ugh); %Y control
    X(2) =  minstim(2,ugh); %X stim
    Y(2) =  minstim(1,ugh); %Y stim
    plot(X(1:2),Y,'Color','[0.7 0.7 0.7]') %plot ind
end %ending "for" loop going through ind
for ugh = 1:10%go through ind
    X(1) =  lownoc(2,ugh);%X control
    Y(1) =  lownoc(1,ugh);%Y control
    X(2) =  lowstim(2,ugh);%X stim
    Y(2) =  lowstim(1,ugh);%Y stim
    plot(X(1:2),Y,'Color','[0.7 0.7 0.7]')%plot ind
end%ending "for" loop going through ind
for ugh = 1:10%go through ind
    X(1) =  highnoc(2,ugh);%X control
    Y(1) =  highnoc(1,ugh);%Y control
    X(2) =  highstim(2,ugh);%X stim
    Y(2) =  highstim(1,ugh);%Y stim
    plot(X(1:2),Y,'Color','[0.7 0.7 0.7]')%plot ind
end%ending "for" loop going through ind
X(1) = ctr(1,1); X(2) = ctr(2,1); Y(1) = y(1,1); Y(2) = y(1,2); %plot avg
plot(X(1:2),Y,'LineWidth',1.5,'Color','k') %avg
X(1) = ctr(1,2); X(2) = ctr(2,2); Y(1) = y(2,1); Y(2) = y(2,2);%plot avg
plot(X(1:2),Y,'LineWidth',1.5,'Color','k')%avg
X(1) = ctr(1,3); X(2) = ctr(2,3); Y(1) = y(3,1); Y(2) = y(3,2);%plot avg
plot(X(1:2),Y,'LineWidth',1.5,'Color','k')%avg
e1 = errorbar(ctr(1,1), y(1,1),errorminus_vert(1,1),errorplus_vert(1,1),'LineStyle','none','LineWidth',2); %NC acc
e1.Color = '[0.99 0.58 0.58]'; %color
hLine = line([ctr(1,1) - errorminus_horz(1), ctr(1,1) + errorplus_horz(1)], [y(1,1), y(1,1)], 'Color', get(e1, 'Color'), 'LineWidth', 2); %horiz
e12 = errorbar(ctr(2,1), y(1,2),errorminus_vert(2,1),errorplus_vert(2,1),'LineStyle','none','LineWidth',2); %NS acc
e12.Color = 'r'; %color
hLine = line([ctr(2,1) - errorminus_horz(1), ctr(2,1) + errorplus_horz(1)], [y(1,2), y(1,2)], 'Color', get(e12, 'Color'), 'LineWidth', 2);%horiz
e2 = errorbar(ctr(1,2), y(2,1),errorminus_vert(1,2),errorplus_vert(1,2),'LineStyle','none','LineWidth',2); %LC acc
e2.Color = '[0.99 0.58 0.58]'; %color
hLine = line([ctr(1,2) - errorminus_horz(1), ctr(1,2) + errorplus_horz(1)], [y(2,1), y(2,1)], 'Color', get(e2, 'Color'), 'LineWidth', 2);%horiz
e22 = errorbar(ctr(2,2), y(2,2),errorminus_vert(2,2),errorplus_vert(2,2),'LineStyle','none','LineWidth',2); %LS acc
e22.Color = 'r'; %color
hLine = line([ctr(2,2) - errorminus_horz(1), ctr(2,2) + errorplus_horz(1)], [y(2,2), y(2,2)], 'Color', get(e22, 'Color'), 'LineWidth', 2);%horiz
e3 = errorbar(ctr(1,3), y(3,1),errorminus_vert(1,3),errorplus_vert(1,3),'LineStyle','none','LineWidth',2); %HC acc
e3.Color = '[0.99 0.58 0.58]'; %color
hLine = line([ctr(1,3) - errorminus_horz(1), ctr(1,3) + errorplus_horz(1)], [y(3,1), y(3,1)], 'Color', get(e3, 'Color'), 'LineWidth', 2);%horiz
e32 = errorbar(ctr(2,3), y(3,2),errorminus_vert(2,3),errorplus_vert(2,3),'LineStyle','none','LineWidth',2); %HS acc
e32.Color = 'r'; %color
hLine = line([ctr(2,3) - errorminus_horz(1), ctr(2,3) + errorplus_horz(1)], [y(3,2), y(3,2)], 'Color', get(e32, 'Color'), 'LineWidth', 2);%horiz
%ylim([0.3 0.85]) %original ylim
ylim([0 0.85]) %ylim axis to 0
XTickLabel = ({'No','Low','High'}); %x label
set(gca,'xtick',[1 1.1 1.2],'xticklabel',XTickLabel,'box','off','FontSize',20) %aesthetics
ylabel('% Correct') %y label
legend({'Ctrl','Stim'},'AutoUpdate','off') %legend
X(1) = ctr(1,1); X(2) = ctr(2,1); Y(1) = 0.73; Y(2) = 0.73; %sig bar
plot(X(1:2),Y,'LineWidth',1.5,'Color','k') %plot
X(1) = ctr(1,2); X(2) = ctr(2,2); Y(1) = 0.71; Y(2) = 0.71;%sig bar
plot(X(1:2),Y,'LineWidth',1.5,'Color','k')%plot
X(1) = ctr(1,3); X(2) = ctr(2,3); Y(1) = 0.62; Y(2) = 0.62;%sig bar
plot(X(1:2),Y,'LineWidth',1.5,'Color','k')%plot

%% iSPN - performance by block type

f = figure; %fig
ctr = []; ydt = []; %space
y = [mean(a2amin_no_left_percent) mean(min_stim_left_percent) mean(a2amin_no_right_percent) mean(min_stim_right_percent); mean(a2alow_no_left_percent) mean(low_stim_left_percent) mean(a2alow_no_right_percent) mean(low_stim_right_percent); mean(a2ahigh_no_left_percent) mean(high_stim_left_percent) mean(a2ahigh_no_right_percent) mean(high_stim_right_percent)]; %performance by block type
a_vert = [std(a2amin_no_left_percent)/sqrt(length(a2amin_no_left_percent)) std(min_stim_left_percent)/sqrt(length(min_stim_left_percent)) std(a2amin_no_right_percent)/sqrt(length(a2amin_no_right_percent)) std(min_stim_right_percent)/sqrt(length(min_stim_right_percent)); std(a2alow_no_left_percent)/sqrt(length(a2alow_no_left_percent)) std(low_stim_left_percent)/sqrt(length(low_stim_left_percent)) std(a2alow_no_right_percent)/sqrt(length(a2alow_no_right_percent)) std(low_stim_right_percent)/sqrt(length(low_stim_right_percent)); std(a2ahigh_no_left_percent)/sqrt(length(a2ahigh_no_left_percent)) std(high_stim_left_percent)/sqrt(length(high_stim_left_percent)) std(a2ahigh_no_right_percent)/sqrt(length(a2ahigh_no_right_percent)) std(high_stim_right_percent)/sqrt(length(high_stim_right_percent))]; %SEM
errorplus_vert=a_vert'; %positive SEM
errorminus_vert=errorplus_vert; %negative SEM
x = [1; 1.1; 1.2]; %x position
a_horz = [0.005;0.005;0.005]; %horizontal line
errorplus_horz=a_horz'; %positive horiz
errorminus_horz=errorplus_horz; %negative horiz
hBar = bar(x,y,'FaceColor',[1 1 1],'EdgeColor',[1 1 1]); %plot perf. by block type
for k1 = 1:size(y,2) %"for" loop going through "y"
    ctr(k1,:) = bsxfun(@plus, hBar(k1).XData, hBar(k1).XOffset');     %ctr
    ydt(k1,:) = hBar(k1).YData;                    %ydt
end %ending "for" loop going through "y"
minnowsL = [0.63 0.6 0.69 0.58 0.67 0.67 0.59 0.65 0.61 0.63]; %NC non pref ind
 minstimwsL = [0.58 0.64 0.56 0.6 0.67 0.53 0.44 0.49 0.54 0.5]; %NS non pref ind
 lownowsL = [0.66 0.59 0.68 0.6 0.64 0.6 0.58 0.67 0.6 0.61]; %LC non pref ind
 lowstimwsL = [0.53 0.64 0.49 0.57 0.63 0.48 0.48 0.5 0.48 0.48]; %LS non pref ind
 highnowsL = [0.54 0.52 0.56 0.53 0.59 0.53 0.5 0.58 0.62 0.56]; %HC non pref ind
 highstimwsL = [0.52 0.6 0.52 0.53 0.57 0.55 0.46 0.52 0.5 0.51]; %HS non pref ind
 minnowsR = [0.69 0.75 0.75 0.67 0.72 0.69 0.7 0.65 0.56 0.65]; %NC pref ind
 minstimwsR = [0.5 0.35 0.52 0.5 0.6 0.52 0.58 0.56 0.48 0.53]; %NS pref ind
 lownowsR = [0.67 0.75 0.69 0.64 0.66 0.62 0.65 0.65 0.56 0.62]; %LC pref ind
 lowstimwsR = [0.58 0.38 0.54 0.48 0.64 0.57 0.51 0.54 0.53 0.54]; %LS pref ind
 highnowsR = [0.63 0.66 0.65 0.56 0.62 0.59 0.65 0.61 0.51 0.59]; %HC pref ind
 highstimwsR = [0.52 0.39 0.54 0.51 0.58 0.5 0.59 0.49 0.53 0.52]; %HS pref ind
 minnowsL(2,:) = ctr(1,1); %center
 minstimwsL(2,:) = ctr(2,1);%center
 minnowsR(2,:) = ctr(3,1);%center
 minstimwsR(2,:) = ctr(4,1);%center
 lownowsL(2,:) = ctr(1,2);%center
 lowstimwsL(2,:) = ctr(2,2);%center
 lownowsR(2,:) = ctr(3,2);%center
 lowstimwsR(2,:) = ctr(4,2);%center
 highnowsL(2,:) = ctr(1,3);%center
 highstimwsL(2,:) = ctr(2,3);%center
 highnowsR(2,:) = ctr(3,3);%center
 highstimwsR(2,:) = ctr(4,3);%center
hold on %hold fig
scatter(minnowsL(2,:),minnowsL(1,:),30,[0.63 0.8 0.38]); %scatter ind
scatter(minstimwsL(2,:),minstimwsL(1,:),30,[0.47 0.67 0.19]);%scatter ind
scatter(lownowsL(2,:),lownowsL(1,:),30,[0.63 0.8 0.38]);%scatter ind
scatter(lowstimwsL(2,:),lowstimwsL(1,:),30,[0.47 0.67 0.19]);%scatter ind
scatter(highnowsL(2,:),highnowsL(1,:),30,[0.63 0.8 0.38]);%scatter ind
scatter(highstimwsL(2,:),highstimwsL(1,:),30,[0.47 0.67 0.19]);%scatter ind
scatter(minnowsR(2,:),minnowsR(1,:),30,[1 0.75 0.91]);%scatter ind
scatter(minstimwsR(2,:),minstimwsR(1,:),30,[0.98 0.51 0.8]);%scatter ind
scatter(lownowsR(2,:),lownowsR(1,:),30,[1 0.75 0.91]);%scatter ind
scatter(lowstimwsR(2,:),lowstimwsR(1,:),30,[0.98 0.51 0.8]);%scatter ind
scatter(highnowsR(2,:),highnowsR(1,:),30,[1 0.75 0.91]);%scatter ind
scatter(highstimwsR(2,:),highstimwsR(1,:),30,[0.98 0.51 0.8]);%scatter ind
X(1) = ctr(1,1); X(2) = ctr(2,1); Y(1) = y(1,1); Y(2) = y(1,2);%avgs
plot(X(1:2),Y,'LineWidth',1.5,'Color','k') %plot line
X(1) = ctr(3,1); X(2) = ctr(4,1); Y(1) = y(1,3); Y(2) = y(1,4);%avgs
plot(X(1:2),Y,'LineWidth',1.5,'Color','k')%plot line
X(1) = ctr(1,2); X(2) = ctr(2,2); Y(1) = y(2,1); Y(2) = y(2,2);%avgs
plot(X(1:2),Y,'LineWidth',1.5,'Color','k')%plot line
X(1) = ctr(3,2); X(2) = ctr(4,2); Y(1) = y(2,3); Y(2) = y(2,4);%avgs
plot(X(1:2),Y,'LineWidth',1.5,'Color','k')%plot line
X(1) = ctr(1,3); X(2) = ctr(2,3); Y(1) = y(3,1); Y(2) = y(3,2);%avgs
plot(X(1:2),Y,'LineWidth',1.5,'Color','k')%plot line
X(1) = ctr(3,3); X(2) = ctr(4,3); Y(1) = y(3,3); Y(2) = y(3,4);%avgs
plot(X(1:2),Y,'LineWidth',1.5,'Color','k')%plot line
e1 = errorbar(ctr(1,1), y(1,1),errorminus_vert(1,1),errorplus_vert(1,1),'LineStyle','none','LineWidth',2); %NC nonpref
e1.Color = '[0.64 0.84 0.48]'; %color
hLine = line([ctr(1,1) - errorminus_horz(1), ctr(1,1) + errorplus_horz(1)], [y(1,1), y(1,1)], 'Color', get(e1, 'Color'), 'LineWidth', 2); %horiz
e12 = errorbar(ctr(2,1), y(1,2),errorminus_vert(2,1),errorplus_vert(2,1),'LineStyle','none','LineWidth',2); %NS nonpref
e12.Color = '[0.29 0.41 0.1]'; %color
hLine = line([ctr(2,1) - errorminus_horz(1), ctr(2,1) + errorplus_horz(1)], [y(1,2), y(1,2)], 'Color', get(e12, 'Color'), 'LineWidth', 2);%horiz
e13 = errorbar(ctr(3,1), y(1,3),errorminus_vert(3,1),errorplus_vert(3,1),'LineStyle','none','LineWidth',2); %NC pref
e13.Color = '[0.98 0.67 0.86]'; %color
hLine = line([ctr(3,1) - errorminus_horz(1), ctr(3,1) + errorplus_horz(1)], [y(1,3), y(1,3)], 'Color', get(e13, 'Color'), 'LineWidth', 2);%horiz
e14 = errorbar(ctr(4,1), y(1,4),errorminus_vert(4,1),errorplus_vert(4,1),'LineStyle','none','LineWidth',2); %NS pref
e14.Color = '[1 0.07 0.65]';%color
hLine = line([ctr(4,1) - errorminus_horz(1), ctr(4,1) + errorplus_horz(1)], [y(1,4), y(1,4)], 'Color', get(e14, 'Color'), 'LineWidth', 2);%horiz
e2 = errorbar(ctr(1,2), y(2,1),errorminus_vert(1,2),errorplus_vert(1,2),'LineStyle','none','LineWidth',2); %LC nonpref
e2.Color = '[0.64 0.84 0.48]'; %color
hLine = line([ctr(1,2) - errorminus_horz(1), ctr(1,2) + errorplus_horz(1)], [y(2,1), y(2,1)], 'Color', get(e2, 'Color'), 'LineWidth', 2);%horiz
e22 = errorbar(ctr(2,2), y(2,2),errorminus_vert(2,2),errorplus_vert(2,2),'LineStyle','none','LineWidth',2); %LS nonpref
e22.Color = '[0.29 0.41 0.1]'; %color
hLine = line([ctr(2,2) - errorminus_horz(1), ctr(2,2) + errorplus_horz(1)], [y(2,2), y(2,2)], 'Color', get(e22, 'Color'), 'LineWidth', 2);%horiz
e23 = errorbar(ctr(3,2), y(2,3),errorminus_vert(3,2),errorplus_vert(3,2),'LineStyle','none','LineWidth',2); %LC pref
e23.Color = '[0.98 0.67 0.86]'; %color
hLine = line([ctr(3,2) - errorminus_horz(1), ctr(3,2) + errorplus_horz(1)], [y(2,3), y(2,3)], 'Color', get(e23, 'Color'), 'LineWidth', 2);%horiz
e24 = errorbar(ctr(4,2), y(2,4),errorminus_vert(4,2),errorplus_vert(4,2),'LineStyle','none','LineWidth',2); %LS pref
e24.Color = '[1 0.07 0.65]';%color
hLine = line([ctr(4,2) - errorminus_horz(1), ctr(4,2) + errorplus_horz(1)], [y(2,4), y(2,4)], 'Color', get(e24, 'Color'), 'LineWidth', 2);%horiz
e3 = errorbar(ctr(1,3), y(3,1),errorminus_vert(1,3),errorplus_vert(1,3),'LineStyle','none','LineWidth',2); %HC nonpref
e3.Color = '[0.64 0.84 0.48]'; %color
hLine = line([ctr(1,3) - errorminus_horz(1), ctr(1,3) + errorplus_horz(1)], [y(3,1), y(3,1)], 'Color', get(e3, 'Color'), 'LineWidth', 2);%horiz
e32 = errorbar(ctr(2,3), y(3,2),errorminus_vert(2,3),errorplus_vert(2,3),'LineStyle','none','LineWidth',2); %HS nonpref
e32.Color = '[0.29 0.41 0.1]'; %color
hLine = line([ctr(2,3) - errorminus_horz(1), ctr(2,3) + errorplus_horz(1)], [y(3,2), y(3,2)], 'Color', get(e32, 'Color'), 'LineWidth', 2);%horiz
e33 = errorbar(ctr(3,3), y(3,3),errorminus_vert(3,3),errorplus_vert(3,3),'LineStyle','none','LineWidth',2); %HC pref
e33.Color = '[0.98 0.67 0.86]'; %color
hLine = line([ctr(3,3) - errorminus_horz(1), ctr(3,3) + errorplus_horz(1)], [y(3,3), y(3,3)], 'Color', get(e33, 'Color'), 'LineWidth', 2);%horiz
e34 = errorbar(ctr(4,3), y(3,4),errorminus_vert(4,3),errorplus_vert(4,3),'LineStyle','none','LineWidth',2); %HS pref
e34.Color = '[1 0.07 0.65]';%color
hLine = line([ctr(4,3) - errorminus_horz(1), ctr(4,3) + errorplus_horz(1)], [y(3,4), y(3,4)], 'Color', get(e34, 'Color'), 'LineWidth', 2);%horiz
%ylim([0.3 0.85]) %original y lim
ylim([0 0.85]) %y lim axis to 0
XTickLabel = ({'No','Low','High'}); %x label
set(gca,'xtick',[1 1.1 1.2],'xticklabel',XTickLabel,'box','off','FontSize',20) %aesthetics
ylabel('% Correct') %y label
legend({'Ctrl Non-Pref','Stim Non-Pref','Ctrl Pref','Stim Pref'},'AutoUpdate','off') %legend
X(1) = ctr(3,1); X(2) = ctr(4,1); Y(1) = 0.76; Y(2) = 0.76; %sig bar
plot(X(1:2),Y,'LineWidth',1.5,'Color','k') %plot
X(1) = ctr(3,2); X(2) = ctr(4,2); Y(1) = 0.76; Y(2) = 0.76;%sig bar
plot(X(1:2),Y,'LineWidth',1.5,'Color','k')%plot
X(1) = ctr(3,3); X(2) = ctr(4,3); Y(1) = 0.67; Y(2) = 0.67;%sig bar
plot(X(1:2),Y,'LineWidth',1.5,'Color','k')%plot

%% iSPN - explore/exploit beta

f = figure; %fig
ctr = []; ydt = []; %space
y = [mean(ispn_noconflowvolctrl_explore_exploit) mean(ispn_noconflowvolstim_explore_exploit); mean(ispn_lowconflowvolctrl_explore_exploit) mean(ispn_lowconflowvolstim_explore_exploit); mean(ispn_highconflowvolctrl_explore_exploit) mean(ispn_highconflowvolstim_explore_exploit)]; %beta
a_vert = [std(ispn_noconflowvolctrl_explore_exploit)/sqrt(length(ispn_noconflowvolctrl_explore_exploit)) std(ispn_noconflowvolstim_explore_exploit)/sqrt(length(ispn_noconflowvolstim_explore_exploit)); std(ispn_lowconflowvolctrl_explore_exploit)/sqrt(length(ispn_lowconflowvolctrl_explore_exploit)) std(ispn_lowconflowvolstim_explore_exploit)/sqrt(length(ispn_lowconflowvolstim_explore_exploit)); std(ispn_highconflowvolctrl_explore_exploit)/sqrt(length(ispn_highconflowvolctrl_explore_exploit)) std(ispn_highconflowvolstim_explore_exploit)/sqrt(length(ispn_highconflowvolstim_explore_exploit))]; %SEM
errorplus_vert=a_vert'; %positive SEM
errorminus_vert=errorplus_vert; %negative SEM
x = [1; 1.1; 1.2]; %x positions
a_horz = [0.01;0.01;0.01]; %horizontal lines
errorplus_horz=a_horz'; %positive horiz
errorminus_horz=errorplus_horz; %negative horiz
hBar = bar(x,y,'FaceColor',[1 1 1],'EdgeColor',[1 1 1]); %plot beta
for k1 = 1:size(y,2) %"for" loop going through y
    ctr(k1,:) = bsxfun(@plus, hBar(k1).XData, hBar(k1).XOffset');     %ctr
    ydt(k1,:) = hBar(k1).YData;                    %ydt
end %ending "for" loop going through y
minnoc = [1.48 0.77 1.44 0.64 0.7 0.73 0.98 1.43 0.45 1.32 ]; %NC ind
minnostim = [1.12 0.48 0.43 0.66 0.1 0.31 0.37 0.17 0.43 0.17]; %NS ind
lownoc = [1.06 0.6 1.3 0.7 0.4 0.4 1.77 0.92 0.28 1.47]; %LC ind
lownostim = [1.02 0.64 0.35 0.44 0.47 0.3 0.11 0.2 0.16 0.45]; %LS ind
highnoc = [1.49 0.66 1.49 0.54 0.2 0.9 1.77 0.33 0.78 1.05 ]; %HC ind
highnostim = [0.61 0.65 0.3 0.52 0.27 0.39 0.17 0.03 0.45 0.43]; %HS ind
minnoc(2,:) = ctr(1,1); %center
minnostim(2,:) = ctr(2,1);%center
lownoc(2,:) = ctr(1,2);%center
lownostim(2,:) = ctr(2,2);%center
highnoc(2,:) = ctr(1,3);%center
highnostim(2,:) = ctr(2,3);%center
hold on %hold fig
for ugh = 1:10 %"for" loop going through ind
    X(1) =  minnoc(2,ugh); %control X
    Y(1) =  minnoc(1,ugh); %control Y
    X(2) =  minnostim(2,ugh); %stim X
    Y(2) =  minnostim(1,ugh); %stim Y
    plot(X(1:2),Y,'Color','[0.7 0.7 0.7]') %plot ind lines
end %ending "for" loop going through ind
for ugh = 1:10%"for" loop going through ind
    X(1) =  lownoc(2,ugh);%control X
    Y(1) =  lownoc(1,ugh);%control Y
    X(2) =  lownostim(2,ugh); %stim X
    Y(2) =  lownostim(1,ugh);%stim Y
    plot(X(1:2),Y,'Color','[0.7 0.7 0.7]')%plot ind lines
end%ending "for" loop going through ind
for ugh = 1:10 %"for" loop going through ind
    X(1) =  highnoc(2,ugh);%control X
    Y(1) =  highnoc(1,ugh);%control Y
    X(2) =  highnostim(2,ugh); %stim X
    Y(2) =  highnostim(1,ugh);%stim Y
    plot(X(1:2),Y,'Color','[0.7 0.7 0.7]')%plot ind lines
end%ending "for" loop going through ind
X(1) = ctr(1,1); X(2) = ctr(2,1); Y(1) = y(1,1); Y(2) = y(1,2); %avg
plot(X(1:2),Y,'LineWidth',1.5,'Color','k') %plot
X(1) = ctr(1,2); X(2) = ctr(2,2); Y(1) = y(2,1); Y(2) = y(2,2);%avg
plot(X(1:2),Y,'LineWidth',1.5,'Color','k')%plot
X(1) = ctr(1,3); X(2) = ctr(2,3); Y(1) = y(3,1); Y(2) = y(3,2);%avg
plot(X(1:2),Y,'LineWidth',1.5,'Color','k')%plot
e1 = errorbar(ctr(1,1), y(1,1),errorminus_vert(1,1),errorplus_vert(1,1),'LineStyle','none','LineWidth',2); %NC beta
e1.Color = '[0.99 0.58 0.58]'; %color
hLine = line([ctr(1,1) - errorminus_horz(1), ctr(1,1) + errorplus_horz(1)], [y(1,1), y(1,1)], 'Color', get(e1, 'Color'), 'LineWidth', 2); %horiz
e12 = errorbar(ctr(2,1), y(1,2),errorminus_vert(2,1),errorplus_vert(2,1),'LineStyle','none','LineWidth',2); %NS beta
e12.Color = 'r'; %color
hLine = line([ctr(2,1) - errorminus_horz(1), ctr(2,1) + errorplus_horz(1)], [y(1,2), y(1,2)], 'Color', get(e12, 'Color'), 'LineWidth', 2);%horiz
e2 = errorbar(ctr(1,2), y(2,1),errorminus_vert(1,2),errorplus_vert(1,2),'LineStyle','none','LineWidth',2); %LC beta
e2.Color = '[0.99 0.58 0.58]'; %color
hLine = line([ctr(1,2) - errorminus_horz(1), ctr(1,2) + errorplus_horz(1)], [y(2,1), y(2,1)], 'Color', get(e2, 'Color'), 'LineWidth', 2);%horiz
e22 = errorbar(ctr(2,2), y(2,2),errorminus_vert(2,2),errorplus_vert(2,2),'LineStyle','none','LineWidth',2); %LS beta
e22.Color = 'r'; %color
hLine = line([ctr(2,2) - errorminus_horz(1), ctr(2,2) + errorplus_horz(1)], [y(2,2), y(2,2)], 'Color', get(e22, 'Color'), 'LineWidth', 2);%horiz
e3 = errorbar(ctr(1,3), y(3,1),errorminus_vert(1,3),errorplus_vert(1,3),'LineStyle','none','LineWidth',2); %HC beta
e3.Color = '[0.99 0.58 0.58]'; %color
hLine = line([ctr(1,3) - errorminus_horz(1), ctr(1,3) + errorplus_horz(1)], [y(3,1), y(3,1)], 'Color', get(e3, 'Color'), 'LineWidth', 2);%horiz
e32 = errorbar(ctr(2,3), y(3,2),errorminus_vert(2,3),errorplus_vert(2,3),'LineStyle','none','LineWidth',2); %HS beta
e32.Color = 'r'; %color
hLine = line([ctr(2,3) - errorminus_horz(1), ctr(2,3) + errorplus_horz(1)], [y(3,2), y(3,2)], 'Color', get(e32, 'Color'), 'LineWidth', 2);%horiz
ylim([0 2]) %ylim
XTickLabel = ({'No','Low','High'}); %x label
set(gca,'xtick',[1 1.1 1.2],'xticklabel',XTickLabel,'box','off','FontSize',20) %aesthetics
ylabel('Explore         vs.         Exploit') %y label
legend({'Ctrl','Stim'},'AutoUpdate','off') %legend
X(1) = ctr(1,1); X(2) = ctr(2,1); Y(1) = 1.51; Y(2) = 1.51; %sig bar
plot(X(1:2),Y,'LineWidth',1.5,'Color','k') %plot
X(1) = ctr(1,2); X(2) = ctr(2,2); Y(1) = 1.8; Y(2) = 1.8;%sig bar
plot(X(1:2),Y,'LineWidth',1.5,'Color','k')%plot
X(1) = ctr(1,3); X(2) = ctr(2,3); Y(1) = 1.8; Y(2) = 1.8;%sig bar
plot(X(1:2),Y,'LineWidth',1.5,'Color','k')%plot

%% iSPN - bias

f = figure; %figure
ctr = []; ydt = []; %space
y = [mean(ispn_noconflowvolctrl_Q_bias) mean(ispn_noconflowvolstim_Q_bias); mean(ispn_lowconflowvolctrl_Q_bias) mean(ispn_lowconflowvolstim_Q_bias); mean(ispn_highconflowvolctrl_Q_bias) mean(ispn_highconflowvolstim_Q_bias)]; %bias
a_vert = [std(ispn_noconflowvolctrl_Q_bias)/sqrt(length(ispn_noconflowvolctrl_Q_bias)) std(ispn_noconflowvolstim_Q_bias)/sqrt(length(ispn_noconflowvolstim_Q_bias)); std(ispn_lowconflowvolctrl_Q_bias)/sqrt(length(ispn_lowconflowvolctrl_Q_bias)) std(ispn_lowconflowvolstim_Q_bias)/sqrt(length(ispn_lowconflowvolstim_Q_bias)); std(ispn_highconflowvolctrl_Q_bias)/sqrt(length(ispn_highconflowvolctrl_Q_bias)) std(ispn_highconflowvolstim_Q_bias)/sqrt(length(ispn_highconflowvolstim_Q_bias))]; %SEM
errorplus_vert=a_vert'; %positive SEM
errorminus_vert=errorplus_vert; %negative SEM
x = [1; 1.1; 1.2]; %x positions
a_horz = [0.01;0.01;0.01]; %horizontal lines
errorplus_horz=a_horz'; %positive horiz
errorminus_horz=errorplus_horz; %negative horiz
hBar = bar(x,y,'FaceColor',[1 1 1],'EdgeColor',[1 1 1]); %plot bias
for k1 = 1:size(y,2) %"for" loop going through y
    ctr(k1,:) = bsxfun(@plus, hBar(k1).XData, hBar(k1).XOffset');      %ctr
    ydt(k1,:) = hBar(k1).YData;                    %ydt
end %ending "for" loop going through y
h = hline(0,'w'); %block 0 line
h.LineWidth = 2; %line width
minnoc = [0.01 0.67 0.64 0.05 0.08 0.15 0.08 0.77 0.02 0.22 ]; %NC ind
minnostim = [-0.34 -0.39 0 -0.12 0.06 0.18 -0.15 -0.38 -0.37 -0.34]; %NS ind
lownoc = [0.56 0.92 0.02 0.07 0.26 0.45 0.45 0.55 0.08 0.09]; %LC ind
lownostim = [-0.52 -0.63 -0.47 -0.12 -0.03 -0.11 -0.01 -0.1 -0.05 0.02]; %LS ind
highnoc = [0.86 1.07 0.86 0.21 0.11 0.33 0.16 0.68 0.03 0.46]; %HC ind
highnostim = [-0.57 -0.08 0.44 -0.12 0.01 0.24 -0.03 -0.11 -0.04 -0.07]; %HS ind
minnoc(2,:) = ctr(1,1); %center
minnostim(2,:) = ctr(2,1);%center
lownoc(2,:) = ctr(1,2);%center
lownostim(2,:) = ctr(2,2);%center
highnoc(2,:) = ctr(1,3);%center
highnostim(2,:) = ctr(2,3);%center
hold on %hold fig
for ugh = 1:10 %"for" loop going through ind
    X(1) =  minnoc(2,ugh); %ctrl X
    Y(1) =  minnoc(1,ugh); %ctrl Y
    X(2) =  minnostim(2,ugh); %stim X
    Y(2) =  minnostim(1,ugh); %stim Y
    plot(X(1:2),Y,'Color','[0.7 0.7 0.7]') %plot individual lines
end %ending "for" loop going through ind
for ugh = 1:10%"for" loop going through ind
    X(1) =  lownoc(2,ugh);%ctrl X
    Y(1) =  lownoc(1,ugh);%ctrl Y
    X(2) =  lownostim(2,ugh);%stim X
    Y(2) =  lownostim(1,ugh);%stim Y
    plot(X(1:2),Y,'Color','[0.7 0.7 0.7]')%plot individual lines
end%ending "for" loop going through ind
for ugh = 1:10%"for" loop going through ind
    X(1) =  highnoc(2,ugh);%ctrl X
    Y(1) =  highnoc(1,ugh);%ctrl Y
    X(2) =  highnostim(2,ugh);%stim X
    Y(2) =  highnostim(1,ugh);%stim Y
    plot(X(1:2),Y,'Color','[0.7 0.7 0.7]')%plot individual lines
end%ending "for" loop going through ind
X(1) = ctr(1,1); X(2) = ctr(2,1); Y(1) = y(1,1); Y(2) = y(1,2); %avg
plot(X(1:2),Y,'LineWidth',1.5,'Color','k') %plot
X(1) = ctr(1,2); X(2) = ctr(2,2); Y(1) = y(2,1); Y(2) = y(2,2);%avg
plot(X(1:2),Y,'LineWidth',1.5,'Color','k')%plot
X(1) = ctr(1,3); X(2) = ctr(2,3); Y(1) = y(3,1); Y(2) = y(3,2);%avg
plot(X(1:2),Y,'LineWidth',1.5,'Color','k')%plot
e1 = errorbar(ctr(1,1), y(1,1),errorminus_vert(1,1),errorplus_vert(1,1),'LineStyle','none','LineWidth',2); %NC bias
e1.Color = '[0.99 0.58 0.58]'; %color
hLine = line([ctr(1,1) - errorminus_horz(1), ctr(1,1) + errorplus_horz(1)], [y(1,1), y(1,1)], 'Color', get(e1, 'Color'), 'LineWidth', 2); %horiz
e12 = errorbar(ctr(2,1), y(1,2),errorminus_vert(2,1),errorplus_vert(2,1),'LineStyle','none','LineWidth',2); %NS bias
e12.Color = 'r'; %color
hLine = line([ctr(2,1) - errorminus_horz(1), ctr(2,1) + errorplus_horz(1)], [y(1,2), y(1,2)], 'Color', get(e12, 'Color'), 'LineWidth', 2);%horiz
e2 = errorbar(ctr(1,2), y(2,1),errorminus_vert(1,2),errorplus_vert(1,2),'LineStyle','none','LineWidth',2); %LC bias
e2.Color = '[0.99 0.58 0.58]'; %color
hLine = line([ctr(1,2) - errorminus_horz(1), ctr(1,2) + errorplus_horz(1)], [y(2,1), y(2,1)], 'Color', get(e2, 'Color'), 'LineWidth', 2);%horiz
e22 = errorbar(ctr(2,2), y(2,2),errorminus_vert(2,2),errorplus_vert(2,2),'LineStyle','none','LineWidth',2); %LS bias
e22.Color = 'r'; %color
hLine = line([ctr(2,2) - errorminus_horz(1), ctr(2,2) + errorplus_horz(1)], [y(2,2), y(2,2)], 'Color', get(e22, 'Color'), 'LineWidth', 2);%horiz
e3 = errorbar(ctr(1,3), y(3,1),errorminus_vert(1,3),errorplus_vert(1,3),'LineStyle','none','LineWidth',2); %HC bias
e3.Color = '[0.99 0.58 0.58]'; %color
hLine = line([ctr(1,3) - errorminus_horz(1), ctr(1,3) + errorplus_horz(1)], [y(3,1), y(3,1)], 'Color', get(e3, 'Color'), 'LineWidth', 2);%horiz
e32 = errorbar(ctr(2,3), y(3,2),errorminus_vert(2,3),errorplus_vert(2,3),'LineStyle','none','LineWidth',2); %HS bias
e32.Color = 'r'; %color
hLine = line([ctr(2,3) - errorminus_horz(1), ctr(2,3) + errorplus_horz(1)], [y(3,2), y(3,2)], 'Color', get(e32, 'Color'), 'LineWidth', 2);%horiz
ylim([-1 1.5]) %ylim
XTickLabel = ({'No','Low','High'}); %x label
set(gca,'xtick',[1 1.1 1.2],'xticklabel',XTickLabel,'box','off','FontSize',20) %aesthetics
ylabel('Bias') %y label
legend({'Ctrl','Stim'},'AutoUpdate','off') %legend
X(1) = ctr(1,1); X(2) = ctr(2,1); Y(1) = 0.8; Y(2) = 0.8; %sig bar
plot(X(1:2),Y,'LineWidth',1.5,'Color','k') %plot
X(1) = ctr(1,2); X(2) = ctr(2,2); Y(1) = 0.95; Y(2) = 0.95; %sig bar
plot(X(1:2),Y,'LineWidth',1.5,'Color','k')%plot
X(1) = ctr(1,3); X(2) = ctr(2,3); Y(1) = 1.08; Y(2) = 1.08; %sig bar
plot(X(1:2),Y,'LineWidth',1.5,'Color','k')%plot

%% iSPN - rew vs unrew LR

f = figure; %figure
ctr = []; ydt = []; %space
y = [mean(a2amin_no_rew_lr) mean(min_stim_rew_lr) mean(a2amin_no_unrew_lr) mean(min_stim_unrew_lr); mean(a2alow_no_rew_lr) mean(low_stim_rew_lr) mean(a2alow_no_unrew_lr) mean(low_stim_unrew_lr); mean(a2ahigh_no_rew_lr) mean(high_stim_rew_lr) mean(a2ahigh_no_unrew_lr) mean(high_stim_unrew_lr)]; %rew vs unrew LR
a_vert = [std(a2amin_no_rew_lr)/sqrt(length(a2amin_no_rew_lr)) std(min_stim_rew_lr)/sqrt(length(min_stim_rew_lr)) std(a2amin_no_unrew_lr)/sqrt(length(a2amin_no_unrew_lr)) std(min_stim_unrew_lr)/sqrt(length(min_stim_unrew_lr)); std(a2alow_no_rew_lr)/sqrt(length(a2alow_no_rew_lr)) std(low_stim_rew_lr)/sqrt(length(low_stim_rew_lr)) std(a2alow_no_unrew_lr)/sqrt(length(a2alow_no_unrew_lr)) std(low_stim_unrew_lr)/sqrt(length(low_stim_unrew_lr)); std(a2ahigh_no_rew_lr)/sqrt(length(a2ahigh_no_rew_lr)) std(high_stim_rew_lr)/sqrt(length(high_stim_rew_lr)) std(a2ahigh_no_unrew_lr)/sqrt(length(a2ahigh_no_unrew_lr)) std(high_stim_unrew_lr)/sqrt(length(high_stim_unrew_lr))]; %SEM
errorplus_vert=a_vert'; %positive SEM
errorminus_vert=errorplus_vert; %negative SEM
x = [1; 1.1; 1.2]; %x position
a_horz = [0.005;0.005;0.005]; %horizontal lines
errorplus_horz=a_horz'; %positive horiz
errorminus_horz=errorplus_horz; %negative horiz
hBar = bar(x,y,'FaceColor',[1 1 1],'EdgeColor',[1 1 1]); %plot LRs
for k1 = 1:size(y,2) %"for" loop going through y
    ctr(k1,:) = bsxfun(@plus, hBar(k1).XData, hBar(k1).XOffset');      %ctr
    ydt(k1,:) = hBar(k1).YData;                     %ydt
end %ending "for" loop going thorugh Y
h = hline(0,'w'); %block 0 line
h.LineWidth = 2; %line width
minnowsL = [0.7 0.55 1 0.71 1 1 1 1 0.38 0.89]; %NC rew ind
 minstimwsL = [0 0.06 0 0.55 0.15 0.76 0 0 0 0]; %NS rew ind
 lownowsL = [0.87 0.56 1 0.62 0.79 1 0.8 0.86 0.8 0.83]; %LC rew ind
 lowstimwsL = [0.2 0 0 0 0 0.02 0 0 0.16 0]; %LS rew ind
 highnowsL = [0.28 0.51 1 0.85 1 0.76 1 1 0.89 0.92]; %HC rew ind
 highstimwsL = [0.44 0 0 0.1 0.62 0 0.52 0.01 0 0]; %HS rew ind
 minnowsR = [0.65 1 1 0.62 1 1 0.87 0.68 0.78 0.95]; %NC unrew ind
 minstimwsR = [0.05 0.05 0.63 0.33 0.86 0.01 0.52 0.78 0.22 0]; %NS unrew ind
 lownowsR = [0.83 0.82 0.94 0.85 0.84 0.99 1 0.95 0.84 0.94]; %LC unrew ind
 lowstimwsR = [0.41 0.08 0.62 0.01 0.98 0 0.02 0.07 0.1 0.01]; %LS unrew ind
 highnowsR = [0.45 0.47 0.92 0.45 0.71 0.72 0.65 0.93 0.48 0.66]; %HC unrew ind
 highstimwsR = [0.62 0.01 0 0 0.63 0 0.02 0.29 0.07 0.03]; %HS unrew ind
 minnowsL(2,:) = ctr(1,1); %center
 minstimwsL(2,:) = ctr(2,1);%center
 minnowsR(2,:) = ctr(3,1);%center
 minstimwsR(2,:) = ctr(4,1);%center
 lownowsL(2,:) = ctr(1,2);%center
 lowstimwsL(2,:) = ctr(2,2);%center
 lownowsR(2,:) = ctr(3,2);%center
 lowstimwsR(2,:) = ctr(4,2);%center
 highnowsL(2,:) = ctr(1,3);%center
 highstimwsL(2,:) = ctr(2,3);%center
 highnowsR(2,:) = ctr(3,3);%center
 highstimwsR(2,:) = ctr(4,3);%center
hold on %hold on
scatter(minnowsL(2,:),minnowsL(1,:),30,[0.62 0.85 0.94]); %scatter ind
scatter(minstimwsL(2,:),minstimwsL(1,:),30,[0 0.45 0.74]);%scatter ind
scatter(lownowsL(2,:),lownowsL(1,:),30,[0.62 0.85 0.94]);%scatter ind
scatter(lowstimwsL(2,:),lowstimwsL(1,:),30,[0 0.45 0.74]);%scatter ind
scatter(highnowsL(2,:),highnowsL(1,:),30,[0.62 0.85 0.94]);%scatter ind
scatter(highstimwsL(2,:),highstimwsL(1,:),30,[0 0.45 0.74]);%scatter ind
scatter(minnowsR(2,:),minnowsR(1,:),30,[0.8 0.8 0.8]);%scatter ind
scatter(minstimwsR(2,:),minstimwsR(1,:),30,[0.6 0.6 0.6]);%scatter ind
scatter(lownowsR(2,:),lownowsR(1,:),30,[0.8 0.8 0.8]);%scatter ind
scatter(lowstimwsR(2,:),lowstimwsR(1,:),30,[0.6 0.6 0.6]);%scatter ind
scatter(highnowsR(2,:),highnowsR(1,:),30,[0.8 0.8 0.8]);%scatter ind
scatter(highstimwsR(2,:),highstimwsR(1,:),30,[0.6 0.6 0.6]);%scatter ind
X(1) = ctr(1,1); X(2) = ctr(2,1); Y(1) = y(1,1); Y(2) = y(1,2); %avg
plot(X(1:2),Y,'LineWidth',1.5,'Color','k') %plot
X(1) = ctr(3,1); X(2) = ctr(4,1); Y(1) = y(1,3); Y(2) = y(1,4);%avg
plot(X(1:2),Y,'LineWidth',1.5,'Color','k')%plot
X(1) = ctr(1,2); X(2) = ctr(2,2); Y(1) = y(2,1); Y(2) = y(2,2);%avg
plot(X(1:2),Y,'LineWidth',1.5,'Color','k')%plot
X(1) = ctr(3,2); X(2) = ctr(4,2); Y(1) = y(2,3); Y(2) = y(2,4);%avg
plot(X(1:2),Y,'LineWidth',1.5,'Color','k')%plot
X(1) = ctr(1,3); X(2) = ctr(2,3); Y(1) = y(3,1); Y(2) = y(3,2);%avg
plot(X(1:2),Y,'LineWidth',1.5,'Color','k')%plot
X(1) = ctr(3,3); X(2) = ctr(4,3); Y(1) = y(3,3); Y(2) = y(3,4);%avg
plot(X(1:2),Y,'LineWidth',1.5,'Color','k')%plot
e1 = errorbar(ctr(1,1), y(1,1),errorminus_vert(1,1),errorplus_vert(1,1),'LineStyle','none','LineWidth',2); %NC rew
e1.Color = '[0.3 0.75 0.93]'; %color
hLine = line([ctr(1,1) - errorminus_horz(1), ctr(1,1) + errorplus_horz(1)], [y(1,1), y(1,1)], 'Color', get(e1, 'Color'), 'LineWidth', 2); %horiz
e12 = errorbar(ctr(2,1), y(1,2),errorminus_vert(2,1),errorplus_vert(2,1),'LineStyle','none','LineWidth',2); %NS rew
e12.Color = 'b'; %color
hLine = line([ctr(2,1) - errorminus_horz(1), ctr(2,1) + errorplus_horz(1)], [y(1,2), y(1,2)], 'Color', get(e12, 'Color'), 'LineWidth', 2);%horiz
e13 = errorbar(ctr(3,1), y(1,3),errorminus_vert(3,1),errorplus_vert(3,1),'LineStyle','none','LineWidth',2); %NC unrew
e13.Color = '[0.7 0.7 0.7]'; %color
hLine = line([ctr(3,1) - errorminus_horz(1), ctr(3,1) + errorplus_horz(1)], [y(1,3), y(1,3)], 'Color', get(e13, 'Color'), 'LineWidth', 2);%horiz
e14 = errorbar(ctr(4,1), y(1,4),errorminus_vert(4,1),errorplus_vert(4,1),'LineStyle','none','LineWidth',2); %NS unrew
e14.Color = '[0.5 0.5 0.5]';%color
hLine = line([ctr(4,1) - errorminus_horz(1), ctr(4,1) + errorplus_horz(1)], [y(1,4), y(1,4)], 'Color', get(e14, 'Color'), 'LineWidth', 2);%horiz
e2 = errorbar(ctr(1,2), y(2,1),errorminus_vert(1,2),errorplus_vert(1,2),'LineStyle','none','LineWidth',2); %LC rew
e2.Color = '[0.3 0.75 0.93]'; %color
hLine = line([ctr(1,2) - errorminus_horz(1), ctr(1,2) + errorplus_horz(1)], [y(2,1), y(2,1)], 'Color', get(e2, 'Color'), 'LineWidth', 2);%horiz
e22 = errorbar(ctr(2,2), y(2,2),errorminus_vert(2,2),errorplus_vert(2,2),'LineStyle','none','LineWidth',2); %LS rew
e22.Color = 'b'; %color
hLine = line([ctr(2,2) - errorminus_horz(1), ctr(2,2) + errorplus_horz(1)], [y(2,2), y(2,2)], 'Color', get(e22, 'Color'), 'LineWidth', 2);%horiz
e23 = errorbar(ctr(3,2), y(2,3),errorminus_vert(3,2),errorplus_vert(3,2),'LineStyle','none','LineWidth',2); %LC unrew
e23.Color = '[0.7 0.7 0.7]'; %color
hLine = line([ctr(3,2) - errorminus_horz(1), ctr(3,2) + errorplus_horz(1)], [y(2,3), y(2,3)], 'Color', get(e23, 'Color'), 'LineWidth', 2);%horiz
e24 = errorbar(ctr(4,2), y(2,4),errorminus_vert(4,2),errorplus_vert(4,2),'LineStyle','none','LineWidth',2); %LS unrew
e24.Color = '[0.5 0.5 0.5]';%color
hLine = line([ctr(4,2) - errorminus_horz(1), ctr(4,2) + errorplus_horz(1)], [y(2,4), y(2,4)], 'Color', get(e24, 'Color'), 'LineWidth', 2);%horiz
e3 = errorbar(ctr(1,3), y(3,1),errorminus_vert(1,3),errorplus_vert(1,3),'LineStyle','none','LineWidth',2); %HC rew
e3.Color = '[0.3 0.75 0.93]'; %color
hLine = line([ctr(1,3) - errorminus_horz(1), ctr(1,3) + errorplus_horz(1)], [y(3,1), y(3,1)], 'Color', get(e3, 'Color'), 'LineWidth', 2);%horiz
e32 = errorbar(ctr(2,3), y(3,2),errorminus_vert(2,3),errorplus_vert(2,3),'LineStyle','none','LineWidth',2); %HS rew
e32.Color = 'b'; %color
hLine = line([ctr(2,3) - errorminus_horz(1), ctr(2,3) + errorplus_horz(1)], [y(3,2), y(3,2)], 'Color', get(e32, 'Color'), 'LineWidth', 2);%horiz
e33 = errorbar(ctr(3,3), y(3,3),errorminus_vert(3,3),errorplus_vert(3,3),'LineStyle','none','LineWidth',2); %HC unrew
e33.Color = '[0.7 0.7 0.7]'; %color
hLine = line([ctr(3,3) - errorminus_horz(1), ctr(3,3) + errorplus_horz(1)], [y(3,3), y(3,3)], 'Color', get(e33, 'Color'), 'LineWidth', 2);%horiz
e34 = errorbar(ctr(4,3), y(3,4),errorminus_vert(4,3),errorplus_vert(4,3),'LineStyle','none','LineWidth',2); %HS unrew
e34.Color = '[0.5 0.5 0.5]';%color
hLine = line([ctr(4,3) - errorminus_horz(1), ctr(4,3) + errorplus_horz(1)], [y(3,4), y(3,4)], 'Color', get(e34, 'Color'), 'LineWidth', 2);%horiz
ylim([-0.2 1.4]) %ylim
XTickLabel = ({'No','Low','High'}); %x label
set(gca,'xtick',[1 1.1 1.2],'xticklabel',XTickLabel,'ytick',[0:0.5:1],'box','off','FontSize',20) %aesthetics
ylabel('Learning Rate') %y label
legend({'Ctrl Rew','Stim Rew','Ctrl Unrew','Stim Unrew'},'AutoUpdate','off') %legend
X(1) = ctr(1,1); X(2) = ctr(2,1); Y(1) = 1.03; Y(2) = 1.03; %sig bar
plot(X(1:2),Y,'LineWidth',1.5,'Color','k') %plot
X(1) = ctr(3,1); X(2) = ctr(4,1); Y(1) = 1.03; Y(2) = 1.03;%sig bar
plot(X(1:2),Y,'LineWidth',1.5,'Color','k')%plot
X(1) = ctr(1,2); X(2) = ctr(2,2); Y(1) = 1.03; Y(2) = 1.03;%sig bar
plot(X(1:2),Y,'LineWidth',1.5,'Color','k')%plot
X(1) = ctr(3,2); X(2) = ctr(4,2); Y(1) = 1.03; Y(2) = 1.03;%sig bar
plot(X(1:2),Y,'LineWidth',1.5,'Color','k')%plot
X(1) = ctr(1,3); X(2) = ctr(2,3); Y(1) = 1.03; Y(2) = 1.03;%sig bar
plot(X(1:2),Y,'LineWidth',1.5,'Color','k')%plot
X(1) = ctr(3,3); X(2) = ctr(4,3); Y(1) = 0.96; Y(2) = 0.96;%sig bar
plot(X(1:2),Y,'LineWidth',1.5,'Color','k')%plot

%% iSPN - scrambled model

f = figure; %figure
ctr = []; ydt = []; %space
y = [mean(a2amin_no_r1) mean(min_stim_r1) mean(noconflowvolscram(:,1)) mean(a2amin_no_c1) mean(min_stim_c1) mean(noconflowvolscram(:,2)); mean(a2alow_no_r1) mean(low_stim_r1) mean(lowconflowvolscram(:,1)) mean(a2alow_no_c1) mean(low_stim_c1) mean(lowconflowvolscram(:,2)); mean(a2ahigh_no_r1) mean(high_stim_r1) mean(highconflowvolscram(:,1)) mean(a2ahigh_no_c1) mean(high_stim_c1) mean(highconflowvolscram(:,2))]; %regression weight
a_vert = [std(a2amin_no_r1)/sqrt(length(a2amin_no_r1)) std(min_stim_r1)/sqrt(length(min_stim_r1)) std(noconflowvolscram(:,1))/sqrt(length(noconflowvolscram(:,1))) std(a2amin_no_c1)/sqrt(length(a2amin_no_c1)) std(min_stim_c1)/sqrt(length(min_stim_c1)) std(noconflowvolscram(:,2))/sqrt(length(noconflowvolscram(:,2))); std(a2alow_no_r1)/sqrt(length(a2alow_no_r1)) std(low_stim_r1)/sqrt(length(low_stim_r1)) std(lowconflowvolscram(:,1))/sqrt(length(lowconflowvolscram(:,1))) std(a2alow_no_c1)/sqrt(length(a2alow_no_c1)) std(low_stim_c1)/sqrt(length(low_stim_c1)) std(lowconflowvolscram(:,2))/sqrt(length(lowconflowvolscram(:,2))); std(a2ahigh_no_r1)/sqrt(length(a2ahigh_no_r1)) std(high_stim_r1)/sqrt(length(high_stim_r1)) std(highconflowvolscram(:,1))/sqrt(length(highconflowvolscram(:,1))) std(a2ahigh_no_c1)/sqrt(length(a2ahigh_no_c1)) std(high_stim_c1)/sqrt(length(high_stim_c1)) std(highconflowvolscram(:,2))/sqrt(length(highconflowvolscram(:,2)))]; %SEM
errorplus_vert=a_vert'; %positive SEM
errorminus_vert=errorplus_vert; %negative SEM
x = [1; 1.1; 1.2]; %x positions
a_horz = [0.005;0.005;0.005]/2; %horizontal line
errorplus_horz=a_horz'; %positive horiz
errorminus_horz=errorplus_horz; %negative horiz
hBar = bar(x,y,'FaceColor',[1 1 1],'EdgeColor',[1 1 1]); %plot regression weights
for k1 = 1:size(y,2) %"for" loop going through y
    ctr(k1,:) = bsxfun(@plus, hBar(k1).XData, hBar(k1).XOffset');     %ctr
    ydt(k1,:) = hBar(k1).YData;                    %ydt
end %ending "for" loop going through y
h = hline(0,'w'); %blocking 0 line
h.LineWidth = 2; %line width
minnoR1 = [0.42 0.66 1 0.48 0.89 0.75 0.65 0.68 0.31 0.61]; %NC outcome
 minstimR1 = [0.22 0.06 0.23 0.19 0.62 0.17 0.09 0.11 0.15 0.13]; %NS outcome
 minscramR1 = noconflowvolscram_ind(1:10,1); %NM outcome
 lownoR1 = [0.62 0.77 0.93 0.41 0.73 0.5 0.5 0.9 0.34 0.55]; %LC outocme
 lowstimR1 = [0.29 -0.04 0.13 0.11 0.58 0.15 0.04 0.15 0 0.09]; %LS outcome
 lowscramR1 = lowconflowvolscram_ind(1:10,1); %LM outcome
 highnoR1 = [0.39 0.76 0.89 0.28 0.77 0.54 0.56 0.7 0.45 0.56]; %HC outcome
 highstimR1 = [0.23 0.05 0.3 0.11 0.6 0.17 -0.01 0.07 0.13 0.1]; %HS outcome
 highscramR1 = highconflowvolscram_ind(1:10,1); %HM outcome
 minnoC1 = [0.88 1.02 0.62 0.27 0.15 0.83 0.97 0.46 0.05 0.59]; %NC choice
 minstimC1 = [0.13 0.33 -0.04 0.2 0.33 -0.03 -0.08 0.73 0.15 -0.18]; %NS choice
 minscramC1 = noconflowvolscram_ind(1:10,2); %NM choice
 lownoC1 = [0.87 1.01 0.7 0.42 0.09 0.78 0.98 0.61 0.01 0.62]; %LC choice
  lowstimC1 = [0.21 0.12 -0.12 0.24 0.14 0.01 -0.05 -0.34 -0.04 -0.11]; %LS choice
 lowscramC1 = lowconflowvolscram_ind(1:10,2); %LM choice
 highnoC1 = [0.92 0.88 0.44 0.41 0.04 0.79 0.98 0.55 -0.01 0.58]; %HC choice
 highstimC1 = [0.37 0.22 0.07 0.17 -0.04 -0.03 -0.04 -0.08 -0.14 -0.07]; %HS choice
 highscramC1 = highconflowvolscram_ind(1:10,2); %HM choice
 minnoR1(2,:) = ctr(1,1); %center
 minstimR1(2,:) = ctr(2,1); %center
 minscramR1(2,:) = ctr(3,1); %center
 minnoC1(2,:) = ctr(4,1); %center
 minstimC1(2,:) = ctr(5,1); %center
 minscramC1(2,:) = ctr(6,1); %center
 lownoR1(2,:) = ctr(1,2); %center
 lowstimR1(2,:) = ctr(2,2); %center
 lowscramR1(2,:) = ctr(3,2); %center
 lownoC1(2,:) = ctr(4,2); %center
 lowstimC1(2,:) = ctr(5,2); %center
lowscramC1(2,:) = ctr(6,2); %center
 highnoR1(2,:) = ctr(1,3); %center
 highstimR1(2,:) = ctr(2,3); %center
 highscramR1(2,:) = ctr(3,3); %center
 highnoC1(2,:) = ctr(4,3); %center
highstimC1(2,:) = ctr(5,3); %center
highscramC1(2,:) = ctr(6,3); %center
hold on %hold on
scatter(minnoR1(2,:),minnoR1(1,:),30,[0.62 0.85 0.94]); %scatter ind
scatter(minstimR1(2,:),minstimR1(1,:),30,[0 0.45 0.74]); %scatter ind
scatter(minscramR1(2,:),minscramR1(1,:),30,[0.8 0.8 0.8]); %scatter ind
scatter(lownoR1(2,:),lownoR1(1,:),30,[0.62 0.85 0.94]); %scatter ind
scatter(lowstimR1(2,:),lowstimR1(1,:),30,[0 0.45 0.74]); %scatter ind
scatter(lowscramR1(2,:),lowscramR1(1,:),30,[0.8 0.8 0.8]); %scatter ind
scatter(highnoR1(2,:),highnoR1(1,:),30,[0.62 0.85 0.94]); %scatter ind
scatter(highstimR1(2,:),highstimR1(1,:),30,[0 0.45 0.74]); %scatter ind
scatter(highscramR1(2,:),highscramR1(1,:),30,[0.8 0.8 0.8]); %scatter ind
scatter(minnoC1(2,:),minnoC1(1,:),30,[0.78 0.62 0.88]); %scatter ind
scatter(minstimC1(2,:),minstimC1(1,:),30,[0.83 0.55 1]); %scatter ind
scatter(minscramC1(2,:),minscramC1(1,:),30,[0.74 0.51 0.58]); %scatter ind
scatter(lownoC1(2,:),lownoC1(1,:),30,[0.78 0.62 0.88]); %scatter ind
scatter(lowstimC1(2,:),lowstimC1(1,:),30,[0.83 0.55 1]); %scatter ind
scatter(lowscramC1(2,:),lowscramC1(1,:),30,[0.74 0.51 0.58]); %scatter ind
scatter(highnoC1(2,:),highnoC1(1,:),30,[0.78 0.62 0.88]); %scatter ind
scatter(highstimC1(2,:),highstimC1(1,:),30,[0.83 0.55 1]); %scatter ind
scatter(highscramC1(2,:),highscramC1(1,:),30,[0.74 0.51 0.58]); %scatter ind
e1 = errorbar(ctr(1,1), y(1,1),errorminus_vert(1,1),errorplus_vert(1,1),'LineStyle','none','LineWidth',2);  %NC outcome
e1.Color = '[0.3 0.75 0.93]'; %color
hLine = line([ctr(1,1) - errorminus_horz(1), ctr(1,1) + errorplus_horz(1)], [y(1,1), y(1,1)], 'Color', get(e1, 'Color'), 'LineWidth', 2); %horiz
e12 = errorbar(ctr(2,1), y(1,2),errorminus_vert(2,1),errorplus_vert(2,1),'LineStyle','none','LineWidth',2); %NS outcome
e12.Color = 'b';  %color
hLine = line([ctr(2,1) - errorminus_horz(1), ctr(2,1) + errorplus_horz(1)], [y(1,2), y(1,2)], 'Color', get(e12, 'Color'), 'LineWidth', 2);%horiz
e13 = errorbar(ctr(3,1), y(1,3),errorminus_vert(3,1),errorplus_vert(3,1),'LineStyle','none','LineWidth',2); %NM outcome
e13.Color = '[0 0.45 0.74]';  %color
hLine = line([ctr(3,1) - errorminus_horz(1), ctr(3,1) + errorplus_horz(1)], [y(1,3), y(1,3)], 'Color', get(e13, 'Color'), 'LineWidth', 2);%horiz
e14 = errorbar(ctr(4,1), y(1,4),errorminus_vert(4,1),errorplus_vert(4,1),'LineStyle','none','LineWidth',2); %NC choice
e14.Color = '[0.89 0.79 1]'; %color
hLine = line([ctr(4,1) - errorminus_horz(1), ctr(4,1) + errorplus_horz(1)], [y(1,4), y(1,4)], 'Color', get(e14, 'Color'), 'LineWidth', 2);%horiz
e15 = errorbar(ctr(5,1), y(1,5),errorminus_vert(5,1),errorplus_vert(5,1),'LineStyle','none','LineWidth',2); %NS choice
e15.Color = '[0.72 0.27 1]';  %color
hLine = line([ctr(5,1) - errorminus_horz(1), ctr(5,1) + errorplus_horz(1)], [y(1,5), y(1,5)], 'Color', get(e15, 'Color'), 'LineWidth', 2);%horiz
e16 = errorbar(ctr(6,1), y(1,6),errorminus_vert(6,1),errorplus_vert(6,1),'LineStyle','none','LineWidth',2); %NM choice
e16.Color = '[0.49 0.18 0.56]'; %color
hLine = line([ctr(6,1) - errorminus_horz(1), ctr(6,1) + errorplus_horz(1)], [y(1,6), y(1,6)], 'Color', get(e16, 'Color'), 'LineWidth', 2);%horiz
e2 = errorbar(ctr(1,2), y(2,1),errorminus_vert(1,2),errorplus_vert(1,2),'LineStyle','none','LineWidth',2); %LC outcome
e2.Color = '[0.3 0.75 0.93]';  %color
hLine = line([ctr(1,2) - errorminus_horz(1), ctr(1,2) + errorplus_horz(1)], [y(2,1), y(2,1)], 'Color', get(e2, 'Color'), 'LineWidth', 2);%horiz
e22 = errorbar(ctr(2,2), y(2,2),errorminus_vert(2,2),errorplus_vert(2,2),'LineStyle','none','LineWidth',2); %LS outcome
e22.Color = 'b';  %color
hLine = line([ctr(2,2) - errorminus_horz(1), ctr(2,2) + errorplus_horz(1)], [y(2,2), y(2,2)], 'Color', get(e22, 'Color'), 'LineWidth', 2);%horiz
e23 = errorbar(ctr(3,2), y(2,3),errorminus_vert(3,2),errorplus_vert(3,2),'LineStyle','none','LineWidth',2); %LM outcome
e23.Color = '[0 0.45 0.74]';  %color
hLine = line([ctr(3,2) - errorminus_horz(1), ctr(3,2) + errorplus_horz(1)], [y(2,3), y(2,3)], 'Color', get(e23, 'Color'), 'LineWidth', 2);%horiz
e24 = errorbar(ctr(4,2), y(2,4),errorminus_vert(4,2),errorplus_vert(4,2),'LineStyle','none','LineWidth',2); %LC choice
e24.Color = '[0.89 0.79 1]'; %color
hLine = line([ctr(4,2) - errorminus_horz(1), ctr(4,2) + errorplus_horz(1)], [y(2,4), y(2,4)], 'Color', get(e24, 'Color'), 'LineWidth', 2);%horiz
e25 = errorbar(ctr(5,2), y(2,5),errorminus_vert(5,2),errorplus_vert(5,2),'LineStyle','none','LineWidth',2); %LS choice
e25.Color = '[0.72 0.27 1]';  %color
hLine = line([ctr(5,2) - errorminus_horz(1), ctr(5,2) + errorplus_horz(1)], [y(2,5), y(2,5)], 'Color', get(e25, 'Color'), 'LineWidth', 2);%horiz
e26 = errorbar(ctr(6,2), y(2,6),errorminus_vert(6,2),errorplus_vert(6,2),'LineStyle','none','LineWidth',2); %LM choice
e26.Color = '[0.49 0.18 0.56]'; %color
hLine = line([ctr(6,2) - errorminus_horz(1), ctr(6,2) + errorplus_horz(1)], [y(2,6), y(2,6)], 'Color', get(e26, 'Color'), 'LineWidth', 2);%horiz
e3 = errorbar(ctr(1,3), y(3,1),errorminus_vert(1,3),errorplus_vert(1,3),'LineStyle','none','LineWidth',2); %HC outcome
e3.Color = '[0.3 0.75 0.93]';  %color
hLine = line([ctr(1,3) - errorminus_horz(1), ctr(1,3) + errorplus_horz(1)], [y(3,1), y(3,1)], 'Color', get(e3, 'Color'), 'LineWidth', 2);%horiz
e32 = errorbar(ctr(2,3), y(3,2),errorminus_vert(2,3),errorplus_vert(2,3),'LineStyle','none','LineWidth',2); %HS outcome
e32.Color = 'b';  %color
hLine = line([ctr(2,3) - errorminus_horz(1), ctr(2,3) + errorplus_horz(1)], [y(3,2), y(3,2)], 'Color', get(e32, 'Color'), 'LineWidth', 2);%horiz
e33 = errorbar(ctr(3,3), y(3,3),errorminus_vert(3,3),errorplus_vert(3,3),'LineStyle','none','LineWidth',2); %HM outcome
e33.Color = '[0 0.45 0.74]';  %color
hLine = line([ctr(3,3) - errorminus_horz(1), ctr(3,3) + errorplus_horz(1)], [y(3,3), y(3,3)], 'Color', get(e33, 'Color'), 'LineWidth', 2);%horiz
e34 = errorbar(ctr(4,3), y(3,4),errorminus_vert(4,3),errorplus_vert(4,3),'LineStyle','none','LineWidth',2); %HC choice
e34.Color = '[0.89 0.79 1]'; %color
hLine = line([ctr(4,3) - errorminus_horz(1), ctr(4,3) + errorplus_horz(1)], [y(3,4), y(3,4)], 'Color', get(e34, 'Color'), 'LineWidth', 2);%horiz
e35 = errorbar(ctr(5,3), y(3,5),errorminus_vert(5,3),errorplus_vert(5,3),'LineStyle','none','LineWidth',2); %HS choice
e35.Color = '[0.72 0.27 1]';  %color
hLine = line([ctr(5,3) - errorminus_horz(1), ctr(5,3) + errorplus_horz(1)], [y(3,5), y(3,5)], 'Color', get(e35, 'Color'), 'LineWidth', 2);%horiz
e36 = errorbar(ctr(6,3), y(3,6),errorminus_vert(6,3),errorplus_vert(6,3),'LineStyle','none','LineWidth',2); %HM choice
e36.Color = '[0.49 0.18 0.56]'; %color
hLine = line([ctr(6,3) - errorminus_horz(1), ctr(6,3) + errorplus_horz(1)], [y(3,6), y(3,6)], 'Color', get(e36, 'Color'), 'LineWidth', 2);%horiz
ylim([-0.4 1.5]) %ylim
XTickLabel = ({'No','Low','High'}); %x label
set(gca,'xtick',[1 1.1 1.2],'xticklabel',XTickLabel,'box','off','FontSize',20) %aesthetics
ylabel('Regression Weight') %y label
legend({'Ctrl Outcome','Stim Outcome','Scrambled Outcome','Ctrl Choice','Stim Choice','Scrambled Choice'},'AutoUpdate','off') %legend

%% iSPN - decision time

f = figure; %figure
if size(a2amin_no_dt,1) == 1 %if the orientaiton is wrong...
    a2amin_no_dt = a2amin_no_dt'; %transpose
    min_stim_dt = min_stim_dt';%transpose
    a2alow_no_dt = a2alow_no_dt';%transpose
    low_stim_dt = low_stim_dt';%transpose
    a2ahigh_no_dt = a2ahigh_no_dt';%transpose
    high_stim_dt = high_stim_dt';%transpose
end %ending "if" statement to fix orientation
distributionPlot(a2amin_no_dt,'XValue',1,'histOri','left','color',[0.97 0.57 0.57],'widthDiv',[2 1]); %NC DT
hold on %hold fig
distributionPlot(a2alow_no_dt,'XValue',2,'histOri','left','color',[0.97 0.57 0.57],'widthDiv',[2 1]) %LC DT
distributionPlot(a2ahigh_no_dt,'XValue',3,'histOri','left','color',[0.97 0.57 0.57],'widthDiv',[2 1]) %HC DT
distributionPlot(min_stim_dt,'XValue',1,'histOri','right','color','r','widthDiv',[2 2]) %NS DT
distributionPlot(low_stim_dt,'XValue',2,'histOri','right','color','r','widthDiv',[2 2]) %LS DT
distributionPlot(high_stim_dt,'XValue',3,'histOri','right','color','r','widthDiv',[2 2]) %HS DT
XTickLabel = ({'No','Low','High'}); %x label
YTickLabel = ({'1','2','3'}); %y label
set(gca,'xlim',[0 4],'xtick',[1:1:3],'xticklabel',XTickLabel,'ytick',[1000:1000:3000],'yticklabel',YTickLabel,'fontsize',15) %aesthetics
ylim([0 3000]) %ylim
ylabel('Decision Time (s)') %y label
child = get(gca,'children'); %children for legend
legend([child(8),child(2)],'Ctrl','Stim','AutoUpdate','off') %legend

%% iSPN - vel profile

f = figure; %figure
subplot(1,3,1) %subplot 1
hold on %make the fig
nonan = []; %place to take out NaNs
for avging = 1:60 %go through our 60 points
    nonan = ispn_ctrl_start(:,avging); %pull out trial
    nonan = nonan(~isnan(nonan)); %take out NaN
    avg2(1,avging) = mean(nonan); %avg it
    avg2(2,avging) = std(nonan)/sqrt(length(nonan)); %get the std
end %ending "for" loop going through our 6 points
test = []; %clearing
test(1,:) = (avg2(1,:)); %the avg
y = test(1,:)/1.5; % your mean vector
x = 1:numel(y); %how many inputs are in y
std_dev = avg2(2,:)/1.5; %std
curve1 = y + std_dev; %upper bound
curve2 = y - std_dev; %lower bound
x2 = [x, fliplr(x)]; %x2
inBetween = [curve1, fliplr(curve2)]; %here are the in between values
fi = fill(x2, inBetween,[0.8 0.8 0.8]); %plotting the shading w/ color specified
set(fi,'edgecolor','white'); %take away edges
hold on; %hold plot
p1 = plot(x, y, 'LineWidth', 2); %plot the line
p1.Color = 'k'; %color
nonan = []; %place to take out NaNs
for avging = 1:60 %go through our 60 points
    nonan = ispn_stim_start(:,avging); %pull out trial
    nonan = nonan(~isnan(nonan)); %take out NaN
    avg2(1,avging) = mean(nonan); %avg it
    avg2(2,avging) = std(nonan)/sqrt(length(nonan)); %get the std
end %ending "for" loop going through our 6 points
hold on %hold it
test(1,:) = (avg2(1,:)); %the avg
y = test(1,:)/1.5; % your mean vector
x = 1:numel(y); %how many inputs are in y
std_dev = avg2(2,:)/1.5; %std
curve1 = y + std_dev; %upper bound
curve2 = y - std_dev; %lower bound
x2 = [x, fliplr(x)]; %x2
inBetween = [curve1, fliplr(curve2)]; %here are the in between values
fi = fill(x2, inBetween,[0.99 0.73 0.73],'facealpha', .6); %plotting the shading w/ color specified
set(fi,'edgecolor','white'); %take away edges
hold on; %hold plot
p1 = plot(x, y, 'LineWidth', 2); %plot the line
p1.Color = 'r'; %color
XTickLabel = ({'-500','Trial Start','+500'}); %x label
set(gca,'xtick',[1 30 60],'xticklabel',XTickLabel,'box','off','fontsize',20) %format
ylabel('Speed (cm/s)'); 
%ylim([1 8]) %original y lim
ylim([0 8]) %y lim axis to 0
xline(30,'--g','LineWidth',2); %alignment line
siggies = []; %where to store sig values
for x = 1:60 %go through each time point
    [h,p] = ttest2(ispn_ctrl_start(:,x),ispn_stim_start(:,x)); %test
    siggies(x) = p; %copy the p
end %ending "for" loop going through time points
for x = 1:length(siggies) %go through sig values
    if siggies(x) < 0.0014 %lowest p
        siggies(x) = 3; %mark with a 3
    elseif siggies(x) > 0.001 && siggies(x) < 0.014 %middle p
        siggies(x) = 1; %mark with a 1
    elseif siggies(x) > 0.01 && siggies(x) < 0.054 %highest p
        siggies(x) = 1; %mark with a 1
    else %if no p...
        siggies(x) = 0; %mark with a 0
    end %ending "if" statements looking for p values
end %ending "for" loop going through sig values
patches = []; %storage
s=1; %formatting
if siggies(1) ~= 0 %if there is a starting sig...
    patches(1,1) = 1; %mark it with the starting sig
    patches(2,1) = siggies(1); %mark the place
    s=2; %format
end %ending "if" statement looking for starting sig
for x = 2:length(siggies) %go through sigs
    if siggies(x) ~= siggies(x-1) %if we have a new sig...
        patches(1,s) = x; %mark the sig value
        patches(2,s) = siggies(x); %mark the location
        s=s+1; %formatting
    end %ending "if" statement looking for new sigs
end %ending "for" loop going through sigs
patches(1,s) = 60; %make the last one 60
patches(2,s) = 0; %make the last one no sig to end it
if size(patches,2) > 1 %if we have some sig...
    for sigmark = 1:length(patches) %going through the info for making the patches
        if patches(2,sigmark) == 1 %if we have a 1...
            color = [1 0.88 0.59]; %set the color
            p1 = patch([patches(1,sigmark) patches(1,sigmark+1) patches(1,sigmark+1) patches(1,sigmark)], [max(ylim)-0.5 max(ylim)-0.5 max(ylim) max(ylim)], color); %draw the patch
            p1.EdgeColor = [1 1 1]; %take off the edges
            hold on %hold
        elseif patches(2,sigmark) == 2 %if we have a 2...
            color = [0.91 0.34 0.1]; %set the orange
            p1 = patch([patches(1,sigmark) patches(1,sigmark+1) patches(1,sigmark+1) patches(1,sigmark)], [max(ylim)-0.5 max(ylim)-0.5 max(ylim) max(ylim)], color); %draw the patch
            p1.EdgeColor = [1 1 1]; %take off the edges
            hold on %hold
        elseif patches(2,sigmark) == 3 %if we have a 3...
            color = [0.93 0.63 0.19]; %set the red
            p1 = patch([patches(1,sigmark) patches(1,sigmark+1) patches(1,sigmark+1) patches(1,sigmark)], [max(ylim)-0.5 max(ylim)-0.5 max(ylim) max(ylim)], color); %draw the patch
            p1.EdgeColor = [1 1 1]; %take off the edges
            hold on     %hold
        end %ending "if" statement looking for sig
    end %ending "for" loop going through to make patches
end %ending "if" statement looking for sigs
subplot(1,3,2) %subplot 2
hold on %make the fig
nonan = []; %place to take out NaNs
for avging = 1:60 %go through our 60 points
    nonan = ispn_ctrl_center(:,avging); %pull out trial
    nonan = nonan(~isnan(nonan)); %take out NaN
    avg2(1,avging) = mean(nonan); %avg it
    avg2(2,avging) = std(nonan)/sqrt(length(nonan)); %get the std
end %ending "for" loop going through our 6 points
test = []; %clearing
test(1,:) = (avg2(1,:)); %the avg
y = test(1,:)/1.5; % your mean vector
x = 1:numel(y); %how many inputs are in y
std_dev = avg2(2,:)/1.5; %std
curve1 = y + std_dev; %upper bound
curve2 = y - std_dev; %lower bound
x2 = [x, fliplr(x)]; %x2
inBetween = [curve1, fliplr(curve2)]; %here are the in between values
fi = fill(x2, inBetween,[0.8 0.8 0.8]); %plotting the shading w/ color specified
set(fi,'edgecolor','white'); %take away edges
hold on; %hold plot
p1 = plot(x, y, 'LineWidth', 2); %plot the line
p1.Color = 'k'; %color
nonan = []; %place to take out NaNs
for avging = 1:60 %go through our 60 points
    nonan = ispn_stim_center(:,avging); %pull out trial
    nonan = nonan(~isnan(nonan)); %take out NaN
    avg2(1,avging) = mean(nonan); %avg it
    avg2(2,avging) = std(nonan)/sqrt(length(nonan)); %get the std
end %ending "for" loop going through our 6 points
hold on %hold it
test(1,:) = (avg2(1,:)); %the avg
y = test(1,:)/1.5; % your mean vector
x = 1:numel(y); %how many inputs are in y
std_dev = avg2(2,:)/1.5; %std
curve1 = y + std_dev; %upper bound
curve2 = y - std_dev; %lower bound
x2 = [x, fliplr(x)]; %x2
inBetween = [curve1, fliplr(curve2)]; %here are the in between values
fi = fill(x2, inBetween,[0.99 0.73 0.73],'facealpha', .6); %plotting the shading w/ color specified
set(fi,'edgecolor','white'); %take away edges
hold on; %hold plot
p1 = plot(x, y, 'LineWidth', 2); %plot the line
p1.Color = 'r'; %color
XTickLabel = ({'-500','Center Entry','+500'}); %x label
set(gca,'xtick',[1 30 60],'xticklabel',XTickLabel,'box','off','fontsize',20) %format
%ylim([1 8]) %original y lim
ylim([0 8]) %y lim axis to 0
xline(30,'--g','LineWidth',2); %alignment line
siggies = []; %where to store sig values
for x = 1:60 %go through each time point
    [h,p] = ttest2(ispn_ctrl_center(:,x),ispn_stim_center(:,x)); %test
    siggies(x) = p; %copy the p
end %ending "for" loop going through time points
for x = 1:length(siggies) %go through sig values
    if siggies(x) < 0.0014 %lowest p
        siggies(x) = 3; %mark with a 3
    elseif siggies(x) > 0.001 && siggies(x) < 0.014 %middle p
        siggies(x) = 1; %mark with a 1
    elseif siggies(x) > 0.01 && siggies(x) < 0.054 %highest p
        siggies(x) = 1; %mark with a 1
    else %if no p...
        siggies(x) = 0; %mark with a 0
    end %ending "if" statements looking for p values
end %ending "for" loop going through sig values
patches = []; %storage
s=1; %formatting
if siggies(1) ~= 0 %if there is a starting sig...
    patches(1,1) = 1; %mark it with the starting sig
    patches(2,1) = siggies(1); %mark the place
    s=2; %format
end %ending "if" statement looking for starting sig
for x = 2:length(siggies) %go through sigs
    if siggies(x) ~= siggies(x-1) %if we have a new sig...
        patches(1,s) = x; %mark the sig value
        patches(2,s) = siggies(x); %mark the location
        s=s+1; %formatting
    end %ending "if" statement looking for new sigs
end %ending "for" loop going through sigs
patches(1,s) = 60; %make the last one 60
patches(2,s) = 0; %make the last one no sig to end it
if size(patches,2) > 1 %if we have some sig...
    for sigmark = 1:length(patches) %going through the info for making the patches
        if patches(2,sigmark) == 1 %if we have a 1...
            color = [1 0.88 0.59]; %set the color
            p1 = patch([patches(1,sigmark) patches(1,sigmark+1) patches(1,sigmark+1) patches(1,sigmark)], [max(ylim)-0.5 max(ylim)-0.5 max(ylim) max(ylim)], color); %draw the patch
            p1.EdgeColor = [1 1 1]; %take off the edges
            hold on %hold
        elseif patches(2,sigmark) == 2 %if we have a 2...
            color = [0.91 0.34 0.1]; %set the orange
            p1 = patch([patches(1,sigmark) patches(1,sigmark+1) patches(1,sigmark+1) patches(1,sigmark)], [max(ylim)-0.5 max(ylim)-0.5 max(ylim) max(ylim)], color); %draw the patch
            p1.EdgeColor = [1 1 1]; %take off the edges
            hold on %hold
        elseif patches(2,sigmark) == 3 %if we have a 3...
            color = [0.93 0.63 0.19]; %set the red
            p1 = patch([patches(1,sigmark) patches(1,sigmark+1) patches(1,sigmark+1) patches(1,sigmark)], [max(ylim)-0.5 max(ylim)-0.5 max(ylim) max(ylim)], color); %draw the patch
            p1.EdgeColor = [1 1 1]; %take off the edges
            hold on     %hold
        end %ending "if" statement looking for sig
    end %ending "for" loop going through to make patches
end %ending "if" statement looking for sigs
subplot(1,3,3) %subplot 3
hold on %make the fig
nonan = []; %place to take out NaNs
for avging = 1:60 %go through our 60 points
    nonan = ispn_ctrl_laser(:,avging); %pull out trial
    nonan = nonan(~isnan(nonan)); %take out NaN
    avg2(1,avging) = mean(nonan); %avg it
    avg2(2,avging) = std(nonan)/sqrt(length(nonan)); %get the std
end %ending "for" loop going through our 6 points
test = []; %clearing
test(1,:) = (avg2(1,:)); %the avg
y = test(1,:)/1.5; % your mean vector
x = 1:numel(y); %how many inputs are in y
std_dev = avg2(2,:)/1.5; %std
curve1 = y + std_dev; %upper bound
curve2 = y - std_dev; %lower bound
x2 = [x, fliplr(x)]; %x2
inBetween = [curve1, fliplr(curve2)]; %here are the in between values
fi = fill(x2, inBetween,[0.8 0.8 0.8]); %plotting the shading w/ color specified
set(fi,'edgecolor','white'); %take away edges
hold on; %hold plot
p1 = plot(x, y, 'LineWidth', 2); %plot the line
p1.Color = 'k'; %color
nonan = []; %place to take out NaNs
for avging = 1:60 %go through our 60 points
    nonan = ispn_stim_laser(:,avging); %pull out trial
    nonan = nonan(~isnan(nonan)); %take out NaN
    avg2(1,avging) = mean(nonan); %avg it
    avg2(2,avging) = std(nonan)/sqrt(length(nonan)); %get the std
end %ending "for" loop going through our 6 points
hold on %hold it
test(1,:) = (avg2(1,:)); %the avg
y = test(1,:)/1.5; % your mean vector
x = 1:numel(y); %how many inputs are in y
std_dev = avg2(2,:)/1.5; %std
curve1 = y + std_dev; %upper bound
curve2 = y - std_dev; %lower bound
x2 = [x, fliplr(x)]; %x2
inBetween = [curve1, fliplr(curve2)]; %here are the in between values
fi = fill(x2, inBetween,[0.99 0.73 0.73],'facealpha', .6); %plotting the shading w/ color specified
set(fi,'edgecolor','white'); %take away edges
hold on; %hold plot
p1 = plot(x, y, 'LineWidth', 2); %plot the line
p1.Color = 'r'; %color
XTickLabel = ({'Laser','+800'}); %x label
set(gca,'xtick',[12 60],'xticklabel',XTickLabel,'box','off','fontsize',20) %format
%ylim([1 8]) %original y lim
ylim([0 8]) %y lim axis to 0
xline(12,'--g','LineWidth',2); %alignment line
color = [0.64 0.85 1]; %color for stim patch
p1 = patch([12 48 48 12], [min(ylim) min(ylim) max(ylim) max(ylim)], color); %draw stim patch
p1.FaceAlpha = 0.2; p1.EdgeColor = [1 1 1]; %translucence
siggies = []; %where to store sig values
for x = 1:60 %go through each time point
    [h,p] = ttest2(ispn_ctrl_laser(:,x),ispn_stim_laser(:,x)); %test
    siggies(x) = p; %copy the p
end %ending "for" loop going through time points
for x = 1:length(siggies) %go through sig values
    if siggies(x) < 0.0014 %lowest p
        siggies(x) = 3; %mark with a 3
    elseif siggies(x) > 0.001 && siggies(x) < 0.014 %middle p
        siggies(x) = 1; %mark with a 1
    elseif siggies(x) > 0.01 && siggies(x) < 0.054 %highest p
        siggies(x) = 1; %mark with a 1
    else %if no p...
        siggies(x) = 0; %mark with a 0
    end %ending "if" statements looking for p values
end %ending "for" loop going through sig values
patches = []; %storage
s=1; %formatting
if siggies(1) ~= 0 %if there is a starting sig...
    patches(1,1) = 1; %mark it with the starting sig
    patches(2,1) = siggies(1); %mark the place
    s=2; %format
end %ending "if" statement looking for starting sig
for x = 2:length(siggies) %go through sigs
    if siggies(x) ~= siggies(x-1) %if we have a new sig...
        patches(1,s) = x; %mark the sig value
        patches(2,s) = siggies(x); %mark the location
        s=s+1; %formatting
    end %ending "if" statement looking for new sigs
end %ending "for" loop going through sigs
patches(1,s) = 60; %make the last one 60
patches(2,s) = 0; %make the last one no sig to end it
if size(patches,2) > 1 %if we have some sig...
    for sigmark = 1:length(patches) %going through the info for making the patches
        if patches(2,sigmark) == 1 %if we have a 1...
            color = [1 0.88 0.59]; %set the color
            p1 = patch([patches(1,sigmark) patches(1,sigmark+1) patches(1,sigmark+1) patches(1,sigmark)], [max(ylim)-0.5 max(ylim)-0.5 max(ylim) max(ylim)], color); %draw the patch
            p1.EdgeColor = [1 1 1]; %take off the edges
            hold on %hold
        elseif patches(2,sigmark) == 2 %if we have a 2...
            color = [0.91 0.34 0.1]; %set the orange
            p1 = patch([patches(1,sigmark) patches(1,sigmark+1) patches(1,sigmark+1) patches(1,sigmark)], [max(ylim)-0.5 max(ylim)-0.5 max(ylim) max(ylim)], color); %draw the patch
            p1.EdgeColor = [1 1 1]; %take off the edges
            hold on %hold
        elseif patches(2,sigmark) == 3 %if we have a 3...
            color = [0.93 0.63 0.19]; %set the red
            p1 = patch([patches(1,sigmark) patches(1,sigmark+1) patches(1,sigmark+1) patches(1,sigmark)], [max(ylim)-0.5 max(ylim)-0.5 max(ylim) max(ylim)], color); %draw the patch
            p1.EdgeColor = [1 1 1]; %take off the edges
            hold on     %hold
        end %ending "if" statement looking for sig
    end %ending "for" loop going through to make patches
end %ending "if" statement looking for sigs
child = get(gca,'children'); %getting children for legend
legend([child(8) child(6) child(2) child(3)],'Ctrl','Stim','p < 0.05','p < 0.01','AutoUpdate','off') %legend

%% dSPN - percent correct

f = figure; %fig
ctr = []; ydt = []; %space
y = [mean(d1min_no_c) mean(d1min_stim_c); mean(d1low_no_c) mean(d1low_stim_c); mean(d1high_no_c) mean(d1high_stim_c)]; %accuracy
a_vert = [std(d1min_no_c)/sqrt(length(d1min_no_c)) std(d1min_stim_c)/sqrt(length(d1min_stim_c)); std(d1low_no_c)/sqrt(length(d1low_no_c)) std(d1low_stim_c)/sqrt(length(d1low_stim_c)); std(d1high_no_c)/sqrt(length(d1high_no_c)) std(d1high_stim_c)/sqrt(length(d1high_stim_c))]; %SEM
errorplus_vert=a_vert'; %positive SEM
errorminus_vert=errorplus_vert; %negative SEM
x = [1; 1.1; 1.2]; %x position
a_horz = [0.01;0.01;0.01]; %horizontal lines
errorplus_horz=a_horz'; %positive horiz
errorminus_horz=errorplus_horz; %negative horiz
hBar = bar(x,y,'FaceColor',[1 1 1],'EdgeColor',[1 1 1]); %plot accuracy
for k1 = 1:size(y,2) %"For" loop going through y
    ctr(k1,:) = bsxfun(@plus, hBar(k1).XData, hBar(k1).XOffset');     %ctr
    ydt(k1,:) = hBar(k1).YData;                    %ydt
end %ending "for" loop going through  y
minnoc = [0.74 0.65 0.67 0.72 0.63 0.73 0.72 0.7 0.71 0.72]; %NC ind
minstim = [0.78 0.7 0.68 0.68 0.67 0.76 0.72 0.64 0.74 0.71]; %NS ind
lownoc = [0.74 0.65 0.65 0.69 0.63 0.74 0.72 0.69 0.69 0.71]; %LC ind
lowstim = [0.7 0.69 0.7 0.69 0.64 0.71 0.71 0.66 0.72 0.7]; %LS ind
highnoc = [0.67 0.57 0.63 0.6 0.57 0.64 0.6 0.59 0.59 0.63]; %HC ind
highstim = [0.66 0.64 0.57 0.59 0.59 0.6 0.62 0.56 0.63 0.6]; %HS ind
minnoc(2,:) = ctr(1,1); %center
minstim(2,:) = ctr(2,1);%center
lownoc(2,:) = ctr(1,2);%center
lowstim(2,:) = ctr(2,2); %center
highnoc(2,:) = ctr(1,3);%center
highstim(2,:) = ctr(2,3);%center
hold on %hold fig
for ugh = 1:10 %"for" loop going through ind
    X(1) =  minnoc(2,ugh); %ctrl x
    Y(1) =  minnoc(1,ugh); %ctrl y
    X(2) =  minstim(2,ugh); %stim x
    Y(2) =  minstim(1,ugh); %stim y
    plot(X(1:2),Y,'Color','[0.7 0.7 0.7]') %plot lines
end %ending "for" loop going through ind
for ugh = 1:10%"for" loop going through ind
    X(1) =  lownoc(2,ugh);%ctrl x
    Y(1) =  lownoc(1,ugh);%ctrl y
    X(2) =  lowstim(2,ugh);%stim x
    Y(2) =  lowstim(1,ugh);%stim y
    plot(X(1:2),Y,'Color','[0.7 0.7 0.7]')%plot lines
end%ending "for" loop going through ind
for ugh = 1:10%"for" loop going through ind
    X(1) =  highnoc(2,ugh);%ctrl x
    Y(1) =  highnoc(1,ugh);%ctrl y
    X(2) =  highstim(2,ugh);%stim x
    Y(2) =  highstim(1,ugh);%stim y
    plot(X(1:2),Y,'Color','[0.7 0.7 0.7]')%plot lines
end%ending "for" loop going through ind
X(1) = ctr(1,1); X(2) = ctr(2,1); Y(1) = y(1,1); Y(2) = y(1,2); %plot avg
plot(X(1:2),Y,'LineWidth',1.5,'Color','k') %avg line
X(1) = ctr(1,2); X(2) = ctr(2,2); Y(1) = y(2,1); Y(2) = y(2,2);%plot avg
plot(X(1:2),Y,'LineWidth',1.5,'Color','k')%avg line
X(1) = ctr(1,3); X(2) = ctr(2,3); Y(1) = y(3,1); Y(2) = y(3,2);%plot avg
plot(X(1:2),Y,'LineWidth',1.5,'Color','k')%avg line
e1 = errorbar(ctr(1,1), y(1,1),errorminus_vert(1,1),errorplus_vert(1,1),'LineStyle','none','LineWidth',2); %NC acc
e1.Color = '[0.3 0.75 0.93]'; %color
hLine = line([ctr(1,1) - errorminus_horz(1), ctr(1,1) + errorplus_horz(1)], [y(1,1), y(1,1)], 'Color', get(e1, 'Color'), 'LineWidth', 2); %horiz
e12 = errorbar(ctr(2,1), y(1,2),errorminus_vert(2,1),errorplus_vert(2,1),'LineStyle','none','LineWidth',2); %NS acc
e12.Color = 'b';  %color
hLine = line([ctr(2,1) - errorminus_horz(1), ctr(2,1) + errorplus_horz(1)], [y(1,2), y(1,2)], 'Color', get(e12, 'Color'), 'LineWidth', 2);%horiz
e2 = errorbar(ctr(1,2), y(2,1),errorminus_vert(1,2),errorplus_vert(1,2),'LineStyle','none','LineWidth',2); %LC acc
e2.Color = '[0.3 0.75 0.93]';  %color
hLine = line([ctr(1,2) - errorminus_horz(1), ctr(1,2) + errorplus_horz(1)], [y(2,1), y(2,1)], 'Color', get(e2, 'Color'), 'LineWidth', 2);%horiz
e22 = errorbar(ctr(2,2), y(2,2),errorminus_vert(2,2),errorplus_vert(2,2),'LineStyle','none','LineWidth',2); %LS acc
e22.Color = 'b';  %color
hLine = line([ctr(2,2) - errorminus_horz(1), ctr(2,2) + errorplus_horz(1)], [y(2,2), y(2,2)], 'Color', get(e22, 'Color'), 'LineWidth', 2);%horiz
e3 = errorbar(ctr(1,3), y(3,1),errorminus_vert(1,3),errorplus_vert(1,3),'LineStyle','none','LineWidth',2); %HC acc
e3.Color = '[0.3 0.75 0.93]';  %color
hLine = line([ctr(1,3) - errorminus_horz(1), ctr(1,3) + errorplus_horz(1)], [y(3,1), y(3,1)], 'Color', get(e3, 'Color'), 'LineWidth', 2);%horiz
e32 = errorbar(ctr(2,3), y(3,2),errorminus_vert(2,3),errorplus_vert(2,3),'LineStyle','none','LineWidth',2); %HS acc
e32.Color = 'b';  %color
hLine = line([ctr(2,3) - errorminus_horz(1), ctr(2,3) + errorplus_horz(1)], [y(3,2), y(3,2)], 'Color', get(e32, 'Color'), 'LineWidth', 2);%horiz
%ylim([0.4 0.9]) %original y lim
ylim([0 0.9]) %y lim axis to 0
XTickLabel = ({'No','Low','High'}); %x label
set(gca,'xtick',[1 1.1 1.2],'xticklabel',XTickLabel,'box','off','FontSize',20) %aesthetics
ylabel('% Correct') %y label
legend({'Ctrl','Stim'},'AutoUpdate','off') %legend

%% dSPN - performance by block type

f = figure; %fig
ctr = []; ydt = []; %space
y = [mean(d1min_no_left_percent) mean(d1min_stim_left_percent) mean(d1min_no_right_percent) mean(d1min_stim_right_percent); mean(d1low_no_left_percent) mean(d1low_stim_left_percent) mean(d1low_no_right_percent) mean(d1low_stim_right_percent); mean(d1high_no_left_percent) mean(d1high_stim_left_percent) mean(d1high_no_right_percent) mean(d1high_stim_right_percent)]; %performance by block type
a_vert = [std(d1min_no_left_percent)/sqrt(length(d1min_no_left_percent)) std(d1min_stim_left_percent)/sqrt(length(d1min_stim_left_percent)) std(d1min_no_right_percent)/sqrt(length(d1min_no_right_percent)) std(d1min_stim_right_percent)/sqrt(length(d1min_stim_right_percent)); std(d1low_no_left_percent)/sqrt(length(d1low_no_left_percent)) std(d1low_stim_left_percent)/sqrt(length(d1low_stim_left_percent)) std(d1low_no_right_percent)/sqrt(length(d1low_no_right_percent)) std(d1low_stim_right_percent)/sqrt(length(d1low_stim_right_percent)); std(d1high_no_left_percent)/sqrt(length(d1high_no_left_percent)) std(d1high_stim_left_percent)/sqrt(length(d1high_stim_left_percent)) std(d1high_no_right_percent)/sqrt(length(d1high_no_right_percent)) std(d1high_stim_right_percent)/sqrt(length(d1high_stim_right_percent))]; %SEM
errorplus_vert=a_vert'; %positive SEM
errorminus_vert=errorplus_vert; %negative SEM
x = [1; 1.1; 1.2]; %x position
a_horz = [0.005;0.005;0.005]; %horizontal lines
errorplus_horz=a_horz'; %positive horiz
errorminus_horz=errorplus_horz; %negative horiz
hBar = bar(x,y,'FaceColor',[1 1 1],'EdgeColor',[1 1 1]); %plot performance by block type
for k1 = 1:size(y,2) %"for" loop going through y
    ctr(k1,:) = bsxfun(@plus, hBar(k1).XData, hBar(k1).XOffset');      %ctr
    ydt(k1,:) = hBar(k1).YData;                    %ydt
end %ending "for" loop going through y
minnowsL = [0.73 0.64 0.64 0.67 0.63 0.69 0.74 0.69 0.65 0.71]; %NC non pref
 minstimwsL = [0.78 0.69 0.75 0.64 0.65 0.8 0.72 0.61 0.73 0.71]; %NS non pref
 lownowsL = [0.73 0.64 0.6 0.65 0.6 0.67 0.7 0.7 0.63 0.67]; %LC non pref
 lowstimwsL = [0.64 0.66 0.7 0.62 0.65 0.66 0.67 0.66 0.7 0.67]; %LS nonpref
 highnowsL = [0.62 0.48 0.53 0.52 0.56 0.65 0.64 0.66 0.51 0.65]; %HC non pref
 highstimwsL = [0.73 0.66 0.61 0.58 0.53 0.72 0.68 0.58 0.59 0.64]; %HS non pref
 minnowsR = [0.76 0.65 0.7 0.76 0.64 0.76 0.7 0.71 0.79 0.72]; %NC pref
 minstimwsR = [0.78 0.73 0.63 0.72 0.69 0.72 0.71 0.68 0.75 0.72]; %NS pref
 lownowsR = [0.75 0.66 0.7 0.73 0.65 0.8 0.75 0.69 0.74 0.75]; %LC pref
 lowstimwsR = [0.75 0.72 0.71 0.77 0.64 0.77 0.75 0.67 0.73 0.73]; %LS pref
 highnowsR = [0.72 0.67 0.74 0.71 0.58 0.62 0.56 0.54 0.69 0.61]; %HC pref
 highstimwsR = [0.6 0.61 0.54 0.61 0.65 0.5 0.57 0.55 0.68]; %HS pref
 minnowsL(2,:) = ctr(1,1); %center
 minstimwsL(2,:) = ctr(2,1);%center
 minnowsR(2,:) = ctr(3,1);%center
 minstimwsR(2,:) = ctr(4,1);%center
 lownowsL(2,:) = ctr(1,2);%center
 lowstimwsL(2,:) = ctr(2,2);%center
 lownowsR(2,:) = ctr(3,2);%center
 lowstimwsR(2,:) = ctr(4,2);%center
 highnowsL(2,:) = ctr(1,3);%center
 highstimwsL(2,:) = ctr(2,3);%center
 highnowsR(2,:) = ctr(3,3);%center
 highstimwsR(2,:) = ctr(4,3);%center
hold on %hold fig
scatter(minnowsL(2,:),minnowsL(1,:),30,[0.63 0.8 0.38]); %scatter ind
scatter(minstimwsL(2,:),minstimwsL(1,:),30,[0.47 0.67 0.19]);%scatter ind
scatter(lownowsL(2,:),lownowsL(1,:),30,[0.63 0.8 0.38]);%scatter ind
scatter(lowstimwsL(2,:),lowstimwsL(1,:),30,[0.47 0.67 0.19]);%scatter ind
scatter(highnowsL(2,:),highnowsL(1,:),30,[0.63 0.8 0.38]);%scatter ind
scatter(highstimwsL(2,:),highstimwsL(1,:),30,[0.47 0.67 0.19]);%scatter ind
scatter(minnowsR(2,:),minnowsR(1,:),30,[1 0.75 0.91]);%scatter ind
scatter(minstimwsR(2,:),minstimwsR(1,:),30,[0.98 0.51 0.8]);%scatter ind
scatter(lownowsR(2,:),lownowsR(1,:),30,[1 0.75 0.91]);%scatter ind
scatter(lowstimwsR(2,:),lowstimwsR(1,:),30,[0.98 0.51 0.8]);%scatter ind
scatter(highnowsR(2,:),highnowsR(1,:),30,[1 0.75 0.91]);%scatter ind
scatter(highstimwsR(2,:),highstimwsR(1,:),30,[0.98 0.51 0.8]);%scatter ind
X(1) = ctr(1,1); X(2) = ctr(2,1); Y(1) = y(1,1); Y(2) = y(1,2); %avg line
plot(X(1:2),Y,'LineWidth',1.5,'Color','k') %plot avg
X(1) = ctr(3,1); X(2) = ctr(4,1); Y(1) = y(1,3); Y(2) = y(1,4);%avg line
plot(X(1:2),Y,'LineWidth',1.5,'Color','k')%plot avg
X(1) = ctr(1,2); X(2) = ctr(2,2); Y(1) = y(2,1); Y(2) = y(2,2);%avg line
plot(X(1:2),Y,'LineWidth',1.5,'Color','k')%plot avg
X(1) = ctr(3,2); X(2) = ctr(4,2); Y(1) = y(2,3); Y(2) = y(2,4);%avg line
plot(X(1:2),Y,'LineWidth',1.5,'Color','k')%plot avg
X(1) = ctr(1,3); X(2) = ctr(2,3); Y(1) = y(3,1); Y(2) = y(3,2);%avg line
plot(X(1:2),Y,'LineWidth',1.5,'Color','k')%plot avg
X(1) = ctr(3,3); X(2) = ctr(4,3); Y(1) = y(3,3); Y(2) = y(3,4);%avg line
plot(X(1:2),Y,'LineWidth',1.5,'Color','k')%plot avg
e1 = errorbar(ctr(1,1), y(1,1),errorminus_vert(1,1),errorplus_vert(1,1),'LineStyle','none','LineWidth',2); %NC nonpref
e1.Color = '[0.64 0.84 0.48]'; %color
hLine = line([ctr(1,1) - errorminus_horz(1), ctr(1,1) + errorplus_horz(1)], [y(1,1), y(1,1)], 'Color', get(e1, 'Color'), 'LineWidth', 2); %horiz
e12 = errorbar(ctr(2,1), y(1,2),errorminus_vert(2,1),errorplus_vert(2,1),'LineStyle','none','LineWidth',2); %NS nonpref
e12.Color = '[0.29 0.41 0.1]'; %color
hLine = line([ctr(2,1) - errorminus_horz(1), ctr(2,1) + errorplus_horz(1)], [y(1,2), y(1,2)], 'Color', get(e12, 'Color'), 'LineWidth', 2);%horiz
e13 = errorbar(ctr(3,1), y(1,3),errorminus_vert(3,1),errorplus_vert(3,1),'LineStyle','none','LineWidth',2); %NC pref
e13.Color = '[0.98 0.67 0.86]'; %color
hLine = line([ctr(3,1) - errorminus_horz(1), ctr(3,1) + errorplus_horz(1)], [y(1,3), y(1,3)], 'Color', get(e13, 'Color'), 'LineWidth', 2);%horiz
e14 = errorbar(ctr(4,1), y(1,4),errorminus_vert(4,1),errorplus_vert(4,1),'LineStyle','none','LineWidth',2); %NS pref
e14.Color = '[1 0.07 0.65]';%color
hLine = line([ctr(4,1) - errorminus_horz(1), ctr(4,1) + errorplus_horz(1)], [y(1,4), y(1,4)], 'Color', get(e14, 'Color'), 'LineWidth', 2);%horiz
e2 = errorbar(ctr(1,2), y(2,1),errorminus_vert(1,2),errorplus_vert(1,2),'LineStyle','none','LineWidth',2); %LC nonpref
e2.Color = '[0.64 0.84 0.48]'; %color
hLine = line([ctr(1,2) - errorminus_horz(1), ctr(1,2) + errorplus_horz(1)], [y(2,1), y(2,1)], 'Color', get(e2, 'Color'), 'LineWidth', 2);%horiz
e22 = errorbar(ctr(2,2), y(2,2),errorminus_vert(2,2),errorplus_vert(2,2),'LineStyle','none','LineWidth',2); %LS nonpref
e22.Color = '[0.29 0.41 0.1]'; %color
hLine = line([ctr(2,2) - errorminus_horz(1), ctr(2,2) + errorplus_horz(1)], [y(2,2), y(2,2)], 'Color', get(e22, 'Color'), 'LineWidth', 2);%horiz
e23 = errorbar(ctr(3,2), y(2,3),errorminus_vert(3,2),errorplus_vert(3,2),'LineStyle','none','LineWidth',2); %LC pref
e23.Color = '[0.98 0.67 0.86]'; %color
hLine = line([ctr(3,2) - errorminus_horz(1), ctr(3,2) + errorplus_horz(1)], [y(2,3), y(2,3)], 'Color', get(e23, 'Color'), 'LineWidth', 2);%horiz
e24 = errorbar(ctr(4,2), y(2,4),errorminus_vert(4,2),errorplus_vert(4,2),'LineStyle','none','LineWidth',2); %LS pref
e24.Color = '[1 0.07 0.65]';%color
hLine = line([ctr(4,2) - errorminus_horz(1), ctr(4,2) + errorplus_horz(1)], [y(2,4), y(2,4)], 'Color', get(e24, 'Color'), 'LineWidth', 2);%horiz
e3 = errorbar(ctr(1,3), y(3,1),errorminus_vert(1,3),errorplus_vert(1,3),'LineStyle','none','LineWidth',2); %HC nonpref
e3.Color = '[0.64 0.84 0.48]'; %color
hLine = line([ctr(1,3) - errorminus_horz(1), ctr(1,3) + errorplus_horz(1)], [y(3,1), y(3,1)], 'Color', get(e3, 'Color'), 'LineWidth', 2);%horiz
e32 = errorbar(ctr(2,3), y(3,2),errorminus_vert(2,3),errorplus_vert(2,3),'LineStyle','none','LineWidth',2); %HS nonpref
e32.Color = '[0.29 0.41 0.1]'; %color
hLine = line([ctr(2,3) - errorminus_horz(1), ctr(2,3) + errorplus_horz(1)], [y(3,2), y(3,2)], 'Color', get(e32, 'Color'), 'LineWidth', 2);%horiz
e33 = errorbar(ctr(3,3), y(3,3),errorminus_vert(3,3),errorplus_vert(3,3),'LineStyle','none','LineWidth',2); %HC pref
e33.Color = '[0.98 0.67 0.86]'; %color
hLine = line([ctr(3,3) - errorminus_horz(1), ctr(3,3) + errorplus_horz(1)], [y(3,3), y(3,3)], 'Color', get(e33, 'Color'), 'LineWidth', 2);%horiz
e34 = errorbar(ctr(4,3), y(3,4),errorminus_vert(4,3),errorplus_vert(4,3),'LineStyle','none','LineWidth',2); %HS pref
e34.Color = '[1 0.07 0.65]';%color
hLine = line([ctr(4,3) - errorminus_horz(1), ctr(4,3) + errorplus_horz(1)], [y(3,4), y(3,4)], 'Color', get(e34, 'Color'), 'LineWidth', 2);%horiz
%ylim([0.4 0.9]) %ylim
ylim([0 0.9]) %ylim to 0
XTickLabel = ({'No','Low','High'}); %x label
set(gca,'xtick',[1 1.1 1.2],'xticklabel',XTickLabel,'box','off','FontSize',20) %aesthetics
ylabel('% Correct') %y label
legend({'Ctrl Non-Pref','Stim Non-Pref','Ctrl Pref','Stim Pref'},'AutoUpdate','off') %legend
X(1) = ctr(1,1); X(2) = ctr(2,1); Y(1) = 0.81; Y(2) = 0.81; %sig bar
plot(X(1:2),Y,'LineWidth',1.5,'Color','k') %plot line
X(1) = ctr(1,3); X(2) = ctr(2,3); Y(1) = 0.74; Y(2) = 0.74; %sig bar
plot(X(1:2),Y,'LineWidth',1.5,'Color','k')%plot line
X(1) = ctr(3,3); X(2) = ctr(4,3); Y(1) = 0.75; Y(2) = 0.75; %sig bar
plot(X(1:2),Y,'LineWidth',1.5,'Color','k')%plot line

%% dSPN - beta

f = figure; %fig
ctr = []; ydt = []; %space
y = [mean(dspn_noconflowvolctrl_explore_exploit) mean(dspn_lowconflowvolctrl_explore_exploit) mean(dspn_highconflowvolctrl_explore_exploit); mean(dspn_noconflowvolstim_explore_exploit) mean(dspn_lowconflowvolstim_explore_exploit) mean(dspn_highconflowvolstim_explore_exploit)]; %beta
a_vert = [std(dspn_noconflowvolctrl_explore_exploit)/sqrt(length(dspn_noconflowvolctrl_explore_exploit)) std(dspn_lowconflowvolctrl_explore_exploit)/sqrt(length(dspn_lowconflowvolctrl_explore_exploit)) std(dspn_highconflowvolctrl_explore_exploit)/sqrt(length(dspn_highconflowvolctrl_explore_exploit)); std(dspn_noconflowvolstim_explore_exploit)/sqrt(length(dspn_noconflowvolstim_explore_exploit)) std(dspn_lowconflowvolstim_explore_exploit)/sqrt(length(dspn_lowconflowvolstim_explore_exploit)) std(dspn_highconflowvolstim_explore_exploit)/sqrt(length(dspn_highconflowvolstim_explore_exploit))]; %SEM
errorplus_vert=a_vert'; %positive SEM
errorminus_vert=errorplus_vert; %negative SEM
x = [1; 1.1]; %x position
a_horz = [0.01;0.01]/2; %horizontal lines
errorplus_horz=a_horz'; %positive horiz
errorminus_horz=errorplus_horz; %negative horiz
hBar = bar(x,y,'FaceColor',[1 1 1],'EdgeColor',[1 1 1]); %plot beta
for k1 = 1:size(y,2) %"for" loop going through y
    ctr(k1,:) = bsxfun(@plus, hBar(k1).XData, hBar(k1).XOffset');     %ctr
    ydt(k1,:) = hBar(k1).YData;                    %ydt
end %ending "for" loop going through y
ctr(1,:) = ctr(1,:)+0.01; %positive offset
ctr(3,:) = ctr(3,:)-0.01; %negative offset
minnobeta = [1.12 1.23 1.13 0.73 1.15 1.1 1.66 0.84 1.27 1.26]; %NC beta
lownobeta = [1.9 1.62 1.1 1.03 0.97 1.36 1.09 1.11 1.59 0.62]; %LC beta
highnobeta = [1.47 1.51 0.9 1.38 0.55 1.12 1.07 1.75 0.6 0.95]; %HC beta
minhighbeta = [1.46 1.32 1.57 1.23 1.5 1.12 1.09 0.82 0.6 1.13]; %NS beta
lowhighbeta = [1.24 1 1.14 1.54 0.93 1.43 1.3 0.69 1.17 1.49]; %LS beta
highhighbeta = [1.13 1.87 1.26 1.19 1.8 1.12 1.29 1.08 0.74 0.96]; %HS beta
minnobeta(2,:) = ctr(1,1); %center
lownobeta(2,:) = ctr(2,1);%center
highnobeta(2,:) = ctr(3,1);%center
minhighbeta(2,:) = ctr(1,2);%center
lowhighbeta(2,:) = ctr(2,2);%center
highhighbeta(2,:) = ctr(3,2);%center
hold on %hold fig
scatter(minnobeta(2,:),minnobeta(1,:),30,[0.64 0.86 1]); %scatter ind
scatter(minhighbeta(2,:),minhighbeta(1,:),30,[0.64 0.86 1]);%scatter ind
scatter(lownobeta(2,:),lownobeta(1,:),30,[0.89 0.75 0.54]);%scatter ind
scatter(lowhighbeta(2,:),lowhighbeta(1,:),30,[0.89 0.75 0.54]);%scatter ind
scatter(highnobeta(2,:),highnobeta(1,:),30,[0.98 0.64 0.5]);%scatter ind
scatter(highhighbeta(2,:),highhighbeta(1,:),30,[0.98 0.64 0.5]);%scatter ind
e1 = errorbar(ctr(1,1), y(1,1),errorminus_vert(1,1),errorplus_vert(1,1),'LineStyle','none','LineWidth',2); %NC beta
e1.Color = '[0 0.45 0.74]'; %color
hLine = line([ctr(1,1) - errorminus_horz(1), ctr(1,1) + errorplus_horz(1)], [y(1,1), y(1,1)], 'Color', get(e1, 'Color'), 'LineWidth', 2); %horiz
e12 = errorbar(ctr(2,1), y(1,2),errorminus_vert(2,1),errorplus_vert(2,1),'LineStyle','none','LineWidth',2); %LC beta
e12.Color = '[0.93 0.69 0.13]'; %color
hLine = line([ctr(2,1) - errorminus_horz(1), ctr(2,1) + errorplus_horz(1)], [y(1,2), y(1,2)], 'Color', get(e12, 'Color'), 'LineWidth', 2);%horiz
e13 = errorbar(ctr(3,1), y(1,3),errorminus_vert(3,1),errorplus_vert(3,1),'LineStyle','none','LineWidth',2); %HC beta
e13.Color = '[0.85 0.33 0.10]'; %color
hLine = line([ctr(3,1) - errorminus_horz(1), ctr(3,1) + errorplus_horz(1)], [y(1,3), y(1,3)], 'Color', get(e13, 'Color'), 'LineWidth', 2);%horiz
e2 = errorbar(ctr(1,2), y(2,1),errorminus_vert(1,2),errorplus_vert(1,2),'LineStyle','none','LineWidth',2); %NS beta
e2.Color = '[0 0.45 0.74]'; %color
hLine = line([ctr(1,2) - errorminus_horz(1), ctr(1,2) + errorplus_horz(1)], [y(2,1), y(2,1)], 'Color', get(e2, 'Color'), 'LineWidth', 2);%horiz
e22 = errorbar(ctr(2,2), y(2,2),errorminus_vert(2,2),errorplus_vert(2,2),'LineStyle','none','LineWidth',2); %LS beta
e22.Color = '[0.93 0.69 0.13]'; %color
hLine = line([ctr(2,2) - errorminus_horz(1), ctr(2,2) + errorplus_horz(1)], [y(2,2), y(2,2)], 'Color', get(e22, 'Color'), 'LineWidth', 2);%horiz
e23 = errorbar(ctr(3,2), y(2,3),errorminus_vert(3,2),errorplus_vert(3,2),'LineStyle','none','LineWidth',2); %HS beta
e23.Color = '[0.85 0.33 0.10]'; %color
hLine = line([ctr(3,2) - errorminus_horz(1), ctr(3,2) + errorplus_horz(1)], [y(2,3), y(2,3)], 'Color', get(e23, 'Color'), 'LineWidth', 2);%horiz
%ylim([0.5 2.5]) %original y lim
ylim([0 2.5]) %ylim axis to 0
XTickLabel = ({'Ctrl','Stim'}); %legend
%set(gca,'xtick',[1
%1.1],'xticklabel',XTickLabel,'ytick',[1:0.5:2],'box','off','FontSize',20)
%%original asethetics
set(gca,'xtick',[1 1.1],'xticklabel',XTickLabel,'box','off','FontSize',20) %aesthetics
ylabel('Explore          vs.          Exploit') %y label
legend({'No','Low','High'},'AutoUpdate','off') %legend
X(1) = ctr(1,1); X(2) = ctr(3,1); Y(1) = 1.95; Y(2) = 1.95; %sig bar
plot(X(1:2),Y,'LineWidth',1.5,'Color','k') %plot line

%% dSPN - rewarded vs unrewarded learning rate

f = figure; %fig
ctr = []; ydt = []; %space
y = [mean(d1min_no_rew_lr) mean(d1min_no_unrew_lr) mean(d1low_no_rew_lr) mean(d1low_no_unrew_lr) mean(d1high_no_rew_lr) mean(d1high_no_unrew_lr); mean(d1min_stim_rew_lr) mean(d1min_stim_unrew_lr) mean(d1low_stim_rew_lr) mean(d1low_stim_unrew_lr) mean(d1high_stim_rew_lr) mean(d1high_stim_unrew_lr)]; %rew/unrew LR
a_vert = [std(d1min_no_rew_lr)/sqrt(length(d1min_no_rew_lr)) std(d1min_no_unrew_lr)/sqrt(length(d1min_no_unrew_lr)) std(d1low_no_rew_lr)/sqrt(length(d1low_no_rew_lr)) std(d1low_no_unrew_lr)/sqrt(length(d1low_no_unrew_lr)) std(d1high_no_rew_lr)/sqrt(length(d1high_no_rew_lr)) std(d1high_no_unrew_lr)/sqrt(length(d1high_no_unrew_lr)); std(d1min_stim_rew_lr)/sqrt(length(d1min_stim_rew_lr)) std(d1min_stim_unrew_lr)/sqrt(length(d1min_stim_unrew_lr)) std(d1low_stim_rew_lr)/sqrt(length(d1low_stim_rew_lr)) std(d1low_stim_unrew_lr)/sqrt(length(d1low_stim_unrew_lr)) std(d1high_stim_rew_lr)/sqrt(length(d1high_stim_rew_lr)) std(d1high_stim_unrew_lr)/sqrt(length(d1high_stim_unrew_lr))]; %SEM
errorplus_vert=a_vert'; %positive SEM
errorminus_vert=errorplus_vert; %negative SEM
x = [1; 1.1]; %x position
a_horz = [0.005;0.005]/2; %horizontal lines
errorplus_horz=a_horz'; %positive horiz
errorminus_horz=errorplus_horz; %negative horiz
hBar = bar(x,y,'FaceColor',[1 1 1],'EdgeColor',[1 1 1]); %plot LRs
for k1 = 1:size(y,2) %"for" loop going through y
    ctr(k1,:) = bsxfun(@plus, hBar(k1).XData, hBar(k1).XOffset');      %ctr
    ydt(k1,:) = hBar(k1).YData;                     %ydt
end %ending "for" loop going through y
minnoREW = [1 0.8 1 0.81 0.18 1 1 0.95 1 1]; %NC rew LR ind
minnoUNREW = [1 1 1 0.98 0.66 1 1 0.74 1 1]; %NC unrew LR ind
lownoREW = [0.71 0.71 1 1 0.69 1 1 0.7 0.67 0.91]; %LC rew LR ind
lownoUNREW = [1 0.92 1 1 0.45 1 1 1 0.8 1]; %LC unrew LR ind
highnoREW = [1 1 0.96 0.75 1 1 1 1 0.68 0.99]; %HC rew LR ind
highnoUNREW = [0.95 1 0.88 0.75 0.66 1 1 1 0.77 1]; %HS unrew LR ind
minstimREW = [1 1 1 1 0.03 1 0.8 1 1 1]; %NS rew LR ind
minstimUNREW = [1 1 0.79 1 0.55 1 1 1 1 1]; %NS unrew LR ind
lowstimREW = [1 1 1 0.58 1 1 1 0.64 1 1]; %LS rew LR ind
lowstimUNREW = [0.93 1 0.69 1 0.7 0.92 1 1 1 1]; %LS unrew LR ind
highstimREW = [0.75 1 0.74 1 0.64 1 0.97 1 0.8 0.85]; %HS rew LR ind
highstimUNREW = [0.97 1 0.67 1 0.97 0.88 1 0.74 1 0.99]; %HS unrew LR ind
 minnoREW(2,:) = ctr(1,1); %center
 minnoUNREW(2,:) = ctr(2,1);%center
 lownoREW(2,:) = ctr(3,1);%center
 lownoUNREW(2,:) = ctr(4,1);%center
 highnoREW(2,:) = ctr(5,1);%center
 highnoUNREW(2,:) = ctr(6,1);%center
minstimREW(2,:) = ctr(1,2);%center
 minstimUNREW(2,:) = ctr(2,2);%center
 lowstimREW(2,:) = ctr(3,2);%center
 lowstimUNREW(2,:) = ctr(4,2);%center
 highstimREW(2,:) = ctr(5,2);%center
 highstimUNREW(2,:) = ctr(6,2);%center
hold on %hold fig
scatter(minnoREW(2,:),minnoREW(1,:),30,[0.28 0.5 0.64]); %scatter ind
scatter(minnoUNREW(2,:),minnoUNREW(1,:),30,[0.37 0.64 0.76]);%scatter ind
scatter(lownoREW(2,:),lownoREW(1,:),30,[1 0.87 0.65]);%scatter ind
scatter(lownoUNREW(2,:),lownoUNREW(1,:),30,[0.94 0.83 0.58]);%scatter ind
scatter(highnoREW(2,:),highnoREW(1,:),30,[1 0.66 0.51]);%scatter ind
scatter(highnoUNREW(2,:),highnoUNREW(1,:),30,[1 0.8 0.72]);%scatter ind
scatter(minstimREW(2,:),minstimREW(1,:),30,[0.28 0.5 0.64]);%scatter ind
scatter(minstimUNREW(2,:),minstimUNREW(1,:),30,[0.37 0.64 0.76]);%scatter ind
scatter(lowstimREW(2,:),lowstimREW(1,:),30,[1 0.87 0.65]);%scatter ind
scatter(lowstimUNREW(2,:),lowstimUNREW(1,:),30,[0.94 0.83 0.58]);%scatter ind
scatter(highstimREW(2,:),highstimREW(1,:),30,[1 0.66 0.51]);%scatter ind
scatter(highstimUNREW(2,:),highstimUNREW(1,:),30,[1 0.8 0.72]);%scatter ind
e1 = errorbar(ctr(1,1), y(1,1),errorminus_vert(1,1),errorplus_vert(1,1),'LineStyle','none','LineWidth',2); %NC rew LR
e1.Color = '[0 0.45 0.74]'; %color
hLine = line([ctr(1,1) - errorminus_horz(1), ctr(1,1) + errorplus_horz(1)], [y(1,1), y(1,1)], 'Color', get(e1, 'Color'), 'LineWidth', 2); %horiz
e12 = errorbar(ctr(2,1), y(1,2),errorminus_vert(2,1),errorplus_vert(2,1),'LineStyle','none','LineWidth',2); %NS unrew LR
e12.Color = '[0.48 0.75 0.93]'; %color
hLine = line([ctr(2,1) - errorminus_horz(1), ctr(2,1) + errorplus_horz(1)], [y(1,2), y(1,2)], 'Color', get(e12, 'Color'), 'LineWidth', 2); %horiz
e13 = errorbar(ctr(3,1), y(1,3),errorminus_vert(3,1),errorplus_vert(3,1),'LineStyle','none','LineWidth',2); %LC rew LR
e13.Color = '[0.93 0.69 0.19]'; %color
hLine = line([ctr(3,1) - errorminus_horz(1), ctr(3,1) + errorplus_horz(1)], [y(1,3), y(1,3)], 'Color', get(e13, 'Color'), 'LineWidth', 2); %horiz
e14 = errorbar(ctr(4,1), y(1,4),errorminus_vert(4,1),errorplus_vert(4,1),'LineStyle','none','LineWidth',2); %LC unrew LR
e14.Color = '[0.91 0.82 0.59]';%color
hLine = line([ctr(4,1) - errorminus_horz(1), ctr(4,1) + errorplus_horz(1)], [y(1,4), y(1,4)], 'Color', get(e14, 'Color'), 'LineWidth', 2); %horiz
e15 = errorbar(ctr(5,1), y(1,5),errorminus_vert(5,1),errorplus_vert(5,1),'LineStyle','none','LineWidth',2); %HC rew LR
e15.Color = '[0.85 0.33 0.1]'; %color
hLine = line([ctr(5,1) - errorminus_horz(1), ctr(5,1) + errorplus_horz(1)], [y(1,5), y(1,5)], 'Color', get(e15, 'Color'), 'LineWidth', 2); %horiz
e16 = errorbar(ctr(6,1), y(1,6),errorminus_vert(6,1),errorplus_vert(6,1),'LineStyle','none','LineWidth',2); %HC unrew LR
e16.Color = '[1 0.62 0.45]';%color
hLine = line([ctr(6,1) - errorminus_horz(1), ctr(6,1) + errorplus_horz(1)], [y(1,6), y(1,6)], 'Color', get(e16, 'Color'), 'LineWidth', 2); %horiz
e2 = errorbar(ctr(1,2), y(2,1),errorminus_vert(1,2),errorplus_vert(1,2),'LineStyle','none','LineWidth',2); %NS rew LR
e2.Color = '[0 0.45 0.74]'; %color
hLine = line([ctr(1,2) - errorminus_horz(1), ctr(1,2) + errorplus_horz(1)], [y(2,1), y(2,1)], 'Color', get(e2, 'Color'), 'LineWidth', 2); %horiz
e22 = errorbar(ctr(2,2), y(2,2),errorminus_vert(2,2),errorplus_vert(2,2),'LineStyle','none','LineWidth',2); %NS unrew LR
e22.Color = '[0.48 0.75 0.93]'; %color
hLine = line([ctr(2,2) - errorminus_horz(1), ctr(2,2) + errorplus_horz(1)], [y(2,2), y(2,2)], 'Color', get(e22, 'Color'), 'LineWidth', 2); %horiz
e23 = errorbar(ctr(3,2), y(2,3),errorminus_vert(3,2),errorplus_vert(3,2),'LineStyle','none','LineWidth',2); %LS rew LR
e23.Color = '[0.93 0.69 0.19]'; %color
hLine = line([ctr(3,2) - errorminus_horz(1), ctr(3,2) + errorplus_horz(1)], [y(2,3), y(2,3)], 'Color', get(e23, 'Color'), 'LineWidth', 2); %horiz
e24 = errorbar(ctr(4,2), y(2,4),errorminus_vert(4,2),errorplus_vert(4,2),'LineStyle','none','LineWidth',2); %LS unrew LR
e24.Color = '[0.91 0.82 0.59]';%color
hLine = line([ctr(4,2) - errorminus_horz(1), ctr(4,2) + errorplus_horz(1)], [y(2,4), y(2,4)], 'Color', get(e24, 'Color'), 'LineWidth', 2); %horiz
e25 = errorbar(ctr(5,2), y(2,5),errorminus_vert(5,2),errorplus_vert(5,2),'LineStyle','none','LineWidth',2); %HS rew LR
e25.Color = '[0.85 0.33 0.1]'; %color
hLine = line([ctr(5,2) - errorminus_horz(1), ctr(5,2) + errorplus_horz(1)], [y(2,5), y(2,5)], 'Color', get(e25, 'Color'), 'LineWidth', 2); %horiz
e26 = errorbar(ctr(6,2), y(2,6),errorminus_vert(6,2),errorplus_vert(6,2),'LineStyle','none','LineWidth',2); %HS unrew LR
e26.Color = '[1 0.62 0.45]';%color
hLine = line([ctr(6,2) - errorminus_horz(1), ctr(6,2) + errorplus_horz(1)], [y(2,6), y(2,6)], 'Color', get(e26, 'Color'), 'LineWidth', 2); %horiz
ylim([0 1.4]) %ylim
XTickLabel = ({'Ctrl','Stim'}); %x label
set(gca,'xtick',[1 1.1],'xticklabel',XTickLabel,'ytick',[0.2:0.2:1],'box','off','FontSize',20) %aesthetics
ylabel('Learning Rate') %y label
legend({'No Rew','No Unrew','Low Rew','Low Unrew','High Rew','High Unrew'},'AutoUpdate','off') %legend
X(1) = ctr(1,1); X(2) = ctr(2,1); Y(1) = 1.06; Y(2) = 1.06; %sig bar
plot(X(1:2),Y,'LineWidth',1.5,'Color','k') %plot line
X(1) = ctr(3,1); X(2) = ctr(4,1); Y(1) =1.06; Y(2) = 1.06;%sig bar
plot(X(1:2),Y,'LineWidth',1.5,'Color','k')%plot line
X(1) = ctr(2,1); X(2) = ctr(6,1); Y(1) = 1.16; Y(2) = 1.16;%sig bar
plot(X(1:2),Y,'LineWidth',1.5,'Color','k') %plot line

%% dSPN - bias

f = figure; %fig
ctr = []; ydt = []; %space
y = [mean(dspn_noconflowvolctrl_Q_bias) mean(dspn_lowconflowvolctrl_Q_bias) mean(dspn_highconflowvolctrl_Q_bias); mean(dspn_noconflowvolstim_Q_bias) mean(dspn_lowconflowvolstim_Q_bias) mean(dspn_highconflowvolstim_Q_bias)]; %bias
a_vert = [std(dspn_noconflowvolctrl_Q_bias)/sqrt(length(dspn_noconflowvolctrl_Q_bias)) std(dspn_lowconflowvolctrl_Q_bias)/sqrt(length(dspn_lowconflowvolctrl_Q_bias)) std(dspn_highconflowvolctrl_Q_bias)/sqrt(length(dspn_highconflowvolctrl_Q_bias)); std(dspn_noconflowvolstim_Q_bias)/sqrt(length(dspn_noconflowvolstim_Q_bias)) std(dspn_lowconflowvolstim_Q_bias)/sqrt(length(dspn_lowconflowvolstim_Q_bias)) std(dspn_highconflowvolstim_Q_bias)/sqrt(length(dspn_highconflowvolstim_Q_bias))]; %SEM
errorplus_vert=a_vert'; %positive SEM
errorminus_vert=errorplus_vert; %negative SEM
x = [1; 1.1]; %x position
a_horz = [0.01;0.01]/2; %horizontal lines
errorplus_horz=a_horz'; %positive horiz
errorminus_horz=errorplus_horz; %negative horiz
hBar = bar(x,y,'FaceColor',[1 1 1],'EdgeColor',[1 1 1]); %plot bias
for k1 = 1:size(y,2) %"for" loop going through y
    ctr(k1,:) = bsxfun(@plus, hBar(k1).XData, hBar(k1).XOffset');      %ctr
    ydt(k1,:) = hBar(k1).YData;                    %ydt
end %ending "for" loop going through y
h = hline(0,'w'); %block 0 line
h.LineWidth = 2; %line width
ctr(1,:) = ctr(1,:)+0.01; %positive offest
ctr(3,:) = ctr(3,:)-0.01; %negative offset
minnobias = [0.28 0.18 0.01 0.35 0.15 0.08 0.07 0.44 0.02 0.31]; %NC bias
lownobias = [0.08 0.27 0.24 0.33 0.03 0.06 0.1 0.39 0.45 0.26]; %LC bias
highnobias = [0.66 0.32 0.6 0.23 0.4 0.27 0.33 0.37 0.4 0.11]; %HC bias
minhighbias = [-0.23 0.27 0.35 -0.25 0.23 -0.24 -0.25 -0.02 0.41 0.24]; %NS bias
lowhighbias = [0.03 0.36 0.47 0.39 0.37 0.01 -0.26 0.44 0.45 0.02]; %LS bias
highhighbias = [0.05 -0.3 -0.42 0.16 -0.41 -0.38 0.03 0.46 0.12 0.14]; %HS bias
minnobias(2,:) = ctr(1,1); %center
lownobias(2,:) = ctr(2,1);%center
highnobias(2,:) = ctr(3,1);%center
minhighbias(2,:) = ctr(1,2);%center
lowhighbias(2,:) = ctr(2,2);%center
highhighbias(2,:) = ctr(3,2);%center
hold on %hold fig
scatter(minnobias(2,:),minnobias(1,:),30,[0.64 0.86 1]); %scatter ind
scatter(minhighbias(2,:),minhighbias(1,:),30,[0.64 0.86 1]); %scatter ind
scatter(lownobias(2,:),lownobias(1,:),30,[0.89 0.75 0.54]); %scatter ind
scatter(lowhighbias(2,:),lowhighbias(1,:),30,[0.89 0.75 0.54]); %scatter ind
scatter(highnobias(2,:),highnobias(1,:),30,[0.98 0.64 0.5]); %scatter ind
scatter(highhighbias(2,:),highhighbias(1,:),30,[0.98 0.64 0.5]); %scatter ind
e1 = errorbar(ctr(1,1), y(1,1),errorminus_vert(1,1),errorplus_vert(1,1),'LineStyle','none','LineWidth',2); %NC bias
e1.Color = '[0 0.45 0.74]'; %color
hLine = line([ctr(1,1) - errorminus_horz(1), ctr(1,1) + errorplus_horz(1)], [y(1,1), y(1,1)], 'Color', get(e1, 'Color'), 'LineWidth', 2); %horiz
e12 = errorbar(ctr(2,1), y(1,2),errorminus_vert(2,1),errorplus_vert(2,1),'LineStyle','none','LineWidth',2); %LC bias
e12.Color = '[0.93 0.69 0.13]'; %color
hLine = line([ctr(2,1) - errorminus_horz(1), ctr(2,1) + errorplus_horz(1)], [y(1,2), y(1,2)], 'Color', get(e12, 'Color'), 'LineWidth', 2);%horiz
e13 = errorbar(ctr(3,1), y(1,3),errorminus_vert(3,1),errorplus_vert(3,1),'LineStyle','none','LineWidth',2); %HC bias
e13.Color = '[0.85 0.33 0.10]'; %color
hLine = line([ctr(3,1) - errorminus_horz(1), ctr(3,1) + errorplus_horz(1)], [y(1,3), y(1,3)], 'Color', get(e13, 'Color'), 'LineWidth', 2);%horiz
e2 = errorbar(ctr(1,2), y(2,1),errorminus_vert(1,2),errorplus_vert(1,2),'LineStyle','none','LineWidth',2); %NS bias
e2.Color = '[0 0.45 0.74]'; %color
hLine = line([ctr(1,2) - errorminus_horz(1), ctr(1,2) + errorplus_horz(1)], [y(2,1), y(2,1)], 'Color', get(e2, 'Color'), 'LineWidth', 2);%horiz
e22 = errorbar(ctr(2,2), y(2,2),errorminus_vert(2,2),errorplus_vert(2,2),'LineStyle','none','LineWidth',2); %LS bias
e22.Color = '[0.93 0.69 0.13]'; %color
hLine = line([ctr(2,2) - errorminus_horz(1), ctr(2,2) + errorplus_horz(1)], [y(2,2), y(2,2)], 'Color', get(e22, 'Color'), 'LineWidth', 2);%horiz
e23 = errorbar(ctr(3,2), y(2,3),errorminus_vert(3,2),errorplus_vert(3,2),'LineStyle','none','LineWidth',2); %HS bias
e23.Color = '[0.85 0.33 0.10]'; %color
hLine = line([ctr(3,2) - errorminus_horz(1), ctr(3,2) + errorplus_horz(1)], [y(2,3), y(2,3)], 'Color', get(e23, 'Color'), 'LineWidth', 2);%horiz
ylim([-0.6 0.8]) %ylim
XTickLabel = ({'Ctrl','Stim'}); %x label
set(gca,'xtick',[1 1.1],'xticklabel',XTickLabel,'box','off','FontSize',20) %aesthetics
ylabel('Bias') %y label
legend({'No','Low','High'},'AutoUpdate','off') %legend

%% dSPN - outcome responsive model

f = figure; %fig
ctr = []; ydt = []; %space
y = [mean(d1high_no_left_percent) mean(d1high_no_right_percent); mean(d1high_stim_left_percent) mean(d1high_stim_right_percent); mean(highconf(:,1)) mean(highconf(:,2))]; %model plotting
a_vert = [std(d1high_no_left_percent)/sqrt(length(d1high_no_left_percent)) std(d1high_no_right_percent)/sqrt(length(d1high_no_right_percent)); std(d1high_stim_left_percent)/sqrt(length(d1high_stim_left_percent)) std(d1high_stim_right_percent)/sqrt(length(d1high_stim_right_percent)); std(highconf(:,1))/sqrt(length(highconf(:,1))) std(highconf(:,2))/sqrt(length(highconf(:,2)))]; %SEM
errorplus_vert=a_vert'; %positive SEM
errorminus_vert=errorplus_vert; %negative SEM
x = [1; 1.1; 1.2]; %x position
a_horz = [0.01;0.01;0.01]; %horizontal lines
errorplus_horz=a_horz'; %positive horiz
errorminus_horz=errorplus_horz; %negative horiz
hBar = bar(x,y,'FaceColor',[1 1 1],'EdgeColor',[1 1 1]); %plot ctrl vs stim vs model performance
for k1 = 1:size(y,2) %"for" loop going through y
    ctr(k1,:) = bsxfun(@plus, hBar(k1).XData, hBar(k1).XOffset');     %ctr
    ydt(k1,:) = hBar(k1).YData;                    %ydt
end %ending "for" loop going through y
highnoleft_percent = [0.62 0.48 0.53 0.52 0.56 0.65 0.64 0.66 0.51 0.65]; %HC nonpref ind
highnoright_percent = [0.72 0.67 0.74 0.71 0.58 0.62 0.56 0.54 0.69 0.61]; %HC pref ind
highstimleft_percent = [0.73 0.66 0.61 0.58 0.53 0.72 0.68 0.58 0.59 0.64]; %HS nonpref ind
highstimright_percent = [0.6 0.61 0.54 0.61 0.65 0.5 0.57 0.55 0.68 0.58]; %HS pref ind
Modelleft_percent(1,:) = highconf_ind(1:10,1); %HM nonpref ind
Modelright_percent(1,:) = highconf_ind(1:10,2); %HM pref ind
highnoleft_percent(2,:) = ctr(1,1); %center
highnoright_percent(2,:) = ctr(2,1); %center
highstimleft_percent(2,:) = ctr(1,2); %center
highstimright_percent(2,:) = ctr(2,2); %center
Modelleft_percent(2,:) = ctr(1,3); %center
Modelright_percent(2,:) = ctr(2,3); %center
hold on %hold fig
for ugh = 1:10 %"for" loop going through individuals
    X(1) =  highnoleft_percent(2,ugh); %non pref x
    Y(1) =  highnoleft_percent(1,ugh); %non pref y
    X(2) =  highnoright_percent(2,ugh); %pref x
    Y(2) =  highnoright_percent(1,ugh); %pref y
    plot(X(1:2),Y,'Color','[0.7 0.7 0.7]') %plot individual lines
end %ending "for" loop going through ind
for ugh = 1:10%"for" loop going through individuals
    X(1) =  highstimleft_percent(2,ugh);%non pref x
    Y(1) =  highstimleft_percent(1,ugh);%non pref y
    X(2) =  highstimright_percent(2,ugh);%pref x
    Y(2) =  highstimright_percent(1,ugh);%pref y
    plot(X(1:2),Y,'Color','[0.7 0.7 0.7]')%plot individual lines
end%ending "for" loop going through ind
for ugh = 1:10%"for" loop going through individuals
    X(1) =  Modelleft_percent(2,ugh);%non pref x
    Y(1) =  Modelleft_percent(1,ugh);%non pref y
    X(2) =  Modelright_percent(2,ugh);%pref x
    Y(2) =  Modelright_percent(1,ugh);%pref y
    plot(X(1:2),Y,'Color','[0.7 0.7 0.7]')%plot individual lines
end%ending "for" loop going through ind
X(1) = ctr(1,1); X(2) = ctr(2,1); Y(1) = y(1,1); Y(2) = y(1,2); %avg line
plot(X(1:2),Y,'LineWidth',1.5,'Color','k') %plot it
X(1) = ctr(1,2); X(2) = ctr(2,2); Y(1) = y(2,1); Y(2) = y(2,2);%avg line
plot(X(1:2),Y,'LineWidth',1.5,'Color','k') %plot it
X(1) = ctr(1,3); X(2) = ctr(2,3); Y(1) = y(3,1); Y(2) = y(3,2);%avg line
plot(X(1:2),Y,'LineWidth',1.5,'Color','k') %plot it
e1 = errorbar(ctr(1,1), y(1,1),errorminus_vert(1,1),errorplus_vert(1,1),'LineStyle','none','LineWidth',2); %HC nonpref
e1.Color = '[0.47 0.67 0.19]'; %color
hLine = line([ctr(1,1) - errorminus_horz(1), ctr(1,1) + errorplus_horz(1)], [y(1,1), y(1,1)], 'Color', get(e1, 'Color'), 'LineWidth', 2); %horiz
e12 = errorbar(ctr(2,1), y(1,2),errorminus_vert(2,1),errorplus_vert(2,1),'LineStyle','none','LineWidth',2); %HC pref
e12.Color = '[1 0.07 0.65]'; %color
hLine = line([ctr(2,1) - errorminus_horz(1), ctr(2,1) + errorplus_horz(1)], [y(1,2), y(1,2)], 'Color', get(e12, 'Color'), 'LineWidth', 2);%horiz
e2 = errorbar(ctr(1,2), y(2,1),errorminus_vert(1,2),errorplus_vert(1,2),'LineStyle','none','LineWidth',2); %HS nonpref
e2.Color = '[0.47 0.67 0.19]'; %color
hLine = line([ctr(1,2) - errorminus_horz(1), ctr(1,2) + errorplus_horz(1)], [y(2,1), y(2,1)], 'Color', get(e2, 'Color'), 'LineWidth', 2);%horiz
e22 = errorbar(ctr(2,2), y(2,2),errorminus_vert(2,2),errorplus_vert(2,2),'LineStyle','none','LineWidth',2); %HS pref
e22.Color = '[1 0.07 0.65]'; %color
hLine = line([ctr(2,2) - errorminus_horz(1), ctr(2,2) + errorplus_horz(1)], [y(2,2), y(2,2)], 'Color', get(e22, 'Color'), 'LineWidth', 2);%horiz
e3 = errorbar(ctr(1,3), y(3,1),errorminus_vert(1,3),errorplus_vert(1,3),'LineStyle','none','LineWidth',2); %HM nonpref
e3.Color = '[0.47 0.67 0.19]'; %color
hLine = line([ctr(1,3) - errorminus_horz(1), ctr(1,3) + errorplus_horz(1)], [y(3,1), y(3,1)], 'Color', get(e3, 'Color'), 'LineWidth', 2);%horiz
e32 = errorbar(ctr(2,3), y(3,2),errorminus_vert(2,3),errorplus_vert(2,3),'LineStyle','none','LineWidth',2); %HM pref
e32.Color = '[1 0.07 0.65]'; %color
hLine = line([ctr(2,3) - errorminus_horz(1), ctr(2,3) + errorplus_horz(1)], [y(3,2), y(3,2)], 'Color', get(e32, 'Color'), 'LineWidth', 2);%horiz
%ylim([0.4 0.9]) %original y lim
ylim([0 0.9]) %ylim axis to 0
XTickLabel = ({'Ctrl','Stim','Model'}); %x label
set(gca,'xtick',[1 1.1 1.2],'xticklabel',XTickLabel,'box','off','FontSize',20) %aesthetics
ylabel('% Correct') %y label
legend({'Non-Pref','Pref'},'AutoUpdate','off') %legend
X(1) = ctr(1,1); X(2) = ctr(2,1); Y(1) = 0.75; Y(2) = 0.75; %sig bar
plot(X(1:2),Y,'LineWidth',1.5,'Color','k') %plot it

%% dSPN - decision time 

f = figure; %fig
ctr = []; ydt = []; %space
y = [(mean(d1min_stim_dt) - mean(d1min_no_dt))/mean(d1min_no_dt); (mean(d1low_stim_dt) - mean(d1low_no_dt))/mean(d1low_no_dt); (mean(d1high_stim_dt) - mean(d1high_no_dt))/mean(d1high_no_dt)]; %DT diff
a = [min_dspn_diff_dt_std; low_dspn_diff_dt_std; high_dspn_diff_dt_std]; %SEM
errorplus_vert=a_vert'; %positive SEM
errorminus_vert=errorplus_vert; %negative SEM
x = [1; 1.05; 1.1]; %x positions
a_horz = [0.04;0.04;0.04]/2; %horizontal lines
errorplus_horz=a_horz'; %positive horiz
errorminus_horz=errorplus_horz; %negative horiz
hBar = bar(x,y,'FaceColor',[1 1 1],'EdgeColor',[1 1 1]); %plot DT diff
for k1 = 1:size(y,2) %"for" loop going through y
    ctr(k1,:) = bsxfun(@plus, hBar(k1).XData, hBar(k1).XOffset');      %ctr
    ydt(k1,:) = hBar(k1).YData;                    %ydt
end %ending "for" loop going through y
h = hline(0,'w'); %block 0 line
h.LineWidth = 2; %line width
minnodt = [-10, 3, 10, 13, -13, 5, 0, 21, 27, 39]/100; %no conf ind diff
lownodt = [-13, -27, -11, 11, -8, 0, -14, -15, 18, 32]/100; %low conf ind diff
highnodt = [-28, -10, -11, 26, -20, -3, -18, 21, 8, 29]/100; %high conf ind diff
minnodt(2,:) = ctr(1,1); %center
lownodt(2,:) = ctr(1,2);%center
highnodt(2,:) = ctr(1,3);%center
hold on %hold fig
h = hline(0,'--k'); %hline at 0
h.Color = [0.64 0.08 0.18]; %color
c = [0.32 0.61 0.81]; %color
scatter(minnodt(2,:),minnodt(1,:),30,c); %scatter ind
scatter(lownodt(2,:),lownodt(1,:),30,c);%scatter ind
scatter(highnodt(2,:),highnodt(1,:),30,c);%scatter ind
e1 = errorbar(x(1), y(1), errorminus_vert(1), errorplus_vert(1), 'LineStyle', 'none', 'LineWidth', 2); %no conf diff
e1.Color = 'b'; %color
hLine = line([x(1) - errorminus_horz(1), x(1) + errorplus_horz(1)], [y(1), y(1)], 'Color', get(e1, 'Color'), 'LineWidth', 2); %horiz
e2 = errorbar(x(2), y(2), errorminus_vert(2), errorplus_vert(2), 'LineStyle', 'none', 'LineWidth', 2); %low conf diff
e2.Color = 'b';  %color
hLine = line([x(2) - errorminus_horz(2), x(2) + errorplus_horz(2)], [y(2), y(2)], 'Color', get(e2, 'Color'), 'LineWidth', 2);%horiz
e3 = errorbar(x(3), y(3), errorminus_vert(3), errorplus_vert(3), 'LineStyle', 'none', 'LineWidth', 2); %high conf diff
e3.Color = 'b';  %color
hLine = line([x(3) - errorminus_horz(3), x(3) + errorplus_horz(3)], [y(3), y(3)], 'Color', get(e3, 'Color'), 'LineWidth', 2); %horiz
ylim([-0.3 0.4]) %ylim
xlim([0.97 1.13]) %xlim
XTickLabel = ({'No','Low','High'}); %x label
set(gca,'xtick',[1 1.05 1.1],'xticklabel',XTickLabel,'box','off','FontSize',20) %aesthetics
ylabel('Decision Time % Change') %y label

%% dSPN - velocity profile

f = figure; %figure
subplot(1,3,1) %subplot 1
hold on %make the fig
nonan = []; %place to take out NaNs
for avging = 1:60 %go through our 60 points
    nonan = dspn_ctrl_start(:,avging); %pull out trial
    nonan = nonan(~isnan(nonan)); %take out NaN
    avg2(1,avging) = mean(nonan); %avg it
    avg2(2,avging) = std(nonan)/sqrt(length(nonan)); %get the std
end %ending "for" loop going through our 6 points
test = []; %clearing
test(1,:) = (avg2(1,:)); %the avg
y = test(1,:)/1.5; % your mean vector
x = 1:numel(y); %how many inputs are in y
std_dev = avg2(2,:)/1.5; %std
curve1 = y + std_dev; %upper bound
curve2 = y - std_dev; %lower bound
x2 = [x, fliplr(x)]; %x2
inBetween = [curve1, fliplr(curve2)]; %here are the in between values
fi = fill(x2, inBetween,[0.8 0.8 0.8]); %plotting the shading w/ color specified
set(fi,'edgecolor','white'); %take away edges
hold on; %hold plot
p1 = plot(x, y, 'LineWidth', 2); %plot the line
p1.Color = 'k'; %color
nonan = []; %place to take out NaNs
for avging = 1:60 %go through our 60 points
    nonan = dspn_stim_start(:,avging); %pull out trial
    nonan = nonan(~isnan(nonan)); %take out NaN
    avg2(1,avging) = mean(nonan); %avg it
    avg2(2,avging) = std(nonan)/sqrt(length(nonan)); %get the std
end %ending "for" loop going through our 6 points
hold on %hold it
test(1,:) = (avg2(1,:)); %the avg
y = test(1,:)/1.5; % your mean vector
x = 1:numel(y); %how many inputs are in y
std_dev = avg2(2,:)/1.5; %std
curve1 = y + std_dev; %upper bound
curve2 = y - std_dev; %lower bound
x2 = [x, fliplr(x)]; %x2
inBetween = [curve1, fliplr(curve2)]; %here are the in between values
fi = fill(x2, inBetween,[0.65 0.92 0.92],'facealpha', .6); %plotting the shading w/ color specified
set(fi,'edgecolor','white'); %take away edges
hold on; %hold plot
p1 = plot(x, y, 'LineWidth', 2); %plot the line
p1.Color = 'b'; %color
XTickLabel = ({'-500','Trial Start','+500'}); %x label
set(gca,'xtick',[1 30 60],'xticklabel',XTickLabel,'box','off','fontsize',20) %format
xlabel('Time (ms)'); %x label
ylabel('Speed (cm/s)');  %ylabel
%ylim([1 8]) %original ylim
ylim([0 8]) %ylim axis to 0
xline(30,'--g','LineWidth',2); %alignment line
siggies = []; %where to store sig values
for x = 1:60 %go through each time point
    [h,p] = ttest2(dspn_ctrl_start(:,x),dspn_stim_start(:,x)); %test
    siggies(x) = p; %copy the p
end %ending "for" loop going through time points
for x = 1:length(siggies) %go through sig values
    if siggies(x) < 0.0014 %lowest p
        siggies(x) = 3; %mark with a 3
    elseif siggies(x) > 0.001 && siggies(x) < 0.014 %middle p
        siggies(x) = 1; %mark with a 1
    elseif siggies(x) > 0.01 && siggies(x) < 0.054 %highest p
        siggies(x) = 1; %mark with a 1
    else %if no p...
        siggies(x) = 0; %mark with a 0
    end %ending "if" statements looking for p values
end %ending "for" loop going through sig values
patches = []; %storage
s=1; %formatting
if siggies(1) ~= 0 %if there is a starting sig...
    patches(1,1) = 1; %mark it with the starting sig
    patches(2,1) = siggies(1); %mark the place
    s=2; %format
end %ending "if" statement looking for starting sig
for x = 2:length(siggies) %go through sigs
    if siggies(x) ~= siggies(x-1) %if we have a new sig...
        patches(1,s) = x; %mark the sig value
        patches(2,s) = siggies(x); %mark the location
        s=s+1; %formatting
    end %ending "if" statement looking for new sigs
end %ending "for" loop going through sigs
patches(1,s) = 60; %make the last one 60
patches(2,s) = 0; %make the last one no sig to end it
if size(patches,2) > 1 %if we have some sig...
for sigmark = 1:length(patches) %going through the info for making the patches
    if patches(2,sigmark) == 1 %if we have a 1...
        color = [1 0.88 0.59]; %set the color
        p1 = patch([patches(1,sigmark) patches(1,sigmark+1) patches(1,sigmark+1) patches(1,sigmark)], [max(ylim)-0.5 max(ylim)-0.5 max(ylim) max(ylim)], color); %draw the patch
        p1.EdgeColor = [1 1 1]; %take off the edges
        hold on %hold
    elseif patches(2,sigmark) == 2 %if we have a 2...
        color = [0.91 0.34 0.1]; %set the orange
        p1 = patch([patches(1,sigmark) patches(1,sigmark+1) patches(1,sigmark+1) patches(1,sigmark)], [max(ylim)-0.5 max(ylim)-0.5 max(ylim) max(ylim)], color); %draw the patch
        p1.EdgeColor = [1 1 1]; %take off the edges
        hold on %hold
    elseif patches(2,sigmark) == 3 %if we have a 3...
        color = [0.93 0.63 0.19]; %set the red
        p1 = patch([patches(1,sigmark) patches(1,sigmark+1) patches(1,sigmark+1) patches(1,sigmark)], [max(ylim)-0.5 max(ylim)-0.5 max(ylim) max(ylim)], color); %draw the patch
        p1.EdgeColor = [1 1 1]; %take off the edges
        hold on     %hold
    end %ending "if" statement looking for sig
end %ending "for" loop going through to make patches
end %ending "if" statement looking for sigs
subplot(1,3,2) %subplot 2
hold on %make the fig
nonan = []; %place to take out NaNs
for avging = 1:60 %go through our 60 points
    nonan = dspn_ctrl_center(:,avging); %pull out trial
    nonan = nonan(~isnan(nonan)); %take out NaN
    avg2(1,avging) = mean(nonan); %avg it
    avg2(2,avging) = std(nonan)/sqrt(length(nonan)); %get the std
end %ending "for" loop going through our 6 points
test = []; %clearing
test(1,:) = (avg2(1,:)); %the avg
y = test(1,:)/1.5; % your mean vector
x = 1:numel(y); %how many inputs are in y
std_dev = avg2(2,:)/1.5; %std
curve1 = y + std_dev; %upper bound
curve2 = y - std_dev; %lower bound
x2 = [x, fliplr(x)]; %x2
inBetween = [curve1, fliplr(curve2)]; %here are the in between values
fi = fill(x2, inBetween,[0.8 0.8 0.8]); %plotting the shading w/ color specified
set(fi,'edgecolor','white'); %take away edges
hold on; %hold plot
p1 = plot(x, y, 'LineWidth', 2); %plot the line
p1.Color = 'k'; %color
nonan = []; %place to take out NaNs
for avging = 1:60 %go through our 60 points
    nonan = dspn_stim_center(:,avging); %pull out trial
    nonan = nonan(~isnan(nonan)); %take out NaN
    avg2(1,avging) = mean(nonan); %avg it
    avg2(2,avging) = std(nonan)/sqrt(length(nonan)); %get the std
end %ending "for" loop going through our 6 points
hold on %hold it
test(1,:) = (avg2(1,:)); %the avg
y = test(1,:)/1.5; % your mean vector
x = 1:numel(y); %how many inputs are in y
std_dev = avg2(2,:)/1.5; %std
curve1 = y + std_dev; %upper bound
curve2 = y - std_dev; %lower bound
x2 = [x, fliplr(x)]; %x2
inBetween = [curve1, fliplr(curve2)]; %here are the in between values
fi = fill(x2, inBetween,[0.65 0.92 0.92],'facealpha', .6); %plotting the shading w/ color specified
set(fi,'edgecolor','white'); %take away edges
hold on; %hold plot
p1 = plot(x, y, 'LineWidth', 2); %plot the line
p1.Color = 'b'; %color
XTickLabel = ({'-500','Center Entry','+500'}); %x label
set(gca,'xtick',[1 30 60],'xticklabel',XTickLabel,'box','off','fontsize',20) %format
xlabel('Time (ms)'); %x label
ylabel('Speed (cm/s)');  %y label
%ylim([1 8]) %original y lim
ylim([0 8]) %y lim axis to 0
xline(30,'--g','LineWidth',2); %alignment line
siggies = []; %where to store sig values
for x = 1:60 %go through each time point
    [h,p] = ttest2(dspn_ctrl_center(:,x),dspn_stim_center(:,x)); %test
    siggies(x) = p; %copy the p
end %ending "for" loop going through time points
for x = 1:length(siggies) %go through sig values
    if siggies(x) < 0.0014 %lowest p
        siggies(x) = 3; %mark with a 3
    elseif siggies(x) > 0.001 && siggies(x) < 0.014 %middle p
        siggies(x) = 1; %mark with a 1
    elseif siggies(x) > 0.01 && siggies(x) < 0.054 %highest p
        siggies(x) = 1; %mark with a 1
    else %if no p...
        siggies(x) = 0; %mark with a 0
    end %ending "if" statements looking for p values
end %ending "for" loop going through sig values
patches = []; %storage
s=1; %formatting
if siggies(1) ~= 0 %if there is a centering sig...
    patches(1,1) = 1; %mark it with the centering sig
    patches(2,1) = siggies(1); %mark the place
    s=2; %format
end %ending "if" statement looking for centering sig
for x = 2:length(siggies) %go through sigs
    if siggies(x) ~= siggies(x-1) %if we have a new sig...
        patches(1,s) = x; %mark the sig value
        patches(2,s) = siggies(x); %mark the location
        s=s+1; %formatting
    end %ending "if" statement looking for new sigs
end %ending "for" loop going through sigs
patches(1,s) = 60; %make the last one 60
patches(2,s) = 0; %make the last one no sig to end it
if size(patches,2) > 1 %if we have some sig...
for sigmark = 1:length(patches) %going through the info for making the patches
    if patches(2,sigmark) == 1 %if we have a 1...
        color = [1 0.88 0.59]; %set the color
        p1 = patch([patches(1,sigmark) patches(1,sigmark+1) patches(1,sigmark+1) patches(1,sigmark)], [max(ylim)-0.5 max(ylim)-0.5 max(ylim) max(ylim)], color); %draw the patch
        p1.EdgeColor = [1 1 1]; %take off the edges
        hold on %hold
    elseif patches(2,sigmark) == 2 %if we have a 2...
        color = [0.91 0.34 0.1]; %set the orange
        p1 = patch([patches(1,sigmark) patches(1,sigmark+1) patches(1,sigmark+1) patches(1,sigmark)], [max(ylim)-0.5 max(ylim)-0.5 max(ylim) max(ylim)], color); %draw the patch
        p1.EdgeColor = [1 1 1]; %take off the edges
        hold on %hold
    elseif patches(2,sigmark) == 3 %if we have a 3...
        color = [0.93 0.63 0.19]; %set the red
        p1 = patch([patches(1,sigmark) patches(1,sigmark+1) patches(1,sigmark+1) patches(1,sigmark)], [max(ylim)-0.5 max(ylim)-0.5 max(ylim) max(ylim)], color); %draw the patch
        p1.EdgeColor = [1 1 1]; %take off the edges
        hold on     %hold
    end %ending "if" statement looking for sig
end %ending "for" loop going through to make patches
end %ending "if" statement looking for sigs
subplot(1,3,3) %subplot 3
hold on %make the fig
nonan = []; %place to take out NaNs
for avging = 1:60 %go through our 60 points
    nonan = dspn_ctrl_laser(:,avging); %pull out trial
    nonan = nonan(~isnan(nonan)); %take out NaN
    avg2(1,avging) = mean(nonan); %avg it
    avg2(2,avging) = std(nonan)/sqrt(length(nonan)); %get the std
end %ending "for" loop going through our 6 points
test = []; %clearing
test(1,:) = (avg2(1,:)); %the avg
y = test(1,:)/1.5; % your mean vector
x = 1:numel(y); %how many inputs are in y
std_dev = avg2(2,:)/1.5; %std
curve1 = y + std_dev; %upper bound
curve2 = y - std_dev; %lower bound
x2 = [x, fliplr(x)]; %x2
inBetween = [curve1, fliplr(curve2)]; %here are the in between values
fi = fill(x2, inBetween,[0.8 0.8 0.8]); %plotting the shading w/ color specified
set(fi,'edgecolor','white'); %take away edges
hold on; %hold plot
p1 = plot(x, y, 'LineWidth', 2); %plot the line
p1.Color = 'k'; %color
nonan = []; %place to take out NaNs
for avging = 1:60 %go through our 60 points
    nonan = dspn_stim_laser(:,avging); %pull out trial
    nonan = nonan(~isnan(nonan)); %take out NaN
    avg2(1,avging) = mean(nonan); %avg it
    avg2(2,avging) = std(nonan)/sqrt(length(nonan)); %get the std
end %ending "for" loop going through our 6 points
hold on %hold it
test(1,:) = (avg2(1,:)); %the avg
y = test(1,:)/1.5; % your mean vector
x = 1:numel(y); %how many inputs are in y
std_dev = avg2(2,:)/1.5; %std
curve1 = y + std_dev; %upper bound
curve2 = y - std_dev; %lower bound
x2 = [x, fliplr(x)]; %x2
inBetween = [curve1, fliplr(curve2)]; %here are the in between values
fi = fill(x2, inBetween,[0.65 0.92 0.92],'facealpha', .6); %plotting the shading w/ color specified
set(fi,'edgecolor','white'); %take away edges
hold on; %hold plot
p1 = plot(x, y, 'LineWidth', 2); %plot the line
p1.Color = 'b'; %color
XTickLabel = ({'Laser','+800'}); %x label
set(gca,'xtick',[12 60],'xticklabel',XTickLabel,'box','off','fontsize',20) %format
xlabel('Time (ms)'); %x label
ylabel('Speed (cm/s)');  %y label
%ylim([1 8]) %y lim
ylim([0 8]) %y lim axis to 0
xline(12,'--g','LineWidth',2); %alignment line
color = [0.64 0.85 1]; %color for stim patch
p1 = patch([12 48 48 12], [min(ylim) min(ylim) max(ylim) max(ylim)], color); %draw stim patch
p1.FaceAlpha = 0.2; p1.EdgeColor = [1 1 1]; %translucence
siggies = []; %where to store sig values
for x = 1:60 %go through each time point
    [h,p] = ttest2(dspn_ctrl_laser(:,x),dspn_stim_laser(:,x)); %test
    siggies(x) = p; %copy the p
end %ending "for" loop going through time points
for x = 1:length(siggies) %go through sig values
    if siggies(x) < 0.0014 %lowest p
        siggies(x) = 3; %mark with a 3
    elseif siggies(x) > 0.001 && siggies(x) < 0.014 %middle p
        siggies(x) = 1; %mark with a 1
    elseif siggies(x) > 0.01 && siggies(x) < 0.054 %highest p
        siggies(x) = 1; %mark with a 1
    else %if no p...
        siggies(x) = 0; %mark with a 0
    end %ending "if" statements looking for p values
end %ending "for" loop going through sig values
patches = []; %storage
s=1; %formatting
if siggies(1) ~= 0 %if there is a lasering sig...
    patches(1,1) = 1; %mark it with the lasering sig
    patches(2,1) = siggies(1); %mark the place
    s=2; %format
end %ending "if" statement looking for lasering sig
for x = 2:length(siggies) %go through sigs
    if siggies(x) ~= siggies(x-1) %if we have a new sig...
        patches(1,s) = x; %mark the sig value
        patches(2,s) = siggies(x); %mark the location
        s=s+1; %formatting
    end %ending "if" statement looking for new sigs
end %ending "for" loop going through sigs
patches(1,s) = 60; %make the last one 60
patches(2,s) = 0; %make the last one no sig to end it
if size(patches,2) > 1 %if we have some sig...
for sigmark = 1:length(patches) %going through the info for making the patches
    if patches(2,sigmark) == 1 %if we have a 1...
        color = [1 0.88 0.59]; %set the color
        p1 = patch([patches(1,sigmark) patches(1,sigmark+1) patches(1,sigmark+1) patches(1,sigmark)], [max(ylim)-0.5 max(ylim)-0.5 max(ylim) max(ylim)], color); %draw the patch
        p1.EdgeColor = [1 1 1]; %take off the edges
        hold on %hold
    elseif patches(2,sigmark) == 2 %if we have a 2...
        color = [0.91 0.34 0.1]; %set the orange
        p1 = patch([patches(1,sigmark) patches(1,sigmark+1) patches(1,sigmark+1) patches(1,sigmark)], [max(ylim)-0.5 max(ylim)-0.5 max(ylim) max(ylim)], color); %draw the patch
        p1.EdgeColor = [1 1 1]; %take off the edges
        hold on %hold
    elseif patches(2,sigmark) == 3 %if we have a 3...
        color = [0.93 0.63 0.19]; %set the red
        p1 = patch([patches(1,sigmark) patches(1,sigmark+1) patches(1,sigmark+1) patches(1,sigmark)], [max(ylim)-0.5 max(ylim)-0.5 max(ylim) max(ylim)], color); %draw the patch
        p1.EdgeColor = [1 1 1]; %take off the edges
        hold on     %hold
    end %ending "if" statement looking for sig
end %ending "for" loop going through to make patches
end %ending "if" statement looking for sigs
child = get(gca,'children'); %getting children for legend
legend([child(7) child(6) child(2) child(3)],'Ctrl','Stim','p < 0.05','p < 0.01','AutoUpdate','off') %legend

%% Summary Figure

figure; %fig
plot(2,2,'Marker','.','MarkerSize',70,'Color',[1 0.65 0.5]) %high certainty
hold on %hold fig
plot(1.5,3,'Marker','.','MarkerSize',70,'Color',[0.93 0.69 0.13])%high conf
plot(2.5,1.5,'Marker','.','MarkerSize',70','Color',[0.89 0.57 0.44]) %high vol
plot(1,1,'Marker','.','MarkerSize',70','Color',[1 0 1]) %ispn
plot(3,1,'Marker','.','MarkerSize',70','Color',[0.98 0.59 0.83]) %dspn
c = colorbar('TickLabels',[]); %color bar
c.Label.String = 'Decision Time'; %color bar label
set(c,'Ydir','reverse') %switch direction
set(gca,'FontSize',24,'box','off','XTickLabel',[],'YTickLabel',[]) %aesthetics
xlabel('Outcome Responsiveness') %x label
ylabel('Bias') %y label
ylim([0.5 3.5]); xlim([0.5 3.5]) %figure limits

% SET COLORMAP WITHIN EDITOR TO "SPRING"

%% supplemental: bias across training

figure; %fig
hold on %hold fig
h = hline(0,'w'); %block 0 line
h.LineWidth = 2; %line width
c = [0.7 0.7 0.7]; %color
for ind = 1:10 %go through individuals
    space(1:14) = ind; %pull out each individual
    scatter(space,bias(:,ind),30,c); %scatter individuals
end %ending "for" loop going through training data
e3 = errorbar(bias_avg,bias_std,'Marker','.','MarkerSize',18,'LineWidth',2); %plotting the bias
e3.Color = [0.5 0.5 0.5]; %color
xlim([0 11]) %setting the x lim
ylim([-0.1 1.8]) %setting y lim
XTickLabel = ({'1','2','3','4','5','-5','-4','-3','-2','-1'}); %x label
set(gca,'box','off','XTick',[1:1:10],'xticklabel',XTickLabel,'FontSize',20) %aesthetic
xlabel('Training Day'); ylabel('Bias'); %labels

%% supplemental: regression bias

f = figure; %fig
ctr = []; ydt = []; %space
y = [mean(min_no_bias); mean(low_no_bias); mean(high_no_bias)]; %regression bias
a_vert = [std(min_no_bias)/sqrt(length(min_no_bias)); std(low_no_bias)/sqrt(length(low_no_bias)); std(high_no_bias)/sqrt(length(high_no_bias))]; %SEM
errorplus_vert=a_vert'; %positive SEM
errorminus_vert=errorplus_vert; %negative SEM
x = [1; 1.05; 1.1]; %x position
a_horz = [0.04;0.04;0.04]/2; %horizontal lines
errorplus_horz=a_horz'; %positive horiz
errorminus_horz=errorplus_horz; %negative horiz
hBar = bar(x,y,'FaceColor',[1 1 1],'EdgeColor',[1 1 1]); %plot bias
for k1 = 1:size(y,2) %"for" loop going through y
    ctr(k1,:) = bsxfun(@plus, hBar(k1).XData, hBar(k1).XOffset');     %ctr
    ydt(k1,:) = hBar(k1).YData;                    %ydt
end %ending "for" loop going through y
h = hline(0,'w'); %blocking 0 line
h.LineWidth = 2; %line width
minnoc = [0.09,0.27,0.14,0.14,0.21,0.15,0.04,0.19,0.17,0.09,0.01,0.12,0.36]; %no conf ind
lownoc = [0,0.33,0.06,0.06,0.51,0.04,0.02,0.22,0.18,0.06,0.1,0.17,0.27]; %low conf ind
highnoc = [0.13,0.24,0.19,0.06,0.47,0.36,0.32,0.41,0.32,0.04,0.06,0.01,0.39]; %high conf ind
minnoc(2,:) = ctr(1,1); %center
lownoc(2,:) = ctr(1,2);%center
highnoc(2,:) = ctr(1,3);%center
hold on %hold fig
c = [0.96 0.65 0.71]; %color
scatter(minnoc(2,:),minnoc(1,:),30,c); %scatter ind
scatter(lownoc(2,:),lownoc(1,:),30,c);%scatter ind
scatter(highnoc(2,:),highnoc(1,:),30,c);%scatter ind
e1 = errorbar(x(1), y(1), errorminus_vert(1), errorplus_vert(1), 'LineStyle', 'none', 'LineWidth', 2); %no conf bias
e1.Color = 'r'; %color
hLine = line([x(1) - errorminus_horz(1), x(1) + errorplus_horz(1)], [y(1), y(1)], 'Color', get(e1, 'Color'), 'LineWidth', 2); %horiz
e2 = errorbar(x(2), y(2), errorminus_vert(2), errorplus_vert(2), 'LineStyle', 'none', 'LineWidth', 2); %low conf bias
e2.Color = 'r'; %color
hLine = line([x(2) - errorminus_horz(2), x(2) + errorplus_horz(2)], [y(2), y(2)], 'Color', get(e2, 'Color'), 'LineWidth', 2);%horiz
e3 = errorbar(x(3), y(3), errorminus_vert(3), errorplus_vert(3), 'LineStyle', 'none', 'LineWidth', 2); %high conf bias
e3.Color = 'r'; %color
hLine = line([x(3) - errorminus_horz(3), x(3) + errorplus_horz(3)], [y(3), y(3)], 'Color', get(e3, 'Color'), 'LineWidth', 2); %horiz
ylim([-0.1 0.6]) %ylim
xlim([0.97 1.13]) %xlim
XTickLabel = ({'No','Low','High'}); %x label
set(gca,'xtick',[1 1.05 1.1],'xticklabel',XTickLabel,'box','off','FontSize',20) %aesthetics
ylabel('Regression Weight') %y label
X(1) = ctr(1,1); X(2) = ctr(1,3); Y(1) = 0.54; Y(2) = 0.54; %sig bar
plot(X(1:2),Y,'LineWidth',1.5,'Color','k') %plot

%% Supplemental: outcome vs choice regression - low vs high volatility

f = figure; %fig
ctr = []; ydt = []; %space
y = [mean(min_no_r1) mean(minhigh_no_r1) mean(min_no_c1) mean(minhigh_no_c1); mean(low_no_r1) mean(lowhigh_no_r1) mean(low_no_c1) mean(lowhigh_no_c1); mean(high_no_r1) mean(highhigh_no_r1) mean(high_no_c1) mean(highhigh_no_c1)]; %regression weights
a_vert = [std(min_no_r1)/sqrt(length(min_no_r1)) std(minhigh_no_r1)/sqrt(length(minhigh_no_r1)) std(min_no_c1)/sqrt(length(min_no_c1)) std(minhigh_no_c1)/sqrt(length(minhigh_no_c1)); std(low_no_r1)/sqrt(length(low_no_r1)) std(lowhigh_no_r1)/sqrt(length(lowhigh_no_r1)) std(low_no_c1)/sqrt(length(low_no_c1)) std(lowhigh_no_c1)/sqrt(length(lowhigh_no_c1)); std(high_no_r1)/sqrt(length(high_no_r1)) std(highhigh_no_r1)/sqrt(length(highhigh_no_r1)) std(high_no_c1)/sqrt(length(high_no_c1)) std(highhigh_no_c1)/sqrt(length(highhigh_no_c1))]; %SEM
errorplus_vert=a_vert'; %positive SEM
errorminus_vert=errorplus_vert; %negative SEM
x = [1; 1.1; 1.2]; %x position
a_horz = [0.005;0.005;0.005]; %horizontal lines
errorplus_horz=a_horz'; %positive horiz
errorminus_horz=errorplus_horz; %negative horiz
hBar = bar(x,y,'FaceColor',[1 1 1],'EdgeColor',[1 1 1]); %plot regression coefficients
for k1 = 1:size(y,2) %"for" loop going through y
    ctr(k1,:) = bsxfun(@plus, hBar(k1).XData, hBar(k1).XOffset');      %ctr
    ydt(k1,:) = hBar(k1).YData;                     %ydt
end %ending "for" loop going through y
h = hline(0,'w'); %line blocking 0
h.LineWidth = 2; %ine width
minnowsL = [0.42 0.66 1 0.48 1.12 1.5 0.84 0.7 1.05 0.89 0.66]; %NL outcome ind
 minhighwsL = [0.79 0.62 0.68 0.31 0.97 1.32 1.58 0.89 1.22 0.85 0.61]; %NH outcome ind
 lownowsL = [0.62 0.77 0.93 0.41 1.45 1.2 0.84 0.78 1.22 0.73 0.81]; %LL outcome ind
 lowhighwsL = [0.5 0.5 0.9 0.34 1 1.36 1.46 0.79 1.09 0.93 0.37]; %LH outcome ind
 highnowsL = [0.39 0.76 0.89 0.28 1.26 1.61 0.91 0.67 1.09 0.77 0.78]; %HL outcome ind
 highhighwsL = [0.54 0.56 0.7 0.45 0.98 1.17 1.03 0.87 1.06 0.78 0.68]; %HH outcome ind
 minnowsR = [0.88 1.02 0.62 0.27 0.72 0.49 0.42 0.7 0.5 0.15 0.36]; %NL choice ind
 minhighwsR = [0.95 1.16 0.46 0.05 0.58 0.79 0.46 1.1 0.56 0.24 0.27]; %NH choice ind
 lownowsR = [0.87 1.01 0.7 0.42 0.72 0.6 0.3 0.7 0.26 0.09 0.5]; %LL choice ind
 lowhighwsR = [0.78 0.98 0.61 0.01 0.93 0.79 0.64 0.75 0.12 0.04 0.47]; %LH choice ind
 highnowsR = [0.92 0.88 0.44 0.41 0.85 0.58 0.36 0.93 0.47 0.04 0.58]; %HL choice ind
 highhighwsR = [0.79 0.98 0.55 0.01 0.68 0.55 0.72 0.79 0.48 0.01 0.43]; %HH choice ind
 minnowsL(2,:) = ctr(1,1); %center
 minhighwsL(2,:) = ctr(2,1);%center
 minnowsR(2,:) = ctr(3,1);%center
 minhighwsR(2,:) = ctr(4,1);%center
 lownowsL(2,:) = ctr(1,2);%center
 lowhighwsL(2,:) = ctr(2,2);%center
 lownowsR(2,:) = ctr(3,2);%center
 lowhighwsR(2,:) = ctr(4,2);%center
 highnowsL(2,:) = ctr(1,3);%center
 highhighwsL(2,:) = ctr(2,3);%center
 highnowsR(2,:) = ctr(3,3);%center
 highhighwsR(2,:) = ctr(4,3);%center
hold on %hold fig
scatter(minnowsL(2,:),minnowsL(1,:),30,[0.61 0.87 0.97]); %scatter ind
scatter(minhighwsL(2,:),minhighwsL(1,:),30,[0.62 0.77 1]);%scatter ind
scatter(lownowsL(2,:),lownowsL(1,:),30,[0.61 0.87 0.97]);%scatter ind
scatter(lowhighwsL(2,:),lowhighwsL(1,:),30,[0.62 0.77 1]);%scatter ind
scatter(highnowsL(2,:),highnowsL(1,:),30,[0.61 0.87 0.97]);%scatter ind
scatter(highhighwsL(2,:),highhighwsL(1,:),30,[0.62 0.77 1]);%scatter ind
scatter(minnowsR(2,:),minnowsR(1,:),30,[0.88 0.71 0.98]);%scatter ind
scatter(minhighwsR(2,:),minhighwsR(1,:),30,[0.91 0.51 1]);%scatter ind
scatter(lownowsR(2,:),lownowsR(1,:),30,[0.88 0.71 0.98]);%scatter ind
scatter(lowhighwsR(2,:),lowhighwsR(1,:),30,[0.91 0.51 1]);%scatter ind
scatter(highnowsR(2,:),highnowsR(1,:),30,[0.88 0.71 0.98]);%scatter ind
scatter(highhighwsR(2,:),highhighwsR(1,:),30,[0.91 0.51 1]);%scatter ind
X(1) = ctr(1,1); X(2) = ctr(2,1); Y(1) = y(1,1); Y(2) = y(1,2); %average
plot(X(1:2),Y,'LineWidth',1.5,'Color','k') %plot lines
X(1) = ctr(3,1); X(2) = ctr(4,1); Y(1) = y(1,3); Y(2) = y(1,4);%average
plot(X(1:2),Y,'LineWidth',1.5,'Color','k') %plot lines
X(1) = ctr(1,2); X(2) = ctr(2,2); Y(1) = y(2,1); Y(2) = y(2,2);%average
plot(X(1:2),Y,'LineWidth',1.5,'Color','k') %plot lines
X(1) = ctr(3,2); X(2) = ctr(4,2); Y(1) = y(2,3); Y(2) = y(2,4);%average
plot(X(1:2),Y,'LineWidth',1.5,'Color','k') %plot lines
X(1) = ctr(1,3); X(2) = ctr(2,3); Y(1) = y(3,1); Y(2) = y(3,2);%average
plot(X(1:2),Y,'LineWidth',1.5,'Color','k') %plot lines
X(1) = ctr(3,3); X(2) = ctr(4,3); Y(1) = y(3,3); Y(2) = y(3,4);%average
plot(X(1:2),Y,'LineWidth',1.5,'Color','k') %plot lines
e1 = errorbar(ctr(1,1), y(1,1),errorminus_vert(1,1),errorplus_vert(1,1),'LineStyle','none','LineWidth',2); %NL outcome
e1.Color = '[0.3 0.75 0.93]'; %color
hLine = line([ctr(1,1) - errorminus_horz(1), ctr(1,1) + errorplus_horz(1)], [y(1,1), y(1,1)], 'Color', get(e1, 'Color'), 'LineWidth', 2); %horiz
e12 = errorbar(ctr(2,1), y(1,2),errorminus_vert(2,1),errorplus_vert(2,1),'LineStyle','none','LineWidth',2); %NH outcome
e12.Color = 'b'; %color
hLine = line([ctr(2,1) - errorminus_horz(1), ctr(2,1) + errorplus_horz(1)], [y(1,2), y(1,2)], 'Color', get(e12, 'Color'), 'LineWidth', 2);%horiz
e13 = errorbar(ctr(3,1), y(1,3),errorminus_vert(3,1),errorplus_vert(3,1),'LineStyle','none','LineWidth',2); %NL choice
e13.Color = '[0.76 0.62 0.79]'; %color
hLine = line([ctr(3,1) - errorminus_horz(1), ctr(3,1) + errorplus_horz(1)], [y(1,3), y(1,3)], 'Color', get(e13, 'Color'), 'LineWidth', 2);%horiz
e14 = errorbar(ctr(4,1), y(1,4),errorminus_vert(4,1),errorplus_vert(4,1),'LineStyle','none','LineWidth',2); %NH choice
e14.Color = '[0.49 0.18 0.56]';%color
hLine = line([ctr(4,1) - errorminus_horz(1), ctr(4,1) + errorplus_horz(1)], [y(1,4), y(1,4)], 'Color', get(e14, 'Color'), 'LineWidth', 2);%horiz
e2 = errorbar(ctr(1,2), y(2,1),errorminus_vert(1,2),errorplus_vert(1,2),'LineStyle','none','LineWidth',2); %LL outcome
e2.Color = '[0.3 0.75 0.93]'; %color
hLine = line([ctr(1,2) - errorminus_horz(1), ctr(1,2) + errorplus_horz(1)], [y(2,1), y(2,1)], 'Color', get(e2, 'Color'), 'LineWidth', 2);%horiz
e22 = errorbar(ctr(2,2), y(2,2),errorminus_vert(2,2),errorplus_vert(2,2),'LineStyle','none','LineWidth',2); %LH outcome
e22.Color = 'b'; %color
hLine = line([ctr(2,2) - errorminus_horz(1), ctr(2,2) + errorplus_horz(1)], [y(2,2), y(2,2)], 'Color', get(e22, 'Color'), 'LineWidth', 2);%horiz
e23 = errorbar(ctr(3,2), y(2,3),errorminus_vert(3,2),errorplus_vert(3,2),'LineStyle','none','LineWidth',2); %LL choice
e23.Color = '[0.76 0.62 0.79]'; %color
hLine = line([ctr(3,2) - errorminus_horz(1), ctr(3,2) + errorplus_horz(1)], [y(2,3), y(2,3)], 'Color', get(e23, 'Color'), 'LineWidth', 2);%horiz
e24 = errorbar(ctr(4,2), y(2,4),errorminus_vert(4,2),errorplus_vert(4,2),'LineStyle','none','LineWidth',2); %LH choice
e24.Color = '[0.49 0.18 0.56]';%color
hLine = line([ctr(4,2) - errorminus_horz(1), ctr(4,2) + errorplus_horz(1)], [y(2,4), y(2,4)], 'Color', get(e24, 'Color'), 'LineWidth', 2);%horiz
e3 = errorbar(ctr(1,3), y(3,1),errorminus_vert(1,3),errorplus_vert(1,3),'LineStyle','none','LineWidth',2); %HL outcome
e3.Color = '[0.3 0.75 0.93]'; %color
hLine = line([ctr(1,3) - errorminus_horz(1), ctr(1,3) + errorplus_horz(1)], [y(3,1), y(3,1)], 'Color', get(e3, 'Color'), 'LineWidth', 2);%horiz
e32 = errorbar(ctr(2,3), y(3,2),errorminus_vert(2,3),errorplus_vert(2,3),'LineStyle','none','LineWidth',2); %HH outcome
e32.Color = 'b'; %color
hLine = line([ctr(2,3) - errorminus_horz(1), ctr(2,3) + errorplus_horz(1)], [y(3,2), y(3,2)], 'Color', get(e32, 'Color'), 'LineWidth', 2);%horiz
e33 = errorbar(ctr(3,3), y(3,3),errorminus_vert(3,3),errorplus_vert(3,3),'LineStyle','none','LineWidth',2);  %HL choice
e33.Color = '[0.76 0.62 0.79]'; %color
hLine = line([ctr(3,3) - errorminus_horz(1), ctr(3,3) + errorplus_horz(1)], [y(3,3), y(3,3)], 'Color', get(e33, 'Color'), 'LineWidth', 2);%horiz
e34 = errorbar(ctr(4,3), y(3,4),errorminus_vert(4,3),errorplus_vert(4,3),'LineStyle','none','LineWidth',2); %HH choice
e34.Color = '[0.49 0.18 0.56]';%color
hLine = line([ctr(4,3) - errorminus_horz(1), ctr(4,3) + errorplus_horz(1)], [y(3,4), y(3,4)], 'Color', get(e34, 'Color'), 'LineWidth', 2);%horiz
ylim([-0.5 2.5]) %ylim
XTickLabel = ({'No','Low','High'}); %x label
set(gca,'xtick',[1 1.1 1.2],'xticklabel',XTickLabel,'box','off','FontSize',20) %aesthetics
ylabel('Regression Weight') %y label
legend({'Low Vol Outcome','High Vol Outcome','Low Vol Choice','High Vol Choice'},'AutoUpdate','off') %legend

%% Supplemental: Low vs High Volatility - Learning Rate

f = figure; %fig
ctr = []; ydt = []; %space
y = [mean(noconflowvolctrl_learning_rate) mean(noconfhighvolctrl_learning_rate); mean(lowconflowvolctrl_learning_rate) mean(lowconfhighvolctrl_learning_rate); mean(highconflowvolctrl_learning_rate) mean(highconfhighvolctrl_learning_rate)]; %LR
a_vert = [std(noconflowvolctrl_learning_rate)/sqrt(length(noconflowvolctrl_learning_rate)) std(noconfhighvolctrl_learning_rate)/sqrt(length(noconfhighvolctrl_learning_rate)); std(lowconflowvolctrl_learning_rate)/sqrt(length(lowconflowvolctrl_learning_rate)) std(lowconfhighvolctrl_learning_rate)/sqrt(length(lowconfhighvolctrl_learning_rate)); std(highconflowvolctrl_learning_rate)/sqrt(length(highconflowvolctrl_learning_rate)) std(highconfhighvolctrl_learning_rate)/sqrt(length(highconfhighvolctrl_learning_rate))]; %SEM
errorplus_vert=a_vert'; %positive SEM
errorminus_vert=errorplus_vert; %negative SEM
x = [1; 1.1; 1.2]; %x position
a_horz = [0.01;0.01;0.01]; %horizontal lines
errorplus_horz=a_horz'; %positive horiz
errorminus_horz=errorplus_horz; %negative horiz
hBar = bar(x,y,'FaceColor',[1 1 1],'EdgeColor',[1 1 1]); %plot LR
for k1 = 1:size(y,2) %"for" loop going through y
    ctr(k1,:) = bsxfun(@plus, hBar(k1).XData, hBar(k1).XOffset'); %ctr    
    ydt(k1,:) = hBar(k1).YData;                    %ydt
end %ending "for" loop going through y
minnoc = [1,1,0.71,1,0.6,1,0.96,1,1,1,0.99]; %NL LR
minnohighvol = [1,0.9,1,1,1,1,1,1,0.82,0.43,1]; %NH LR
lownoc = [0.8,0.71,0.65,1,0.99,1,1,1,1,0.5,0.67,1,0.63]; %LL LR
lownohighvol = [0.45,0.93,0.71,1,0.97,1,1,0.92,0.97,0.84,1]; %LH LR
highnoc = [0.54,0.69,0.54,1,1,1,1,0.79,1,0.91,0.94,0.4,0.85]; %HL LR
highnohighvol = [0.68,0.85,1,0.73,1,1,1,1,0.67,0.93,1]; %HH LR
minnoc(2,:) = ctr(1,1); %center
minnohighvol(2,:) = ctr(2,1);%center
lownoc(2,:) = ctr(1,2);%center
lownohighvol(2,:) = ctr(2,2);%center
highnoc(2,:) = ctr(1,3);%center
highnohighvol(2,:) = ctr(2,3);%center
hold on %hold fig
for ugh = 1:11 %"for" loop going through ind
    X(1) =  minnoc(2,ugh); %low vol X
    Y(1) =  minnoc(1,ugh); %low vol Y
    X(2) =  minnohighvol(2,ugh); %high vol X
    Y(2) =  minnohighvol(1,ugh); %high vol Y
    plot(X(1:2),Y,'Color','[0.7 0.7 0.7]') %plot ind lines
end %ending "for" loop going through ind
for ugh = 1:11%"for" loop going through ind
    X(1) =  lownoc(2,ugh);%low vol X
    Y(1) =  lownoc(1,ugh); %low vol Y
    X(2) =  lownohighvol(2,ugh); %high vol X
    Y(2) =  lownohighvol(1,ugh);%high vol Y
    plot(X(1:2),Y,'Color','[0.7 0.7 0.7]')%plot ind lines
end%ending "for" loop going through ind
for ugh = 1:11%"for" loop going through ind
    X(1) =  highnoc(2,ugh);%low vol X
    Y(1) =  highnoc(1,ugh); %low vol Y
    X(2) =  highnohighvol(2,ugh); %high vol X
    Y(2) =  highnohighvol(1,ugh);%high vol Y
    plot(X(1:2),Y,'Color','[0.7 0.7 0.7]')%plot ind lines
end%ending "for" loop going through ind
X(1) = ctr(1,1); X(2) = ctr(2,1); Y(1) = y(1,1); Y(2) = y(1,2); %plot avg
plot(X(1:2),Y,'LineWidth',1.5,'Color','k') %avg line
X(1) = ctr(1,2); X(2) = ctr(2,2); Y(1) = y(2,1); Y(2) = y(2,2);%plot avg
plot(X(1:2),Y,'LineWidth',1.5,'Color','k')%avg line
X(1) = ctr(1,3); X(2) = ctr(2,3); Y(1) = y(3,1); Y(2) = y(3,2);%plot avg
plot(X(1:2),Y,'LineWidth',1.5,'Color','k')%avg line
e1 = errorbar(ctr(1,1), y(1,1),errorminus_vert(1,1),errorplus_vert(1,1),'LineStyle','none','LineWidth',2); %NL LR
e1.Color = '[0.4 0.4 0.4]'; %color
hLine = line([ctr(1,1) - errorminus_horz(1), ctr(1,1) + errorplus_horz(1)], [y(1,1), y(1,1)], 'Color', get(e1, 'Color'), 'LineWidth', 2); %horiz
e12 = errorbar(ctr(2,1), y(1,2),errorminus_vert(2,1),errorplus_vert(2,1),'LineStyle','none','LineWidth',2); %NH LR
e12.Color = 'k'; %color
hLine = line([ctr(2,1) - errorminus_horz(1), ctr(2,1) + errorplus_horz(1)], [y(1,2), y(1,2)], 'Color', get(e12, 'Color'), 'LineWidth', 2);%horiz
e2 = errorbar(ctr(1,2), y(2,1),errorminus_vert(1,2),errorplus_vert(1,2),'LineStyle','none','LineWidth',2);  %LL LR
e2.Color = '[0.4 0.4 0.4]'; %color
hLine = line([ctr(1,2) - errorminus_horz(1), ctr(1,2) + errorplus_horz(1)], [y(2,1), y(2,1)], 'Color', get(e2, 'Color'), 'LineWidth', 2);%horiz
e22 = errorbar(ctr(2,2), y(2,2),errorminus_vert(2,2),errorplus_vert(2,2),'LineStyle','none','LineWidth',2); %LH LR
e22.Color = 'k'; %color
hLine = line([ctr(2,2) - errorminus_horz(1), ctr(2,2) + errorplus_horz(1)], [y(2,2), y(2,2)], 'Color', get(e22, 'Color'), 'LineWidth', 2);%horiz
e3 = errorbar(ctr(1,3), y(3,1),errorminus_vert(1,3),errorplus_vert(1,3),'LineStyle','none','LineWidth',2); %HL LR
e3.Color = '[0.4 0.4 0.4]'; %color
hLine = line([ctr(1,3) - errorminus_horz(1), ctr(1,3) + errorplus_horz(1)], [y(3,1), y(3,1)], 'Color', get(e3, 'Color'), 'LineWidth', 2);%horiz
e32 = errorbar(ctr(2,3), y(3,2),errorminus_vert(2,3),errorplus_vert(2,3),'LineStyle','none','LineWidth',2); %HH LR
e32.Color = 'k'; %color
hLine = line([ctr(2,3) - errorminus_horz(1), ctr(2,3) + errorplus_horz(1)], [y(3,2), y(3,2)], 'Color', get(e32, 'Color'), 'LineWidth', 2);%horiz
%ylim([0.4 1.2]) %original y lim
ylim([0 1.2]) %y lim axis to 0
XTickLabel = ({'No','Low','High'}); %x label
%set(gca,'xtick',[1 1.1
%1.2],'xticklabel',XTickLabel,'ytick',[0.6:0.2:1],'box','off','FontSize',20)
%%original aesthetics
set(gca,'xtick',[1 1.1 1.2],'xticklabel',XTickLabel,'box','off','FontSize',20) %aesthetics
ylabel('Learning Rate') %y label
legend({'Low Vol','High Vol'},'AutoUpdate','off') %legend

%% Supplemental: iSPN Stim Learning Rate

f = figure; %fig
ctr = []; ydt = []; %space
y = [mean(ispn_noconflowvolctrl_learning_rate) mean(ispn_noconflowvolstim_learning_rate); mean(ispn_lowconflowvolctrl_learning_rate) mean(ispn_lowconflowvolstim_learning_rate); mean(ispn_highconflowvolctrl_learning_rate) mean(ispn_highconflowvolstim_learning_rate)]; %LR
a_vert = [std(ispn_noconflowvolctrl_learning_rate)/sqrt(length(ispn_noconflowvolctrl_learning_rate)) std(ispn_noconflowvolstim_learning_rate)/sqrt(length(ispn_noconflowvolstim_learning_rate)); std(ispn_lowconflowvolctrl_learning_rate)/sqrt(length(ispn_lowconflowvolctrl_learning_rate)) std(ispn_lowconflowvolstim_learning_rate)/sqrt(length(ispn_lowconflowvolstim_learning_rate)); std(ispn_highconflowvolctrl_learning_rate)/sqrt(length(ispn_highconflowvolctrl_learning_rate)) std(ispn_highconflowvolstim_learning_rate)/sqrt(length(ispn_highconflowvolstim_learning_rate))]; %SEM
errorplus_vert=a_vert'; %positive SEM
errorminus_vert=errorplus_vert; %negative SEM
x = [1; 1.1; 1.2]; %x position
a_horz = [0.01;0.01;0.01]; %horizontal lines
errorplus_horz=a_horz'; %positive horiz
errorminus_horz=errorplus_horz; %negative horiz
hBar = bar(x,y,'FaceColor',[1 1 1],'EdgeColor',[1 1 1]); %plot LR
for k1 = 1:size(y,2) %"for" loop going through y
    ctr(k1,:) = bsxfun(@plus, hBar(k1).XData, hBar(k1).XOffset');      %ctr
    ydt(k1,:) = hBar(k1).YData;                    %ydt
end %ending "for" loop going through y
minnoc = [1,1,0.71,1,0.6,1,0.9,1,1,1]; %NC LR
minnostim = [0.08 0.03 0.1 0.21 0.5 0.75 0.48 0.9 0.9 0.29]; %NS LR
lownoc = [0.8,0.71,0.65,1,0.99,0.45,0.93,0.71,1,0.97]; %LC LR
lownostim = [0.03 0.28 0.04 0.4 0.9 0.35 0.3 0.42 0.9 0.33]; %LS LR
highnoc = [0.54,0.69,0.54,1,1,0.68,0.85,1,0.73,1]; %HC LR
highnostim = [0.04 0.08 0.03 0.44 0.31 0.2 0.2 0.6 0.69 0.09]; %HS LR
minnoc(2,:) = ctr(1,1); %center
minnostim(2,:) = ctr(2,1); %center
lownoc(2,:) = ctr(1,2); %center
lownostim(2,:) = ctr(2,2); %center
highnoc(2,:) = ctr(1,3); %center
highnostim(2,:) = ctr(2,3); %center
hold on %hold fig
for ugh = 1:10 %"for" loop going through ind
    X(1) =  minnoc(2,ugh); %control X
    Y(1) =  minnoc(1,ugh); %control Y
    X(2) =  minnostim(2,ugh); %stim X
    Y(2) =  minnostim(1,ugh); %stim Y
    plot(X(1:2),Y,'Color','[0.7 0.7 0.7]') %plot individual lines
end %ending "for" loop going through ind
for ugh = 1:10%"for" loop going through ind
    X(1) =  lownoc(2,ugh);%control X
    Y(1) =  lownoc(1,ugh); %control Y
    X(2) =  lownostim(2,ugh);%stim X
    Y(2) =  lownostim(1,ugh);%stim Y
    plot(X(1:2),Y,'Color','[0.7 0.7 0.7]') %plot individual lines
end%ending "for" loop going through ind
for ugh = 1:10%"for" loop going through ind
    X(1) =  highnoc(2,ugh);%control X
    Y(1) =  highnoc(1,ugh); %control Y
    X(2) =  highnostim(2,ugh);%stim X
    Y(2) =  highnostim(1,ugh);%stim Y
    plot(X(1:2),Y,'Color','[0.7 0.7 0.7]') %plot individual lines
end%ending "for" loop going through ind
X(1) = ctr(1,1); X(2) = ctr(2,1); Y(1) = y(1,1); Y(2) = y(1,2); %average lines
plot(X(1:2),Y,'LineWidth',1.5,'Color','k') %plot avg
X(1) = ctr(1,2); X(2) = ctr(2,2); Y(1) = y(2,1); Y(2) = y(2,2);%average lines
plot(X(1:2),Y,'LineWidth',1.5,'Color','k') %plot avg
X(1) = ctr(1,3); X(2) = ctr(2,3); Y(1) = y(3,1); Y(2) = y(3,2);%average lines
plot(X(1:2),Y,'LineWidth',1.5,'Color','k') %plot avg
e1 = errorbar(ctr(1,1), y(1,1),errorminus_vert(1,1),errorplus_vert(1,1),'LineStyle','none','LineWidth',2); %NC LR
e1.Color = '[0.99 0.58 0.58]'; %color
hLine = line([ctr(1,1) - errorminus_horz(1), ctr(1,1) + errorplus_horz(1)], [y(1,1), y(1,1)], 'Color', get(e1, 'Color'), 'LineWidth', 2); %horiz
e12 = errorbar(ctr(2,1), y(1,2),errorminus_vert(2,1),errorplus_vert(2,1),'LineStyle','none','LineWidth',2); %NS LR
e12.Color = 'r'; %color
hLine = line([ctr(2,1) - errorminus_horz(1), ctr(2,1) + errorplus_horz(1)], [y(1,2), y(1,2)], 'Color', get(e12, 'Color'), 'LineWidth', 2);%horiz
e2 = errorbar(ctr(1,2), y(2,1),errorminus_vert(1,2),errorplus_vert(1,2),'LineStyle','none','LineWidth',2); %LC LR
e2.Color = '[0.99 0.58 0.58]'; %color
hLine = line([ctr(1,2) - errorminus_horz(1), ctr(1,2) + errorplus_horz(1)], [y(2,1), y(2,1)], 'Color', get(e2, 'Color'), 'LineWidth', 2);%horiz
e22 = errorbar(ctr(2,2), y(2,2),errorminus_vert(2,2),errorplus_vert(2,2),'LineStyle','none','LineWidth',2); %LS LR
e22.Color = 'r'; %color
hLine = line([ctr(2,2) - errorminus_horz(1), ctr(2,2) + errorplus_horz(1)], [y(2,2), y(2,2)], 'Color', get(e22, 'Color'), 'LineWidth', 2);%horiz
e3 = errorbar(ctr(1,3), y(3,1),errorminus_vert(1,3),errorplus_vert(1,3),'LineStyle','none','LineWidth',2); %HC LR
e3.Color = '[0.99 0.58 0.58]'; %color
hLine = line([ctr(1,3) - errorminus_horz(1), ctr(1,3) + errorplus_horz(1)], [y(3,1), y(3,1)], 'Color', get(e3, 'Color'), 'LineWidth', 2);%horiz
e32 = errorbar(ctr(2,3), y(3,2),errorminus_vert(2,3),errorplus_vert(2,3),'LineStyle','none','LineWidth',2); %HS LR
e32.Color = 'r'; %color
hLine = line([ctr(2,3) - errorminus_horz(1), ctr(2,3) + errorplus_horz(1)], [y(3,2), y(3,2)], 'Color', get(e32, 'Color'), 'LineWidth', 2);%horiz
ylim([0 1.2]) %ylim
XTickLabel = ({'No','Low','High'}); %x label
set(gca,'xtick',[1 1.1 1.2],'xticklabel',XTickLabel,'ytick',[0.2:0.2:1],'box','off','FontSize',20) %aesthetics
ylabel('Learning Rate') %ylabel
legend({'Ctrl','Stim'},'AutoUpdate','off') %legend
X(1) = ctr(1,1); X(2) = ctr(2,1); Y(1) = 1.03; Y(2) = 1.03; %sig bar
plot(X(1:2),Y,'LineWidth',1.5,'Color','k') %plot line
X(1) = ctr(1,2); X(2) = ctr(2,2); Y(1) = 1.03; Y(2) = 1.03;%sig bar
plot(X(1:2),Y,'LineWidth',1.5,'Color','k')%plot line
X(1) = ctr(1,3); X(2) = ctr(2,3); Y(1) = 1.03; Y(2) = 1.03;%sig bar
plot(X(1:2),Y,'LineWidth',1.5,'Color','k')%plot line

%% Supplemental: iSPN - Rew/Unrew/Bias Regression

f = figure; %fig
ctr = []; ydt = []; %space
y = [mean(a2amin_no_rew1) mean(min_stim_rew1) mean(a2amin_no_unrew1)*-1 mean(min_stim_unrew1)*-1 mean(a2amin_no_bias) mean(min_stim_bias); mean(a2alow_no_rew1) mean(low_stim_rew1) mean(a2alow_no_unrew1)*-1 mean(low_stim_unrew1)*-1 mean(a2alow_no_bias) mean(low_stim_bias); mean(a2ahigh_no_rew1) mean(high_stim_rew1) mean(a2ahigh_no_unrew1)*-1 mean(high_stim_unrew1)*-1 mean(a2ahigh_no_bias) mean(high_stim_bias)]; %regression coefficients
a_vert = [std(a2amin_no_rew1)/sqrt(length(a2amin_no_rew1)) std(min_stim_rew1)/sqrt(length(min_stim_rew1)) std(a2amin_no_unrew1)/sqrt(length(a2amin_no_unrew1)) std(min_stim_unrew1)/sqrt(length(min_stim_unrew1)) std(a2amin_no_bias)/sqrt(length(a2amin_no_bias)) std(min_stim_bias)/sqrt(length(min_stim_bias)); std(a2alow_no_rew1)/sqrt(length(a2alow_no_rew1)) std(low_stim_rew1)/sqrt(length(low_stim_rew1)) std(a2alow_no_unrew1)/sqrt(length(a2alow_no_unrew1)) std(low_stim_unrew1)/sqrt(length(low_stim_unrew1)) std(a2alow_no_bias)/sqrt(length(a2alow_no_bias)) std(low_stim_bias)/sqrt(length(low_stim_bias)); std(a2ahigh_no_rew1)/sqrt(length(a2ahigh_no_rew1)) std(high_stim_rew1)/sqrt(length(high_stim_rew1)) std(a2ahigh_no_unrew1)/sqrt(length(a2ahigh_no_unrew1)) std(high_stim_unrew1)/sqrt(length(high_stim_unrew1)) std(a2ahigh_no_bias)/sqrt(length(a2ahigh_no_bias)) std(high_stim_bias)/sqrt(length(high_stim_bias))]; %SEM
errorplus_vert=a_vert'; %positive SEM
errorminus_vert=errorplus_vert; %negative SEM
x = [1; 1.1; 1.2]; %x position
a_horz = [0.005;0.005;0.005]/2; %horizontal lines
errorplus_horz=a_horz'; %positive horiz
errorminus_horz=errorplus_horz; %negative horiz
hBar = bar(x,y,'FaceColor',[1 1 1],'EdgeColor',[1 1 1]); %plot regressions
for k1 = 1:size(y,2) %"for" loop going through y
    ctr(k1,:) = bsxfun(@plus, hBar(k1).XData, hBar(k1).XOffset');     %ctr
    ydt(k1,:) = hBar(k1).YData;                    %ydt
end %ending "for" loop going through y
h = hline(0,'w'); %block 0 line
h.LineWidth = 2; %line width
minnoREW = [1.29 1.68 1.62 0.74 1.04 1.59 1.63 1.14 0.26 1.2]; %NC rew
 minstimREW = [0.35 0.39 0.19 0.39 0.95 0.13 0.01 0.62 0.3 -0.04]; %NS rew
 lownoREW = [1.49 1.78 1.62 0.84 0.82 1.28 1.48 1.51 0.35 1.17]; %LC rew
 lowstimREW = [0.49 0.07 0.01 0.34 0.73 0.15 -0.02 -0.19 -0.04 -0.02]; %LS rew
 highnoREW = [1.32 1.64 1.33 0.69 0.81 1.33 1.54 1.25 0.44 1.4]; %HC rew
 highstimREW = [0.6 0.27 0.38 0.27 0.56 0.14 -0.04 -0.01 -0.01 0.02]; %HS rew
 minnoUNREW = [0.06 0.07 0.38 0.21 0.74 0.08 0.02 0.22 0.36 0.02]; %NC unrew
 minstimUNREW = [0.09 -0.26 0.27 -0.02 0.29 0.2 0.17 0.85 0.01 0.31]; %NS unrew
 lownoUNREW = [0.05 0.04 0.23 0.01 0.64 0.09 0.08 0.28 0.33 0.07]; %LC unrew
 lowstimUNREW = [0.08 -0.04 0.26 -0.12 0.44 0.15 0.08 0.49 0.05 0.2]; %LS unrew
 highnoUNREW = [0.03 0.03 0.44 0.03 0.73 0.05 0.01 0.15 0.47 0.01]; %HC unrew
 highstimUNREW = [-0.14 -0.17 0.23 -0.06 0.64 0.2 0.05 0.15 0.28 0.18]; %HS unrew
  minnoBIAS = [0.18 0.22 0.14 0.14 0.09 0.05 0.23 0 0.04 0.06]; %NC bias
 minstimBIAS = [-0.13 -0.51 -0.07 -0.19 -0.11 -0.03 0.32 0.17 -0.17 0.07]; %NS bias
 lownoBIAS = [0.28 0.21 0.06 0.06 0.06 0.04 0.09 0.03 0.06 0.02]; %LC bias
 lowstimBIAS = [0.1 -0.43 0.1 -0.15 0.02 0.17 0.04 0.09 0.11 0.11]; %LS bias
 highnoBIAS = [0.18 0.14 0.19 0.06 0.04 0.09 0.29 0.06 0.06 0.04]; %HC bias
 highstimBIAS = [-0.01 -0.44 0.01 -0.04 0.14 -0.09 0.29 -0.1 0.03 0.02]; %HS bias
 minnoREW(2,:) = ctr(1,1); %center
 minstimREW(2,:) = ctr(2,1);  %center
 minnoUNREW(2,:) = ctr(3,1); %center
 minstimUNREW(2,:) = ctr(4,1); %center
 minnoBIAS(2,:) = ctr(5,1); %center
 minstimBIAS(2,:) = ctr(6,1); %center
 lownoREW(2,:) = ctr(1,2); %center
 lowstimREW(2,:) = ctr(2,2); %center
 lownoUNREW(2,:) = ctr(3,2); %center
 lowstimUNREW(2,:) = ctr(4,2); %center
 lownoBIAS(2,:) = ctr(5,2); %center
lowstimBIAS(2,:) = ctr(6,2); %center
 highnoREW(2,:) = ctr(1,3); %center
 highstimREW(2,:) = ctr(2,3); %center
 highnoUNREW(2,:) = ctr(3,3); %center
 highstimUNREW(2,:) = ctr(4,3); %center
highnoBIAS(2,:) = ctr(5,3); %center
highstimBIAS(2,:) = ctr(6,3); %center
hold on %hold fig
scatter(minnoREW(2,:),minnoREW(1,:),30,[0.62 0.85 0.94]); %scatter ind
scatter(minstimREW(2,:),minstimREW(1,:),30,[0 0.45 0.74]);%scatter ind
scatter(lownoREW(2,:),lownoREW(1,:),30,[0.62 0.85 0.94]);%scatter ind
scatter(lowstimREW(2,:),lowstimREW(1,:),30,[0 0.45 0.74]);%scatter ind
scatter(highnoREW(2,:),highnoREW(1,:),30,[0.62 0.85 0.94]);%scatter ind
scatter(highstimREW(2,:),highstimREW(1,:),30,[0 0.45 0.74]);%scatter ind
scatter(minnoUNREW(2,:),minnoUNREW(1,:),30,[0.8 0.8 0.8]);%scatter ind
scatter(minstimUNREW(2,:),minstimUNREW(1,:),30,[0.6 0.6 0.6]);%scatter ind
scatter(lownoUNREW(2,:),lownoUNREW(1,:),30,[0.8 0.8 0.8]);%scatter ind
scatter(lowstimUNREW(2,:),lowstimUNREW(1,:),30,[0.6 0.6 0.6]);%scatter ind
scatter(highnoUNREW(2,:),highnoUNREW(1,:),30,[0.8 0.8 0.8]);%scatter ind
scatter(highstimUNREW(2,:),highstimUNREW(1,:),30,[0.6 0.6 0.6]);%scatter ind
scatter(minnoBIAS(2,:),minnoBIAS(1,:),30,[0.99 0.76 0.76]);%scatter ind
scatter(minstimBIAS(2,:),minstimBIAS(1,:),30,[0.96 0.65 0.71]);%scatter ind
scatter(lownoBIAS(2,:),lownoBIAS(1,:),30,[0.99 0.76 0.76]);%scatter ind
scatter(lowstimBIAS(2,:),lowstimBIAS(1,:),30,[0.96 0.65 0.71]);%scatter ind
scatter(highnoBIAS(2,:),highnoBIAS(1,:),30,[0.99 0.76 0.76]);%scatter ind
scatter(highstimBIAS(2,:),highstimBIAS(1,:),30,[0.96 0.65 0.71]);%scatter ind
X(1) = ctr(1,1); X(2) = ctr(2,1); Y(1) = y(1,1); Y(2) = y(1,2); %avg line
plot(X(1:2),Y,'LineWidth',1.5,'Color','k') %plot line
X(1) = ctr(3,1); X(2) = ctr(4,1); Y(1) = y(1,3); Y(2) = y(1,4);%avg line
plot(X(1:2),Y,'LineWidth',1.5,'Color','k')%plot line
X(1) = ctr(5,1); X(2) = ctr(6,1); Y(1) = y(1,5); Y(2) = y(1,6);%avg line
plot(X(1:2),Y,'LineWidth',1.5,'Color','k')%plot line
X(1) = ctr(1,2); X(2) = ctr(2,2); Y(1) = y(2,1); Y(2) = y(2,2);%avg line
plot(X(1:2),Y,'LineWidth',1.5,'Color','k')%plot line
X(1) = ctr(3,2); X(2) = ctr(4,2); Y(1) = y(2,3); Y(2) = y(2,4);%avg line
plot(X(1:2),Y,'LineWidth',1.5,'Color','k')%plot line
X(1) = ctr(5,2); X(2) = ctr(6,2); Y(1) = y(2,5); Y(2) = y(2,6);%avg line
plot(X(1:2),Y,'LineWidth',1.5,'Color','k')%plot line
X(1) = ctr(1,3); X(2) = ctr(2,3); Y(1) = y(3,1); Y(2) = y(3,2);%avg line
plot(X(1:2),Y,'LineWidth',1.5,'Color','k')%plot line
X(1) = ctr(3,3); X(2) = ctr(4,3); Y(1) = y(3,3); Y(2) = y(3,4);%avg line
plot(X(1:2),Y,'LineWidth',1.5,'Color','k')%plot line
X(1) = ctr(5,3); X(2) = ctr(6,3); Y(1) = y(3,5); Y(2) = y(3,6);%avg line
plot(X(1:2),Y,'LineWidth',1.5,'Color','k')%plot line
e1 = errorbar(ctr(1,1), y(1,1),errorminus_vert(1,1),errorplus_vert(1,1),'LineStyle','none','LineWidth',2); %NC rew
e1.Color = '[0.3 0.75 0.93]'; %color
hLine = line([ctr(1,1) - errorminus_horz(1), ctr(1,1) + errorplus_horz(1)], [y(1,1), y(1,1)], 'Color', get(e1, 'Color'), 'LineWidth', 2); %horiz
e12 = errorbar(ctr(2,1), y(1,2),errorminus_vert(2,1),errorplus_vert(2,1),'LineStyle','none','LineWidth',2); %NS rew
e12.Color = 'b'; %color
hLine = line([ctr(2,1) - errorminus_horz(1), ctr(2,1) + errorplus_horz(1)], [y(1,2), y(1,2)], 'Color', get(e12, 'Color'), 'LineWidth', 2);%horiz
e13 = errorbar(ctr(3,1), y(1,3),errorminus_vert(3,1),errorplus_vert(3,1),'LineStyle','none','LineWidth',2); %NC unrew
e13.Color = '[0.7 0.7 0.7]'; %color
hLine = line([ctr(3,1) - errorminus_horz(1), ctr(3,1) + errorplus_horz(1)], [y(1,3), y(1,3)], 'Color', get(e13, 'Color'), 'LineWidth', 2);%horiz
e14 = errorbar(ctr(4,1), y(1,4),errorminus_vert(4,1),errorplus_vert(4,1),'LineStyle','none','LineWidth',2); %NS unrew
e14.Color = '[0.5 0.5 0.5]';%color
hLine = line([ctr(4,1) - errorminus_horz(1), ctr(4,1) + errorplus_horz(1)], [y(1,4), y(1,4)], 'Color', get(e14, 'Color'), 'LineWidth', 2);%horiz
e15 = errorbar(ctr(5,1), y(1,5),errorminus_vert(5,1),errorplus_vert(5,1),'LineStyle','none','LineWidth',2); %NC bias
e15.Color = '[0.99 0.58 0.58]'; %color
hLine = line([ctr(5,1) - errorminus_horz(1), ctr(5,1) + errorplus_horz(1)], [y(1,5), y(1,5)], 'Color', get(e15, 'Color'), 'LineWidth', 2);%horiz
e16 = errorbar(ctr(6,1), y(1,6),errorminus_vert(6,1),errorplus_vert(6,1),'LineStyle','none','LineWidth',2); %NS bias
e16.Color = 'r';%color
hLine = line([ctr(6,1) - errorminus_horz(1), ctr(6,1) + errorplus_horz(1)], [y(1,6), y(1,6)], 'Color', get(e16, 'Color'), 'LineWidth', 2);%horiz
e2 = errorbar(ctr(1,2), y(2,1),errorminus_vert(1,2),errorplus_vert(1,2),'LineStyle','none','LineWidth',2); %LC rew
e2.Color = '[0.3 0.75 0.93]'; %color
hLine = line([ctr(1,2) - errorminus_horz(1), ctr(1,2) + errorplus_horz(1)], [y(2,1), y(2,1)], 'Color', get(e2, 'Color'), 'LineWidth', 2);%horiz
e22 = errorbar(ctr(2,2), y(2,2),errorminus_vert(2,2),errorplus_vert(2,2),'LineStyle','none','LineWidth',2); %LS rew
e22.Color = 'b'; %color
hLine = line([ctr(2,2) - errorminus_horz(1), ctr(2,2) + errorplus_horz(1)], [y(2,2), y(2,2)], 'Color', get(e22, 'Color'), 'LineWidth', 2);%horiz
e23 = errorbar(ctr(3,2), y(2,3),errorminus_vert(3,2),errorplus_vert(3,2),'LineStyle','none','LineWidth',2); %LC unrew
e23.Color = '[0.7 0.7 0.7]'; %color
hLine = line([ctr(3,2) - errorminus_horz(1), ctr(3,2) + errorplus_horz(1)], [y(2,3), y(2,3)], 'Color', get(e23, 'Color'), 'LineWidth', 2);%horiz
e24 = errorbar(ctr(4,2), y(2,4),errorminus_vert(4,2),errorplus_vert(4,2),'LineStyle','none','LineWidth',2); %LS unrew
e24.Color = '[0.5 0.5 0.5]';%color
hLine = line([ctr(4,2) - errorminus_horz(1), ctr(4,2) + errorplus_horz(1)], [y(2,4), y(2,4)], 'Color', get(e24, 'Color'), 'LineWidth', 2);%horiz
e25 = errorbar(ctr(5,2), y(2,5),errorminus_vert(5,2),errorplus_vert(5,2),'LineStyle','none','LineWidth',2); %LC bias
e25.Color = '[0.99 0.58 0.58]'; %color
hLine = line([ctr(5,2) - errorminus_horz(1), ctr(5,2) + errorplus_horz(1)], [y(2,5), y(2,5)], 'Color', get(e25, 'Color'), 'LineWidth', 2);%horiz
e26 = errorbar(ctr(6,2), y(2,6),errorminus_vert(6,2),errorplus_vert(6,2),'LineStyle','none','LineWidth',2); %LS bias
e26.Color = 'r';%color
hLine = line([ctr(6,2) - errorminus_horz(1), ctr(6,2) + errorplus_horz(1)], [y(2,6), y(2,6)], 'Color', get(e26, 'Color'), 'LineWidth', 2);%horiz
e3 = errorbar(ctr(1,3), y(3,1),errorminus_vert(1,3),errorplus_vert(1,3),'LineStyle','none','LineWidth',2); %HC rew
e3.Color = '[0.3 0.75 0.93]'; %color
hLine = line([ctr(1,3) - errorminus_horz(1), ctr(1,3) + errorplus_horz(1)], [y(3,1), y(3,1)], 'Color', get(e3, 'Color'), 'LineWidth', 2);%horiz
e32 = errorbar(ctr(2,3), y(3,2),errorminus_vert(2,3),errorplus_vert(2,3),'LineStyle','none','LineWidth',2); %HS rew
e32.Color = 'b'; %color
hLine = line([ctr(2,3) - errorminus_horz(1), ctr(2,3) + errorplus_horz(1)], [y(3,2), y(3,2)], 'Color', get(e32, 'Color'), 'LineWidth', 2);%horiz
e33 = errorbar(ctr(3,3), y(3,3),errorminus_vert(3,3),errorplus_vert(3,3),'LineStyle','none','LineWidth',2); %HC unrew
e33.Color = '[0.7 0.7 0.7]'; %color
hLine = line([ctr(3,3) - errorminus_horz(1), ctr(3,3) + errorplus_horz(1)], [y(3,3), y(3,3)], 'Color', get(e33, 'Color'), 'LineWidth', 2);%horiz
e34 = errorbar(ctr(4,3), y(3,4),errorminus_vert(4,3),errorplus_vert(4,3),'LineStyle','none','LineWidth',2); %HS unrew
e34.Color = '[0.5 0.5 0.5]';%color
hLine = line([ctr(4,3) - errorminus_horz(1), ctr(4,3) + errorplus_horz(1)], [y(3,4), y(3,4)], 'Color', get(e34, 'Color'), 'LineWidth', 2);%horiz
e35 = errorbar(ctr(5,3), y(3,5),errorminus_vert(5,3),errorplus_vert(5,3),'LineStyle','none','LineWidth',2); %HC bias
e35.Color = '[0.99 0.58 0.58]'; %color
hLine = line([ctr(5,3) - errorminus_horz(1), ctr(5,3) + errorplus_horz(1)], [y(3,5), y(3,5)], 'Color', get(e35, 'Color'), 'LineWidth', 2);%horiz
e36 = errorbar(ctr(6,3), y(3,6),errorminus_vert(6,3),errorplus_vert(6,3),'LineStyle','none','LineWidth',2); %HS bias
e36.Color = 'r';%color
hLine = line([ctr(6,3) - errorminus_horz(1), ctr(6,3) + errorplus_horz(1)], [y(3,6), y(3,6)], 'Color', get(e36, 'Color'), 'LineWidth', 2);%horiz
ylim([-0.6 2]) %ylim
XTickLabel = ({'No','Low','High'}); %x label
set(gca,'xtick',[1 1.1 1.2],'xticklabel',XTickLabel,'box','off','FontSize',20) %aesthetics
ylabel('Regression Weight') %y label
legend({'Ctrl Rew','Stim Rew','Ctrl Unrew','Stim Unrew','Ctrl Bias','Stim Bias'},'AutoUpdate','off') %legend
X(1) = ctr(1,1); X(2) = ctr(2,1); Y(1) = 1.74; Y(2) = 1.74; %sig bar
plot(X(1:2),Y,'LineWidth',1.5,'Color','k') %plot line
X(1) = ctr(5,1); X(2) = ctr(6,1); Y(1) = 0.38; Y(2) = 0.38; %sig bar
plot(X(1:2),Y,'LineWidth',1.5,'Color','k')%plot line
X(1) = ctr(1,2); X(2) = ctr(2,2); Y(1) = 1.84; Y(2) = 1.84; %sig bar
plot(X(1:2),Y,'LineWidth',1.5,'Color','k')%plot line
X(1) = ctr(1,3); X(2) = ctr(2,3); Y(1) = 1.7; Y(2) = 1.7; %sig bar
plot(X(1:2),Y,'LineWidth',1.5,'Color','k')%plot line

%% Supplemental: iSPN - WSLS

f = figure; %fig
ctr = []; ydt = []; %space
y = [mean(ws_ctrl) mean(ws_stim); mean(ls_ctrl) mean(ls_stim)]; %WSLS
a_vert = [std(ws_ctrl)/sqrt(length(ws_ctrl)) std(ws_stim)/sqrt(length(ws_stim)); std(ls_ctrl)/sqrt(length(ls_ctrl)) std(ls_stim)/sqrt(length(ls_stim))]; %SEM
errorplus_vert=a_vert'; %positive SEM
errorminus_vert=errorplus_vert; %negative SEM
x = [1; 1.1]; %x position
a_horz = [0.01;0.01]; %horizontal lines
errorplus_horz=a_horz'; %positive horiz
errorminus_horz=errorplus_horz; %negative horiz
hBar = bar(x,y,'FaceColor',[1 1 1],'EdgeColor',[1 1 1]); %plot WSLS
for k1 = 1:size(y,2) %"for" loop going through y
    ctr(k1,:) = bsxfun(@plus, hBar(k1).XData, hBar(k1).XOffset');     %ctr
    ydt(k1,:) = hBar(k1).YData;                     %ydt
end %ending "for" loop going through y
minnoc = [0.74 0.73 0.67 0.83 0.92 0.88 0.84 0.56 0.89 0.73]; %WC ind
minstim = [0.52 0.64 0.62 0.56 0.67 0.87 0.81 0.58 0.66 0.5]; %WS ind
lownoc = [0.58 0.5 0.42 0.57 0.34 0.43 0.58 0.55 0.55 0.58]; %LC ind
lowstim = [0.59 0.47 0.46 0.46 0.6 0.58 0.64 0.58 0.62 0.38]; %LS ind
minnoc(2,:) = ctr(1,1); %center
minstim(2,:) = ctr(2,1); %center
lownoc(2,:) = ctr(1,2); %center
lowstim(2,:) = ctr(2,2); %center
hold on %hold fig
for ugh = 1:10 %"for" loop going through ind
    X(1) =  minnoc(2,ugh); %control X
    Y(1) =  minnoc(1,ugh); %control Y
    X(2) =  minstim(2,ugh); %stim X
    Y(2) =  minstim(1,ugh); %stim Y
    plot(X(1:2),Y,'Color','[0.7 0.7 0.7]') %plot individual lines
end %ending "for" loop going through ind
for ugh = 1:10%"for" loop going through ind
    X(1) =  lownoc(2,ugh); %control X
    Y(1) =  lownoc(1,ugh);%control Y
    X(2) =  lowstim(2,ugh); %stim X
    Y(2) =  lowstim(1,ugh); %stim Y
    plot(X(1:2),Y,'Color','[0.7 0.7 0.7]')%plot individual lines
end%ending "for" loop going through ind
X(1) = ctr(1,1); X(2) = ctr(2,1); Y(1) = y(1,1); Y(2) = y(1,2); %avg line
plot(X(1:2),Y,'LineWidth',1.5,'Color','k') %plot line
X(1) = ctr(1,2); X(2) = ctr(2,2); Y(1) = y(2,1); Y(2) = y(2,2);%avg line
plot(X(1:2),Y,'LineWidth',1.5,'Color','k')%plot line
e1 = errorbar(ctr(1,1), y(1,1),errorminus_vert(1,1),errorplus_vert(1,1),'LineStyle','none','LineWidth',2); %ctrl rew
e1.Color = '[0.99 0.58 0.58]'; %color
hLine = line([ctr(1,1) - errorminus_horz(1), ctr(1,1) + errorplus_horz(1)], [y(1,1), y(1,1)], 'Color', get(e1, 'Color'), 'LineWidth', 2); %horiz
e12 = errorbar(ctr(2,1), y(1,2),errorminus_vert(2,1),errorplus_vert(2,1),'LineStyle','none','LineWidth',2); %stim rew
e12.Color = 'r'; %color
hLine = line([ctr(2,1) - errorminus_horz(1), ctr(2,1) + errorplus_horz(1)], [y(1,2), y(1,2)], 'Color', get(e12, 'Color'), 'LineWidth', 2);%horiz
e2 = errorbar(ctr(1,2), y(2,1),errorminus_vert(1,2),errorplus_vert(1,2),'LineStyle','none','LineWidth',2); %ctrl unrew
e2.Color = '[0.99 0.58 0.58]'; %color
hLine = line([ctr(1,2) - errorminus_horz(1), ctr(1,2) + errorplus_horz(1)], [y(2,1), y(2,1)], 'Color', get(e2, 'Color'), 'LineWidth', 2);%horiz
e22 = errorbar(ctr(2,2), y(2,2),errorminus_vert(2,2),errorplus_vert(2,2),'LineStyle','none','LineWidth',2); %stim unrew
e22.Color = 'r'; %color
hLine = line([ctr(2,2) - errorminus_horz(1), ctr(2,2) + errorplus_horz(1)], [y(2,2), y(2,2)], 'Color', get(e22, 'Color'), 'LineWidth', 2);%horiz
ylim([0.2 1]) %ylim
hline(0.5,'--k') %50% line
XTickLabel = ({'Rew','Unrew'}); %x label
set(gca,'xtick',[1 1.1 1.2],'xticklabel',XTickLabel,'box','off','FontSize',20) %aesthetics
ylabel('P(Repeat)') %y label
legend({'Ctrl','Stim'},'AutoUpdate','off') %legend
X(1) = ctr(1,1); X(2) = ctr(2,1); Y(1) = 0.95; Y(2) = 0.95; %sig bar
plot(X(1:2),Y,'LineWidth',1.5,'Color','k') %plot line

%% Supplemental: iSPN - time in center proportion

f = figure; %fig 
ctr = []; ydt = []; %space
y = [mean(ispn_noconflowvolctrl_ratio) mean(ispn_noconflowvolstim_ratio); mean(ispn_lowconflowvolctrl_ratio) mean(ispn_lowconflowvolstim_ratio); mean(ispn_highconflowvolctrl_ratio) mean(ispn_highconflowvolstim_ratio)]; %time in center vs total trial time ratio
a_vert = [std(ispn_noconflowvolctrl_ratio)/sqrt(length(ispn_noconflowvolctrl_ratio)) std(ispn_noconflowvolstim_ratio)/sqrt(length(ispn_noconflowvolstim_ratio)); std(ispn_lowconflowvolctrl_ratio)/sqrt(length(ispn_lowconflowvolctrl_ratio)) std(ispn_lowconflowvolstim_ratio)/sqrt(length(ispn_lowconflowvolstim_ratio)); std(ispn_highconflowvolctrl_ratio)/sqrt(length(ispn_highconflowvolctrl_ratio)) std(ispn_highconflowvolstim_ratio)/sqrt(length(ispn_highconflowvolstim_ratio))]; %SEM
errorplus_vert=a_vert'; %positive SEM
errorminus_vert=errorplus_vert; %negative SEM
x = [1; 1.1; 1.2]; %x position
a_horz = [0.01;0.01;0.01]; %horizontal lines
errorplus_horz=a_horz'; %positive horiz
errorminus_horz=errorplus_horz; %negative horiz
hBar = bar(x,y,'FaceColor',[1 1 1],'EdgeColor',[1 1 1]); %plot ratios
for k1 = 1:size(y,2) %"for" loop going through y
    ctr(k1,:) = bsxfun(@plus, hBar(k1).XData, hBar(k1).XOffset');     %ctr
    ydt(k1,:) = hBar(k1).YData;                    %ydt
end %ending "for" loop going through y
minnoc = [0.15 0.17 0.2 0.2 0.28 0.18 0.21 0.2 0.21 0.28]; %NC ind
minstim = [0.28 0.24 0.2 0.22 0.34 0.32 0.22 0.22 0.22 0.34]; %NS ind
lownoc = [0.13 0.16 0.19 0.14 0.23 0.21 0.2 0.2 0.22 0.2]; %LC ind
lowstim = [0.25 0.19 0.24 0.21 0.23 0.29 0.21 0.23 0.25 0.24]; %LS ind
highnoc = [0.16 0.2 0.18 0.2 0.28 0.2 0.2 0.22 0.24 0.2]; %HC ind
highstim = [0.21 0.23 0.24 0.26 0.34 0.25 0.22 0.28 0.25 0.29]; %HS ind
minnoc(2,:) = ctr(1,1); %center
minstim(2,:) = ctr(2,1);%center
lownoc(2,:) = ctr(1,2);%center
lowstim(2,:) = ctr(2,2);%center
highnoc(2,:) = ctr(1,3);%center
highstim(2,:) = ctr(2,3);%center
hold on %hold fig
for ugh = 1:10 %"for" loop going through ind
    X(1) =  minnoc(2,ugh); %control X
    Y(1) =  minnoc(1,ugh); %control Y
    X(2) =  minstim(2,ugh); %stim X
    Y(2) =  minstim(1,ugh); %stim Y
    plot(X(1:2),Y,'Color','[0.7 0.7 0.7]') %plot individual lines
end %ending "for" loop going through ind
for ugh = 1:10%"for" loop going through ind
    X(1) =  lownoc(2,ugh);%control X
    Y(1) =  lownoc(1,ugh);%control Y
    X(2) =  lowstim(2,ugh); %stim X
    Y(2) =  lowstim(1,ugh);%stim Y
    plot(X(1:2),Y,'Color','[0.7 0.7 0.7]')%plot individual lines
end%ending "for" loop going through ind
for ugh = 1:10%"for" loop going through ind
    X(1) =  highnoc(2,ugh);%control X
    Y(1) =  highnoc(1,ugh);%control Y
    X(2) =  highstim(2,ugh); %stim X
    Y(2) =  highstim(1,ugh);%stim Y
    plot(X(1:2),Y,'Color','[0.7 0.7 0.7]')%plot individual lines
end%ending "for" loop going through ind
X(1) = ctr(1,1); X(2) = ctr(2,1); Y(1) = y(1,1); Y(2) = y(1,2); %avg line
plot(X(1:2),Y,'LineWidth',1.5,'Color','k') %plot line
X(1) = ctr(1,2); X(2) = ctr(2,2); Y(1) = y(2,1); Y(2) = y(2,2);%avg line
plot(X(1:2),Y,'LineWidth',1.5,'Color','k') %plot line
X(1) = ctr(1,3); X(2) = ctr(2,3); Y(1) = y(3,1); Y(2) = y(3,2);%avg line
plot(X(1:2),Y,'LineWidth',1.5,'Color','k') %plot line
e1 = errorbar(ctr(1,1), y(1,1),errorminus_vert(1,1),errorplus_vert(1,1),'LineStyle','none','LineWidth',2); %NC ratio
e1.Color = '[0.99 0.58 0.58]'; %color
hLine = line([ctr(1,1) - errorminus_horz(1), ctr(1,1) + errorplus_horz(1)], [y(1,1), y(1,1)], 'Color', get(e1, 'Color'), 'LineWidth', 2); %horiz
e12 = errorbar(ctr(2,1), y(1,2),errorminus_vert(2,1),errorplus_vert(2,1),'LineStyle','none','LineWidth',2); %NS ratio
e12.Color = 'r'; %color
hLine = line([ctr(2,1) - errorminus_horz(1), ctr(2,1) + errorplus_horz(1)], [y(1,2), y(1,2)], 'Color', get(e12, 'Color'), 'LineWidth', 2); %horiz
e2 = errorbar(ctr(1,2), y(2,1),errorminus_vert(1,2),errorplus_vert(1,2),'LineStyle','none','LineWidth',2); %LC ratio
e2.Color = '[0.99 0.58 0.58]'; %color
hLine = line([ctr(1,2) - errorminus_horz(1), ctr(1,2) + errorplus_horz(1)], [y(2,1), y(2,1)], 'Color', get(e2, 'Color'), 'LineWidth', 2); %horiz
e22 = errorbar(ctr(2,2), y(2,2),errorminus_vert(2,2),errorplus_vert(2,2),'LineStyle','none','LineWidth',2); %LS ratio
e22.Color = 'r'; %color
hLine = line([ctr(2,2) - errorminus_horz(1), ctr(2,2) + errorplus_horz(1)], [y(2,2), y(2,2)], 'Color', get(e22, 'Color'), 'LineWidth', 2); %horiz
e3 = errorbar(ctr(1,3), y(3,1),errorminus_vert(1,3),errorplus_vert(1,3),'LineStyle','none','LineWidth',2); %HC ratio
e3.Color = '[0.99 0.58 0.58]'; %color
hLine = line([ctr(1,3) - errorminus_horz(1), ctr(1,3) + errorplus_horz(1)], [y(3,1), y(3,1)], 'Color', get(e3, 'Color'), 'LineWidth', 2); %horiz
e32 = errorbar(ctr(2,3), y(3,2),errorminus_vert(2,3),errorplus_vert(2,3),'LineStyle','none','LineWidth',2); %HS ratio
e32.Color = 'r'; %color
hLine = line([ctr(2,3) - errorminus_horz(1), ctr(2,3) + errorplus_horz(1)], [y(3,2), y(3,2)], 'Color', get(e32, 'Color'), 'LineWidth', 2); %horiz
%ylim([0.1 0.4]) %original ylim
ylim([0 0.4]) %ylim axis to 0
XTickLabel = ({'No','Low','High'}); %x label
set(gca,'xtick',[1 1.1 1.2],'xticklabel',XTickLabel,'box','off','FontSize',20) %aesthetics
ylabel('Time in Center / Trial Time') %y label
legend({'Ctrl','Stim'},'AutoUpdate','off') %legend
X(1) = ctr(1,1); X(2) = ctr(2,1); Y(1) = 0.35; Y(2) = 0.35; %sig bar
plot(X(1:2),Y,'LineWidth',1.5,'Color','k') %plot line
X(1) = ctr(1,2); X(2) = ctr(2,2); Y(1) = 0.3; Y(2) = 0.3; %sig bar
plot(X(1:2),Y,'LineWidth',1.5,'Color','k') %plot line
X(1) = ctr(1,3); X(2) = ctr(2,3); Y(1) = 0.35; Y(2) = 0.35; %sig bar
plot(X(1:2),Y,'LineWidth',1.5,'Color','k') %plot line

%% Supplemental: dSPN - outcome/choice regression

f = figure; %fig
ctr = []; ydt = []; %space
y = [mean(d1min_no_r1) mean(d1min_stim_r1) mean(d1min_no_c1) mean(d1min_stim_c1); mean(d1low_no_r1) mean(d1low_stim_r1) mean(d1low_no_c1) mean(d1low_stim_c1); mean(d1high_no_r1) mean(d1high_stim_r1) mean(d1high_no_c1) mean(d1high_stim_c1)]; %regression weights
a_vert = [std(d1min_no_r1)/sqrt(length(d1min_no_r1)) std(d1min_stim_r1)/sqrt(length(d1min_stim_r1)) std(d1min_no_c1)/sqrt(length(d1min_no_c1)) std(d1min_stim_c1)/sqrt(length(d1min_stim_c1)); std(d1low_no_r1)/sqrt(length(d1low_no_r1)) std(d1low_stim_r1)/sqrt(length(d1low_stim_r1)) std(d1low_no_c1)/sqrt(length(d1low_no_c1)) std(d1low_stim_c1)/sqrt(length(d1low_stim_c1)); std(d1high_no_r1)/sqrt(length(d1high_no_r1)) std(d1high_stim_r1)/sqrt(length(d1high_stim_r1)) std(d1high_no_c1)/sqrt(length(d1high_no_c1)) std(d1high_stim_c1)/sqrt(length(d1high_stim_c1))]; %SEM
errorplus_vert=a_vert'; %positive SEM
errorminus_vert=errorplus_vert; %negative SEM
x = [1; 1.1; 1.2]; %x position
a_horz = [0.005;0.005;0.005]; %horizontal line
errorplus_horz=a_horz'; %positive horiz
errorminus_horz=errorplus_horz; %negative horiz
hBar = bar(x,y,'FaceColor',[1 1 1],'EdgeColor',[1 1 1]); %plot regression
for k1 = 1:size(y,2) %"for" loop going through y
    ctr(k1,:) = bsxfun(@plus, hBar(k1).XData, hBar(k1).XOffset');     %ctr
    ydt(k1,:) = hBar(k1).YData;                    %ydt
end %ending "for" loop going through y
h = hline(0,'w'); %block 0
h.LineWidth = 2; %line width
minnoR = [1.5 0.84 0.7 1.05 0.66 1.32 1.58 0.89 1.22 1.25]; %NC outcome
 minhighR = [1.32 1.02 0.41 0.93 0.71 1.51 0.99 0.63 1.37 1.13]; %NS outcome
 lownoR = [1.2 0.84 0.78 1.22 0.81 1.36 1.46 0.79 1.09 1.18]; %LC outcome
 lowhighR = [0.86 0.98 0.68 1.11 0.76 1.14 1.19 0.85 1.36 1.14]; %LS outcome
 highnoR = [1.61 0.91 0.67 1.09 0.78 1.17 1.03 0.87 1.06 1.08]; %HC outcome
 highhighR = [1.48 1.15 0.37 0.94 0.71 1.13 1.09 0.58 1.17 0.99]; %HS outcome
 minnoC = [0.49 0.42 0.7 0.5 0.36 0.79 0.46 1.1 0.56 0.73]; %NC choice
 minhighC = [0.86 0.95 0.89 0.48 0.35 1 0.69 0.74 0.28 0.68]; %NS choice
 lownoC = [0.6 0.3 0.7 0.26 0.5 0.79 0.64 0.75 0.12 0.58]; %LC choice
 lowhighC = [0.56 0.67 1.03 0.19 0.32 0.94 0.88 0.5 0.32 0.66]; %LS choice
 highnoC = [0.58 0.36 0.93 0.47 0.58 0.55 0.72 0.79 0.48 0.69]; %HC choice
 highhighC = [0.5 0.99 1.06 0.01 0.45 0.67 0.78 0.8 0.28 0.63]; %HS choice
 minnoR(2,:) = ctr(1,1); %center
 minhighR(2,:) = ctr(2,1);%center
 minnoC(2,:) = ctr(3,1);%center
 minhighC(2,:) = ctr(4,1);%center
 lownoR(2,:) = ctr(1,2);%center
 lowhighR(2,:) = ctr(2,2);%center
 lownoC(2,:) = ctr(3,2);%center
 lowhighC(2,:) = ctr(4,2);%center
 highnoR(2,:) = ctr(1,3);%center
 highhighR(2,:) = ctr(2,3);%center
 highnoC(2,:) = ctr(3,3);%center
 highhighC(2,:) = ctr(4,3);%center
hold on %hold fig
scatter(minnoR(2,:),minnoR(1,:),30,[0.61 0.87 0.97]); %scatter ind
scatter(minhighR(2,:),minhighR(1,:),30,[0.62 0.77 1]); %scatter ind
scatter(lownoR(2,:),lownoR(1,:),30,[0.61 0.87 0.97]); %scatter ind
scatter(lowhighR(2,:),lowhighR(1,:),30,[0.62 0.77 1]); %scatter ind
scatter(highnoR(2,:),highnoR(1,:),30,[0.61 0.87 0.97]); %scatter ind
scatter(highhighR(2,:),highhighR(1,:),30,[0.62 0.77 1]); %scatter ind
scatter(minnoC(2,:),minnoC(1,:),30,[0.88 0.71 0.98]); %scatter ind
scatter(minhighC(2,:),minhighC(1,:),30,[0.91 0.51 1]); %scatter ind
scatter(lownoC(2,:),lownoC(1,:),30,[0.88 0.71 0.98]); %scatter ind
scatter(lowhighC(2,:),lowhighC(1,:),30,[0.91 0.51 1]); %scatter ind
scatter(highnoC(2,:),highnoC(1,:),30,[0.88 0.71 0.98]); %scatter ind
scatter(highhighC(2,:),highhighC(1,:),30,[0.91 0.51 1]); %scatter ind
X(1) = ctr(1,1); X(2) = ctr(2,1); Y(1) = y(1,1); Y(2) = y(1,2); %avg lines
plot(X(1:2),Y,'LineWidth',1.5,'Color','k') %plot line
X(1) = ctr(3,1); X(2) = ctr(4,1); Y(1) = y(1,3); Y(2) = y(1,4); %avg lines
plot(X(1:2),Y,'LineWidth',1.5,'Color','k') %plot line
X(1) = ctr(1,2); X(2) = ctr(2,2); Y(1) = y(2,1); Y(2) = y(2,2); %avg lines
plot(X(1:2),Y,'LineWidth',1.5,'Color','k') %plot line
X(1) = ctr(3,2); X(2) = ctr(4,2); Y(1) = y(2,3); Y(2) = y(2,4); %avg lines
plot(X(1:2),Y,'LineWidth',1.5,'Color','k') %plot line
X(1) = ctr(1,3); X(2) = ctr(2,3); Y(1) = y(3,1); Y(2) = y(3,2); %avg lines
plot(X(1:2),Y,'LineWidth',1.5,'Color','k') %plot line
X(1) = ctr(3,3); X(2) = ctr(4,3); Y(1) = y(3,3); Y(2) = y(3,4); %avg lines
plot(X(1:2),Y,'LineWidth',1.5,'Color','k') %plot line
e1 = errorbar(ctr(1,1), y(1,1),errorminus_vert(1,1),errorplus_vert(1,1),'LineStyle','none','LineWidth',2); %NC outcome
e1.Color = '[0.3 0.75 0.93]'; %color
hLine = line([ctr(1,1) - errorminus_horz(1), ctr(1,1) + errorplus_horz(1)], [y(1,1), y(1,1)], 'Color', get(e1, 'Color'), 'LineWidth', 2); %horiz
e12 = errorbar(ctr(2,1), y(1,2),errorminus_vert(2,1),errorplus_vert(2,1),'LineStyle','none','LineWidth',2); %NS outcome
e12.Color = 'b';  %color
hLine = line([ctr(2,1) - errorminus_horz(1), ctr(2,1) + errorplus_horz(1)], [y(1,2), y(1,2)], 'Color', get(e12, 'Color'), 'LineWidth', 2); %horiz
e13 = errorbar(ctr(3,1), y(1,3),errorminus_vert(3,1),errorplus_vert(3,1),'LineStyle','none','LineWidth',2); %NC choice
e13.Color = '[0.76 0.62 0.79]';  %color
hLine = line([ctr(3,1) - errorminus_horz(1), ctr(3,1) + errorplus_horz(1)], [y(1,3), y(1,3)], 'Color', get(e13, 'Color'), 'LineWidth', 2); %horiz
e14 = errorbar(ctr(4,1), y(1,4),errorminus_vert(4,1),errorplus_vert(4,1),'LineStyle','none','LineWidth',2); %NS choice
e14.Color = '[0.49 0.18 0.56]'; %color
hLine = line([ctr(4,1) - errorminus_horz(1), ctr(4,1) + errorplus_horz(1)], [y(1,4), y(1,4)], 'Color', get(e14, 'Color'), 'LineWidth', 2); %horiz
e2 = errorbar(ctr(1,2), y(2,1),errorminus_vert(1,2),errorplus_vert(1,2),'LineStyle','none','LineWidth',2); %LC outcome
e2.Color = '[0.3 0.75 0.93]';  %color
hLine = line([ctr(1,2) - errorminus_horz(1), ctr(1,2) + errorplus_horz(1)], [y(2,1), y(2,1)], 'Color', get(e2, 'Color'), 'LineWidth', 2); %horiz
e22 = errorbar(ctr(2,2), y(2,2),errorminus_vert(2,2),errorplus_vert(2,2),'LineStyle','none','LineWidth',2); %LS outcome
e22.Color = 'b';  %color
hLine = line([ctr(2,2) - errorminus_horz(1), ctr(2,2) + errorplus_horz(1)], [y(2,2), y(2,2)], 'Color', get(e22, 'Color'), 'LineWidth', 2); %horiz
e23 = errorbar(ctr(3,2), y(2,3),errorminus_vert(3,2),errorplus_vert(3,2),'LineStyle','none','LineWidth',2); %LC choice
e23.Color = '[0.76 0.62 0.79]';  %color
hLine = line([ctr(3,2) - errorminus_horz(1), ctr(3,2) + errorplus_horz(1)], [y(2,3), y(2,3)], 'Color', get(e23, 'Color'), 'LineWidth', 2); %horiz
e24 = errorbar(ctr(4,2), y(2,4),errorminus_vert(4,2),errorplus_vert(4,2),'LineStyle','none','LineWidth',2); %LS choice
e24.Color = '[0.49 0.18 0.56]'; %color
hLine = line([ctr(4,2) - errorminus_horz(1), ctr(4,2) + errorplus_horz(1)], [y(2,4), y(2,4)], 'Color', get(e24, 'Color'), 'LineWidth', 2); %horiz
e3 = errorbar(ctr(1,3), y(3,1),errorminus_vert(1,3),errorplus_vert(1,3),'LineStyle','none','LineWidth',2); %HC outcome
e3.Color = '[0.3 0.75 0.93]';  %color
hLine = line([ctr(1,3) - errorminus_horz(1), ctr(1,3) + errorplus_horz(1)], [y(3,1), y(3,1)], 'Color', get(e3, 'Color'), 'LineWidth', 2); %horiz
e32 = errorbar(ctr(2,3), y(3,2),errorminus_vert(2,3),errorplus_vert(2,3),'LineStyle','none','LineWidth',2); %HS outcome
e32.Color = 'b';  %color
hLine = line([ctr(2,3) - errorminus_horz(1), ctr(2,3) + errorplus_horz(1)], [y(3,2), y(3,2)], 'Color', get(e32, 'Color'), 'LineWidth', 2); %horiz
e33 = errorbar(ctr(3,3), y(3,3),errorminus_vert(3,3),errorplus_vert(3,3),'LineStyle','none','LineWidth',2); %HC choice
e33.Color = '[0.76 0.62 0.79]';  %color
hLine = line([ctr(3,3) - errorminus_horz(1), ctr(3,3) + errorplus_horz(1)], [y(3,3), y(3,3)], 'Color', get(e33, 'Color'), 'LineWidth', 2); %horiz
e34 = errorbar(ctr(4,3), y(3,4),errorminus_vert(4,3),errorplus_vert(4,3),'LineStyle','none','LineWidth',2); %HS outcome
e34.Color = '[0.49 0.18 0.56]'; %color
hLine = line([ctr(4,3) - errorminus_horz(1), ctr(4,3) + errorplus_horz(1)], [y(3,4), y(3,4)], 'Color', get(e34, 'Color'), 'LineWidth', 2); %horiz
ylim([-0.2 1.8]) %ylim
XTickLabel = ({'No','Low','High'}); %x label
set(gca,'xtick',[1 1.1 1.2],'xticklabel',XTickLabel,'box','off','FontSize',20) %aesthetics
ylabel('Regression Weight') %y label
legend({'Ctrl Outcome','Stim Outcome','Ctrl Choice','Stim Choice'},'AutoUpdate','off') %legend

%% Supplemental: dSPN - Learning Rate

f = figure; %fig
ctr = []; ydt = []; %space
y = [mean(dspn_noconflowvolctrl_learning_rate) mean(dspn_noconflowvolstim_learning_rate); mean(dspn_lowconflowvolctrl_learning_rate) mean(dspn_lowconflowvolstim_learning_rate); mean(dspn_highconflowvolctrl_learning_rate) mean(dspn_highconflowvolstim_learning_rate)]; %LR
a_vert = [std(dspn_noconflowvolctrl_learning_rate)/sqrt(length(dspn_noconflowvolctrl_learning_rate)) std(dspn_noconflowvolstim_learning_rate)/sqrt(length(dspn_noconflowvolstim_learning_rate)); std(dspn_lowconflowvolctrl_learning_rate)/sqrt(length(dspn_lowconflowvolctrl_learning_rate)) std(dspn_lowconflowvolstim_learning_rate)/sqrt(length(dspn_lowconflowvolstim_learning_rate)); std(dspn_highconflowvolctrl_learning_rate)/sqrt(length(dspn_highconflowvolctrl_learning_rate)) std(dspn_highconflowvolstim_learning_rate)/sqrt(length(dspn_highconflowvolstim_learning_rate))]; %SEM
errorplus_vert=a_vert'; %positive SEM
errorminus_vert=errorplus_vert; %negative SEM
x = [1; 1.1; 1.2]; %x position
a_horz = [0.01;0.01;0.01]; %horizontal lines
errorplus_horz=a_horz'; %positive horiz
errorminus_horz=errorplus_horz; %negative horiz
hBar = bar(x,y,'FaceColor',[1 1 1],'EdgeColor',[1 1 1]); %plot LRs
for k1 = 1:size(y,2) %"for" loop going through y
    ctr(k1,:) = bsxfun(@plus, hBar(k1).XData, hBar(k1).XOffset');     %ctr
    ydt(k1,:) = hBar(k1).YData;                    %ydt
end %ending "for" loop going through y
minnoc = [1,1,0.82,0.43,1,0.96,1,1,1,0.99]; %NC
minnohighvol = [1,1,1,1,1,1,0.73,0.82,1,1]; %NS
lownoc = [1,0.92,0.97,0.84,1,1,0.5,0.67,1,0.63]; %LC
lownohighvol = [0.84,1,1,1,0.81,0.97,1,0.86,1,1]; %LS
highnoc = [1,1,0.67,0.93,1,1,0.91,0.94,0.4,0.85]; %HC
highnohighvol = [0.96,0.89,0.8,1,0.86,0.86,0.83,0.37,1,1];%HS
minnoc(2,:) = ctr(1,1); %center
minnohighvol(2,:) = ctr(2,1); %center
lownoc(2,:) = ctr(1,2); %center
lownohighvol(2,:) = ctr(2,2); %center
highnoc(2,:) = ctr(1,3); %center
highnohighvol(2,:) = ctr(2,3); %center
hold on %hold fig
for ugh = 1:10 %"for" loop going through ind
    X(1) =  minnoc(2,ugh); %control X
    Y(1) =  minnoc(1,ugh); %control Y
    X(2) =  minnohighvol(2,ugh); %stim X
    Y(2) =  minnohighvol(1,ugh); %stim Y
    plot(X(1:2),Y,'Color','[0.7 0.7 0.7]') %plot ind lines
end %ending "for" loop going through ind
for ugh = 1:10 %"for" loop going through ind
    X(1) =  lownoc(2,ugh);%control X
    Y(1) =  lownoc(1,ugh);%control Y
    X(2) =  lownohighvol(2,ugh);%stim X
    Y(2) =  lownohighvol(1,ugh); %stim Y
    plot(X(1:2),Y,'Color','[0.7 0.7 0.7]')%plot ind lines
end%ending "for" loop going through ind
for ugh = 1:10 %"for" loop going through ind
    X(1) =  highnoc(2,ugh);%control X
    Y(1) =  highnoc(1,ugh);%control Y
    X(2) =  highnohighvol(2,ugh);%stim X
    Y(2) =  highnohighvol(1,ugh); %stim Y
    plot(X(1:2),Y,'Color','[0.7 0.7 0.7]')%plot ind lines
end%ending "for" loop going through ind
X(1) = ctr(1,1); X(2) = ctr(2,1); Y(1) = y(1,1); Y(2) = y(1,2); %avg
plot(X(1:2),Y,'LineWidth',1.5,'Color','k') %plot line
X(1) = ctr(1,2); X(2) = ctr(2,2); Y(1) = y(2,1); Y(2) = y(2,2);%avg
plot(X(1:2),Y,'LineWidth',1.5,'Color','k')%plot line
X(1) = ctr(1,3); X(2) = ctr(2,3); Y(1) = y(3,1); Y(2) = y(3,2);%avg
plot(X(1:2),Y,'LineWidth',1.5,'Color','k')%plot line
e1 = errorbar(ctr(1,1), y(1,1),errorminus_vert(1,1),errorplus_vert(1,1),'LineStyle','none','LineWidth',2); %NC
e1.Color = '[0.3 0.75 0.93]'; %color
hLine = line([ctr(1,1) - errorminus_horz(1), ctr(1,1) + errorplus_horz(1)], [y(1,1), y(1,1)], 'Color', get(e1, 'Color'), 'LineWidth', 2); %horiz
e12 = errorbar(ctr(2,1), y(1,2),errorminus_vert(2,1),errorplus_vert(2,1),'LineStyle','none','LineWidth',2); %NS
e12.Color = 'b'; %color
hLine = line([ctr(2,1) - errorminus_horz(1), ctr(2,1) + errorplus_horz(1)], [y(1,2), y(1,2)], 'Color', get(e12, 'Color'), 'LineWidth', 2);%horiz
e2 = errorbar(ctr(1,2), y(2,1),errorminus_vert(1,2),errorplus_vert(1,2),'LineStyle','none','LineWidth',2); %LC
e2.Color = '[0.3 0.75 0.93]'; %color
hLine = line([ctr(1,2) - errorminus_horz(1), ctr(1,2) + errorplus_horz(1)], [y(2,1), y(2,1)], 'Color', get(e2, 'Color'), 'LineWidth', 2);%horiz
e22 = errorbar(ctr(2,2), y(2,2),errorminus_vert(2,2),errorplus_vert(2,2),'LineStyle','none','LineWidth',2); %LS
e22.Color = 'b'; %color
hLine = line([ctr(2,2) - errorminus_horz(1), ctr(2,2) + errorplus_horz(1)], [y(2,2), y(2,2)], 'Color', get(e22, 'Color'), 'LineWidth', 2);%horiz
e3 = errorbar(ctr(1,3), y(3,1),errorminus_vert(1,3),errorplus_vert(1,3),'LineStyle','none','LineWidth',2); %HC
e3.Color = '[0.3 0.75 0.93]'; %color
hLine = line([ctr(1,3) - errorminus_horz(1), ctr(1,3) + errorplus_horz(1)], [y(3,1), y(3,1)], 'Color', get(e3, 'Color'), 'LineWidth', 2);%horiz
e32 = errorbar(ctr(2,3), y(3,2),errorminus_vert(2,3),errorplus_vert(2,3),'LineStyle','none','LineWidth',2); %HS
e32.Color = 'b'; %color
hLine = line([ctr(2,3) - errorminus_horz(1), ctr(2,3) + errorplus_horz(1)], [y(3,2), y(3,2)], 'Color', get(e32, 'Color'), 'LineWidth', 2);%horiz
%ylim([0.2 1.2]) %original ylim
ylim([0 1.2]) %ylim axis to 0
XTickLabel = ({'No','Low','High'}); %x label
%set(gca,'xtick',[1 1.1
%1.2],'xticklabel',XTickLabel,'ytick',[0.4:0.2:1],'box','off','FontSize',20)
%%original aesthetics
set(gca,'xtick',[1 1.1 1.2],'xticklabel',XTickLabel,'box','off','FontSize',20) %aesthetics ylim to 0
ylabel('Learning Rate') %y label
legend({'Ctrl','Stim'},'AutoUpdate','off') %legend

%% Supplemental: dSPN - DT % Change

f = figure; %fig
ctr = []; ydt = []; %space
y = [(mean(d1min_stim_dt) - mean(d1min_no_dt))/mean(d1min_no_dt); (mean(d1low_stim_dt) - mean(d1low_no_dt))/mean(d1low_no_dt); (mean(d1high_stim_dt) - mean(d1high_no_dt))/mean(d1high_no_dt)]; %DT diff
a_vert = [dspn_noconf_dt_diff_std; dspn_lowconf_dt_diff_std; dspn_highconf_dt_diff_std;]; %SEM
errorplus_vert=a_vert'; %positive SEM
errorminus_vert=errorplus_vert; %negative SEM
x = [1; 1.05; 1.1]; %x position
a_horz = [0.04;0.04;0.04]/2; %horizontal line
errorplus_horz=a_horz'; %positive horiz
errorminus_horz=errorplus_horz; %negative horiz
hBar = bar(x,y,'FaceColor',[1 1 1],'EdgeColor',[1 1 1]); %plot DT diff
for k1 = 1:size(y,2) %"for" loop going through y
    ctr(k1,:) = bsxfun(@plus, hBar(k1).XData, hBar(k1).XOffset');      %ctr
    ydt(k1,:) = hBar(k1).YData;                    %ydt
end %ending "for" loop going through y
h = hline(0,'w'); %block 0
h.LineWidth = 2; %line width
minnodt = [-10, 3, 10, 13, -13, 5, 0, 21, 27, 39]/100; %no conf ind
lownodt = [-13, -27, -11, 11, -8, 0, -14, -15, 18, 32]/100; %low conf ind
highnodt = [-28, -10, -11, 26, -20, -3, -18, 21, 8, 29]/100; %high conf ind
minnodt(2,:) = ctr(1,1); %center
lownodt(2,:) = ctr(1,2); %center
highnodt(2,:) = ctr(1,3); %center
hold on %hold fig
h = hline(0,'--k'); %0 line
h.Color = [0.64 0.08 0.18]; %color
c = [0.32 0.61 0.81]; %scatter color
scatter(minnodt(2,:),minnodt(1,:),30,c); %scatter ind
scatter(lownodt(2,:),lownodt(1,:),30,c); %scatter ind
scatter(highnodt(2,:),highnodt(1,:),30,c); %scatter ind
e1 = errorbar(x(1), y(1), errorminus_vert(1), errorplus_vert(1), 'LineStyle', 'none', 'LineWidth', 2); %no conf
e1.Color = 'b'; %color
hLine = line([x(1) - errorminus_horz(1), x(1) + errorplus_horz(1)], [y(1), y(1)], 'Color', get(e1, 'Color'), 'LineWidth', 2); %horiz
e2 = errorbar(x(2), y(2), errorminus_vert(2), errorplus_vert(2), 'LineStyle', 'none', 'LineWidth', 2);%low conf
e2.Color = 'b'; %color
hLine = line([x(2) - errorminus_horz(2), x(2) + errorplus_horz(2)], [y(2), y(2)], 'Color', get(e2, 'Color'), 'LineWidth', 2);%horiz
e3 = errorbar(x(3), y(3), errorminus_vert(3), errorplus_vert(3), 'LineStyle', 'none', 'LineWidth', 2); %high conf
e3.Color = 'b'; %color
hLine = line([x(3) - errorminus_horz(3), x(3) + errorplus_horz(3)], [y(3), y(3)], 'Color', get(e3, 'Color'), 'LineWidth', 2); %horiz
ylim([-0.3 0.4]) %ylim
xlim([0.97 1.13]) %xlim
XTickLabel = ({'No','Low','High'}); %x label
set(gca,'xtick',[1 1.05 1.1],'xticklabel',XTickLabel,'box','off','FontSize',20) %aesthetics
ylabel('Decision Time % Change') %y label