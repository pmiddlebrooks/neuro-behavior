%% percent correct - non-preferred vs preferred

ctrl_left_percent = [];
ctrl_left_percent = a2amin_no_left_percent;
L = length(ctrl_left_percent);
ctrl_left_percent(L+1:L+length(a2alow_no_left_percent)) = a2alow_no_left_percent;
L = length(ctrl_left_percent);
ctrl_left_percent(L+1:L+length(a2ahigh_no_left_percent)) = a2ahigh_no_left_percent;
ctrl_right_percent = [];
ctrl_right_percent = a2amin_no_right_percent;
L = length(ctrl_right_percent);
ctrl_right_percent(L+1:L+length(a2alow_no_right_percent)) = a2alow_no_right_percent;
L = length(ctrl_right_percent);
ctrl_right_percent(L+1:L+length(a2ahigh_no_right_percent)) = a2ahigh_no_right_percent;
stim_left_percent = [];
stim_left_percent = min_stim_left_percent;
L = length(stim_left_percent);
stim_left_percent(L+1:L+length(low_stim_left_percent)) = low_stim_left_percent;
L = length(stim_left_percent);
stim_left_percent(L+1:L+length(high_stim_left_percent)) = high_stim_left_percent;
stim_right_percent = [];
stim_right_percent = min_stim_right_percent;
L = length(stim_right_percent);
stim_right_percent(L+1:L+length(low_stim_right_percent)) = low_stim_right_percent;
L = length(stim_right_percent);
stim_right_percent(L+1:L+length(high_stim_right_percent)) = high_stim_right_percent;

sigs = [];
[h,p] = kstest2(ctrl_left_percent,stim_left_percent);
sigs(1) = p;
[h,p] = kstest2(ctrl_right_percent,stim_right_percent);
sigs(2) = p;

f = figure;
ctr = []; ydt = [];
y = [mean(ctrl_left_percent) mean(stim_left_percent); mean(ctrl_right_percent) mean(stim_right_percent)];
a_vert = [std(ctrl_left_percent)/sqrt(length(ctrl_left_percent)) std(stim_left_percent)/sqrt(length(stim_left_percent)); std(ctrl_right_percent)/sqrt(length(ctrl_right_percent)) std(stim_right_percent)/sqrt(length(stim_right_percent))];
errorplus_vert = a_vert';
errorminus_vert=errorplus_vert;
x = [1:2];
a_horz = [0.1 0.1; 0.1 0.1];
errorplus_horz=a_horz';
errorminus_horz=errorplus_horz;
hBar = bar(x,y,'FaceColor',[1 1 1],'EdgeColor',[1 1 1]);
for k1 = 1:size(y,2)
    ctr(k1,:) = bsxfun(@plus, hBar(k1).XData, hBar(k1).XOffset');     
    ydt(k1,:) = hBar(k1).YData;                    
end
ctrlL = [0.61 0.57 0.64 0.57 0.63 0.6 0.56 0.63 0.61 0.6];
stimL = [0.54 0.63 0.52 0.57 0.62 0.52 0.46 0.5 0.51 0.5];
ctrlR = [0.66 0.72 0.7 0.62 0.67 0.63 0.67 0.64 0.54 0.62];
stimR = [0.53 0.37 0.53 0.5 0.61 0.53 0.56 0.53 0.51 0.53];
 ctrlL(2,:) = ctr(1,1);
 stimL(2,:) = ctr(2,1);
 ctrlR(2,:) = ctr(1,2);
 stimR(2,:) = ctr(2,2);
hold on
for ugh = 1:10
    X(1) =  ctrlL(2,ugh);
    Y(1) =  ctrlL(1,ugh);
    X(2) =  stimL(2,ugh);
    Y(2) =  stimL(1,ugh);
    plot(X(1:2),Y,'Color','[0.7 0.7 0.7]')
end
for ugh = 1:10
    X(1) =  ctrlR(2,ugh);
    Y(1) =  ctrlR(1,ugh);
    X(2) =  stimR(2,ugh);
    Y(2) =  stimR(1,ugh);
    plot(X(1:2),Y,'Color','[0.7 0.7 0.7]')
end
X(1) = ctr(1,1); X(2) = ctr(2,1); Y(1) = y(1,1); Y(2) = y(1,2);
plot(X(1:2),Y,'LineWidth',1.5,'Color','k')
X(1) = ctr(1,2); X(2) = ctr(2,2); Y(1) = y(2,1); Y(2) = y(2,2);
plot(X(1:2),Y,'LineWidth',1.5,'Color','k')
e1 = errorbar(ctr(1,1), y(1,1),errorminus_vert(1,1),errorplus_vert(1,1),'LineStyle','none','LineWidth',2); 
e1.Color = '[1 0.51 0.51]'; 
hLine = line([ctr(1,1) - errorminus_horz(1), ctr(1,1) + errorplus_horz(1)], [y(1,1), y(1,1)], 'Color', get(e1, 'Color'), 'LineWidth', 2);
e12 = errorbar(ctr(2,1), y(1,2),errorminus_vert(2,1),errorplus_vert(2,1),'LineStyle','none','LineWidth',2); 
e12.Color = 'r'; 
hLine = line([ctr(2,1) - errorminus_horz(1), ctr(2,1) + errorplus_horz(1)], [y(1,2), y(1,2)], 'Color', get(e12, 'Color'), 'LineWidth', 2);
e2 = errorbar(ctr(1,2), y(2,1),errorminus_vert(1,2),errorplus_vert(1,2),'LineStyle','none','LineWidth',2); 
e2.Color = '[1 0.51 0.51]'; 
hLine = line([ctr(1,2) - errorminus_horz(1), ctr(1,2) + errorplus_horz(1)], [y(2,1), y(2,1)], 'Color', get(e2, 'Color'), 'LineWidth', 2);
e22 = errorbar(ctr(2,2), y(2,2),errorminus_vert(2,2),errorplus_vert(2,2),'LineStyle','none','LineWidth',2); 
e22.Color = 'r'; 
hLine = line([ctr(2,2) - errorminus_horz(1), ctr(2,2) + errorplus_horz(1)], [y(2,2), y(2,2)], 'Color', get(e22, 'Color'), 'LineWidth', 2);
ylim([0.3 0.8])
XTickLabel = ({'Non-Pref','Pref'});
set(gca,'xticklabel',XTickLabel,'box','off','FontSize',20)
ylabel('P(Correct)')
X(1) = ctr(1,2); X(2) = ctr(2,2); Y(1) = 0.75; Y(2) = 0.75;
plot(X(1:2),Y,'LineWidth',1.5,'Color','k')

%% learning rate - rewarded vs unrewarded

ctrl_rew_lr = [];
ctrl_rew_lr = a2amin_no_rew_lr;
L = length(ctrl_rew_lr);
ctrl_rew_lr(L+1:L+length(a2alow_no_rew_lr)) = a2alow_no_rew_lr;
L = length(ctrl_rew_lr);
ctrl_rew_lr(L+1:L+length(a2ahigh_no_rew_lr)) = a2ahigh_no_rew_lr;
ctrl_unrew_lr = [];
ctrl_unrew_lr = a2amin_no_unrew_lr;
L = length(ctrl_unrew_lr);
ctrl_unrew_lr(L+1:L+length(a2alow_no_unrew_lr)) = a2alow_no_unrew_lr;
L = length(ctrl_unrew_lr);
ctrl_unrew_lr(L+1:L+length(a2ahigh_no_unrew_lr)) = a2ahigh_no_unrew_lr;
stim_rew_lr = [];
stim_rew_lr = min_stim_rew_lr;
L = length(stim_rew_lr);
stim_rew_lr(L+1:L+length(low_stim_rew_lr)) = low_stim_rew_lr;
L = length(stim_rew_lr);
stim_rew_lr(L+1:L+length(high_stim_rew_lr)) = high_stim_rew_lr;
stim_unrew_lr = [];
stim_unrew_lr = min_stim_unrew_lr;
L = length(stim_unrew_lr);
stim_unrew_lr(L+1:L+length(low_stim_unrew_lr)) = low_stim_unrew_lr;
L = length(stim_unrew_lr);
stim_unrew_lr(L+1:L+length(high_stim_unrew_lr)) = high_stim_unrew_lr;

sigs = [];
[h,p] = kstest2(ctrl_rew_lr,stim_rew_lr);
sigs(1) = p;
[h,p] = kstest2(ctrl_unrew_lr,stim_unrew_lr);
sigs(2) = p;

f = figure;
ctr = []; ydt = [];
y = [mean(ctrl_rew_lr) mean(stim_rew_lr); mean(ctrl_unrew_lr) mean(stim_unrew_lr)];
a_vert = [std(ctrl_rew_lr)/sqrt(length(ctrl_rew_lr)) std(stim_rew_lr)/sqrt(length(stim_rew_lr)); std(ctrl_unrew_lr)/sqrt(length(ctrl_unrew_lr)) std(stim_unrew_lr)/sqrt(length(stim_unrew_lr))];
errorplus_vert = a_vert';
errorminus_vert=errorplus_vert;
x = [1:2];
a_horz = [0.1 0.1; 0.1 0.1];
errorplus_horz=a_horz';
errorminus_horz=errorplus_horz;
hBar = bar(x,y,'FaceColor',[1 1 1],'EdgeColor',[1 1 1]);
for k1 = 1:size(y,2)
    ctr(k1,:) = bsxfun(@plus, hBar(k1).XData, hBar(k1).XOffset');     
    ydt(k1,:) = hBar(k1).YData;                    
end
ctrlL = [0.62 0.54 1 0.73 0.93 0.92 0.93 0.95 0.69 0.88];
stimL = [0.21 0.02 0 0.22 0.26 0.26 0.17 0 0.05 0];
ctrlR = [0.63 0.76 0.89 0.61 0.85 0.9 0.84 0.85 0.7 0.85];
stimR = [0.36 0.05 0.42 0.11 0.82 0 0.19 0.38 0.13 0.01];
 ctrlL(2,:) = ctr(1,1);
 stimL(2,:) = ctr(2,1);
 ctrlR(2,:) = ctr(1,2);
 stimR(2,:) = ctr(2,2);
hold on
h = hline(0,'w');
h.LineWidth = 2;
for ugh = 1:10
    X(1) =  ctrlL(2,ugh);
    Y(1) =  ctrlL(1,ugh);
    X(2) =  stimL(2,ugh);
    Y(2) =  stimL(1,ugh);
    plot(X(1:2),Y,'Color','[0.7 0.7 0.7]')
end
for ugh = 1:10
    X(1) =  ctrlR(2,ugh);
    Y(1) =  ctrlR(1,ugh);
    X(2) =  stimR(2,ugh);
    Y(2) =  stimR(1,ugh);
    plot(X(1:2),Y,'Color','[0.7 0.7 0.7]')
end
X(1) = ctr(1,1); X(2) = ctr(2,1); Y(1) = y(1,1); Y(2) = y(1,2);
plot(X(1:2),Y,'LineWidth',1.5,'Color','k')
X(1) = ctr(1,2); X(2) = ctr(2,2); Y(1) = y(2,1); Y(2) = y(2,2);
plot(X(1:2),Y,'LineWidth',1.5,'Color','k')
e1 = errorbar(ctr(1,1), y(1,1),errorminus_vert(1,1),errorplus_vert(1,1),'LineStyle','none','LineWidth',2); 
e1.Color = '[1 0.51 0.51]'; 
hLine = line([ctr(1,1) - errorminus_horz(1), ctr(1,1) + errorplus_horz(1)], [y(1,1), y(1,1)], 'Color', get(e1, 'Color'), 'LineWidth', 2);
e12 = errorbar(ctr(2,1), y(1,2),errorminus_vert(2,1),errorplus_vert(2,1),'LineStyle','none','LineWidth',2); 
e12.Color = 'r'; 
hLine = line([ctr(2,1) - errorminus_horz(1), ctr(2,1) + errorplus_horz(1)], [y(1,2), y(1,2)], 'Color', get(e12, 'Color'), 'LineWidth', 2);
e2 = errorbar(ctr(1,2), y(2,1),errorminus_vert(1,2),errorplus_vert(1,2),'LineStyle','none','LineWidth',2); 
e2.Color = '[1 0.51 0.51]'; 
hLine = line([ctr(1,2) - errorminus_horz(1), ctr(1,2) + errorplus_horz(1)], [y(2,1), y(2,1)], 'Color', get(e2, 'Color'), 'LineWidth', 2);
e22 = errorbar(ctr(2,2), y(2,2),errorminus_vert(2,2),errorplus_vert(2,2),'LineStyle','none','LineWidth',2); 
e22.Color = 'r'; 
hLine = line([ctr(2,2) - errorminus_horz(1), ctr(2,2) + errorplus_horz(1)], [y(2,2), y(2,2)], 'Color', get(e22, 'Color'), 'LineWidth', 2);
ylim([-0.2 1.2])
XTickLabel = ({'Rew','Unrew'});
set(gca,'xticklabel',XTickLabel,'box','off','FontSize',20)
ylabel('Learning Rate')
X(1) = ctr(1,1); X(2) = ctr(2,1); Y(1) = 1.03; Y(2) = 1.03;
plot(X(1:2),Y,'LineWidth',1.5,'Color','k')
X(1) = ctr(1,2); X(2) = ctr(2,2); Y(1) = 0.93; Y(2) = 0.93;
plot(X(1:2),Y,'LineWidth',1.5,'Color','k')

%% regression weight - outcome vs choice - control vs stim vs scrambled model

ctrl_r1 = [];
ctrl_r1 = a2amin_no_r1;
L = length(ctrl_r1);
ctrl_r1(L+1:L+length(a2alow_no_r1)) = a2alow_no_r1;
L = length(ctrl_r1);
ctrl_r1(L+1:L+length(a2ahigh_no_r1)) = a2ahigh_no_r1;
ctrl_c1 = [];
ctrl_c1 = a2amin_no_c1;
L = length(ctrl_c1);
ctrl_c1(L+1:L+length(a2alow_no_c1)) = a2alow_no_c1;
L = length(ctrl_c1);
ctrl_c1(L+1:L+length(a2ahigh_no_c1)) = a2ahigh_no_c1;
stim_r1 = [];
stim_r1 = min_stim_r1;
L = length(stim_r1);
stim_r1(L+1:L+length(low_stim_r1)) = low_stim_r1;
L = length(stim_r1);
stim_r1(L+1:L+length(high_stim_r1)) = high_stim_r1;
stim_c1 = [];
stim_c1 = min_stim_c1;
L = length(stim_c1);
stim_c1(L+1:L+length(low_stim_c1)) = low_stim_c1;
L = length(stim_c1);
stim_c1(L+1:L+length(high_stim_c1)) = high_stim_c1;
model_r1 = [];
model_r1 = noconflowvolscram(:,1);
L = length(model_r1);
model_r1(L+1:L+length(lowconflowvolscram(:,1))) = lowconflowvolscram(:,1);
L = length(model_r1);
model_r1(L+1:L+length(highconflowvolscram(:,1))) = highconflowvolscram(:,1);
model_c1 = [];
model_c1 = noconflowvolscram(:,2);
L = length(model_c1);
model_c1(L+1:L+length(lowconflowvolscram(:,2))) = lowconflowvolscram(:,2);
L = length(model_c1);
model_c1(L+1:L+length(highconflowvolscram(:,2))) = highconflowvolscram(:,2);

sigs = [];
[h,p] = kstest2(ctrl_r1,stim_r1);
sigs(1,1) = p;
[h,p] = kstest2(ctrl_r1,model_r1);
sigs(1,2) = p;
[h,p] = kstest2(stim_r1,model_r1);
sigs(1,3) = p;
[h,p] = kstest2(ctrl_c1,stim_c1);
sigs(2,1) = p;
[h,p] = kstest2(ctrl_c1,model_c1);
sigs(2,2) = p;
[h,p] = kstest2(stim_c1,model_c1);
sigs(2,3) = p;

f = figure;
ctr = []; ydt = [];
y = [mean(ctrl_r1) mean(stim_r1) mean(model_r1); mean(ctrl_c1) mean(stim_c1) mean(model_r1)];
a_vert = [std(ctrl_r1)/sqrt(length(ctrl_r1)) std(stim_r1)/sqrt(length(stim_r1)) std(model_r1)/sqrt(length(model_r1)); std(ctrl_c1)/sqrt(length(ctrl_c1)) std(stim_c1)/sqrt(length(stim_c1)) std(model_c1)/sqrt(length(model_c1))];
errorplus_vert = a_vert';
errorminus_vert=errorplus_vert;
x = [1:2];
a_horz = [0.1 0.1 0.1; 0.1 0.1 0.1];
errorplus_horz=a_horz';
errorminus_horz=errorplus_horz;
hBar = bar(x,y,'FaceColor',[1 1 1],'EdgeColor',[1 1 1]);
for k1 = 1:size(y,2)
    ctr(k1,:) = bsxfun(@plus, hBar(k1).XData, hBar(k1).XOffset');     
    ydt(k1,:) = hBar(k1).YData;                    
end
ctrlL = [0.48 0.73 0.94 0.39 0.8 0.6 0.57 0.76 0.37 0.57];
stimL = [0.25 0.02 0.22 0.14 0.6 0.16 0.04 0.11 0.09 0.11];
modelL = [-0.05 0.02 0.05 0.02 0.06 0 0.03 0.23 0.05 -0.05];
ctrlR = [0.89 0.97 0.59 0.37 0.09 0.8 0.98 0.54 0.02 0.6];
stimR = [0.24 0.22 -0.03 0.2 0.14 -0.02 -0.06 0.1 -0.01 -0.12];
modelR = [0.08 -0.01 -0.02 0 -0.04 -0.02 -0.06 -0.03 -0.07 0.09];
 ctrlL(2,:) = ctr(1,1);
 stimL(2,:) = ctr(2,1);
 modelL(2,:) = ctr(3,1);
 ctrlR(2,:) = ctr(1,2);
 stimR(2,:) = ctr(2,2);
 modelR(2,:) = ctr(3,2);
hold on
h = hline(0,'w');
h.LineWidth = 2;
scatter(ctrlL(2,:),ctrlL(1,:),30,[0.71 0.36 0.36]);
scatter(stimL(2,:),stimL(1,:),30,[0.67 0.01 0.01]);
scatter(modelL(2,:),modelL(1,:),30,[0.8 0.8 0.8]);
scatter(ctrlR(2,:),ctrlR(1,:),30,[0.71 0.36 0.36]);
scatter(stimR(2,:),stimR(1,:),30,[0.67 0.01 0.01]);
scatter(modelR(2,:),modelR(1,:),30,[0.8 0.8 0.8]);
e1 = errorbar(ctr(1,1), y(1,1),errorminus_vert(1,1),errorplus_vert(1,1),'LineStyle','none','LineWidth',2); 
e1.Color = '[1 0.51 0.51]'; 
hLine = line([ctr(1,1) - errorminus_horz(1), ctr(1,1) + errorplus_horz(1)], [y(1,1), y(1,1)], 'Color', get(e1, 'Color'), 'LineWidth', 2);
e12 = errorbar(ctr(2,1), y(1,2),errorminus_vert(2,1),errorplus_vert(2,1),'LineStyle','none','LineWidth',2); 
e12.Color = 'r'; 
hLine = line([ctr(2,1) - errorminus_horz(1), ctr(2,1) + errorplus_horz(1)], [y(1,2), y(1,2)], 'Color', get(e12, 'Color'), 'LineWidth', 2);
e13 = errorbar(ctr(3,1), y(1,3),errorminus_vert(3,1),errorplus_vert(3,1),'LineStyle','none','LineWidth',2); 
e13.Color = '[0.5 0.5 0.5]'; 
hLine = line([ctr(3,1) - errorminus_horz(1), ctr(3,1) + errorplus_horz(1)], [y(1,3), y(1,3)], 'Color', get(e13, 'Color'), 'LineWidth', 2);
e2 = errorbar(ctr(1,2), y(2,1),errorminus_vert(1,2),errorplus_vert(1,2),'LineStyle','none','LineWidth',2); 
e2.Color = '[1 0.51 0.51]'; 
hLine = line([ctr(1,2) - errorminus_horz(1), ctr(1,2) + errorplus_horz(1)], [y(2,1), y(2,1)], 'Color', get(e2, 'Color'), 'LineWidth', 2);
e22 = errorbar(ctr(2,2), y(2,2),errorminus_vert(2,2),errorplus_vert(2,2),'LineStyle','none','LineWidth',2); 
e22.Color = 'r'; 
hLine = line([ctr(2,2) - errorminus_horz(1), ctr(2,2) + errorplus_horz(1)], [y(2,2), y(2,2)], 'Color', get(e22, 'Color'), 'LineWidth', 2);
e23 = errorbar(ctr(3,2), y(2,3),errorminus_vert(3,2),errorplus_vert(3,2),'LineStyle','none','LineWidth',2); 
e23.Color = '[0.5 0.5 0.5]'; 
hLine = line([ctr(3,2) - errorminus_horz(1), ctr(3,2) + errorplus_horz(1)], [y(2,3), y(2,3)], 'Color', get(e23, 'Color'), 'LineWidth', 2);
ylim([-0.2 1])
XTickLabel = ({'Outcome','Choice'});
set(gca,'xticklabel',XTickLabel,'ytick',[-0.2:0.4:1],'box','off','FontSize',20)
ylabel('Regression Weight')

%% learning rate

ctrl_lr = [];
ctrl_lr = a2amin_no_lr;
L = length(ctrl_lr);
ctrl_lr(L+1:L+length(a2alow_no_lr)) = a2alow_no_lr;
L = length(ctrl_lr);
ctrl_lr(L+1:L+length(a2ahigh_no_lr)) = a2ahigh_no_lr;
stim_lr = [];
stim_lr = min_stim_lr;
L = length(stim_lr);
stim_lr(L+1:L+length(low_stim_lr)) = low_stim_lr;
L = length(stim_lr);
stim_lr(L+1:L+length(high_stim_lr)) = high_stim_lr;

sigs = [];
[h,p] = kstest2(ctrl_lr,stim_lr);
sigs(1,1) = p;

f = figure;
ctr = []; ydt = [];
y = [mean(ctrl_lr); mean(stim_lr)];
a_vert = [std(ctrl_lr)/sqrt(length(ctrl_lr)); std(stim_lr)/sqrt(length(stim_lr))];
errorplus_vert = a_vert';
errorminus_vert=errorplus_vert;
x = [1:2];
a_horz = [0.1; 0.1];
errorplus_horz=a_horz';
errorminus_horz=errorplus_horz;
hBar = bar(x,y,'FaceColor',[1 1 1],'EdgeColor',[1 1 1]);
for k1 = 1:size(y,2)
    ctr(k1,:) = bsxfun(@plus, hBar(k1).XData, hBar(k1).XOffset');     
    ydt(k1,:) = hBar(k1).YData;                    
end
ctrlL = [0.78 0.8 0.63 1 0.86 0.71 0.89 0.9 0.91 0.99];
stimL = [0.05 0.13 0.06 0.35 0.57 0.43 0.33 0.64 0.83 0.24];
 ctrlL(2,:) = ctr(1,1);
 stimL(2,:) = ctr(1,2);
hold on
for ugh = 1:10
    X(1) =  ctrlL(2,ugh);
    Y(1) =  ctrlL(1,ugh);
    X(2) =  stimL(2,ugh);
    Y(2) =  stimL(1,ugh);
    plot(X(1:2),Y,'Color','[0.7 0.7 0.7]')
end
e1 = errorbar(ctr(1,1), y(1,1),errorminus_vert(1,1),errorplus_vert(1,1),'LineStyle','none','LineWidth',2); 
e1.Color = '[1 0.51 0.51]'; 
hLine = line([ctr(1,1) - errorminus_horz(1), ctr(1,1) + errorplus_horz(1)], [y(1,1), y(1,1)], 'Color', get(e1, 'Color'), 'LineWidth', 2);
e2 = errorbar(ctr(1,2), y(2,1),errorminus_vert(1,2),errorplus_vert(1,2),'LineStyle','none','LineWidth',2); 
e2.Color = 'r'; 
hLine = line([ctr(1,2) - errorminus_horz(1), ctr(1,2) + errorplus_horz(1)], [y(2,1), y(2,1)], 'Color', get(e2, 'Color'), 'LineWidth', 2);
ylim([0 1.2])
XTickLabel = ({'Ctrl','Stim'});
set(gca,'xticklabel',XTickLabel,'box','off','FontSize',20)
ylabel('Learning Rate')
X(1) = ctr(1,1); X(2) = ctr(1,2); Y(1) = 1.03; Y(2) = 1.03;
plot(X(1:2),Y,'LineWidth',1.5,'Color','k')

%% regression weight - rewarded vs unrewarded vs bias

ctrl_rew1 = [];
ctrl_rew1 = a2amin_no_rew1;
L = length(ctrl_rew1);
ctrl_rew1(L+1:L+length(a2alow_no_rew1)) = a2alow_no_rew1;
L = length(ctrl_rew1);
ctrl_rew1(L+1:L+length(a2ahigh_no_rew1)) = a2ahigh_no_rew1;
stim_rew1 = [];
stim_rew1 = min_stim_rew1;
L = length(stim_rew1);
stim_rew1(L+1:L+length(low_stim_rew1)) = low_stim_rew1;
L = length(stim_rew1);
stim_rew1(L+1:L+length(high_stim_rew1)) = high_stim_rew1;
ctrl_unrew1 = [];
ctrl_unrew1 = a2amin_no_unrew1;
L = length(ctrl_unrew1);
ctrl_unrew1(L+1:L+length(a2alow_no_unrew1)) = a2alow_no_unrew1;
L = length(ctrl_unrew1);
ctrl_unrew1(L+1:L+length(a2ahigh_no_unrew1)) = a2ahigh_no_unrew1;
ctrl_unrew1 = ctrl_unrew1*-1;
stim_unrew1 = [];
stim_unrew1 = min_stim_unrew1;
L = length(stim_unrew1);
stim_unrew1(L+1:L+length(low_stim_unrew1)) = low_stim_unrew1;
L = length(stim_unrew1);
stim_unrew1(L+1:L+length(high_stim_unrew1)) = high_stim_unrew1;
stim_unrew1 = stim_unrew1*-1;
ctrl_bias = [];
ctrl_bias = a2amin_no_bias;
L = length(ctrl_bias);
ctrl_bias(L+1:L+length(a2alow_no_bias)) = a2alow_no_bias;
L = length(ctrl_bias);
ctrl_bias(L+1:L+length(a2ahigh_no_bias)) = a2ahigh_no_bias;
stim_bias = [];
stim_bias = min_stim_bias;
L = length(stim_bias);
stim_bias(L+1:L+length(low_stim_bias)) = low_stim_bias;
L = length(stim_bias);
stim_bias(L+1:L+length(high_stim_bias)) = high_stim_bias;

sigs = [];
[h,p] = kstest2(ctrl_rew1,stim_rew1);
sigs(1,1) = p;
[h,p] = kstest2(ctrl_unrew1,stim_unrew1);
sigs(1,2) = p;
[h,p] = kstest2(ctrl_bias,stim_bias);
sigs(1,3) = p;

f = figure;
ctr = []; ydt = [];
y = [mean(ctrl_rew1) mean(stim_rew1); mean(ctrl_unrew1) mean(stim_unrew1); mean(ctrl_bias) mean(stim_bias)];
a_vert = [std(ctrl_rew1)/sqrt(length(ctrl_rew1)) std(stim_rew1)/sqrt(length(stim_rew1)); std(ctrl_unrew1)/sqrt(length(ctrl_unrew1)) std(stim_unrew1)/sqrt(length(stim_unrew1)); std(ctrl_bias)/sqrt(length(ctrl_bias)) std(stim_bias)/sqrt(length(stim_bias))];
errorplus_vert = a_vert';
errorminus_vert=errorplus_vert;
x = [1:3];
a_horz = [0.1 0.1; 0.1 0.1; 0.1 0.1];
errorplus_horz=a_horz';
errorminus_horz=errorplus_horz;
hBar = bar(x,y,'FaceColor',[1 1 1],'EdgeColor',[1 1 1]);
for k1 = 1:size(y,2)
    ctr(k1,:) = bsxfun(@plus, hBar(k1).XData, hBar(k1).XOffset');     
    ydt(k1,:) = hBar(k1).YData;                    
end
ctrlR = [1.37 1.7 1.52 0.76 0.89 1.4 1.55 1.3 0.35 1.26];
stimR = [0.48 0.24 0.19 0.33 0.75 0.14 -0.02 0.14 0.08 -0.01];
ctrlU = [-0.11 0.05 0.05 0.02 0.04 0.07 0.04 0.15 0.39 0.03];
stimU = [0.01 -0.16 0.25 -0.07 0.46 0.18 0.1 0.5 0.11 0.23];
ctrlB = [0.21 0.19 0.13 0.09 0.06 0.06 0.2 0.03 0.05 0.04];
stimB = [-0.01 -0.46 0.01 -0.13 0.02 0.02 0.22 0.05 -0.01 0.07];
 ctrlR(2,:) = ctr(1,1);
 stimR(2,:) = ctr(2,1);
  ctrlU(2,:) = ctr(1,2);
 stimU(2,:) = ctr(2,2);
  ctrlB(2,:) = ctr(1,3);
 stimB(2,:) = ctr(2,3);
hold on
h = hline(0,'w');
h.LineWidth = 2;
for ugh = 1:10
    X(1) =  ctrlR(2,ugh);
    Y(1) =  ctrlR(1,ugh);
    X(2) =  stimR(2,ugh);
    Y(2) =  stimR(1,ugh);
    plot(X(1:2),Y,'Color','[0.7 0.7 0.7]')
end
for ugh = 1:10
    X(1) =  ctrlU(2,ugh);
    Y(1) =  ctrlU(1,ugh);
    X(2) =  stimU(2,ugh);
    Y(2) =  stimU(1,ugh);
    plot(X(1:2),Y,'Color','[0.7 0.7 0.7]')
end
for ugh = 1:10
    X(1) =  ctrlB(2,ugh);
    Y(1) =  ctrlB(1,ugh);
    X(2) =  stimB(2,ugh);
    Y(2) =  stimB(1,ugh);
    plot(X(1:2),Y,'Color','[0.7 0.7 0.7]')
end
e1 = errorbar(ctr(1,1), y(1,1),errorminus_vert(1,1),errorplus_vert(1,1),'LineStyle','none','LineWidth',2); 
e1.Color = '[1 0.51 0.51]'; 
hLine = line([ctr(1,1) - errorminus_horz(1), ctr(1,1) + errorplus_horz(1)], [y(1,1), y(1,1)], 'Color', get(e1, 'Color'), 'LineWidth', 2);
e12 = errorbar(ctr(2,1), y(1,2),errorminus_vert(2,1),errorplus_vert(2,1),'LineStyle','none','LineWidth',2); 
e12.Color = 'r'; 
hLine = line([ctr(2,1) - errorminus_horz(1), ctr(2,1) + errorplus_horz(1)], [y(1,2), y(1,2)], 'Color', get(e12, 'Color'), 'LineWidth', 2);
e2 = errorbar(ctr(1,2), y(2,1),errorminus_vert(1,2),errorplus_vert(1,2),'LineStyle','none','LineWidth',2); 
e2.Color = '[1 0.51 0.51]'; 
hLine = line([ctr(1,2) - errorminus_horz(1), ctr(1,2) + errorplus_horz(1)], [y(2,1), y(2,1)], 'Color', get(e2, 'Color'), 'LineWidth', 2);
e22 = errorbar(ctr(2,2), y(2,2),errorminus_vert(2,2),errorplus_vert(2,2),'LineStyle','none','LineWidth',2); 
e22.Color = 'r'; 
hLine = line([ctr(2,2) - errorminus_horz(1), ctr(2,2) + errorplus_horz(1)], [y(2,2), y(2,2)], 'Color', get(e22, 'Color'), 'LineWidth', 2);
e3 = errorbar(ctr(1,3), y(3,1),errorminus_vert(1,3),errorplus_vert(1,3),'LineStyle','none','LineWidth',2); 
e3.Color = '[1 0.51 0.51]'; 
hLine = line([ctr(1,3) - errorminus_horz(1), ctr(1,3) + errorplus_horz(1)], [y(3,1), y(3,1)], 'Color', get(e3, 'Color'), 'LineWidth', 2);
e32 = errorbar(ctr(2,3), y(3,2),errorminus_vert(2,3),errorplus_vert(2,3),'LineStyle','none','LineWidth',2); 
e32.Color = 'r'; 
hLine = line([ctr(2,3) - errorminus_horz(1), ctr(2,3) + errorplus_horz(1)], [y(3,2), y(3,2)], 'Color', get(e32, 'Color'), 'LineWidth', 2);
ylim([-0.5 2])
XTickLabel = ({'Rew','Unrew','Bias'});
set(gca,'xticklabel',XTickLabel,'box','off','FontSize',20)
ylabel('Regression Weight')
X(1) = ctr(1,1); X(2) = ctr(2,1); Y(1) = 1.73; Y(2) = 1.73;
plot(X(1:2),Y,'LineWidth',1.5,'Color','k')
X(1) = ctr(1,3); X(2) = ctr(2,3); Y(1) = 0.25; Y(2) = 0.25;
plot(X(1:2),Y,'LineWidth',1.5,'Color','k')

%% repetition probability - rewarded vs unrewarded

ctrl_ws = [];
ctrl_ws = a2amin_no_ws_left;
L = length(ctrl_ws);
ctrl_ws(L+1:L+length(a2alow_no_ws_left)) = a2alow_no_ws_left;
L = length(ctrl_ws);
ctrl_ws(L+1:L+length(a2ahigh_no_ws_left)) = a2ahigh_no_ws_left;
L = length(ctrl_ws);
ctrl_ws(L+1:L+length(a2amin_no_ws_right)) = a2amin_no_ws_right;
L = length(ctrl_ws);
ctrl_ws(L+1:L+length(a2alow_no_ws_right)) = a2alow_no_ws_right;
L = length(ctrl_ws);
ctrl_ws(L+1:L+length(a2ahigh_no_ws_right)) = a2ahigh_no_ws_right;
stim_ws = [];
stim_ws = min_stim_ws_left;
L = length(stim_ws);
stim_ws(L+1:L+length(low_stim_ws_left)) = low_stim_ws_left;
L = length(stim_ws);
stim_ws(L+1:L+length(high_stim_ws_left)) = high_stim_ws_left;
L = length(stim_ws);
stim_ws(L+1:L+length(min_stim_ws_right)) = min_stim_ws_right;
L = length(stim_ws);
stim_ws(L+1:L+length(low_stim_ws_right)) = low_stim_ws_right;
L = length(stim_ws);
stim_ws(L+1:L+length(high_stim_ws_right)) = high_stim_ws_right;
ctrl_ls = [];
ctrl_ls = a2amin_no_ls_left;
L = length(ctrl_ls);
ctrl_ls(L+1:L+length(a2alow_no_ls_left)) = a2alow_no_ls_left;
L = length(ctrl_ls);
ctrl_ls(L+1:L+length(a2ahigh_no_ls_left)) = a2ahigh_no_ls_left;
L = length(ctrl_ls);
ctrl_ls(L+1:L+length(a2amin_no_ls_right)) = a2amin_no_ls_right;
L = length(ctrl_ls);
ctrl_ls(L+1:L+length(a2alow_no_ls_right)) = a2alow_no_ls_right;
L = length(ctrl_ls);
ctrl_ls(L+1:L+length(a2ahigh_no_ls_right)) = a2ahigh_no_ls_right;
ctrl_ls = 1 - ctrl_ls;
stim_ls = [];
stim_ls = min_stim_ls_left;
L = length(stim_ls);
stim_ls(L+1:L+length(low_stim_ls_left)) = low_stim_ls_left;
L = length(stim_ls);
stim_ls(L+1:L+length(high_stim_ls_left)) = high_stim_ls_left;
L = length(stim_ls);
stim_ls(L+1:L+length(min_stim_ls_right)) = min_stim_ls_right;
L = length(stim_ls);
stim_ls(L+1:L+length(low_stim_ls_right)) = low_stim_ls_right;
L = length(stim_ls);
stim_ls(L+1:L+length(high_stim_ls_right)) = high_stim_ls_right;
stim_ls = 1 - stim_ls;

sigs = [];
[h,p] = kstest2(ctrl_ws,stim_ws);
sigs(1,1) = p;
[h,p] = kstest2(ctrl_ls,stim_ls);
sigs(1,2) = p;

f = figure;
ctr = []; ydt = [];
y = [mean(ctrl_ws) mean(stim_ws); mean(ctrl_ls) mean(stim_ls)];
a_vert = [std(ctrl_ws)/sqrt(length(ctrl_ws)) std(stim_ws)/sqrt(length(stim_ws)); std(ctrl_ls)/sqrt(length(ctrl_ls)) std(stim_ls)/sqrt(length(stim_ls))];
errorplus_vert = a_vert';
errorminus_vert=errorplus_vert;
x = [1:2];
a_horz = [0.1 0.1; 0.1 0.1];
errorplus_horz=a_horz';
errorminus_horz=errorplus_horz;
hBar = bar(x,y,'FaceColor',[1 1 1],'EdgeColor',[1 1 1]);
for k1 = 1:size(y,2)
    ctr(k1,:) = bsxfun(@plus, hBar(k1).XData, hBar(k1).XOffset');     
    ydt(k1,:) = hBar(k1).YData;                    
end
ctrlR = [0.85 0.93 0.82 0.75 0.77 0.65 0.74 0.79 0.71 0.62];
stimR = [0.5 0.75 0.58 0.66 0.35 0.51 0.76 0.42 0.52 0.49];
ctrlU = [0.67 0.5 0.38 0.58 0.55 0.36 0.62 0.44 0.6 0.46];
stimU = [0.6 0.41 0.46 0.66 0.27 0.39 0.49 0.58 0.48 0.51];
 ctrlR(2,:) = ctr(1,1);
 stimR(2,:) = ctr(2,1);
  ctrlU(2,:) = ctr(1,2);
 stimU(2,:) = ctr(2,2);
hold on
hline(0.5,'--k');
for ugh = 1:10
    X(1) =  ctrlR(2,ugh);
    Y(1) =  ctrlR(1,ugh);
    X(2) =  stimR(2,ugh);
    Y(2) =  stimR(1,ugh);
    plot(X(1:2),Y,'Color','[0.7 0.7 0.7]')
end
for ugh = 1:10
    X(1) =  ctrlU(2,ugh);
    Y(1) =  ctrlU(1,ugh);
    X(2) =  stimU(2,ugh);
    Y(2) =  stimU(1,ugh);
    plot(X(1:2),Y,'Color','[0.7 0.7 0.7]')
end
e1 = errorbar(ctr(1,1), y(1,1),errorminus_vert(1,1),errorplus_vert(1,1),'LineStyle','none','LineWidth',2); 
e1.Color = '[1 0.51 0.51]'; 
hLine = line([ctr(1,1) - errorminus_horz(1), ctr(1,1) + errorplus_horz(1)], [y(1,1), y(1,1)], 'Color', get(e1, 'Color'), 'LineWidth', 2);
e12 = errorbar(ctr(2,1), y(1,2),errorminus_vert(2,1),errorplus_vert(2,1),'LineStyle','none','LineWidth',2); 
e12.Color = 'r'; 
hLine = line([ctr(2,1) - errorminus_horz(1), ctr(2,1) + errorplus_horz(1)], [y(1,2), y(1,2)], 'Color', get(e12, 'Color'), 'LineWidth', 2);
e2 = errorbar(ctr(1,2), y(2,1),errorminus_vert(1,2),errorplus_vert(1,2),'LineStyle','none','LineWidth',2); 
e2.Color = '[1 0.51 0.51]'; 
hLine = line([ctr(1,2) - errorminus_horz(1), ctr(1,2) + errorplus_horz(1)], [y(2,1), y(2,1)], 'Color', get(e2, 'Color'), 'LineWidth', 2);
e22 = errorbar(ctr(2,2), y(2,2),errorminus_vert(2,2),errorplus_vert(2,2),'LineStyle','none','LineWidth',2); 
e22.Color = 'r'; 
hLine = line([ctr(2,2) - errorminus_horz(1), ctr(2,2) + errorplus_horz(1)], [y(2,2), y(2,2)], 'Color', get(e22, 'Color'), 'LineWidth', 2);
ylim([0.2 1])
XTickLabel = ({'Rew','Unrew'});
set(gca,'xticklabel',XTickLabel,'box','off','FontSize',20)
ylabel('P(Repeat Decision)')
X(1) = ctr(1,1); X(2) = ctr(2,1); Y(1) = 0.96; Y(2) = 0.96;
plot(X(1:2),Y,'LineWidth',1.5,'Color','k')

%% time in center / trial time

ctrl_ws = [];
ctrl_ws = ispn_noconflowvolctrl_ratio;
L = length(ctrl_ws);
ctrl_ws(L+1:L+length(ispn_lowconflowvolctrl_ratio)) = ispn_lowconflowvolctrl_ratio;
L = length(ctrl_ws);
ctrl_ws(L+1:L+length(ispn_highconflowvolctrl_ratio)) = ispn_highconflowvolctrl_ratio;
stim_ws = [];
stim_ws = ispn_noconflowvolstim_ratio;
L = length(stim_ws);
stim_ws(L+1:L+length(ispn_lowconflowvolstim_ratio)) = ispn_lowconflowvolstim_ratio;
L = length(stim_ws);
stim_ws(L+1:L+length(ispn_highconflowvolstim_ratio)) = ispn_highconflowvolstim_ratio;

sigs = [];
[h,p] = kstest2(ctrl_ws,stim_ws);
sigs(1,1) = p;

f = figure;
ctr = []; ydt = [];
y = [mean(ctrl_ws); mean(stim_ws)];
a_vert = [std(ctrl_ws)/sqrt(length(ctrl_ws)); std(stim_ws)/sqrt(length(stim_ws))];
errorplus_vert = a_vert';
errorminus_vert=errorplus_vert;
x = [1:2];
a_horz = [0.1; 0.1];
errorplus_horz=a_horz';
errorminus_horz=errorplus_horz;
hBar = bar(x,y,'FaceColor',[1 1 1],'EdgeColor',[1 1 1]);
for k1 = 1:size(y,2)
    ctr(k1,:) = bsxfun(@plus, hBar(k1).XData, hBar(k1).XOffset');     
    ydt(k1,:) = hBar(k1).YData;                    
end
ctrlR = [0.15 0.18 0.19 0.18 0.26 0.2 0.2 0.21 0.22 0.23];
stimR = [0.25 0.22 0.23 0.23 0.3 0.29 0.22 0.24 0.24 0.29];
 ctrlR(2,:) = ctr(1,1);
 stimR(2,:) = ctr(1,2);
hold on
for ugh = 1:10
    X(1) =  ctrlR(2,ugh);
    Y(1) =  ctrlR(1,ugh);
    X(2) =  stimR(2,ugh);
    Y(2) =  stimR(1,ugh);
    plot(X(1:2),Y,'Color','[0.7 0.7 0.7]')
end
e1 = errorbar(ctr(1,1), y(1,1),errorminus_vert(1,1),errorplus_vert(1,1),'LineStyle','none','LineWidth',2); 
e1.Color = '[1 0.51 0.51]'; 
hLine = line([ctr(1,1) - errorminus_horz(1), ctr(1,1) + errorplus_horz(1)], [y(1,1), y(1,1)], 'Color', get(e1, 'Color'), 'LineWidth', 2);
e2 = errorbar(ctr(1,2), y(2,1),errorminus_vert(1,2),errorplus_vert(1,2),'LineStyle','none','LineWidth',2); 
e2.Color = 'r'; 
hLine = line([ctr(1,2) - errorminus_horz(1), ctr(1,2) + errorplus_horz(1)], [y(2,1), y(2,1)], 'Color', get(e2, 'Color'), 'LineWidth', 2);
ylim([0.1 0.35])
XTickLabel = ({'Ctrl','Stim'});
set(gca,'xticklabel',XTickLabel,'box','off','FontSize',20)
ylabel('Time in Center / Trial Time')
X(1) = ctr(1,1); X(2) = ctr(1,2); Y(1) = 0.33; Y(2) = 0.33;
plot(X(1:2),Y,'LineWidth',1.5,'Color','k')

%% decision time

ctrl_dt = [];
ctrl_dt = a2amin_no_dt;
L = length(ctrl_dt);
ctrl_dt(L+1:L+length(a2alow_no_dt)) = a2alow_no_dt;
L = length(ctrl_dt);
ctrl_dt(L+1:L+length(a2ahigh_no_dt)) = a2ahigh_no_dt;
stim_dt = [];
stim_dt = min_stim_dt;
L = length(stim_dt);
stim_dt(L+1:L+length(low_stim_dt)) = low_stim_dt;
L = length(stim_dt);
stim_dt(L+1:L+length(high_stim_dt)) = high_stim_dt;

control = [520 484 466 570 470 830 602 624 667 617];
stim = [731 1397 949 667 607 1212 2274 1248 1705 647];
control(2,:) = 0.75;
stim(2,:) = 1.25;

sigs = [];
[h,p] = kstest2(ctrl_dt,stim_dt);
sigs(1,1) = p;

f = figure;
if size(ctrl_dt,1) == 1
ctrl_dt = ctrl_dt';
stim_dt = stim_dt';
end
distributionPlot(ctrl_dt,'XValue',1,'histOri','left','color',[0.97 0.57 0.57],'widthDiv',[2 1]);
hold on
distributionPlot(stim_dt,'XValue',1,'histOri','right','color','r','widthDiv',[2 2])
for ugh = 1:10
    X(1) =  control(2,ugh);
    Y(1) =  control(1,ugh);
    X(2) =  stim(2,ugh);
    Y(2) =  stim(1,ugh);
    plot(X(1:2),Y,'Color','[0.7 0.7 0.7]')
end
YTickLabel = ({'1','2','3'});
set(gca,'xlim',[0 2],'xtick',[],'ytick',[1000:1000:3000],'yticklabel',YTickLabel,'fontsize',20)
ylim([0 3000])
ylabel('Decision Time (s)')