%% WSLS - OR

for environment = 1:6 %go through all environments doing the streamlined way to pull data out
    if environment == 1 %starting "if" statement to load each individual overlapped
        filename = load('C:\Users\julie\OneDrive\Documents\MATLAB\example data\NoConfLowVolControl_withDTs.mat');
    elseif environment == 2 %starting "if" statement to load each individual overlapped
        filename = load('C:\Users\julie\OneDrive\Documents\MATLAB\example data\LowConfLowVolControl_withDTs.mat');
    elseif environment == 3 %starting "if" statement to load each individual overlapped
        filename = load('C:\Users\julie\OneDrive\Documents\MATLAB\example data\HighConfLowVolControl_withDTs.mat');
    elseif environment == 4 %starting "if" statement to load each individual overlapped
        filename = load('C:\Users\julie\OneDrive\Documents\MATLAB\example data\NoConfHighVolControl_withDTs.mat');
    elseif environment == 5 %starting "if" statement to load each individual overlapped
        filename = load('C:\Users\julie\OneDrive\Documents\MATLAB\example data\LowConfHighVolControl_withDTs.mat');
    elseif environment == 6 %starting "if" statement to load each individual overlapped
        filename = load('C:\Users\julie\OneDrive\Documents\MATLAB\example data\HighConfHighVolControl_withDTs.mat');
    end %ending "if" statement for environments
    overlap = filename.overlap; %pull out cells
    for session = 1:length(overlap) %go through "overlap"
        data = overlap{1,session}; %pull out raw data
        wantedEM = []; %space
        wem = 1; %format
        for WEM = 2:length(data) %go through data
            if data(WEM,2) ~= data(WEM-1,2) && data(WEM-1,2) == 0 %if a turn is detected...
                wantedEM(wem) = data(WEM,2); %mark it
                wem=wem+1; %format
            end %ending "if" statement looking for turns
        end %ending "for" loop going through data
        choice = []; %space
        right = [5 6 7 8]; %right decision EMs
        for c = 1:length(wantedEM) %go through EMs
            if ismember(wantedEM(c),right) %if a right direction turn...
                choice(c) = 1; %mark 1
            else %if left...
                choice(c) = 0; %mark 0
            end %ending "if" statement for right direction turn
        end %ending "for" loop going through EMs
        outcome = []; %space
        reward = [1 3 5 7]; %reward decision EMs
        for r = 1:length(wantedEM) %go through EMs
            if ismember(wantedEM(r),reward) %if a reward direction turn...
                outcome(r) = 1; %mark 1
            else %if left...
                outcome(r) = 0; %mark 0
            end %ending "if" statement for reward direction turn
        end %ending "for" loop going through EMs
        winstay = []; loseswitch = []; %space
        WS = 1; LS = 1; %format
        for ws = 1:length(outcome)-1 %go through all outcomes but last one because we don't know what that one would have led to
            if outcome(ws) == 1 %if rewarded...
                if choice(ws) == choice(ws+1) %and the choice was repeated...
                    winstay(WS) = 1; %mark 1
                    WS=WS+1; %format
                else %if the choice was switched...
                    winstay(WS) = 0; %mark 0
                    WS=WS+1; %format
                end %ending "if" statement for repeat
            else %if unrewarded...
                if choice (ws) ~= choice(ws+1) %and the choice was switched...
                    loseswitch(LS) = 1; %mark 1
                    LS=LS+1; %format
                else %if the choice was repeated...
                    loseswitch(LS) = 0; %mark 0
                    LS=LS+1; %format
                end %ending "if" statement for switch
            end %ending "if" statement for rewarded
        end %ending "for" loop going through outcomes
        if environment == 1 && session == 1 %first session from first environment...
            savespot = 1; %we start in first spot
        else %if we are beyond the first session...
            savespot = savespot+1; %move down one spot for saving the data
        end %ending "if" statement figuring out where to save the data
        ALLDATA(savespot,1) = sum(winstay)/length(winstay); %save WS proportion
        ALLDATA(savespot,2) = sum(loseswitch)/length(loseswitch); %save LS proportion
        ALLDATA(savespot,3) = overlap{3,session}; %save mouse ID
    end %ending "for" loop going through overlap
end %ending "for" loop going through environments

%% Correct WS vs Incorrect WS - OR

ALLDATA = []; %space
for environment = 2:6 %go through all environments doing the streamlined way to pull data out
    if environment == 2 %starting "if" statement to load each individual overlapped
        filename = load('C:\Users\julie\OneDrive\Documents\MATLAB\example data\LowConfLowVolControl_withDTs.mat');
    elseif environment == 3 %starting "if" statement to load each individual overlapped
        filename = load('C:\Users\julie\OneDrive\Documents\MATLAB\example data\HighConfLowVolControl_withDTs.mat');
    elseif environment == 5 %starting "if" statement to load each individual overlapped
        filename = load('C:\Users\julie\OneDrive\Documents\MATLAB\example data\LowConfHighVolControl_withDTs.mat');
    elseif environment == 6 %starting "if" statement to load each individual overlapped
        filename = load('C:\Users\julie\OneDrive\Documents\MATLAB\example data\HighConfHighVolControl_withDTs.mat');
    end %ending "if" statement for environments
    overlap = filename.overlap; %pull out cells
    for session = 1:length(overlap) %go through "overlap"
        data = overlap{1,session}; %pull out raw data
        wantedEM = []; %space
        wem = 1; %format
        for WEM = 2:length(data) %go through data
            if data(WEM,2) ~= data(WEM-1,2) && data(WEM-1,2) == 0 %if a turn is detected...
                wantedEM(wem) = data(WEM,2); %mark it
                wem=wem+1; %format
            end %ending "if" statement looking for turns
        end %ending "for" loop going through data
        choice = []; %space
        right = [5 6 7 8]; %right decision EMs
        for c = 1:length(wantedEM) %go through EMs
            if ismember(wantedEM(c),right) %if a right direction turn...
                choice(c) = 1; %mark 1
            else %if left...
                choice(c) = 0; %mark 0
            end %ending "if" statement for right direction turn
        end %ending "for" loop going through EMs
        outcome = []; %space
        reward = [1 3 5 7]; %reward decision EMs
        for r = 1:length(wantedEM) %go through EMs
            if ismember(wantedEM(r),reward) %if a reward direction turn...
                outcome(r) = 1; %mark 1
            else %if left...
                outcome(r) = 0; %mark 0
            end %ending "if" statement for reward direction turn
        end %ending "for" loop going through EMs
        correct = []; %space
        cor = [1 2 5 6]; %cor decision EMs
        for COR = 1:length(wantedEM) %go through EMs
            if ismember(wantedEM(COR),cor) %if a correct direction turn...
                correct(COR) = 1; %mark 1
            else %if incorrect...
                correct(COR) = 0; %mark 0
            end %ending "if" statement for correct direction turn
        end %ending "for" loop going through EMs
        correct_winstay = []; incorrect_winstay = []; %space
        cws = 1; iws = 1; %format
        for ws = 1:length(outcome)-1 %go through all outcomes but last one because we don't know what that one would have led to
            if correct(ws) == 1 %if correct...
                if outcome(ws) == 1 %if rewarded...
                    if choice(ws) == choice(ws+1) %and the choice was repeated...
                        correct_winstay(cws) = 1; %mark 1
                        cws=cws+1; %format
                    else %if the choice was switched...
                        correct_winstay(cws) = 0; %mark 0
                        cws=cws+1; %format
                    end %ending "if" statement for repeat
                end %ending "if" statement looking for rewarded outcomes
            else %if incorrect......
                if outcome(ws) == 1 %if rewarded...
                    if choice(ws) == choice(ws+1) %and the choice was repeated...
                        incorrect_winstay(iws) = 1; %mark 1
                        iws=iws+1; %format
                    else %if the choice was switched...
                        incorrect_winstay(iws) = 0; %mark 0
                        iws=iws+1; %format
                    end %ending "if" statement for repeat
                end %ending "if" statement looking for rewarded outcomes
            end %ending "if" statement for correct vs incorrect
        end %ending "for" loop going through outcomes
        if environment == 2 && session == 1 %first session from first environment (startin with low conf though!)...
            savespot = 1; %we start in first spot
        else %if we are beyond the first session...
            savespot = savespot+1; %move down one spot for saving the data
        end %ending "if" statement figuring out where to save the data
        ALLDATA(savespot,1) = sum(correct_winstay)/length(correct_winstay); %save WS proportion
        ALLDATA(savespot,2) = sum(incorrect_winstay)/length(incorrect_winstay); %save LS proportion
        ALLDATA(savespot,3) = overlap{3,session}; %save mouse ID
    end %ending "for" loop going through overlap
end %ending "for" loop going through environments

%% N-back - OR

ALLDATA = []; %space 
for environment = 1:6 %go through all environments doing the streamlined way to pull data out
    if environment == 1 %starting "if" statement to load each individual overlapped
        filename = load('C:\Users\julie\OneDrive\Documents\MATLAB\example data\NoConfLowVolControl_withDTs.mat');
    elseif environment == 2 %starting "if" statement to load each individual overlapped
        filename = load('C:\Users\julie\OneDrive\Documents\MATLAB\example data\LowConfLowVolControl_withDTs.mat');
    elseif environment == 3 %starting "if" statement to load each individual overlapped
        filename = load('C:\Users\julie\OneDrive\Documents\MATLAB\example data\HighConfLowVolControl_withDTs.mat');
    elseif environment == 4 %starting "if" statement to load each individual overlapped
        filename = load('C:\Users\julie\OneDrive\Documents\MATLAB\example data\NoConfHighVolControl_withDTs.mat');
    elseif environment == 5 %starting "if" statement to load each individual overlapped
        filename = load('C:\Users\julie\OneDrive\Documents\MATLAB\example data\LowConfHighVolControl_withDTs.mat');
    elseif environment == 6 %starting "if" statement to load each individual overlapped
        filename = load('C:\Users\julie\OneDrive\Documents\MATLAB\example data\HighConfHighVolControl_withDTs.mat');
    end %ending "if" statement for environments
    overlap = filename.overlap; %pull out cells
    for session = 1:length(overlap) %go through "overlap"
        DATA = overlap{1,session}; %pull out raw data
        back = 5; %the number of trials back we are looking
        data = []; %re-setting "data" from it being used earlier in the code
        data.days = 1; %the number of sessions you are using
        data.b_lag.raw = NaN((back*2)+1,data.days); %making NaNs the size of # of trials back by the # of sessions
        wantedEM = []; %space
        wem = 1; %format
        for WEM = 2:length(DATA) %go through data
            if DATA(WEM,2) ~= DATA(WEM-1,2) && DATA(WEM-1,2) == 0 %if a turn is detected...
                wantedEM(wem) = DATA(WEM,2); %mark it
                wem=wem+1; %format
            end %ending "if" statement looking for turns
        end %ending "for" loop going through data
        rew = [1 3 5 7]; %rewarded EMs
        data.trials = length(wantedEM); %the relevant trials
        data.rewards = []; %place to store reward info
        for rrr = 1:length(wantedEM) %going through the EMs
            if ismember(wantedEM(rrr),rew) %if it is a rewarded trial...
                data.rewards(rrr) = 1; %tag it with a 1
            else %otherwise...
                data.rewards(rrr) = 0; %tag it with a 0
            end %ending "if" statement
        end %ending "for" loop
        rights = [5 6 7 8];
        data.choices = []; %place to store choice info
        for cc = 1:length(wantedEM) %going through the EMs
            if ismember(wantedEM(cc),rights) %if it is a right decision...
                data.choices(cc) = 1; %tag it with a 1
            else %otherwise...
                data.choices(cc) = 2; %tag it with a 2
            end %ending "if" statement
        end %ending "for" loop
        data.rewards = data.rewards'; %transposing the orientation
        data.choices = data.choices'; %transposing the orientation
        for zzz=1:data.days %going through all the sessions
            x_rew = zeros(data.trials,back); %a set of 0s to store rewarded info
            x_unr = zeros(data.trials,back); %a set of 0s to store unrewarded info
            for k=1:data.trials %going through the # of trials
                     for j=1:back %going through the # of trials back we are using
                        if k-j >=1 %if we are past a trial # where there are at least "n" trials back...
                            if data.rewards(k-j,zzz) == 1 %if that trial back is tagged with a 1 for rewarded... (doing this across each session)
                                 if data.choices(k-j,zzz) == 1 %if that trial back is tagged with a 1 for right...
                                     x_rew(k,j) = 1; %fill it in as a 1 in the rewarded info
                                 end %ending "if" statement for right
                                 if data.choices(k-j,zzz) == 2 %if that trial back is tagged with a 2 for left...
                                     x_rew(k,j) = -1; %fill it in as a -1 in the rewarded info
                                 end %ending "if" statement for left
                            end %ending "if" statement for rewarded
                            if data.rewards(k-j,zzz) == 0 %if that trial back is tagged with a 0 for unrewarded... (doing this for each session)
                                 if data.choices(k-j,zzz) == 1 %if that trial back is tagged with a 1 for right...
                                     x_unr(k,j) = 1; %fill it in as a 1 in the unrewarded info
                                 end %ending "if" statement for right
                                 if data.choices(k-j,zzz) == 2 %if that trial back is tagged with a 2 for left...
                                     x_unr(k,j) = -1; %fill it in as a -1 in the unrewarded info
                                 end %ending "if" statement for left
                            end %ending "if" statement for unrewarded
                        end %ending "if" statement for that specific trial back
                     end %ending "for" loop going through "n" back
            end %ending "for" loop going through number of trials
            choices2 = data.choices(:,zzz); %make a second place of choice information for all trials across all days
            choices2(find(choices2(:,1) == 2),1) = 0; %substitute any 2s for 0s (left)
            x_preds = [x_rew,x_unr]; %the predictors are the rewarded & unrewarded choice info
            data.b_lag.raw(:,zzz)=glmfit(x_preds,choices2,'binomial','constant','on'); %doing the regression
        end %ending "for" loop going through data.days
        if environment == 1 && session == 1 %first session from first environment...
            savespot = 1; %we start in first spot
        else %if we are beyond the first session...
            savespot = savespot+1; %move down one spot for saving the data
        end %ending "if" statement figuring out where to save the data
        ALLDATA(savespot,1:5) = data.b_lag.raw(2:6); %save rewarded coefficients
        ALLDATA(savespot,6) = overlap{3,session}; %save mouse ID
    end %ending "for" loop going through overlap
end %ending "for" loop going through environments

%% DT by WSLS - OR

ALLDATA = []; %space
for environment = 1:6 %go through all environments doing the streamlined way to pull data out
    if environment == 1 %starting "if" statement to load each individual overlapped
        filename = load('C:\Users\julie\OneDrive\Documents\MATLAB\example data\NoConfLowVolControl_withDTs.mat');
    elseif environment == 2 %starting "if" statement to load each individual overlapped
        filename = load('C:\Users\julie\OneDrive\Documents\MATLAB\example data\LowConfLowVolControl_withDTs.mat');
    elseif environment == 3 %starting "if" statement to load each individual overlapped
        filename = load('C:\Users\julie\OneDrive\Documents\MATLAB\example data\HighConfLowVolControl_withDTs.mat');
    elseif environment == 4 %starting "if" statement to load each individual overlapped
        filename = load('C:\Users\julie\OneDrive\Documents\MATLAB\example data\NoConfHighVolControl_withDTs.mat');
    elseif environment == 5 %starting "if" statement to load each individual overlapped
        filename = load('C:\Users\julie\OneDrive\Documents\MATLAB\example data\LowConfHighVolControl_withDTs.mat');
    elseif environment == 6 %starting "if" statement to load each individual overlapped
        filename = load('C:\Users\julie\OneDrive\Documents\MATLAB\example data\HighConfHighVolControl_withDTs.mat');
    end %ending "if" statement for environments
    overlap = filename.overlap; %pull out cells
    for session = 1:length(overlap) %go through "overlap"
        data = overlap{1,session}; %pull out raw data
        wantedEM = []; %space
        wem = 1; %format
        for WEM = 2:length(data) %go through data
            if data(WEM,2) ~= data(WEM-1,2) && data(WEM-1,2) == 0 %if a turn is detected...
                wantedEM(wem) = data(WEM,2); %mark it
                wem=wem+1; %format
            end %ending "if" statement looking for turns
        end %ending "for" loop going through data
        choice = []; %space
        right = [5 6 7 8]; %right decision EMs
        for c = 1:length(wantedEM) %go through EMs
            if ismember(wantedEM(c),right) %if a right direction turn...
                choice(c) = 1; %mark 1
            else %if left...
                choice(c) = 0; %mark 0
            end %ending "if" statement for right direction turn
        end %ending "for" loop going through EMs
        outcome = []; %space
        reward = [1 3 5 7]; %reward decision EMs
        for r = 1:length(wantedEM) %go through EMs
            if ismember(wantedEM(r),reward) %if a reward direction turn...
                outcome(r) = 1; %mark 1
            else %if left...
                outcome(r) = 0; %mark 0
            end %ending "if" statement for reward direction turn
        end %ending "for" loop going through EMs
        outcome_wsls = []; %space
        for owsls = 1:length(wantedEM)-1 %go through all but last trial because we don't know what that next one would be
            if outcome(owsls) == 1 %if rewarded...
                if choice(owsls) == choice(owsls+1) %and if repeat...
                    outcome_wsls(owsls) = 1; %mark 1
                else %and if switch...
                    outcome_wsls(owsls) = 2; %mark 2
                end %ending "if" statement for choice
            else %if unrewarded...
                if choice(owsls) == choice(owsls+1) %and if repeat...
                    outcome_wsls(owsls) = 4; %mark 4
                else %and if switch...
                    outcome_wsls(owsls) = 3; %mark 3
                end %ending "if" statement for choice
            end %ending "if" statement for outcome
        end %ending "for" loop going through EMs
        xy = []; %place to store location data
        xy(1,:) = data(:,13); %copy over the x info from data file
        xy(2,:) = data(:,14); %copy over y info from data file
        disp3=sqrt(diff2(xy(1,:),3).^2+diff2(xy(2,:),3).^2);  %equation from Eric for velocity
        turn1 = smooth(disp3,30)'; %smooth the velocity data
        turn1(2,:) = data(1:end-3,17); %fill in the second part of velocity data with the hall location
        stuff1 = NaN(length(turn1),1); %place to store hall 1
        stuff2 = NaN(length(turn1),1); %place to store hall 2
        stuff3 = NaN(length(turn1),1); %place to store hall 3
        stuff0 = NaN(length(turn1),1); %place to store center
        for splittingup = 1:length(turn1) %going through the velocity data
            if turn1(2,splittingup) == 1 %if the animal was in hall 1...
                stuff1(splittingup) = turn1(1,splittingup); %fill that in stuff 1
            elseif turn1(2,splittingup) == 2 %if the animal was in hall 2...
                stuff2(splittingup) = turn1(1,splittingup); %fill that in stuff 2
            elseif turn1(2,splittingup) == 3 %if the animal was in hall 3...
                stuff3(splittingup) = turn1(1,splittingup); %fill that in stuff 3
            elseif turn1(2,splittingup) == 0 %if the animal was in the center...
                stuff0(splittingup) = turn1(1,splittingup); %fill that in stuff 0
            end %ending "if" statement looking for location
        end %ending "for" loop going through velocity data
        stuff1(:,2) = data(1:end-3,1); %add in time info
        stuff2(:,2) = data(1:end-3,1); %add in time info
        stuff3(:,2) = data(1:end-3,1); %add in time info
        stuff0(:,2) = data(1:end-3,1); %add in time info
        turn1(3,:) = data(1:end-3,2); %fill in the EMs to turn1
        turn1(4,:) = 1:length(turn1); %fill in a spacer
        allempositions = []; %place to store all EM positions
        emspot = 1; %formatting
        ems = [1:8]; %any em
        for stars = 2:length(turn1) %going through the data
            if ismember(turn1(3,stars),ems) && turn1(3,stars-1) == 0 %if a turn is first detected...
                allempositions(emspot) = turn1(4,stars); %mark down that spot position number
                emspot = emspot + 1; %formatting
            end %ending "if" statement looking for ems
        end %ending "for" loop gooing through the data
        veltimes = []; %place to store a lot of the info
        veltimes(1,:) = turn1(1,:); %copy over the velocities
        veltimes(2,:) = data(1:end-3,1); %copy over the times
        veltimes(3,:) = 1:length(veltimes); %copy over the position # (to go with allempositions)
        cutoffvalues = []; %place to store the cutoff velocity for each turn
        hopethisworks = []; %going to try finding the min each time & then clearing it
        cutofftracker = 1; %formatting
        for merp = 1:length(allempositions)-1 %going through all the marked turns except for the last one
            hopethisworks = veltimes(1,allempositions(merp):allempositions(merp+1)); %temporarily copy over the vels between two turns
            sorted = sort(hopethisworks); %sort them from low to high
            cutoff = sorted(round(length(sorted)*0.15)); %find the velocity that cuts off the slowest 15% of values
            cutoffvalues(cutofftracker) = cutoff; %copy over the cutoff value
            cutofftracker = cutofftracker + 1; %formatting
            hopethisworks = []; %clearing for the next run
        end %ending "for" loop going through the empositions
        for eek = 1:length(allempositions)-1 %going through the ems
            for hoping = 1:length(veltimes) %going through veltimes
                if (allempositions(eek) < veltimes(3,hoping)) && (veltimes(3,hoping) < allempositions(eek + 1)) %if we are at a position between two marked turns...
                    if veltimes(1,hoping) < cutoffvalues(eek) %if the velocity falls below the cutoff of what we are calling "decision time"...
                        veltimes(4,hoping) = 1; %mark it as a 1 in the bottom row
                    end %ending "if" statement looking for the values falling below cutoff
                end %ending "if" statement looking at vels that fall between the EMs
            end %ending "for" loop going through veltimes
        end %ending "for" loop going through allempositions
        crossing = []; %storing when the threshold is crossed for the first time
        crossingdown = []; %slowing down
        downspot = 1; %formatting
        for firstinstance = 1:length(allempositions)-1 %going through the ems
            crossing(1,:) = veltimes(4,allempositions(firstinstance):allempositions(firstinstance + 1));  %temporarily copy over the 0s & 1s
            crossing(2,:) = veltimes(3,allempositions(firstinstance):allempositions(firstinstance + 1)); %temporarily copy over the positions
            for cutoffsearch = 2:length(crossing)-1 %going through crossing (second through second to last value)
                if crossing(1,cutoffsearch) == 1 && crossing(1,cutoffsearch - 1) == 0 %if the threshold is crossed going down...
                    crossing(1,cutoffsearch) = 2; %change its value to a 2
                end %ending "if" statement looking for when the threshold is crossed
            end %ending "for" loop going through crossing
            for separating = 1:length(crossing) %going through crossing
                if crossing(1,separating) == 2 %if the mouse crossed threshold going down...
                    crossingdown(downspot) = crossing(2,separating); %copy over the position
                    downspot = downspot + 1; %formatting
                end %ending "if" statement looking for slowing down
            end %ending "for" loop going through crossing
            downtime = min(crossingdown); %first time slowing down
            for finaltry = 1:length(crossing) %going through crossing
                if crossing(2,finaltry) == downtime %if it is down time...
                    crossing(1,finaltry) = 4; %change that value to a 4
                else %otherwise...
                    crossing(1,finaltry) = 0; %change everything to a 0
                end %ending "if" statement to find exact start/stop
            end %ending "for" loop going through crossing   
            veltimes(4,allempositions(firstinstance):allempositions(firstinstance + 1)) = crossing(1,:); %replace the veltimes with the cleared data
            crossing = []; crossingdown = []; %clear out this for next run
            downspot = 1; %formatting
        end %ending "for" loop going through to mark only first time a min is reached
        start = []; %decision time start
        t = 1; %formatting
        for testing = 1:length(veltimes) %going through veltimes
            if veltimes(4,testing) == 4 %if it is the start of decision time...
                start(t,1) = veltimes(1,testing); %copy over vel
                start(t,2) = veltimes(2,testing); %copy over time
                t = t + 1; %formatting
            end %ending "if" statement looking for starts
        end %ending "for" loop going through veltimes
        cutoffvalues = []; %place to store the cutoff velocity for each turn SPEEDING BACK UP
        hopethisworks = []; %going to try finding the min each time & then clearing it
        cutofftracker = 1; %formatting
        for merp = 1:length(allempositions)-1 %going through all the marked turns except for the last one
            hopethisworks = veltimes(1,allempositions(merp):allempositions(merp+1)); %temporarily copy over the vels between two turns
            sorted = sort(hopethisworks); %sort them from low to high
            cutoff = sorted(round(length(sorted)*0.25)); %find the velocity that cuts off the slowest 25% of values
            cutoffvalues(cutofftracker) = cutoff; %copy over the cutoff value
            cutofftracker = cutofftracker + 1; %formatting
            hopethisworks = []; %clearing for the next run
        end %ending "for" loop going through the empositions
        for eek = 1:length(allempositions)-1 %going through the ems
            for hoping = 1:length(veltimes) %going through veltimes
                if (allempositions(eek) < veltimes(3,hoping)) && (veltimes(3,hoping) < allempositions(eek + 1)) %if we are at a position between two marked turns...
                    if veltimes(1,hoping) < cutoffvalues(eek) %if the velocity falls below the cutoff of what we are calling "decision time"...
                        veltimes(4,hoping) = 1; %mark it as a 1 in the bottom row
                    end %ending "if" statement looking for the values falling below cutoff
                end %ending "if" statement looking at vels that fall between the EMs
            end %ending "for" loop going through veltimes
        end %ending "for" loop going through allempositions
        crossing = []; %storing when the threshold is crossed for the last time
        crossingup = []; %speeding up
        upspot = 1; %formatting
        for firstinstance = 1:length(allempositions)-1 %going through the ems
            crossing(1,:) = veltimes(4,allempositions(firstinstance):allempositions(firstinstance + 1));  %temporarily copy over the 0s & 1s
            crossing(2,:) = veltimes(3,allempositions(firstinstance):allempositions(firstinstance + 1)); %temporarily copy over the positions
            for cutoffsearch = 2:length(crossing)-1 %going through crossing (second through second to last value)
                if crossing(1,cutoffsearch) == 1 && crossing (1,cutoffsearch + 1) == 0 %if the threshold is crossed going up...
                    crossing(1,cutoffsearch) = 3; %change its value to a 3
                end %ending "if" statements looking for when the threshold is crossed
            end %ending "for" loop going through crossing
            for separating = 1:length(crossing) %going through crossing
                if crossing(1,separating) == 3 %if the mouse crossed threshold going up...
                    crossingup(upspot) = crossing(2,separating); %copy over the position
                    upspot = upspot + 1; %formatting
                end %ending "if" statement looking for up
            end %ending "for" loop going through crossing
            uptime = max(crossingup); %final time speeding up
            for finaltry = 1:length(crossing) %going through crossing
                if crossing(2,finaltry) == uptime %if it is up time...
                    crossing(1,finaltry) = 5; %change that value to a 5
                else %otherwise...
                    crossing(1,finaltry) = 0; %change everything to a 0
                end %ending "if" statement to find exact start/stop
            end %ending "for" loop going through crossing   
            veltimes(4,allempositions(firstinstance):allempositions(firstinstance + 1)) = crossing(1,:); %replace the veltimes with the cleared data
            crossing = []; crossingup = []; %clear out this for next run
            upspot = 1; %formatting
        end %ending "for" loop going through to mark only first time a min is reached
        stop = []; %decision time stop
        p = 1; %formatting
        for testing = 1:length(veltimes) %going through veltimes
            if veltimes(4,testing) == 5 %if it is the end of decision time...
                stop(p,1) = veltimes(1,testing); %copy over vel
                stop(p,2) = veltimes(2,testing); %copy over time
                p = p + 1; %formatting
            end %ending "if" statement looking for sta[rts/stops
        end %ending "for" loop going through veltimes
        decisiontimes = []; %storing DTs
        decisiontimes(:,1) = start(:,2); %copy over the DT starts
        decisiontimes(:,2) = stop(1:length(start),2); %copy over the DT stops (based off start length bc getting weird bug sometimes where there are more stops than starts)
        for calculating = 1:length(decisiontimes) %going through DTs
            decisiontimes(calculating,3) = decisiontimes(calculating,2) - decisiontimes(calculating,1); %calculating DTs
        end %ending "for" loop going through DTs
        L = length(outcome_wsls); %set L to EMs
        space = NaN(4,L); %NaN space
        for separate = 1:L %go through the number of trials...
            space(outcome_wsls(separate),separate) = decisiontimes(separate,3); %fill it all in
        end %ending "for" loop going through the number of trials
        nonNaNIndices = ~isnan(space(1,:));
        winstay = space(1,nonNaNIndices);
        nonNaNIndices = ~isnan(space(2,:));
        winswitch = space(2,nonNaNIndices);
        nonNaNIndices = ~isnan(space(3,:));
        loseswitch = space(3,nonNaNIndices);
        nonNaNIndices = ~isnan(space(4,:));
        losestay = space(4,nonNaNIndices);
        if environment == 1 && session == 1 %first session from first environment...
            savespot = 1; %we start in first spot
        else %if we are beyond the first session...
            savespot = savespot+1; %move down one spot for saving the data
        end %ending "if" statement figuring out where to save the data
        ALLDATA(savespot,1) = nanmedian(winstay); %store win stay
        ALLDATA(savespot,2) = nanmedian(winswitch); %store win switch
        ALLDATA(savespot,3) = nanmedian(loseswitch); %store lose switch
        ALLDATA(savespot,4) = nanmedian(losestay); %store lose stay
        ALLDATA(savespot,5) = overlap{3,session}; %save mouse ID
    end %ending "for" loop going through overlap
end %ending "for" loop going through environment

%% Speed by WSLS - OR

ALLDATA = []; %space
for environment = 1:6 %go through all environments doing the streamlined way to pull data out
    if environment == 1 %starting "if" statement to load each individual overlapped
        filename = load('C:\Users\julie\OneDrive\Documents\MATLAB\example data\NoConfLowVolControl_withDTs.mat');
    elseif environment == 2 %starting "if" statement to load each individual overlapped
        filename = load('C:\Users\julie\OneDrive\Documents\MATLAB\example data\LowConfLowVolControl_withDTs.mat');
    elseif environment == 3 %starting "if" statement to load each individual overlapped
        filename = load('C:\Users\julie\OneDrive\Documents\MATLAB\example data\HighConfLowVolControl_withDTs.mat');
    elseif environment == 4 %starting "if" statement to load each individual overlapped
        filename = load('C:\Users\julie\OneDrive\Documents\MATLAB\example data\NoConfHighVolControl_withDTs.mat');
    elseif environment == 5 %starting "if" statement to load each individual overlapped
        filename = load('C:\Users\julie\OneDrive\Documents\MATLAB\example data\LowConfHighVolControl_withDTs.mat');
    elseif environment == 6 %starting "if" statement to load each individual overlapped
        filename = load('C:\Users\julie\OneDrive\Documents\MATLAB\example data\HighConfHighVolControl_withDTs.mat');
    end %ending "if" statement for environments
    overlap = filename.overlap; %pull out cells
    for session = 1:length(overlap) %go through "overlap"
        data = overlap{1,session}; %pull out raw data
        wantedEM = []; %space
        wem = 1; %format
        for WEM = 2:length(data) %go through data
            if data(WEM,2) ~= data(WEM-1,2) && data(WEM-1,2) == 0 %if a turn is detected...
                wantedEM(wem) = data(WEM,2); %mark it
                wem=wem+1; %format
            end %ending "if" statement looking for turns
        end %ending "for" loop going through data
        choice = []; %space
        right = [5 6 7 8]; %right decision EMs
        for c = 1:length(wantedEM) %go through EMs
            if ismember(wantedEM(c),right) %if a right direction turn...
                choice(c) = 1; %mark 1
            else %if left...
                choice(c) = 0; %mark 0
            end %ending "if" statement for right direction turn
        end %ending "for" loop going through EMs
        outcome = []; %space
        reward = [1 3 5 7]; %reward decision EMs
        for r = 1:length(wantedEM) %go through EMs
            if ismember(wantedEM(r),reward) %if a reward direction turn...
                outcome(r) = 1; %mark 1
            else %if left...
                outcome(r) = 0; %mark 0
            end %ending "if" statement for reward direction turn
        end %ending "for" loop going through EMs
        outcome_wsls = []; %space
        for owsls = 1:length(wantedEM)-1 %go through all but last trial because we don't know what that next one would be
            if outcome(owsls) == 1 %if rewarded...
                if choice(owsls) == choice(owsls+1) %and if repeat...
                    outcome_wsls(owsls) = 1; %mark 1
                else %and if switch...
                    outcome_wsls(owsls) = 2; %mark 2
                end %ending "if" statement for choice
            else %if unrewarded...
                if choice(owsls) == choice(owsls+1) %and if repeat...
                    outcome_wsls(owsls) = 4; %mark 4
                else %and if switch...
                    outcome_wsls(owsls) = 3; %mark 3
                end %ending "if" statement for choice
            end %ending "if" statement for outcome
        end %ending "for" loop going through EMs
        xy = []; %place to store location data
        xy(1,:) = data(:,13); %copy over the x info from data file
        xy(2,:) = data(:,14); %copy over y info from data file
        disp3=sqrt(diff2(xy(1,:),3).^2+diff2(xy(2,:),3).^2)+3; %equation for velocity from Eric
        turn1 = smooth(disp3,30)'; %smooth the velocity data
        turn1(2,:) = data(1:end-3,17); %fill in the second part of velocity data with the hall location
        stuff1 = NaN(length(turn1),1); %place to store hall 1
        stuff2 = NaN(length(turn1),1); %place to store hall 2
        stuff3 = NaN(length(turn1),1); %place to store hall 3
        stuff0 = NaN(length(turn1),1); %place to store center
        for splittingup = 1:length(turn1) %going through the velocity data
            if turn1(2,splittingup) == 1 %if the animal was in hall 1...
                stuff1(splittingup) = turn1(1,splittingup); %fill that in stuff 1
            elseif turn1(2,splittingup) == 2 %if the animal wa sin hall 2...
                stuff2(splittingup) = turn1(1,splittingup); %fill that in stuff 2
            elseif turn1(2,splittingup) == 3 %if the animal was in hall 3...
                stuff3(splittingup) = turn1(1,splittingup); %fill that in stuff 3
            elseif turn1(2,splittingup) == 0 %if the animal was in the center...
                stuff0(splittingup) = turn1(1,splittingup); %fill that in stuff 0
            end %ending "if" statement looking for location
        end %ending "for" loop going through velocity data
        EM_vel = []; %place to store the EMs with the velocities
        EM_vel(:,1) = wantedEM; %pull out the trialwise EMs
        turn1(3,:) = data(1:end-3,2); %add EMs to the turn data
        emtracker = 1; %a way to make sure vels line up properly with EMs
        for velatentry = 2:length(turn1) %going through the turn data
            if turn1(3,velatentry) ~= 0 && turn1(3,velatentry-1) == 0 %if an EM turn was detected...
                EM_vel(emtracker,2) = turn1(1,velatentry); %copy over the velocity at the point the EM was detected
                emtracker = emtracker + 1; %slide down one
            end %ending "if" statement looking for EMs
        end %ending "for" loop going through turn data
        L = length(outcome_wsls); %set trial number
        space = NaN(4,L); %NaN space
        for separate = 1:L %go through the number of trials...
            space(outcome_wsls(separate),separate) = EM_vel(separate+1,2); %fill it all in (one offset because we don't have WSLS for trial 1)
        end %ending "for" loop going through the number of trials
        nonNaNIndices = ~isnan(space(1,:));
        winstay = space(1,nonNaNIndices);
        nonNaNIndices = ~isnan(space(2,:));
        winswitch = space(2,nonNaNIndices);
        nonNaNIndices = ~isnan(space(3,:));
        loseswitch = space(3,nonNaNIndices);
        nonNaNIndices = ~isnan(space(4,:));
        losestay = space(4,nonNaNIndices);
        if environment == 1 && session == 1 %first session from first environment...
            savespot = 1; %we start in first spot
        else %if we are beyond the first session...
            savespot = savespot+1; %move down one spot for saving the data
        end %ending "if" statement figuring out where to save the data
        ALLDATA(savespot,1) = nanmean(winstay)/3; %store win stay
        ALLDATA(savespot,2) = nanmean(winswitch)/3; %store win switch
        ALLDATA(savespot,3) = nanmean(loseswitch)/3; %store lose switch
        ALLDATA(savespot,4) = nanmean(losestay)/3; %store lose stay
        ALLDATA(savespot,5) = overlap{3,session}; %save mouse ID
    end %ending "for" loop going through overlap
end %ending "for" loop going through environments

%% R > C > 0 - OR

ALLDATA = []; %space 
for environment = 1:6 %go through all environments doing the streamlined way to pull data out
    if environment == 1 %starting "if" statement to load each individual overlapped
        filename = load('C:\Users\julie\OneDrive\Documents\MATLAB\example data\NoConfLowVolControl_withDTs.mat');
    elseif environment == 2 %starting "if" statement to load each individual overlapped
        filename = load('C:\Users\julie\OneDrive\Documents\MATLAB\example data\LowConfLowVolControl_withDTs.mat');
    elseif environment == 3 %starting "if" statement to load each individual overlapped
        filename = load('C:\Users\julie\OneDrive\Documents\MATLAB\example data\HighConfLowVolControl_withDTs.mat');
    elseif environment == 4 %starting "if" statement to load each individual overlapped
        filename = load('C:\Users\julie\OneDrive\Documents\MATLAB\example data\NoConfHighVolControl_withDTs.mat');
    elseif environment == 5 %starting "if" statement to load each individual overlapped
        filename = load('C:\Users\julie\OneDrive\Documents\MATLAB\example data\LowConfHighVolControl_withDTs.mat');
    elseif environment == 6 %starting "if" statement to load each individual overlapped
        filename = load('C:\Users\julie\OneDrive\Documents\MATLAB\example data\HighConfHighVolControl_withDTs.mat');
    end %ending "if" statement for environments
    overlap = filename.overlap; %pull out cells
    for session = 1:length(overlap) %go through "overlap"
        DATA = overlap{1,session}; %pull out raw data
        back = 5; %the number of trials back we are looking
        data = []; %re-setting "data" from it being used earlier in the code
        data.days = 1; %running one session
        data.b_lag.raw = NaN((back*2)+1,data.days); %making NaNs the size of # of trials back by the # of sessions
        wantedEM = []; %space
        wem = 1; %format
        for WEM = 2:length(DATA) %go through data
            if DATA(WEM,2) ~= DATA(WEM-1,2) && DATA(WEM-1,2) == 0 %if a turn is detected...
                wantedEM(wem) = DATA(WEM,2); %mark it
                wem=wem+1; %format
            end %ending "if" statement looking for turns
        end %ending "for" loop going through data
        rew = [1 3 5 7]; %rewarded EMs
        data.trials = length(wantedEM); %the relevant trials
        data.rewards = []; %place to store reward info
        for number = 1:length(wantedEM) %going through the EMs
            if ismember(wantedEM(number),rew) %if it is a rewarded trial...
                data.rewards(number) = 1; %tag it with a 1
            else %otherwise...
                data.rewards(number) = 0; %tag it with a 0
            end %ending "if" statement
        end %ending "for" loop
        RIGHTS = [5 6 7 8];
        data.choices = []; %place to store choice info
        for cc = 1:length(wantedEM) %going through the EMs
            if ismember(wantedEM(cc),RIGHTS) %if it is a right decision...
                data.choices(cc) = 1; %tag it with a 1
            else %otherwise...
                data.choices(cc) = 2; %tag it with a 2
            end %ending "if" statement
        end %ending "for" loop
        data.rewards = data.rewards'; %transposing the orientation
        data.choices = data.choices'; %transposing the orientation
        for zzz=1 %going through all the sessions
            x_rew = zeros(data.trials,back); %a set of 0s to store reward info
            x_choice = zeros(data.trials,back); %a set of 0s to store choice info
            for k=1:data.trials %going through the # of trials
                     for j=1:back %going through the # of trials back we are using
                        if k-j >=1 %if we are past a trial # where there are at least "n" trials back...
                            if data.rewards(k-j,zzz) == 1 %if that trial back is tagged with a 1 for rewarded... (doing this across each session)
                                 if data.choices(k-j,zzz) == 1 %if that trial back is tagged with a 1 for right...
                                     x_rew(k,j) = 1; %fill it in as a 1 in the rewarded info
                                 end %ending "if" statement for right
                                 if data.choices(k-j,zzz) == 2 %if that trial back is tagged with a 2 for left...
                                     x_rew(k,j) = -1; %fill it in as a -1 in the rewarded info
                                 end %ending "if" statement for left
                            end %ending "if" statement for rewarded
                            if data.rewards(k-j,zzz) == 0 %if that trial back is tagged with a 0 for unrewarded... (doing this for each session)
                                 if data.choices(k-j,zzz) == 1 %if that trial back is tagged with a 1 for right...
                                     x_rew(k,j) = -1; %fill it in as a -1 in the reward info
                                 end %ending "if" statement for right
                                 if data.choices(k-j,zzz) == 2 %if that trial back is tagged with a 2 for left...
                                     x_rew(k,j) = 1; %fill it in as a 1 in the reward info
                                 end %ending "if" statement for left
                            end %ending "if" statement for unrewarded
                        end %ending "if" statement for that specific trial back
                     end %ending "for" loop going through "n" back
            end %ending "for" loop going through number of trials
            for k=1:data.trials %going through the # of trials
                     for j=1:back %going through the # of trials back we are using
                        if k-j >=1 %if we are past a trial # where there are at least "n" trials back...
                            if data.choices(k-j,zzz) == 1 %if that trial back is tagged with a 1 for right
                                 if data.choices(k-j,zzz) == 1 %if that trial back is tagged with a 1 for right...
                                     x_choice(k,j) = 1; %fill it in as a 1 in the rewarded info
                                 end %ending "if" statement for right
                                 if data.choices(k-j,zzz) == 2 %if that trial back is tagged with a 2 for left...
                                     x_choice(k,j) = -1; %fill it in as a -1 in the rewarded info
                                 end %ending "if" statement for left
                            end %ending "if" statement for rewarded
                            if data.choices(k-j,zzz) == 2 %if that trial back is tagged with a 2 for left... (doing this for each session)
                                 if data.choices(k-j,zzz) == 1 %if that trial back is tagged with a 1 for right...
                                     x_choice(k,j) = 1; %fill it in as a -1 in the choice info
                                 end %ending "if" statement for right
                                 if data.choices(k-j,zzz) == 2 %if that trial back is tagged with a 2 for left...
                                     x_choice(k,j) = -1; %fill it in as a 1 in the choice info
                                 end %ending "if" statement for left
                            end %ending "if" statement for unrewarded
                        end %ending "if" statement for that specific trial back
                     end %ending "for" loop going through "n" back
            end %ending "for" loop going through number of trials
            choices2 = data.choices(:,zzz); %make a second place of choice information for all trials across all days
            choices2(find(choices2(:,1) == 2),1) = 0; %substitute any 2s for 0s (left)
            x_preds = [x_rew,x_choice]; %the predictors are the rewarded & choice info
            data.b_lag.raw(:,zzz)=glmfit(x_preds,choices2,'binomial','constant','on'); %doing the regression
        end %ending "for" loop going through all the sessions
        rewardregression = data.b_lag.raw(2); %printing info
        choiceregression = data.b_lag.raw(7); %printing info
        if environment == 1 && session == 1 %first session from first environment...
            savespot = 1; %we start in first spot
        else %if we are beyond the first session...
            savespot = savespot+1; %move down one spot for saving the data
        end %ending "if" statement figuring out where to save the data
        ALLDATA(savespot,1) = rewardregression; %store rew
        ALLDATA(savespot,2) = choiceregression; %store choice
        ALLDATA(savespot,3) = overlap{3,session}; %store ID
    end %ending "for" loop going through overlap
end %ending "for" loop going through environments

%% Training bias - bias

for session = 1:10 %we have 10 training sessions
    for bymouse = 1:length(training) %this will go through each # mouse we have
        data = []; %re-setting "data" from it being used earlier in the code
        data.days = 1; %the number of sessions you are using
        data2 = training{bymouse,session}; %pull out raw data
        EM = data2(:,2); %makes a variable for just EMs
        wantedEM = []; %making an open vector for EMs that we want
        for q = 2:length(EM) %starting with second value & running through all EMs
            if EM(q)~=EM(q-1) %if an EM is different from the one preceding it...
            wantedEM = [wantedEM,EM(q)]; %...include those EMs in the wantedEM vector
            end %ending the "if" statement
        end %ending the "for" loop
        wantedEM(ismember(wantedEM,[0,11,12])) = []; %retains only EMs pertaining to turns
        rew = [1 3 5 7]; %rewarded EMs
        data.trials = length(wantedEM); %the relevant trials
        data.rewards = []; %place to store reward info
        for rrr = 1:length(wantedEM) %going through the EMs
            if ismember(wantedEM(rrr),rew) %if it is a rewarded trial...
                data.rewards(rrr) = 1; %tag it with a 1
            else %otherwise...
                data.rewards(rrr) = 0; %tag it with a 0
            end %ending "if" statement
        end %ending "for" loop
        rights = [5 6 7 8];
        data.choices = []; %place to store choice info
        for cc = 1:length(wantedEM) %going through the EMs
            if ismember(wantedEM(cc),rights) %if it is a right decision...
                data.choices(cc) = 1; %tag it with a 1
            else %otherwise...
                data.choices(cc) = 2; %tag it with a 2
            end %ending "if" statement
        end %ending "for" loop
        data.rewards = data.rewards'; %transposing the orientation
        data.choices = data.choices'; %transposing the orientation
        data.bs.raw = NaN(data.days, 3); %making NaNs for the number of days we are running for the number of betas being used
        for i=1:data.days %going through the number of days
            LLfit = @(betas) sum(-fit_Qvalue_V2(data.choices(:,i), data.rewards(:,i), betas, data.trials)); %use the choice, reward, & trial info to fit betas established in fit_Qvalue function
            [data.bs.raw(i,:), bvals] = fmincon(LLfit,[0.5, 5, 0],[],[],[],[],[0, 0.001, -10],[1, Inf, 10]); %establishing the limits for the values of the different betas
        end %ending "for" loop going through the days
        learning_rate = data.bs.raw(1); %printing the learning rate
        explore_exploit = data.bs.raw(2); %printing the beta
        Q_bias = data.bs.raw(3); %printing the Q bias term
        if Q_bias < 0 %if we have a negative value...
            Q_bias = abs(Q_bias); %make absolute value
        end %ending "if" statement fixing for diff. directions
        bias(bymouse,session) = Q_bias; %fill it in the matrix
    end %ending "for" loop going by mouse
end %ending "for" loop going through the training data

%% Q & Reg Bias - bias

for environment = 1:6 %go through all environments doing the streamlined way to pull data out
    if environment == 1 %starting "if" statement to load each individual overlapped
        filename = load('C:\Users\julie\OneDrive\Documents\MATLAB\example data\NoConfLowVolControl_withDTs.mat');
    elseif environment == 2 %starting "if" statement to load each individual overlapped
        filename = load('C:\Users\julie\OneDrive\Documents\MATLAB\example data\LowConfLowVolControl_withDTs.mat');
    elseif environment == 3 %starting "if" statement to load each individual overlapped
        filename = load('C:\Users\julie\OneDrive\Documents\MATLAB\example data\HighConfLowVolControl_withDTs.mat');
    elseif environment == 4 %starting "if" statement to load each individual overlapped
        filename = load('C:\Users\julie\OneDrive\Documents\MATLAB\example data\NoConfHighVolControl_withDTs.mat');
    elseif environment == 5 %starting "if" statement to load each individual overlapped
        filename = load('C:\Users\julie\OneDrive\Documents\MATLAB\example data\LowConfHighVolControl_withDTs.mat');
    elseif environment == 6 %starting "if" statement to load each individual overlapped
        filename = load('C:\Users\julie\OneDrive\Documents\MATLAB\example data\HighConfHighVolControl_withDTs.mat');
    end %ending "if" statement for environments
    overlap = filename.overlap; %pull out cells
    for session = 1:length(overlap) %go through "overlap"
        data = []; %re-setting "data" from it being used earlier in the code
        data.days = 1; %the number of sessions you are using
        data2 = overlap{1,session}; %pull out raw data
        EM = data2(:,2); %makes a variable for just EMs
        wantedEM = []; %making an open vector for EMs that we want
        for q = 2:length(EM) %starting with second value & running through all EMs
            if EM(q)~=EM(q-1) %if an EM is different from the one preceding it...
            wantedEM = [wantedEM,EM(q)]; %...include those EMs in the wantedEM vector
            end %ending the "if" statement
        end %ending the "for" loop
        wantedEM(ismember(wantedEM,[0,11,12])) = []; %retains only EMs pertaining to turns
        rew = [1 3 5 7]; %rewarded EMs
        data.trials = length(wantedEM); %the relevant trials
        data.rewards = []; %place to store reward info
        for rrr = 1:length(wantedEM) %going through the EMs
            if ismember(wantedEM(rrr),rew) %if it is a rewarded trial...
                data.rewards(rrr) = 1; %tag it with a 1
            else %otherwise...
                data.rewards(rrr) = 0; %tag it with a 0
            end %ending "if" statement
        end %ending "for" loop
        rights = [5 6 7 8];
        data.choices = []; %place to store choice info
        for cc = 1:length(wantedEM) %going through the EMs
            if ismember(wantedEM(cc),rights) %if it is a right decision...
                data.choices(cc) = 1; %tag it with a 1
            else %otherwise...
                data.choices(cc) = 2; %tag it with a 2
            end %ending "if" statement
        end %ending "for" loop
        data.rewards = data.rewards'; %transposing the orientation
        data.choices = data.choices'; %transposing the orientation
        data.bs.raw = NaN(data.days, 3); %making NaNs for the number of days we are running for the number of betas being used
        for i=1:data.days %going through the number of days
            LLfit = @(betas) sum(-fit_Qvalue_V2(data.choices(:,i), data.rewards(:,i), betas, data.trials)); %use the choice, reward, & trial info to fit betas established in fit_Qvalue function
            [data.bs.raw(i,:), bvals] = fmincon(LLfit,[0.5, 5, 0],[],[],[],[],[0, 0.001, -10],[1, Inf, 10]); %establishing the limits for the values of the different betas
        end %ending "for" loop going through the days
        learning_rate = data.bs.raw(1); %printing the learning rate
        explore_exploit = data.bs.raw(2); %printing the beta
        Q_bias = data.bs.raw(3); %printing the Q bias term
        if Q_bias < 0 %if we have a negative value...
            Q_bias = abs(Q_bias); %make absolute value
        end %ending "if" statement fixing for diff. directions
        if environment == 1 && session == 1 %first session from first environment...
            savespot = 1; %we start in first spot
        else %if we are beyond the first session...
            savespot = savespot+1; %move down one spot for saving the data
        end %ending "if" statement figuring out where to save the data
        back = 5; %the number of trials back we are looking
        data = []; %re-setting "data" from it being used earlier in the code
        data.days = 1; %the number of sessions you are using
        data.b_lag.raw = NaN((back*2)+1,data.days); %making NaNs the size of # of trials back by the # of sessions
        rew = [1 3 5 7]; %rewarded EMs
        data.trials = length(wantedEM); %the relevant trials
        data.rewards = []; %place to store reward info
        for rrr = 1:length(wantedEM) %going through the EMs
            if ismember(wantedEM(rrr),rew) %if it is a rewarded trial...
                data.rewards(rrr) = 1; %tag it with a 1
            else %otherwise...
                data.rewards(rrr) = 0; %tag it with a 0
            end %ending "if" statement
        end %ending "for" loop
        rights = [5 6 7 8];
        data.choices = []; %place to store choice info
        for cc = 1:length(wantedEM) %going through the EMs
            if ismember(wantedEM(cc),rights) %if it is a right decision...
                data.choices(cc) = 1; %tag it with a 1
            else %otherwise...
                data.choices(cc) = 2; %tag it with a 2
            end %ending "if" statement
        end %ending "for" loop
        data.rewards = data.rewards'; %transposing the orientation
        data.choices = data.choices'; %transposing the orientation
        for zzz=1:data.days %going through all the sessions
            x_rew = zeros(data.trials,back); %a set of 0s to store rewarded info
            x_unr = zeros(data.trials,back); %a set of 0s to store unrewarded info
            for k=1:data.trials %going through the # of trials
                     for j=1:back %going through the # of trials back we are using
                        if k-j >=1 %if we are past a trial # where there are at least "n" trials back...
                            if data.rewards(k-j,zzz) == 1 %if that trial back is tagged with a 1 for rewarded... (doing this across each session)
                                 if data.choices(k-j,zzz) == 1 %if that trial back is tagged with a 1 for right...
                                     x_rew(k,j) = 1; %fill it in as a 1 in the rewarded info
                                 end %ending "if" statement for right
                                 if data.choices(k-j,zzz) == 2 %if that trial back is tagged with a 2 for left...
                                     x_rew(k,j) = -1; %fill it in as a -1 in the rewarded info
                                 end %ending "if" statement for left
                            end %ending "if" statement for rewarded
                            if data.rewards(k-j,zzz) == 0 %if that trial back is tagged with a 0 for unrewarded... (doing this for each session)
                                 if data.choices(k-j,zzz) == 1 %if that trial back is tagged with a 1 for right...
                                     x_unr(k,j) = 1; %fill it in as a 1 in the unrewarded info
                                 end %ending "if" statement for right
                                 if data.choices(k-j,zzz) == 2 %if that trial back is tagged with a 2 for left...
                                     x_unr(k,j) = -1; %fill it in as a -1 in the unrewarded info
                                 end %ending "if" statement for left
                            end %ending "if" statement for unrewarded
                        end %ending "if" statement for that specific trial back
                     end %ending "for" loop going through "n" back
            end %ending "for" loop going through number of trials
            choices2 = data.choices(:,zzz); %make a second place of choice information for all trials across all days
            choices2(find(choices2(:,1) == 2),1) = 0; %substitute any 2s for 0s (left)
            x_preds = [x_rew,x_unr]; %the predictors are the rewarded & unrewarded choice info
            data.b_lag.raw(:,zzz)=glmfit(x_preds,choices2,'binomial','constant','on'); %doing the regression
        end %ending "for" loop going through all the sessions
        reg_bias = data.b_lag.raw(1); %regression bias
        if reg_bias < 0 %if negative...
            reg_bias = abs(reg_bias); %take the absolute value
        end %ending "if" statement for if negative
        ALLDATA(savespot,1) = Q_bias; %save Q bias
        ALLDATA(savespot,2) = reg_bias; %save regression bias
        ALLDATA(savespot,3) = overlap{3,session}; %save mouse ID
    end %ending "for" loop going through overlap
end %ending "for" loop going through environment

%% lose-stay by bias - bias

ALLDATA = []; %space
rightmouse = [1 3 10 5 7 2 6 101]; %right mice
for environment = 1:6 %go through all environments doing the streamlined way to pull data out
    if environment == 1 %starting "if" statement to load each individual overlapped
        filename = load('C:\Users\julie\OneDrive\Documents\MATLAB\example data\NoConfLowVolControl_withDTs.mat');
    elseif environment == 2 %starting "if" statement to load each individual overlapped
        filename = load('C:\Users\julie\OneDrive\Documents\MATLAB\example data\LowConfLowVolControl_withDTs.mat');
    elseif environment == 3 %starting "if" statement to load each individual overlapped
        filename = load('C:\Users\julie\OneDrive\Documents\MATLAB\example data\HighConfLowVolControl_withDTs.mat');
    elseif environment == 4 %starting "if" statement to load each individual overlapped
        filename = load('C:\Users\julie\OneDrive\Documents\MATLAB\example data\NoConfHighVolControl_withDTs.mat');
    elseif environment == 5 %starting "if" statement to load each individual overlapped
        filename = load('C:\Users\julie\OneDrive\Documents\MATLAB\example data\LowConfHighVolControl_withDTs.mat');
    elseif environment == 6 %starting "if" statement to load each individual overlapped
        filename = load('C:\Users\julie\OneDrive\Documents\MATLAB\example data\HighConfHighVolControl_withDTs.mat');
    end %ending "if" statement for environments
    overlap = filename.overlap; %pull out cells
    for session = 1:length(overlap) %go through "overlap"
        data = []; %clearing from regression
        data = overlap{1,session}; %pull out raw data
        wantedEM = []; %space
        wem = 1; %format
        for WEM = 2:length(data) %go through data
            if data(WEM,2) ~= data(WEM-1,2) && data(WEM-1,2) == 0 %if a turn is detected...
                wantedEM(wem) = data(WEM,2); %mark it
                wem=wem+1; %format
            end %ending "if" statement looking for turns
        end %ending "for" loop going through data
        choice = []; %space
        right = [5 6 7 8]; %right decision EMs
        for c = 1:length(wantedEM) %go through EMs
            if ismember(wantedEM(c),right) %if a right direction turn...
                choice(c) = 1; %mark 1
            else %if left...
                choice(c) = 0; %mark 0
            end %ending "if" statement for right direction turn
        end %ending "for" loop going through EMs
        outcome = []; %space
        reward = [1 3 5 7]; %reward decision EMs
        for r = 1:length(wantedEM) %go through EMs
            if ismember(wantedEM(r),reward) %if a reward direction turn...
                outcome(r) = 1; %mark 1
            else %if left...
                outcome(r) = 0; %mark 0
            end %ending "if" statement for reward direction turn
        end %ending "for" loop going through EMs
        preferred = []; %space
        if ismember(overlap{3,session},rightmouse) %if preferred direction is right...
            pref = [5 6 7 8]; %set pref
        else %if left...
            pref = [1 2 3 4]; %set pref
        end %ending "if" statement setting preference
        for p = 1:length(wantedEM) %go through EMs
            if ismember(wantedEM(p),pref) %if preferred...
                preferred(p) = 1; %mark 1
            else %if non-preferred...
                preferred(p) = 0; %mark 0
            end %ending "if" statement setting preference binary
        end %ending "for" loop going through EMs
        losestay = []; %space
        LS = 1; %format
        for ws = 1:length(outcome)-1 %go through all outcomes but last one because we don't know what that one would have led to
            if outcome(ws) == 0 %if unrewarded...
                if choice(ws) == choice(ws+1) %and the choice was repeated...
                    losestay(1,LS) = 1; %mark 1
                    losestay(2,LS) = preferred(ws+1); %mark if the choice was pref or not
                    LS=LS+1; %format
                end %ending "if" statement for switch
            end %ending "if" statement for rewarded
        end %ending "for" loop going through outcomes
        if environment == 1 && session == 1 %first session from first environment...
            savespot = 1; %we start in first spot
        else %if we are beyond the first session...
            savespot = savespot+1; %move down one spot for saving the data
        end %ending "if" statement figuring out where to save the data
        ALLDATA(savespot,1) = sum(losestay(2,:))/length(losestay); %save proportion of lose-stay that were into pref direction
        ALLDATA(savespot,2) = overlap{3,session}; %save mouse ID
    end %ending "for" loop going through overlap
end %ending "for" loop going through environments

%% win-switch bias

ALLDATA = []; %space
rightmouse = [1 3 10 5 7 2 6 101]; %right mice
for environment = 1:6 %go through all environments doing the streamlined way to pull data out
    if environment == 1 %starting "if" statement to load each individual overlapped
        filename = load('C:\Users\julie\OneDrive\Documents\MATLAB\example data\NoConfLowVolControl_withDTs.mat');
    elseif environment == 2 %starting "if" statement to load each individual overlapped
        filename = load('C:\Users\julie\OneDrive\Documents\MATLAB\example data\LowConfLowVolControl_withDTs.mat');
    elseif environment == 3 %starting "if" statement to load each individual overlapped
        filename = load('C:\Users\julie\OneDrive\Documents\MATLAB\example data\HighConfLowVolControl_withDTs.mat');
    elseif environment == 4 %starting "if" statement to load each individual overlapped
        filename = load('C:\Users\julie\OneDrive\Documents\MATLAB\example data\NoConfHighVolControl_withDTs.mat');
    elseif environment == 5 %starting "if" statement to load each individual overlapped
        filename = load('C:\Users\julie\OneDrive\Documents\MATLAB\example data\LowConfHighVolControl_withDTs.mat');
    elseif environment == 6 %starting "if" statement to load each individual overlapped
        filename = load('C:\Users\julie\OneDrive\Documents\MATLAB\example data\HighConfHighVolControl_withDTs.mat');
    end %ending "if" statement for environments
    overlap = filename.overlap; %pull out cells
    for session = 1:length(overlap) %go through "overlap"
        data = []; %clearing from regression
        data = overlap{1,session}; %pull out raw data
        wantedEM = []; %space
        wem = 1; %format
        for WEM = 2:length(data) %go through data
            if data(WEM,2) ~= data(WEM-1,2) && data(WEM-1,2) == 0 %if a turn is detected...
                wantedEM(wem) = data(WEM,2); %mark it
                wem=wem+1; %format
            end %ending "if" statement looking for turns
        end %ending "for" loop going through data
        choice = []; %space
        right = [5 6 7 8]; %right decision EMs
        for c = 1:length(wantedEM) %go through EMs
            if ismember(wantedEM(c),right) %if a right direction turn...
                choice(c) = 1; %mark 1
            else %if left...
                choice(c) = 0; %mark 0
            end %ending "if" statement for right direction turn
        end %ending "for" loop going through EMs
        outcome = []; %space
        reward = [1 3 5 7]; %reward decision EMs
        for r = 1:length(wantedEM) %go through EMs
            if ismember(wantedEM(r),reward) %if a reward direction turn...
                outcome(r) = 1; %mark 1
            else %if left...
                outcome(r) = 0; %mark 0
            end %ending "if" statement for reward direction turn
        end %ending "for" loop going through EMs
        preferred = []; %space
        if ismember(overlap{3,session},rightmouse) %if preferred direction is right...
            pref = [5 6 7 8]; %set pref
        else %if left...
            pref = [1 2 3 4]; %set pref
        end %ending "if" statement setting preference
        for p = 1:length(wantedEM) %go through EMs
            if ismember(wantedEM(p),pref) %if preferred...
                preferred(p) = 1; %mark 1
            else %if non-preferred...
                preferred(p) = 0; %mark 0
            end %ending "if" statement setting preference binary
        end %ending "for" loop going through EMs
        winswitch = []; %space
        WS = 1; %format
        for ws = 1:length(outcome)-1 %go through all outcomes but last one because we don't know what that one would have led to
            if outcome(ws) == 1 %if rewarded...
                if choice(ws) ~= choice(ws+1) %and the choice was switched...
                    winswitch(1,WS) = 1; %mark 1
                    winswitch(2,WS) = preferred(ws+1); %mark if the choice was pref or not
                    WS=WS+1; %format
                end %ending "if" statement for switch
            end %ending "if" statement for rewarded
        end %ending "for" loop going through outcomes
        if environment == 1 && session == 1 %first session from first environment...
            savespot = 1; %we start in first spot
        else %if we are beyond the first session...
            savespot = savespot+1; %move down one spot for saving the data
        end %ending "if" statement figuring out where to save the data
        ALLDATA(savespot,1) = sum(winswitch(2,:))/length(winswitch); %save proportion of win-switch that were into pref direction
        ALLDATA(savespot,2) = overlap{3,session}; %save mouse ID
    end %ending "for" loop going through overlap
end %ending "for" loop going through environments

%% perseverative errors - bias

ALLDATA = []; %space
rightmouse = [1 3 10 5 7 2 6 101]; %right mice
for environment = 1:6 %go through all environments doing the streamlined way to pull data out
    if environment == 1 %starting "if" statement to load each individual overlapped
        filename = load('C:\Users\julie\OneDrive\Documents\MATLAB\example data\NoConfLowVolControl_withDTs.mat');
    elseif environment == 2 %starting "if" statement to load each individual overlapped
        filename = load('C:\Users\julie\OneDrive\Documents\MATLAB\example data\LowConfLowVolControl_withDTs.mat');
    elseif environment == 3 %starting "if" statement to load each individual overlapped
        filename = load('C:\Users\julie\OneDrive\Documents\MATLAB\example data\HighConfLowVolControl_withDTs.mat');
    elseif environment == 4 %starting "if" statement to load each individual overlapped
        filename = load('C:\Users\julie\OneDrive\Documents\MATLAB\example data\NoConfHighVolControl_withDTs.mat');
    elseif environment == 5 %starting "if" statement to load each individual overlapped
        filename = load('C:\Users\julie\OneDrive\Documents\MATLAB\example data\LowConfHighVolControl_withDTs.mat');
    elseif environment == 6 %starting "if" statement to load each individual overlapped
        filename = load('C:\Users\julie\OneDrive\Documents\MATLAB\example data\HighConfHighVolControl_withDTs.mat');
    end %ending "if" statement for environments
    overlap = filename.overlap; %pull out cells
    for session = 1:length(overlap) %go through "overlap"
        data = overlap{1,session}; %pulling out raw data
        baitedDir = data(:,4); %identifying the "correct" direction: 0 means left; 1 means right
        baitedDir(:,2) = data(:,2); %filling second column with EMs
        correctDir = []; %empty vector that will be filled with correct directions for each trial
        for w = 2:length(baitedDir) %setting w to go through baitedDir starting at 2
            if baitedDir(w,2)~=baitedDir(w-1,2) && baitedDir(w,2)~=0 && baitedDir(w,2)~=11 && baitedDir(w,2)~=12 %if the EM for a trial indicates a turn...
               correctDir = [correctDir,baitedDir(w,1)]; %...fill in the correct direction for that trial #
            end %ending "if" statement
        end %ending "for" loop
        wantedEM = []; %space
        wem = 1; %format
        for WEM = 2:length(data) %go through data
            if data(WEM,2) ~= data(WEM-1,2) && data(WEM-1,2) == 0 %when a turn is detected...
                wantedEM(wem) = data(WEM,2); %store it
                wem=wem+1; %format
            end %ending "if" statement looking for turns
        end %ending "for" loop going through data
        nTrials = length(wantedEM); %pulling out the number of trials done
        LEFTs = [1 2 3 4]; %EMs for left turns
        RIGHTs = [5 6 7 8]; %EMs for right turns
        rights = []; %space
        rights = NaN(nTrials*2); %create a vector that will be filled for right turns
        L = length(correctDir); %set length
        rights(1:nTrials,1) = wantedEM; %fill "rights" with the EMs associated with turns in first column
        rights(ismember(rights,LEFTs)) = 0; %set any left turn to a 0 value
        rights(ismember(rights,RIGHTs)) = 1; %set any right turn to a 1 value
        rights(1:nTrials,2) = correctDir; %fill "rights" with the correct block direction
        rights_2 = []; %setting up a place to store just around block switch info
        w = 1; %setting up the column info
        for q = 2:length(correctDir) %going through the correct direction part of rights
            if rights(q,2) == 0 && rights(q-1,2) == 1 %if the block switched from right to left
                rights_2(w,1) = rights(q-10,1); %fill in 1-10 in row with 10 trials prior to switch
                rights_2(w,2) = rights(q-9,1);
                rights_2(w,3) = rights(q-8,1);
                rights_2(w,4) = rights(q-7,1);
                rights_2(w,5) = rights(q-6,1);
                rights_2(w,6) = rights(q-5,1);
                rights_2(w,7) = rights(q-4,1);
                rights_2(w,8) = rights(q-3,1);
                rights_2(w,9) = rights(q-2,1);
                rights_2(w,10) = rights(q-1,1);
                rights_2(w,11) = rights(q,1);
                rights_2(w,12) = rights(q+1,1); %fill in 11-20 in row with 10 trials after the switch
                rights_2(w,13) = rights(q+2,1);
                rights_2(w,14) = rights(q+3,1);
                rights_2(w,15) = rights(q+4,1);
                rights_2(w,16) = rights(q+5,1);
                rights_2(w,17) = rights(q+6,1);
                rights_2(w,18) = rights(q+7,1);
                rights_2(w,19) = rights(q+8,1);
                rights_2(w,20) = rights(q+9,1);
                rights_2(w,21) = rights(q+10,1);
                w = w+1; %add 1 to w
            end %ending "if" statement
        end %ending "for" loop
        for e = 1:21 %going through the 20 trials in the data set
            rights_2(w,e) = nanmean(rights_2(1:w-1,e)); %putting in the average for all the switches
        end %ending "for" loop
        rights = NaN(nTrials*2); %create a vector that will be filled for the other kind of transition
        rights(1:nTrials,1) = wantedEM; %fill "rights" with the EMs associated with turns in first column
        rights(ismember(rights,LEFTs)) = 0; %set any left turn to a 0 value
        rights(ismember(rights,RIGHTs)) = 1; %set any right turn to a 1 value
        rights(1:nTrials,2) = correctDir; %fill "rights" with the correct block direction
        rights_3 = []; %setting up a place to store just around block switch info
        rr = 1; %setting up the column info
        for q = 2:length(correctDir) %going through the correct direction part of pulls
            if rights(q,2) == 1 && rights(q-1,2) == 0 %if the block switched from left to right
                rights_3(rr,1) = rights(q-10,1); %fill in 1-10 in row with 10 trials prior to switch
                rights_3(rr,2) = rights(q-9,1);
                rights_3(rr,3) = rights(q-8,1);
                rights_3(rr,4) = rights(q-7,1);
                rights_3(rr,5) = rights(q-6,1);
                rights_3(rr,6) = rights(q-5,1);
                rights_3(rr,7) = rights(q-4,1);
                rights_3(rr,8) = rights(q-3,1);
                rights_3(rr,9) = rights(q-2,1);
                rights_3(rr,10) = rights(q-1,1);
                rights_3(rr,11) = rights(q,1);
                rights_3(rr,12) = rights(q+1,1); %fill in 11-20 in row with 10 trials after the switch
                rights_3(rr,13) = rights(q+2,1);
                rights_3(rr,14) = rights(q+3,1);
                rights_3(rr,15) = rights(q+4,1);
                rights_3(rr,16) = rights(q+5,1);
                rights_3(rr,17) = rights(q+6,1);
                rights_3(rr,18) = rights(q+7,1);
                rights_3(rr,19) = rights(q+8,1);
                rights_3(rr,20) = rights(q+9,1);
                rights_3(rr,21) = rights(q+10,1);
                rr = rr+1; %add 1 to rr
            end %ending "if" statement
        end %ending "for" loop
        for e = 1:21 %going through the 20 trials in the data set
            rights_3(rr,e) = nanmean(rights_3(1:rr-1,e)); %putting in the average for all the switches
        end %ending "for" loop
        L1(1,:) = [1:21]; %first row is spots on x axis
        L1(2,:) = movmean(rights_2(w,:)',[4 0]); %second row is moving average travel line RtoL
        L2(1,:) = [1:21]; %first row is spots on x axis
        L2(2,:) = 0.5; %second row is the 50% cross line
        if L1 (2,12:21) < 0.5 %if it is already in the L range...
            L1(2,12) = 0.5; %artificially fill in the first one
        end %ending "if" statement looking for it already being in L
        P = InterX(L1,L2); %finding the intersection
        P2 = []; %place to store just trial number
        P2 = P(1,:); %copy over just the trial numbers
        for x = 1:length(P2) %going through all the cross points
            if P2(1,x) < 11.1 %take out any pre-block switch
                P2(:,x) = NaN; %make a NaN
            end %ending "if" statement looking for pre-block crossings
        end %ending "for" loop going through cross points
        crosspoints = P2(1,:); %copy over cross points to new place
        crosspoints = crosspoints(~isnan(crosspoints)); %take out NaNs
        cross50_RtoL = crosspoints(1,1)-11; %first cross point
        if ismember(overlap{3,session},rightmouse) %if pref right...
            pref_pers = cross50_RtoL; %store pref
        else %if pref left...
            nonpref_pers = cross50_RtoL; %store nonpref
        end %ending "if" statement separating by preference
        L1(2,:) = movmean(rights_3(rr,:)',[4 0]); %second row is the moving average travel line LtoR
        P = InterX(L1,L2); %finding the intersection
        P2 = []; %place to store just trial number
        P2 = P(1,:); %copy over just the trial numbers
        for x = 1:length(P2) %going through all the cross points
            if P2(1,x) < 11.1 %take out any pre-block switch
                P2(:,x) = NaN; %make a NaN
            end %ending "if" statement looking for pre-block crossings
        end %ending "for" loop going through cross points
        crosspoints = P2(1,:); %copy over cross points to new place
        crosspoints = crosspoints(~isnan(crosspoints)); %take out NaNs
        cross50_LtoR = crosspoints(1,1)-11; %first cross point
        if ismember(overlap{3,session},rightmouse) %if pref right...
            nonpref_pers = cross50_LtoR; %store pref
        else %if pref left...
            pref_pers = cross50_LtoR; %store nonpref
        end %ending "if" statement separating by preference
        if environment == 1 && session == 1 %first session from first environment...
            savespot = 1; %we start in first spot
        else %if we are beyond the first session...
            savespot = savespot+1; %move down one spot for saving the data
        end %ending "if" statement figuring out where to save the data
        ALLDATA(savespot,1) = pref_pers; %save preferred
        ALLDATA(savespot,2) = nonpref_pers; %save non-preferred
        ALLDATA(savespot,3) = overlap{3,session}; %save mouse ID
    end %ending "for" loop going through overlap
end %ending "for" loop going through environment

%% perseverative errors vs bias proportion - bias

ALLDATA = []; %space
rightmouse = [1 3 10 5 7 2 6 101]; %right mice
for environment = 1:6 %go through all environments doing the streamlined way to pull data out
    if environment == 1 %starting "if" statement to load each individual overlapped
        filename = load('C:\Users\julie\OneDrive\Documents\MATLAB\example data\NoConfLowVolControl_withDTs.mat');
    elseif environment == 2 %starting "if" statement to load each individual overlapped
        filename = load('C:\Users\julie\OneDrive\Documents\MATLAB\example data\LowConfLowVolControl_withDTs.mat');
    elseif environment == 3 %starting "if" statement to load each individual overlapped
        filename = load('C:\Users\julie\OneDrive\Documents\MATLAB\example data\HighConfLowVolControl_withDTs.mat');
    elseif environment == 4 %starting "if" statement to load each individual overlapped
        filename = load('C:\Users\julie\OneDrive\Documents\MATLAB\example data\NoConfHighVolControl_withDTs.mat');
    elseif environment == 5 %starting "if" statement to load each individual overlapped
        filename = load('C:\Users\julie\OneDrive\Documents\MATLAB\example data\LowConfHighVolControl_withDTs.mat');
    elseif environment == 6 %starting "if" statement to load each individual overlapped
        filename = load('C:\Users\julie\OneDrive\Documents\MATLAB\example data\HighConfHighVolControl_withDTs.mat');
    end %ending "if" statement for environments
    overlap = filename.overlap; %pull out cells
    for session = 1:length(overlap) %go through "overlap"
        data = overlap{1,session}; %pulling out raw data
        baitedDir = data(:,4); %identifying the "correct" direction: 0 means left; 1 means right
        baitedDir(:,2) = data(:,2); %filling second column with EMs
        correctDir = []; %empty vector that will be filled with correct directions for each trial
        for w = 2:length(baitedDir) %setting w to go through baitedDir starting at 2
            if baitedDir(w,2)~=baitedDir(w-1,2) && baitedDir(w,2)~=0 && baitedDir(w,2)~=11 && baitedDir(w,2)~=12 %if the EM for a trial indicates a turn...
               correctDir = [correctDir,baitedDir(w,1)]; %...fill in the correct direction for that trial #
            end %ending "if" statement
        end %ending "for" loop
        wantedEM = []; %space
        wem = 1; %format
        for WEM = 2:length(data) %go through data
            if data(WEM,2) ~= data(WEM-1,2) && data(WEM-1,2) == 0 %when a turn is detected...
                wantedEM(wem) = data(WEM,2); %store it
                wem=wem+1; %format
            end %ending "if" statement looking for turns
        end %ending "for" loop going through data
        nTrials = length(wantedEM); %pulling out the number of trials done
        LEFTs = [1 2 3 4]; %EMs for left turns
        RIGHTs = [5 6 7 8]; %EMs for right turns
        rights = []; %space
        rights = NaN(nTrials*2); %create a vector that will be filled for right turns
        L = length(correctDir); %set length
        rights(1:nTrials,1) = wantedEM; %fill "rights" with the EMs associated with turns in first column
        rights(ismember(rights,LEFTs)) = 0; %set any left turn to a 0 value
        rights(ismember(rights,RIGHTs)) = 1; %set any right turn to a 1 value
        rights(1:nTrials,2) = correctDir; %fill "rights" with the correct block direction
        rights_2 = []; %setting up a place to store just around block switch info
        w = 1; %setting up the column info
        for q = 2:length(correctDir) %going through the correct direction part of rights
            if rights(q,2) == 0 && rights(q-1,2) == 1 %if the block switched from right to left
                rights_2(w,1) = rights(q-10,1); %fill in 1-10 in row with 10 trials prior to switch
                rights_2(w,2) = rights(q-9,1);
                rights_2(w,3) = rights(q-8,1);
                rights_2(w,4) = rights(q-7,1);
                rights_2(w,5) = rights(q-6,1);
                rights_2(w,6) = rights(q-5,1);
                rights_2(w,7) = rights(q-4,1);
                rights_2(w,8) = rights(q-3,1);
                rights_2(w,9) = rights(q-2,1);
                rights_2(w,10) = rights(q-1,1);
                rights_2(w,11) = rights(q,1);
                rights_2(w,12) = rights(q+1,1); %fill in 11-20 in row with 10 trials after the switch
                rights_2(w,13) = rights(q+2,1);
                rights_2(w,14) = rights(q+3,1);
                rights_2(w,15) = rights(q+4,1);
                rights_2(w,16) = rights(q+5,1);
                rights_2(w,17) = rights(q+6,1);
                rights_2(w,18) = rights(q+7,1);
                rights_2(w,19) = rights(q+8,1);
                rights_2(w,20) = rights(q+9,1);
                rights_2(w,21) = rights(q+10,1);
                w = w+1; %add 1 to w
            end %ending "if" statement
        end %ending "for" loop
        for e = 1:21 %going through the 20 trials in the data set
            rights_2(w,e) = nanmean(rights_2(1:w-1,e)); %putting in the average for all the switches
        end %ending "for" loop
        rights = NaN(nTrials*2); %create a vector that will be filled for the other kind of transition
        rights(1:nTrials,1) = wantedEM; %fill "rights" with the EMs associated with turns in first column
        rights(ismember(rights,LEFTs)) = 0; %set any left turn to a 0 value
        rights(ismember(rights,RIGHTs)) = 1; %set any right turn to a 1 value
        rights(1:nTrials,2) = correctDir; %fill "rights" with the correct block direction
        rights_3 = []; %setting up a place to store just around block switch info
        rr = 1; %setting up the column info
        for q = 2:length(correctDir) %going through the correct direction part of pulls
            if rights(q,2) == 1 && rights(q-1,2) == 0 %if the block switched from left to right
                rights_3(rr,1) = rights(q-10,1); %fill in 1-10 in row with 10 trials prior to switch
                rights_3(rr,2) = rights(q-9,1);
                rights_3(rr,3) = rights(q-8,1);
                rights_3(rr,4) = rights(q-7,1);
                rights_3(rr,5) = rights(q-6,1);
                rights_3(rr,6) = rights(q-5,1);
                rights_3(rr,7) = rights(q-4,1);
                rights_3(rr,8) = rights(q-3,1);
                rights_3(rr,9) = rights(q-2,1);
                rights_3(rr,10) = rights(q-1,1);
                rights_3(rr,11) = rights(q,1);
                rights_3(rr,12) = rights(q+1,1); %fill in 11-20 in row with 10 trials after the switch
                rights_3(rr,13) = rights(q+2,1);
                rights_3(rr,14) = rights(q+3,1);
                rights_3(rr,15) = rights(q+4,1);
                rights_3(rr,16) = rights(q+5,1);
                rights_3(rr,17) = rights(q+6,1);
                rights_3(rr,18) = rights(q+7,1);
                rights_3(rr,19) = rights(q+8,1);
                rights_3(rr,20) = rights(q+9,1);
                rights_3(rr,21) = rights(q+10,1);
                rr = rr+1; %add 1 to rr
            end %ending "if" statement
        end %ending "for" loop
        for e = 1:21 %going through the 20 trials in the data set
            rights_3(rr,e) = nanmean(rights_3(1:rr-1,e)); %putting in the average for all the switches
        end %ending "for" loop
        L1(1,:) = [1:21]; %first row is spots on x axis
        L1(2,:) = movmean(rights_2(w,:)',[4 0]); %second row is moving average travel line RtoL
        L2(1,:) = [1:21]; %first row is spots on x axis
        L2(2,:) = 0.5; %second row is the 50% cross line
        P = InterX(L1,L2); %finding the intersection
        P2 = []; %place to store just trial number
        P2 = P(1,:); %copy over just the trial numbers
        for x = 1:length(P2) %going through all the cross points
            if P2(1,x) < 11.1 %take out any pre-block switch
                P2(:,x) = NaN; %make a NaN
            end %ending "if" statement looking for pre-block crossings
        end %ending "for" loop going through cross points
        crosspoints = P2(1,:); %copy over cross points to new place
        crosspoints = crosspoints(~isnan(crosspoints)); %take out NaNs
        cross50_RtoL = crosspoints(1,1)-11; %first cross point
        if ismember(overlap{3,session},rightmouse) %if pref right...
            pref_pers = cross50_RtoL; %store pref
        else %if pref left...
            nonpref_pers = cross50_RtoL; %store nonpref
        end %ending "if" statement separating by preference
        L1(2,:) = movmean(rights_3(rr,:)',[4 0]); %second row is the moving average travel line LtoR
        P = InterX(L1,L2); %finding the intersection
        P2 = []; %place to store just trial number
        P2 = P(1,:); %copy over just the trial numbers
        for x = 1:length(P2) %going through all the cross points
            if P2(1,x) < 11.1 %take out any pre-block switch
                P2(:,x) = NaN; %make a NaN
            end %ending "if" statement looking for pre-block crossings
        end %ending "for" loop going through cross points
        crosspoints = P2(1,:); %copy over cross points to new place
        crosspoints = crosspoints(~isnan(crosspoints)); %take out NaNs
        cross50_LtoR = crosspoints(1,1)-11; %first cross point
        if ismember(overlap{3,session},rightmouse) %if pref right...
            nonpref_pers = cross50_LtoR; %store pref
        else %if pref left...
            pref_pers = cross50_LtoR; %store nonpref
        end %ending "if" statement separating by preference
        total_pers = nonpref_pers + pref_pers; %how many perseveratie errors tend to get made?
        pref_percent = pref_pers/total_pers; %proportion of pref pers
        if ismember(overlap{3,session},rightmouse) %if pref right...
            BIAS = [5 6 7 8]; %bias EM
        else %if pref left...
            BIAS = [1 2 3 4]; %bias EM
        end %ending "if" statement setting EMs
        for b = 1:length(wantedEM) %go through the EMs
            if ismember(wantedEM(b),BIAS) %if pref...
                wantedEM(b) = 1; %change to 1
            else %if nonpref...
                wantedEM(b) = 0; %change to 0
            end %ending "if" statement for pref vs nonpref
        end %ending "for" loop going through EMs
        bias_percent = sum(wantedEM)/length(wantedEM); %bias percent
        diff_from_expected = pref_percent - bias_percent; %how different is perseverative error breakdown from total trial breakdown
        if environment == 1 && session == 1 %first session from first environment...
            savespot = 1; %we start in first spot
        else %if we are beyond the first session...
            savespot = savespot+1; %move down one spot for saving the data
        end %ending "if" statement figuring out where to save the data
        ALLDATA(savespot,1) = diff_from_expected; %save preferred diff from expected
        ALLDATA(savespot,2) = overlap{3,session}; %save mouse ID
    end %ending "for" loop going through overlap
end %ending "for" loop going through environment

%% regressive errors - bias

ALLDATA = []; %space
rightmouse = [1 3 10 5 7 2 6 101]; %right mice
for environment = 1:6 %go through all environments doing the streamlined way to pull data out
    if environment == 1 %starting "if" statement to load each individual overlapped
        filename = load('C:\Users\julie\OneDrive\Documents\MATLAB\example data\NoConfLowVolControl_withDTs.mat');
    elseif environment == 2 %starting "if" statement to load each individual overlapped
        filename = load('C:\Users\julie\OneDrive\Documents\MATLAB\example data\LowConfLowVolControl_withDTs.mat');
    elseif environment == 3 %starting "if" statement to load each individual overlapped
        filename = load('C:\Users\julie\OneDrive\Documents\MATLAB\example data\HighConfLowVolControl_withDTs.mat');
    elseif environment == 4 %starting "if" statement to load each individual overlapped
        filename = load('C:\Users\julie\OneDrive\Documents\MATLAB\example data\NoConfHighVolControl_withDTs.mat');
    elseif environment == 5 %starting "if" statement to load each individual overlapped
        filename = load('C:\Users\julie\OneDrive\Documents\MATLAB\example data\LowConfHighVolControl_withDTs.mat');
    elseif environment == 6 %starting "if" statement to load each individual overlapped
        filename = load('C:\Users\julie\OneDrive\Documents\MATLAB\example data\HighConfHighVolControl_withDTs.mat');
    end %ending "if" statement for environments
    overlap = filename.overlap; %pull out cells
    for session = 1:length(overlap) %go through "overlap"
        data = overlap{1,session}; %pulling out raw data
        for y = 1:length(data(:,9)) %going through the column that tracks block #
            if data(y,9) == 2 && data(y-1,9) == 1 %when the block has changed to the second block...
                trial_switch = data(y,11)+1; %code the next trial number as the first trial of block 2
            end %ending "if" statement
        end %ending "for" loop
        wantedEM = []; %space
        wem = 1; %format
        for WEM = 2:length(data) %go through data
            if data(WEM,2) ~= data(WEM-1,2) && data(WEM-1,2) == 0 %when a turn is detected...
                wantedEM(wem) = data(WEM,2); %store it
                wem=wem+1; %format
            end %ending "if" statement looking for turns
        end %ending "for" loop going through data
        relevant_EMs = wantedEM(trial_switch:end); %only pulling out EMs from block 2 onward
        blocks = NaN(length(data(:,9)),1); %making nans to keep track of blocks
        for b = 2:length(data(:,9)) %going through the block counter
            if data(b,11) ~= data(b-1,11) %if a new trial has been counted...
                blocks(b) = data(b,9); %fill in which block number we are in in the "blocks" spot
            end %ending "if" statement
        end %ending "for" loop
        blocks = blocks(~isnan(blocks)); %take out the NaNs
        relevant_blocks = blocks(trial_switch:end); %pulling out only blocks 2+
        relevant_EMs(2,:) = relevant_blocks(1:end); %fill in the second row with block counter
        lastblock = max(relevant_blocks); %counting how many blocks there are
        relevant_EMs(3,:) = 1:length(relevant_EMs); %adding a trial counter
        relevant_EMs(4,:) = 0; %filling in a tracker row with 0s
        x = [1:lastblock]; %making a vector of x values
        corrects = [1 2 5 6]; %correct trial EMs
        for r = 1:length(relevant_EMs) %going through the EMs
            if ismember(relevant_EMs(1,r),corrects) %if a correct trial has been counted...
                relevant_EMs(4,r) = x(relevant_EMs(2,r)); %mark it in the fourth row according to which block we are in
            end %ending "if" statement
        end %ending "for" loop
        breakdown = zeros(length(x),length(relevant_EMs)); %place to store a breakdown of the data by block
        for xx = 1:length(relevant_EMs) %going through the block counter
            if ismember(relevant_EMs(4,xx),x) %if it has been tagged as a correct trial...
                breakdown(relevant_EMs(4,xx),xx) = relevant_EMs(3,xx); %fill in the associated column for the block number with the trial count that the correct trial happened on
            end %ending "if" statement
        end %ending "for" loop
        breakdown(find(breakdown == 0)) = 1000; %substitute 0s for 1000s
        for bb = 1:length(x) %going through each block
            x(2,bb) = min(breakdown(bb,:)); %fill in the second row of x with the first correct trial number of each block
        end %ending "for" loop
        space = NaN(length(x),length(relevant_EMs)); %NaN space
        for s = 2:length(x)-1 %go through x except for the first block & last block...
            col = 1; %re-start at 1
            for rem = 1:length(relevant_EMs) %go through the EM combined data...
                if relevant_EMs(3,rem) > x(2,s) && relevant_EMs(3,rem) < x(2,s+1) %if we are after the correct switch but before the next correct switch...
                    space(s,col) = relevant_EMs(1,rem); %fill in the EM
                    col=col+1; %format
                end %ending "if" statement for trial timing
            end %ending "for" loop going through combined data
        end %ending "for" loop going through blocks
        block_direction = []; %space
        bd = 1; %format
        for BD = 2:length(data)
            if data(BD,4) ~= data(BD-1,4) %when a block change happens...
                block_direction(bd) = data(BD,4); %mark it
                bd=bd+1; %format
            end %ending "if" statement for block change
        end %ending "for" loop going through data
        LI = [3 4]; %left incorrect
        RI = [7 8]; %right incorrect
        for incorrect = 1:length(block_direction)-1 %go through the blocks (but not last one in case never crossed properly)
            counter = 0; %always start the counter at 0
            if block_direction(1,incorrect) == 0 %when we are dealing with a left block...
                for counting = 1:length(space) %going through the saved space...
                    if ismember(space(incorrect+1,counting),RI) %when a right incorrect happens...
                        counter=counter+1; %add 1 to the counter
                    end %ending "if" statement counting incorrects
                end %ending "for" loop going through saved space
            else %if a right block...
                for counting = 1:length(space) %going through the saved space...
                    if ismember(space(incorrect+1,counting),LI) %when a left incorrect happens...
                        counter=counter+1; %add 1 to the counter
                    end %ending "if" statement counting incorrects
                end %ending "for" loop going through saved space
            end %ending "if" statement for left block
            block_direction(2,incorrect) = counter; %mark the counter
        end %ending "for" loop going through the blocks
        if ismember(overlap{3,session},rightmouse) %if right mouse...
            pref = 1; %pref blocks are rights
        else %if left mouse...
            pref = 0; %pref blocks are lefts
        end %ending "if" statement for right or left mouse
        pref_reg = []; nonpref_reg = []; %space
        pp = 1; np = 1; %format
        for sep = 1:length(block_direction) %go through blocks
            if block_direction(1,sep) == pref %if a pref block...
                pref_reg(pp) = block_direction(2,sep); %mark info
                pp=pp+1; %format
            else %if a non-pref block...
                nonpref_reg(np) = block_direction(2,sep); %mark info
                np=np+1; %format
            end %ending "if" statement for pref vs nonpref
        end %ending "for" loop going through blocks
        if environment == 1 && session == 1 %first session from first environment...
            savespot = 1; %we start in first spot
        else %if we are beyond the first session...
            savespot = savespot+1; %move down one spot for saving the data
        end %ending "if" statement figuring out where to save the data
        ALLDATA(savespot,1) = mean(nonpref_reg); %save preferred (flip because into block)
        ALLDATA(savespot,2) = mean(pref_reg); %save non-preferred (flip because into block)
        ALLDATA(savespot,3) = overlap{3,session}; %save mouse ID
    end %ending "for" loop going through overlap
end %ending "for" loop going through environment

%% Regressive Errors vs Expected Bias - bias

ALLDATA = []; %space
rightmouse = [1 3 10 5 7 2 6 101]; %right mice
for environment = 1:6 %go through all environments doing the streamlined way to pull data out
    if environment == 1 %starting "if" statement to load each individual overlapped
        filename = load('C:\Users\julie\OneDrive\Documents\MATLAB\example data\NoConfLowVolControl_withDTs.mat');
    elseif environment == 2 %starting "if" statement to load each individual overlapped
        filename = load('C:\Users\julie\OneDrive\Documents\MATLAB\example data\LowConfLowVolControl_withDTs.mat');
    elseif environment == 3 %starting "if" statement to load each individual overlapped
        filename = load('C:\Users\julie\OneDrive\Documents\MATLAB\example data\HighConfLowVolControl_withDTs.mat');
    elseif environment == 4 %starting "if" statement to load each individual overlapped
        filename = load('C:\Users\julie\OneDrive\Documents\MATLAB\example data\NoConfHighVolControl_withDTs.mat');
    elseif environment == 5 %starting "if" statement to load each individual overlapped
        filename = load('C:\Users\julie\OneDrive\Documents\MATLAB\example data\LowConfHighVolControl_withDTs.mat');
    elseif environment == 6 %starting "if" statement to load each individual overlapped
        filename = load('C:\Users\julie\OneDrive\Documents\MATLAB\example data\HighConfHighVolControl_withDTs.mat');
    end %ending "if" statement for environments
    overlap = filename.overlap; %pull out cells
    for session = 1:length(overlap) %go through "overlap"
        data = overlap{1,session}; %pulling out raw data
        for y = 1:length(data(:,9)) %going through the column that tracks block #
            if data(y,9) == 2 && data(y-1,9) == 1 %when the block has changed to the second block...
                trial_switch = data(y,11)+1; %code the next trial number as the first trial of block 2
            end %ending "if" statement
        end %ending "for" loop
        wantedEM = []; %space
        wem = 1; %format
        for WEM = 2:length(data) %go through data
            if data(WEM,2) ~= data(WEM-1,2) && data(WEM-1,2) == 0 %when a turn is detected...
                wantedEM(wem) = data(WEM,2); %store it
                wem=wem+1; %format
            end %ending "if" statement looking for turns
        end %ending "for" loop going through data
        relevant_EMs = wantedEM(trial_switch:end); %only pulling out EMs from block 2 onward
        blocks = NaN(length(data(:,9)),1); %making nans to keep track of blocks
        for b = 2:length(data(:,9)) %going through the block counter
            if data(b,11) ~= data(b-1,11) %if a new trial has been counted...
                blocks(b) = data(b,9); %fill in which block number we are in in the "blocks" spot
            end %ending "if" statement
        end %ending "for" loop
        blocks = blocks(~isnan(blocks)); %take out the NaNs
        relevant_blocks = blocks(trial_switch:end); %pulling out only blocks 2+
        relevant_EMs(2,:) = relevant_blocks(1:end); %fill in the second row with block counter
        lastblock = max(relevant_blocks); %counting how many blocks there are
        relevant_EMs(3,:) = 1:length(relevant_EMs); %adding a trial counter
        relevant_EMs(4,:) = 0; %filling in a tracker row with 0s
        x = [1:lastblock]; %making a vector of x values
        corrects = [1 2 5 6]; %correct trial EMs
        for r = 1:length(relevant_EMs) %going through the EMs
            if ismember(relevant_EMs(1,r),corrects) %if a correct trial has been counted...
                relevant_EMs(4,r) = x(relevant_EMs(2,r)); %mark it in the fourth row according to which block we are in
            end %ending "if" statement
        end %ending "for" loop
        breakdown = zeros(length(x),length(relevant_EMs)); %place to store a breakdown of the data by block
        for xx = 1:length(relevant_EMs) %going through the block counter
            if ismember(relevant_EMs(4,xx),x) %if it has been tagged as a correct trial...
                breakdown(relevant_EMs(4,xx),xx) = relevant_EMs(3,xx); %fill in the associated column for the block number with the trial count that the correct trial happened on
            end %ending "if" statement
        end %ending "for" loop
        breakdown(find(breakdown == 0)) = 1000; %substitute 0s for 1000s
        for bb = 1:length(x) %going through each block
            x(2,bb) = min(breakdown(bb,:)); %fill in the second row of x with the first correct trial number of each block
        end %ending "for" loop
        space = NaN(length(x),length(relevant_EMs)); %NaN space
        for s = 2:length(x)-1 %go through x except for the first block & last block...
            col = 1; %re-start at 1
            for rem = 1:length(relevant_EMs) %go through the EM combined data...
                if relevant_EMs(3,rem) > x(2,s) && relevant_EMs(3,rem) < x(2,s+1) %if we are after the correct switch but before the next correct switch...
                    space(s,col) = relevant_EMs(1,rem); %fill in the EM
                    col=col+1; %format
                end %ending "if" statement for trial timing
            end %ending "for" loop going through combined data
        end %ending "for" loop going through blocks
        block_direction = []; %space
        bd = 1; %format
        for BD = 2:length(data)
            if data(BD,4) ~= data(BD-1,4) %when a block change happens...
                block_direction(bd) = data(BD,4); %mark it
                bd=bd+1; %format
            end %ending "if" statement for block change
        end %ending "for" loop going through data
        LI = [3 4]; %left incorrect
        RI = [7 8]; %right incorrect
        for incorrect = 1:length(block_direction)-1 %go through the blocks (but not last one in case never crossed properly)
            counter = 0; %always start the counter at 0
            if block_direction(1,incorrect) == 0 %when we are dealing with a left block...
                for counting = 1:length(space) %going through the saved space...
                    if ismember(space(incorrect+1,counting),RI) %when a right incorrect happens...
                        counter=counter+1; %add 1 to the counter
                    end %ending "if" statement counting incorrects
                end %ending "for" loop going through saved space
            else %if a right block...
                for counting = 1:length(space) %going through the saved space...
                    if ismember(space(incorrect+1,counting),LI) %when a left incorrect happens...
                        counter=counter+1; %add 1 to the counter
                    end %ending "if" statement counting incorrects
                end %ending "for" loop going through saved space
            end %ending "if" statement for left block
            block_direction(2,incorrect) = counter; %mark the counter
        end %ending "for" loop going through the blocks
        if ismember(overlap{3,session},rightmouse) %if right mouse...
            pref = 1; %pref blocks are rights
        else %if left mouse...
            pref = 0; %pref blocks are lefts
        end %ending "if" statement for right or left mouse
        pref_reg = []; nonpref_reg = []; %space
        pp = 1; np = 1; %format
        for sep = 1:length(block_direction) %go through blocks
            if block_direction(1,sep) == pref %if a pref block...
                pref_reg(pp) = block_direction(2,sep); %mark info
                pp=pp+1; %format
            else %if a non-pref block...
                nonpref_reg(np) = block_direction(2,sep); %mark info
                np=np+1; %format
            end %ending "if" statement for pref vs nonpref
        end %ending "for" loop going through blocks
        total_reg = mean(nonpref_reg) + mean(pref_reg); %how many reg errors tend to get made?
        pref_percent = mean(pref_reg)/total_reg; %proportion of reg pers
        if ismember(overlap{3,session},rightmouse) %if pref right...
            BIAS = [5 6 7 8]; %bias EM
        else %if pref left...
            BIAS = [1 2 3 4]; %bias EM
        end %ending "if" statement setting EMs
        for b = 1:length(wantedEM) %go through the EMs
            if ismember(wantedEM(b),BIAS) %if pref...
                wantedEM(b) = 1; %change to 1
            else %if nonpref...
                wantedEM(b) = 0; %change to 0
            end %ending "if" statement for pref vs nonpref
        end %ending "for" loop going through EMs
        bias_percent = sum(wantedEM)/length(wantedEM); %bias percent
        diff_from_expected = pref_percent - bias_percent; %how different is perseverative error breakdown from total trial breakdown
        if environment == 1 && session == 1 %first session from first environment...
            savespot = 1; %we start in first spot
        else %if we are beyond the first session...
            savespot = savespot+1; %move down one spot for saving the data
        end %ending "if" statement figuring out where to save the data
        ALLDATA(savespot,1) = diff_from_expected; %save diff_from_expected
        ALLDATA(savespot,2) = overlap{3,session}; %save mouse ID
    end %ending "for" loop going through overlap
end %ending "for" loop going through environment

%% switch into bias - bias

ALLDATA = []; %space
rightmouse = [1 3 10 5 7 2 6 101]; %right mice
for environment = 1:6 %go through all environments doing the streamlined way to pull data out
    if environment == 1 %starting "if" statement to load each individual overlapped
        filename = load('C:\Users\julie\OneDrive\Documents\MATLAB\example data\NoConfLowVolControl_withDTs.mat');
    elseif environment == 2 %starting "if" statement to load each individual overlapped
        filename = load('C:\Users\julie\OneDrive\Documents\MATLAB\example data\LowConfLowVolControl_withDTs.mat');
    elseif environment == 3 %starting "if" statement to load each individual overlapped
        filename = load('C:\Users\julie\OneDrive\Documents\MATLAB\example data\HighConfLowVolControl_withDTs.mat');
    elseif environment == 4 %starting "if" statement to load each individual overlapped
        filename = load('C:\Users\julie\OneDrive\Documents\MATLAB\example data\NoConfHighVolControl_withDTs.mat');
    elseif environment == 5 %starting "if" statement to load each individual overlapped
        filename = load('C:\Users\julie\OneDrive\Documents\MATLAB\example data\LowConfHighVolControl_withDTs.mat');
    elseif environment == 6 %starting "if" statement to load each individual overlapped
        filename = load('C:\Users\julie\OneDrive\Documents\MATLAB\example data\HighConfHighVolControl_withDTs.mat');
    end %ending "if" statement for environments
    overlap = filename.overlap; %pull out cells
    for session = 1:length(overlap) %go through "overlap"
        data = overlap{1,session}; %pulling out raw data
        xy = []; %place to store location data
        xy(1,:) = data(:,13); %copy over the x info from data file
        xy(2,:) = data(:,14); %copy over y info from data file
        if min(xy(2,:)) < 40 %this should be a good way to XY determine which maze...
            maze = 239; %set the maze data to 239
        else %if we have the other orientation...
            maze = 241; %set maze data to 241
        end %ending "if" statement trying to mathematically separate mazes
        if maze == 239 %we are only running the analysis for 239 mice
            disp3=sqrt(diff2(xy(1,:),3).^2+diff2(xy(2,:),3).^2);  %equation from Eric for velocity
            turn1 = smooth(disp3,30)'; %smooth the velocity data
            turn1(2,:) = data(1:end-3,17); %fill in the second part of velocity data with the hall location
            stuff1 = NaN(length(turn1),1); %place to store hall 1
            stuff2 = NaN(length(turn1),1); %place to store hall 2
            stuff3 = NaN(length(turn1),1); %place to store hall 3
            stuff0 = NaN(length(turn1),1); %place to store center
            for splittingup = 1:length(turn1) %going through the velocity data
                if turn1(2,splittingup) == 1 %if the animal was in hall 1...
                    stuff1(splittingup) = turn1(1,splittingup); %fill that in stuff 1
                elseif turn1(2,splittingup) == 2 %if the animal was in hall 2...
                    stuff2(splittingup) = turn1(1,splittingup); %fill that in stuff 2
                elseif turn1(2,splittingup) == 3 %if the animal was in hall 3...
                    stuff3(splittingup) = turn1(1,splittingup); %fill that in stuff 3
                elseif turn1(2,splittingup) == 0 %if the animal was in the center...
                    stuff0(splittingup) = turn1(1,splittingup); %fill that in stuff 0
                end %ending "if" statement looking for location
            end %ending "for" loop going through velocity data
            stuff1(:,2) = data(1:end-3,1); %add in time info
            stuff2(:,2) = data(1:end-3,1); %add in time info
            stuff3(:,2) = data(1:end-3,1); %add in time info
            stuff0(:,2) = data(1:end-3,1); %add in time info
            turn1(3,:) = data(1:end-3,2); %fill in the EMs to turn1
            turn1(4,:) = 1:length(turn1); %fill in a spacer
            allempositions = []; %place to store all EM positions
            emspot = 1; %formatting
            ems = [1:8]; %any em
            for stars = 2:length(turn1) %going through the data
                if ismember(turn1(3,stars),ems) && turn1(3,stars-1) == 0 %if a turn is first detected...
                    allempositions(emspot) = turn1(4,stars); %mark down that spot position number
                    emspot = emspot + 1; %formatting
                end %ending "if" statement looking for ems
            end %ending "for" loop gooing through the data
            veltimes = []; %place to store a lot of the info
            veltimes(1,:) = turn1(1,:); %copy over the velocities
            veltimes(2,:) = data(1:end-3,1); %copy over the times
            veltimes(3,:) = 1:length(veltimes); %copy over the position # (to go with allempositions)
            cutoffvalues = []; %place to store the cutoff velocity for each turn
            hopethisworks = []; %going to try finding the min each time & then clearing it
            cutofftracker = 1; %formatting
            for merp = 1:length(allempositions)-1 %going through all the marked turns except for the last one
                hopethisworks = veltimes(1,allempositions(merp):allempositions(merp+1)); %temporarily copy over the vels between two turns
                sorted = sort(hopethisworks); %sort them from low to high
                cutoff = sorted(round(length(sorted)*0.15)); %find the velocity that cuts off the slowest 15% of values
                cutoffvalues(cutofftracker) = cutoff; %copy over the cutoff value
                cutofftracker = cutofftracker + 1; %formatting
                hopethisworks = []; %clearing for the next run
            end %ending "for" loop going through the empositions
            for eek = 1:length(allempositions)-1 %going through the ems
                for hoping = 1:length(veltimes) %going through veltimes
                    if (allempositions(eek) < veltimes(3,hoping)) && (veltimes(3,hoping) < allempositions(eek + 1)) %if we are at a position between two marked turns...
                        if veltimes(1,hoping) < cutoffvalues(eek) %if the velocity falls below the cutoff of what we are calling "decision time"...
                            veltimes(4,hoping) = 1; %mark it as a 1 in the bottom row
                        end %ending "if" statement looking for the values falling below cutoff
                    end %ending "if" statement looking at vels that fall between the EMs
                end %ending "for" loop going through veltimes
            end %ending "for" loop going through allempositions
            crossing = []; %storing when the threshold is crossed for the first time
            crossingdown = []; %slowing down
            downspot = 1; %formatting
            for firstinstance = 1:length(allempositions)-1 %going through the ems
                crossing(1,:) = veltimes(4,allempositions(firstinstance):allempositions(firstinstance + 1));  %temporarily copy over the 0s & 1s
                crossing(2,:) = veltimes(3,allempositions(firstinstance):allempositions(firstinstance + 1)); %temporarily copy over the positions
                for cutoffsearch = 2:length(crossing)-1 %going through crossing (second through second to last value)
                    if crossing(1,cutoffsearch) == 1 && crossing(1,cutoffsearch - 1) == 0 %if the threshold is crossed going down...
                        crossing(1,cutoffsearch) = 2; %change its value to a 2
                    end %ending "if" statement looking for when the threshold is crossed
                end %ending "for" loop going through crossing
                for separating = 1:length(crossing) %going through crossing
                    if crossing(1,separating) == 2 %if the mouse crossed threshold going down...
                        crossingdown(downspot) = crossing(2,separating); %copy over the position
                        downspot = downspot + 1; %formatting
                    end %ending "if" statement looking for slowing down
                end %ending "for" loop going through crossing
                    downtime = min(crossingdown); %first time slowing down
                for finaltry = 1:length(crossing) %going through crossing
                    if crossing(2,finaltry) == downtime %if it is down time...
                        crossing(1,finaltry) = 4; %change that value to a 4
                    else %otherwise...
                        crossing(1,finaltry) = 0; %change everything to a 0
                    end %ending "if" statement to find exact start/stop
                end %ending "for" loop going through crossing   
                veltimes(4,allempositions(firstinstance):allempositions(firstinstance + 1)) = crossing(1,:); %replace the veltimes with the cleared data
                crossing = []; crossingdown = []; %clear out this for next run
                downspot = 1; %formatting
            end %ending "for" loop going through to mark only first time a min is reached
            start = []; %decision time start
            t = 1; %formatting
            for testing = 1:length(veltimes) %going through veltimes
                if veltimes(4,testing) == 4 %if it is the start of decision time...
                    start(t,1) = veltimes(1,testing); %copy over vel
                    start(t,2) = veltimes(2,testing); %copy over time
                    t = t + 1; %formatting
                end %ending "if" statement looking for starts
            end %ending "for" loop going through veltimes
            timebracket = []; %going to try doing mins & max's for time to look at
            timebracket(:,1) = start(:,2);
            for TB = 2:length(allempositions) %this has already found the lines in data when turns are detected
                timebracket(TB-1,2) = data(allempositions(TB),1); %copy the times from those lines into the bracket
            end %ending "for" loop looking for EM timing
            trialcount = data(end,11); %number of trials done
            x = NaN(length(data),trialcount); %place for x
            y = NaN(length(data),trialcount); %place for y
            for gothrough = 1:length(data) %going through data
                if data(gothrough,11) > 0 && data(gothrough,11) < length(timebracket)+1 %getting through the first trial but also not doing one for after the last...
                    if timebracket(data(gothrough,11),1) <= data(gothrough,1) && data(gothrough,1) <= timebracket(data(gothrough,11),2) %if it falls within the time frame associated for that trial...
                        x(gothrough,data(gothrough,11)) = data(gothrough,13); %copy over the x info in a column by column basis for trial #
                        y(gothrough,data(gothrough,11)) = data(gothrough,14); %copy over the y info in a column by column basis for trial #
                    end %ending "if" statement trying to get only bracketed times
                end %ending "if" statement making sure we get through the first trial & not doing the very last one bc we don't know what the next choice would have been
            end %ending "for" loop going through data
            xcombo = []; %combining the info
            for combo = 1:trialcount-1 %going for each trial
                col = x(:,combo); %pull out just the column for each trial
                col2 = col(~isnan(col)); %take out the nans
                if combo > 1 %if we are past the first trial...
                    if length(col2) < size(xcombo,2) %if we have a shorter series of data...
                        col2(end+1:length(xcombo)) = NaN; %fill it in with NaN
                    end %ending "if" statement fixing shorter series
                    if length(col2) > size(xcombo,2) %if we have a longer series of data...
                        xcombo(:,end:length(col2)) = NaN; %fill in xcombo with NaN
                    end %ending "if" statement fixing longer series
                end %ending "if" statement getting past first trial
                xcombo(combo,:) = col2; %copy over
            end
            ycombo = []; %combining the info
            for combo = 1:trialcount-1 %going for each trial
                col = y(:,combo); %pull out just the column for each trial
                col2 = col(~isnan(col)); %take out the nans
                if combo > 1 %if we are past the first trial...
                    if length(col2) < size(ycombo,2) %if we have a shorter series of data...
                        col2(end+1:length(ycombo)) = NaN; %fill it in with NaN
                    end %ending "if" statement fiying shorter series
                    if length(col2) > size(ycombo,2) %if we have a longer series of data...
                        ycombo(:,end:length(col2)) = NaN; %fill in ycombo with NaN
                    end %ending "if" statement fiying longer series
                end %ending "if" statement getting past first trial
                ycombo(combo,:) = col2; %copy over
            end
            EM = data(:,2); %makes a variable for just EMs
            wantedEM = []; %making an open vector for EMs that we want
            for q = 2:length(EM) %starting with second value & running through all EMs
                if EM(q)~=EM(q-1) %if an EM is different from the one preceding it...
                    wantedEM = [wantedEM,EM(q)]; %...include those EMs in the wantedEM vector
                end %ending the "if" statement
            end %ending the "for" loop
            wantedEM(ismember(wantedEM,[0,11,12])) = []; %retains only EMs pertaining to turns
            EMs = wantedEM; %pull out the EMs
            left = [1:4]; %left EMs
            for em = 1:length(EMs) %going through EMs
                if ismember(EMs(em),left) %if it is a left...
                    EMs(em) = 0; %set it to 0
                else %if it is a right...
                    EMs(em) = 9; %set it to 9 (want to be out of bounds of normal EMs)
                end %ending "if" statement separating L/R
            end %ending "for" loop going through EMs
            EMs(2,1) = data(1,17); %fill in the first one
            counter = 2; %moving hall to EM info
            for hall = 2:length(data) %going to try sep by halls too
                if data(hall,2) ~= data(hall-1,2) && data(hall-1,2) == 0 %when a turn is detected...
                    EMs(2,counter) = data(hall,17); %pull over the hall number turned out of to the em file IMPORTANT THAT IT IS WHERE THEY TURNED OUT OF & NEXT HALL IS WHERE THEY TURNED INTO!
                    counter = counter + 1; %formatting
                end %ending "if" statement looking for turns
            end %ending "for" loop going through data
            commits = []; %space
            c = 1; %format
            s1L = 1; %formatting
            LX1 = []; %space
            LY1 = []; %space
            LOUTCOMEs = []; %space to line up each row with an EM for rewarded vs unrewarded
            rew = [1 3 5 7]; %rewarded EMs
            outcome = []; %space
            for o = 1:length(wantedEM) %go through EMs
                if ismember(wantedEM(o),rew) %if rewarded...
                    outcome(o) = 1; %mark 1
                else %if unrewarded...
                    outcome(o) = 0; %mark 0
                end %ending "if" statement separating by outcome
            end %ending "for" loop going through EMs
            for sep = 2:trialcount %going through again
                if EMs(1,sep) == 0 && EMs(2,sep) == 1 %left out of hall #...
                    LX1(s1L,:) = xcombo(sep-1,:); %pull out x data
                    LY1(s1L,:) = ycombo(sep-1,:); %pull out y data
                    LOUTCOMEs(1,s1L) = outcome(1,sep-1); %copy outcome that lead to that turn
                    s1L = s1L + 1; %formatting
                end %ending "if" statement finding hall 1 lefts
            end %ending "for" loop going through data
            s1R = 1; %formatting
            RX1 = []; %space
            RY1 = []; %space
            ROUTCOMEs = []; %space to line up each row with an EM
            for sep = 2:trialcount %going through again
                if EMs(1,sep) == 9 && EMs(2,sep) == 1 %right out of hall #...
                    RX1(s1R,:) = xcombo(sep-1,:); %pull out x data
                    RY1(s1R,:) = ycombo(sep-1,:); %pull out y data
                    ROUTCOMEs(1,s1R) = outcome(1,sep-1); %copy outcome of that trial
                    s1R = s1R + 1; %formatting
                end %ending "if" statement finding hall 1 rights
            end %ending "for" loop going through data
            cutoffhigh = 110; %hall 1 cutoff
            cutofflow = 40; %hall 1 spout cutoff
            midline = 160; %midline
                for tc = 1:size(LY1,1) %going through trials
                    for time = 1:length(LY1) %going through time
                        if LY1(tc,time) < cutofflow || LY1(tc,time) > cutoffhigh %if we are past the center cutoff & not at the spout...
                            LY1(tc,time) = NaN; %make it NaN
                            LX1(tc,time) = NaN; %make it NaN
                        end %ending "if" statement looking for center
                    end %ending "for" loop going row by row
                end %ending "for" loop going column by column
                for tc = 1:size(RY1,1) %going through trials
                    for time = 1:length(RY1) %going through time
                        if RY1(tc,time) < cutofflow || RY1(tc,time) > cutoffhigh %if we are past the center cutoff & not at the spout...
                            RY1(tc,time) = NaN; %make it NaN
                            RX1(tc,time) = NaN; %make it NaN
                        end %ending "if" statement looking for center
                    end %ending "for" loop going row by row
                end %ending "for" loop going column by column
                for l = 1:size(LX1,1) %go through the lefts
                    if nanmean(LX1(l,:)) > midline %if we have a committed left...
                        commits(1,c) = 1; %mark as committed
                        commits(2,c) = LOUTCOMEs(1,l); %copy outcome
                        c=c+1; %format
                    else %if we have a switchie...
                        commits(1,c) = 0; %mark as committed
                        commits(2,c) = LOUTCOMEs(1,l); %copy outcome
                        c=c+1; %format
                    end %ending "if" statement separating by committed
                end %ending "for" loop going through the lefts
                for l = 1:size(RX1,1) %go through the rights
                    if nanmean(RX1(l,:)) < midline %if we have a committed right...
                        commits(1,c) = 1; %mark as committed
                        commits(2,c) = ROUTCOMEs(1,l); %copy outcome
                        c=c+1; %format
                    else %if we have a switchie...
                        commits(1,c) = 0; %mark as committed
                        commits(2,c) = ROUTCOMEs(1,l); %copy outcome
                        c=c+1; %format
                    end %ending "if" statement separating by committed
                end %ending "for" loop going through the rights
            s2L = 1; %formatting
            LX2 = []; %space
            LY2 = []; %space
            LOUTCOMEs = []; %space to line up each row with an EM
            for sep = 2:trialcount %going through again
                if EMs(1,sep) == 0 && EMs(2,sep) == 2 %left out of hall #...
                    LX2(s2L,:) = xcombo(sep-1,:); %pull out x data
                    LY2(s2L,:) = ycombo(sep-1,:); %pull out y data
                    LOUTCOMEs(1,s2L) = outcome(1,sep-1); %dt
                    s2L = s2L + 1; %formatting
                end %ending "if" statement finding hall 1 lefts
            end %ending "for" loop going through data
            s2R = 1; %formatting
            RX2 = []; %space
            RY2 = []; %space
            ROUTCOMEs = []; %space to line up each row with an EM
            for sep = 2:trialcount %going through again
                if EMs(1,sep) == 9 && EMs(2,sep) == 2 %right out of hall #...
                    RX2(s2R,:) = xcombo(sep-1,:); %pull out x data
                    RY2(s2R,:) = ycombo(sep-1,:); %pull out y data
                    ROUTCOMEs(1,s2R) = outcome(1,sep-1); %dt
                    s2R = s2R + 1; %formatting
                end %ending "if" statement finding hall 1 rights
            end %ending "for" loop going through data
            cutoffhigh = 230; %cutoff
            cutofflow = 180; %spout cutoff
            midline = 150; %midline
            for tc = 1:size(LX2,1) %going through trials
                for time = 1:length(LX2) %going through time
                    if LX2(tc,time) < cutofflow || LX2(tc,time) > cutoffhigh %if we are past the center cutoff & not at the spout...
                        LX2(tc,time) = NaN; %make it NaN
                        LX2(tc,time) = NaN; %make it NaN
                    end %ending "if" statement looking for center
                end %ending "for" loop going row by row
            end %ending "for" loop going column by column
            for tc = 1:size(RX2,1) %going through trials
                for time = 1:length(RX2) %going through time
                    if RX2(tc,time) < cutofflow || RX2(tc,time) > cutoffhigh %if we are past the center cutoff & not at the spout...
                        RX2(tc,time) = NaN; %make it NaN
                        RX2(tc,time) = NaN; %make it NaN
                    end %ending "if" statement looking for center
                end %ending "for" loop going row by row
            end %ending "for" loop going column by column
            for l = 1:size(LY2,1) %go through the lefts
                if nanmean(LY2(l,:)) > midline %if we have a committed left...
                    commits(1,c) = 1; %mark as committed
                    commits(2,c) = LOUTCOMEs(1,l); %copy outcome
                    c=c+1; %format
                else %if we have a switchie...
                    commits(1,c) = 0; %mark as committed
                    commits(2,c) = LOUTCOMEs(1,l); %copy outcome
                    c=c+1; %format
                end %ending "if" statement separating by committed
            end %ending "for" loop going through the lefts
            for l = 1:size(RY2,1) %go through the rights
                if nanmean(RY2(l,:)) < midline %if we have a committed right...
                    commits(1,c) = 1; %mark as committed
                    commits(2,c) = ROUTCOMEs(1,l); %copy outcome
                    c=c+1; %format
                else %if we have a switchie...
                    commits(1,c) = 0; %mark as committed
                    commits(2,c) = ROUTCOMEs(1,l); %copy outcome
                    c=c+1; %format
                end %ending "if" statement separating by committed
            end %ending "for" loop going through the rights
            s3L = 1; %formatting
            LX3 = []; %space
            LY3 = []; %space
            LOUTCOMEs = []; %space to line up each row with an EM
            for sep = 2:trialcount %going through again
                if EMs(1,sep) == 0 && EMs(2,sep) == 3 %left out of hall #...
                    LX3(s3L,:) = xcombo(sep-1,:); %pull out x data
                    LY3(s3L,:) = ycombo(sep-1,:); %pull out y data
                    LOUTCOMEs(1,s3L) = outcome(1,sep-1); %dt
                    s3L = s3L + 1; %formatting
                end %ending "if" statement finding hall 1 lefts
            end %ending "for" loop going through data
            s3R = 1; %formatting
            RX3 = []; %space
            RY3 = []; %space
            ROUTCOMEs = []; %space to line up each row with an EM
            for sep = 2:trialcount %going through again
                if EMs(1,sep) == 9 && EMs(2,sep) == 3 %right out of hall #...
                    RX3(s3R,:) = xcombo(sep-1,:); %pull out x data
                    RY3(s3R,:) = ycombo(sep-1,:); %pull out y data
                    ROUTCOMEs(1,s3R) = outcome(1,sep-1); %dt
                    s3R = s3R + 1; %formatting
                end %ending "if" statement finding hall 3 rights
            end %ending "for" loop going through data
            cutoffhigh = 120; %cutoff
            cutofflow = 90; %spout cutoff
            midline = 150; %midline
            for tc = 1:size(LX3,1) %going through trials
                for time = 1:length(LX3) %going through time
                    if LX3(tc,time) < cutofflow || LX3(tc,time) > cutoffhigh %if we are past the center cutoff & not at the spout...
                        LX3(tc,time) = NaN; %make it NaN
                        LX3(tc,time) = NaN; %make it NaN
                    end %ending "if" statement looking for center
                end %ending "for" loop going row by row
            end %ending "for" loop going column by column
            for tc = 1:size(RX3,1) %going through trials
                for time = 1:length(RX3) %going through time
                    if RX3(tc,time) < cutofflow || RX3(tc,time) > cutoffhigh %if we are past the center cutoff & not at the spout...
                        RX3(tc,time) = NaN; %make it NaN
                        RX3(tc,time) = NaN; %make it NaN
                    end %ending "if" statement looking for center
                end %ending "for" loop going row by row
            end %ending "for" loop going column by column
            for l = 1:size(LY3,1) %go through the lefts
                if nanmean(LY3(l,:)) < midline %if we have a committed left...
                    commits(1,c) = 1; %mark as committed
                    commits(2,c) = LOUTCOMEs(1,l); %copy outcome
                    c=c+1; %format
                else %if we have a switchie...
                    commits(1,c) = 0; %mark as committed
                    commits(2,c) = LOUTCOMEs(1,l); %copy outcome
                        c=c+1; %format
                end %ending "if" statement separating by committed
            end %ending "for" loop going through the lefts
            for l = 1:size(RY3,1) %go through the rights
                if nanmean(RY3(l,:)) > midline %if we have a committed right...
                    commits(1,c) = 1; %mark as committed
                    commits(2,c) = ROUTCOMEs(1,l); %copy outcome
                    c=c+1; %format
                else %if we have a switchie...
                    commits(1,c) = 0; %mark as committed
                        commits(2,c) = ROUTCOMEs(1,l); %copy outcome
                    c=c+1; %format
                end %ending "if" statement separating by committed
            end %ending "for" loop going through the right
            LR = []; %space
            for lr = 1:length(EMs) %go through the EMs
                if EMs(1,lr) == 9 %if we have a 9...
                    LR(lr) = 1; %change to 1
                else %if we have a left...
                    LR(lr) = 0; %change to a 0
                end %ending "if" statement for left vs right
            end %ending "for" loop going through EMs
            if ismember(overlap{3,session},rightmouse) %if mouse is preferring right...
                pref = 1; %we like right
            else %if not...
                pref = 0; %we like left
            end %ending "if" statement for preference
            commits(2,:) = LR(2:length(commits)+1); %fill in LR instead of outcome
            Pref_Com = 0; Pref_Switch = 0; Nonpref_Com = 0; Nonpref_Switch = 0; %starting at 0 for each count
            for fourtype = 1:length(commits) %go through commits
                if commits(1,fourtype) == 1 %if a commit...
                    if pref == 1 %if we have a right mouse...
                        if commits(2,fourtype) == 1 %if pref...
                            Pref_Com = Pref_Com + 1; %count
                        else %if nonpref...
                            Nonpref_Com = Nonpref_Com + 1; %count
                        end %ending "if" statement for rew vs unrew
                    else %if left mouse...
                        if commits(2,fourtype) == 0 %if pref...
                            Pref_Com = Pref_Com + 1; %count
                        else %if nonpref...
                            Nonpref_Com = Nonpref_Com + 1; %count
                        end %ending "if" statement for rew vs unrew
                    end %ending "if" statement for pref vs nonpref
                else %if a switch...
                    if pref == 1 %if we have a right mouse...
                        if commits(2,fourtype) == 1 %if pref...
                            Pref_Switch = Pref_Switch + 1; %count
                        else %if nonpref...
                            Nonpref_Switch = Nonpref_Switch + 1; %count
                        end %ending "if" statement for rew vs unrew
                    else %if left mouse...
                        if commits(2,fourtype) == 0 %if pref...
                            Pref_Switch = Pref_Switch + 1; %count
                        else %if nonpref...
                            Nonpref_Switch = Nonpref_Switch + 1; %count
                        end %ending "if" statement for rew vs unrew
                    end %ending "if" statement for pref vs nonpref
                end %ending "if" statement for com vs switch
            end %ending "for" loop going through commits dataset
            if environment == 1 && session == 1 %if we are in the first session of the first environment...
                savespot = 1; %we are saving in spot 1
            else %if we are any other...
                savespot = savespot+1; %move to the next spot
            end %ending "if" statement setting the savespot
            ALLDATA_reg(savespot,1) = Pref_Com / length(commits); %saving the pref com %
            ALLDATA_reg(savespot,2) = Nonpref_Com / length(commits); %saving nonpref com %
            ALLDATA_reg(savespot,3) = Pref_Switch / length(commits); %saving pref switch %
            ALLDATA_reg(savespot,4) = Nonpref_Switch / length(commits); %saving unrew switch %
            ALLDATA_reg(savespot,5) = overlap{3,session}; %store mouse ID
        end %ending "if" statement only doing all the calculation/saving for if the maze is 239
    end %ending "for" loop going through "overlap"
end %ending "for" loop going through all the environments ONLY DOING CONTROL HERE THOUGH

%% Switch into Bias vs Expected Bias from Session

ALLDATA = []; %space
rightmouse = [1 3 10 5 7 2 6 101]; %right mice
for environment = 1:6 %go through all environments doing the streamlined way to pull data out
    if environment == 1 %starting "if" statement to load each individual overlapped
        filename = load('C:\Users\julie\OneDrive\Documents\MATLAB\example data\NoConfLowVolControl_withDTs.mat');
    elseif environment == 2 %starting "if" statement to load each individual overlapped
        filename = load('C:\Users\julie\OneDrive\Documents\MATLAB\example data\LowConfLowVolControl_withDTs.mat');
    elseif environment == 3 %starting "if" statement to load each individual overlapped
        filename = load('C:\Users\julie\OneDrive\Documents\MATLAB\example data\HighConfLowVolControl_withDTs.mat');
    elseif environment == 4 %starting "if" statement to load each individual overlapped
        filename = load('C:\Users\julie\OneDrive\Documents\MATLAB\example data\NoConfHighVolControl_withDTs.mat');
    elseif environment == 5 %starting "if" statement to load each individual overlapped
        filename = load('C:\Users\julie\OneDrive\Documents\MATLAB\example data\LowConfHighVolControl_withDTs.mat');
    elseif environment == 6 %starting "if" statement to load each individual overlapped
        filename = load('C:\Users\julie\OneDrive\Documents\MATLAB\example data\HighConfHighVolControl_withDTs.mat');
    end %ending "if" statement for environments
    overlap = filename.overlap; %pull out cells
    for session = 1:length(overlap) %go through "overlap"
        data = overlap{1,session}; %pulling out raw data
        xy = []; %place to store location data
        xy(1,:) = data(:,13); %copy over the x info from data file
        xy(2,:) = data(:,14); %copy over y info from data file
        if min(xy(2,:)) < 40 %this should be a good way to XY determine which maze...
            maze = 239; %set the maze data to 239
        else %if we have the other orientation...
            maze = 241; %set maze data to 241
        end %ending "if" statement trying to mathematically separate mazes
        if maze == 239 %we are only running the analysis for 239 mice
            disp3=sqrt(diff2(xy(1,:),3).^2+diff2(xy(2,:),3).^2);  %equation from Eric for velocity
            turn1 = smooth(disp3,30)'; %smooth the velocity data
            turn1(2,:) = data(1:end-3,17); %fill in the second part of velocity data with the hall location
            stuff1 = NaN(length(turn1),1); %place to store hall 1
            stuff2 = NaN(length(turn1),1); %place to store hall 2
            stuff3 = NaN(length(turn1),1); %place to store hall 3
            stuff0 = NaN(length(turn1),1); %place to store center
            for splittingup = 1:length(turn1) %going through the velocity data
                if turn1(2,splittingup) == 1 %if the animal was in hall 1...
                    stuff1(splittingup) = turn1(1,splittingup); %fill that in stuff 1
                elseif turn1(2,splittingup) == 2 %if the animal was in hall 2...
                    stuff2(splittingup) = turn1(1,splittingup); %fill that in stuff 2
                elseif turn1(2,splittingup) == 3 %if the animal was in hall 3...
                    stuff3(splittingup) = turn1(1,splittingup); %fill that in stuff 3
                elseif turn1(2,splittingup) == 0 %if the animal was in the center...
                    stuff0(splittingup) = turn1(1,splittingup); %fill that in stuff 0
                end %ending "if" statement looking for location
            end %ending "for" loop going through velocity data
            stuff1(:,2) = data(1:end-3,1); %add in time info
            stuff2(:,2) = data(1:end-3,1); %add in time info
            stuff3(:,2) = data(1:end-3,1); %add in time info
            stuff0(:,2) = data(1:end-3,1); %add in time info
            turn1(3,:) = data(1:end-3,2); %fill in the EMs to turn1
            turn1(4,:) = 1:length(turn1); %fill in a spacer
            allempositions = []; %place to store all EM positions
            emspot = 1; %formatting
            ems = [1:8]; %any em
            for stars = 2:length(turn1) %going through the data
                if ismember(turn1(3,stars),ems) && turn1(3,stars-1) == 0 %if a turn is first detected...
                    allempositions(emspot) = turn1(4,stars); %mark down that spot position number
                    emspot = emspot + 1; %formatting
                end %ending "if" statement looking for ems
            end %ending "for" loop gooing through the data
            veltimes = []; %place to store a lot of the info
            veltimes(1,:) = turn1(1,:); %copy over the velocities
            veltimes(2,:) = data(1:end-3,1); %copy over the times
            veltimes(3,:) = 1:length(veltimes); %copy over the position # (to go with allempositions)
            cutoffvalues = []; %place to store the cutoff velocity for each turn
            hopethisworks = []; %going to try finding the min each time & then clearing it
            cutofftracker = 1; %formatting
            for merp = 1:length(allempositions)-1 %going through all the marked turns except for the last one
                hopethisworks = veltimes(1,allempositions(merp):allempositions(merp+1)); %temporarily copy over the vels between two turns
                sorted = sort(hopethisworks); %sort them from low to high
                cutoff = sorted(round(length(sorted)*0.15)); %find the velocity that cuts off the slowest 15% of values
                cutoffvalues(cutofftracker) = cutoff; %copy over the cutoff value
                cutofftracker = cutofftracker + 1; %formatting
                hopethisworks = []; %clearing for the next run
            end %ending "for" loop going through the empositions
            for eek = 1:length(allempositions)-1 %going through the ems
                for hoping = 1:length(veltimes) %going through veltimes
                    if (allempositions(eek) < veltimes(3,hoping)) && (veltimes(3,hoping) < allempositions(eek + 1)) %if we are at a position between two marked turns...
                        if veltimes(1,hoping) < cutoffvalues(eek) %if the velocity falls below the cutoff of what we are calling "decision time"...
                            veltimes(4,hoping) = 1; %mark it as a 1 in the bottom row
                        end %ending "if" statement looking for the values falling below cutoff
                    end %ending "if" statement looking at vels that fall between the EMs
                end %ending "for" loop going through veltimes
            end %ending "for" loop going through allempositions
            crossing = []; %storing when the threshold is crossed for the first time
            crossingdown = []; %slowing down
            downspot = 1; %formatting
            for firstinstance = 1:length(allempositions)-1 %going through the ems
                crossing(1,:) = veltimes(4,allempositions(firstinstance):allempositions(firstinstance + 1));  %temporarily copy over the 0s & 1s
                crossing(2,:) = veltimes(3,allempositions(firstinstance):allempositions(firstinstance + 1)); %temporarily copy over the positions
                for cutoffsearch = 2:length(crossing)-1 %going through crossing (second through second to last value)
                    if crossing(1,cutoffsearch) == 1 && crossing(1,cutoffsearch - 1) == 0 %if the threshold is crossed going down...
                        crossing(1,cutoffsearch) = 2; %change its value to a 2
                    end %ending "if" statement looking for when the threshold is crossed
                end %ending "for" loop going through crossing
                for separating = 1:length(crossing) %going through crossing
                    if crossing(1,separating) == 2 %if the mouse crossed threshold going down...
                        crossingdown(downspot) = crossing(2,separating); %copy over the position
                        downspot = downspot + 1; %formatting
                    end %ending "if" statement looking for slowing down
                end %ending "for" loop going through crossing
                    downtime = min(crossingdown); %first time slowing down
                for finaltry = 1:length(crossing) %going through crossing
                    if crossing(2,finaltry) == downtime %if it is down time...
                        crossing(1,finaltry) = 4; %change that value to a 4
                    else %otherwise...
                        crossing(1,finaltry) = 0; %change everything to a 0
                    end %ending "if" statement to find exact start/stop
                end %ending "for" loop going through crossing   
                veltimes(4,allempositions(firstinstance):allempositions(firstinstance + 1)) = crossing(1,:); %replace the veltimes with the cleared data
                crossing = []; crossingdown = []; %clear out this for next run
                downspot = 1; %formatting
            end %ending "for" loop going through to mark only first time a min is reached
            start = []; %decision time start
            t = 1; %formatting
            for testing = 1:length(veltimes) %going through veltimes
                if veltimes(4,testing) == 4 %if it is the start of decision time...
                    start(t,1) = veltimes(1,testing); %copy over vel
                    start(t,2) = veltimes(2,testing); %copy over time
                    t = t + 1; %formatting
                end %ending "if" statement looking for starts
            end %ending "for" loop going through veltimes
            timebracket = []; %going to try doing mins & max's for time to look at
            timebracket(:,1) = start(:,2);
            for TB = 2:length(allempositions) %this has already found the lines in data when turns are detected
                timebracket(TB-1,2) = data(allempositions(TB),1); %copy the times from those lines into the bracket
            end %ending "for" loop looking for EM timing
            trialcount = data(end,11); %number of trials done
            x = NaN(length(data),trialcount); %place for x
            y = NaN(length(data),trialcount); %place for y
            for gothrough = 1:length(data) %going through data
                if data(gothrough,11) > 0 && data(gothrough,11) < length(timebracket)+1 %getting through the first trial but also not doing one for after the last...
                    if timebracket(data(gothrough,11),1) <= data(gothrough,1) && data(gothrough,1) <= timebracket(data(gothrough,11),2) %if it falls within the time frame associated for that trial...
                        x(gothrough,data(gothrough,11)) = data(gothrough,13); %copy over the x info in a column by column basis for trial #
                        y(gothrough,data(gothrough,11)) = data(gothrough,14); %copy over the y info in a column by column basis for trial #
                    end %ending "if" statement trying to get only bracketed times
                end %ending "if" statement making sure we get through the first trial & not doing the very last one bc we don't know what the next choice would have been
            end %ending "for" loop going through data
            xcombo = []; %combining the info
            for combo = 1:trialcount-1 %going for each trial
                col = x(:,combo); %pull out just the column for each trial
                col2 = col(~isnan(col)); %take out the nans
                if combo > 1 %if we are past the first trial...
                    if length(col2) < size(xcombo,2) %if we have a shorter series of data...
                        col2(end+1:length(xcombo)) = NaN; %fill it in with NaN
                    end %ending "if" statement fixing shorter series
                    if length(col2) > size(xcombo,2) %if we have a longer series of data...
                        xcombo(:,end:length(col2)) = NaN; %fill in xcombo with NaN
                    end %ending "if" statement fixing longer series
                end %ending "if" statement getting past first trial
                xcombo(combo,:) = col2; %copy over
            end
            ycombo = []; %combining the info
            for combo = 1:trialcount-1 %going for each trial
                col = y(:,combo); %pull out just the column for each trial
                col2 = col(~isnan(col)); %take out the nans
                if combo > 1 %if we are past the first trial...
                    if length(col2) < size(ycombo,2) %if we have a shorter series of data...
                        col2(end+1:length(ycombo)) = NaN; %fill it in with NaN
                    end %ending "if" statement fiying shorter series
                    if length(col2) > size(ycombo,2) %if we have a longer series of data...
                        ycombo(:,end:length(col2)) = NaN; %fill in ycombo with NaN
                    end %ending "if" statement fiying longer series
                end %ending "if" statement getting past first trial
                ycombo(combo,:) = col2; %copy over
            end
            EM = data(:,2); %makes a variable for just EMs
            wantedEM = []; %making an open vector for EMs that we want
            for q = 2:length(EM) %starting with second value & running through all EMs
                if EM(q)~=EM(q-1) %if an EM is different from the one preceding it...
                    wantedEM = [wantedEM,EM(q)]; %...include those EMs in the wantedEM vector
                end %ending the "if" statement
            end %ending the "for" loop
            wantedEM(ismember(wantedEM,[0,11,12])) = []; %retains only EMs pertaining to turns
            EMs = wantedEM; %pull out the EMs
            left = [1:4]; %left EMs
            for em = 1:length(EMs) %going through EMs
                if ismember(EMs(em),left) %if it is a left...
                    EMs(em) = 0; %set it to 0
                else %if it is a right...
                    EMs(em) = 9; %set it to 9 (want to be out of bounds of normal EMs)
                end %ending "if" statement separating L/R
            end %ending "for" loop going through EMs
            EMs(2,1) = data(1,17); %fill in the first one
            counter = 2; %moving hall to EM info
            for hall = 2:length(data) %going to try sep by halls too
                if data(hall,2) ~= data(hall-1,2) && data(hall-1,2) == 0 %when a turn is detected...
                    EMs(2,counter) = data(hall,17); %pull over the hall number turned out of to the em file IMPORTANT THAT IT IS WHERE THEY TURNED OUT OF & NEXT HALL IS WHERE THEY TURNED INTO!
                    counter = counter + 1; %formatting
                end %ending "if" statement looking for turns
            end %ending "for" loop going through data
            commits = []; %space
            c = 1; %format
            s1L = 1; %formatting
            LX1 = []; %space
            LY1 = []; %space
            LOUTCOMEs = []; %space to line up each row with an EM for rewarded vs unrewarded
            rew = [1 3 5 7]; %rewarded EMs
            outcome = []; %space
            for o = 1:length(wantedEM) %go through EMs
                if ismember(wantedEM(o),rew) %if rewarded...
                    outcome(o) = 1; %mark 1
                else %if unrewarded...
                    outcome(o) = 0; %mark 0
                end %ending "if" statement separating by outcome
            end %ending "for" loop going through EMs
            for sep = 2:trialcount %going through again
                if EMs(1,sep) == 0 && EMs(2,sep) == 1 %left out of hall #...
                    LX1(s1L,:) = xcombo(sep-1,:); %pull out x data
                    LY1(s1L,:) = ycombo(sep-1,:); %pull out y data
                    LOUTCOMEs(1,s1L) = outcome(1,sep-1); %copy outcome that lead to that turn
                    s1L = s1L + 1; %formatting
                end %ending "if" statement finding hall 1 lefts
            end %ending "for" loop going through data
            s1R = 1; %formatting
            RX1 = []; %space
            RY1 = []; %space
            ROUTCOMEs = []; %space to line up each row with an EM
            for sep = 2:trialcount %going through again
                if EMs(1,sep) == 9 && EMs(2,sep) == 1 %right out of hall #...
                    RX1(s1R,:) = xcombo(sep-1,:); %pull out x data
                    RY1(s1R,:) = ycombo(sep-1,:); %pull out y data
                    ROUTCOMEs(1,s1R) = outcome(1,sep-1); %copy outcome of that trial
                    s1R = s1R + 1; %formatting
                end %ending "if" statement finding hall 1 rights
            end %ending "for" loop going through data
            cutoffhigh = 110; %hall 1 cutoff
            cutofflow = 40; %hall 1 spout cutoff
            midline = 160; %midline
                for tc = 1:size(LY1,1) %going through trials
                    for time = 1:length(LY1) %going through time
                        if LY1(tc,time) < cutofflow || LY1(tc,time) > cutoffhigh %if we are past the center cutoff & not at the spout...
                            LY1(tc,time) = NaN; %make it NaN
                            LX1(tc,time) = NaN; %make it NaN
                        end %ending "if" statement looking for center
                    end %ending "for" loop going row by row
                end %ending "for" loop going column by column
                for tc = 1:size(RY1,1) %going through trials
                    for time = 1:length(RY1) %going through time
                        if RY1(tc,time) < cutofflow || RY1(tc,time) > cutoffhigh %if we are past the center cutoff & not at the spout...
                            RY1(tc,time) = NaN; %make it NaN
                            RX1(tc,time) = NaN; %make it NaN
                        end %ending "if" statement looking for center
                    end %ending "for" loop going row by row
                end %ending "for" loop going column by column
                for l = 1:size(LX1,1) %go through the lefts
                    if nanmean(LX1(l,:)) > midline %if we have a committed left...
                        commits(1,c) = 1; %mark as committed
                        commits(2,c) = LOUTCOMEs(1,l); %copy outcome
                        c=c+1; %format
                    else %if we have a switchie...
                        commits(1,c) = 0; %mark as committed
                        commits(2,c) = LOUTCOMEs(1,l); %copy outcome
                        c=c+1; %format
                    end %ending "if" statement separating by committed
                end %ending "for" loop going through the lefts
                for l = 1:size(RX1,1) %go through the rights
                    if nanmean(RX1(l,:)) < midline %if we have a committed right...
                        commits(1,c) = 1; %mark as committed
                        commits(2,c) = ROUTCOMEs(1,l); %copy outcome
                        c=c+1; %format
                    else %if we have a switchie...
                        commits(1,c) = 0; %mark as committed
                        commits(2,c) = ROUTCOMEs(1,l); %copy outcome
                        c=c+1; %format
                    end %ending "if" statement separating by committed
                end %ending "for" loop going through the rights
            s2L = 1; %formatting
            LX2 = []; %space
            LY2 = []; %space
            LOUTCOMEs = []; %space to line up each row with an EM
            for sep = 2:trialcount %going through again
                if EMs(1,sep) == 0 && EMs(2,sep) == 2 %left out of hall #...
                    LX2(s2L,:) = xcombo(sep-1,:); %pull out x data
                    LY2(s2L,:) = ycombo(sep-1,:); %pull out y data
                    LOUTCOMEs(1,s2L) = outcome(1,sep-1); %dt
                    s2L = s2L + 1; %formatting
                end %ending "if" statement finding hall 1 lefts
            end %ending "for" loop going through data
            s2R = 1; %formatting
            RX2 = []; %space
            RY2 = []; %space
            ROUTCOMEs = []; %space to line up each row with an EM
            for sep = 2:trialcount %going through again
                if EMs(1,sep) == 9 && EMs(2,sep) == 2 %right out of hall #...
                    RX2(s2R,:) = xcombo(sep-1,:); %pull out x data
                    RY2(s2R,:) = ycombo(sep-1,:); %pull out y data
                    ROUTCOMEs(1,s2R) = outcome(1,sep-1); %dt
                    s2R = s2R + 1; %formatting
                end %ending "if" statement finding hall 1 rights
            end %ending "for" loop going through data
            cutoffhigh = 230; %cutoff
            cutofflow = 180; %spout cutoff
            midline = 150; %midline
            for tc = 1:size(LX2,1) %going through trials
                for time = 1:length(LX2) %going through time
                    if LX2(tc,time) < cutofflow || LX2(tc,time) > cutoffhigh %if we are past the center cutoff & not at the spout...
                        LX2(tc,time) = NaN; %make it NaN
                        LX2(tc,time) = NaN; %make it NaN
                    end %ending "if" statement looking for center
                end %ending "for" loop going row by row
            end %ending "for" loop going column by column
            for tc = 1:size(RX2,1) %going through trials
                for time = 1:length(RX2) %going through time
                    if RX2(tc,time) < cutofflow || RX2(tc,time) > cutoffhigh %if we are past the center cutoff & not at the spout...
                        RX2(tc,time) = NaN; %make it NaN
                        RX2(tc,time) = NaN; %make it NaN
                    end %ending "if" statement looking for center
                end %ending "for" loop going row by row
            end %ending "for" loop going column by column
            for l = 1:size(LY2,1) %go through the lefts
                if nanmean(LY2(l,:)) > midline %if we have a committed left...
                    commits(1,c) = 1; %mark as committed
                    commits(2,c) = LOUTCOMEs(1,l); %copy outcome
                    c=c+1; %format
                else %if we have a switchie...
                    commits(1,c) = 0; %mark as committed
                    commits(2,c) = LOUTCOMEs(1,l); %copy outcome
                    c=c+1; %format
                end %ending "if" statement separating by committed
            end %ending "for" loop going through the lefts
            for l = 1:size(RY2,1) %go through the rights
                if nanmean(RY2(l,:)) < midline %if we have a committed right...
                    commits(1,c) = 1; %mark as committed
                    commits(2,c) = ROUTCOMEs(1,l); %copy outcome
                    c=c+1; %format
                else %if we have a switchie...
                    commits(1,c) = 0; %mark as committed
                    commits(2,c) = ROUTCOMEs(1,l); %copy outcome
                    c=c+1; %format
                end %ending "if" statement separating by committed
            end %ending "for" loop going through the rights
            s3L = 1; %formatting
            LX3 = []; %space
            LY3 = []; %space
            LOUTCOMEs = []; %space to line up each row with an EM
            for sep = 2:trialcount %going through again
                if EMs(1,sep) == 0 && EMs(2,sep) == 3 %left out of hall #...
                    LX3(s3L,:) = xcombo(sep-1,:); %pull out x data
                    LY3(s3L,:) = ycombo(sep-1,:); %pull out y data
                    LOUTCOMEs(1,s3L) = outcome(1,sep-1); %dt
                    s3L = s3L + 1; %formatting
                end %ending "if" statement finding hall 1 lefts
            end %ending "for" loop going through data
            s3R = 1; %formatting
            RX3 = []; %space
            RY3 = []; %space
            ROUTCOMEs = []; %space to line up each row with an EM
            for sep = 2:trialcount %going through again
                if EMs(1,sep) == 9 && EMs(2,sep) == 3 %right out of hall #...
                    RX3(s3R,:) = xcombo(sep-1,:); %pull out x data
                    RY3(s3R,:) = ycombo(sep-1,:); %pull out y data
                    ROUTCOMEs(1,s3R) = outcome(1,sep-1); %dt
                    s3R = s3R + 1; %formatting
                end %ending "if" statement finding hall 3 rights
            end %ending "for" loop going through data
            cutoffhigh = 120; %cutoff
            cutofflow = 90; %spout cutoff
            midline = 150; %midline
            for tc = 1:size(LX3,1) %going through trials
                for time = 1:length(LX3) %going through time
                    if LX3(tc,time) < cutofflow || LX3(tc,time) > cutoffhigh %if we are past the center cutoff & not at the spout...
                        LX3(tc,time) = NaN; %make it NaN
                        LX3(tc,time) = NaN; %make it NaN
                    end %ending "if" statement looking for center
                end %ending "for" loop going row by row
            end %ending "for" loop going column by column
            for tc = 1:size(RX3,1) %going through trials
                for time = 1:length(RX3) %going through time
                    if RX3(tc,time) < cutofflow || RX3(tc,time) > cutoffhigh %if we are past the center cutoff & not at the spout...
                        RX3(tc,time) = NaN; %make it NaN
                        RX3(tc,time) = NaN; %make it NaN
                    end %ending "if" statement looking for center
                end %ending "for" loop going row by row
            end %ending "for" loop going column by column
            for l = 1:size(LY3,1) %go through the lefts
                if nanmean(LY3(l,:)) < midline %if we have a committed left...
                    commits(1,c) = 1; %mark as committed
                    commits(2,c) = LOUTCOMEs(1,l); %copy outcome
                    c=c+1; %format
                else %if we have a switchie...
                    commits(1,c) = 0; %mark as committed
                    commits(2,c) = LOUTCOMEs(1,l); %copy outcome
                        c=c+1; %format
                end %ending "if" statement separating by committed
            end %ending "for" loop going through the lefts
            for l = 1:size(RY3,1) %go through the rights
                if nanmean(RY3(l,:)) > midline %if we have a committed right...
                    commits(1,c) = 1; %mark as committed
                    commits(2,c) = ROUTCOMEs(1,l); %copy outcome
                    c=c+1; %format
                else %if we have a switchie...
                    commits(1,c) = 0; %mark as committed
                        commits(2,c) = ROUTCOMEs(1,l); %copy outcome
                    c=c+1; %format
                end %ending "if" statement separating by committed
            end %ending "for" loop going through the right
            LR = []; %space
            for lr = 1:length(EMs) %go through the EMs
                if EMs(1,lr) == 9 %if we have a 9...
                    LR(lr) = 1; %change to 1
                else %if we have a left...
                    LR(lr) = 0; %change to a 0
                end %ending "if" statement for left vs right
            end %ending "for" loop going through EMs
            if ismember(overlap{3,session},rightmouse) %if mouse is preferring right...
                pref = 1; %we like right
            else %if not...
                pref = 0; %we like left
            end %ending "if" statement for preference
            commits(2,:) = LR(2:length(commits)+1); %fill in LR instead of outcome
            Pref_Com = 0; Pref_Switch = 0; Nonpref_Com = 0; Nonpref_Switch = 0; %starting at 0 for each count
            for fourtype = 1:length(commits) %go through commits
                if commits(1,fourtype) == 1 %if a commit...
                    if pref == 1 %if we have a right mouse...
                        if commits(2,fourtype) == 1 %if pref...
                            Pref_Com = Pref_Com + 1; %count
                        else %if nonpref...
                            Nonpref_Com = Nonpref_Com + 1; %count
                        end %ending "if" statement for rew vs unrew
                    else %if left mouse...
                        if commits(2,fourtype) == 0 %if pref...
                            Pref_Com = Pref_Com + 1; %count
                        else %if nonpref...
                            Nonpref_Com = Nonpref_Com + 1; %count
                        end %ending "if" statement for rew vs unrew
                    end %ending "if" statement for pref vs nonpref
                else %if a switch...
                    if pref == 1 %if we have a right mouse...
                        if commits(2,fourtype) == 1 %if pref...
                            Pref_Switch = Pref_Switch + 1; %count
                        else %if nonpref...
                            Nonpref_Switch = Nonpref_Switch + 1; %count
                        end %ending "if" statement for rew vs unrew
                    else %if left mouse...
                        if commits(2,fourtype) == 0 %if pref...
                            Pref_Switch = Pref_Switch + 1; %count
                        else %if nonpref...
                            Nonpref_Switch = Nonpref_Switch + 1; %count
                        end %ending "if" statement for rew vs unrew
                    end %ending "if" statement for pref vs nonpref
                end %ending "if" statement for com vs switch
            end %ending "for" loop going through commits dataset
            total_switch = Nonpref_Switch + Pref_Switch; %total switches
            pref_percent = Pref_Switch/total_switch; %proportion of switch
            if ismember(overlap{3,session},rightmouse) %if pref right...
                BIAS = [5 6 7 8]; %bias EM
            else %if pref left...
                BIAS = [1 2 3 4]; %bias EM
            end %ending "if" statement setting EMs
            for b = 1:length(wantedEM) %go through the EMs
                if ismember(wantedEM(b),BIAS) %if pref...
                    wantedEM(b) = 1; %change to 1
                else %if nonpref...
                    wantedEM(b) = 0; %change to 0
                end %ending "if" statement for pref vs nonpref
            end %ending "for" loop going through EMs
            bias_percent = sum(wantedEM)/length(wantedEM); %bias percent
            diff_from_expected = pref_percent - bias_percent; %how different is switch breakdown from total trial breakdown
            if environment == 1 && session == 1 %first session from first environment...
                savespot = 1; %we start in first spot
            else %if we are beyond the first session...
                savespot = savespot+1; %move down one spot for saving the data
            end %ending "if" statement figuring out where to save the data
            ALLDATA(savespot,1) = diff_from_expected; %save diff_from_expected
            ALLDATA(savespot,2) = overlap{3,session}; %save mouse ID
        end %ending "if" statement for only maze 239
    end %ending "for" loop going through sessions
end %ending "for" loop going through environment

%% bias/error correlation

ALLDATA = []; %space
for environment = 1:6 %go through all environments doing the streamlined way to pull data out
    if environment == 1 %starting "if" statement to load each individual overlapped
        filename = load('C:\Users\julie\OneDrive\Documents\MATLAB\example data\NoConfLowVolControl_withDTs.mat');
    elseif environment == 2 %starting "if" statement to load each individual overlapped
        filename = load('C:\Users\julie\OneDrive\Documents\MATLAB\example data\LowConfLowVolControl_withDTs.mat');
    elseif environment == 3 %starting "if" statement to load each individual overlapped
        filename = load('C:\Users\julie\OneDrive\Documents\MATLAB\example data\HighConfLowVolControl_withDTs.mat');
    elseif environment == 4 %starting "if" statement to load each individual overlapped
        filename = load('C:\Users\julie\OneDrive\Documents\MATLAB\example data\NoConfHighVolControl_withDTs.mat');
    elseif environment == 5 %starting "if" statement to load each individual overlapped
        filename = load('C:\Users\julie\OneDrive\Documents\MATLAB\example data\LowConfHighVolControl_withDTs.mat');
    elseif environment == 6 %starting "if" statement to load each individual overlapped
        filename = load('C:\Users\julie\OneDrive\Documents\MATLAB\example data\HighConfHighVolControl_withDTs.mat');
    end %ending "if" statement for environments
    overlap = filename.overlap; %pull out cells
    for session = 1:length(overlap) %go through "overlap"
        data = []; %re-setting "data" from it being used earlier in the code
        data.days = 1; %the number of sessions you are using
        data2 = overlap{1,session}; %pull out raw data
        EM = data2(:,2); %makes a variable for just EMs
        wantedEM = []; %making an open vector for EMs that we want
        for q = 2:length(EM) %starting with second value & running through all EMs
            if EM(q)~=EM(q-1) %if an EM is different from the one preceding it...
            wantedEM = [wantedEM,EM(q)]; %...include those EMs in the wantedEM vector
            end %ending the "if" statement
        end %ending the "for" loop
        wantedEM(ismember(wantedEM,[0,11,12])) = []; %retains only EMs pertaining to turns
        rew = [1 3 5 7]; %rewarded EMs
        data.trials = length(wantedEM); %the relevant trials
        data.rewards = []; %place to store reward info
        for rrr = 1:length(wantedEM) %going through the EMs
            if ismember(wantedEM(rrr),rew) %if it is a rewarded trial...
                data.rewards(rrr) = 1; %tag it with a 1
            else %otherwise...
                data.rewards(rrr) = 0; %tag it with a 0
            end %ending "if" statement
        end %ending "for" loop
        rights = [5 6 7 8];
        data.choices = []; %place to store choice info
        for cc = 1:length(wantedEM) %going through the EMs
            if ismember(wantedEM(cc),rights) %if it is a right decision...
                data.choices(cc) = 1; %tag it with a 1
            else %otherwise...
                data.choices(cc) = 2; %tag it with a 2
            end %ending "if" statement
        end %ending "for" loop
        data.rewards = data.rewards'; %transposing the orientation
        data.choices = data.choices'; %transposing the orientation
        data.bs.raw = NaN(data.days, 3); %making NaNs for the number of days we are running for the number of betas being used
        for i=1:data.days %going through the number of days
            LLfit = @(betas) sum(-fit_Qvalue_V2(data.choices(:,i), data.rewards(:,i), betas, data.trials)); %use the choice, reward, & trial info to fit betas established in fit_Qvalue function
            [data.bs.raw(i,:), bvals] = fmincon(LLfit,[0.5, 5, 0],[],[],[],[],[0, 0.001, -10],[1, Inf, 10]); %establishing the limits for the values of the different betas
        end %ending "for" loop going through the days
        learning_rate = data.bs.raw(1); %printing the learning rate
        explore_exploit = data.bs.raw(2); %printing the beta
        Q_bias = data.bs.raw(3); %printing the Q bias term
        if Q_bias < 0 %if we have a negative value...
            Q_bias = abs(Q_bias); %make absolute value
        end %ending "if" statement fixing for diff. directions
        error = [3 4 7 8]; %error EMs
        for e = 1:length(wantedEM) %go through EMs
            if ismember(wantedEM(e),error) %if error...
                wantedEM(e) = 1; %mark 1
            else %if correct...
                wantedEM(e) = 0; %mark 0
            end %ending "if" statement for error vs correct
        end %ending "for" loop going through EMs
        ER = sum(wantedEM)/length(wantedEM); %error proportion
        if environment == 1 && session == 1 %first session from first environment...
            savespot = 1; %we start in first spot
        else %if we are beyond the first session...
            savespot = savespot+1; %move down one spot for saving the data
        end %ending "if" statement figuring out where to save the data
        ALLDATA(savespot,1) = Q_bias; %save bias
        ALLDATA(savespot,2) = ER; %save error
        ALLDATA(savespot,3) = overlap{3,session}; %save mouse ID
    end %ending "for" loop going through overlap
end %ending "for" loop going through environment

%% Random Lose-Stay vs Lose-Switch - exploration

for environment = 1:6 %go through all environments doing the streamlined way to pull data out
    if environment == 1 %starting "if" statement to load each individual overlapped
        filename = load('C:\Users\julie\OneDrive\Documents\MATLAB\example data\NoConfLowVolControl_withDTs.mat');
    elseif environment == 2 %starting "if" statement to load each individual overlapped
        filename = load('C:\Users\julie\OneDrive\Documents\MATLAB\example data\LowConfLowVolControl_withDTs.mat');
    elseif environment == 3 %starting "if" statement to load each individual overlapped
        filename = load('C:\Users\julie\OneDrive\Documents\MATLAB\example data\HighConfLowVolControl_withDTs.mat');
    elseif environment == 4 %starting "if" statement to load each individual overlapped
        filename = load('C:\Users\julie\OneDrive\Documents\MATLAB\example data\NoConfHighVolControl_withDTs.mat');
    elseif environment == 5 %starting "if" statement to load each individual overlapped
        filename = load('C:\Users\julie\OneDrive\Documents\MATLAB\example data\LowConfHighVolControl_withDTs.mat');
    elseif environment == 6 %starting "if" statement to load each individual overlapped
        filename = load('C:\Users\julie\OneDrive\Documents\MATLAB\example data\HighConfHighVolControl_withDTs.mat');
    end %ending "if" statement for environments
    overlap = filename.overlap; %pull out cells
    for session = 1:length(overlap) %go through "overlap"
        data = []; %clearing from regression
        data = overlap{1,session}; %pull out raw data
        wantedEM = []; %space
        wem = 1; %format
        for WEM = 2:length(data) %go through data
            if data(WEM,2) ~= data(WEM-1,2) && data(WEM-1,2) == 0 %if a turn is detected...
                wantedEM(wem) = data(WEM,2); %mark it
                wem=wem+1; %format
            end %ending "if" statement looking for turns
        end %ending "for" loop going through data
        choice = []; %space
        right = [5 6 7 8]; %right decision EMs
        for c = 1:length(wantedEM) %go through EMs
            if ismember(wantedEM(c),right) %if a right direction turn...
                choice(c) = 1; %mark 1
            else %if left...
                choice(c) = 0; %mark 0
            end %ending "if" statement for right direction turn
        end %ending "for" loop going through EMs
        outcome = []; %space
        reward = [1 3 5 7]; %reward decision EMs
        for r = 1:length(wantedEM) %go through EMs
            if ismember(wantedEM(r),reward) %if a reward direction turn...
                outcome(r) = 1; %mark 1
            else %if left...
                outcome(r) = 0; %mark 0
            end %ending "if" statement for reward direction turn
        end %ending "for" loop going through EMs
        lose = []; %space to store lose-stay vs lose-switch decisions
        l = 1; %format
        for L = 1:length(outcome)-1 %go through outcomes except the last one because we don't know what that next decision would be
            if outcome(L) == 0 %if unrewarded...
                if choice(L) ~= choice(L+1) %if switch...
                    lose(l) = 1; %mark 1 for lose-switch
                    l=l+1; %format
                else %if stay...
                    lose(l) = 0; %mark 0 for lose-stay
                    l=l+1; %format
                end %ending "if" statement for switch vs stay
            end %ending "if" statement for unrewarded outcomes
        end %ending "for" loop going through outcomes
        entropy = []; %space
        for e = 1:length(lose)-1 %go through lose data except the last one because we don't know what the next decision would be
            if lose(e) == lose(e+1) %if the same decision was made in a row...
                entropy(e) = 0; %0 score for entropy
            else %if a different decision was made in a row...
                entropy(e) = 1; %1 score for entropy
            end %ending "if" statement for repeated vs different decisions in a row
        end %ending "for" loop going through lose data
        if environment == 1 && session == 1 %first session from first environment...
            savespot = 1; %we start in first spot
        else %if we are beyond the first session...
            savespot = savespot+1; %move down one spot for saving the data
        end %ending "if" statement figuring out where to save the data
        ALLDATA(savespot,1) = sum(entropy)/length(entropy); %save entropy score
        ALLDATA(savespot,2) = overlap{3,session}; %save mouse ID
    end %ending "for" loop going through overlap
end %ending "for" loop going through environments

%% rewarded vs unrewarded randomness - exploration

ALLDATA = []; %space
for environment = 1:6 %go through all environments doing the streamlined way to pull data out
    if environment == 1 %starting "if" statement to load each individual overlapped
        filename = load('C:\Users\julie\OneDrive\Documents\MATLAB\example data\NoConfLowVolControl_withDTs.mat');
    elseif environment == 2 %starting "if" statement to load each individual overlapped
        filename = load('C:\Users\julie\OneDrive\Documents\MATLAB\example data\LowConfLowVolControl_withDTs.mat');
    elseif environment == 3 %starting "if" statement to load each individual overlapped
        filename = load('C:\Users\julie\OneDrive\Documents\MATLAB\example data\HighConfLowVolControl_withDTs.mat');
    elseif environment == 4 %starting "if" statement to load each individual overlapped
        filename = load('C:\Users\julie\OneDrive\Documents\MATLAB\example data\NoConfHighVolControl_withDTs.mat');
    elseif environment == 5 %starting "if" statement to load each individual overlapped
        filename = load('C:\Users\julie\OneDrive\Documents\MATLAB\example data\LowConfHighVolControl_withDTs.mat');
    elseif environment == 6 %starting "if" statement to load each individual overlapped
        filename = load('C:\Users\julie\OneDrive\Documents\MATLAB\example data\HighConfHighVolControl_withDTs.mat');
    end %ending "if" statement for environments
    overlap = filename.overlap; %pull out cells
    for session = 1:length(overlap) %go through "overlap"
        data = []; %clearing from regression
        data = overlap{1,session}; %pull out raw data
        wantedEM = []; %space for EMs
        w = 1; %format
        left = [1 2 3 4]; %left EMs
        rew = [1 3 5 7]; %rewarded EMs
        for EM = 2:length(data) %go through the raw data file
            if data(EM,2) ~= data(EM-1,2) && data(EM-1,2) == 0 %when a turn happens...
                wantedEM(w) = data(EM,2); %copy DTs
                w=w+1; %format
            end %ending "if" statement looking for turn detection
        end %ending "for" loop going through raw data file
        RL_binary = []; RU_binary = []; %format
        for rlb = 1:length(wantedEM) %go through EMs
            if ismember(wantedEM(rlb),left) %if left...
                RL_binary(rlb) = 0; %mark as 0
            else %if right...
                RL_binary(rlb) = 1; %mark as 0
            end %ending "if" statement separating left & right
        end %ending "for" loop going through the EMs
        for rub = 1:length(wantedEM) %go through EMs
            if ismember(wantedEM(rub),rew) %if rewarded...
                RU_binary(rub) = 1; %mark as 1
            else %if unrewarded...
                RU_binary(rub) = 0; %mark as 0
            end %ending "if" statement separating left & right
        end %ending "for" loop going through the EMs
        if session == 1 %if we are in the first session...
            EM_length = length(wantedEM)-1; %set the max EMs we can use (minus 1 because of the variability subtraction)
        else %if we are past session 1...
            if EM_length > length(wantedEM)-1 %if we have fewer EMs than what we said was our max...
                EM_length = length(wantedEM)-1; %reset to the new value
            end %ending "if" statement looking for new maximum
        end %ending "if" statement doing different rules by session number
        variability = []; %space to store choice variability
        for v = 2:length(RL_binary) %go through right left binary
            if RL_binary(v) == RL_binary(v-1) %if the same...
                variability(v-1) = 0; %0 variability score
            else %if different...
                variability(v-1) = 1; %1 variability score
            end %ending "if" statement separating same vs different choices
        end %ending "for" loop going through EMs
        rew_variability = []; unrew_variability = []; %space
        rv = 1; uv = 1; %format
        for trials = 1:length(RU_binary)-5 %go through RU binary (but not last 5 trials because we need at least 5 post for analysis)
            if RU_binary(trials) == 1 %if the trial was rewarded...
                rew_variability(rv,1:5) = variability(trials:trials+4); %fill in the five trials after that
                rv=rv+1; %format
            else %if the trial was unrewarded...
                unrew_variability(uv,1:5) = variability(trials:trials+4); %fill in the five trials after that
                uv=uv+1; %format
            end %ending "if" statement for rewarded vs unrewarded
        end %ending "for" loop going through RU binary
        for avging = 1:5 %go through each of the 5 trials
            REW(avging) = mean(rew_variability(:,avging)); %rewarded
            UNREW(avging) = mean(unrew_variability(:,avging)); %unrewarded
        end %ending "for" loop going through the 5 trials
        if environment == 1 && session == 1 %first session from first environment...
            savespot = 1; %we start in first spot
        else %if we are beyond the first session...
            savespot = savespot+1; %move down one spot for saving the data
        end %ending "if" statement figuring out where to save the data
        ALLDATA{savespot,1} = REW; %save rewarded
        ALLDATA{savespot,2} = UNREW; %save unrewarded
        ALLDATA{savespot,3} = overlap(3,session); %save mouse ID
    end %ending "for" loop going through overlap
end %ending "for" loop going through environments

%% Decision Time Reward Rate Correlation - exploration

ALLDATA = []; %space
for environment = 1:6 %go through all environments doing the streamlined way to pull data out
    if environment == 1 %starting "if" statement to load each individual overlapped
        filename = load('C:\Users\julie\OneDrive\Documents\MATLAB\example data\NoConfLowVolControl_withDTs.mat');
    elseif environment == 2 %starting "if" statement to load each individual overlapped
        filename = load('C:\Users\julie\OneDrive\Documents\MATLAB\example data\LowConfLowVolControl_withDTs.mat');
    elseif environment == 3 %starting "if" statement to load each individual overlapped
        filename = load('C:\Users\julie\OneDrive\Documents\MATLAB\example data\HighConfLowVolControl_withDTs.mat');
    elseif environment == 4 %starting "if" statement to load each individual overlapped
        filename = load('C:\Users\julie\OneDrive\Documents\MATLAB\example data\NoConfHighVolControl_withDTs.mat');
    elseif environment == 5 %starting "if" statement to load each individual overlapped
        filename = load('C:\Users\julie\OneDrive\Documents\MATLAB\example data\LowConfHighVolControl_withDTs.mat');
    elseif environment == 6 %starting "if" statement to load each individual overlapped
        filename = load('C:\Users\julie\OneDrive\Documents\MATLAB\example data\HighConfHighVolControl_withDTs.mat');
    end %ending "if" statement for environments
    overlap = filename.overlap; %pull out cells
    for session = 1:length(overlap) %go through "overlap"
        data = []; %clearing from regression
        data = overlap{1,session}; %pull out raw data
        xy = []; %place to store location data
        xy(1,:) = data(:,13); %copy over the x info from data file
        xy(2,:) = data(:,14); %copy over y info from data file
        disp3=sqrt(diff2(xy(1,:),3).^2+diff2(xy(2,:),3).^2);  %equation from Eric for velocity
        turn1 = smooth(disp3,30)'; %smooth the velocity data
        turn1(2,:) = data(1:end-3,17); %fill in the second part of velocity data with the hall location
        stuff1 = NaN(length(turn1),1); %place to store hall 1
        stuff2 = NaN(length(turn1),1); %place to store hall 2
        stuff3 = NaN(length(turn1),1); %place to store hall 3
        stuff0 = NaN(length(turn1),1); %place to store center
        for splittingup = 1:length(turn1) %going through the velocity data
            if turn1(2,splittingup) == 1 %if the animal was in hall 1...
                stuff1(splittingup) = turn1(1,splittingup); %fill that in stuff 1
            elseif turn1(2,splittingup) == 2 %if the animal was in hall 2...
                stuff2(splittingup) = turn1(1,splittingup); %fill that in stuff 2
            elseif turn1(2,splittingup) == 3 %if the animal was in hall 3...
                stuff3(splittingup) = turn1(1,splittingup); %fill that in stuff 3
            elseif turn1(2,splittingup) == 0 %if the animal was in the center...
                stuff0(splittingup) = turn1(1,splittingup); %fill that in stuff 0
            end %ending "if" statement looking for location
        end %ending "for" loop going through velocity data
        stuff1(:,2) = data(1:end-3,1); %add in time info
        stuff2(:,2) = data(1:end-3,1); %add in time info
        stuff3(:,2) = data(1:end-3,1); %add in time info
        stuff0(:,2) = data(1:end-3,1); %add in time info
        turn1(3,:) = data(1:end-3,2); %fill in the DTs to turn1
        turn1(4,:) = 1:length(turn1); %fill in a spacer
        allempositions = []; %place to store all EM positions
        emspot = 1; %formatting
        ems = [1:8]; %any em
        for stars = 2:length(turn1) %going through the data
            if ismember(turn1(3,stars),ems) && turn1(3,stars-1) == 0 %if a turn is first detected...
                allempositions(emspot) = turn1(4,stars); %mark down that spot position number
                emspot = emspot + 1; %formatting
            end %ending "if" statement looking for ems
        end %ending "for" loop gooing through the data
        veltimes = []; %place to store a lot of the info
        veltimes(1,:) = turn1(1,:); %copy over the velocities
        veltimes(2,:) = data(1:end-3,1); %copy over the times
        veltimes(3,:) = 1:length(veltimes); %copy over the position # (to go with allempositions)
        cutoffvalues = []; %place to store the cutoff velocity for each turn
        hopethisworks = []; %going to try finding the min each time & then clearing it
        cutofftracker = 1; %formatting
        for merp = 1:length(allempositions)-1 %going through all the marked turns except for the last one
            hopethisworks = veltimes(1,allempositions(merp):allempositions(merp+1)); %temporarily copy over the vels between two turns
            sorted = sort(hopethisworks); %sort them from low to high
            cutoff = sorted(round(length(sorted)*0.15)); %find the velocity that cuts off the slowest 15% of values
            cutoffvalues(cutofftracker) = cutoff; %copy over the cutoff value
            cutofftracker = cutofftracker + 1; %formatting
            hopethisworks = []; %clearing for the next run
        end %ending "for" loop going through the empositions
        for eek = 1:length(allempositions)-1 %going through the ems
            for hoping = 1:length(veltimes) %going through veltimes
                if (allempositions(eek) < veltimes(3,hoping)) && (veltimes(3,hoping) < allempositions(eek + 1)) %if we are at a position between two marked turns...
                    if veltimes(1,hoping) < cutoffvalues(eek) %if the velocity falls below the cutoff of what we are calling "decision time"...
                        veltimes(4,hoping) = 1; %mark it as a 1 in the bottom row
                    end %ending "if" statement looking for the values falling below cutoff
                end %ending "if" statement looking at vels that fall between the DTs
            end %ending "for" loop going through veltimes
        end %ending "for" loop going through allempositions
        crossing = []; %storing when the threshold is crossed for the first time
        crossingdown = []; %slowing down
        downspot = 1; %formatting
        for firstinstance = 1:length(allempositions)-1 %going through the ems
            crossing(1,:) = veltimes(4,allempositions(firstinstance):allempositions(firstinstance + 1));  %temporarily copy over the 0s & 1s
            crossing(2,:) = veltimes(3,allempositions(firstinstance):allempositions(firstinstance + 1)); %temporarily copy over the positions
            for cutoffsearch = 2:length(crossing)-1 %going through crossing (second through second to last value)
                if crossing(1,cutoffsearch) == 1 && crossing(1,cutoffsearch - 1) == 0 %if the threshold is crossed going down...
                    crossing(1,cutoffsearch) = 2; %change its value to a 2
                end %ending "if" statement looking for when the threshold is crossed
            end %ending "for" loop going through crossing
            for separating = 1:length(crossing) %going through crossing
                if crossing(1,separating) == 2 %if the mouse crossed threshold going down...
                    crossingdown(downspot) = crossing(2,separating); %copy over the position
                    downspot = downspot + 1; %formatting
                end %ending "if" statement looking for slowing down
            end %ending "for" loop going through crossing
            downtime = min(crossingdown); %first time slowing down
            for finaltry = 1:length(crossing) %going through crossing
                if crossing(2,finaltry) == downtime %if it is down time...
                    crossing(1,finaltry) = 4; %change that value to a 4
                else %otherwise...
                    crossing(1,finaltry) = 0; %change everything to a 0
                end %ending "if" statement to find exact start/stop
            end %ending "for" loop going through crossing   
            veltimes(4,allempositions(firstinstance):allempositions(firstinstance + 1)) = crossing(1,:); %replace the veltimes with the cleared data
            crossing = []; crossingdown = []; %clear out this for next run
            downspot = 1; %formatting
        end %ending "for" loop going through to mark only first time a min is reached
        start = []; %decision time start
        t = 1; %formatting
        for testing = 1:length(veltimes) %going through veltimes
            if veltimes(4,testing) == 4 %if it is the start of decision time...
                start(t,1) = veltimes(1,testing); %copy over vel
                start(t,2) = veltimes(2,testing); %copy over time
                t = t + 1; %formatting
            end %ending "if" statement looking for starts
        end %ending "for" loop going through veltimes
        cutoffvalues = []; %place to store the cutoff velocity for each turn SPEEDING BACK UP
        hopethisworks = []; %going to try finding the min each time & then clearing it
        cutofftracker = 1; %formatting
        for merp = 1:length(allempositions)-1 %going through all the marked turns except for the last one
            hopethisworks = veltimes(1,allempositions(merp):allempositions(merp+1)); %temporarily copy over the vels between two turns
            sorted = sort(hopethisworks); %sort them from low to high
            cutoff = sorted(round(length(sorted)*0.25)); %find the velocity that cuts off the slowest 25% of values
            cutoffvalues(cutofftracker) = cutoff; %copy over the cutoff value
            cutofftracker = cutofftracker + 1; %formatting
            hopethisworks = []; %clearing for the next run
        end %ending "for" loop going through the empositions
        for eek = 1:length(allempositions)-1 %going through the ems
            for hoping = 1:length(veltimes) %going through veltimes
                if (allempositions(eek) < veltimes(3,hoping)) && (veltimes(3,hoping) < allempositions(eek + 1)) %if we are at a position between two marked turns...
                    if veltimes(1,hoping) < cutoffvalues(eek) %if the velocity falls below the cutoff of what we are calling "decision time"...
                        veltimes(4,hoping) = 1; %mark it as a 1 in the bottom row
                    end %ending "if" statement looking for the values falling below cutoff
                end %ending "if" statement looking at vels that fall between the DTs
            end %ending "for" loop going through veltimes
        end %ending "for" loop going through allempositions
        crossing = []; %storing when the threshold is crossed for the last time
        crossingup = []; %speeding up
        upspot = 1; %formatting
        for firstinstance = 1:length(allempositions)-1 %going through the ems
            crossing(1,:) = veltimes(4,allempositions(firstinstance):allempositions(firstinstance + 1));  %temporarily copy over the 0s & 1s
            crossing(2,:) = veltimes(3,allempositions(firstinstance):allempositions(firstinstance + 1)); %temporarily copy over the positions
            for cutoffsearch = 2:length(crossing)-1 %going through crossing (second through second to last value)
                if crossing(1,cutoffsearch) == 1 && crossing (1,cutoffsearch + 1) == 0 %if the threshold is crossed going up...
                    crossing(1,cutoffsearch) = 3; %change its value to a 3
                end %ending "if" statements looking for when the threshold is crossed
            end %ending "for" loop going through crossing
            for separating = 1:length(crossing) %going through crossing
                if crossing(1,separating) == 3 %if the mouse crossed threshold going up...
                    crossingup(upspot) = crossing(2,separating); %copy over the position
                    upspot = upspot + 1; %formatting
                end %ending "if" statement looking for up
            end %ending "for" loop going through crossing
            uptime = max(crossingup); %final time speeding up
            for finaltry = 1:length(crossing) %going through crossing
                if crossing(2,finaltry) == uptime %if it is up time...
                    crossing(1,finaltry) = 5; %change that value to a 5
                else %otherwise...
                    crossing(1,finaltry) = 0; %change everything to a 0
                end %ending "if" statement to find exact start/stop
            end %ending "for" loop going through crossing   
            veltimes(4,allempositions(firstinstance):allempositions(firstinstance + 1)) = crossing(1,:); %replace the veltimes with the cleared data
            crossing = []; crossingup = []; %clear out this for next run
            upspot = 1; %formatting
        end %ending "for" loop going through to mark only first time a min is reached
        stop = []; %decision time stop
        p = 1; %formatting
        for testing = 1:length(veltimes) %going through veltimes
            if veltimes(4,testing) == 5 %if it is the end of decision time...
                stop(p,1) = veltimes(1,testing); %copy over vel
                stop(p,2) = veltimes(2,testing); %copy over time
                p = p + 1; %formatting
            end %ending "if" statement looking for sta[rts/stops
        end %ending "for" loop going through veltimes
        decisiontimes = []; %storing DTs
        decisiontimes(:,1) = start(:,2); %copy over the DT starts
        decisiontimes(:,2) = stop(1:length(start),2); %copy over the DT stops (based off start length bc getting weird bug sometimes where there are more stops than starts)
        for calculating = 1:length(decisiontimes) %going through DTs
            decisiontimes(calculating,3) = decisiontimes(calculating,2) - decisiontimes(calculating,1); %calculating DTs
        end %ending "for" loop going through DTs
        variability = []; %space to store choice variability
        for v = 2:length(decisiontimes) %go through DTs
            variability(1,v-1) = abs(decisiontimes(v,3) - decisiontimes(v-1,3)); %what is the variability between the DTs?
            variability(2,v-1) = decisiontimes(v,1); %fill in the time
        end %ending "for" loop going through DTs
        block = NaN(30,length(variability)); %space for each time block
        for timebin = 1:30 %doing one time bin per minute
            for trialnumber = 1:length(variability) %go through variability...
                if timebin == 1 %if we are in the first time bin...
                    if variability(2,trialnumber) < timebin*60*1000 %if we are in our time bin (translated to ms)...
                        block(timebin,trialnumber) = variability(1,trialnumber); %fill in the variability score in the associated row
                    end %ending "if" statement looking for our time bin
                else %if we are beyond the first time bin...
                    if variability(2,trialnumber) < timebin*60*1000 && variability(2,trialnumber) > (timebin-1)*60*1000 %if we are in our time bin (translated to ms)...
                        block(timebin,trialnumber) = variability(1,trialnumber); %fill in the variability score in the associated row
                    end %ending "if" statement looking for our time bin
                end %ending "if" statement doing the rules for first vs beyond first time bin
            end %ending "for" loop going through variability
        end %ending "for" loop going through time bins
        rew = [1 3 5 7]; %rewarded EMs
        RewEMs = []; %space
        rem = 1; %format
        for re = 2:length(data) %go through data
            if data(re,2) ~= data(re-1,2) && data(re-1,2) == 0 %when a turn happens...
                if ismember(data(re,2),rew) %if rewarded...
                    RewEMs(1,rem) = 1; %mark as 1
                    RewEMs(2,rem) = data(re,1); %mark time
                    rem=rem+1; %format
                else %if unrewarded...
                    RewEMs(1,rem) = 0; %mark as 0
                    RewEMs(2,rem) = data(re,1); %mark time
                    rem=rem+1; %format
                end %ending "if" statement separating rewarded from unrewarded
            end %ending "if" statement looking for EMs
        end %ending "for" loop going through data
        RewBlock = NaN(30,length(RewEMs)); %space for each time RewBlock
        for timebin = 1:30 %doing one time bin per minute
            for trialnumber = 1:length(RewEMs) %go through RewEMs...
                if timebin == 1 %if we are in the first time bin...
                    if RewEMs(2,trialnumber) < timebin*60*1000 %if we are in our time bin (translated to ms)...
                        RewBlock(timebin,trialnumber) = RewEMs(1,trialnumber); %fill in the RewEMs score in the associated row
                    end %ending "if" statement looking for our time bin
                else %if we are beyond the first time bin...
                    if RewEMs(2,trialnumber) < timebin*60*1000 && RewEMs(2,trialnumber) > (timebin-1)*60*1000 %if we are in our time bin (translated to ms)...
                        RewBlock(timebin,trialnumber) = RewEMs(1,trialnumber); %fill in the RewEMs score in the associated row
                    end %ending "if" statement looking for our time bin
                end %ending "if" statement doing the rules for first vs beyond first time bin
            end %ending "for" loop going through RewEMs
        end %ending "for" loop going through time bins
        align = []; %space
        for A = 1:30 %go through each min
            align(A,1) = nanmean(block(A,:)); %fill in mean variability score during that time block
            align(A,2) = nansum(RewBlock(A,:)); %fill in # of rewards gained in that time block
        end %ending "for" loop going through each min
        if session == 1 %if we are in the first session...
            EM_length = max(align(:,2)); %set the max RPMs we can use
        else %if we are past session 1...
            if EM_length > max(align(:,2)) %if we have fewer RPMs than what we said was our max...
                EM_length = max(align(:,2)); %reset to the new value
            end %ending "if" statement looking for new maximum
        end %ending "if" statement doing different rules by session number
        [correlationValue, pValue] = corr(align(:,1), align(:,2)); %calculate the correlation value & p-value
        CORR(1) = correlationValue; %store
        CORR(2) = pValue; %store
        if environment == 1 && session == 1 %first session from first environment...
            savespot = 1; %we start in first spot
        else %if we are beyond the first session...
            savespot = savespot+1; %move down one spot for saving the data
        end %ending "if" statement figuring out where to save the data
        ALLDATA{savespot,1} = align; %save variability score & # of rewards
        ALLDATA{savespot,2} = CORR; %save correlation & p value
        ALLDATA{savespot,3} = overlap(3,session); %save mouse ID
    end %ending "for" loop going through overlap
end %ending "for" loop going through environments

%% Center Reward Rate Correlation - exploration

ALLDATA = []; %space
for environment = 1:6 %go through all environments doing the streamlined way to pull data out
    if environment == 1 %starting "if" statement to load each individual overlapped
        filename = load('C:\Users\julie\OneDrive\Documents\MATLAB\example data\NoConfLowVolControl_withDTs.mat');
    elseif environment == 2 %starting "if" statement to load each individual overlapped
        filename = load('C:\Users\julie\OneDrive\Documents\MATLAB\example data\LowConfLowVolControl_withDTs.mat');
    elseif environment == 3 %starting "if" statement to load each individual overlapped
        filename = load('C:\Users\julie\OneDrive\Documents\MATLAB\example data\HighConfLowVolControl_withDTs.mat');
    elseif environment == 4 %starting "if" statement to load each individual overlapped
        filename = load('C:\Users\julie\OneDrive\Documents\MATLAB\example data\NoConfHighVolControl_withDTs.mat');
    elseif environment == 5 %starting "if" statement to load each individual overlapped
        filename = load('C:\Users\julie\OneDrive\Documents\MATLAB\example data\LowConfHighVolControl_withDTs.mat');
    elseif environment == 6 %starting "if" statement to load each individual overlapped
        filename = load('C:\Users\julie\OneDrive\Documents\MATLAB\example data\HighConfHighVolControl_withDTs.mat');
    end %ending "if" statement for environments
    overlap = filename.overlap; %pull out cells
    for session = 1:length(overlap) %go through "overlap"
        data = []; %clearing from regression
        data = overlap{1,session}; %pull out raw data
        times = 1; %format
        CE = []; %space
        for centerentry = 2:length(data) %go through the data
            if data(centerentry,17) == 0 && data(centerentry-1,17) ~= 0 %when the mouse enters the center...
                CE(times,1) = data(centerentry,1); %store the time
            elseif data(centerentry,17) ~= 0 && data(centerentry-1,17) == 0 %when the mouse leaves the center...
                CE(times,2) = data(centerentry,1); %store the time
                CE(times,3) = data(centerentry,11);
                times = times+1; %format
            end %ending "if" statement looking for when mouse enters the center
        end %ending "for" loop going through the data
        CE(times,3) = CE(times-1,3)+1; %fill in an extra EM marker for making sure we get one time point per EM
        actual = []; %space
        A = 1; %format
        for fixing = 1:length(CE)-1 %go through CE data outside of last fake data point
            if CE(fixing,3) < CE(fixing+1,3) %if it is the true center entry data...
                actual(A,1) = CE(fixing,1); %fill in entry time
                actual(A,2) = CE(fixing,2); %fill in exit time
                actual(A,3) = actual(A,2) - actual(A,1); %fill in time spent in center
                A=A+1; %format
            end %ending "if" statement looking for true center entry point
        end %ending "for" loop going through CE data
        variability = []; %space to store choice variability
        for v = 2:length(actual) %go through CEs
            variability(1,v-1) = abs(actual(v,3) - actual(v-1,3)); %what is the variability between the CEs?
            variability(2,v-1) = actual(v,1); %mark the time
        end %ending "for" loop going through CEs
        block = NaN(30,length(variability)); %space for each time block
        for timebin = 1:30 %doing one time bin per minute
            for trialnumber = 1:length(variability) %go through variability...
                if timebin == 1 %if we are in the first time bin...
                    if variability(2,trialnumber) < timebin*60*1000 %if we are in our time bin (translated to ms)...
                        block(timebin,trialnumber) = variability(1,trialnumber); %fill in the variability score in the associated row
                    end %ending "if" statement looking for our time bin
                else %if we are beyond the first time bin...
                    if variability(2,trialnumber) < timebin*60*1000 && variability(2,trialnumber) > (timebin-1)*60*1000 %if we are in our time bin (translated to ms)...
                        block(timebin,trialnumber) = variability(1,trialnumber); %fill in the variability score in the associated row
                    end %ending "if" statement looking for our time bin
                end %ending "if" statement doing the rules for first vs beyond first time bin
            end %ending "for" loop going through variability
        end %ending "for" loop going through time bins
        rew = [1 3 5 7]; %rewarded EMs
        RewEMs = []; %space
        rem = 1; %format
        for re = 2:length(data) %go through data
            if data(re,2) ~= data(re-1,2) && data(re-1,2) == 0 %when a turn happens...
                if ismember(data(re,2),rew) %if rewarded...
                    RewEMs(1,rem) = 1; %mark as 1
                    RewEMs(2,rem) = data(re,1); %mark time
                    rem=rem+1; %format
                else %if unrewarded...
                    RewEMs(1,rem) = 0; %mark as 0
                    RewEMs(2,rem) = data(re,1); %mark time
                    rem=rem+1; %format
                end %ending "if" statement separating rewarded from unrewarded
            end %ending "if" statement looking for EMs
        end %ending "for" loop going through data
        RewBlock = NaN(30,length(RewEMs)); %space for each time RewBlock
        for timebin = 1:30 %doing one time bin per minute
            for trialnumber = 1:length(RewEMs) %go through RewEMs...
                if timebin == 1 %if we are in the first time bin...
                    if RewEMs(2,trialnumber) < timebin*60*1000 %if we are in our time bin (translated to ms)...
                        RewBlock(timebin,trialnumber) = RewEMs(1,trialnumber); %fill in the RewEMs score in the associated row
                    end %ending "if" statement looking for our time bin
                else %if we are beyond the first time bin...
                    if RewEMs(2,trialnumber) < timebin*60*1000 && RewEMs(2,trialnumber) > (timebin-1)*60*1000 %if we are in our time bin (translated to ms)...
                        RewBlock(timebin,trialnumber) = RewEMs(1,trialnumber); %fill in the RewEMs score in the associated row
                    end %ending "if" statement looking for our time bin
                end %ending "if" statement doing the rules for first vs beyond first time bin
            end %ending "for" loop going through RewEMs
        end %ending "for" loop going through time bins
        align = []; %space
        for A = 1:30 %go through each min
            align(A,1) = nanmean(block(A,:)); %fill in mean variability score during that time block
            align(A,2) = nansum(RewBlock(A,:)); %fill in # of rewards gained in that time block
        end %ending "for" loop going through each min
        if session == 1 %if we are in the first session...
            EM_length = max(align(:,2)); %set the max RPMs we can use
        else %if we are past session 1...
            if EM_length > max(align(:,2)) %if we have fewer RPMs than what we said was our max...
                EM_length = max(align(:,2)); %reset to the new value
            end %ending "if" statement looking for new maximum
        end %ending "if" statement doing different rules by session number
        [correlationValue, pValue] = corr(align(:,1), align(:,2)); %calculate the correlation value & p-value
        CORR(1) = correlationValue; %store
        CORR(2) = pValue; %store
        if environment == 1 && session == 1 %first session from first environment...
            savespot = 1; %we start in first spot
        else %if we are beyond the first session...
            savespot = savespot+1; %move down one spot for saving the data
        end %ending "if" statement figuring out where to save the data
        ALLDATA{savespot,1} = align; %save variability score & # of rewards
        ALLDATA{savespot,2} = CORR; %save correlation & p value
        ALLDATA{savespot,3} = overlap(3,session); %save mouse ID
    end %ending "for" loop going through overlap
end %ending "for" loop going through environments

%% Speed Reward Rate Correlation - exploration

ALLDATA = []; %space
for environment = 1:6 %go through all environments doing the streamlined way to pull data out
    if environment == 1 %starting "if" statement to load each individual overlapped
        filename = load('C:\Users\julie\OneDrive\Documents\MATLAB\example data\NoConfLowVolControl_withDTs.mat');
    elseif environment == 2 %starting "if" statement to load each individual overlapped
        filename = load('C:\Users\julie\OneDrive\Documents\MATLAB\example data\LowConfLowVolControl_withDTs.mat');
    elseif environment == 3 %starting "if" statement to load each individual overlapped
        filename = load('C:\Users\julie\OneDrive\Documents\MATLAB\example data\HighConfLowVolControl_withDTs.mat');
    elseif environment == 4 %starting "if" statement to load each individual overlapped
        filename = load('C:\Users\julie\OneDrive\Documents\MATLAB\example data\NoConfHighVolControl_withDTs.mat');
    elseif environment == 5 %starting "if" statement to load each individual overlapped
        filename = load('C:\Users\julie\OneDrive\Documents\MATLAB\example data\LowConfHighVolControl_withDTs.mat');
    elseif environment == 6 %starting "if" statement to load each individual overlapped
        filename = load('C:\Users\julie\OneDrive\Documents\MATLAB\example data\HighConfHighVolControl_withDTs.mat');
    end %ending "if" statement for environments
    overlap = filename.overlap; %pull out cells
    for session = 1:length(overlap) %go through "overlap"
        data = []; %clearing from regression
        data = overlap{1,session}; %pull out raw data
        xy = []; %place to store location data
        xy(1,:) = data(:,13); %copy over the x info from data file
        xy(2,:) = data(:,14); %copy over y info from data file
        disp3=sqrt(diff2(xy(1,:),3).^2+diff2(xy(2,:),3).^2)+3; %equation for velocity from Eric
        turn1 = smooth(disp3,30)'; %smooth the velocity data
        turn1(2,:) = data(1:end-3,17); %fill in the second part of velocity data with the hall location
        stuff1 = NaN(length(turn1),1); %place to store hall 1
        stuff2 = NaN(length(turn1),1); %place to store hall 2
        stuff3 = NaN(length(turn1),1); %place to store hall 3
        stuff0 = NaN(length(turn1),1); %place to store center
        for splittingup = 1:length(turn1) %going through the velocity data
            if turn1(2,splittingup) == 1 %if the animal was in hall 1...
                stuff1(splittingup) = turn1(1,splittingup); %fill that in stuff 1
            elseif turn1(2,splittingup) == 2 %if the animal wa sin hall 2...
                stuff2(splittingup) = turn1(1,splittingup); %fill that in stuff 2
            elseif turn1(2,splittingup) == 3 %if the animal was in hall 3...
                stuff3(splittingup) = turn1(1,splittingup); %fill that in stuff 3
            elseif turn1(2,splittingup) == 0 %if the animal was in the center...
                stuff0(splittingup) = turn1(1,splittingup); %fill that in stuff 0
            end %ending "if" statement looking for location
        end %ending "for" loop going through velocity data
        EM_vel = []; %place to store the EMs with the velocities
        ev = 1; %format
        for turndetection = 2:length(data) %go through data
            if data(turndetection,2) ~= data(turndetection-1,2) && data(turndetection-1,2) == 0 %when a turn is detected...
                EM_vel(ev,1) = data(turndetection,2); %fill in EM
                EM_vel(ev,3) = data(turndetection,1); %fill in time
                ev=ev+1; %format
            end %ending "if" statement looking for turn detection
        end %ending "for" loop going through data
        turn1(3,:) = data(1:end-3,2); %add EMs to the turn data
        emtracker = 1; %a way to make sure vels line up properly with EMs
        for velatentry = 2:length(turn1) %going through the turn data
            if turn1(3,velatentry) ~= 0 && turn1(3,velatentry-1) == 0 %if an EM turn was detected...
                EM_vel(emtracker,2) = turn1(1,velatentry); %copy over the velocity at the point the EM was detected
                emtracker = emtracker + 1; %slide down one
            end %ending "if" statement looking for EMs
        end %ending "for" loop going through turn data
        speed = EM_vel(:,2)/3; %copy over data
        variability = []; %space to store choice variability
        for v = 2:length(speed) %go through SAEs
            variability(1,v-1) = abs(speed(v) - speed(v-1)); %what is the variability between the SAEs?
            variability(2,v-1) = EM_vel(v,3); %fill in time
        end %ending "for" loop going through SAEs
        block = NaN(30,length(variability)); %space for each time block
        for timebin = 1:30 %doing one time bin per minute
            for trialnumber = 1:length(variability) %go through variability...
                if timebin == 1 %if we are in the first time bin...
                    if variability(2,trialnumber) < timebin*60*1000 %if we are in our time bin (translated to ms)...
                        block(timebin,trialnumber) = variability(1,trialnumber); %fill in the variability score in the associated row
                    end %ending "if" statement looking for our time bin
                else %if we are beyond the first time bin...
                    if variability(2,trialnumber) < timebin*60*1000 && variability(2,trialnumber) > (timebin-1)*60*1000 %if we are in our time bin (translated to ms)...
                        block(timebin,trialnumber) = variability(1,trialnumber); %fill in the variability score in the associated row
                    end %ending "if" statement looking for our time bin
                end %ending "if" statement doing the rules for first vs beyond first time bin
            end %ending "for" loop going through variability
        end %ending "for" loop going through time bins
        rew = [1 3 5 7]; %rewarded EMs
        RewEMs = []; %space
        rem = 1; %format
        for re = 2:length(data) %go through data
            if data(re,2) ~= data(re-1,2) && data(re-1,2) == 0 %when a turn happens...
                if ismember(data(re,2),rew) %if rewarded...
                    RewEMs(1,rem) = 1; %mark as 1
                    RewEMs(2,rem) = data(re,1); %mark time
                    rem=rem+1; %format
                else %if unrewarded...
                    RewEMs(1,rem) = 0; %mark as 0
                    RewEMs(2,rem) = data(re,1); %mark time
                    rem=rem+1; %format
                end %ending "if" statement separating rewarded from unrewarded
            end %ending "if" statement looking for EMs
        end %ending "for" loop going through data
        RewBlock = NaN(30,length(RewEMs)); %space for each time RewBlock
        for timebin = 1:30 %doing one time bin per minute
            for trialnumber = 1:length(RewEMs) %go through RewEMs...
                if timebin == 1 %if we are in the first time bin...
                    if RewEMs(2,trialnumber) < timebin*60*1000 %if we are in our time bin (translated to ms)...
                        RewBlock(timebin,trialnumber) = RewEMs(1,trialnumber); %fill in the RewEMs score in the associated row
                    end %ending "if" statement looking for our time bin
                else %if we are beyond the first time bin...
                    if RewEMs(2,trialnumber) < timebin*60*1000 && RewEMs(2,trialnumber) > (timebin-1)*60*1000 %if we are in our time bin (translated to ms)...
                        RewBlock(timebin,trialnumber) = RewEMs(1,trialnumber); %fill in the RewEMs score in the associated row
                    end %ending "if" statement looking for our time bin
                end %ending "if" statement doing the rules for first vs beyond first time bin
            end %ending "for" loop going through RewEMs
        end %ending "for" loop going through time bins
        align = []; %space
        for A = 1:30 %go through each min
            align(A,1) = nanmean(block(A,:)); %fill in mean variability score during that time block
            align(A,2) = nansum(RewBlock(A,:)); %fill in # of rewards gained in that time block
        end %ending "for" loop going through each min
        if session == 1 %if we are in the first session...
            EM_length = max(align(:,2)); %set the max RPMs we can use
        else %if we are past session 1...
            if EM_length > max(align(:,2)) %if we have fewer RPMs than what we said was our max...
                EM_length = max(align(:,2)); %reset to the new value
            end %ending "if" statement looking for new maximum
        end %ending "if" statement doing different rules by session number
        [correlationValue, pValue] = corr(align(:,1), align(:,2)); %calculate the correlation value & p-value
        CORR(1) = correlationValue; %store
        CORR(2) = pValue; %store
        if environment == 1 && session == 1 %first session from first environment...
            savespot = 1; %we start in first spot
        else %if we are beyond the first session...
            savespot = savespot+1; %move down one spot for saving the data
        end %ending "if" statement figuring out where to save the data
        ALLDATA{savespot,1} = align; %save variability score & # of rewards
        ALLDATA{savespot,2} = CORR; %save correlation & p value
        ALLDATA{savespot,3} = overlap(3,session); %save mouse ID
    end %ending "for" loop going through overlap
end %ending "for" loop going through environments

%% First Block Beta - exploration

for environment = 1:6 %go through all environments doing the streamlined way to pull data out
    if environment == 1 %starting "if" statement to load each individual overlapped
        filename = load('C:\Users\julie\OneDrive\Documents\MATLAB\example data\NoConfLowVolControl_withDTs.mat');
    elseif environment == 2 %starting "if" statement to load each individual overlapped
        filename = load('C:\Users\julie\OneDrive\Documents\MATLAB\example data\LowConfLowVolControl_withDTs.mat');
    elseif environment == 3 %starting "if" statement to load each individual overlapped
        filename = load('C:\Users\julie\OneDrive\Documents\MATLAB\example data\HighConfLowVolControl_withDTs.mat');
    elseif environment == 4 %starting "if" statement to load each individual overlapped
        filename = load('C:\Users\julie\OneDrive\Documents\MATLAB\example data\NoConfHighVolControl_withDTs.mat');
    elseif environment == 5 %starting "if" statement to load each individual overlapped
        filename = load('C:\Users\julie\OneDrive\Documents\MATLAB\example data\LowConfHighVolControl_withDTs.mat');
    elseif environment == 6 %starting "if" statement to load each individual overlapped
        filename = load('C:\Users\julie\OneDrive\Documents\MATLAB\example data\HighConfHighVolControl_withDTs.mat');
    end %ending "if" statement for environments
    overlap = filename.overlap; %pull out cells
    for session = 1:length(overlap) %go through "overlap"
        data2 = overlap{1,session}; %pull out raw data
        EM = data2(:,2); %makes a variable for just EMs
        wantedEM = []; %making an open vector for EMs that we want
        for q = 2:length(EM) %starting with second value & running through all EMs
            if EM(q)~=EM(q-1) %if an EM is different from the one preceding it...
            wantedEM = [wantedEM,EM(q)]; %...include those EMs in the wantedEM vector
            end %ending the "if" statement
        end %ending the "for" loop
        wantedEM(ismember(wantedEM,[0,11,12])) = []; %retains only EMs pertaining to turns
        b=1; %format
        for blocks = 2:length(data2) %go through data
            if data2(blocks,2) ~= data2(blocks-1,2) && data2(blocks-1,2) == 0 %if a turn is detected...
                wantedEM(2,b) = data2(blocks,9); %fill in the block that trial happened in
                b=b+1; %format
            end %ending "if" statement looking for turns
        end %ending "for" loop going through data
        block1 = []; restofsession = []; %space
        ros = 1; %format
        for b1 = 1:length(wantedEM) %go through EMs
            if wantedEM(2,b1) == 1 %if we are in the first block...
                block1(b1) = wantedEM(1,b1); %fill in
            else %if we are past the first block...
                restofsession(ros) = wantedEM(1,b1); %fill in
                ros=ros+1; %format
            end %ending "if" statement separating by block
        end %ending "for" loop going through EMs
        rew = [1 3 5 7]; %rewarded EMs
        data = []; %space
        data.days = 1; %the number of sessions you are using
        data.trials = length(block1); %the relevant trials
        data.rewards = []; %place to store reward info
        for rrr = 1:length(block1) %going through the EMs
            if ismember(block1(rrr),rew) %if it is a rewarded trial...
                data.rewards(rrr) = 1; %tag it with a 1
            else %otherwise...
                data.rewards(rrr) = 0; %tag it with a 0
            end %ending "if" statement
        end %ending "for" loop
        rights = [5 6 7 8];
        data.choices = []; %place to store choice info
        for cc = 1:length(block1) %going through the EMs
            if ismember(block1(cc),rights) %if it is a right decision...
                data.choices(cc) = 1; %tag it with a 1
            else %otherwise...
                data.choices(cc) = 2; %tag it with a 2
            end %ending "if" statement
        end %ending "for" loop
        data.rewards = data.rewards'; %transposing the orientation
        data.choices = data.choices'; %transposing the orientation
        data.bs.raw = NaN(data.days, 3); %making NaNs for the number of days we are running for the number of betas being used
        for i=1:data.days %going through the number of days
            LLfit = @(betas) sum(-fit_Qvalue_V2(data.choices(:,i), data.rewards(:,i), betas, data.trials)); %use the choice, reward, & trial info to fit betas established in fit_Qvalue function
            [data.bs.raw(i,:), bvals] = fmincon(LLfit,[0.5, 5, 0],[],[],[],[],[0, 0.001, -10],[1, Inf, 10]); %establishing the limits for the values of the different betas
        end %ending "for" loop going through the days
        block1_learning_rate = data.bs.raw(1); %printing the learning rate
        block1_explore_exploit = data.bs.raw(2); %printing the beta
        block1_Q_bias = data.bs.raw(3); %printing the Q bias term
        data = []; %space
        data.days = 1; %the number of sessions you are using
        data.trials = length(restofsession); %the relevant trials
        data.rewards = []; %place to store reward info
        for rrr = 1:length(restofsession) %going through the EMs
            if ismember(restofsession(rrr),rew) %if it is a rewarded trial...
                data.rewards(rrr) = 1; %tag it with a 1
            else %otherwise...
                data.rewards(rrr) = 0; %tag it with a 0
            end %ending "if" statement
        end %ending "for" loop
        rights = [5 6 7 8];
        data.choices = []; %place to store choice info
        for cc = 1:length(restofsession) %going through the EMs
            if ismember(restofsession(cc),rights) %if it is a right decision...
                data.choices(cc) = 1; %tag it with a 1
            else %otherwise...
                data.choices(cc) = 2; %tag it with a 2
            end %ending "if" statement
        end %ending "for" loop
        data.rewards = data.rewards'; %transposing the orientation
        data.choices = data.choices'; %transposing the orientation
        data.bs.raw = NaN(data.days, 3); %making NaNs for the number of days we are running for the number of betas being used
        for i=1:data.days %going through the number of days
            LLfit = @(betas) sum(-fit_Qvalue_V2(data.choices(:,i), data.rewards(:,i), betas, data.trials)); %use the choice, reward, & trial info to fit betas established in fit_Qvalue function
            [data.bs.raw(i,:), bvals] = fmincon(LLfit,[0.5, 5, 0],[],[],[],[],[0, 0.001, -10],[1, Inf, 10]); %establishing the limits for the values of the different betas
        end %ending "for" loop going through the days
        restofsession_learning_rate = data.bs.raw(1); %printing the learning rate
        restofsession_explore_exploit = data.bs.raw(2); %printing the beta
        restofsession_Q_bias = data.bs.raw(3); %printing the Q bias term
        if environment == 1 && session == 1 %first session from first environment...
            savespot = 1; %we start in first spot
        else %if we are beyond the first session...
            savespot = savespot+1; %move down one spot for saving the data
        end %ending "if" statement figuring out where to save the data
        ALLDATA(savespot,1) = block1_explore_exploit; %save explore exploit beta
        ALLDATA(savespot,2) = restofsession_explore_exploit; %save beta for rest of session
        ALLDATA(savespot,3) = overlap{3,session}; %save ID
    end %ending "for" loop going through overlap
end %ending "for" loop going through environments

%% End Session Beta - exploration

for environment = 1:6 %go through all environments doing the streamlined way to pull data out
    if environment == 1 %starting "if" statement to load each individual overlapped
        filename = load('C:\Users\julie\OneDrive\Documents\MATLAB\example data\NoConfLowVolControl_withDTs.mat');
    elseif environment == 2 %starting "if" statement to load each individual overlapped
        filename = load('C:\Users\julie\OneDrive\Documents\MATLAB\example data\LowConfLowVolControl_withDTs.mat');
    elseif environment == 3 %starting "if" statement to load each individual overlapped
        filename = load('C:\Users\julie\OneDrive\Documents\MATLAB\example data\HighConfLowVolControl_withDTs.mat');
    elseif environment == 4 %starting "if" statement to load each individual overlapped
        filename = load('C:\Users\julie\OneDrive\Documents\MATLAB\example data\NoConfHighVolControl_withDTs.mat');
    elseif environment == 5 %starting "if" statement to load each individual overlapped
        filename = load('C:\Users\julie\OneDrive\Documents\MATLAB\example data\LowConfHighVolControl_withDTs.mat');
    elseif environment == 6 %starting "if" statement to load each individual overlapped
        filename = load('C:\Users\julie\OneDrive\Documents\MATLAB\example data\HighConfHighVolControl_withDTs.mat');
    end %ending "if" statement for environments
    overlap = filename.overlap; %pull out cells
    for session = 1:length(overlap) %go through "overlap"
        data2 = overlap{1,session}; %pull out raw data
        EM = data2(:,2); %makes a variable for just EMs
        wantedEM = []; %making an open vector for EMs that we want
        for q = 2:length(EM) %starting with second value & running through all EMs
            if EM(q)~=EM(q-1) %if an EM is different from the one preceding it...
            wantedEM = [wantedEM,EM(q)]; %...include those EMs in the wantedEM vector
            end %ending the "if" statement
        end %ending the "for" loop
        wantedEM(ismember(wantedEM,[0,11,12])) = []; %retains only EMs pertaining to turns
        r = 0; %counter for rewards
        rew = [1 3 5 7]; %rewarded EMs
        for R = 1:length(wantedEM) %go through EMs
            if r < 101 %if we are within first 100 rewards...
                counter = R; %mark which trial we are at
                if ismember(wantedEM(R),rew) %if rewarded...
                    r=r+1; %add 1 to the counter
                end %ending "if" statement for rewarded
            end %ending "if" statement for first 100 rewards
        end %ending "for" loop going through EMs
        firstrewards = []; restofsession = []; %space
        ros = 1; %format
        for fr = 1:length(wantedEM) %go through EMs
            if fr <= counter %if we are in the first 100 rewards...
                firstrewards(fr) = wantedEM(fr); %fill in
            else %if we are past the first block...
                restofsession(ros) = wantedEM(fr); %fill in
                ros=ros+1; %format
            end %ending "if" statement separating by block
        end %ending "for" loop going through EMs
        if length(firstrewards) < length(wantedEM) %if we have a useable sesion...
            data = []; %space
            data.days = 1; %the number of sessions you are using
            data.trials = length(firstrewards); %the relevant trials
            data.rewards = []; %place to store reward info
            for rrr = 1:length(firstrewards) %going through the EMs
                if ismember(firstrewards(rrr),rew) %if it is a rewarded trial...
                    data.rewards(rrr) = 1; %tag it with a 1
                else %otherwise...
                    data.rewards(rrr) = 0; %tag it with a 0
                end %ending "if" statement
            end %ending "for" loop
            rights = [5 6 7 8];
            data.choices = []; %place to store choice info
            for cc = 1:length(firstrewards) %going through the EMs
                if ismember(firstrewards(cc),rights) %if it is a right decision...
                    data.choices(cc) = 1; %tag it with a 1
                else %otherwise...
                    data.choices(cc) = 2; %tag it with a 2
                end %ending "if" statement
            end %ending "for" loop
            data.rewards = data.rewards'; %transposing the orientation
            data.choices = data.choices'; %transposing the orientation
            data.bs.raw = NaN(data.days, 3); %making NaNs for the number of days we are running for the number of betas being used
            for i=1:data.days %going through the number of days
                LLfit = @(betas) sum(-fit_Qvalue_V2(data.choices(:,i), data.rewards(:,i), betas, data.trials)); %use the choice, reward, & trial info to fit betas established in fit_Qvalue function
                [data.bs.raw(i,:), bvals] = fmincon(LLfit,[0.5, 5, 0],[],[],[],[],[0, 0.001, -10],[1, Inf, 10]); %establishing the limits for the values of the different betas
            end %ending "for" loop going through the days
            firstrewards_learning_rate = data.bs.raw(1); %printing the learning rate
            firstrewards_explore_exploit = data.bs.raw(2); %printing the beta
            firstrewards_Q_bias = data.bs.raw(3); %printing the Q bias term
            data = []; %space
            data.days = 1; %the number of sessions you are using
            data.trials = length(restofsession); %the relevant trials
            data.rewards = []; %place to store reward info
            for rrr = 1:length(restofsession) %going through the EMs
                if ismember(restofsession(rrr),rew) %if it is a rewarded trial...
                    data.rewards(rrr) = 1; %tag it with a 1
                else %otherwise...
                    data.rewards(rrr) = 0; %tag it with a 0
                end %ending "if" statement
            end %ending "for" loop
            rights = [5 6 7 8];
            data.choices = []; %place to store choice info
            for cc = 1:length(restofsession) %going through the EMs
                if ismember(restofsession(cc),rights) %if it is a right decision...
                    data.choices(cc) = 1; %tag it with a 1
                else %otherwise...
                    data.choices(cc) = 2; %tag it with a 2
                end %ending "if" statement
            end %ending "for" loop
            data.rewards = data.rewards'; %transposing the orientation
            data.choices = data.choices'; %transposing the orientation
            data.bs.raw = NaN(data.days, 3); %making NaNs for the number of days we are running for the number of betas being used
            for i=1:data.days %going through the number of days
                LLfit = @(betas) sum(-fit_Qvalue_V2(data.choices(:,i), data.rewards(:,i), betas, data.trials)); %use the choice, reward, & trial info to fit betas established in fit_Qvalue function
                [data.bs.raw(i,:), bvals] = fmincon(LLfit,[0.5, 5, 0],[],[],[],[],[0, 0.001, -10],[1, Inf, 10]); %establishing the limits for the values of the different betas
            end %ending "for" loop going through the days
            restofsession_learning_rate = data.bs.raw(1); %printing the learning rate
            restofsession_explore_exploit = data.bs.raw(2); %printing the beta
            restofsession_Q_bias = data.bs.raw(3); %printing the Q bias term
        end %end "if" statement only running usable sessions
        if environment == 1 && session == 1 %first session from first environment...
            savespot = 1; %we start in first spot
        else %if we are beyond the first session...
            savespot = savespot+1; %move down one spot for saving the data
        end %ending "if" statement figuring out where to save the data
        ALLDATA(savespot,1) = firstrewards_explore_exploit; %save explore exploit beta
        ALLDATA(savespot,2) = restofsession_explore_exploit; %save beta for rest of session
        ALLDATA(savespot,3) = overlap{3,session}; %save ID
    end %ending "for" loop going through overlap
end %ending "for" loop going through environments