for environment = 1:9 %go through all environments doing the streamlined way to pull DATA out
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
    elseif environment == 7 %starting "if" statement to load each individual overlapped
        filename = load('C:\Users\julie\OneDrive\Documents\MATLAB\example DATA\NoConfLowVolStim_withDTs.mat');
    elseif environment == 8 %starting "if" statement to load each individual overlapped
        filename = load('C:\Users\julie\OneDrive\Documents\MATLAB\example DATA\LowConfLowVolStim_withDTs.mat');
    elseif environment == 9 %starting "if" statement to load each individual overlapped
        filename = load('C:\Users\julie\OneDrive\Documents\MATLAB\example DATA\HighConfLowVolStim_withDTs.mat');
    end %ending "if" statement for environments
    overlap = filename.overlap; %pull out cells
    for session = 1:length(overlap) %go through "overlap"
        DATA = overlap{1,session}; %raw DATA
        setpoint = 0; %starting with a set point of 0
        for timestop = 1:length(DATA) %go through data
            if setpoint == 0 %if we are in the looking phase...
                if DATA(timestop,1) > 1800000 %if we are past the 30 min mark...
                    DATA(timestop:end,:) = []; %clear it
                    setpoint = 1; %change to a 1
                end %ending "if" statement if we are past 30 min mark...
            end %ending "for" loop going through data
        end %ending "for" loop going through data
        wantedEM = []; %space
        wem = 1; %format
        for WEM = 2:length(DATA) %go throguh DATA
            if DATA(WEM,2) ~= DATA(WEM-1,2) && DATA(WEM-1,2) == 0 %turn detection
                wantedEM(wem) = DATA(WEM,2); %store EM
                wem=wem+1; %format
            end %ending "if" statement for turns
        end %ending "for" loop going through DATA
        correct = [1 2 5 6]; %correct EMs
        cor = []; %space
        for c = 1:length(wantedEM) %go through EMs
            if ismember(wantedEM(c),correct) %if correct...
                cor(c) = 1; %mark 1
            else %if incorrect...
                cor(c) = 0; %mark 0
            end %ending "if" statement for correctness
        end %ending "for" loop going through EMs
        accuracy = sum(cor) / length(cor); %accuracy calculation
        data.days = 1; %the number of sessions you are using
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
        LR = data.bs.raw(1); %learning rate
        beta = data.bs.raw(2); %beta
        Q = data.bs.raw(3); %Q bias term
        Q = abs(Q); %make sure bias is positive
        xy = []; %place to store location DATA
        xy(1,:) = DATA(:,13); %copy over the x info from DATA file
        xy(2,:) = DATA(:,14); %copy over y info from DATA file
        disp3=sqrt(diff2(xy(1,:),3).^2+diff2(xy(2,:),3).^2);  %equation from Eric for velocity
        turn1 = smooth(disp3,30)'; %smooth the velocity DATA
        turn1(2,:) = DATA(1:end-3,17); %fill in the second part of velocity DATA with the hall location
        stuff1 = NaN(length(turn1),1); %place to store hall 1
        stuff2 = NaN(length(turn1),1); %place to store hall 2
        stuff3 = NaN(length(turn1),1); %place to store hall 3
        stuff0 = NaN(length(turn1),1); %place to store center
        for splittingup = 1:length(turn1) %going through the velocity DATA
            if turn1(2,splittingup) == 1 %if the animal was in hall 1...
                stuff1(splittingup) = turn1(1,splittingup); %fill that in stuff 1
            elseif turn1(2,splittingup) == 2 %if the animal was in hall 2...
                stuff2(splittingup) = turn1(1,splittingup); %fill that in stuff 2
            elseif turn1(2,splittingup) == 3 %if the animal was in hall 3...
                stuff3(splittingup) = turn1(1,splittingup); %fill that in stuff 3
            elseif turn1(2,splittingup) == 0 %if the animal was in the center...
                stuff0(splittingup) = turn1(1,splittingup); %fill that in stuff 0
            end %ending "if" statement looking for location
        end %ending "for" loop going through velocity DATA
        stuff1(:,2) = DATA(1:end-3,1); %add in time info
        stuff2(:,2) = DATA(1:end-3,1); %add in time info
        stuff3(:,2) = DATA(1:end-3,1); %add in time info
        stuff0(:,2) = DATA(1:end-3,1); %add in time info
        turn1(3,:) = DATA(1:end-3,2); %fill in the EMs to turn1
        turn1(4,:) = 1:length(turn1); %fill in a spacer
        allempositions = []; %place to store all EM positions
        emspot = 1; %formatting
        ems = [1:8]; %any em
        for stars = 2:length(turn1) %going through the DATA
            if ismember(turn1(3,stars),ems) && turn1(3,stars-1) == 0 %if a turn is first detected...
                allempositions(emspot) = turn1(4,stars); %mark down that spot position number
                emspot = emspot + 1; %formatting
            end %ending "if" statement looking for ems
        end %ending "for" loop gooing through the DATA
        veltimes = []; %place to store a lot of the info
        veltimes(1,:) = turn1(1,:); %copy over the velocities
        veltimes(2,:) = DATA(1:end-3,1); %copy over the times
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
            veltimes(4,allempositions(firstinstance):allempositions(firstinstance + 1)) = crossing(1,:); %replace the veltimes with the cleared DATA
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
            veltimes(4,allempositions(firstinstance):allempositions(firstinstance + 1)) = crossing(1,:); %replace the veltimes with the cleared DATA
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
        for cleaningup = 1:length(decisiontimes) %cleaning up any possible negative decision times due to bugs
            if decisiontimes(cleaningup,3) < 0 %if we have a weird neg value...
                decisiontimes(cleaningup,3) = NaN; %NaN it out
            end %ending "if" statement looking for negatives
        end %ending "for" loop for cleaning out bugs
        DT = nanmedian(decisiontimes(:,3)); %DT
        if environment == 1 %environment
            noconflowvol_ctrl(session,1) = accuracy; %store
            noconflowvol_ctrl(session,2) = LR; %store
            noconflowvol_ctrl(session,3) = beta; %store
            noconflowvol_ctrl(session,4) = Q; %store
            noconflowvol_ctrl(session,5) = DT; %store
            noconflowvol_ctrl(session,6) = overlap{3,session}; %ID
        elseif environment == 2 %environment
            lowconflowvol_ctrl(session,1) = accuracy; %store
            lowconflowvol_ctrl(session,2) = LR; %store
            lowconflowvol_ctrl(session,3) = beta; %store
            lowconflowvol_ctrl(session,4) = Q; %store
            lowconflowvol_ctrl(session,5) = DT; %store
            lowconflowvol_ctrl(session,6) = overlap{3,session}; %ID
        elseif environment == 3 %environment
            highconflowvol_ctrl(session,1) = accuracy; %store
            highconflowvol_ctrl(session,2) = LR; %store
            highconflowvol_ctrl(session,3) = beta; %store
            highconflowvol_ctrl(session,4) = Q; %store
            highconflowvol_ctrl(session,5) = DT; %store
            highconflowvol_ctrl(session,6) = overlap{3,session}; %ID
        elseif environment == 4 %environment
            noconfhighvol_ctrl(session,1) = accuracy; %store
            noconfhighvol_ctrl(session,2) = LR; %store
            noconfhighvol_ctrl(session,3) = beta; %store
            noconfhighvol_ctrl(session,4) = Q; %store
            noconfhighvol_ctrl(session,5) = DT; %store
            noconfhighvol_ctrl(session,6) = overlap{3,session}; %ID
        elseif environment == 5 %environment
            lowconfhighvol_ctrl(session,1) = accuracy; %store
            lowconfhighvol_ctrl(session,2) = LR; %store
            lowconfhighvol_ctrl(session,3) = beta; %store
            lowconfhighvol_ctrl(session,4) = Q; %store
            lowconfhighvol_ctrl(session,5) = DT; %store
            lowconfhighvol_ctrl(session,6) = overlap{3,session}; %ID
        elseif environment == 6 %environment
            highconfhighvol_ctrl(session,1) = accuracy; %store
            highconfhighvol_ctrl(session,2) = LR; %store
            highconfhighvol_ctrl(session,3) = beta; %store
            highconfhighvol_ctrl(session,4) = Q; %store
            highconfhighvol_ctrl(session,5) = DT; %store
            highconfhighvol_ctrl(session,6) = overlap{3,session}; %ID
        elseif environment == 7 %environment
            noconflowvol_stim(session,1) = accuracy; %store
            noconflowvol_stim(session,2) = LR; %store
            noconflowvol_stim(session,3) = beta; %store
            noconflowvol_stim(session,4) = Q; %store
            noconflowvol_stim(session,5) = DT; %store
            noconflowvol_stim(session,6) = overlap{3,session}; %ID
        elseif environment == 8 %environment
            lowconflowvol_stim(session,1) = accuracy; %store
            lowconflowvol_stim(session,2) = LR; %store
            lowconflowvol_stim(session,3) = beta; %store
            lowconflowvol_stim(session,4) = Q; %store
            lowconflowvol_stim(session,5) = DT; %store
            lowconflowvol_stim(session,6) = overlap{3,session}; %ID
        elseif environment == 9 %environment
            highconflowvol_stim(session,1) = accuracy; %store
            highconflowvol_stim(session,2) = LR; %store
            highconflowvol_stim(session,3) = beta; %store
            highconflowvol_stim(session,4) = Q; %store
            highconflowvol_stim(session,5) = DT; %store
            highconflowvol_stim(session,6) = overlap{3,session}; %ID
        end %ending "if" statement saving data
    end %ending "for" loop going through sessions
    ispn = [1 5 7 9 11]; %iSPN
    dspn = [3 10 2 4 6 8 12]; %dSPN
    if environment == 1 %environment check
        i=1; d=1; %format
        for genotype = 1:length(noconflowvol_ctrl) %go through computed data
            if ismember(noconflowvol_ctrl(genotype,6),ispn) %if ispn...
                ispn_noconflowvol_ctrl(i,1:6) = noconflowvol_ctrl(genotype,1:6); %store data
                i=i+1; %format
            elseif ismember(noconflowvol_ctrl(genotype,6),dspn) %if dspn...
                dspn_noconflowvol_ctrl(d,1:6) = noconflowvol_ctrl(genotype,1:6); %store data
                d=d+1; %format
            end %ending "if" statement separating by genotype
        end %ending "for" loop going through computed data
    elseif environment == 2 %environment check
        i=1; d=1; %format
        for genotype = 1:length(lowconflowvol_ctrl) %go through computed data
            if ismember(lowconflowvol_ctrl(genotype,6),ispn) %if ispn...
                ispn_lowconflowvol_ctrl(i,1:6) = lowconflowvol_ctrl(genotype,1:6); %store data
                i=i+1; %format
            elseif ismember(lowconflowvol_ctrl(genotype,6),dspn) %if dspn...
                dspn_lowconflowvol_ctrl(d,1:6) = lowconflowvol_ctrl(genotype,1:6); %store data
                d=d+1; %format
            end %ending "if" statement separating by genotype
        end %ending "for" loop going through computed data
    elseif environment == 3 %environment check
        i=1; d=1; %format
        for genotype = 1:length(highconflowvol_ctrl) %go through computed data
            if ismember(highconflowvol_ctrl(genotype,6),ispn) %if ispn...
                ispn_highconflowvol_ctrl(i,1:6) = highconflowvol_ctrl(genotype,1:6); %store data
                i=i+1; %format
            elseif ismember(highconflowvol_ctrl(genotype,6),dspn) %if dspn...
                dspn_highconflowvol_ctrl(d,1:6) = highconflowvol_ctrl(genotype,1:6); %store data
                d=d+1; %format
            end %ending "if" statement separating by genotype
        end %ending "for" loop going through computed data
    elseif environment == 4 %environment check
        i=1; d=1; %format
        for genotype = 1:length(noconfhighvol_ctrl) %go through computed data
            if ismember(noconfhighvol_ctrl(genotype,6),ispn) %if ispn...
                ispn_noconfhighvol_ctrl(i,1:6) = noconfhighvol_ctrl(genotype,1:6); %store data
                i=i+1; %format
            elseif ismember(noconfhighvol_ctrl(genotype,6),dspn) %if dspn...
                dspn_noconfhighvol_ctrl(d,1:6) = noconfhighvol_ctrl(genotype,1:6); %store data
                d=d+1; %format
            end %ending "if" statement separating by genotype
        end %ending "for" loop going through computed data
    elseif environment == 5 %environment check
        i=1; d=1; %format
        for genotype = 1:length(lowconfhighvol_ctrl) %go through computed data
            if ismember(lowconfhighvol_ctrl(genotype,6),ispn) %if ispn...
                ispn_lowconfhighvol_ctrl(i,1:6) = lowconfhighvol_ctrl(genotype,1:6); %store data
                i=i+1; %format
            elseif ismember(lowconfhighvol_ctrl(genotype,6),dspn) %if dspn...
                dspn_lowconfhighvol_ctrl(d,1:6) = lowconfhighvol_ctrl(genotype,1:6); %store data
                d=d+1; %format
            end %ending "if" statement separating by genotype
        end %ending "for" loop going through computed data
    elseif environment == 6 %environment check
        i=1; d=1; %format
        for genotype = 1:length(highconfhighvol_ctrl) %go through computed data
            if ismember(highconfhighvol_ctrl(genotype,6),ispn) %if ispn...
                ispn_highconfhighvol_ctrl(i,1:6) = highconfhighvol_ctrl(genotype,1:6); %store data
                i=i+1; %format
            elseif ismember(highconfhighvol_ctrl(genotype,6),dspn) %if dspn...
                dspn_highconfhighvol_ctrl(d,1:6) = highconfhighvol_ctrl(genotype,1:6); %store data
                d=d+1; %format
            end %ending "if" statement separating by genotype
        end %ending "for" loop going through computed data
    elseif environment == 7 %environment check
        i=1; d=1; %format
        for genotype = 1:length(noconflowvol_stim) %go through computed data
            if ismember(noconflowvol_stim(genotype,6),ispn) %if ispn...
                ispn_noconflowvol_stim(i,1:6) = noconflowvol_stim(genotype,1:6); %store data
                i=i+1; %format
            elseif ismember(noconflowvol_stim(genotype,6),dspn) %if dspn...
                dspn_noconflowvol_stim(d,1:6) = noconflowvol_stim(genotype,1:6); %store data
                d=d+1; %format
            end %ending "if" statement separating by genotype
        end %ending "for" loop going through computed data
    elseif environment == 8 %environment check
        i=1; d=1; %format
        for genotype = 1:length(lowconflowvol_stim) %go through computed data
            if ismember(lowconflowvol_stim(genotype,6),ispn) %if ispn...
                ispn_lowconflowvol_stim(i,1:6) = lowconflowvol_stim(genotype,1:6); %store data
                i=i+1; %format
            elseif ismember(lowconflowvol_stim(genotype,6),dspn) %if dspn...
                dspn_lowconflowvol_stim(d,1:6) = lowconflowvol_stim(genotype,1:6); %store data
                d=d+1; %format
            end %ending "if" statement separating by genotype
        end %ending "for" loop going through computed data
    elseif environment == 9 %environment check
        i=1; d=1; %format
        for genotype = 1:length(highconflowvol_stim) %go through computed data
            if ismember(highconflowvol_stim(genotype,6),ispn) %if ispn...
                ispn_highconflowvol_stim(i,1:6) = highconflowvol_stim(genotype,1:6); %store data
                i=i+1; %format
            elseif ismember(highconflowvol_stim(genotype,6),dspn) %if dspn...
                dspn_highconflowvol_stim(d,1:6) = highconflowvol_stim(genotype,1:6); %store data
                d=d+1; %format
            end %ending "if" statement separating by genotype
        end %ending "for" loop going through computed data
    end %ending "if" statement for environments
end %ending "for" loop going through environments

MEAN(1) = (nanmean(highconflowvol_ctrl(:,1)) - nanmean(noconflowvol_ctrl(:,1))) / nanmean(noconflowvol_ctrl(:,1));
MEAN(2) = (nanmean(highconflowvol_ctrl(:,2)) - nanmean(noconflowvol_ctrl(:,2))) / nanmean(noconflowvol_ctrl(:,2));
MEAN(3) = (nanmean(highconflowvol_ctrl(:,3)) - nanmean(noconflowvol_ctrl(:,3))) / nanmean(noconflowvol_ctrl(:,3));
MEAN(4) = (nanmean(highconflowvol_ctrl(:,4)) - nanmean(noconflowvol_ctrl(:,4))) / nanmean(noconflowvol_ctrl(:,4));
MEAN(5) = (nanmean(highconflowvol_ctrl(:,5)) - nanmean(noconflowvol_ctrl(:,5))) / nanmean(noconflowvol_ctrl(:,5));
figure;
ctr = []; ydt = [];
y = [MEAN(1); MEAN(2); MEAN(3); MEAN(4); MEAN(5)]*100;
a_vert = [nanstd(individual3(:,1))/sqrt(length(noconflowvol_ctrl(:,1))); nanstd(individual3(:,2))/sqrt(length(noconflowvol_ctrl(:,2))); nanstd(individual3(:,3))/sqrt(length(noconflowvol_ctrl(:,3))); nanstd(individual3(:,4))/sqrt(length(noconflowvol_ctrl(:,4))); nanstd(individual3(:,5))/sqrt(length(noconflowvol_ctrl(:,5)))]*100;
errorplus_vert=a_vert';
errorminus_vert=errorplus_vert;
x = [1:5];
hBar = bar(x,y);
for C = 1:5
    colors(C,1:3) = [0.8 0.8 0.8];
end
for k1 = 1:length(x)
    hBar.FaceColor = 'flat';  % Set the FaceColor property to 'flat' to enable individual coloring
    hBar.CData(k1,:) = colors(k1,:);  % Assign the colors
end
for k1 = 1:size(y,2)
    ctr(k1,:) = bsxfun(@plus, hBar(k1).XData, hBar(k1).XOffset');     
    ydt(k1,:) = hBar(k1).YData;                    
end
hold on
e1 = errorbar(x(1), y(1), errorminus_vert(1), errorplus_vert(1), 'LineStyle', 'none', 'LineWidth', 2);
e1.Color = 'k'; 
e2 = errorbar(x(2), y(2), errorminus_vert(2), errorplus_vert(2), 'LineStyle', 'none', 'LineWidth', 2);
e2.Color = 'k'; 
e3 = errorbar(x(3), y(3), errorminus_vert(3), errorplus_vert(3), 'LineStyle', 'none', 'LineWidth', 2);
e3.Color = 'k'; 
e4 = errorbar(x(4), y(4), errorminus_vert(4), errorplus_vert(4), 'LineStyle', 'none', 'LineWidth', 2);
e4.Color = 'k'; 
e5 = errorbar(x(5), y(5), errorminus_vert(5), errorplus_vert(5), 'LineStyle', 'none', 'LineWidth', 2);
e5.Color = 'k'; 
ylim([-30 40])
XTickLabel = ({'Acc.','LR','Exp.','Bias','DT'});
set(gca,'xticklabel',XTickLabel,'box','off','FontSize',20)
ylabel('% Change due to High Conf')

miniMEAN(1,1) = (nanmean(noconfhighvol_ctrl(:,1)) - nanmean(noconflowvol_ctrl(:,1))) / nanmean(noconflowvol_ctrl(:,1));
miniMEAN(1,2) = (nanmean(noconfhighvol_ctrl(:,2)) - nanmean(noconflowvol_ctrl(:,2))) / nanmean(noconflowvol_ctrl(:,2));
miniMEAN(1,3) = (nanmean(noconfhighvol_ctrl(:,3)) - nanmean(noconflowvol_ctrl(:,3))) / nanmean(noconflowvol_ctrl(:,3));
miniMEAN(1,4) = (nanmean(noconfhighvol_ctrl(:,4)) - nanmean(noconflowvol_ctrl(:,4))) / nanmean(noconflowvol_ctrl(:,4));
miniMEAN(1,5) = (nanmean(noconfhighvol_ctrl(:,5)) - nanmean(noconflowvol_ctrl(:,5))) / nanmean(noconflowvol_ctrl(:,5));
miniMEAN(2,1) = (nanmean(lowconfhighvol_ctrl(:,1)) - nanmean(lowconflowvol_ctrl(:,1))) / nanmean(lowconflowvol_ctrl(:,1));
miniMEAN(2,2) = (nanmean(lowconfhighvol_ctrl(:,2)) - nanmean(lowconflowvol_ctrl(:,2))) / nanmean(lowconflowvol_ctrl(:,2));
miniMEAN(2,3) = (nanmean(lowconfhighvol_ctrl(:,3)) - nanmean(lowconflowvol_ctrl(:,3))) / nanmean(lowconflowvol_ctrl(:,3));
miniMEAN(2,4) = (nanmean(lowconfhighvol_ctrl(:,4)) - nanmean(lowconflowvol_ctrl(:,4))) / nanmean(lowconflowvol_ctrl(:,4));
miniMEAN(2,5) = (nanmean(lowconfhighvol_ctrl(:,5)) - nanmean(lowconflowvol_ctrl(:,5))) / nanmean(lowconflowvol_ctrl(:,5));
miniMEAN(3,1) = (nanmean(highconfhighvol_ctrl(:,1)) - nanmean(highconflowvol_ctrl(:,1))) / nanmean(highconflowvol_ctrl(:,1));
miniMEAN(3,2) = (nanmean(highconfhighvol_ctrl(:,2)) - nanmean(highconflowvol_ctrl(:,2))) / nanmean(highconflowvol_ctrl(:,2));
miniMEAN(3,3) = (nanmean(highconfhighvol_ctrl(:,3)) - nanmean(highconflowvol_ctrl(:,3))) / nanmean(highconflowvol_ctrl(:,3));
miniMEAN(3,4) = (nanmean(highconfhighvol_ctrl(:,4)) - nanmean(highconflowvol_ctrl(:,4))) / nanmean(highconflowvol_ctrl(:,4));
miniMEAN(3,5) = (nanmean(highconfhighvol_ctrl(:,5)) - nanmean(highconflowvol_ctrl(:,5))) / nanmean(highconflowvol_ctrl(:,5));
for m = 1:5
    MEAN(m) = mean(miniMEAN(:,m));
end
figure;
ctr = []; ydt = [];
y = [MEAN(1); MEAN(2); MEAN(3); MEAN(4); MEAN(5)]*100;
a_vert = [nanstd(individual3(:,1))/sqrt(length(noconflowvol_ctrl(:,1))); nanstd(individual3(:,2))/sqrt(length(noconflowvol_ctrl(:,2))); nanstd(individual3(:,3))/sqrt(length(noconflowvol_ctrl(:,3))); nanstd(individual3(:,4))/sqrt(length(noconflowvol_ctrl(:,4))); nanstd(individual3(:,5))/sqrt(length(noconflowvol_ctrl(:,5)))]*100;
errorplus_vert=a_vert';
errorminus_vert=errorplus_vert;
x = [1:5];
hBar = bar(x,y);
for C = 1:5
    colors(C,1:3) = [0.8 0.8 0.8];
end
for k1 = 1:length(x)
    hBar.FaceColor = 'flat';  % Set the FaceColor property to 'flat' to enable individual coloring
    hBar.CData(k1,:) = colors(k1,:);  % Assign the colors
end
for k1 = 1:size(y,2)
    ctr(k1,:) = bsxfun(@plus, hBar(k1).XData, hBar(k1).XOffset');     
    ydt(k1,:) = hBar(k1).YData;                    
end
hold on
e1 = errorbar(x(1), y(1), errorminus_vert(1), errorplus_vert(1), 'LineStyle', 'none', 'LineWidth', 2);
e1.Color = 'k'; 
e2 = errorbar(x(2), y(2), errorminus_vert(2), errorplus_vert(2), 'LineStyle', 'none', 'LineWidth', 2);
e2.Color = 'k'; 
e3 = errorbar(x(3), y(3), errorminus_vert(3), errorplus_vert(3), 'LineStyle', 'none', 'LineWidth', 2);
e3.Color = 'k'; 
e4 = errorbar(x(4), y(4), errorminus_vert(4), errorplus_vert(4), 'LineStyle', 'none', 'LineWidth', 2);
e4.Color = 'k'; 
e5 = errorbar(x(5), y(5), errorminus_vert(5), errorplus_vert(5), 'LineStyle', 'none', 'LineWidth', 2);
e5.Color = 'k'; 
ylim([-30 40])
XTickLabel = ({'Acc.','LR','Exp.','Bias','DT'});
set(gca,'xticklabel',XTickLabel,'box','off','FontSize',20)
ylabel('% Change due to High Vol')

miniMEAN(1,1) = (nanmean(ispn_noconflowvol_stim(:,1)) - nanmean(ispn_noconflowvol_ctrl(:,1))) / nanmean(ispn_noconflowvol_ctrl(:,1));
miniMEAN(1,2) = (nanmean(ispn_noconflowvol_stim(:,2)) - nanmean(ispn_noconflowvol_ctrl(:,2))) / nanmean(ispn_noconflowvol_ctrl(:,2));
miniMEAN(1,3) = (nanmean(ispn_noconflowvol_stim(:,3)) - nanmean(ispn_noconflowvol_ctrl(:,3))) / nanmean(ispn_noconflowvol_ctrl(:,3));
miniMEAN(1,4) = (nanmean(ispn_noconflowvol_stim(:,4)) - nanmean(ispn_noconflowvol_ctrl(:,4))) / nanmean(ispn_noconflowvol_ctrl(:,4));
miniMEAN(1,5) = (nanmean(ispn_noconflowvol_stim(:,5)) - nanmean(ispn_noconflowvol_ctrl(:,5))) / nanmean(ispn_noconflowvol_ctrl(:,5));
miniMEAN(2,1) = (nanmean(ispn_lowconflowvol_stim(:,1)) - nanmean(ispn_lowconflowvol_ctrl(:,1))) / nanmean(ispn_lowconflowvol_ctrl(:,1));
miniMEAN(2,2) = (nanmean(ispn_lowconflowvol_stim(:,2)) - nanmean(ispn_lowconflowvol_ctrl(:,2))) / nanmean(ispn_lowconflowvol_ctrl(:,2));
miniMEAN(2,3) = (nanmean(ispn_lowconflowvol_stim(:,3)) - nanmean(ispn_lowconflowvol_ctrl(:,3))) / nanmean(ispn_lowconflowvol_ctrl(:,3));
miniMEAN(2,4) = (nanmean(ispn_lowconflowvol_stim(:,4)) - nanmean(ispn_lowconflowvol_ctrl(:,4))) / nanmean(ispn_lowconflowvol_ctrl(:,4));
miniMEAN(2,5) = (nanmean(ispn_lowconflowvol_stim(:,5)) - nanmean(ispn_lowconflowvol_ctrl(:,5))) / nanmean(ispn_lowconflowvol_ctrl(:,5));
miniMEAN(3,1) = (nanmean(ispn_highconflowvol_stim(:,1)) - nanmean(ispn_highconflowvol_ctrl(:,1))) / nanmean(ispn_highconflowvol_ctrl(:,1));
miniMEAN(3,2) = (nanmean(ispn_highconflowvol_stim(:,2)) - nanmean(ispn_highconflowvol_ctrl(:,2))) / nanmean(ispn_highconflowvol_ctrl(:,2));
miniMEAN(3,3) = (nanmean(ispn_highconflowvol_stim(:,3)) - nanmean(ispn_highconflowvol_ctrl(:,3))) / nanmean(ispn_highconflowvol_ctrl(:,3));
miniMEAN(3,4) = (nanmean(ispn_highconflowvol_stim(:,4)) - nanmean(ispn_highconflowvol_ctrl(:,4))) / nanmean(ispn_highconflowvol_ctrl(:,4));
miniMEAN(3,5) = (nanmean(ispn_highconflowvol_stim(:,5)) - nanmean(ispn_highconflowvol_ctrl(:,5))) / nanmean(ispn_highconflowvol_ctrl(:,5));
for m = 1:5
    MEAN(m) = mean(miniMEAN(:,m));
end
figure;
ctr = []; ydt = [];
y = [MEAN(1); MEAN(2); MEAN(3); MEAN(4); MEAN(5)]*100;
a_vert = [nanstd(individual3(:,1))/sqrt(length(ispn_noconflowvol_ctrl(:,1))); nanstd(individual3(:,2))/sqrt(length(ispn_noconflowvol_ctrl(:,2))); nanstd(individual3(:,3))/sqrt(length(ispn_noconflowvol_ctrl(:,3))); nanstd(individual3(:,4))/sqrt(length(ispn_noconflowvol_ctrl(:,4))); nanstd(individual3(:,5))/sqrt(length(ispn_noconflowvol_ctrl(:,5)))]*100;
errorplus_vert=a_vert';
errorminus_vert=errorplus_vert;
x = [1:5];
hBar = bar(x,y);
for C = 1:5
    colors(C,1:3) = [0.8 0.8 0.8];
end
for k1 = 1:length(x)
    hBar.FaceColor = 'flat';  % Set the FaceColor property to 'flat' to enable individual coloring
    hBar.CData(k1,:) = colors(k1,:);  % Assign the colors
end
for k1 = 1:size(y,2)
    ctr(k1,:) = bsxfun(@plus, hBar(k1).XData, hBar(k1).XOffset');     
    ydt(k1,:) = hBar(k1).YData;                    
end
hold on
e1 = errorbar(x(1), y(1), errorminus_vert(1), errorplus_vert(1), 'LineStyle', 'none', 'LineWidth', 2);
e1.Color = 'k'; 
e2 = errorbar(x(2), y(2), errorminus_vert(2), errorplus_vert(2), 'LineStyle', 'none', 'LineWidth', 2);
e2.Color = 'k'; 
e3 = errorbar(x(3), y(3), errorminus_vert(3), errorplus_vert(3), 'LineStyle', 'none', 'LineWidth', 2);
e3.Color = 'k'; 
e4 = errorbar(x(4), y(4), errorminus_vert(4), errorplus_vert(4), 'LineStyle', 'none', 'LineWidth', 2);
e4.Color = 'k'; 
e5 = errorbar(x(5), y(5), errorminus_vert(5), errorplus_vert(5), 'LineStyle', 'none', 'LineWidth', 2);
e5.Color = 'k'; 
ylim([-60 100])
XTickLabel = ({'Acc.','LR','Exp.','Bias','DT'});
set(gca,'xticklabel',XTickLabel,'box','off','FontSize',20)
ylabel('% Change due to iSPN Stim')

miniMEAN(1,1) = (nanmean(dspn_noconflowvol_stim(:,1)) - nanmean(dspn_noconflowvol_ctrl(:,1))) / nanmean(dspn_noconflowvol_ctrl(:,1));
miniMEAN(1,2) = (nanmean(dspn_noconflowvol_stim(:,2)) - nanmean(dspn_noconflowvol_ctrl(:,2))) / nanmean(dspn_noconflowvol_ctrl(:,2));
miniMEAN(1,3) = (nanmean(dspn_noconflowvol_stim(:,3)) - nanmean(dspn_noconflowvol_ctrl(:,3))) / nanmean(dspn_noconflowvol_ctrl(:,3));
miniMEAN(1,4) = (nanmean(dspn_noconflowvol_stim(:,4)) - nanmean(dspn_noconflowvol_ctrl(:,4))) / nanmean(dspn_noconflowvol_ctrl(:,4));
miniMEAN(1,5) = (nanmean(dspn_noconflowvol_stim(:,5)) - nanmean(dspn_noconflowvol_ctrl(:,5))) / nanmean(dspn_noconflowvol_ctrl(:,5));
miniMEAN(2,1) = (nanmean(dspn_lowconflowvol_stim(:,1)) - nanmean(dspn_lowconflowvol_ctrl(:,1))) / nanmean(dspn_lowconflowvol_ctrl(:,1));
miniMEAN(2,2) = (nanmean(dspn_lowconflowvol_stim(:,2)) - nanmean(dspn_lowconflowvol_ctrl(:,2))) / nanmean(dspn_lowconflowvol_ctrl(:,2));
miniMEAN(2,3) = (nanmean(dspn_lowconflowvol_stim(:,3)) - nanmean(dspn_lowconflowvol_ctrl(:,3))) / nanmean(dspn_lowconflowvol_ctrl(:,3));
miniMEAN(2,4) = (nanmean(dspn_lowconflowvol_stim(:,4)) - nanmean(dspn_lowconflowvol_ctrl(:,4))) / nanmean(dspn_lowconflowvol_ctrl(:,4));
miniMEAN(2,5) = (nanmean(dspn_lowconflowvol_stim(:,5)) - nanmean(dspn_lowconflowvol_ctrl(:,5))) / nanmean(dspn_lowconflowvol_ctrl(:,5));
miniMEAN(3,1) = (nanmean(dspn_highconflowvol_stim(:,1)) - nanmean(dspn_highconflowvol_ctrl(:,1))) / nanmean(dspn_highconflowvol_ctrl(:,1));
miniMEAN(3,2) = (nanmean(dspn_highconflowvol_stim(:,2)) - nanmean(dspn_highconflowvol_ctrl(:,2))) / nanmean(dspn_highconflowvol_ctrl(:,2));
miniMEAN(3,3) = (nanmean(dspn_highconflowvol_stim(:,3)) - nanmean(dspn_highconflowvol_ctrl(:,3))) / nanmean(dspn_highconflowvol_ctrl(:,3));
miniMEAN(3,4) = (nanmean(dspn_highconflowvol_stim(:,4)) - nanmean(dspn_highconflowvol_ctrl(:,4))) / nanmean(dspn_highconflowvol_ctrl(:,4));
miniMEAN(3,5) = (nanmean(dspn_highconflowvol_stim(:,5)) - nanmean(dspn_highconflowvol_ctrl(:,5))) / nanmean(dspn_highconflowvol_ctrl(:,5));
for m = 1:5
    MEAN(m) = mean(miniMEAN(:,m));
end
figure;
ctr = []; ydt = [];
y = [MEAN(1); MEAN(2); MEAN(3); MEAN(4); MEAN(5)]*100;
a_vert = [nanstd(individual3(:,1))/sqrt(length(dspn_noconflowvol_ctrl(:,1))); nanstd(individual3(:,2))/sqrt(length(dspn_noconflowvol_ctrl(:,2))); nanstd(individual3(:,3))/sqrt(length(dspn_noconflowvol_ctrl(:,3))); nanstd(individual3(:,4))/sqrt(length(dspn_noconflowvol_ctrl(:,4))); nanstd(individual3(:,5))/sqrt(length(dspn_noconflowvol_ctrl(:,5)))]*100;
errorplus_vert=a_vert';
errorminus_vert=errorplus_vert;
x = [1:5];
hBar = bar(x,y);
for C = 1:5
    colors(C,1:3) = [0.8 0.8 0.8];
end
for k1 = 1:length(x)
    hBar.FaceColor = 'flat';  % Set the FaceColor property to 'flat' to enable individual coloring
    hBar.CData(k1,:) = colors(k1,:);  % Assign the colors
end
for k1 = 1:size(y,2)
    ctr(k1,:) = bsxfun(@plus, hBar(k1).XData, hBar(k1).XOffset');     
    ydt(k1,:) = hBar(k1).YData;                    
end
hold on
e1 = errorbar(x(1), y(1), errorminus_vert(1), errorplus_vert(1), 'LineStyle', 'none', 'LineWidth', 2);
e1.Color = 'k'; 
e2 = errorbar(x(2), y(2), errorminus_vert(2), errorplus_vert(2), 'LineStyle', 'none', 'LineWidth', 2);
e2.Color = 'k'; 
e3 = errorbar(x(3), y(3), errorminus_vert(3), errorplus_vert(3), 'LineStyle', 'none', 'LineWidth', 2);
e3.Color = 'k'; 
e4 = errorbar(x(4), y(4), errorminus_vert(4), errorplus_vert(4), 'LineStyle', 'none', 'LineWidth', 2);
e4.Color = 'k'; 
e5 = errorbar(x(5), y(5), errorminus_vert(5), errorplus_vert(5), 'LineStyle', 'none', 'LineWidth', 2);
e5.Color = 'k'; 
ylim([-60 90])
XTickLabel = ({'Acc.','LR','Exp.','Bias','DT'});
set(gca,'xticklabel',XTickLabel,'box','off','FontSize',20)
ylabel('% Change due to dSPN Stim')