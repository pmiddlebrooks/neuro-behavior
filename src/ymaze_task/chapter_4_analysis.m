%% Accuracy vs Chance vs 100% - prob matching

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
        correct = [1 2 5 6]; %correct EMs
        WEM = 1; %format
        for wem = 2:length(data) %go through data
            if data(wem,2) ~= data(wem-1,2) && data(wem-1,2) == 0 %turn detection...
                if ismember(data(wem,2),correct) %if correct turn...
                    wantedEM(WEM) = 1; %mark 1
                    WEM=WEM+1; %format
                else %if incorrect...
                    wantedEM(WEM) = 0; %mark 0
                    WEM=WEM+1; %format
                end %ending "if" statement for correct vs incorrect
            end %ending "if" statement looking for turns
        end %ending "for" loop going through data
        accuracy = sum(wantedEM)/length(wantedEM); %calculate accuracy
        if environment == 1 %environment
            noconflowvol_ctrl(session,1) = accuracy; %store
            noconflowvol_ctrl(session,2) = overlap{3,session}; %store
        elseif environment == 2 %environment
            lowconflowvol_ctrl(session,1) = accuracy; %store
            lowconflowvol_ctrl(session,2) = overlap{3,session}; %store
        elseif environment == 3 %environment
            highconflowvol_ctrl(session,1) = accuracy; %store
            highconflowvol_ctrl(session,2) = overlap{3,session}; %store
        elseif environment == 4 %environment
            noconfhighvol_ctrl(session,1) = accuracy; %store
            noconfhighvol_ctrl(session,2) = overlap{3,session}; %store
        elseif environment == 5 %environment
            lowconfhighvol_ctrl(session,1) = accuracy; %store
            lowconfhighvol_ctrl(session,2) = overlap{3,session}; %store
        elseif environment == 6 %environment
            highconfhighvol_ctrl(session,1) = accuracy; %store
            highconfhighvol_ctrl(session,2) = overlap{3,session}; %store
        end %ending "if" statement for environments
    end %ending "for" loop going through sessions
end %ending "for" loop going through environments

%% Control Probability Matching - prob. matching

for environment = 1:9 %go through all environments doing the streamlined way to pull data out
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
        filename = load('C:\Users\julie\OneDrive\Documents\MATLAB\example data\NoConfLowVolStim_withDTs.mat');
    elseif environment == 8 %starting "if" statement to load each individual overlapped
        filename = load('C:\Users\julie\OneDrive\Documents\MATLAB\example data\LowConfLowVolStim_withDTs.mat');
    elseif environment == 9 %starting "if" statement to load each individual overlapped
        filename = load('C:\Users\julie\OneDrive\Documents\MATLAB\example data\HighConfLowVolStim_withDTs.mat');
    end %ending "if" statement for environments
    overlap = filename.overlap; %pull out cells
    for session = 1:length(overlap) %go through "overlap"
        data = overlap{1,session}; %pull out data
        wantedEM = []; %space
        w=1; %format
        for ems = 2:length(data) %go through data
            if data(ems,2) ~= 0 && data(ems-1,2) == 0 %when a turn happens...
                wantedEM(w,1) = data(ems,2); %pull over EM
                w=w+1; %format
            end %ending "if" statement looking for turns
        end %ending "for" loop going through data
        C = [1 2 5 6]; %correct EMs
        I = [3 4 7 8]; %incorrect EMs
        rew = [1 3 5 7]; %rewarded EMs
        c = []; i = []; %space
        cor = 1; incor = 1; %format
        for sep = 1:length(wantedEM) %go through wantedEM
            if ismember(wantedEM(sep),C) %if correct...
                c(cor) = wantedEM(sep); %copy
                cor=cor+1; %format
            elseif ismember(wantedEM(sep),I) %if incorrect...
                i(incor) = wantedEM(sep); %copy
                incor=incor+1; %format
            end %ending "if" statement separating into the two types
        end %ending "for" loop going through wantedEM
        for binary = 1:length(c) %go through c
            if ismember(c(binary),rew) %if rewarded...
                c(binary) = 1; %change to a 1
            else %if unrewarded...
                c(binary) = 0; %change to a 0
            end %ending "if" statement turning into binary
        end %ending "for" loop going through c
        for binary = 1:length(i) %go through i
            if ismember(i(binary),rew) %if rewarded...
                i(binary) = 1; %change to a 1
            else %if unrewarded...
                i(binary) = 0; %change to a 0
            end %ending "if" statement turning into binary
        end %ending "for" loop going through i
        correct_rewprob = sum(c) / length(c); %actual reward probability of the correct decisions
        correct_choiceprob = length(c) / (length(c) + length(i)); %proportion chosen
        incorrect_rewprob = sum(i) / length(i); %actual reward probability of the incorrect decisions
        incorrect_choiceprob = length(i) / (length(i) + length(c)); %proportion chosen
        correct_difference = correct_choiceprob - correct_rewprob; %probability matching?
        incorrect_difference = incorrect_choiceprob - incorrect_rewprob; %probability matching?
        if environment == 1 %environment
            noconflowvol_ctrl(session,1) = correct_rewprob; %store
            noconflowvol_ctrl(session,2) = correct_choiceprob; %store
            noconflowvol_ctrl(session,3) = correct_difference; %store
            noconflowvol_ctrl(session,4) = incorrect_rewprob; %store
            noconflowvol_ctrl(session,5) = incorrect_choiceprob; %store
            noconflowvol_ctrl(session,6) = incorrect_difference; %store
            noconflowvol_ctrl(session,7) = overlap{3,session}; %ID
        elseif environment == 2 %environment
            lowconflowvol_ctrl(session,1) = correct_rewprob; %store
            lowconflowvol_ctrl(session,2) = correct_choiceprob; %store
            lowconflowvol_ctrl(session,3) = correct_difference; %store
            lowconflowvol_ctrl(session,4) = incorrect_rewprob; %store
            lowconflowvol_ctrl(session,5) = incorrect_choiceprob; %store
            lowconflowvol_ctrl(session,6) = incorrect_difference; %store
            lowconflowvol_ctrl(session,7) = overlap{3,session}; %ID
        elseif environment == 3 %environment
            highconflowvol_ctrl(session,1) = correct_rewprob; %store
            highconflowvol_ctrl(session,2) = correct_choiceprob; %store
            highconflowvol_ctrl(session,3) = correct_difference; %store
            highconflowvol_ctrl(session,4) = incorrect_rewprob; %store
            highconflowvol_ctrl(session,5) = incorrect_choiceprob; %store
            highconflowvol_ctrl(session,6) = incorrect_difference; %store
            highconflowvol_ctrl(session,7) = overlap{3,session}; %ID
        elseif environment == 4 %environment
            noconfhighvol_ctrl(session,1) = correct_rewprob; %store
            noconfhighvol_ctrl(session,2) = correct_choiceprob; %store
            noconfhighvol_ctrl(session,3) = correct_difference; %store
            noconfhighvol_ctrl(session,4) = incorrect_rewprob; %store
            noconfhighvol_ctrl(session,5) = incorrect_choiceprob; %store
            noconfhighvol_ctrl(session,6) = incorrect_difference; %store
            noconfhighvol_ctrl(session,7) = overlap{3,session}; %ID
        elseif environment == 5 %environment
            lowconfhighvol_ctrl(session,1) = correct_rewprob; %store
            lowconfhighvol_ctrl(session,2) = correct_choiceprob; %store
            lowconfhighvol_ctrl(session,3) = correct_difference; %store
            lowconfhighvol_ctrl(session,4) = incorrect_rewprob; %store
            lowconfhighvol_ctrl(session,5) = incorrect_choiceprob; %store
            lowconfhighvol_ctrl(session,6) = incorrect_difference; %store
            lowconfhighvol_ctrl(session,7) = overlap{3,session}; %ID
        elseif environment == 6 %environment
            highconfhighvol_ctrl(session,1) = correct_rewprob; %store
            highconfhighvol_ctrl(session,2) = correct_choiceprob; %store
            highconfhighvol_ctrl(session,3) = correct_difference; %store
            highconfhighvol_ctrl(session,4) = incorrect_rewprob; %store
            highconfhighvol_ctrl(session,5) = incorrect_choiceprob; %store
            highconfhighvol_ctrl(session,6) = incorrect_difference; %store
            highconfhighvol_ctrl(session,7) = overlap{3,session}; %ID
        elseif environment == 7 %environment
            noconflowvol_stim(session,1) = correct_rewprob; %store
            noconflowvol_stim(session,2) = correct_choiceprob; %store
            noconflowvol_stim(session,3) = correct_difference; %store
            noconflowvol_stim(session,4) = incorrect_rewprob; %store
            noconflowvol_stim(session,5) = incorrect_choiceprob; %store
            noconflowvol_stim(session,6) = incorrect_difference; %store
            noconflowvol_stim(session,7) = overlap{3,session}; %ID
        elseif environment == 8 %environment
            lowconflowvol_stim(session,1) = correct_rewprob; %store
            lowconflowvol_stim(session,2) = correct_choiceprob; %store
            lowconflowvol_stim(session,3) = correct_difference; %store
            lowconflowvol_stim(session,4) = incorrect_rewprob; %store
            lowconflowvol_stim(session,5) = incorrect_choiceprob; %store
            lowconflowvol_stim(session,6) = incorrect_difference; %store
            lowconflowvol_stim(session,7) = overlap{3,session}; %ID
        elseif environment == 9 %environment
            highconflowvol_stim(session,1) = correct_rewprob; %store
            highconflowvol_stim(session,2) = correct_choiceprob; %store
            highconflowvol_stim(session,3) = correct_difference; %store
            highconflowvol_stim(session,4) = incorrect_rewprob; %store
            highconflowvol_stim(session,5) = incorrect_choiceprob; %store
            highconflowvol_stim(session,6) = incorrect_difference; %store
            highconflowvol_stim(session,7) = overlap{3,session}; %ID
        end %ending "if" statement saving data
    end %ending "for" loop going through sessions
    ispn = [1 5 7 9 11]; %iSPN
    dspn = [2 4 6 8 12]; %dSPN
    if environment == 1 %environment check
        i=1; d=1; %format
        for genotype = 1:length(noconflowvol_ctrl) %go through computed data
            if ismember(noconflowvol_ctrl(genotype,7),ispn) %if ispn...
                ispn_noconflowvol_ctrl(i,1:6) = noconflowvol_ctrl(genotype,1:6); %store data
                i=i+1; %format
            elseif ismember(noconflowvol_ctrl(genotype,7),dspn) %if dspn...
                dspn_noconflowvol_ctrl(d,1:6) = noconflowvol_ctrl(genotype,1:6); %store data
                d=d+1; %format
            end %ending "if" statement separating by genotype
        end %ending "for" loop going through computed data
    elseif environment == 2 %environment check
        i=1; d=1; %format
        for genotype = 1:length(lowconflowvol_ctrl) %go through computed data
            if ismember(lowconflowvol_ctrl(genotype,7),ispn) %if ispn...
                ispn_lowconflowvol_ctrl(i,1:6) = lowconflowvol_ctrl(genotype,1:6); %store data
                i=i+1; %format
            elseif ismember(lowconflowvol_ctrl(genotype,7),dspn) %if dspn...
                dspn_lowconflowvol_ctrl(d,1:6) = lowconflowvol_ctrl(genotype,1:6); %store data
                d=d+1; %format
            end %ending "if" statement separating by genotype
        end %ending "for" loop going through computed data
    elseif environment == 3 %environment check
        i=1; d=1; %format
        for genotype = 1:length(highconflowvol_ctrl) %go through computed data
            if ismember(highconflowvol_ctrl(genotype,7),ispn) %if ispn...
                ispn_highconflowvol_ctrl(i,1:6) = highconflowvol_ctrl(genotype,1:6); %store data
                i=i+1; %format
            elseif ismember(highconflowvol_ctrl(genotype,7),dspn) %if dspn...
                dspn_highconflowvol_ctrl(d,1:6) = highconflowvol_ctrl(genotype,1:6); %store data
                d=d+1; %format
            end %ending "if" statement separating by genotype
        end %ending "for" loop going through computed data
    elseif environment == 4 %environment check
        i=1; d=1; %format
        for genotype = 1:length(noconfhighvol_ctrl) %go through computed data
            if ismember(noconfhighvol_ctrl(genotype,7),ispn) %if ispn...
                ispn_noconfhighvol_ctrl(i,1:6) = noconfhighvol_ctrl(genotype,1:6); %store data
                i=i+1; %format
            elseif ismember(noconfhighvol_ctrl(genotype,7),dspn) %if dspn...
                dspn_noconfhighvol_ctrl(d,1:6) = noconfhighvol_ctrl(genotype,1:6); %store data
                d=d+1; %format
            end %ending "if" statement separating by genotype
        end %ending "for" loop going through computed data
    elseif environment == 5 %environment check
        i=1; d=1; %format
        for genotype = 1:length(lowconfhighvol_ctrl) %go through computed data
            if ismember(lowconfhighvol_ctrl(genotype,7),ispn) %if ispn...
                ispn_lowconfhighvol_ctrl(i,1:6) = lowconfhighvol_ctrl(genotype,1:6); %store data
                i=i+1; %format
            elseif ismember(lowconfhighvol_ctrl(genotype,7),dspn) %if dspn...
                dspn_lowconfhighvol_ctrl(d,1:6) = lowconfhighvol_ctrl(genotype,1:6); %store data
                d=d+1; %format
            end %ending "if" statement separating by genotype
        end %ending "for" loop going through computed data
    elseif environment == 6 %environment check
        i=1; d=1; %format
        for genotype = 1:length(highconfhighvol_ctrl) %go through computed data
            if ismember(highconfhighvol_ctrl(genotype,7),ispn) %if ispn...
                ispn_highconfhighvol_ctrl(i,1:6) = highconfhighvol_ctrl(genotype,1:6); %store data
                i=i+1; %format
            elseif ismember(highconfhighvol_ctrl(genotype,7),dspn) %if dspn...
                dspn_highconfhighvol_ctrl(d,1:6) = highconfhighvol_ctrl(genotype,1:6); %store data
                d=d+1; %format
            end %ending "if" statement separating by genotype
        end %ending "for" loop going through computed data
    elseif environment == 7 %environment check
        i=1; d=1; %format
        for genotype = 1:length(noconflowvol_stim) %go through computed data
            if ismember(noconflowvol_stim(genotype,7),ispn) %if ispn...
                ispn_noconflowvol_stim(i,1:6) = noconflowvol_stim(genotype,1:6); %store data
                i=i+1; %format
            elseif ismember(noconflowvol_stim(genotype,7),dspn) %if dspn...
                dspn_noconflowvol_stim(d,1:6) = noconflowvol_stim(genotype,1:6); %store data
                d=d+1; %format
            end %ending "if" statement separating by genotype
        end %ending "for" loop going through computed data
    elseif environment == 8 %environment check
        i=1; d=1; %format
        for genotype = 1:length(lowconflowvol_stim) %go through computed data
            if ismember(lowconflowvol_stim(genotype,7),ispn) %if ispn...
                ispn_lowconflowvol_stim(i,1:6) = lowconflowvol_stim(genotype,1:6); %store data
                i=i+1; %format
            elseif ismember(lowconflowvol_stim(genotype,7),dspn) %if dspn...
                dspn_lowconflowvol_stim(d,1:6) = lowconflowvol_stim(genotype,1:6); %store data
                d=d+1; %format
            end %ending "if" statement separating by genotype
        end %ending "for" loop going through computed data
    elseif environment == 9 %environment check
        i=1; d=1; %format
        for genotype = 1:length(highconflowvol_stim) %go through computed data
            if ismember(highconflowvol_stim(genotype,7),ispn) %if ispn...
                ispn_highconflowvol_stim(i,1:6) = highconflowvol_stim(genotype,1:6); %store data
                i=i+1; %format
            elseif ismember(highconflowvol_stim(genotype,7),dspn) %if dspn...
                dspn_highconflowvol_stim(d,1:6) = highconflowvol_stim(genotype,1:6); %store data
                d=d+1; %format
            end %ending "if" statement separating by genotype
        end %ending "for" loop going through computed data
    end %ending "if" statement for environments
end %ending "for" loop going through environments

%% distance from OPC - opc

for environment = 1:9 %go through all environments doing the streamlined way to pull data out
    if environment == 1 %starting "if" statement to load each individual overlapped
        filename = load('C:\Users\julie\OneDrive\Documents\MATLAB\example data\NoConfLowVolControl_withDTs.mat');
        conf = 1; vol = 2; condition = 0; %parameters
    elseif environment == 2 %starting "if" statement to load each individual overlapped
        filename = load('C:\Users\julie\OneDrive\Documents\MATLAB\example data\LowConfLowVolControl_withDTs.mat');
         conf = 2; vol = 2; condition = 0;
    elseif environment == 3 %starting "if" statement to load each individual overlapped
        filename = load('C:\Users\julie\OneDrive\Documents\MATLAB\example data\HighConfLowVolControl_withDTs.mat');
         conf = 3; vol = 2; condition = 0;
    elseif environment == 4 %starting "if" statement to load each individual overlapped
        filename = load('C:\Users\julie\OneDrive\Documents\MATLAB\example data\NoConfHighVolControl_withDTs.mat');
         conf = 1; vol = 3; condition = 0;
    elseif environment == 5 %starting "if" statement to load each individual overlapped
        filename = load('C:\Users\julie\OneDrive\Documents\MATLAB\example data\LowConfHighVolControl_withDTs.mat');
         conf = 2; vol = 3; condition = 0;
    elseif environment == 6 %starting "if" statement to load each individual overlapped
        filename = load('C:\Users\julie\OneDrive\Documents\MATLAB\example data\HighConfHighVolControl_withDTs.mat');
         conf = 3; vol = 3; condition = 0;
    elseif environment == 7 %starting "if" statement to load each individual overlapped
        filename = load('C:\Users\julie\OneDrive\Documents\MATLAB\example data\NoConfLowVolStim_withDTs.mat');
         conf = 1; vol = 2; condition = 1;
    elseif environment == 8 %starting "if" statement to load each individual overlapped
        filename = load('C:\Users\julie\OneDrive\Documents\MATLAB\example data\LowConfLowVolStim_withDTs.mat');
         conf = 2; vol = 2; condition = 1;
    elseif environment == 9 %starting "if" statement to load each individual overlapped
        filename = load('C:\Users\julie\OneDrive\Documents\MATLAB\example data\HighConfLowVolStim_withDTs.mat');
         conf = 3; vol = 2; condition = 1;
    end %ending "if" statement for environments
    overlap = filename.overlap; %pull out cells
    for session = 1:length(overlap) %go through "overlap"
        data = overlap{1,session}; %pull out data
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
        dt_non = []; %space
        dt_non(:,1) = decisiontimes(:,3); %copy over DTs
        for N = 2:length(decisiontimes) %go through dts
            dt_non(N-1,2) = decisiontimes(N,1) - decisiontimes(N-1,2); %calculating each trial's non-dt (total execution time)
        end %ending "for" loop going through DTs to calculate non-DTs
        relationship = []; %space to store DT / non-DT
        for r = 1:length(dt_non)-1 %last one won't have any non-DT
            relationship(r) = dt_non(r,1) / dt_non(r,2); %do the calculation
        end %ending "for" loop going through each trial
        division = nanmedian(relationship); %save the median
        EM = data(:,2); %makes a variable for just EMs
        wantedEM = []; %making an open vector for EMs that we want
        for q = 2:length(EM) %starting with second value & running through all EMs
            if EM(q)~=EM(q-1) %if an EM is different from the one preceding it...
                wantedEM = [wantedEM,EM(q)]; %...include those EMs in the wantedEM vector
            end %ending the "if" statement
        end %ending the "for" loop
        wantedEM(ismember(wantedEM,[0,11,12])) = []; %retains only EMs pertaining to turns
        incorrect = [3 4 7 8]; %incorrect EMs
        binary = []; %space
        for IB = 1:length(wantedEM) %go through EMs
            if ismember(wantedEM(IB),incorrect) %if incorrect...
                binary(IB) = 1; %fill in a 1
            else %if correct...
                binary(IB) = 0; %fill in a 0
            end %ending "if" statement making binary
        end %ending "for" loop going through EMs
        pIncor = sum(binary)/length(binary); %percent incorrect
        animal_id = overlap{3,session}; %animal ID
        opc = []; %space to load in OPC
        o=1; %format
        for OPC = 1:length(OPC_curve) %go through opc curve
            if OPC_curve(OPC,4) == animal_id && OPC_curve(OPC,5) == conf && OPC_curve(OPC,6) == vol && OPC_curve(OPC,7) == condition %if we have our criteria met...
                opc(o,1:2) = OPC_curve(OPC,1:2); %copy curve data
                o=o+1; %format
                q_value = OPC_curve(OPC,3); %mark q value
            end %ending "if" statement looking for parameters
        end %ending "for" loop going through opc curve
        y_interp = interp1(opc(:,1), opc(:,2), pIncor, 'linear'); %Interpolate the y-value of the curve at x_point
        vertical_distance = abs(division - y_interp); % Calculate the vertical distance
        if environment == 1 %environment
            noconflowvol_ctrl(session,1) = vertical_distance; %store
            noconflowvol_ctrl(session,2) = q_value; %store
            noconflowvol_ctrl(session,3) = overlap{3,session}; %store
        elseif environment == 2 %environment
            lowconflowvol_ctrl(session,1) = vertical_distance; %store
            lowconflowvol_ctrl(session,2) = q_value; %store
            lowconflowvol_ctrl(session,3) = overlap{3,session}; %store
        elseif environment == 3 %environment
            highconflowvol_ctrl(session,1) = vertical_distance; %store
            highconflowvol_ctrl(session,2) = q_value; %store
            highconflowvol_ctrl(session,3) = overlap{3,session}; %store
        elseif environment == 4 %environment
            noconfhighvol_ctrl(session,1) = vertical_distance; %store
            noconfhighvol_ctrl(session,2) = q_value; %store
            noconfhighvol_ctrl(session,3) = overlap{3,session}; %store
        elseif environment == 5 %environment
            lowconfhighvol_ctrl(session,1) = vertical_distance; %store
            lowconfhighvol_ctrl(session,2) = q_value; %store
            lowconfhighvol_ctrl(session,3) = overlap{3,session}; %store
        elseif environment == 6 %environment
            highconfhighvol_ctrl(session,1) = vertical_distance; %store
            highconfhighvol_ctrl(session,2) = q_value; %store
            highconfhighvol_ctrl(session,3) = overlap{3,session}; %store
        elseif environment == 7 %environment
            noconflowvol_stim(session,1) = vertical_distance; %store
            noconflowvol_stim(session,2) = q_value; %store
            noconflowvol_stim(session,3) = overlap{3,session}; %store
        elseif environment == 8 %environment
            lowconflowvol_stim(session,1) = vertical_distance; %store
            lowconflowvol_stim(session,2) = q_value; %store
            lowconflowvol_stim(session,3) = overlap{3,session}; %store
        elseif environment == 9 %environment
            highconflowvol_stim(session,1) = vertical_distance; %store
            highconflowvol_stim(session,2) = q_value; %store
            highconflowvol_stim(session,3) = overlap{3,session}; %store
        end %ending "if" statement saving data
    end %ending "for" loop going through sessions
    ispn = [1 5 7 9 11]; %iSPN
    dspn = [2 4 6 8 12]; %dSPN
    if environment == 1 %environment check
        i=1; d=1; %format
        for genotype = 1:length(noconflowvol_ctrl) %go through computed data
            if ismember(noconflowvol_ctrl(genotype,3),ispn) %if ispn...
                ispn_noconflowvol_ctrl(i,1:3) = noconflowvol_ctrl(genotype,1:3); %store data
                i=i+1; %format
            elseif ismember(noconflowvol_ctrl(genotype,3),dspn) %if dspn...
                dspn_noconflowvol_ctrl(d,1:3) = noconflowvol_ctrl(genotype,1:3); %store data
                d=d+1; %format
            end %ending "if" statement separating by genotype
        end %ending "for" loop going through computed data
    elseif environment == 2 %environment check
        i=1; d=1; %format
        for genotype = 1:length(lowconflowvol_ctrl) %go through computed data
            if ismember(lowconflowvol_ctrl(genotype,3),ispn) %if ispn...
                ispn_lowconflowvol_ctrl(i,1:3) = lowconflowvol_ctrl(genotype,1:3); %store data
                i=i+1; %format
            elseif ismember(lowconflowvol_ctrl(genotype,3),dspn) %if dspn...
                dspn_lowconflowvol_ctrl(d,1:3) = lowconflowvol_ctrl(genotype,1:3); %store data
                d=d+1; %format
            end %ending "if" statement separating by genotype
        end %ending "for" loop going through computed data
    elseif environment == 3 %environment check
        i=1; d=1; %format
        for genotype = 1:length(highconflowvol_ctrl) %go through computed data
            if ismember(highconflowvol_ctrl(genotype,3),ispn) %if ispn...
                ispn_highconflowvol_ctrl(i,1:3) = highconflowvol_ctrl(genotype,1:3); %store data
                i=i+1; %format
            elseif ismember(highconflowvol_ctrl(genotype,3),dspn) %if dspn...
                dspn_highconflowvol_ctrl(d,1:3) = highconflowvol_ctrl(genotype,1:3); %store data
                d=d+1; %format
            end %ending "if" statement separating by genotype
        end %ending "for" loop going through computed data
    elseif environment == 4 %environment check
        i=1; d=1; %format
        for genotype = 1:length(noconfhighvol_ctrl) %go through computed data
            if ismember(noconfhighvol_ctrl(genotype,3),ispn) %if ispn...
                ispn_noconfhighvol_ctrl(i,1:3) = noconfhighvol_ctrl(genotype,1:3); %store data
                i=i+1; %format
            elseif ismember(noconfhighvol_ctrl(genotype,3),dspn) %if dspn...
                dspn_noconfhighvol_ctrl(d,1:3) = noconfhighvol_ctrl(genotype,1:3); %store data
                d=d+1; %format
            end %ending "if" statement separating by genotype
        end %ending "for" loop going through computed data
    elseif environment == 5 %environment check
        i=1; d=1; %format
        for genotype = 1:length(lowconfhighvol_ctrl) %go through computed data
            if ismember(lowconfhighvol_ctrl(genotype,3),ispn) %if ispn...
                ispn_lowconfhighvol_ctrl(i,1:3) = lowconfhighvol_ctrl(genotype,1:3); %store data
                i=i+1; %format
            elseif ismember(lowconfhighvol_ctrl(genotype,3),dspn) %if dspn...
                dspn_lowconfhighvol_ctrl(d,1:3) = lowconfhighvol_ctrl(genotype,1:3); %store data
                d=d+1; %format
            end %ending "if" statement separating by genotype
        end %ending "for" loop going through computed data
    elseif environment == 6 %environment check
        i=1; d=1; %format
        for genotype = 1:length(highconfhighvol_ctrl) %go through computed data
            if ismember(highconfhighvol_ctrl(genotype,3),ispn) %if ispn...
                ispn_highconfhighvol_ctrl(i,1:3) = highconfhighvol_ctrl(genotype,1:3); %store data
                i=i+1; %format
            elseif ismember(highconfhighvol_ctrl(genotype,3),dspn) %if dspn...
                dspn_highconfhighvol_ctrl(d,1:3) = highconfhighvol_ctrl(genotype,1:3); %store data
                d=d+1; %format
            end %ending "if" statement separating by genotype
        end %ending "for" loop going through computed data
    elseif environment == 7 %environment check
        i=1; d=1; %format
        for genotype = 1:length(noconflowvol_stim) %go through computed data
            if ismember(noconflowvol_stim(genotype,3),ispn) %if ispn...
                ispn_noconflowvol_stim(i,1:3) = noconflowvol_stim(genotype,1:3); %store data
                i=i+1; %format
            elseif ismember(noconflowvol_stim(genotype,3),dspn) %if dspn...
                dspn_noconflowvol_stim(d,1:3) = noconflowvol_stim(genotype,1:3); %store data
                d=d+1; %format
            end %ending "if" statement separating by genotype
        end %ending "for" loop going through computed data
    elseif environment == 8 %environment check
        i=1; d=1; %format
        for genotype = 1:length(lowconflowvol_stim) %go through computed data
            if ismember(lowconflowvol_stim(genotype,3),ispn) %if ispn...
                ispn_lowconflowvol_stim(i,1:3) = lowconflowvol_stim(genotype,1:3); %store data
                i=i+1; %format
            elseif ismember(lowconflowvol_stim(genotype,3),dspn) %if dspn...
                dspn_lowconflowvol_stim(d,1:3) = lowconflowvol_stim(genotype,1:3); %store data
                d=d+1; %format
            end %ending "if" statement separating by genotype
        end %ending "for" loop going through computed data
    elseif environment == 9 %environment check
        i=1; d=1; %format
        for genotype = 1:length(highconflowvol_stim) %go through computed data
            if ismember(highconflowvol_stim(genotype,3),ispn) %if ispn...
                ispn_highconflowvol_stim(i,1:3) = highconflowvol_stim(genotype,1:3); %store data
                i=i+1; %format
            elseif ismember(highconflowvol_stim(genotype,3),dspn) %if dspn...
                dspn_highconflowvol_stim(d,1:3) = highconflowvol_stim(genotype,1:3); %store data
                d=d+1; %format
            end %ending "if" statement separating by genotype
        end %ending "for" loop going through computed data
    end %ending "if" statement for environments
end %ending "for" loop going through environments

%% cognitive energy cost - cec

for environment = 1:9 %go through all environments doing the streamlined way to pull data out
    if environment == 1 %starting "if" statement to load each individual overlapped
        filename = load('C:\Users\julie\OneDrive\Documents\MATLAB\example data\NoConfLowVolControl_withDTs.mat');
        conf = 1; vol = 2; condition = 0; %parameters
    elseif environment == 2 %starting "if" statement to load each individual overlapped
        filename = load('C:\Users\julie\OneDrive\Documents\MATLAB\example data\LowConfLowVolControl_withDTs.mat');
         conf = 2; vol = 2; condition = 0;
    elseif environment == 3 %starting "if" statement to load each individual overlapped
        filename = load('C:\Users\julie\OneDrive\Documents\MATLAB\example data\HighConfLowVolControl_withDTs.mat');
         conf = 3; vol = 2; condition = 0;
    elseif environment == 4 %starting "if" statement to load each individual overlapped
        filename = load('C:\Users\julie\OneDrive\Documents\MATLAB\example data\NoConfHighVolControl_withDTs.mat');
         conf = 1; vol = 3; condition = 0;
    elseif environment == 5 %starting "if" statement to load each individual overlapped
        filename = load('C:\Users\julie\OneDrive\Documents\MATLAB\example data\LowConfHighVolControl_withDTs.mat');
         conf = 2; vol = 3; condition = 0;
    elseif environment == 6 %starting "if" statement to load each individual overlapped
        filename = load('C:\Users\julie\OneDrive\Documents\MATLAB\example data\HighConfHighVolControl_withDTs.mat');
         conf = 3; vol = 3; condition = 0;
    elseif environment == 7 %starting "if" statement to load each individual overlapped
        filename = load('C:\Users\julie\OneDrive\Documents\MATLAB\example data\NoConfLowVolStim_withDTs.mat');
         conf = 1; vol = 2; condition = 1;
    elseif environment == 8 %starting "if" statement to load each individual overlapped
        filename = load('C:\Users\julie\OneDrive\Documents\MATLAB\example data\LowConfLowVolStim_withDTs.mat');
         conf = 2; vol = 2; condition = 1;
    elseif environment == 9 %starting "if" statement to load each individual overlapped
        filename = load('C:\Users\julie\OneDrive\Documents\MATLAB\example data\HighConfLowVolStim_withDTs.mat');
         conf = 3; vol = 2; condition = 1;
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
        xy = []; %place to store location data2
        xy(1,:) = data2(:,13); %copy over the x info from data2 file
        xy(2,:) = data2(:,14); %copy over y info from data2 file
        disp3=sqrt(diff2(xy(1,:),3).^2+diff2(xy(2,:),3).^2);  %equation from Eric for velocity
        turn1 = smooth(disp3,30)'; %smooth the velocity data2
        turn1(2,:) = data2(1:end-3,17); %fill in the second part of velocity data2 with the hall location
        stuff1 = NaN(length(turn1),1); %place to store hall 1
        stuff2 = NaN(length(turn1),1); %place to store hall 2
        stuff3 = NaN(length(turn1),1); %place to store hall 3
        stuff0 = NaN(length(turn1),1); %place to store center
        for splittingup = 1:length(turn1) %going through the velocity data2
            if turn1(2,splittingup) == 1 %if the animal was in hall 1...
                stuff1(splittingup) = turn1(1,splittingup); %fill that in stuff 1
            elseif turn1(2,splittingup) == 2 %if the animal was in hall 2...
                stuff2(splittingup) = turn1(1,splittingup); %fill that in stuff 2
            elseif turn1(2,splittingup) == 3 %if the animal was in hall 3...
                stuff3(splittingup) = turn1(1,splittingup); %fill that in stuff 3
            elseif turn1(2,splittingup) == 0 %if the animal was in the center...
                stuff0(splittingup) = turn1(1,splittingup); %fill that in stuff 0
            end %ending "if" statement looking for location
        end %ending "for" loop going through velocity data2
        stuff1(:,2) = data2(1:end-3,1); %add in time info
        stuff2(:,2) = data2(1:end-3,1); %add in time info
        stuff3(:,2) = data2(1:end-3,1); %add in time info
        stuff0(:,2) = data2(1:end-3,1); %add in time info
        turn1(3,:) = data2(1:end-3,2); %fill in the EMs to turn1
        turn1(4,:) = 1:length(turn1); %fill in a spacer
        allempositions = []; %place to store all EM positions
        emspot = 1; %formatting
        ems = [1:8]; %any em
        for stars = 2:length(turn1) %going through the data2
            if ismember(turn1(3,stars),ems) && turn1(3,stars-1) == 0 %if a turn is first detected...
                allempositions(emspot) = turn1(4,stars); %mark down that spot position number
                emspot = emspot + 1; %formatting
            end %ending "if" statement looking for ems
        end %ending "for" loop gooing through the data2
        veltimes = []; %place to store a lot of the info
        veltimes(1,:) = turn1(1,:); %copy over the velocities
        veltimes(2,:) = data2(1:end-3,1); %copy over the times
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
            veltimes(4,allempositions(firstinstance):allempositions(firstinstance + 1)) = crossing(1,:); %replace the veltimes with the cleared data2
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
            veltimes(4,allempositions(firstinstance):allempositions(firstinstance + 1)) = crossing(1,:); %replace the veltimes with the cleared data2
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
        dt_non = []; %space
        detection_time = []; %space
        EMs = []; %space
        DT = 1; %format
        for dt = 2:length(data2) %go through data
            if data2(dt,2) ~= data2(dt-1,2) && data2(dt-1,2) == 0 %turn detection...
                detection_time(DT) = data2(dt,1); %copy the turn detection time
                EMs(DT) = data2(dt,2); %copy the EM
                DT = DT + 1; %format
            end %ending "if" statement for turn detection
        end %ending "for" loop going through data
        for dtn = 1:length(detection_time)-1 %go through detection times
            dt_non(dtn) = detection_time(dtn+1) - detection_time(dtn); %calculate the T
        end %ending "for" loop going through detection times
        relationship = []; %space to store DT / non-DT
        L = length(decisiontimes); %set the length
        for r = 1:L %last one won't have any non-DT
            relationship(r) = decisiontimes(r,3) / dt_non(r); %do the calculation
        end %ending "for" loop going through each trial
        division = nanmean(relationship); %save the average
        e = learning_rate + explore_exploit * division - abs(Q_bias); %calculate cognitive energy
        if e < 0 %if we have a negative...
            e = 0; %set to 0
        end %ending "if" statement for negative energy
        reward = [1 3 5 7]; %rewarded EMs
        for r = 1:length(EMs) %go through EMs
            if ismember(EMs(r),reward) %if rewarded...
                EMs(r) = 1; %mark 1
            else %if unrewarded...
                EMs(r) = 0; %mark 0
            end %ending "if" statement for rewarded vs unrewarded
        end %ending "for" loop going through EMs
        time = data2(end,1); %how long was the session?
        time = time / 1000; %ms to s
        time = time / 60; %s to min
        reward_rate = sum(EMs) / time; %calculate reward rate
        if environment == 1 %environment
            noconflowvol_ctrl(session,1) = e; %store
            noconflowvol_ctrl(session,2) = reward_rate; %store
            noconflowvol_ctrl(session,3) = overlap{3,session}; %store
        elseif environment == 2 %environment
            lowconflowvol_ctrl(session,1) = e; %store
            lowconflowvol_ctrl(session,2) = reward_rate; %store
            lowconflowvol_ctrl(session,3) = overlap{3,session}; %store
        elseif environment == 3 %environment
            highconflowvol_ctrl(session,1) = e; %store
            highconflowvol_ctrl(session,2) = reward_rate; %store
            highconflowvol_ctrl(session,3) = overlap{3,session}; %store
        elseif environment == 4 %environment
            noconfhighvol_ctrl(session,1) = e; %store
            noconfhighvol_ctrl(session,2) = reward_rate; %store
            noconfhighvol_ctrl(session,3) = overlap{3,session}; %store
        elseif environment == 5 %environment
            lowconfhighvol_ctrl(session,1) = e; %store
            lowconfhighvol_ctrl(session,2) = reward_rate; %store
            lowconfhighvol_ctrl(session,3) = overlap{3,session}; %store
        elseif environment == 6 %environment
            highconfhighvol_ctrl(session,1) = e; %store
            highconfhighvol_ctrl(session,2) = reward_rate; %store
            highconfhighvol_ctrl(session,3) = overlap{3,session}; %store
        elseif environment == 7 %environment
            noconflowvol_stim(session,1) = e; %store
            noconflowvol_stim(session,2) = reward_rate; %store
            noconflowvol_stim(session,3) = overlap{3,session}; %store
        elseif environment == 8 %environment
            lowconflowvol_stim(session,1) = e; %store
            lowconflowvol_stim(session,2) = reward_rate; %store
            lowconflowvol_stim(session,3) = overlap{3,session}; %store
        elseif environment == 9 %environment
            highconflowvol_stim(session,1) = e; %store
            highconflowvol_stim(session,2) = reward_rate; %store
            highconflowvol_stim(session,3) = overlap{3,session}; %store
        end %ending "if" statement saving data
    end %ending "for" loop going through sessions
    ispn = [1 5 7 9 11]; %iSPN
    dspn = [2 4 6 8 12]; %dSPN
    if environment == 1 %environment check
        i=1; d=1; %format
        for genotype = 1:length(noconflowvol_ctrl) %go through computed data
            if ismember(noconflowvol_ctrl(genotype,3),ispn) %if ispn...
                ispn_noconflowvol_ctrl(i,1:3) = noconflowvol_ctrl(genotype,1:3); %store data
                i=i+1; %format
            elseif ismember(noconflowvol_ctrl(genotype,3),dspn) %if dspn...
                dspn_noconflowvol_ctrl(d,1:3) = noconflowvol_ctrl(genotype,1:3); %store data
                d=d+1; %format
            end %ending "if" statement separating by genotype
        end %ending "for" loop going through computed data
    elseif environment == 2 %environment check
        i=1; d=1; %format
        for genotype = 1:length(lowconflowvol_ctrl) %go through computed data
            if ismember(lowconflowvol_ctrl(genotype,3),ispn) %if ispn...
                ispn_lowconflowvol_ctrl(i,1:3) = lowconflowvol_ctrl(genotype,1:3); %store data
                i=i+1; %format
            elseif ismember(lowconflowvol_ctrl(genotype,3),dspn) %if dspn...
                dspn_lowconflowvol_ctrl(d,1:3) = lowconflowvol_ctrl(genotype,1:3); %store data
                d=d+1; %format
            end %ending "if" statement separating by genotype
        end %ending "for" loop going through computed data
    elseif environment == 3 %environment check
        i=1; d=1; %format
        for genotype = 1:length(highconflowvol_ctrl) %go through computed data
            if ismember(highconflowvol_ctrl(genotype,3),ispn) %if ispn...
                ispn_highconflowvol_ctrl(i,1:3) = highconflowvol_ctrl(genotype,1:3); %store data
                i=i+1; %format
            elseif ismember(highconflowvol_ctrl(genotype,3),dspn) %if dspn...
                dspn_highconflowvol_ctrl(d,1:3) = highconflowvol_ctrl(genotype,1:3); %store data
                d=d+1; %format
            end %ending "if" statement separating by genotype
        end %ending "for" loop going through computed data
    elseif environment == 4 %environment check
        i=1; d=1; %format
        for genotype = 1:length(noconfhighvol_ctrl) %go through computed data
            if ismember(noconfhighvol_ctrl(genotype,3),ispn) %if ispn...
                ispn_noconfhighvol_ctrl(i,1:3) = noconfhighvol_ctrl(genotype,1:3); %store data
                i=i+1; %format
            elseif ismember(noconfhighvol_ctrl(genotype,3),dspn) %if dspn...
                dspn_noconfhighvol_ctrl(d,1:3) = noconfhighvol_ctrl(genotype,1:3); %store data
                d=d+1; %format
            end %ending "if" statement separating by genotype
        end %ending "for" loop going through computed data
    elseif environment == 5 %environment check
        i=1; d=1; %format
        for genotype = 1:length(lowconfhighvol_ctrl) %go through computed data
            if ismember(lowconfhighvol_ctrl(genotype,3),ispn) %if ispn...
                ispn_lowconfhighvol_ctrl(i,1:3) = lowconfhighvol_ctrl(genotype,1:3); %store data
                i=i+1; %format
            elseif ismember(lowconfhighvol_ctrl(genotype,3),dspn) %if dspn...
                dspn_lowconfhighvol_ctrl(d,1:3) = lowconfhighvol_ctrl(genotype,1:3); %store data
                d=d+1; %format
            end %ending "if" statement separating by genotype
        end %ending "for" loop going through computed data
    elseif environment == 6 %environment check
        i=1; d=1; %format
        for genotype = 1:length(highconfhighvol_ctrl) %go through computed data
            if ismember(highconfhighvol_ctrl(genotype,3),ispn) %if ispn...
                ispn_highconfhighvol_ctrl(i,1:3) = highconfhighvol_ctrl(genotype,1:3); %store data
                i=i+1; %format
            elseif ismember(highconfhighvol_ctrl(genotype,3),dspn) %if dspn...
                dspn_highconfhighvol_ctrl(d,1:3) = highconfhighvol_ctrl(genotype,1:3); %store data
                d=d+1; %format
            end %ending "if" statement separating by genotype
        end %ending "for" loop going through computed data
    elseif environment == 7 %environment check
        i=1; d=1; %format
        for genotype = 1:length(noconflowvol_stim) %go through computed data
            if ismember(noconflowvol_stim(genotype,3),ispn) %if ispn...
                ispn_noconflowvol_stim(i,1:3) = noconflowvol_stim(genotype,1:3); %store data
                i=i+1; %format
            elseif ismember(noconflowvol_stim(genotype,3),dspn) %if dspn...
                dspn_noconflowvol_stim(d,1:3) = noconflowvol_stim(genotype,1:3); %store data
                d=d+1; %format
            end %ending "if" statement separating by genotype
        end %ending "for" loop going through computed data
    elseif environment == 8 %environment check
        i=1; d=1; %format
        for genotype = 1:length(lowconflowvol_stim) %go through computed data
            if ismember(lowconflowvol_stim(genotype,3),ispn) %if ispn...
                ispn_lowconflowvol_stim(i,1:3) = lowconflowvol_stim(genotype,1:3); %store data
                i=i+1; %format
            elseif ismember(lowconflowvol_stim(genotype,3),dspn) %if dspn...
                dspn_lowconflowvol_stim(d,1:3) = lowconflowvol_stim(genotype,1:3); %store data
                d=d+1; %format
            end %ending "if" statement separating by genotype
        end %ending "for" loop going through computed data
    elseif environment == 9 %environment check
        i=1; d=1; %format
        for genotype = 1:length(highconflowvol_stim) %go through computed data
            if ismember(highconflowvol_stim(genotype,3),ispn) %if ispn...
                ispn_highconflowvol_stim(i,1:3) = highconflowvol_stim(genotype,1:3); %store data
                i=i+1; %format
            elseif ismember(highconflowvol_stim(genotype,3),dspn) %if dspn...
                dspn_highconflowvol_stim(d,1:3) = highconflowvol_stim(genotype,1:3); %store data
                d=d+1; %format
            end %ending "if" statement separating by genotype
        end %ending "for" loop going through computed data
    end %ending "if" statement for environments
end %ending "for" loop going through environments