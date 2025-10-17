% If you run this code, it will generate poisson spike times for 100
% neurons for 200 trials. Reaches occur every 10 seconds @ 8, 18, 28s...
% Columns of CC are spike time and neuron ID. Could you run your d2 code on
% it?   
% 
% All neurons ramp up/down their activity to 4x their baseline. The
% baseline is normally distributed around 4hz, so 0.1% will fire less than
% 1hz, but 93% will fire above 2.5hz.
% 
% After some thinking with Keith H, I just want to check what independently
% modulated neurons would do for your d2 metric, e.g. in this case there is
% no network phenomenon - just 100 neurons that are individually and
% independently tuned to the same thing (they could come from 100 different
% brains). 

duration = 2000; % Duration of simulation in seconds
dt = 0.001; % Time step in seconds
t_vec = 0:dt:duration-dt;
facInc=4; %how much larger is peak activity?
nCount=70;

preRampT=300; %ms
maxT=700;
postRampT=1000;
baselineT=10e3-(preRampT+maxT+postRampT);

CC=[];

for nn=1:nCount
% Create a time-varying rate signal (e.g., a sine wave)
rate_avg = abs(randn(1)+4); % Baseline rate. normal dist around 4hz
%rate_modulation = 10 * sin(2 * pi * 2 * t_vec); % 2 Hz modulation
rate_modulation1 = [zeros(1,baselineT), ...
    rate_avg*facInc/preRampT:rate_avg*facInc/preRampT:rate_avg*facInc, ...
    ones(1,maxT)*rate_avg*facInc,...
    fliplr(rate_avg*facInc/postRampT:rate_avg*facInc/postRampT:rate_avg*facInc)];
rate_modulation=repmat(rate_modulation1,[1, length(t_vec)/length(rate_modulation1)]);
rate = rate_avg + rate_modulation;
rate(rate < 0) = 0; % Rates cannot be negative

% Generate spike counts per bin using poissrnd
spike_counts = poissrnd(rate * dt);

% Convert spike counts to a binary spike train
spike_train = double(spike_counts > 0);
spike_times = find(spike_train>0)/1000;

% figure; hist(spike_times,0:.2:75); xlim([0 74.5])

CC=[CC;[spike_times',repmat(nn,[length(spike_times),1])]];
end

% Make variables to emulate Mark's data formatting
R = 1000 * (8 : 10 : duration)';
R = [R, zeros(length(R), 1)];

Block = ones(length(R), 3);
Block(1:floor(length(R)/2),3) = 2;

idchan = zeros(nCount, 7);
idchan(:,1) = 1:max(unique(CC(:,2)));
idchan(:,4) = 1; % make all neurons "good"
idchan(:,end) = 2;
% Make a CSV file with spiking data
CSV = CC;

%
saveDir = fullfile(paths.dropPath, 'reach_task/data');
save(fullfile(saveDir, 'makeSpikes.mat'), 'Block', 'R', 'idchan', 'CSV')