duration = 2000; % Duration of simulation in seconds
dt = 0.001; % Time step in seconds
t_vec = 0:dt:duration-dt;
facInc=4; %how much larger is peak activity?
nCount=3

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

figure; hist(spike_times,0:.2:75); xlim([0 74.5])

CC=[CC;[spike_times',repmat(nn,[length(spike_times),1])]];
end