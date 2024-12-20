%upload data

behave=readtable('/Users/eyttri/Downloads/neurobehavioral_data/ag25290_day2_iter13.tsv',"FileType","text",'Delimiter','\t');
behave=table2array(behave);
spikeid=readtable('/Users/eyttri/Downloads/neurobehavioral_data/spike_clusters.txt',"FileType","text",'Delimiter','\t');
spikeid=table2array(spikeid);
spiket=table2array(readtable('/Users/eyttri/Downloads/neurobehavioral_data/spike_times.txt',"FileType","text",'Delimiter','\t'));

behave=readtable('/Users/eyttri/Downloads/neurobehavioral_data/ag25290_day2_iter13.tsv',"FileType","text",'Delimiter','\t');
behave=table2array(behave);
spikeid=readtable('/Users/eyttri/Downloads/neurobehavioral_data/spike_clusters.txt',"FileType","text",'Delimiter','\t');
spikeid=table2array(spikeid);
spiket=table2array(readtable('/Users/eyttri/Downloads/neurobehavioral_data/spike_times.txt',"FileType","text",'Delimiter','\t'));



%reminder: behavioral samp rate=60hz
% bstart columns are: time start(s), ID, dur(s)
startInd=find(abs(diff(behave))>0)+1;
bstart=[[1/60,behave(1)];[startInd/60,behave(startInd)]];
bstart=[bstart,[diff(bstart(:,1));nan]];
bstart100=bstart(bstart(:,3)>.09,:);
cc=[spikeid,spiket/30000]; %dividing by 30k converts to seconds
%read cluster_info and determine what neuron ID's correspond to what
%categories
t = readtable('/Users/eyttri/Downloads/neurobehavioral_data/cluster_info.tsv',"FileType","text",'Delimiter', '\t');
goods=[];
goodn=[];
muas=[];
muan=[];
depths=table2array(t(:,7));
fr=table2array(t(:,8));
for nn=1:size(t,1)
    aa=t{nn,9};
    if any(aa{1}=='g')
        goods=[goods,nn];
        goodn=[goodn,table2array(t(nn,1))];
    elseif any(aa{1}=='m')
        muas=[muas,nn];
        muan=[muan,table2array(t(nn,1))];
    end
end
% end
neuron=goods;


%get activity
binSpike=[];
for n = goods %for each good neuron
    thisN=cc(cc(:,1)==n,2); %get spike times for a given neuron
    hist1=hist(thisN,0:.05:30*60); %generate histogram with step size (50ms) %20*60 = 20 min
    hist1=hist1(1:end-1); % throw out the huge last value because I'm cutting off at 20min
    % thisH=hist1(1:end-1);
    thisH=hist1 + [hist1(2:end), 0]+ [hist1(3 :end), 0, 0]+[hist1(4:end),0 0 0]; % make bin size 4x shift size
    binSpike=[binSpike;thisH];
end


%visualize bands, activity
figure; imagesc(zscore(binSpike')');
caxis([-1 3])
%time: 1 sample = 50ms


%tsne + plot
bb=behave([1:3:(20*60*60)]+(3*60*60*60));
bb(bb<1)=30;
Y2=tsne(zscore(binSpike(sum(abs(zscore(binSpike(:,220:400)')))>100,:)'),'Exaggeration',90);
%I picked a bit of time around one of the bands to limit to active
neurons
ccol='krygmcyrbgmckrbgmckrygmcyrbgmckrbgmckrygmcyrbgmckrbgmc';
figure; hold on;
for i=1:length(Y2)
    plot3(Y2(i,1),Y2(i,2),bb(i),'.','Color',ccol(bb(i)));%[bb(i)/28 .6 1-bb(i)/28])
end