%% Creates a test dataset (small) from real data for reach

reachDataFile = fullfile(paths.dropPath, 'reach_data/Y4_06-Oct-2023 14_14_53_NeuroBeh.mat');

load(reachDataFile);

opts = neuro_behavior_options;
opts.minFiringRate = .1;
opts.frameSize = .001;

% % Find the first reach after block switch
% block2FirstIdx = find(ismember(Block(:,3), 3:4), 1);
% block2FirstSec = R(block2FirstIdx)/1000;
% 
% beforeAfter = 120 * 1000;
% beforeAfterReach = 5;
% 
% opts.collectStart = block2FirstSec - beforeAfter;
% opts.collectFor = beforeAfter;

Block = Block(1:15,:);
Block([9 10 12 15],3) = 4;
Block([11 13 14],3) = 3;

R = R(1:15,:);

savePath = fullfile(paths.dropPath, 'reach_data', 'reach_test.mat');
save(savePath, 'R', 'Block', 'README', 'CSV', 'idchan', '-mat');
