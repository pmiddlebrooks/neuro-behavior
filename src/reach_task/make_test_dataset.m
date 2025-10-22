%% Creates a test dataset (small) from real data for reach

reachDataFile = fullfile(paths.reachDataPath, 'Y4_06-Oct-2023 14_14_53_NeuroBeh.mat');
% reachDataFile = fullfile(paths.reachDataPath, 'AB2_01-May-2023 15_34_59_NeuroBeh.mat');

load(reachDataFile);

% % Find the first reach after block switch
% block2FirstIdx = find(ismember(Block(:,3), 3:4), 1);
% block2FirstSec = R(block2FirstIdx)/1000;
% 
% beforeAfter = 120 * 1000;
% beforeAfterReach = 5;
% 
% opts.collectStart = block2FirstSec - beforeAfter;
% opts.collectFor = beforeAfter;

nTrial = 50;
Block = Block(1:nTrial,:);
Block(26:end,3) = 4;
Block(40:50,3) = 3;

R = R(1:nTrial,:);

savePath = fullfile(paths.reachDataPath, 'reach_test.mat');
save(savePath, 'R', 'Block', 'README', 'CSV', 'idchan', '-mat');
