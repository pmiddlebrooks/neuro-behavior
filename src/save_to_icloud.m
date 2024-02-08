function save_to_icloud(fileName)
%%
if exist('E:/Projects', 'dir')
    cloudPath = 'E:/iCloudDrive/Figures';
elseif exist('/Users/paulmiddlebrooks', 'dir')
    cloudPath = '/Users/paulmiddlebrooks/Library/Mobile Documents/com~apple~CloudDocs/Figures';
end

saveas(fullfile(cloudPath, fileName))