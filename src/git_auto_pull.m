function git_auto_pull

%%
if exist('/Users/paulmiddlebrooks/Projects/', 'dir')
        pathsToUpdate = {'/Users/paulmiddlebrooks/Projects/neuro-behavior', ...
            '/Users/paulmiddlebrooks/Projects/dynamics', ...
            '/Users/paulmiddlebrooks/Projects/figure_tools', ...
            '/Users/paulmiddlebrooks/Projects/ymaze/src', ...
            '/Users/paulmiddlebrooks/Projects/ridgeRegress'};
elseif exist('E:/Projects', 'dir')
        pathsToUpdate = {'E:/Projects/neuro-behavior/', ...
            'E:/Projects/dynamics/', ...
            'E:/Projects/figure_tools/', ...
            'E:/Projects/ymaze/', ...
            'E:/Projects/ridgeRegress/'};
end

for i = 1 : length(pathsToUpdate)
    % Navigate to the specified folder
    cd(pathsToUpdate{i});
    disp(pathsToUpdate{i})
    % Git add
    system('git pull');

    pause(3)
end
cd(pathsToUpdate{1})
cd src/
end