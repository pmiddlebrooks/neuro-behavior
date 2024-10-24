function git_auto_push

if exist('/Users/paulmiddlebrooks/Projects/', 'dir')
        pathsToUpdate = {'/Users/paulmiddlebrooks/Projects/neuro-behavior', ...
            '/Users/paulmiddlebrooks/Projects/dynamics', ...
            '/Users/paulmiddlebrooks/Projects/figure_tools', ...
            '/Users/paulmiddlebrooks/Projects/ridgeRegress'};
elseif exist('E:/Projects', 'dir')
        pathsToUpdate = {'E:/Projects/neuro-behavior/', ...
            'E:/Projects/dynamics/', ...
            'E:/Projects/figure_tools/', ...
            'E:/Projects/ridgeRegress/'};
end

for i = 1 : length(pathsToUpdate)
    % Navigate to the specified folder
    cd(pathsToUpdate{i});
fprintf('Updating %s\n', pathsToUpdate{i})
    % Git add
    system('git add .');

    % Git commit
    system('git commit -m "auto-update"');

    % Git push
    system('git push origin main');
    pause(5)
end
cd(pathsToUpdate{1})
cd src/
end