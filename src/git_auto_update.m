function git_auto_update(location)

switch location
    case 'home'
        pathsToUpdate = {'/Users/paulmiddlebrooks/Projects/neuro-behavior', ...
            '/Users/paulmiddlebrooks/Projects/dynamics', ...
            '/Users/paulmiddlebrooks/Projects/figure-tools', ...
            '/Users/paulmiddlebrooks/Projects/ridgeRegress'};
    case 'lab'
        pathsToUpdate = {'E:/Projects/neuro-behavior/', ...
            'E:/Projects/dynamics/', ...
            'E:/Projects/figure-tools/', ...
            'E:/Projects/ridgeRegress/'};
end

for i = 1 : length(pathsToUpdate)
    % Navigate to the specified folder
    cd(pathsToUpdate{i});

    % Git add
    system('git add .');

    % Git commit
    system('git commit -m "auto-update"');

    % Git push
    system('git push origin main');
    pause(5)
end
cd 'E:/Projects/neuro-behavior/src'
end