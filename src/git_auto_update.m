function git_auto_update(location)

switch location
    case 'home'
        pathsToUpdate = {''};
    case 'lab'
        pathsToUpdate = {'E:/Projects/neuro-behavior/', ...
            'E:/Projects/dynamics/', ...
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
end