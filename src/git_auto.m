function git_auto

%%
% Prompt user to choose operation
fprintf('Choose operation:\n');
fprintf('1. push - add, commit, and push changes\n');
fprintf('2. pull - pull latest changes\n');
operation = input('Enter "push" or "pull": ', 's');

% Validate input
if ~ismember(lower(operation), {'push', 'pull'})
    fprintf('Invalid input. Please enter "push" or "pull".\n');
    return;
end

% Set paths based on system
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

% Execute chosen operation
for i = 1 : length(pathsToUpdate)
    % Navigate to the specified folder
    cd(pathsToUpdate{i});
    fprintf('Processing %s\n', pathsToUpdate{i});
    
    if strcmpi(operation, 'push')
        % Git add
        system('git add .');
        
        % Git commit
        system('git commit -m "auto-update"');
        
        % Git push
        system('git push origin main');
        pause(5);
    else % pull
        % Git pull
        system('git pull');
        pause(3);
    end
end

% Return to first project's src directory
cd(pathsToUpdate{1});
cd src/;

fprintf('Git %s operation completed for all repositories.\n', operation);
end 