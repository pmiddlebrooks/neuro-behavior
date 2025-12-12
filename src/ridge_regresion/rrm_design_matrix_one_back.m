function [oneBackDesign, oneBackLabels] = rrm_design_matrix_one_back(data, opts)
% Make  design matrix for 1-back behavior (and labels for the 1-back regressors)


durFrames = floor(sum(data.Dur) / opts.frameSize);
[bhvIdx, ~, ~] = unique(data.ID);
data.StartFrame = 1 + floor(data.StartTime ./ opts.frameSize); % adjust start times to frame time based on frameSize


% pre-allocate the design matrix
% oneBackDesign = zeros(durFrames, opts.framesPerTrial * length(opts.behaviors));
oneBackDesign = [];
oneBackLabels = {};
% column = 1;

for iCurr = 1 : length(opts.behaviors)
    for jPrev = 1 : length(opts.behaviors)
        if ~strcmp(opts.behaviors{iCurr}, opts.behaviors{jPrev}) % If not regressing a behavior against itself


            % Find in data the index the current behavior starts, when it is
            % preceded by the oneBack behavior, given that the
            % current behavior is valid

            % Use this version to ensure only the current behavior is
            % valid, ignoring whether the previous behavior is valid
            oneBackIdx = data.ID == bhvIdx(iCurr) & data.prevBhvID == bhvIdx(jPrev) & data.Valid;

            % Use this version to ensure both the current behavior and
            % previous behaviors are valid
            % oneBackIdx = data.ID == bhvIdx(iCurr) & data.prevBhvID == bhvIdx(jPrev) & opts.validBhv(:, iCurr) & [0; opts.validBhv(:, jPrev)(1:end-1)];
            
            if sum(oneBackIdx) >= opts.minOneBackNum

                % Find the frame the current behavior starts, when it is
                % preceded by the oneBack behavior
                currBhvFrame = data.StartFrame(oneBackIdx);
                insertVector = zeros(durFrames, 1);
                insertVector(currBhvFrame) = 1;

                % oneBackDesign(prvBhvFrame, column) = 1;
                oneBackDesign = [oneBackDesign, insertVector];

                % label it as <current behavior> after <previous behavior>
                iLabel = [opts.behaviors{jPrev}, ' then ', opts.behaviors{iCurr}];
                oneBackLabels = [oneBackLabels, iLabel];

                % fprintf('%s: \t %d\n', iLabel, sum(oneBackIdx))
            end
            % column = column + 1;
        end
    end
end


% Only keep one-back regressors with the most instances
nOneBack = sum(oneBackDesign, 1);
[b,idx] = sort(nOneBack, 'descend');

% warning('Something is wrong with these numbers: see rrm_design_matrix_one_back.m line52, etc.')
for k = 1 : opts.nOneBackKeep
    fprintf('%d trials of %s\n', b(k), oneBackLabels{idx(k)})
end

oneBackDesign = oneBackDesign(:, idx(1:opts.nOneBackKeep));
oneBackLabels = oneBackLabels(idx(1:opts.nOneBackKeep));

