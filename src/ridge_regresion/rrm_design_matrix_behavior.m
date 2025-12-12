function [designMatrix, regressorLabels, regressorBhv] = rrm_design_matrix_behavior(data, opts)


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  Make labels for the time regressors

regWindow = int8(-opts.mPreTime/opts.frameSize : opts.mPostTime/opts.frameSize - 1);

% Initialize a cell array to store the combinations
regressorLabels = cell(numel(regWindow), numel(opts.behaviors));
regressorBhv = nan(numel(regWindow), numel(opts.behaviors));

% Loop through the vector and cell array to create combinations
for i = 1:numel(regWindow)
    for j = 1:numel(opts.behaviors)
        regressorLabels{i, j} = [num2str(regWindow(i)), ' ', opts.behaviors{j}];
        regressorBhv(i, j) = opts.bhvCodes(j);
    end
end

% Display the cell array of combinations
% disp(regressorLabels);
regressorLabels = reshape(regressorLabels, [1, numel(regressorLabels)]);
regressorBhv = reshape(regressorBhv, [1, numel(regressorBhv)]);



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  Design matrix for behavior time regressors
[bhvIdx, ia, ic] = unique(data.ID);



durFrames = floor(sum(data.Dur) / opts.frameSize);

% pre-allocate the design matrix
framesPerBout = int8((opts.mPreTime + opts.mPostTime) / opts.frameSize);
timeRegressor = eye(framesPerBout);
bhvDesign = zeros(durFrames, framesPerBout * length(opts.behaviors));

for i = 1 : length(data.ID)


    % Can we use this behavior?
    if data.Valid(i) && data.ID(i) ~= -1 && data.StartFrame(i) > 1

        % which behavior index is this?
        iBhvIdx = ic(i);

        xCoord = (iBhvIdx * framesPerBout + 1 : (iBhvIdx + 1) * framesPerBout) - framesPerBout;

        % data.StartFrame(i)

        % Normal (full time) frames get pasted in as identity matrix
        if data.StartFrame(i) >= opts.mPreTime/opts.frameSize + 1 && data.StartFrame(i) <= durFrames - opts.mPostTime/opts.frameSize
            yCoord = data.StartFrame(i) - opts.mPreTime/opts.frameSize : data.StartFrame(i) + opts.mPostTime/opts.frameSize -1;
            bhvDesign(yCoord, xCoord) = timeRegressor;
            % Take care of the first few frames
        elseif data.StartFrame(i) <= opts.mPreTime/opts.frameSize - 1  %
            insertReg = timeRegressor(opts.mPreTime/opts.frameSize - data.StartFrame(i) + 2 : end, :);
            yCoord = 1 : size(insertReg, 1);
            % yCoord = opts.mPreTime/opts.frameSize - data.StartFrame(i) + 2 : opts.mPreTime/opts.frameSize - data.StartFrame(i) + 2 + size(timeRegressor, 1);
            bhvDesign(yCoord, xCoord) = insertReg;
            % Take care of the last few frames
        elseif data.StartFrame(i) > durFrames - opts.mPostTime/opts.frameSize
            insertReg = timeRegressor(1 : opts.mPreTime/opts.frameSize + durFrames - data.StartFrame(i) + 1, :);
            yCoord = data.StartFrame(i) - opts.mPreTime/opts.frameSize : durFrames;
            bhvDesign(yCoord, xCoord) = insertReg;

        end

        % imagesc(bhvDesign)
        % pause
    end
end
% imagesc(bhvDesign)

designMatrix = bhvDesign;
