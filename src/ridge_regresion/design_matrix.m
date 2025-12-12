function [fullMat, eventIdx] = design_matrix(events, eventType, opts)
% function to generate design matrix from a column matrix with binary
% events. eventType defines the type of design matrix that is generated.
% (1 = fullTrial, 2 = post-event, 3 = peri-event)
% 
% Edited from makeDesignMatrix.m in RidgeRegression

nFrame = opts.nFramePerTrial;
fullMat = cell(1,length(eventType));
eventIdx = cell(1,length(eventType));
events = reshape(events, nFrame, [], length(eventType)); %reshape to trials
trialCnt = size(events,2); %nr of trials

for iRegs = 1 : length(eventType)
    
    % determine index for current event type
    if eventType(iRegs) == 1
        kernelIdx = 0 : nFrame-1; %index for whole trial
    elseif eventType(iRegs) == 2
        kernelIdx = 0 : opts.sPostTime/opts.frameSize; %index for design matrix to cover post event activity
    elseif eventType(iRegs) == 3
        kernelIdx = [-(opts.mPreTime/opts.frameSize: -1 : 1) 0 (1:opts.mPostTimeopts.frameSize)]; %index for design matrix to cover pre- and post event activity
    else
        error('Unknown event type. Must be a value between 1 and 3.')
    end
    
    % run over trials
    dMat = cell(1,trialCnt);    
    for iTrials = 1 : trialCnt
        
        % Get the zero lag regressor.
        trace = logical(events(:,iTrials,iRegs));
        
        % create full design matrix
        cIdx = bsxfun(@plus,find(trace),kernelIdx);
        cIdx(cIdx < 1) = 0;
        cIdx(cIdx > nFrame) = nFrame; 
        cIdx = bsxfun(@plus,cIdx,(0:nFrame:nFrame*length(kernelIdx)-1));  % gets the indices of each successive frame's regressor (creates an eye(size(kernel)))
        cIdx(cIdx < 1) = nFrame;
        cIdx(cIdx > (nFrame * length(kernelIdx))) = nFrame * length(kernelIdx);
        
        dMat{iTrials} = false(nFrame, length(kernelIdx));
        dMat{iTrials}(cIdx(:)) = true;
        dMat{iTrials}(end,:) = false; %don't use last timepoint of design matrix to avoid confusion with indexing.
        dMat{iTrials}(end,2:end) = dMat{iTrials}(end-1,1:end-1); %replace with shifted version of previous timepoint
        
    end
    fullMat{iRegs} = cat(1, dMat{:}); %combine all trials
    cIdx = sum(fullMat{iRegs},1) > 0; %don't use empty regressors
    fullMat{iRegs} = fullMat{iRegs}(:,cIdx);
    eventIdx{iRegs} = repmat(iRegs,sum(cIdx),1); %keep index on how many regressor were created
end

fullMat = cat(2,fullMat{:}); %combine all regressors into larger matrix
eventIdx = cat(1,eventIdx{:}); %combine index so we need what is what

end