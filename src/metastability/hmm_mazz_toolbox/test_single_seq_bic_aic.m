%% Test script for single-sequence BIC/AIC implementation
% This script tests the new single-sequence handling in fun_HMM_BIC_AIC.m


% Create synthetic single-sequence data
nUnits = 5;
nTimeBins = 1000;
binSize = 0.01; % 10ms bins

% Generate synthetic spike data (single trial)
spikes = struct();
for u = 1:nUnits
    % Generate random spike times for this unit
    spikeTimes = sort(rand(1, randi([50, 200]))) * 10; % 0-10 seconds
    spikes(u).spk = spikeTimes;
end

% Create window for training (entire duration)
win_train = [0, 10]; % 10 second window

% Set up HMM parameters
HmmParam = struct();
HmmParam.BinSize = binSize;
HmmParam.MinDur = 0.05;
HmmParam.MinP = 0.8;
HmmParam.NumSteps = 5; % reduced for testing
HmmParam.NumRuns = 10; % reduced for testing
HmmParam.VarStates = [2, 3, 4, 5]; % test different state numbers

% Test BIC penalty function
HmmParam.NP = @(NumStates,gnunits,logT)(NumStates.*(NumStates-1)+NumStates.*gnunits+NumStates-1)*logT;

% Add single-sequence cross-validation parameters
HmmParam.singleSeqXval = struct();
HmmParam.singleSeqXval.K = 3; % 3-fold cross-validation

fprintf('Testing single-sequence BIC/AIC implementation...\n');
fprintf('Data: %d units, %d time bins (%.1f seconds)\n', nUnits, nTimeBins, nTimeBins*binSize);
fprintf('Testing %d state numbers: %s\n', length(HmmParam.VarStates), mat2str(HmmParam.VarStates));

try
    % Test the BIC/AIC function
    tic;
    [LLtottemp, hmm_all_data, hmm_all_bestfit, temp_SkipSpikesSess] = hmm.fun_HMM_BIC_AIC(spikes, win_train, HmmParam);
    elapsed_time = toc;
    
    fprintf('\nSUCCESS! Function completed in %.2f seconds\n', elapsed_time);
    fprintf('Results:\n');
    fprintf('  Number of states tested: %d\n', length(HmmParam.VarStates));
    fprintf('  LLtottemp size: %s\n', mat2str(size(LLtottemp)));
    fprintf('  hmm_all_data size: %s\n', mat2str(size(hmm_all_data)));
    fprintf('  hmm_all_bestfit size: %s\n', mat2str(size(hmm_all_bestfit)));
    
    % Display results for each state number
    fprintf('\nBIC/AIC Results:\n');
    fprintf('States\tBIC/AIC Score\n');
    fprintf('------\t-------------\n');
    for i = 1:length(HmmParam.VarStates)
        fprintf('%d\t%.3f\n', HmmParam.VarStates(i), LLtottemp(i));
    end
    
    % Find best number of states
    [~, bestIdx] = min(LLtottemp);
    bestStates = HmmParam.VarStates(bestIdx);
    fprintf('\nBest number of states: %d (BIC/AIC = %.3f)\n', bestStates, LLtottemp(bestIdx));
    
catch ME
    fprintf('ERROR: %s\n', ME.message);
    fprintf('Stack trace:\n');
    for i = 1:length(ME.stack)
        fprintf('  %s (line %d)\n', ME.stack(i).name, ME.stack(i).line);
    end
end

fprintf('\nTest completed.\n');
