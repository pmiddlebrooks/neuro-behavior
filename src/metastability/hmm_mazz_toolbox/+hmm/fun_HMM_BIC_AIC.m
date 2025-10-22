function [LLtottemp,hmm_all_data,hmm_all_bestfit,temp_SkipSpikesSess]=fun_HMM_BIC_AIC(Spikes,win_train,HmmParam)

% BINNING SPIKES
% create sequences from 5 s prior to cue to 5 s post delivery
[sequence, temp_SkipSpikesSess]=hmm.fun_HMM_binning(Spikes,HmmParam,win_train);
[ntrials, gnunits]=size(Spikes);

LLtottemp=zeros(numel(HmmParam.VarStates),1);
NStates=numel(HmmParam.VarStates);

% PGM edit to add single-sequence handling for BIC/AIC
% Single-sequence BIC/AIC: if only one sequence present, use cross-validation approach
if numel(sequence)==1
    % For single sequences, use cross-validation similar to XVAL but with BIC/AIC penalty
    % Expect HmmParam.singleSeqXval.K specifying number of folds (optional)
    Kss=5; % default number of folds
    if any(strcmp(fieldnames(HmmParam),'singleSeqXval')) && any(strcmp(fieldnames(HmmParam.singleSeqXval),'K'))
        Kss=HmmParam.singleSeqXval.K;
    end
    
    seqData=sequence.data; % emissions matrix: units x time or 1 x time (per Spikes2Seq)
    T=size(seqData,2); % total number of time bins
    edges=round(linspace(1,T+1,Kss+1));
    
    % accumulate per-state log-likelihood across folds
    LLacc=NaN(Kss,NStates);
    hmm_all_data=repmat(struct('tpm',[],'epm',[],'LLtrain',[]),1,NStates);
    
    for k=1:Kss
        holdIdx=edges(k):edges(k+1)-1;
        trainIdx=[1:edges(k)-1, edges(k+1):T];
        seqTrainSingle=struct('data', seqData(:,trainIdx));
        
        % train as single sequence (no repeated state-1 starts)
        fit_k=hmm.fun_HMM_training(seqTrainSingle,gnunits,HmmParam);
        
        % evaluate on hold-out segment for each state count
        for s=1:NStates
            esttr=fit_k(s).tpm;
            estemis=fit_k(s).epm;
            MinValue=1e-100;
            % MinValue=1e-10;
            esttr(esttr<MinValue)=MinValue;
            estemis(estemis<MinValue)=MinValue;
            holdSeq=seqData(:,holdIdx);
% fprintf('Emission matrix analysis:\n');
%     state_emissions = estemis(s, 1:end-1); % Exclude silence column
%     min_emission = min(state_emissions);
%     max_emission = max(state_emissions);
%     mean_emission = mean(state_emissions);
%     fprintf('  State %d: min=%.6f, max=%.6f, mean=%.6f\n', s, min_emission, max_emission, mean_emission);
            [~,logpseq]=hmmdecode(holdSeq,esttr,estemis);
            LLacc(k,s)=logpseq;
        end
        
        % store the last fold's fit (or could return best by avg LL)
        if k==Kss
            for s=1:NStates
                hmm_all_data(1,s)=fit_k(s);
            end
        end
    end
    
    % Calculate average log-likelihood across folds and add BIC/AIC penalty
    avgLL=mean(LLacc,1);
    LLtottemp(1:NStates,1)=avgLL'+HmmParam.NP(HmmParam.VarStates',gnunits,log(T));
    
    % Create bestfit structure (same as hmm_all_data for single sequence)
    hmm_all_bestfit=hmm_all_data;
    
else
    % Original multi-trial path
    T=0; % total number of datapoints
    for trial=1:ntrials
        T=T+size(sequence(trial).data,2);
    end
    %---------
    % TRAINING
    %---------
    hmm_all_data=hmm.fun_HMM_training(sequence,gnunits,HmmParam); 
    % rows=NumSteps runs with different initial conditions
    % cols=VarStates runs with different # of hidden states
    % for each choice of number of states in VarStates, select the best out of
    % NumSteps runs
    % -> This is needed to avoid getting stuck in local minima (HMM is non-convex)
    ind_step=zeros(1,numel(HmmParam.VarStates));
    hmm_all_bestfit=repmat(struct('tpm',[],'epm',[],'LLtrain',[]),1,numel(HmmParam.VarStates));
    for st_cnt=1:numel(HmmParam.VarStates)
        tempLL=cell2mat(arrayfun(@(x)x.LLtrain,hmm_all_data(:,st_cnt),'uniformoutput',false));
        [~,ind_step(st_cnt)]=min(tempLL); % find index of initial cond. with highest LL
        hmm_all_bestfit(1,st_cnt)=hmm_all_data(ind_step(st_cnt),st_cnt); % epm fit
    end
    LL=[hmm_all_bestfit(1,:).LLtrain]';
    LLtottemp(1:numel(HmmParam.VarStates),1)=LL+HmmParam.NP(HmmParam.VarStates',gnunits,log(T));
end
