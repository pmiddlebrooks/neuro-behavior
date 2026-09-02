function allGood = cluster_quality_mask(ci, opts)
% CLUSTER_QUALITY_MASK - Keep units labeled good, mua, or real
%
% Variables:
%   ci   - Cluster table from load_session_cluster_info (cluster_info.tsv
%          and/or cluster_rf.tsv)
%   opts - Options; opts.useMulti (default true) includes mua;
%          opts.sessionName used only in error messages
%
% Goal:
%   Return a logical mask for units to keep. In group and/or rf_label,
%   accept any of 'good', 'mua', or 'real' (case-insensitive). Empty Phy
%   group columns (header-only cluster_group.tsv) contribute nothing.
%
% This is the single place that defines the good / mua / real unit mask.

    if nargin < 2
        opts = struct();
    end

    nUnits = height(ci);
    allGood = false(nUnits, 1);
    usedSources = {};

    useMulti = true;
    if isfield(opts, 'useMulti') && ~isempty(opts.useMulti)
        useMulti = logical(opts.useMulti);
    end

    if useMulti
        acceptedLabels = ["good", "mua", "real"];
        labelNote = 'good, mua, or real';
    else
        acceptedLabels = ["good", "real"];
        labelNote = 'good or real';
    end

    varNames = ci.Properties.VariableNames;
    qualityColumns = {'group', 'rf_label'};
    foundQualityColumn = false;

    for iCol = 1:numel(qualityColumns)
        colName = qualityColumns{iCol};
        if ~ismember(colName, varNames)
            continue
        end
        foundQualityColumn = true;
        colLabel = lower(strtrim(string(ci.(colName))));
        colMatch = ismember(colLabel, acceptedLabels);
        if any(colMatch)
            allGood = allGood | colMatch;
            usedSources{end+1} = sprintf('%s (%s)', colName, labelNote); %#ok<AGROW>
        end
    end

    if ~foundQualityColumn
        sessionLabel = 'session';
        if isfield(opts, 'sessionName') && ~isempty(opts.sessionName)
            sessionLabel = opts.sessionName;
        end
        error(['No group or rf_label quality column in %s. ', ...
            'Need cluster_info.tsv and/or cluster_rf.tsv.'], sessionLabel);
    end

    fprintf('Unit quality mask (%s): keeping %d / %d units\n', ...
        strjoin(usedSources, ' | '), sum(allGood), nUnits);
end
