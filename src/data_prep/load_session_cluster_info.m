function ci = load_session_cluster_info(sessionFolder, sessionName)
% LOAD_SESSION_CLUSTER_INFO - Load cluster metadata and assign brain areas
%
% Variables:
%   sessionFolder - Path to the session directory
%   sessionName   - Session folder name (for logging; optional)
%
% Goal:
%   Load cluster_info.tsv when present; otherwise load cluster_rf.tsv.
%   If cluster_info is missing rf_label, merge it from cluster_rf.tsv.
%   Orient depth so 0 is superficial (M23) and 3840 is deep (VS), then
%   assign ci.area from get_brain_area_depth_ranges. Units with depth < 0
%   are labeled noise in group and/or rf_label so they are not accepted.

    if nargin < 2
        sessionName = '';
    end

    clusterInfoPath = fullfile(sessionFolder, 'cluster_info.tsv');
    clusterRfPath = fullfile(sessionFolder, 'cluster_rf.tsv');

    if isfile(clusterInfoPath)
        ci = readtable(clusterInfoPath, 'FileType', 'text', 'Delimiter', '\t');
        clusterFileUsed = 'cluster_info.tsv';
    elseif isfile(clusterRfPath)
        ci = readtable(clusterRfPath, 'FileType', 'text', 'Delimiter', '\t');
        clusterFileUsed = 'cluster_rf.tsv';
    else
        error('Neither cluster_info.tsv nor cluster_rf.tsv found in %s', sessionFolder);
    end

    fprintf('Loaded cluster metadata from %s\n', clusterFileUsed);

    if ~ismember('rf_label', ci.Properties.VariableNames) && isfile(clusterRfPath)
        rfTable = readtable(clusterRfPath, 'FileType', 'text', 'Delimiter', '\t');
        ci = merge_rf_label(ci, rfTable);
        fprintf('Merged rf_label from cluster_rf.tsv into %s\n', clusterFileUsed);
    end

    if ismember('depth', ci.Properties.VariableNames)
        ci = sortrows(ci, 'depth');
        ci.depth = 3840 - ci.depth;
        ci = flipud(ci);

        belowSurface = ci.depth < 0;
        if any(belowSurface)
            ci = assign_noise_for_negative_depth(ci, belowSurface);
        end

        [m23, m56, cc, ds, vs, depthSource] = get_brain_area_depth_ranges(sessionFolder);
        if strcmp(depthSource, 'session')
            if ~isempty(sessionName)
                fprintf('Using brain_area_depths.mat for %s\n', sessionName);
            else
                fprintf('Using brain_area_depths.mat in %s\n', sessionFolder);
            end
        end

        area = repmat({''}, height(ci), 1);
        area(ci.depth >= m23(1) & ci.depth <= m23(2)) = {'M23'};
        area(ci.depth >= m56(1) & ci.depth <= m56(2)) = {'M56'};
        area(ci.depth >= cc(1) & ci.depth <= cc(2)) = {'CC'};
        area(ci.depth >= ds(1) & ci.depth <= ds(2)) = {'DS'};
        area(ci.depth >= vs(1) & ci.depth <= vs(2)) = {'VS'};
        ci.area = area;
    end
end

function ci = assign_noise_for_negative_depth(ci, belowSurface)
% ASSIGN_NOISE_FOR_NEGATIVE_DEPTH - Mark above-surface units as noise
%
% Variables:
%   ci           - Cluster table with group and/or rf_label
%   belowSurface - Logical mask of units with oriented depth < 0
%
% Goal:
%   Set group and/or rf_label to 'noise' for those units, matching whichever
%   quality column(s) are present so cluster_quality_mask will drop them.

    qualityColumns = {'group', 'rf_label'};
    varNames = ci.Properties.VariableNames;
    updatedCols = {};
    for iCol = 1:numel(qualityColumns)
        colName = qualityColumns{iCol};
        if ~ismember(colName, varNames)
            continue
        end
        colVals = ci.(colName);
        if iscell(colVals)
            colVals(belowSurface) = {'noise'};
        elseif isstring(colVals)
            colVals(belowSurface) = "noise";
        elseif iscategorical(colVals)
            if ~ismember('noise', categories(colVals))
                colVals = addcats(colVals, {'noise'});
            end
            colVals(belowSurface) = 'noise';
        else
            colVals = cellstr(string(colVals));
            colVals(belowSurface) = {'noise'};
        end
        ci.(colName) = colVals;
        updatedCols{end+1} = colName; %#ok<AGROW>
    end

    if isempty(updatedCols)
        warning('load_session_cluster_info:NoQualityColumn', ...
            'Could not set depth < 0 units to noise (no group or rf_label column).');
        return
    end
    fprintf('Assigned noise to %d units with depth < 0 (%s)\n', ...
        sum(belowSurface), strjoin(updatedCols, ', '));
end

function ci = merge_rf_label(ci, rfTable)
% MERGE_RF_LABEL - Add rf_label from cluster_rf.tsv onto the cluster table
%
% Variables:
%   ci      - cluster_info (or similar) table
%   rfTable - table with cluster_id/id and rf_label
%
% Goal:
%   Align rf_label rows to ci by cluster id.

    if ~ismember('rf_label', rfTable.Properties.VariableNames)
        error('cluster_rf.tsv must contain an rf_label column');
    end

    ciIds = cluster_id_column(ci);
    rfIds = cluster_id_column(rfTable);
    [tf, loc] = ismember(ciIds, rfIds);
    rfLabel = strings(height(ci), 1);
    rfVals = strtrim(string(rfTable.rf_label));
    rfLabel(tf) = rfVals(loc(tf));
    ci.rf_label = cellstr(rfLabel);
end

function clusterIds = cluster_id_column(clusterTable)
% CLUSTER_ID_COLUMN - cluster_id or id vector from a cluster table

    varNames = clusterTable.Properties.VariableNames;
    if ismember('cluster_id', varNames)
        clusterIds = clusterTable.cluster_id;
    elseif ismember('id', varNames)
        clusterIds = clusterTable.id;
    else
        error('Cluster table must contain cluster_id or id');
    end
end
