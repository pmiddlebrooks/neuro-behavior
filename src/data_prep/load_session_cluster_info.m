function ci = load_session_cluster_info(sessionFolder, sessionName)
% LOAD_SESSION_CLUSTER_INFO - Load cluster metadata and assign brain areas
%
% Variables:
%   sessionFolder - Path to the session directory
%   sessionName   - Session folder name (for logging; optional)
%
% Goal:
%   Prefer cluster_rf.tsv when it exists (quality labels are the source of
%   truth). Fall back to cluster_info.tsv if cluster_rf.tsv is absent.
%   Depth and other non-quality columns are copied from cluster_info when
%   cluster_rf lacks them. Orient depth so 0 is superficial (M23) and 3840
%   is deep (VS), then assign ci.area from get_brain_area_depth_ranges.
%   Units with depth < 0 are labeled noise in group and/or rf_label so they
%   are not accepted.

    if nargin < 2
        sessionName = '';
    end

    clusterInfoPath = fullfile(sessionFolder, 'cluster_info.tsv');
    clusterRfPath = fullfile(sessionFolder, 'cluster_rf.tsv');

    if isfile(clusterRfPath)
        ci = read_tsv_table(clusterRfPath);
        clusterFileUsed = 'cluster_rf.tsv';
        if isfile(clusterInfoPath)
            infoTable = read_tsv_table(clusterInfoPath);
            ci = merge_cluster_info_metadata(ci, infoTable);
            fprintf('Merged depth/metadata from cluster_info.tsv into cluster_rf.tsv\n');
        end
    elseif isfile(clusterInfoPath)
        ci = read_tsv_table(clusterInfoPath);
        clusterFileUsed = 'cluster_info.tsv';
    else
        error('Neither cluster_rf.tsv nor cluster_info.tsv found in %s', sessionFolder);
    end

    fprintf('Loaded cluster metadata from %s\n', clusterFileUsed);

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

function ci = merge_cluster_info_metadata(ci, infoTable)
% MERGE_CLUSTER_INFO_METADATA - Copy non-quality columns from cluster_info
%
% Variables:
%   ci        - Primary table (cluster_rf.tsv)
%   infoTable - cluster_info.tsv table, aligned by cluster id
%
% Goal:
%   Fill columns that cluster_rf does not have (especially depth) from
%   cluster_info. Quality labels (group, rf_label) stay on ci so
%   cluster_rf remains the quality source of truth.

    skipFields = {'group', 'rf_label'};
    ciIds = cluster_id_column(ci);
    infoIds = cluster_id_column(infoTable);
    [tf, loc] = ismember(ciIds, infoIds);
    infoFields = infoTable.Properties.VariableNames;
    ciFields = ci.Properties.VariableNames;

    for iField = 1:numel(infoFields)
        fieldName = infoFields{iField};
        if any(strcmp(fieldName, skipFields)) || ismember(fieldName, ciFields)
            continue
        end
        infoVals = infoTable.(fieldName);
        mergedVals = empty_like_column(infoVals, height(ci));
        if any(tf)
            mergedVals(tf) = infoVals(loc(tf));
        end
        ci.(fieldName) = mergedVals;
    end
end

function col = empty_like_column(templateCol, nRows)
% EMPTY_LIKE_COLUMN - nRows placeholder matching templateCol's type

    if iscell(templateCol)
        col = repmat({''}, nRows, 1);
    elseif isstring(templateCol)
        col = strings(nRows, 1);
    elseif iscategorical(templateCol)
        col = categorical(repmat({''}, nRows, 1));
    elseif isnumeric(templateCol)
        col = nan(nRows, 1);
    elseif islogical(templateCol)
        col = false(nRows, 1);
    else
        col = cell(nRows, 1);
    end
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
