function allGood = cluster_quality_mask(ci, opts)
% CLUSTER_QUALITY_MASK - Units to keep from cluster_info quality labels
%
% Variables:
%   ci   - cluster_info table
%   opts - Options; opts.useMulti (default true) includes mua with good;
%          opts.sessionName used only in error messages
%
% Goal:
%   If ci.group exists and has non-empty labels, keep 'good' (and 'mua' when
%   opts.useMulti is true). Otherwise keep units with ci.rf_label == 'real'.
%   Empty Phy group columns (header-only cluster_group.tsv) fall back to rf_label.

    useGroup = ismember('group', ci.Properties.VariableNames) && ...
        has_nonempty_quality_labels(ci.group);

    if useGroup
        useMulti = true;
        if isfield(opts, 'useMulti') && ~isempty(opts.useMulti)
            useMulti = logical(opts.useMulti);
        end
        groupLabel = strtrim(string(ci.group));
        if useMulti
            allGood = groupLabel == "good" | groupLabel == "mua";
            fprintf('Using cluster_info.group (good + mua) for unit quality\n');
        else
            allGood = groupLabel == "good";
            fprintf('Using cluster_info.group (good only) for unit quality\n');
        end
    else
        if ~ismember('rf_label', ci.Properties.VariableNames)
            sessionLabel = 'session';
            if isfield(opts, 'sessionName') && ~isempty(opts.sessionName)
                sessionLabel = opts.sessionName;
            end
            error(['cluster_info.tsv has no usable group column and no rf_label ', ...
                'column in %s'], sessionLabel);
        end
        rfLabel = strtrim(string(ci.rf_label));
        allGood = rfLabel == "real";
        fprintf('Using cluster_info.rf_label (real) for unit quality\n');
    end
end

function tf = has_nonempty_quality_labels(vals)
% HAS_NONEMPTY_QUALITY_LABELS - True if any quality label is non-blank
%
% Variables:
%   vals - cluster_info quality column (group or similar)
%
% Goal:
%   Treat a column of empty strings / missing / NaN as unused so loading can
%   fall back to rf_label.

    if isempty(vals)
        tf = false;
        return
    end

    if isnumeric(vals)
        tf = any(~isnan(vals(:)));
        return
    end

    labelStr = strtrim(string(vals));
    tf = any(strlength(labelStr) > 0 & ~ismissing(labelStr));
end
