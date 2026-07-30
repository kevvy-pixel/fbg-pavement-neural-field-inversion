function workspaceDir = fbg_workspace(requested)
%FBG_WORKSPACE Resolve a writable workspace without assuming package write access.
if nargin < 1
    requested = "";
end

candidates = strings(0, 1);
if strlength(string(requested)) > 0
    candidates(end + 1) = string(requested); %#ok<AGROW>
end
if ispc
    userProfile = string(getenv("USERPROFILE"));
    if strlength(userProfile) > 0
        candidates(end + 1) = fullfile(userProfile, "Documents", ...
            "MATLAB", "FBG_NeuralField_Workspace"); %#ok<AGROW>
    end
else
    userHome = string(getenv("HOME"));
    if strlength(userHome) > 0
        candidates(end + 1) = fullfile(userHome, "MATLAB", ...
            "FBG_NeuralField_Workspace"); %#ok<AGROW>
    end
end
candidates(end + 1) = fullfile(tempdir, "FBG_NeuralField_Workspace");

for i = 1:numel(candidates)
    candidate = char(candidates(i));
    fid = -1;
    try
        if ~isfolder(candidate)
            mkdir(candidate);
        end
        probe = fullfile(candidate, "fbg_write_probe.tmp");
        fid = fopen(probe, "w");
        if fid > 0
            fprintf(fid, "ok");
            fclose(fid);
            delete(probe);
            workspaceDir = candidate;
            return;
        end
    catch
        if fid > 0
            fclose(fid);
        end
    end
end
error("FBG:Workspace", "No writable workspace directory could be resolved.");
end
