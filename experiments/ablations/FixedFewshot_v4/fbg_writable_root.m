function workRoot = fbg_writable_root(preferredBase)
%FBG_WRITABLE_ROOT Find a writable location for data and experiment outputs.

arguments
    preferredBase (1,1) string = ""
end

candidates = strings(0,1);
if strlength(preferredBase) > 0
    candidates(end+1,1) = preferredBase;
end
candidates(end+1,1) = string(pwd);

matlabUserPath = string(userpath);
if strlength(matlabUserPath) > 0
    userParts = split(matlabUserPath,pathsep);
    userParts = userParts(strlength(userParts) > 0);
    candidates = [candidates;userParts(:)];
end
candidates(end+1,1) = string(tempdir);
candidates = unique(candidates,'stable');

failureMessages = strings(0,1);
for i = 1:numel(candidates)
    base = candidates(i);
    candidate = fullfile(base,"FBG_NeuralField_Workspace");
    fid = -1;
    try
        if ~exist(candidate,'dir')
            mkdir(candidate);
        end
        testFile = string(tempname(candidate)) + ".tmp";
        fid = fopen(testFile,'w');
        if fid < 0
            failureMessages(end+1,1) = candidate + ": fopen denied"; %#ok<AGROW>
            continue;
        end
        fprintf(fid,'write test');
        fclose(fid);
        fid = -1;
        delete(testFile);
        workRoot = string(candidate);
        return;
    catch ME
        if fid >= 0
            fclose(fid);
        end
        failureMessages(end+1,1) = candidate + ": " + string(ME.message); %#ok<AGROW>
    end
end

error('Could not find a writable work directory. Attempts:\n%s', ...
    strjoin(failureMessages,newline));
end
