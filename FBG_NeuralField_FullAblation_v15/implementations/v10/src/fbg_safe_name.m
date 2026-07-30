function name = fbg_safe_name(value)
%FBG_SAFE_NAME Convert a display name into a portable folder name.
name = regexprep(char(string(value)), "[^A-Za-z0-9_-]", "_");
end
