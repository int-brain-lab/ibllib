function userdir = getuserdir()
% userdir = io.getuserdir()
% returns the home directory for the current user. Windows/Linux tested,
% untested on Mac.

switch true
    case ispc;   userdir= getenv('USERPROFILE');
    case isunix; userdir= getenv('HOME');
    case ismac;  userdir = getenv('HOME'); % NOT TESTED
end