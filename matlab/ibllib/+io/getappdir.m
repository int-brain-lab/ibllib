function userdir = getappdir()
% userdir = io.getappdir()
% returns appdata dir to store preferences

switch true
    case ispc;   userdir= getenv('APPDATA');
    case isunix; userdir= getenv('HOME');
    case ismac;  userdir = getenv('HOME'); % NOT TESTED
end