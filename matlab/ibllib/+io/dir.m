function file_list = dir(chem, varargin)
% file_list = io.dir(folder_path, filter)
% file_list = io.dir(folder_path, filter, 'recursive', false)
% recursive: default is true (not implemented yet)
% returns a cell-array list of full file paths 

% TODO recursive function

%% Input parser
persistent p; % for recursive calls, avoids setting this up each time
p = inputParser;
addRequired(p, 'chem');
addOptional(p,'extfilt', '*.*');
addParameter(p,'recursive', true)
parse(p,chem,varargin{:});
for fn = fieldnames(p.Results)', eval([fn{1} '= p.Results.' (fn{1}) ';']); end

%% dir part
if chem(end) ~= filesep, chem = [ chem, filesep]; end
file_list = dir([chem extfilt]);
% removes the current and previous directory from the list
file_list = file_list(arrayfun(@(x) ~all(x.name=='.'), file_list));
file_list = arrayfun(@(x) [x.folder filesep x.name], file_list, 'UniformOutput', false);

