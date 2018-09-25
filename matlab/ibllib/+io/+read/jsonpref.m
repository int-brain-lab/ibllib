function [par] = jsonpref(str_params)
% [par] = io.read.jsonpref(str_params)

par = [];
par_file = [io.getappdir filesep '.' str_params];
if ~exist(par_file,'file'), return, end

fid = fopen(par_file); 
par = fread(fid,Inf,'*char')';
fclose(fid);

par = jsondecode(par);
end

