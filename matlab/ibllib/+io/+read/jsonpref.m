function [par] = jsonpref(str_params)
% [par] = io.read.jsonpref(str_params)

par = [];
par_file = [io.getappdir filesep '.' str_params];
if ~exist(par_file,'file'), return, end

par = io.read.json(par_file);
end

