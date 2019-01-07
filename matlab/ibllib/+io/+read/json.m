function [data] = json(file_json)
% [data] = io.read.json(file_json)

data = '';
if ~exist(file_json,'file')
    warning([file_json ' doesn''t exist'])
    return
end

fid = fopen(file_json); 
data = fread(fid,Inf,'*char')';
fclose(fid);

data = jsondecode(data);
end

