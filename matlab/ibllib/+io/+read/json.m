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

if exist('jsondecode', 'builtin')
    data = jsondecode(data);
else
    if ~exist('loadjson', 'file')
        error( ['Old version of Matlab (<2018), should install JSONlab', char(10)...
            'https://github.com/fangq/jsonlab', char(10), ...
            'https://www.mathworks.com/matlabcentral/fileexchange/33381-jsonlab-a-toolbox-to-encode-decode-json-files'])
    end
    data = loadjson(data);
end

