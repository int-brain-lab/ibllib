function json(filename, s)
% io.write.json(filename, structure)
% from a structure (or other variable at your own risk) writes a
% human-readable (ie not 1km long one-liner) indented json.

if exist('jsonencode', 'builtin')
    sj = jsonencode(s);
    sj = strrep(sj, '{', ['{' newline ' ']);
    sj = strrep(sj, ',', [',' newline ' ']);
    sj = strrep(sj, '}', [newline '}']);
    sj = strrep(sj, ':', ': ');
else
    if ~exist('savejson', 'file')
        error( ['Old version of Matlab (<2018), should install JSONlab', char(10)...
            'https://github.com/fangq/jsonlab', char(10), ...
            'https://www.mathworks.com/matlabcentral/fileexchange/33381-jsonlab-a-toolbox-to-encode-decode-json-files'])
    end
    sj = savejson('', s);
end

fid = fopen(filename,'w+');
fwrite(fid, [sj newline]);
fclose(fid);
