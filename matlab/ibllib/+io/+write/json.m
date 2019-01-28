function json(filename, s)
% io.write.json(filename, structure)
% from a structure (or other variable at your own risk) writes a
% human-readable (ie not 1km long one-liner) indented json.

sj = jsonencode(s);
sj = strrep(sj, '{', ['{' newline ' ']);
sj = strrep(sj, ',', [',' newline ' ']);
sj = strrep(sj, '}', [newline '}']);
sj = strrep(sj, ':', ': ');

fid = fopen(filename,'w+');
fwrite(fid, [sj newline]);
fclose(fid);
