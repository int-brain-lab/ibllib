function par_file = jsonpref(str_params, par)
% par_file = io.write.jsonpref(str_params, par)

par_file = [io.getappdir filesep '.' str_params];

str_json = jsonencode(par);
str_json = strrep(str_json, '","', ['",' char(10) '    "']) ;
str_json = strrep(str_json, '{', ['{' char(10) '    ']) ;
str_json = strrep(str_json, '}', [ char(10) '}']) ;


fid = fopen(par_file, 'w+'); 
par = fwrite(fid,str_json,'char')';
fclose(fid);

end