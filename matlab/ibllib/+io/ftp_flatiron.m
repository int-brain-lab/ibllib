par = io.read.jsonpref('one_params')

cpar = {...
    [par.FTP_DATA_SERVER(7:end)], ...
    par.FTP_DATA_SERVER_LOGIN, ...
    par.FTP_DATA_SERVER_PWD, ...
    };



ftp_obj = ftp(cpar{:});
ftp_obj.binary;
%%
isa(ftp_obj, 'ftp')
isvalid(ftp_obj)


% mget
%%

try
a =1;
a(2)
catch err
   err.addCause('turlu')
   rethrow(err)
end
    
    
    
    
    
