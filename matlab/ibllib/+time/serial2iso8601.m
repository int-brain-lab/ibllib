function str_dt = serial2iso8601(dt)
% str_dt = time.serial2iso8601(dt)
%ISO 8601: '2018-05-22T14:35:22.99585' or '2018-05-22T14:35:22' 
% the latter is enforced for the django REST API

str_dt = datestr( dt, 'yyyy-mm-ddTHH:MM:SS');
