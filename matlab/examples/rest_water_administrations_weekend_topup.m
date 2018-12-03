%% Parameters
admin_dates = [datenum([2018 12 05 11 50 23]), datenum([2018 12 06 11 50 23])];
waterType = 'Water'; % Type of water to be given
lab = 'zadorlab'; % Which lab are you in?
subjects = {}; % this will query all alive and water restricted subjects from the lab
% subjects = {'IBL_1', 'IBL_10'};% it is safer to introduce a list of subject nicknames
alyx_url ='https://dev.alyx.internationalbrainlab.org';% NB: this is a good practice to test first on dev.alyx, look at the result
% online and then run on the production database

%% Function
one = One('alyx_url', alyx_url);
ac = one.alyx_client;
if isempty(subjects)
subjects = ac.get(sprintf('/subjects?stock=False&alive=True&lab=%s&water_restricted=True', lab));
end
% Validate water type
validTypes = ac.get('/water-type').name;
assert(any(strcmp(waterType, validTypes)));
for s = 1:length(subjects.nickname)
  wtr = subjects.remaining_water(s);
  wtr = ceil(wtr*10^2)/10^2; % Round up
  if wtr > 0
    for dat = admin_dates
        wa_ = struct(...
            'subject', subjects.nickname{s},...
            'date_time', time.serial2iso8601(dat),...  
            'water_type', waterType,...
            'user', ac.user,...
            'water_administered', wtr, ...
            'adlib', false);
      rep = ac.post('/water-administrations', wa_);
      disp(rep)
    end
  end
end