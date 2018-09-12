function setup()
%SETUP Summary of this function goes here
%   Detailed explanation goes here
AlyxClient.setup()


prefs = getpref('One');
if isempty(prefs)
    prefs = struct('base_url','https://test.alyx.internationalbrainlab.org',...
        'user','test_user',...
        'password','');
end

%% Get it back afterwards
par.alyx = getpref('Alyx')


end

