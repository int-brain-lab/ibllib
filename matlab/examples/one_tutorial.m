% https://ibllib.readthedocs.io/en/latest/_static/one_demo.html
% one = One;
one = One('alyx_url','https://test.alyx.internationalbrainlab.org',...
              'alyx_login','test_user','alyx_pwd','TapetesBloc18');

%%
[eid, ses] = one.search('users', 'olivier', 'date_range', datenum([2018 8 24 ; 2018 8 24])) ;

%%
one.list(eid)

one.list(eid,'keyword', 'all')

one.list([],'keyword', 'labs')
one.list([],'keyword', 'datasets')
one.list([],'keyword', 'users')
one.list([],'keyword', 'subjects')

%%

dataset_types = {'clusters.templateWaveforms', 'clusters.probes', 'clusters.depths'};
eid = 'cf264653-2deb-44cb-aa84-89b82507028a';
[wf, pr, d ]= one.load(eid, 'data' ,dataset_types);

figure,
imagesc(squeeze(wf(1,:,:)), 'Parent', subplot(2,1,1)), colormap('bone')
plot(subplot(2,1,2), squeeze(wf(1,:,:)))

%%
D = one.load(eid, 'data' ,dataset_types, 'dclass_output', true);

disp(D.local_path)
disp(D.dataset_type)
disp('dimensions: ')
disp(cellfun(@(x) length(x), D.data))
%%
[eid, ses_info ]= one.search('subjects', 'flowers');
D = one.load(eid);

%% load empty dataset
eid = 'cf264653-2deb-44cb-aa84-89b82507028a';
dataset_types = {'clusters.probes', 'thisDataset.IveJustMadeUp', 'clusters.depths'};
[t, empty, cl ] = one.load(eid, 'data', dataset_types)
isempty(empty) % true !


%% Search 
eid = one.search('subject', 'flowers');

[eid, ses_info] = one.search('users',{'test_user', 'olivier'});
[eid, ses_info] = one.search('users', 'olivier');

drange = datenum(['2018-08-24'; '2018-08-24']);
eid = one.search('users','olivier', 'date_range', drange);

