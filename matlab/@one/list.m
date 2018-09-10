function dataset_list = list(self, eid)
% dtypes = one.list(eid) : returns a cell array containing dataset types belonging to the current session
% dtypes = one.list('86e27228-8708-48d8-96ed-9aa61ab951db');
% dtypes = one.list('https://test.alyx.internationalbrainlab.org/sessions/86e27228-8708-48d8-96ed-9aa61ab951db');

session_info = self.alyx_client.get_session(eid);
session_data_info = session_info.data_dataset_session_related;
dataset_list = unique(session_data_info.dataset_type);
