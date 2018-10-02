
%% Register Folder to endpoint (Todo Parameters in AlyxClient)
chem = '/home/owinter/Documents/IBL/Nick/Subjects/clns0730/2018-07-30/1'
self = AlyxClient();
fn = io.dir(chem);
% remove the path from the file names
fn = cellfun(@(x) strrep(x, [chem filesep] , ''),  fn, 'UniformOutput', false);
split_path= strsplit(chem, filesep);
% create the session structure
session = struct;
session.subject = split_path{end-2}; % 'clns0730'
session.procedures = {'Behavior training/tasks'};
session.narrative = 'auto-generated session';
session.start_time =  datestr(datenum(split_path{end-1},'yyyy-mm-dd') + 0.5, 'yyyy-mm-ddTHH:MM:SS') ;
session.type = 'Base';
session.number = split_path{end};
session.users = {'olivier'};


%% First step is to make sure the subject exist
[matching_subjects] = self.get(['/subjects?nickname=' session.subject '&user=olivier'])

if isempty(matching_subjects)
    subject = struct('nickname', session.subject, 'responsible_user',{'olivier'},'protocol_number','1');
    subject.project = '<Project_test_IBL>';
    subject.genotype = {'',''}
    rep = self.post('/subjects', subject)
end
%% First step is to make sure the subject exists, also that the sessions exists, if it doesn't everything crashes
drange = datenum(session.start_time, 'yyyy-mm-ddTHH:MM:SS');
drange = datestr(drange+[0 0], ',yyyy-mm-dd')
drange = flatten(drange')';
drange = drange(2:end);

%%
[matching_sessions] = self.get(['/sessions?'...
    'type=' session.type, ...
    '&subject=' session.subject,...
    '&user=' session.users{1}, ...
    '&number=' session.number, ...
    '&date_range=' drange]);
matching_sessions
%%
if isempty(matching_sessions)
    rep = self.post('/sessions', session);
end
% todo create base session if not already in
% todo create number session if not already in
% base_submit = self.post('sessions', session);

%% Go Over the files
repos = self.get('/data-repository');
formats = self.get('/data-formats');
dtypes = self.get('/dataset-types');


%% 
% todo ensure the dataset types are ok
D =  struct('created_by', 'olivier' );
D.dns =  'ibl.flatironinstitute.org';
D.path = 'clns0730/2018-07-30/1';
% D.data_repository = 'flatiron_mainenlab';
D.filenames = fn;
[rep] = self.post('/register-file', D);

%  arrayfun(@(x) x.name, datasetTypes, 'UniformOutput', false)
%     {'Block'                             }
%     {'cwFeedback.rewardVolume'           }
%     {'cwFeedback.times'                  }
%     {'cwFeedback.type'                   }
%     {'cwGoCue.times'                     }
%     {'cwResponse.choice'                 }
%     {'cwResponse.times'                  }
%     {'cwStimOn.contrastLeft'             }
%     {'cwStimOn.contrastRight'            }
%     {'cwStimOn.times'                    }
%     {'cwTrials.inclTrials'               }
%     {'cwTrials.intervals'                }
%     {'cwTrials.repNum'                   }
%     {'expDefinition'                     }
%     {'galvoLog'                          }
%     {'Hardware Info'                     }
%     {'lfp.raw'                           }
%     {'Parameters'                        }
%     {'photometry.calciumLeft_normalized' }
%     {'photometry.calciumRight_normalized'}
%     {'photometry.timestamps'             }
%     {'unknown'                           }
%     {'wheel.position'                    }
%     {'wheel.timestamps'                  }
%     {'wheel.velocity'                    }


% TODO label the unknown datasets as such...

%%
[matching_sessions] = self.get(['/sessions?type=Base&subject=' session.subject]);





