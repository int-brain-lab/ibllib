classdef AlyxClient
    properties (SetAccess=private)
        base_url = ''
        user = ''
        timeout = 30
        weboptions = []
    end
    
    properties (SetAccess=private, Hidden=true)
        password = ''
        token = ''
        headers = ''
    end
    
    methods
        function self = AlyxClient()
            % setup weboptions for 
            self = self.getparameters();
            self.weboptions = weboptions('MediaType','application/json','Timeout',self.timeout );      
            self = self.authenticate();
        end
    end
    
    methods (Access=private)
        function self = authenticate(self)
            rep = self.post('/auth-token', struct('username', self.user, 'password', self.password));
            self.token = rep.token;
            self.weboptions.HeaderFields = { 'Authorization', ['Token ' self.token]};
        end
        
        function self = getparameters(self)
            % This may well be a temporary way to setup the client. At least it doesn't require a parameter files
            prefs = getpref('Alyx');
            if isempty(prefs),
                self.setup;
                prefs = getpref('Alyx');
            end
            for ff = fields(prefs)'
                eval(['self.' ff{1} '='  'prefs.' ff{1} ';']);
            end
        end
    end
    
    methods (Access = public)
        function rep = post(self,end_point, request_struct)
            % rep = post(url, request_struct)
            url = [self.base_url  end_point];
            rep = webwrite(url,  jsonencode(request_struct), setfield( self.weboptions, 'RequestMethod', 'post') );
        end
        
         function rep = get(self,end_point)
             % rep = get(url)
            url = [self.base_url  end_point];
            rep = webread(url, self.weboptions);
            rep = flatten(rep);
         end
        
         function create_session(self, session_structure)
             % self.create_session(session_structure)
            %  session =  struct with fields: 
            %        subject: 'clns0730'
            %     procedures: {'Behavior training/tasks'}
            %      narrative: 'auto-generated session'
            %     start_time: '2018-07-30T12:00:00'
            %           type: 'Base'
            %         number: '1'
            %          users: {'olivier'}
             
             
             
             
         end
    end
    
        
    methods (Static)
        function setup()
            % AlyxClient.setup()
            % Prompts the user for base_url, user and password and stores for subsequent uses.
            % Will change once we settle on a proper way to handle parameters
            prefs = getpref('Alyx');
            if isempty(prefs), prefs = struct('base_url','','user','','password',''); end
            base_url = input(['base_url (example: alyx.cortexlab.net), (current: ' prefs.base_url ') '], 's');
            if ~isempty(base_url), prefs.base_url = base_url; end
            user = input(['user (example:olivier), (current: ' prefs.user ')'], 's'); 
            if ~isempty(user), prefs.user = user; end
            password = passwordUI();
            if ~isempty(password), prefs.password = password; end
            for ff = fields(prefs)'
                setpref('Alyx', ff{1},  prefs.(ff{1}));
            end
        end
    end
end