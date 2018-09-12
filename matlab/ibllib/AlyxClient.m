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
        function self = AlyxClient(varargin)
            % ac = AlyxClient()
            % ac = AlyxClient('user','test','password','pass','base_url',...
            %                 'https://test.alyx.internationalbrainlab.org');
            prefs = self.getparameters();
            % Handle input arguments, input arguments always overwrite preferences
            p = inputParser;
            addParameter(p,'user', prefs.user, @isstr)
            addParameter(p,'password', prefs.password, @isstr)
            addParameter(p,'base_url', prefs.base_url, @isstr)
            parse(p,varargin{:});
            for fn = fieldnames(p.Results)'; eval(['self.' fn{1} '= p.Results.' (fn{1}) ';']); end
            if isempty(self.password), self.password = prefs.password; end
            if isempty(self.base_url), self.base_url = prefs.base_url; end
            if isempty(self.user)    , self.user     = prefs.user;     end
            % setup weboptions for REST queries
            self.weboptions = weboptions('MediaType','application/json','Timeout',self.timeout );      
            self = self.authenticate();
        end
    end
    
    methods (Access=private)
        function self = authenticate(self)
            % REST query to authenticate against Alyx and get an access token
            rep = self.post('/auth-token', struct('username', self.user, 'password', self.password));
            self.token = rep.token;
            self.weboptions.HeaderFields = { 'Authorization', ['Token ' self.token]};
        end
        
        function prefs = getparameters(self)
            % Get parameters from preferences
            prefs = getpref('Alyx');
            if isempty(prefs)
                self.setup;
                prefs = getpref('Alyx');
            end
        end
    end
    
    methods (Access = public)        
         function rep = get(self,endpoint_url)
             % rep = get(url)
             % rep = ac.get('/sessions/86e27228-8708-48d8-96ed-9aa61ab951db')
             % rep = ac.get('https://test.alyx.internationalbrainlab.org/sessions/86e27228-8708-48d8-96ed-9aa61ab951db')
            if ~(strfind(endpoint_url, self.base_url)==1)
                endpoint_url = [self.base_url  endpoint_url];
            end
            rep = webread(endpoint_url, self.weboptions);
            rep = flatten(rep);
         end
         
         function session_info = get_session(self, session_url)
             % session_info = ac.get_session('86e27228-8708-48d8-96ed-9aa61ab951db')
             % session_info = ac.get_session('https://test.alyx.internationalbrainlab.org/sessions/86e27228-8708-48d8-96ed-9aa61ab951db') 
            if isempty(strfind(session_url, self.base_url))
                session_url = [self.base_url '/sessions/' session_url];
            end
            session_info = self.get(session_url);
         end
         
         function rep = post(self,end_point, request_struct)
            % rep = post(url, request_struct)
            url = [self.base_url  end_point];
            rep = webwrite(url,  jsonencode(request_struct), setfield(self.weboptions, 'RequestMethod', 'post') );
        end
%          function create_session(self, session_structure)
%              % self.create_session(session_structure)
%             %  session =  struct with fields: 
%             %        subject: 'clns0730'
%             %     procedures: {'Behavior training/tasks'}
%             %      narrative: 'auto-generated session'
%             %     start_time: '2018-07-30T12:00:00'
%             %           type: 'Base'
%             %         number: '1'
%             %          users: {'olivier'}
%          end
    end
    
        
    methods (Static)
        function setup()
            % AlyxClient.setup()
            % Prompts the user for base_url, user and password and stores for subsequent uses.
            prefs = getpref('Alyx');
            if isempty(prefs)
                prefs = struct('base_url','https://test.alyx.internationalbrainlab.org',...
                               'user','test_user',...
                               'password','');
            end
            % prompt for address
            base_url = input(['Alyx full URL: (example: https://test.alyx.internationalbrainlab.org), (current: ' prefs.base_url ') '], 's');
            if ~isempty(base_url)
                prefs.base_url = base_url;
            end
            % prompts for user
            user = input(['Alyx username (example:test_user), (current: ' prefs.user ')'], 's'); 
            if ~isempty(user)
                prefs.user = user;
            end
            % prompts for password
%             prefs.password
            password = passwordUI();
            if ~isempty(password)
                prefs.password = password;
            end
            % assign properties
            for ff = fields(prefs)'
                setpref('Alyx', ff{1},  prefs.(ff{1}));
            end
        end
    end
end
