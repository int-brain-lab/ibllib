classdef one
    %ONE Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        alyx_client
        ftp
        par
    end
    
    methods
        function self = one(varargin)
            % myone = one()
            % myone = one('alyx_url','https://test.alyx.internationalbrainlab.org',...
            %             'alyx_login','test_user','alyx_pwd','pass');
            
            % Init preferences and parse input arguments
            self.par = io.read.jsonpref('one_params');
            if isempty(self.par), self.par = self.setup; end 
            p = inputParser;
            addParameter(p,'alyx_login', self.par.ALYX_LOGIN, @isstr)
            addParameter(p,'alyx_url', self.par.ALYX_URL, @isstr)
            addParameter(p,'alyx_pwd', self.par.ALYX_PWD, @isstr)
            parse(p,varargin{:});
            for fn = fieldnames(p.Results)', eval([fn{1} '= p.Results.' (fn{1}) ';']); end
            % Instantiate the Alyx Client connection
            try
                self.alyx_client = AlyxClient('user',alyx_login,'password',alyx_pwd,'base_url',alyx_url);
            catch err
                warning(['Error ocurred while instantiating Alyx client: ' err.message])
                rethrow(err)
            end
            % Instantiate the FTP connection
            try
                self.ftp = ftp(self.par.FTP_DATA_SERVER(7:end), ...
                               self.par.FTP_DATA_SERVER_LOGIN, ...
                               self.par.FTP_DATA_SERVER_PWD);
                self.ftp.binary;
            catch err
                warning(['Error ocurred while instantiating FTP client to FlatIron: ' err.message])
                rethrow(err)
            end
        end
    end
    
    methods (Access=private)
       par = get_params(self) 
    end
    
    methods
        session_info = info(self, eeid)
        varargout = list(self, eid, varargin)
        varargout = load(self, eid, varargin)
        [eids, ses] = search(self, varargin)
    end
    
    methods (Static)
         par = setup()
    end
end

