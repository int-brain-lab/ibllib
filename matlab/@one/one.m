classdef one
    %ONE Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        alyx_client
        ftp
        params
    end
    
    methods
        function self = one(varargin)
            % myone = one()
            % myone = one('base_url','https://test.alyx.internationalbrainlab.org',...
            %             'user','test_user','password','pass');
            p = inputParser;
            addParameter(p,'user', '', @isstr)
            addParameter(p,'password', '', @isstr)
            addParameter(p,'base_url', '', @isstr)
            parse(p,varargin{:});
            for fn = fieldnames(p.Results)', eval([fn{1} '= p.Results.' (fn{1}) ';']); end
            self.alyx_client = AlyxClient('user',user,'password',password,'base_url',base_url);
        end
    end
    
    methods (Access=private)
       par = get_params(self) 
    end
    
    methods
        session_info = info(self, eeid)
        dataset_list = list(self, eid)
        varargout = load(self, eeid, varargin)
        search_result = search(self, varargin)
    end
end

