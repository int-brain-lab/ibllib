function password = gui_password(varargin)
% gui_password()
%   PW = gui_password() brings up a dialog for entering a password.
%       The dialog masks the characters with a "*".
%   PW = gui_password(...) accepts parameters
%       DialogTitle : title of window
%       uiMode - 'Query' or 'Create'. 'Query' asks the user to type the
%              password. 'Create' asks the user to create a new password.
%              This requires the user to type the password twice. Default
%              is 'Query'.
%       minChar  - a positive number that indicates the minimum number of
%              characters the password should have. This is only used when
%              the MODE is 'Create'. Default is 5.
%       PasswordType - 'AlphaOnly', 'NumOnly', 'AlphaNum', or 'AlphaNumSpecial'.
%              Restricts the type of password when the MODE is 'Create'.
%              Default is 'AlphaOnly'.
%                'AlphaOnly' - Only alphabets.
%                'NumOnly'   - Only numbers.
%                'AlphaNum'  - Must have both alphabets and numbers but no
%                              special characters.
%                'AlphaNumSpecial' - Must have alphabets, numbers, and
%                              special characters.
%
%               Special characters:  !"#$%&'()*+,-./:;<=>?@[\]^_`{|}
%
%   Examples:
%       pw = gui_password()
%       pw = gui_password('DialogTitle', 'This will be displayed on top')
%       pw = gui_password('uiMode', 'Create', 'PasswordType', 'AlphaNum')
%       pw = gui_password('Create', 'AlphaOnly', 8)

% Version History:
%   1.0 - Oct 2010.

% Jiro Doke
% Copyright 2010 The MathWorks, Inc.

[uiMode, passwordType, minChar, DialogTitle]=deal([]);
modstr = {'Query', 'Create', 'AlphaOnly', 'NumOnly', 'AlphaNum','AlphaNumSpecial'};
ptyp = {'AlphaOnly', 'NumOnly', 'AlphaNum', 'AlphaNumSpecial'};
p = inputParser;
addParameter(p,'uiMode', 'Query', @(str) any(cellfun(@(x) strcmpi(x,str), modstr)))
addParameter(p,'passwordType', 'AlphaOnly', @(str) any(cellfun(@(x) strcmpi(x,str), ptyp)))
addParameter(p,'minChar', 5, @isscalar)
addParameter(p,'DialogTitle', 'Enter Password:', @isstr)
parse(p,varargin{:});
for fn = fieldnames(p.Results)', eval([fn{1} '= p.Results.' (fn{1}) ';']); end

num = 1;
canceled = false;

while true
    
    fh = figure(...
        'Visible', 'off', ...
        'Name', DialogTitle, ...
        'NumberTitle', 'off', ...
        'Units', 'Pixels', ...
        'Position', [0, 0, 500, 50], ...
        'Toolbar', 'none', ...
        'Menubar', 'none', ...
        'CloseRequestFcn', @closeFcn, ...
        'WindowStyle', 'modal', ...
        'KeyPressFcn', @passwordKeyPressFcn);
    
    th = uicontrol(...
        'Style', 'edit', ...
        'Units', 'Pixels', ...
        'Position', [10, 10, 480, 30], ...
        'BackgroundColor', 'white', ...
        'Enable', 'inactive', ...
        'String', '_', ...
        'FontName', 'FixedWidth', ...
        'FontSize', 10);
    
    movegui(fh, 'center');
    set(fh, 'Visible', 'on');
    
    % Default password
    password = '';
    
    uiwait(fh);
    drawnow;
    
    if canceled
        password = '';
        break;
    elseif strcmp(uiMode, 'Query')
        % "password" already has the characters
        break;
    else
        if num == 1  % First time through
            if isempty(password)
                return;
            end
            
            % Make sure the valid characters were typed
            s = validatePassword();
            if s   % if so, save that, and go to "retype"
                password1 = password;
                num = 2;
            end
        else         % Retype
            % Check to see if the two passwords match
            if isequal(password, password1)  % if so, break (OK)
                break;
            else   % if not, notify that they did not match, and go back to first try
                uiwait(warndlg('The passwords do not match', 'Error', 'modal'));
                num = 1;
            end
        end
    end
    
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Nested Functions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%--------------------------------------------------------------------------
    function closeFcn(obj, edata) %#ok<INUSD>
        canceled = true;
        delete(obj);
    end

%--------------------------------------------------------------------------
    function success = validatePassword()
        success = false;
        
        % Check password length
        if length(password) < minChar
            uiwait(warndlg(sprintf('The password must be at least %d characters long.', minChar), 'Error', 'modal'));
            return;
        end
        
        % Check valid use of characters
        switch passwordType
            case 'AlphaOnly'        % Only alphabets
                if ~all(isstrprop(password, 'alpha'))
                    uiwait(warndlg('The password must be all alphabets.', 'Error', 'modal'));
                    return;
                end
            case 'NumOnly'          % Only numbers
                if ~all(isstrprop(password, 'digit'))
                    uiwait(warndlg('The password must be all numbers.', 'Error', 'modal'));
                    return;
                end
            case 'AlphaNum'         % Alphabets and numbers
                if ~all(isstrprop(password, 'alphanum')) || ...
                        ~(any(isstrprop(password, 'alpha')) && any(isstrprop(password, 'digit')))
                    uiwait(warndlg('The password must have alphabets and numbers.', 'Error', 'modal'));
                    return;
                end
            case 'AlphaNumSpecial'  % Alphabets, numbers, and special characters
                if ~all(isstrprop(password, 'graphic')) || ...
                        ~(any(isstrprop(password, 'alpha')) && any(isstrprop(password, 'digit')) && ...
                        any(~(isstrprop(password, 'alpha') | isstrprop(password, 'digit'))))
                    uiwait(warndlg('The password must have alphabets, numbers, and special characters.', 'Error', 'modal'));
                    return;
                end
        end
        success = true;
    end

%--------------------------------------------------------------------------
    function passwordKeyPressFcn(obj, edata)
        switch edata.Key
            case 'return'
                delete(obj);
                return;
            case 'escape'
                canceled = true;
                delete(obj);
                return;
            case {'backspace', 'delete'}
                if ~isempty(password)
                    password(end) = '';
                end
            otherwise
                c = edata.Character;
                if ~isempty(c)
                    if c >= '!' && c <= '}'
                        password = [password, c];
                    else
                        disp('Unrecognized character');
                    end
                end
        end
        set(th, 'String', [repmat('*', 1, length(password)), '_']);
    end

end