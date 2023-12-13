"""(DEPRECATED) Globus SDK utility functions.

This has been deprecated in favour of the one.remote.globus module.
"""
import re
import sys
import os
from pathlib import Path
import warnings
import traceback
import logging

import globus_sdk as globus
from iblutil.io import params


for line in traceback.format_stack():
    print(line.strip())

msg = 'ibllib.io.globus has been deprecated. Use one.remote.globus instead. See stack above'
warnings.warn(msg, DeprecationWarning)
logging.getLogger(__name__).warning(msg)


def as_globus_path(path):
    """
    (DEPRECATED) Convert a path into one suitable for the Globus TransferClient.

    NB: If using tilda in path, the home folder of your Globus Connect instance must be the same as
    the OS home dir.

    :param path: A path str or Path instance
    :return: A formatted path string

    Examples:
        # A Windows path
        >>> as_globus_path('E:\\FlatIron\\integration')
        >>> '/E/FlatIron/integration'

        # A relative POSIX path
        >>> as_globus_path('../data/integration')
        >>> '/mnt/data/integration'

        # A globus path
        >>> as_globus_path('/E/FlatIron/integration')
        >>> '/E/FlatIron/integration'
    TODO Remove in favour of one.remote.globus.as_globus_path
    """
    msg = 'ibllib.io.globus.as_globus_path has been deprecated. Use one.remote.globus.as_globus_path instead.'
    warnings.warn(msg, DeprecationWarning)

    path = str(path)
    if (
        re.match(r'/[A-Z]($|/)', path)
        if sys.platform in ('win32', 'cygwin')
        else Path(path).is_absolute()
    ):
        return path
    path = Path(path).resolve()
    if path.drive:
        path = '/' + str(path.as_posix().replace(':', '', 1))
    return str(path)


def _login(globus_client_id, refresh_tokens=False):
    # TODO Import from one.remove.globus
    client = globus.NativeAppAuthClient(globus_client_id)
    client.oauth2_start_flow(refresh_tokens=refresh_tokens)

    authorize_url = client.oauth2_get_authorize_url()
    print('Please go to this URL and login: {0}'.format(authorize_url))
    auth_code = input(
        'Please enter the code you get after login here: ').strip()

    token_response = client.oauth2_exchange_code_for_tokens(auth_code)
    globus_transfer_data = token_response.by_resource_server['transfer.api.globus.org']

    token = dict(refresh_token=globus_transfer_data['refresh_token'],
                 access_token=globus_transfer_data['access_token'],
                 expires_at_seconds=globus_transfer_data['expires_at_seconds'],
                 )
    return token


def login(globus_client_id):
    msg = 'ibllib.io.globus.login has been deprecated. Use one.remote.globus.Globus instead.'
    warnings.warn(msg, DeprecationWarning)

    token = _login(globus_client_id, refresh_tokens=False)
    authorizer = globus.AccessTokenAuthorizer(token['access_token'])
    tc = globus.TransferClient(authorizer=authorizer)
    return tc


def setup(globus_client_id, str_app='globus/default'):
    msg = 'ibllib.io.globus.setup has been deprecated. Use one.remote.globus.Globus instead.'
    warnings.warn(msg, DeprecationWarning)
    # Lookup and manage consents there
    # https://auth.globus.org/v2/web/consents
    gtok = _login(globus_client_id, refresh_tokens=True)
    params.write(str_app, gtok)


def login_auto(globus_client_id, str_app='globus/default'):
    msg = 'ibllib.io.globus.login_auto has been deprecated. Use one.remote.globus.Globus instead.'
    warnings.warn(msg, DeprecationWarning)
    token = params.read(str_app, {})
    required_fields = {'refresh_token', 'access_token', 'expires_at_seconds'}
    if not (token and required_fields.issubset(token.as_dict())):
        raise ValueError("Token file doesn't exist, run ibllib.io.globus.setup first")
    client = globus.NativeAppAuthClient(globus_client_id)
    client.oauth2_start_flow(refresh_tokens=True)
    authorizer = globus.RefreshTokenAuthorizer(token.refresh_token, client)
    return globus.TransferClient(authorizer=authorizer)


def get_local_endpoint():
    msg = 'ibllib.io.globus.get_local_endpoint has been deprecated. Use one.remote.globus.get_local_endpoint_id instead.'
    warnings.warn(msg, DeprecationWarning)

    if sys.platform == 'win32' or sys.platform == 'cygwin':
        id_path = Path(os.environ['LOCALAPPDATA']).joinpath("Globus Connect")
    else:
        id_path = Path.home().joinpath(".globusonline", "lta")
    with open(id_path / "client-id.txt", 'r') as fid:
        globus_id = fid.read()
    return globus_id.strip()
