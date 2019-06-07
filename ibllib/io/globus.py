import globus_sdk as globus
from ibllib.io import params


def _login(globus_client_id, refresh_tokens=False):

    client = globus.NativeAppAuthClient(globus_client_id)
    client.oauth2_start_flow(refresh_tokens=refresh_tokens)

    authorize_url = client.oauth2_get_authorize_url()
    print('Please go to this URL and login: {0}'.format(authorize_url))
    auth_code = input(
        'Please enter the code you get after login here: ').strip()

    token_response = client.oauth2_exchange_code_for_tokens(auth_code)
    globus_transfer_data = token_response.by_resource_server['transfer.api.globus.org']

    token = dict(refresh_token=globus_transfer_data['refresh_token'],
                 transfer_token=globus_transfer_data['access_token'],
                 expires_at_s=globus_transfer_data['expires_at_seconds'],
                 )
    return token


def login(globus_client_id):
    token = _login(globus_client_id, refresh_tokens=False)
    authorizer = globus.AccessTokenAuthorizer(token['transfer_token'])
    tc = globus.TransferClient(authorizer=authorizer)
    return tc


def setup(globus_client_id, str_app='globus'):
    # Lookup and manage consents there
    # https://auth.globus.org/v2/web/consents
    gtok = _login(globus_client_id, refresh_tokens=True)
    params.write(str_app, gtok)


def login_auto(globus_client_id, str_app='globus'):
    token = params.read(str_app)
    if not token:
        raise ValueError("Token file doesn't exist, run ibllib.io.globus.setup first")
    client = globus.NativeAppAuthClient(globus_client_id)
    client.oauth2_start_flow(refresh_tokens=True)
    authorizer = globus.RefreshTokenAuthorizer(token.transfer_rt, client)
    return globus.TransferClient(authorizer=authorizer)
