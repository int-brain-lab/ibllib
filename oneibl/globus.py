from ibllib.io import globus
from one.api import OneAlyx


class OneGlobus(OneAlyx):

    def __init__(self, **kwargs):
        # Init connection to the database
        super(OneGlobus, self).__init__(**kwargs)
        # Init connection to Globus if needed
        self._tc = globus.login_auto(self.alyx._par.GLOBUS['CLIENT_ID'], str_app='globus_one')

    def setup(self):
        super(OneGlobus, self).setup()
        globus.login_auto(self.alyx._par.GLOBUS['CLIENT_ID'], str_app='globus_one')

    def download(self, eids):
        pass


# transfer_object = globus.TransferData(
#     self._globus_transfer_client,
#     source_endpoint=par.GLOBUS['SERVER_ENDPOINT'],
#     destination_endpoint=par.GLOBUS['LOCAL_ENDPOINT'],
#     verify_checksum=True,
#     sync_level='checksum',
#     label='ONE_transfer_python')
