from ibllib.io import params as iopar
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Set HTTP_DATA_SERVER_PWD from secret')
    parser.add_argument('secret', help='secret input HTTP_DATA_SERVER_PWD')
    parser.add_argument('secret2', help='secret input ALYX_PWD')
    args = parser.parse_args()

    pars = iopar.read('one_params').as_dict()
    pars['HTTP_DATA_SERVER_PWD'] = args.secret
    pars['HTTP_DATA_SERVER'] = "https://ibl.flatironinstitute.org"
    pars['ALYX_PWD'] = args.secret2
    pars['ALYX_LOGIN'] = 'test_user'
    pars['ALYX_URL'] = 'https://test.alyx.internationalbrainlab.org'

    iopar.write('one_params', pars)
