from ibllib.io import params as iopar
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Set HTTP_DATA_SERVER_PWD from secret')
    parser.add_argument('secret', help='secret input')
    args = parser.parse_args()

    pars = iopar.read('one_params').as_dict()
    pars['HTTP_DATA_SERVER_PWD'] = args.secret
    iopar.write('one_params', pars)
