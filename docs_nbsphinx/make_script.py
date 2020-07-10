import os
import sys
import argparse
import subprocess

root = os.path.abspath(os.path.join(os.path.dirname(__file__)))
scripts_path = os.path.join(root, 'scripts')
sys.path.insert(1, scripts_path)
from execute_notebooks import process_notebooks


def make_documentation(execute, documentation, clean, github, message):

    print("Cleaning up previous documentation")
    os.system("make clean")
    nb_path = []
    print(execute)

    if execute:
        #from execute_notebooks import process_notebooks
        nb_path = os.path.join(root, 'notebooks')
        process_notebooks(nb_path, execute=True)
        print("Finished processing notebooks")

    if documentation:
        print("Making documentation")
        os.system("make html")

    if clean:
    # Clean up notebooks both in directory and in build directory
        print("Cleaning up notebooks")
        build_nb_path = os.path.join(root, '_build', 'html', 'notebooks')
        if not nb_path:
            nb_path = os.path.join(root, 'notebooks')
        process_notebooks(nb_path, execute=False, cleanup=True)
        process_notebooks(build_nb_path, execute=False, cleanup=True)

    if github:
        # Need to figure out how to do this
        if not message:
            message = "commit now"
        subprocess.call(['scripts\gh_push.sh', message], shell=True)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Make IBL documentation')

    #parser.add_argument('-e', '--execute', default=True, required=False, action='store_true',
    #                    help='Execute notebooks')
    parser.add_argument('-e', '--execute', default=False, action='store_true')
    parser.add_argument('-d', '--documentation', default=True, required=False, action='store_true',
                        help='Make documentation')
    parser.add_argument('-c', '--cleanup', default=True, required=False, action='store_true',
                        help='Cleanup notebooks once documentation made')
    parser.add_argument('-gh', '--github', default=False, required=False, action='store_false',
                        help='Push documentation to gh-pages')
    parser.add_argument('-m', '--message', default=None, required=False, type=str,
                        help='Commit message')
    args = parser.parse_args()
    print(args)
    make_documentation(execute=args.execute, documentation=args.documentation, clean=args.cleanup,
                       github=args.github, message=args.message)