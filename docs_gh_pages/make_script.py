import os
import sys
import argparse
import subprocess
from pathlib import Path

root = os.path.abspath(os.path.join(os.path.dirname(__file__)))
scripts_path = os.path.join(root, 'scripts')
sys.path.insert(1, scripts_path)
from execute_notebooks import process_notebooks  # noqa: E402

nb_path = os.path.join(root, 'notebooks')
nb_path_external = os.path.join(Path(root).parent, 'examples', 'one', 'notebooks')


def make_documentation(execute, documentation, clean, github, message):

    if execute:
        process_notebooks(nb_path, execute=True)
        process_notebooks(nb_path_external, execute=True)
        print("Finished processing notebooks")

    if documentation:
        print("Cleaning up previous documentation")
        os.system("make clean")
        print("Making documentation")
        os.system("make html")

    # Clean up the build path regardless
    build_nb_path = os.path.join(root, '_build', 'html', 'notebooks')
    build_nb_external_path = os.path.join(root, '_build', 'html', 'notebooks_external')
    process_notebooks(build_nb_path, execute=False, cleanup=True)
    process_notebooks(build_nb_external_path, execute=False, cleanup=True)

    # Clean up notebooks in directory if also specified
    if clean:
        print("Cleaning up notebooks")
        process_notebooks(nb_path, execute=False, cleanup=True)
        process_notebooks(nb_path_external, execute=False, cleanup=True)

    # If github is True push the latest documentation to gh-pages
    if github:
        # Need to figure out how to do this
        if not message:
            message = "commit latest documentation"

        subprocess.call(['scripts\gh_push.sh', message], shell=True)  # noqa: E605


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Make IBL documentation')

    parser.add_argument('-e', '--execute', default=False, action='store_true',
                        help='Execute notebooks')
    parser.add_argument('-d', '--documentation', default=False, action='store_true',
                        help='Make documentation')
    parser.add_argument('-c', '--cleanup', default=False, action='store_true',
                        help='Cleanup notebooks once documentation made')
    parser.add_argument('-gh', '--github', default=False, action='store_true',
                        help='Push documentation to gh-pages')
    parser.add_argument('-m', '--message', default=None, required=False, type=str,
                        help='Commit message')
    args = parser.parse_args()
    print(args)
    make_documentation(execute=args.execute, documentation=args.documentation, clean=args.cleanup,
                       github=args.github, message=args.message)
