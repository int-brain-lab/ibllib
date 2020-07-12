# Script to push html doc files to gh pages

# Repo information
ORG=mayofaulkner
REPO=ibllib

commit_message = $1
# Clone the gh-pages branch to local documentation directory
git clone -b gh-pages "https://github.com/$ORG/$REPO.git" gh-pages
cd gh-pages
#rm -r gh-pages

# Copy everything from output of build into gh-pages branch
cp -R ../_build/html/* ./

echo commit_message
# Add and commit all changes
git add -A .
git commit -m "commit_message: run final nbsphinx version"

# Push the changes
git push -q origin gh-pages



