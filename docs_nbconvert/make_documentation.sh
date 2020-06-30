
# Copy the notebooks to a new directory
cp -r notebooks notebooks_processed

# Clean up any previous documentation
make clean

# Make the new documentation
make html

# Push to gh-pages
bash scripts/gh_push.sh

# Clean up everything
rm -r notebooks_processed
rm -rf gh-pages
rm -r rst-notebooks

make clean

rm -r _build
