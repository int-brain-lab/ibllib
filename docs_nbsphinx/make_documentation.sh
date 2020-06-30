# Clean up any previous documentation
make clean

# Make the new documentation
make html

#Clean up
python scripts/cleanup.py

# Push to gh-pages
bash scripts/gh_push.sh

# Clean up everything
rm -rf gh-pages

make clean

rm -r _build
