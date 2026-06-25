#!/usr/bin/env bash
# Entry point for the Lightning CI job.
# cwd is repo root (set by dispatch command). Repo already cloned/checked out.
set -uo pipefail   # NOT -e: we want coveralls to report even on test failure

source scripts/ci/prep_env.sh
source /workspace/venv/bin/activate   # ensure venv active in this shell; should be unnecessary

echo "Running full test suite under coverage (PY_VERSION=${PY_VERSION})"

export ONE_SAVE_ON_DELETE=false  # prevent ONE from writing cache tables to disk
export NO_PROGRESSBARS=1  # supress progress bars in test output (for cleaner logs)
export INTEGRATION_DATA_WRITABLE=0  # s3 data connection is read only, so integration tests must not attempt to write to it

# INTEGRATION_DATA_DIR is set by CI (Step 3) -> integration tests run.
# In Step 1 it's unset -> integration tests auto-skip, unit tests run.
set +e
coverage run --rcfile "scripts/ci/.coveragerc" -m unittest discover -t . -p "test_*.py"
TEST_EXIT=$?
set -e

coverage report || true

echo "Reporting partial coverage to Coveralls (parallel)"
coveralls --service=github-actions || echo "coveralls upload failed (non-fatal)"

exit "${TEST_EXIT}"