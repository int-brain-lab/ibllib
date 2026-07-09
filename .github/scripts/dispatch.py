#!/usr/bin/env python3
"""Runner-side. Submits one blocking Lightning job per matrix leg.

------------------------------------------------------------------------------
COVERALLS BUILD CORRELATION (highest-risk item — read if coverage won't merge)
------------------------------------------------------------------------------
The per-leg `coveralls` report runs INSIDE the Lightning image, which has none
of GitHub's native CI env vars. The finalize step (`coveralls --finish`) runs
on the GH runner, which DOES. For coveralls.io to merge the parallel reports
with the finish, both sides must resolve to the SAME build id. We forward the
GitHub CI context below so the in-image coveralls computes the same build
identity as the runner-side finish. If, after a first run, the parallel
reports and the finish show up as SEPARATE/un-merged builds, align the build
id explicitly (e.g. COVERALLS_SERVICE_NUMBER) on both sides. Verify on Step 1.
------------------------------------------------------------------------------
"""
import os
import re
import sys

from lightning_sdk import Job, Machine, Status


def sanitize(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9-]", "-", s).strip("-").lower()


PY_VERSION = os.environ["PY_VERSION"]
GITHUB_SHA = os.environ["GITHUB_SHA"]
GITHUB_RUN_ID = os.environ["GITHUB_RUN_ID"]
RUN_ATTEMPT = os.environ.get("GITHUB_RUN_ATTEMPT", "1")
GITHUB_REPO = os.environ["GITHUB_REPOSITORY"]      # "owner/repo"
GITHUB_REF = os.environ.get("GITHUB_REF_NAME", "")
REPO_URL = os.environ["REPO_URL"]
TEAMSPACE = os.environ["LIGHTNING_TEAMSPACE"]    # "owner/teamspace"
INTEGRATION_DATA_DIR = os.environ.get("INTEGRATION_DATA_DIR", "/data")
COVERALLS_TOKEN = os.environ["COVERALLS_REPO_TOKEN"]
PR_NUMBER = os.environ.get("PR_NUMBER", "")
JOB_TIMEOUT = int(os.environ.get("JOB_TIMEOUT_SECONDS", "7200"))
# Base image for the Lightning job. python:3.12 verified to ship git/bash/pip.
# uv (installed at runtime) provides the actual test Python via PY_VERSION, so
# this base Python version is only used to bootstrap `pip install uv`.
CI_IMAGE = "python:3.12"

# GITHUB_RUN_ID is only unique per-repo, and one teamspace serves many repos.
repo_slug = sanitize(GITHUB_REPO)
py_slug = PY_VERSION.replace(".", "")
JOB_NAME = f"{repo_slug}-py{py_slug}-{GITHUB_RUN_ID}-{RUN_ATTEMPT}"
# NOTE: verify max job-name length; truncate/hash repo_slug if too long.

owner, _, teamspace = TEAMSPACE.partition("/")

# Bootstrap inside the container (python:3.12 — verified git/bash/pip):
#   guard -> clone @ commit -> hand off to repo-versioned CI scripts.
#
# For pull_request events, GITHUB_SHA is a virtual merge commit
# (refs/pull/<N>/merge) that is not available in a regular clone.
# Fetch the PR merge ref explicitly so the Lightning job can check it out.
if PR_NUMBER:
    checkout_cmd = (
        f"git fetch origin refs/pull/{PR_NUMBER}/merge && "
        "git checkout FETCH_HEAD"
    )
else:
    checkout_cmd = 'git checkout "$GITHUB_SHA"'

command = (
    "set -e && "
    "command -v git >/dev/null || { echo 'ERROR: git not present in image'; exit 127; } && "
    'git clone --no-checkout "$REPO_URL" /workspace/repo && '
    "cd /workspace/repo && "
    f"{checkout_cmd} && "
    "bash scripts/ci/run_tests.sh"
)

print(f"🚀 Submitting {JOB_NAME} (image={CI_IMAGE}, py={PY_VERSION})", flush=True)

job = Job.run(
    name=JOB_NAME,
    image=CI_IMAGE,
    machine=Machine.CPU,
    command=command,
    env={
        # --- runtime / bootstrap ---
        "PY_VERSION": PY_VERSION,
        "REPO_URL": REPO_URL,
        "INTEGRATION_DATA_DIR": INTEGRATION_DATA_DIR,
        # --- Coveralls auth + parallel grouping ---
        "COVERALLS_REPO_TOKEN": COVERALLS_TOKEN,
        "COVERALLS_PARALLEL": "true",
        "COVERALLS_FLAG_NAME": f"py{PY_VERSION}",   # label only; need not be unique
        # --- CI context forwarded for build-id correlation (see header) ---
        "GITHUB_ACTIONS": "true",
        "GITHUB_SHA": GITHUB_SHA,
        "GITHUB_RUN_ID": GITHUB_RUN_ID,
        "GITHUB_REPOSITORY": GITHUB_REPO,
        "GITHUB_REF_NAME": GITHUB_REF,
        **({"GITHUB_PR_NUMBER": PR_NUMBER} if PR_NUMBER else {}),
    },
    path_mappings={"/data": "ibl-brain-wide-map-private"},
    teamspace=teamspace,
    org=owner,
    interruptible=False,
)

try:
    job.wait(interval=15, timeout=JOB_TIMEOUT, stop_on_timeout=True)
except TimeoutError:
    print(f"⏰ Job exceeded {JOB_TIMEOUT}s and was stopped.", flush=True)

print("================ JOB LOGS ================", flush=True)
try:
    print(job.logs, flush=True)
except Exception as e:  # noqa: BLE001
    print(f"⚠️ Could not fetch logs: {e}", flush=True)
print("=========================================", flush=True)

status = job.status
print(f"🏁 Final status: {status}", flush=True)
sys.exit(0 if status == Status.Completed else 1)
