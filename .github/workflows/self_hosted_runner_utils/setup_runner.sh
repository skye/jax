#!/bin/bash

# Copyright 2022 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# More or less copied from
# https://github.com/iree-org/iree/tree/main/build_tools/github_actions/runner/config

set -ex

if [ "$#" -ne 3 ]; then
  echo "Usage: setup_runner.sh <runner name> <tags> <github token>"
fi

runner_name=$1
runner_tags=$2
runner_token=$3

# Secret fourth argument for setting the repo URL. Useful for testing with forks.
jax_repo_url=$4
if [ -z "${jax_repo_url}" ]; then
  jax_repo_url="https://github.com/google/jax"
fi

# Create `runner` user. This user won't have sudo access unless you ssh into the
# GCP VM as `runner` using gcloud. Don't do that!
sudo useradd runner -m

# Find the latest actions-runner download
# e.g. https://github.com/actions/runner/releases/download/v2.298.2/actions-runner-linux-x64-2.298.2.tar.gz
actions_runner_download_regexp='https://github.com/actions/runner/releases/'\
'download/v[0-9.]\+/actions-runner-linux-x64-[0-9.]\+\.tar\.gz'
# Use `head -n 1` because there are multiple instances of the same URL
actions_runner_download=$(
  curl -s -X GET 'https://api.github.com/repos/actions/runner/releases/latest' |
    grep -o $actions_runner_download_regexp |
    head -n 1)
echo "actions_runner_download: $actions_runner_download"

# Run the rest of the setup as `runner`
sudo -i -u runner bash -ex <<EOF

cd ~/

git clone $jax_repo_url

# Based on https://github.com/google/jax/settings/actions/runners/new
mkdir actions-runner && cd actions-runner
curl -o actions-runner-linux-x64.tar.gz -L $actions_runner_download
tar xzf ./actions-runner-linux-x64.tar.gz

# Register the runner with Github
./config.sh --unattended \
--url $jax_repo_url \
--labels $runner_tags \
--token $runner_token \
--name $runner_name

# Setup pre-job hook
cat ~/jax/.github/workflows/self_hosted_runner_utils/runner.env | envsubst >> ~/actions-runner/.env

# Setup Github Actions Runner to automatically start on reboot (e.g. due to TPU
# VM maintenance events)
echo "@reboot $HOME/jax/.github/workflows/self_hosted_runner_utils/start_github_runner.sh" | crontab -

EOF
