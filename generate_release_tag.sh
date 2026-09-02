#!/bin/bash

#
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

set -euo pipefail

error_echo() { echo -e "\033[31mError: $*\033[0m" >&2; }

VERSION_REGEX="^[0-9]+\.[0-9]+\.[0-9]+(dev[0-9]+)?$"

read -p "Version? (provide the next x.y.z or x.y.zdevN format) : " TAG
if [[ ! $TAG =~ $VERSION_REGEX ]]; then
    error_echo "Invalid semver format: $TAG"
    exit 1
fi

if ! git checkout -b "tag-${TAG}"; then
    error_echo "Failed to create tag branch"
    exit 1
fi

PYPROJECT_PATH="./python/pyproject.toml"

cp "${PYPROJECT_PATH}" "${PYPROJECT_PATH}.bak"

if ! sed -i.bak "s/^dynamic = \[\"version\"\]$/version = \"${TAG}\"/" "${PYPROJECT_PATH}"; then
    error_echo "Failed to update version in pyproject.toml"
    exit 1
fi

if ! grep -q "^version = \"${TAG}\"" "${PYPROJECT_PATH}"; then
    error_echo "Version replacement verification failed"
    exit 1
fi

NEW_WHEEL_NAME_FOR_PYLANCE="ve-pylance"
echo "Replacing the project name from pylance to ${NEW_WHEEL_NAME_FOR_PYLANCE}"

if ! sed -i.bak '/^\[project\]$/,/^\[/ s/^name[[:space:]]*=[[:space:]]*"pylance"$/name = "'"${NEW_WHEEL_NAME_FOR_PYLANCE}"'"/' "${PYPROJECT_PATH}"; then
    error_echo "Failed to update project name"
    exit 1
fi

if ! grep -q "^name = \"${NEW_WHEEL_NAME_FOR_PYLANCE}\"" "${PYPROJECT_PATH}"; then
    error_echo "Project name replacement verification failed"
    exit 1
fi

rm -f "${PYPROJECT_PATH}.bak"

if ! git add "${PYPROJECT_PATH}"; then
    error_echo "Git add failed"
    exit 1
fi

if ! git commit -m "release: version ${TAG} 🚀"; then
    error_echo "Git commit failed"
    exit 1
fi

echo "creating git tag : ${TAG}"
if ! git tag "${TAG}"; then
    error_echo "Git tag creation failed"
    exit 1
fi

if ! git push -u origin "tag-${TAG}" "${TAG}"; then
    error_echo "Git push failed"
    exit 1
fi

echo "Successfully pushed new tag: ${TAG} to remote."
