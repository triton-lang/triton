#!/bin/sh

# Copyright (C) 2024 Red Hat, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

username=""
userid=""

usage() {
  cat >&2 <<EOF
Usage: $0
   -u | --user <username>
   -g | --gid <userid>
EOF
  exit 1
}

# Parse command-line arguments
args=$(getopt -o u:g: --long user:,gid: -n "$0" -- "$@")
if [ $? -ne 0 ]; then usage; fi

eval set -- "$args"
while [ $# -gt 0 ]; do
  case $1 in
    -h | --help)
      usage
      ;;
    -u | --user)
      username=$2
      shift 2
      ;;
    -g | --gid)
      userid=$2
      shift 2
      ;;
    --)
      shift
      break
      ;;
    *)
      echo "Unsupported option: $1" >&2
      usage
      ;;
  esac
done

# Validate required parameters
if [ -z "$username" ] || [ -z "$userid" ]; then
  echo "Error: --user and --gid are required." >&2
  usage
fi

USER_NAME="$username"
USER_UID="$userid"
USER_GID="$USER_UID"
HOME_DIR="/home/$USER_NAME"

# Exit if the user is root
if [ "$USER_NAME" = "root" ]; then
  exit 0
fi

if ! [ $(getent group $USER_NAME) ]; then
  groupadd --gid $USER_GID $USER_NAME
fi

if ! [ $(getent passwd $USER_NAME) ]; then
  useradd --uid $USER_UID --gid $USER_GID -m $USER_NAME
fi

# Ensure $HOME exists when starting
if [ ! -d "${HOME}" ]; then
  mkdir -p "${HOME}"
fi

# Add current (arbitrary) user to /etc/passwd and /etc/group
if [ -w /etc/passwd ]; then
  echo "${USER_NAME:-user}:x:$(id -u):0:${USER_NAME:-user} user:${HOME}:/bin/bash" >> /etc/passwd
  echo "${USER_NAME:-user}:x:$(id -u):" >> /etc/group
fi

# Fix up permissions
chown $USER_NAME:$USER_GID -R /home/$USER_NAME
chown $USER_NAME:$USER_GID -R /opt
mkdir -p /run/user/$USER_UID
chown $USER_NAME:$USER_GID /run/user/$USER_UID
