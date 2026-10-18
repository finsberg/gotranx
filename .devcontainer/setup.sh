#!/usr/bin/env bash
set -e # Exit immediately if a command exits with a non-zero status

chsh -s $(which zsh) root || true
if ! command -v starship &> /dev/null; then
  echo "Installing Starship prompt..."
  curl -sS https://starship.rs/install.sh | sh -s -- -y
fi
# apt-get update && export DEBIAN_FRONTEND=noninteractive \
#     && apt-get -y install --no-install-recommends \
#         build-essential ca-certificates coreutils curl environment-modules \
#         gfortran git gpg lsb-release \
#         unzip zip bzip2 xz-utils patch file gnupg2 make tar gzip zstd \
#     && apt-get clean \
#     && rm -rf /var/lib/apt/lists/*
python3 -m pip install --upgrade pip
python3 -m pip install -e .[dev]
pre-commit install
# 3. Print success message
echo "Container setup complete!"
