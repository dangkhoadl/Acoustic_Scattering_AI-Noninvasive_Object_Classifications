#!/usr/bin/env bash
#
set -u           # Detect unset variable
set -e           # Exit on non-zero return code from any command
set -o pipefail  # Exit if any of the commands in the pipeline will
                 # return non-zero return code


python local/download_pretrained.py