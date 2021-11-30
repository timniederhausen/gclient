#!/usr/bin/env bash
# Copyright (c) 2009 The Chromium Authors. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.

base_dir=$(dirname "$0")

PYTHONDONTWRITEBYTECODE=1 exec python3 "$base_dir/gclient.py" "$@"
