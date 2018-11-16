# Copyright 2016-2018 The Van Valen Lab at the California Institute of
# Technology (Caltech), with support from the Paul Allen Family Foundation,
# Google, & National Institutes of Health (NIH) under Grant U24CA224309-01.
# All rights reserved.
#
# Licensed under a modified Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.github.com/vanvalenlab/kiosk-training/LICENSE
#
# The Work provided may be used for non-commercial academic purposes only.
# For any other use of the Work, including commercial use, please contact:
# vanvalenlab@gmail.com
#
# Neither the name of Caltech nor the names of its contributors may be used
# to endorse or promote products derived from this software without specific
# prior written permission.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Constants and environment variables"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from decouple import config


# remove leading/trailing "/"s from cloud bucket folder names
_strip = lambda x: '/'.join(y for y in x.split('/') if y)

# Debug Mode
DEBUG = config('DEBUG', cast=bool, default=False)

# Top level folder to save logs inside bucket
LOG_PREFIX = _strip(config('LOG_PREFIX', cast=str, default='tensorboard_logs'))

# Top level folder to save exported models inside bucket
EXPORT_PREFIX = _strip(config('EXPORT_PREFIX', cast=str, default='models'))

# Hash Prefix - filter out training jobs
HASH_PREFIX = _strip(config('HASH_PREFIX', cast=str, default='train'))

# Redis client connection
REDIS_HOST = config('REDIS_HOST', default='redis-master')
REDIS_PORT = config('REDIS_PORT', default=6379, cast=int)

# Status of hashes marked for training
STATUS = config('STATUS', default='new')

# Cloud storage
CLOUD_PROVIDER = config('CLOUD_PROVIDER', cast=str, default='aws').lower()

# AWS credentials
AWS_REGION = config('AWS_REGION', default='us-east-1')
AWS_S3_BUCKET = config('AWS_S3_BUCKET', default='default-bucket')
AWS_ACCESS_KEY_ID = config('AWS_ACCESS_KEY_ID', default='specify_me')
AWS_SECRET_ACCESS_KEY = config('AWS_SECRET_ACCESS_KEY', default='specify_me')

# Google credentials
GCLOUD_STORAGE_BUCKET = config('GKE_BUCKET', default='default-bucket')

# Application directories
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOWNLOAD_DIR = os.path.join(ROOT_DIR, 'download')
NOTEBOOK_DIR = os.path.join(ROOT_DIR, 'notebooks')

# Overwrite directories with environment variabls
DOWNLOAD_DIR = config('DOWNLOAD_DIR', default=DOWNLOAD_DIR)
NOTEBOOK_DIR = config('NOTEBOOK_DIR', default=NOTEBOOK_DIR)

LOG_DIR = '{protocol}://{bucket}/{folder}'.format(
    protocol='s3' if CLOUD_PROVIDER == 'aws' else 'gs',
    bucket=AWS_S3_BUCKET if CLOUD_PROVIDER == 'aws' else GCLOUD_STORAGE_BUCKET,
    folder=LOG_PREFIX)

EXPORT_DIR = '{protocol}://{bucket}/{folder}'.format(
    protocol='s3' if CLOUD_PROVIDER == 'aws' else 'gs',
    bucket=AWS_S3_BUCKET if CLOUD_PROVIDER == 'aws' else GCLOUD_STORAGE_BUCKET,
    folder=EXPORT_PREFIX)
