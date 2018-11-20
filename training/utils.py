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
"""Write a deepcell training notebook and execute it"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import subprocess

import deepcell  # pylint: disable=E0401

from training import settings
from training import storage


logger = logging.getLogger('training.utils')


def get_hash_with_status(redis, status='new'):
    """Iterate over hash values in redis,
    yielding each with a status equal to watch_status
    # Returns: Iterator of all hashes with a valid status
    """
    try:
        keys = redis.keys()
        logger.debug('Found %s redis keys', len(keys))
    except:
        keys = []

    prefix = settings.HASH_PREFIX
    for key in keys:
        # Check if the key is a hash
        if redis.type(key) == 'hash' and str(key).startswith(prefix):
            if redis.hget(key, 'status') == status:
                logger.debug('Found new key: %s', key)
                return key

    logger.error('Could not find a redis hash with status "%s".', status)
    return None


def get_storage_client(cloud_provider):
    """Returns the Storage Client appropriate for the cloud provider
    # Arguments:
        cloud_provider: Indicates which cloud platform (AWS vs GKE)
    # Returns:
        storage_client: Client for interacting with the cloud.
    """
    if cloud_provider == 'aws':
        storage_client = storage.S3Storage(settings.AWS_S3_BUCKET)
    elif cloud_provider == 'gke':
        storage_client = storage.GoogleStorage(settings.GCLOUD_STORAGE_BUCKET)
    else:
        errmsg = 'Bad value for CLOUD_PROVIDER: %s'
        logger.error(errmsg, cloud_provider)
        raise ValueError(errmsg % cloud_provider)
    return storage_client


def make_notebook(data, **kwargs):
    """Use the training parameters to create a deepcell training notebook
    # Arguments:
        data: the path to the properly formatted directory of data
        kwargs: named key/value pairs from the redis hash
    """
    if not data:
        raise ValueError('`data` is required to download training data')

    try:
        notebook_path = deepcell.notebooks.train.make_notebook(
            data,
            model_name=kwargs.get('model_name'),
            train_type=kwargs.get('training_type', 'conv'),
            field_size=int(kwargs.get('field', 61)),
            ndim=int(kwargs.get('ndim', 2)),
            optimizer=kwargs.get('optimizer', 'sgd'),
            skips=int(kwargs.get('skips', 0)),
            epochs=int(kwargs.get('epochs', 10)),
            normalization=kwargs.get('normalization', 'std'),
            transform=kwargs.get('transform', 'watershed'),
            distance_bins=int(kwargs.get('distance_bins', 4)),
            erosion_width=int(kwargs.get('erosion_width', 0)),
            dilation_radius=int(kwargs.get('dilation_radius', 1)),
            output_dir=settings.NOTEBOOK_DIR,
            export_dir=settings.EXPORT_DIR,
            log_dir=settings.LOG_DIR)

    except Exception as err:
        logger.error('Failed to write training notebook: %s', err)
        raise err

    logger.info('Saved notebook to %s', notebook_path)
    return notebook_path


def run_notebook(notebook_path):
    """Create a training notebook with deepcell and run it
    # Arguments:
        notebook_path: path to generated training notebook
    """
    cmd = [
        'jupyter', 'nbconvert',
        '--to', 'notebook',
        '--ExecutePreprocessor.timeout=-1',
        '--execute', notebook_path
    ]

    logger.debug('Executing subprocess: `%s`', ' '.join(cmd))

    try:
        output = subprocess.check_output(cmd, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as err:
        logger.error('Encountered error while running the notebook: %s', err)
        raise Exception('{} : {}'.format(err, err.stdout.decode('utf-8')))

    output = output.decode('utf-8')  # convert output bytes to string
    logger.debug('Subprocess Output: %s', output)

    return output
