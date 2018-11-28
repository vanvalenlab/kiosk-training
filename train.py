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
"""The entrypoint for the training job"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import datetime
import logging
import tempfile

from redis import StrictRedis

from training import settings
from training import utils


def initialize_logger(debug_mode=False):
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter('[%(levelname)s]:[%(name)s]: %(message)s')
    console = logging.StreamHandler(stream=sys.stdout)
    console.setFormatter(formatter)

    if debug_mode:
        console.setLevel(logging.DEBUG)
    else:
        console.setLevel(logging.INFO)

    logger.addHandler(console)


if __name__ == '__main__':
    initialize_logger(settings.DEBUG)

    _logger = logging.getLogger(__file__)

    storage_client = utils.get_storage_client(settings.CLOUD_PROVIDER)

    redis = StrictRedis(
        host=settings.REDIS_HOST,
        port=settings.REDIS_PORT,
        decode_responses=True,
        charset='utf-8')

    training_hash = utils.get_hash_with_status(redis, settings.STATUS)

    if training_hash is None:
        # could not find a hash with status == STATUS
        sys.exit(0)

    hash_values = redis.hgetall(training_hash)

    try:
        with tempfile.TemporaryDirectory() as tempdir:
            data_path = hash_values.get('file_name')
            local_path = storage_client.download(data_path, tempdir)

            model_name = '{ts}_{dataset}_{type}_{transform}'.format(
                ts=datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S'),
                dataset=os.path.splitext(os.path.basename(local_path))[0],
                type=hash_values.get('training_type', 'conv'),
                transform=hash_values.get('transform', 'watershed'))

            notebook_path = utils.make_notebook(
                local_path,
                model_name=model_name,
                **hash_values)

            redis.hmset(training_hash, {
                'model': model_name,
                'status': 'training'
            })

            _logger.debug('Updated model %s status to "training"', model_name)

            result = utils.run_notebook(notebook_path)

            redis.expire(training_hash, 10)

            exit_status = 0

    except Exception as err:
        _logger.error('Encountered %s during training: %s',
                      type(err).__name__, err)

        redis.hmset(training_hash, {
            'reason': '{}'.format(err),
            'status': 'failed'
        })
        redis.expire(training_hash, 10)
        exit_status = 1

    _logger.info('Exiting with status: %s', exit_status)
    sys.exit(exit_status)
