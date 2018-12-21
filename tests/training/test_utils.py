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
"""Tests for training utility functions"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pytest

from training import utils
from training import settings


class TestUtils(object):

    def test_get_hash_with_status(self):

        class Redis(object):
            def __init__(self, prefix, status):
                self.prefix = prefix
                self.status = status

            def keys(self):
                return ['%s_hash_1' % self.prefix, '%s_hash_2' % self.prefix,
                        'hash_1', 'hash_2', '%s_string_1' % self.prefix]

            def type(self, thing):
                if 'hash' in str(thing).lower():
                    return 'hash'
                return 'string'

            def hget(self, thing, field):
                if field == 'status':
                    return self.status
                return 'other'

        prefix = settings.HASH_PREFIX
        status = 'new'
        redis = Redis(prefix, status)
        rhash = utils.get_hash_with_status(redis, status)
        assert rhash == '%s_hash_1' % prefix
        assert not utils.get_hash_with_status(redis, 'no_status')

        def bad_keys():
            raise ZeroDivisionError

        redis.keys = bad_keys
        rhash = utils.get_hash_with_status(redis, 'status')
        assert rhash is None

    def test_make_notebook(self):
        # test bad input data
        with np.testing.assert_raises(ValueError):
            _ = utils.make_notebook(None)
        # test ImportError raised if deepcell not found
        with np.testing.assert_raises(ImportError):
            _ = utils.make_notebook('random_path')
