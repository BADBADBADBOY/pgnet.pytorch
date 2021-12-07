from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

__dir__ = os.path.dirname(__file__)
sys.path.append(__dir__)
sys.path.append(os.path.join(__dir__, '..'))
from e2e_utils.pgnet_pp_utils import PGNet_PostProcess


class PGPostProcess(object):
    """
    The post process for PGNet.
    """

    def __init__(self, character_dict_path, valid_set, score_thresh, mode,
                 **kwargs):
        self.character_dict_path = character_dict_path
        self.valid_set = valid_set
        self.score_thresh = score_thresh
        self.mode = mode

        # c++ la-nms is faster, but only support python 3.5
        self.is_python35 = False
        if sys.version_info.major == 3 and sys.version_info.minor == 5:
            self.is_python35 = True

    def __call__(self, outs_dict, shape_list):
        post = PGNet_PostProcess(self.character_dict_path, self.valid_set,
                                 self.score_thresh, outs_dict, shape_list)
        if self.mode == 'fast':
            data = post.pg_postprocess_fast()
        else:
            data = post.pg_postprocess_slow()
        return data