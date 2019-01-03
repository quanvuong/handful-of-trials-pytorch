from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


def get_required_argument(dotmap, key, message, default=None):
    val = dotmap.get(key, default)
    if val is default:
        raise ValueError(message)
    return val
