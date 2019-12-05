#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author: Shiva Manne <s.manne@rare-technologies.com>
# Copyright (C) 2019 RaRe Technologies s.r.o.

"""General functions used for any2vec models.

One of the goals of this module is to provide an abstraction over the Cython
extensions for FastText.  If they are not available, then the module substitutes
slower Python versions in their place.

Another related set of FastText functionality is computing ngrams for a word.
The :py:func:`compute_ngrams` and :py:func:`compute_ngrams_bytes` hashes achieve that.

Closely related is the functionality for hashing ngrams, implemented by the
:py:func:`ft_hash` and :py:func:`ft_hash_broken` functions.
The module exposes "working" and "broken" hash functions in order to maintain
backwards compatibility with older versions of Gensim.

For compatibility with older Gensim, use :py:func:`compute_ngrams` and
:py:func:`ft_hash_broken` to has each ngram.  For compatibility with the
current Facebook implementation, use :py:func:`compute_ngrams_bytes` and
:py:func:`ft_hash_bytes`.

"""

import logging
from gensim import utils
import gensim.models.keyedvectors

from numpy import zeros, dtype, float32 as REAL, ascontiguousarray, frombuffer

from six.moves import range
from six import iteritems, PY2

logger = logging.getLogger(__name__)


#
# UTF-8 bytes that begin with 10 are subsequent bytes of a multi-byte sequence,
# as opposed to a new character.
#
_MB_MASK = 0xC0
_MB_START = 0x80


def _byte_to_int_py3(b):
    return b


def _byte_to_int_py2(b):
    return ord(b)


_byte_to_int = _byte_to_int_py2 if PY2 else _byte_to_int_py3


def _is_utf8_continue(b):
    return _byte_to_int(b) & _MB_MASK == _MB_START


try:
    from gensim.models._utils_any2vec import (
        compute_ngrams,
        compute_ngrams_bytes,
        ft_hash_broken,
        ft_hash_bytes,
    )
except ImportError:
    raise utils.NO_CYTHON


def ft_ngram_hashes(word, minn, maxn, num_buckets, fb_compatible=True):
    """Calculate the ngrams of the word and hash them.

    Parameters
    ----------
    word : str
        The word to calculate ngram hashes for.
    minn : int
        Minimum ngram length
    maxn : int
        Maximum ngram length
    num_buckets : int
        The number of buckets
    fb_compatible : boolean, optional
        True for compatibility with the Facebook implementation.
        False for compatibility with the old Gensim implementation.

    Returns
    -------
        A list of hashes (integers), one per each detected ngram.

    """
    if fb_compatible:
        encoded_ngrams = compute_ngrams_bytes(word, minn, maxn)
        hashes = [ft_hash_bytes(n) % num_buckets for n in encoded_ngrams]
    else:
        text_ngrams = compute_ngrams(word, minn, maxn)
        hashes = [ft_hash_broken(n) % num_buckets for n in text_ngrams]
    return hashes


