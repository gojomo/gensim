#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author: Gensim Contributors
# Copyright (C) 2020 RaRe Technologies s.r.o.
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

"""
TokenCorpus

TK: Usage examples
"""

import logging
import sys
from timeit import default_timer
from collections import Counter
import itertools

from gensim.utils import keep_vocab_item, pickle, GENSIM_PICKLE_PROTOCOL


logger = logging.getLogger(__name__)


class TokenSurvey(object):
    def __init__(
            self, corpus_iterable=None, raw_tally=None, corpus_count=0,
            progress_per=10000, prune_at_size=None, prune_until=0.5, prune_callbacks=(),
            items='tokens',
    ):
        """
        TokenSurvey is a utility for surveying a corpus, to discover its vocabulary/token-frequencies, etc.

        It's used internally by the `.build_vocab()` steps of Word2Vec, Doc2Vec, & FastText, but can also
        be used outside those methods. For example it could be used to scan & save the token statistics of a
        corpus for re-use in multiple future models (saving that time-consuming step for as long as the corpus
        doesn't change.

        Parameters
        ----------
        corpus_iterable : iterable of (list of tokens), optional
            Either iterable-of-lists-of-tokens
        raw_tally : collections.Counter of (token -> int count), optional
            Initial counts from another source, if available
        corpus_count : int, optional
            Initial count of texts to assume, if raw_tally provided
        prune_at_size : int, optional
            Limits the RAM during vocabulary building; if there are more unique tokens than this, then prune
            the infrequent ones before continuing. Every 10 million token types need about 1GB of RAM.
        prune_until : float, optional
            If the `prune_at_size` is reached, callbacks run then least-frequent tokens discarded until
            the number of remaining keys is this proportion of the full `prune_at_size`. Default is 0.5.
        prune_callbacks : sequence of single-argument functions, optional
            If provided, these will be called one-at-a-time, with the TokenSurvey itself as the
            argument, both whenever the `prune_at_size` threshold is reached, and then finally at the
            end of the survey, thus allowing any particular discard/retention policy to be applied.
        items: str, optional
            A display name for the kind of tokens surveyed. Default is 'tokens'.
        """
        self.prune_at_size = prune_at_size or sys.maxsize
        self.prune_until = prune_until
        self.tenured_tokens = set()
        self.raw_tally = raw_tally or Counter()
        self.corpus_count = corpus_count
        self.total_tokens = sum(self.raw_tally.values())
        self.max_raw_int = -1  # useful for detecting plain-int tags
        self.prune_count = 0  # total number of keys (w/ dups) pruned
        self.prune_floor = 0  # highest token count ever pruned
        self.prune_total = 0  # total words in corpus disregarded via prunes
        self.items = items

        if corpus_iterable:
            self.update_with(corpus_iterable, prune_callbacks=prune_callbacks, progress_per=progress_per)

    def update_with(self, corpus_iterable, prune_callbacks=(), progress_per=10000, trim_rule=None):
        total_tokens = 0

        if trim_rule:
            prune_callbacks = prune_callbacks + (trim_rule_callback(trim_rule),)
        effective_prune_at_size = self.prune_at_size

        interval_start = default_timer() - 0.00001  # guard against next sample being identical
        interval_count = 0
        item_count = 0
        for tokens in corpus_iterable:
            if item_count == 0:
                # ensure 1st item isn't a plain string, indicating a common error
                if isinstance(tokens, str):
                    logger.warning(
                        f"Each 'corpus_iterable' item should be a list of {self.items} (usually unicode strings). "
                        f"First item here is instead plain {type(tokens)}.",
                    )

            self.raw_tally.update(tokens)
            item_count += 1
            total_tokens += len(tokens)
            reached_size = len(self.raw_tally)

            if item_count % progress_per == 0:
                interval_rate = (total_tokens - interval_count) / (default_timer() - interval_start)
                logger.info(
                    f"PROGRESS: at text #{item_count}, processed {total_tokens} {self.items} ({interval_rate}/s), "
                    f"{reached_size} unique {self.items}"
                )
                interval_start = default_timer()
                interval_count = total_tokens

            if reached_size >= effective_prune_at_size:
                logger.info(f"survey reached size {reached_size}, beginning callback & threshold token prune")

                if prune_callbacks:
                    for callback in prune_callbacks:
                        callback(self)
                    logger.info(f"prune_callbacks removed {reached_size - len(self.raw_tally)} tokens")

                target_size = int(self.prune_until * self.prune_at_size)
                before_infrequent_pruning = len(self.raw_tally)
                highest_pruned_count = 0
                infrequent_pruned = 0
                infrequent_candidates = self.raw_tally.most_common()
                infrequent_candidates.reverse()
                for key, count in infrequent_candidates:
                    if len(self.raw_tally) <= target_size:
                        # reached target, report & continue
                        break
                    # TODO: option to write pruned tallies to file, for later precise-retally?
                    # TODO: can any trick ensure at least one example of a pruned key is remembered?
                    highest_pruned_count = count
                    infrequent_pruned += 1
                    self.prune_token(key)
                else:
                    # serious problem: could never reach target_size
                    # disable frequency-pruning with loud warning
                    logger.severe(
                        f"infrequent token pruning can't reach {target_size}; still {len(self.raw_tally)} keys. "
                        f"Maybe too many tenured tokens ({len(self.tenured_tokens)})? Doubling prune threshold to "
                        f"risk memory-error rather than fail here."
                    )
                    effective_prune_at_size *= 2
                logger.info(
                    f"pruned {infrequent_pruned} least-frequent tokens with counts up to {highest_pruned_count} "
                    f"(before {before_infrequent_pruning}, after {len(self.raw_tally)}",
                )

        # finished, but warn again if prune_at_size couldn't be respected
        if effective_prune_at_size > self.prune_at_size:
            logger.severe(
                f"Infrequent token pruning couldn't satisfy prune targets; final size {len(self.raw_tally)} "
                f"may have outgrown requested purge_at_size. Check results and consider adjusting parameters."
            )

        # application of callbacks at finish (so guaranteed to run at least once)
        if prune_callbacks:
            before_size = len(self.raw_tally)
            for callback in prune_callbacks:
                callback(self)
            pruned = before_size - len(self.raw_tally)
            logger.info(f"end-of-batch prune_callbacks removed {pruned} tokens")

        self.corpus_count += item_count
        self.total_tokens += total_tokens
        self.max_raw_int = max(itertools.chain((-1,), (t for t in self.raw_tally.keys() if isinstance(t, int))))

    def tenure_token(self, token):
        """Make a token immune to low-frequency-based pruning."""
        self.tenured_tokens.add(token)

    def prune_token(self, token):
        """Remove named token"""
        count = self.raw_tally.pop(token)
        self.prune_total += count
        self.prune_floor = max(self.prune_floor, count)
        self.prune_count += 1

    def combine_survey(self, other_survey):
        """Tally another survey's values into this, so this becomes a combined survey."""
        self.raw_tally.update(other_survey.raw_tally)
        self.corpus_count += other_survey.corpus_count
        self.total_tokens = sum(self.raw_tally.values())

        self.prune_count += other_survey.prune_count
        self.prune_floor = max(self.prune_floor, other_survey.prune_floor)
        self.prune_total += other_survey.prune_total

    def __len__(self):
        return len(self.raw_tally)

    def int_centric_items(self):
        """Report tokens/counts of self.raw_tally with all plain-int tokens first,
        padded with non-present ints, then any remaining items."""

        non_int_tokens = sum(1 for t in self.raw_tally.keys() if not isinstance(t, int))
        if self.max_raw_int > len(self.raw_tally):
            logger.warn(
                f"TokenSurvey of only {len(self.raw_tally)} unique tokens is reporting "
                f"larger highest plain-int token {self.max_raw_int}. This is usually an error."
            )
        return_list = [None] * ((self.max_raw_int + 1) + non_int_tokens)
        for i in range(self.max_raw_int + 1):
            return_list[i] = (i, self.raw_tally.get(i, 0))
        for i, token in enumerate(k for k in self.raw_tally.keys() if not isinstance(k, int)):
            return_list[self.max_raw_int + 1 + i] = (token, self.raw_tally[token])
        return return_list

    def save(self, filename):
        """Convenience imperative pickling"""
        with open(filename) as f:
            pickle.dump(self, f, protocol=GENSIM_PICKLE_PROTOCOL)


def trim_rule_callback(trim_rule):
    """Create callback for a legacy trim_rule"""
    def apply_trim_rule(token_survey):
        for token, count in token_survey.raw_tally.most_common():
            if not keep_vocab_item(token, count, 0, trim_rule):
                token_survey.prune_token(token)
    return apply_trim_rule
