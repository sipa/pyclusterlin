# Copyright (c) 2025 Pieter Wuille
# Distributed under the MIT software license, see the accompanying
# file LICENSE or http://www.opensource.org/licenses/mit-license.php.

"""Python reimplementation of the spanning-forest linearization algorithm."""

import json
import random
import heapq
import unittest
from collections.abc import Callable
from pathlib import Path
from typing import Any
from depgraph import (SetInfo, DepGraph, DepGraphFormatter, is_topological, by_goodness,
                      FeeFracByGoodness)

class SpanningForestState:
    """Data structure representing the state of the SFL algorithm."""

    __slots__ = ("_rng", "_parents", "_children", "_txchunk", "_active_deps", "_transactions",
                 "_depgraph")

    def __init__(self, depgraph: DepGraph, rng_seed: int) -> None:
        """Construct an SFL state object, with every transaction in its own singleton chunk."""
        # Internal RNG.
        self._rng = random.Random(rng_seed)
        # The transactions in the cluster, randomly permuted.
        self._transactions = list(depgraph.positions())
        self._rng.shuffle(self._transactions)
        # For each transaction, the parents (immutable).
        self._parents = {tx: depgraph.reduced_parents(tx) for tx in depgraph.positions()}
        # For each transaction, the children (immutable).
        self._children = {tx: depgraph.reduced_children(tx) for tx in depgraph.positions()}
        # Chunk for every transaction. Note that the SetInfo object is shared among all
        # transactions in a given chunk.
        self._txchunk: dict[int, SetInfo] = {}
        # Information about all its active dependencies (initially, none). It stores the SetInfo
        # of the would-be top chunk that would be created by deactivating said dependency.
        self._active_deps: dict[tuple[int, int], SetInfo] = {}
        # Store the depgraph this is for.
        self._depgraph = depgraph
        # Create a chunk for every transaction.
        for tx in depgraph.positions():
            self._txchunk[tx] = SetInfo.make_singleton(depgraph, tx)

    def _update_deps(self, txn: set[int], query: int, dep_adjust: SetInfo, subtract: bool) -> None:
        """Update all dependencies between transactions in txn that have query in their top set,
           adding/removing dep_adjust from the top set."""
        for (par, chl), dep in self._active_deps.items():
            if query in dep.transactions and par in txn and chl in txn:
                if subtract:
                    dep -= dep_adjust
                else:
                    dep += dep_adjust

    def _update_chunk(self, chunk: SetInfo) -> None:
        """Update all transactions in chunk to have chunk as their chunk."""
        for tx in chunk.transactions:
            self._txchunk[tx] = chunk

    def _activate(self, par: int, chl: int) -> SetInfo:
        """Activate the dependency chl->par, and return the new chunk."""
        # The dependency cannot be active already.
        assert (par, chl) not in self._active_deps
        # Get the parent and child chunk's information.
        top_chunk = self._txchunk[par]
        bottom_chunk = self._txchunk[chl]
        assert top_chunk.transactions.isdisjoint(bottom_chunk.transactions)
        # We will reuse the bottom chunk for the new combined chunk.
        self._update_deps(top_chunk.transactions, par, bottom_chunk, False)
        self._update_deps(bottom_chunk.transactions, chl, top_chunk, False)
        # Mark the dependency active, and merge the top chunk into the bottom chunk.
        self._active_deps[(par, chl)] = top_chunk
        bottom_chunk += top_chunk
        self._update_chunk(bottom_chunk)
        return bottom_chunk

    def _deactivate(self, par: int, chl: int) -> tuple[SetInfo, SetInfo]:
        """Deactivate the dependency chl->par, and return the corresponding chunks."""
        # The dependency must be active already.
        assert (par, chl) in self._active_deps
        # Get the dependency's information (its would-be top chunk), and deactivate it.
        top_chunk = self._active_deps.pop((par, chl))
        # Get the old chunk's information, which will become the bottom chunk.
        bottom_chunk = self._txchunk[par]
        bottom_chunk -= top_chunk
        # Update the transactions and dependencies.
        self._update_deps(top_chunk.transactions, par, bottom_chunk, True)
        self._update_deps(bottom_chunk.transactions, chl, top_chunk, True)
        self._update_chunk(top_chunk)
        self._update_chunk(bottom_chunk)
        return top_chunk, bottom_chunk

    def _merge_chunks(self, top_chunk: SetInfo, bottom_chunk: SetInfo) -> SetInfo | None:
        """Activate a random dependency from the bottom chunk to the top chunk. Returns the merged
           chunk, or None if no dependencies between them exist."""
        # Gather the dependencies between the two chunks.
        candidate_deps: list[tuple[int, int]] = []
        for tx in top_chunk.transactions:
            for child in self._children[tx] & bottom_chunk.transactions:
                candidate_deps.append((tx, child))
        # If none are found, fail.
        if not candidate_deps:
            return None
        # Pick an index into those dependencies.
        par, chl = self._rng.choice(candidate_deps)
        # Activate.
        return self._activate(par, chl)

    def _get_reachable(self, start: set[int], downward: bool) -> set[int]:
        """Get the set of transactions reachable from start in upward or downward direction."""
        ret: set[int] = set()
        for tx in start:
            ret |= self._children[tx] if downward else self._parents[tx]
        return ret - start

    def _merge_step(self, chunk: SetInfo, downward: bool) -> SetInfo | None:
        """Perform an upward or downward merge step, on the specified chunk. Returns the merged
           chunk, or None if no merge took place."""
        # The candidate chunks to merge with.
        candidates: list[SetInfo] = []
        # The feerates to compare with. Initially, this is equal to the chunk's own feerate. It is
        # updated to be the feerate of candidates whenever any are found.
        candidate_feerate = chunk.feerate
        # Explore chunks that can be reached from chunk_txn, with appropriate feerate.
        todo = self._get_reachable(chunk.transactions, downward)
        while todo:
            new_chunk = self._txchunk[next(iter(todo))]
            todo -= new_chunk.transactions
            # Compare feerate of new chunk with existing candidate(s), if any.
            comp = new_chunk.feerate.compare_feerate(candidate_feerate)
            comp = comp if downward else -comp
            # Ignore if this is worse than candidate_feerate.
            if comp < 0:
                continue
            # Replace candidates if this is strictly better than existing candidates.
            if comp > 0:
                candidate_feerate = new_chunk.feerate
                candidates = []
            candidates.append(new_chunk)
        # If no candidates exist, don't do anything.
        if not candidates:
            return None
        # Pick a random candidate.
        merge_chunk = self._rng.choice(candidates)
        # Merge the chunks.
        if downward:
            merge_chunk, chunk = chunk, merge_chunk
        return self._merge_chunks(merge_chunk, chunk)

    def _merge_sequence(self, tx: int, downward: bool) -> SetInfo:
        """Perform a merge sequence in the specified direction starting with tx, and return the
           resulting chunk (regardless of whether a merge tool place)."""
        # Find the specified transaction's chunk.
        chunk = self._txchunk[tx]
        # Perform merge steps as long as there are any to perform on chunk.
        while True:
            if (result := self._merge_step(chunk, downward)) is None:
                return chunk
            chunk = result

    def load_linearization(self, lin: list[int]) -> None:
        """Load an existing linearization. If it is topological, the SFL state will be
           topological. If not, make_topological() still needs to be called."""
        for tx in lin:
            self._merge_sequence(tx, False)

    def make_topological(self) -> None:
        """Make the graph topological."""
        # Iterate while there exist potentially suboptimal chunks.
        optimal: set[int] = set()
        while len(optimal) != len(self._transactions):
            for tx in self._transactions:
                if tx in optimal:
                    continue
                chunk = self._txchunk[tx]
                # Mark all transactions of this chunk as optimal, since we will be processing them.
                # If a merge occurs, the resulting transactions will be made suboptimal again
                # below.
                optimal |= chunk.transactions
                # Try a merge step, in both directions, in random order. If one succeeds, mark the
                # resulting chunk as suboptimal and continue with another chunk.
                flip = self._rng.getrandbits(1)
                for direction in range(2):
                    if (result := self._merge_step(chunk, flip != direction)) is not None:
                        optimal -= result.transactions
                        break

    def optimize(self) -> None:
        """Make the SFL state optimal (i.e., the output of get_linearization() will be optimal
           in the convexified feerate diagram sense). Requires that the state is topological."""
        # The set of transactions which belong to optimal chunks.
        optimal: set[int] = set()
        # Loop while there are non-optimal chunks.
        while len(optimal) != len(self._transactions):
            for loop_tx in self._transactions:
                # Find a non-optimal chunk, and start processing it.
                if loop_tx in optimal:
                    continue
                chunk = self._txchunk[loop_tx]
                optimal |= chunk.transactions
                # Find the set of active dependencies within the chunk whose top set has higher
                # feerate than the entire chunk (and thus higher feerate than the bottom set).
                candidate_deps: list[tuple[int, int]] = []
                for tx in chunk.transactions:
                    for par in self._parents[tx]:
                        if (dep_data := self._active_deps.get((par, tx))) is None:
                            continue
                        if dep_data.feerate.compare_feerate(chunk.feerate) > 0:
                            candidate_deps.append((par, tx))
                # If any is found, pick a random dependency from that list, deactivate it, merge
                # the resulting chunks to make the state topological again, and then mark the
                # result of those merges suboptimal again.
                if candidate_deps:
                    par, chl = self._rng.choice(candidate_deps)
                    self._deactivate(par, chl)
                    optimal -= self._merge_sequence(par, False).transactions
                    optimal -= self._merge_sequence(chl, True).transactions

    def minimize(self) -> None:
        """Make the SFL state minimal (i.e., all chunks will be split into their smallest
           equal-feerate components. Requires that the state is optimal."""
        # For every transaction, what the pivot in its chunk is, and the direction (False = try
        # to find a way to split the chunk with the pivot in the top part, True = same with pivot
        # in the bottom part).
        txinfo: dict[int, tuple[int, bool]] = {}
        # The set of transactions which belong to minimal chunks.
        optimal: set[int] = set()
        # Loop while there are non-minimal chunks.
        while len(optimal) != len(self._transactions):
            for loop_tx in self._transactions:
                # Find a non-minimal chunk and start processing it.
                if loop_tx in optimal:
                    continue
                chunk = self._txchunk[loop_tx]
                # If no pivot is assigned yet for this chunk, pick loop_tx as pivot, and start
                # with trying to move the pivot up.
                if loop_tx not in txinfo:
                    for tx in chunk.transactions:
                        txinfo[tx] = (loop_tx, False)
                pivot, direction = txinfo[loop_tx]
                # Build a list of active dependencies within the chunk whose top set has equal
                # feerate to the bottom set, and has the pivot in the expected place (top or
                # bottom, depending on direction).
                candidate_deps: list[tuple[int, int]] = []
                for tx in chunk.transactions:
                    for par in self._parents[tx]:
                        if (dep_data := self._active_deps.get((par, tx))) is None:
                            continue
                        if chunk.feerate.compare_feerate(dep_data.feerate) <= 0:
                            if (pivot in dep_data.transactions) == direction:
                                candidate_deps.append((par, tx))
                # If any is found, pick a random dependency from the list.
                if candidate_deps:
                    par, chl = self._rng.choice(candidate_deps)
                    # Deactivate it, splitting the chunk into a top and bottom part.
                    self._deactivate(par, chl)
                    # See if we need to merge the bottom part back with the top part. No other
                    # merges are possible, as the state is already optimal.
                    merged = self._merge_chunks(self._txchunk[chl], self._txchunk[par])
                    if merged is None:
                        # If merging fails, because there is no dependency from bottom on top, we
                        # succeeded in permanently splitting the chunk. Reset the txinfo, so we can
                        # continue with attempting to split the components further.
                        for tx in self._txchunk[par].transactions | self._txchunk[chl].transactions:
                            del txinfo[tx]
                    continue
                if direction:
                    # If such dependency is found, a split is not possible. If direction was True
                    # already, we have tried both directions and the chunk is minimal.
                    optimal |= chunk.transactions
                else:
                    # If not, retry with direction=True.
                    for tx in chunk.transactions:
                        txinfo[tx] = (pivot, True)

    def get_linearization(self, fallback_key: Callable[[int], int]=lambda x: x) -> list[int]:
        """Produce a linearization. Requires that the SFL state is topological."""
        # A heap of chunks which have no unmet dependencies, as (feerate, max_fallback, chunk_rep)
        # tuples.
        ready_chunks: list[tuple[FeeFracByGoodness, int, int]] = []
        # A dict of first_tx_in_chunk -> unmet dependencies.
        chunk_deps: dict[int, int] = {}
        ret: list[int] = []
        # Compute for each chunk how many out-of-chunk unmet dependencies it has, and add those
        # with none to the ready_chunks heap, sorted by decreasing feerate.
        done_tx: set[int] = set()
        for chunk in self._txchunk.values():
            # Make sure not to process the same chunk multiple times.
            if not chunk.transactions.isdisjoint(done_tx):
                continue
            done_tx |= chunk.transactions
            # Count out-of-chunk dependencies this chunk has.
            deps = 0
            for tx in chunk.transactions:
                deps += len(self._parents[tx] - chunk.transactions)
            # Store and if no dependencies found, add to ready_chunks.
            rep = next(iter(chunk.transactions))
            chunk_deps[rep] = deps
            if deps == 0:
                max_fallback = max(fallback_key(tx) for tx in chunk.transactions)
                heapq.heappush(ready_chunks, (by_goodness(chunk.feerate),
                                              max_fallback, rep))
        # Loop over the ready chunks, producing an output linearization for each.
        while ready_chunks:
            # Pop the highest-feerate chunk off the heap.
            _chunk_feerate, _max_fallback_key, chunk_rep = heapq.heappop(ready_chunks)
            chunk = self._txchunk[chunk_rep]
            # A heap of transactions which have no unmet dependencies, as (feerate, fallback, idx)
            # tuples.
            ready_tx: list[tuple[FeeFracByGoodness, int, int]] = []
            # A dict of tx -> unmet dependencies.
            tx_deps: dict[int, int] = {}
            # Compute for each transaction how many unmet dependencies it has, and add those with
            # none to the ready_tx list.
            for tx in chunk.transactions:
                tx_deps[tx] = len(self._parents[tx] & chunk.transactions)
                if tx_deps[tx] == 0:
                    heapq.heappush(ready_tx, (by_goodness(self._depgraph.feerate(tx)),
                                              fallback_key(tx), tx))
            # Loop over the ready transactions, adding each to the output linearization and
            # reducing the relevant unmet dependency counts for its children.
            while ready_tx:
                # Pop a transactions off the list.
                _tx_feerate, _fallback_key, tx = heapq.heappop(ready_tx)
                # Add it to the output linearization.
                ret.append(tx)
                # Loop over its children, whose unmet dependency counts need to be reduced.
                for chl in self._children[tx]:
                    if chl in chunk.transactions:
                        # The child is within the chunk, reduce the per-tx unmet dependency count
                        # of that transaction.
                        tx_deps[chl] -= 1
                        if tx_deps[chl] == 0:
                            # The child has no unmet dependencies left, add it to the ready list.
                            heapq.heappush(ready_tx, (by_goodness(self._depgraph.feerate(chl)),
                                                      fallback_key(chl), chl))
                    else:
                        # The child is in another chunk, reduce the per-chunk out-of-chunk unmet
                        # dependency count for that chunk.
                        chl_chunk = self._txchunk[chl]
                        chl_chunk_rep = next(iter(chl_chunk.transactions))
                        chunk_deps[chl_chunk_rep] -= 1
                        if chunk_deps[chl_chunk_rep] == 0:
                            # The child chunk has no out-of-chunk unmet dependencies left, add it
                            # to the ready heap.
                            max_fallback = max(fallback_key(tx) for tx in chl_chunk.transactions)
                            heapq.heappush(ready_chunks, (by_goodness(chl_chunk.feerate),
                                                          max_fallback, chl_chunk_rep))
        return ret

def linearize(depgraph: DepGraph,
              input_linearization: list[int] | None=None,
              fix_linearization: list[int] | None=None,
              minimize: bool=True, fallback_key: Callable[[int], Any]=lambda x: x) -> list[int]:
    """Produce an optimal linearization for the given graph."""
    sfl = SpanningForestState(depgraph, random.getrandbits(64))
    # Construct a topological SFL state from the input.
    if input_linearization:
        # A valid linearization is provided already.
        assert not fix_linearization
        sfl.load_linearization(input_linearization)
    elif fix_linearization:
        # Potentially invalid linearization is provided. Load it, but call make_topological() to
        # fix it up.
        sfl.load_linearization(fix_linearization)
        sfl.make_topological()
    else:
        # No input linearization is provided. Create a topological state from scratch.
        sfl.make_topological()
    # Always optimize the state.
    sfl.optimize()
    # Minimize the chunks if requested.
    if minimize:
        sfl.minimize()
    # Produce an output linearization.
    return sfl.get_linearization(fallback_key)

class TestSFL(unittest.TestCase):
    """Unit tests for the SFL algorithm."""

    def test_optimal(self) -> None:
        """Compare linearizations with known-optimal chunk feerate diagrams."""

        data_file = Path(__file__).resolve().parent / 'linearization_tests.json'
        with open(data_file, "r", encoding='utf-8') as input_file:
            data = json.load(fp=input_file)['optimal_linearizations']
            for ser_hex, expected_linearization in data:
                ser = bytes.fromhex(ser_hex)
                dg = DepGraphFormatter().deserialize(ser)
                assert dg is not None
                for _ in range(10):
                    lin = linearize(dg)
                    assert is_topological(dg, lin)
                    self.assertEqual(lin, expected_linearization)

if __name__ == '__main__':
    unittest.main()
