# Copyright (c) 2025 Pieter Wuille
# Distributed under the MIT software license, see the accompanying
# file LICENSE or http://www.opensource.org/licenses/mit-license.php.

"""Python reimplementation of the spanning-forest linearization algorithm."""

import json
import random
import heapq
import unittest
from pathlib import Path
from typing import Optional
from depgraph import FeeFrac, SetInfo, DepGraph, DepGraphFormatter, compute_chunking, is_topological

class SpanningForestState:
    """Data structure representing the state of the SFL algorithm."""

    __slots__ = ("_rng", "_parents", "_children", "_txchunk", "_active_deps", "_transactions")

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
        # Create a chunk for every transaction.
        for tx in depgraph.positions():
            self._txchunk[tx] = SetInfo.make_singleton(depgraph, tx)

    def _walk(self, start: int, chunk: SetInfo, dep_adjust: SetInfo, subtract: bool) -> None:
        """Walk all transactions of a chunk, update the chunk representatives and dependency
           info."""
        # The set of transactions that are known to need visiting still.
        todo = {start}
        # The set of transactions that have been visited already.
        done: set[int] = set()
        while todo != done:
            for tx_idx in todo - done:
                # Visit the transaction and mark it as processed.
                self._txchunk[tx_idx] = chunk
                done.add(tx_idx)
                # Iterate over all its parents.
                for par in self._parents[tx_idx]:
                    if par in done or (par, tx_idx) not in self._active_deps:
                        continue
                    # If the dependency to this parent is active, and it has not been visited
                    # already, we need to process it.
                    todo.add(par)
                # Iterate over all its children.
                for chl in self._children[tx_idx]:
                    info = self._active_deps.get((tx_idx, chl))
                    if info is None or chl in done:
                        continue
                    # If the dependency to this child is active, and it has not been visited
                    # already, we need to adjust the dependency, and process the child.
                    if subtract:
                        info -= dep_adjust
                    else:
                        info += dep_adjust
                    todo.add(chl)

    def _activate(self, par: int, chl: int) -> SetInfo:
        """Activate the dependency chl->par, and return the new chunk."""
        # The dependency cannot be active already.
        assert (par, chl) not in self._active_deps
        # Get the parent and child chunk's information.
        top_chunk = self._txchunk[par]
        bottom_chunk = self._txchunk[chl]
        assert top_chunk.transactions.isdisjoint(bottom_chunk.transactions)
        # We will reuse the bottom chunk for the new combined chunk.
        self._walk(par, bottom_chunk, bottom_chunk, False)
        self._walk(chl, bottom_chunk, top_chunk, False)
        # Mark the dependency active, and merge the top chunk into the bottom chunk.
        self._active_deps[(par, chl)] = top_chunk
        bottom_chunk += top_chunk
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
        self._walk(par, top_chunk, bottom_chunk, True)
        self._walk(chl, bottom_chunk, top_chunk, True)
        return top_chunk, bottom_chunk

    def _merge_chunks(self, top_chunk: SetInfo, bottom_chunk: SetInfo) -> Optional[SetInfo]:
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

    def _merge_step(self, chunk: SetInfo, downward: bool) -> Optional[SetInfo]:
        """Perform an upward or downward merge step, on the specified chunk. Returns the merged
           chunk, or None if no merge took place."""
        # Locate chunk information
        chunk_txn = chunk.transactions
        explored = set(chunk_txn)
        # The candidate chunks to merge with.
        candidates: list[SetInfo] = []
        # The feerates to compare with. Initially, this is equal to the chunk's own feerate. It is
        # updated to be the feerate of candidates whenever any are found.
        candidate_feerate = chunk.feerate
        # Explore chunks that can be reached from chunk_txn, with appropriate feerate.
        for tx in chunk_txn:
            newly_reached = (self._children[tx] if downward else self._parents[tx]) - explored
            explored |= newly_reached
            while newly_reached:
                new_chunk = self._txchunk[next(iter(newly_reached))]
                newly_reached -= new_chunk.transactions
                # Compare feerate of new chunk with existing candidate(s), if any.
                comp = new_chunk.feerate.compare(candidate_feerate)
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
            result = self._merge_step(chunk, downward)
            if result is None:
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
                    result = self._merge_step(chunk, flip != direction)
                    if result is not None:
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
                        dep_data = self._active_deps.get((par, tx))
                        if dep_data is None:
                            continue
                        if dep_data.feerate.compare(chunk.feerate) > 0:
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
                        dep_data = self._active_deps.get((par, tx))
                        if dep_data is None:
                            continue
                        if chunk.feerate <= dep_data.feerate:
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
                else:
                    # If such dependency is found, a split is not possible. If direction was True
                    # already, we have tried both directions and the chunk is minimal. If not, retry
                    # with direction=True.
                    if direction:
                        optimal |= chunk.transactions
                    else:
                        for tx in chunk.transactions:
                            txinfo[tx] = (pivot, True)

    def get_linearization(self) -> list[int]:
        """Produce a linearization. Requires that the SFL state is topological."""
        # A heap of chunks which have no unmet dependencies, as (-feerate, chunk_rep) pairs.
        ready_chunks: list[tuple[FeeFrac, int]] = []
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
                heapq.heappush(ready_chunks, (-chunk.feerate, rep))
        # Loop over the ready chunks, producing an output linearization for each.
        while ready_chunks:
            # Pop the highest-feerate chunk off the heap.
            _, chunk_rep = heapq.heappop(ready_chunks)
            chunk = self._txchunk[chunk_rep]
            # A list of transactions which have no unmet dependencies.
            ready_tx: list[int] = []
            # A dict of tx -> unmet dependencies.
            tx_deps: dict[int, int] = {}
            # Compute for each transaction how many unmet dependencies it has, and add those with
            # none to the ready_tx list.
            for tx in chunk.transactions:
                tx_deps[tx] = len(self._parents[tx] & chunk.transactions)
                if tx_deps[tx] == 0:
                    ready_tx.append(tx)
            # Loop over the ready transactions, adding each to the output linearization and
            # reducing the relevant unmet dependency counts for its children.
            while ready_tx:
                # Pop a transactions off the list.
                tx = ready_tx.pop()
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
                            ready_tx.append(chl)
                    else:
                        # The child is in another chunk, reduce the per-chunk out-of-chunk unmet
                        # dependency count for that chunk.
                        chl_chunk = self._txchunk[chl]
                        chl_chunk_rep = next(iter(chl_chunk.transactions))
                        chunk_deps[chl_chunk_rep] -= 1
                        if chunk_deps[chl_chunk_rep] == 0:
                            # The child chunk has no out-of-chunk unmet dependencies left, add it
                            # to the ready heap.
                            heapq.heappush(ready_chunks, (-chl_chunk.feerate, chl_chunk_rep))
        return ret

def linearize(depgraph: DepGraph) -> list[int]:
    """Produce an optimal linearization for the given graph."""
    sfl = SpanningForestState(depgraph, random.getrandbits(64))
    sfl.make_topological()
    sfl.optimize()
    sfl.minimize()
    return sfl.get_linearization()

class TestSFL(unittest.TestCase):
    """Unit tests for the SFL algorithm."""

    def test_optimal(self):
        """Compare linearizations with known-optimal chunk feerate diagrams."""

        data_file = Path(__file__).resolve().parent / 'linearization_tests.json'
        with open(data_file, "r", encoding='utf-8') as input_file:
            data = json.load(fp=input_file)['optimal_linearization_chunkings']
            for ser_hex, expected_diagram in data:
                ser = bytes.fromhex(ser_hex)
                dg = DepGraphFormatter().deserialize(ser)
                assert dg is not None
                expected_diagram.sort()
                for _ in range(10):
                    lin = linearize(dg)
                    assert is_topological(dg, lin)
                    chunking = compute_chunking(dg, lin)
                    diagram = [[si.feerate.fee, si.feerate.size] for si in chunking]
                    diagram.sort()
                    self.assertEqual(diagram, expected_diagram)

if __name__ == '__main__':
    unittest.main()
