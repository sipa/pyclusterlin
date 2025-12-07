# Copyright (c) 2024 Pieter Wuille
# Distributed under the MIT software license, see the accompanying
# file LICENSE or http://www.opensource.org/licenses/mit-license.php.

"""Python reimplementation of the cluster mempool DepGraph code and its serialization."""

from __future__ import annotations
from dataclasses import dataclass
from typing import BinaryIO
from serialization import Formatter, varint_formatter, signed_varint_formatter

@dataclass(slots=True)
class FeeFrac:
    """A class representing a fee/size pair."""
    fee: int
    size: int

    @classmethod
    def make_empty(cls) -> FeeFrac:
        """Construct an empty FeeFrac (0 fee, 0 size)."""
        return cls(0, 0)

    def __add__(self, other: FeeFrac) -> FeeFrac:
        """Add two FeeFracs."""
        return FeeFrac(self.fee + other.fee, self.size + other.size)

    def __sub__(self, other: FeeFrac) -> FeeFrac:
        """Subtract two FeeFracs."""
        return FeeFrac(self.fee - other.fee, self.size - other.size)

    def __neg__(self) -> FeeFrac:
        """Return a FeeFrac whose feerate is the negation of this one.
           Note: this is not the same as (empty - self)."""
        return FeeFrac(-self.fee, self.size)

    def __iadd__(self, other: FeeFrac) -> FeeFrac:
        """Increment a FeeFrac."""
        self.fee += other.fee
        self.size += other.size
        return self

    def __isub__(self, other: FeeFrac) -> FeeFrac:
        """Decrement a FeeFrac."""
        self.fee -= other.fee
        self.size -= other.size
        return self

    def compare(self, other: FeeFrac) -> int:
        """Compare two FeeFracs by feerate."""
        return self.fee * other.size - other.fee * self.size

    def __lt__(self, other: FeeFrac) -> bool:
        """Total ordering of FeeFracs: first increasing feerate, then decreasing size."""
        return self.compare(other) < 0

    def __le__(self, other: FeeFrac) -> bool:
        """Total ordering of FeeFracs: first increasing feerate, then decreasing size."""
        return self.compare(other) <= 0

    def copy(self) -> FeeFrac:
        """Construct a non-sharing copy of this FeeFrac."""
        return FeeFrac(self.fee, self.size)

    def __str__(self) -> str:
        """Convert this FeeFrac to a string."""
        return f"{self.fee}/{self.size}"

def compare_feefrac(a: FeeFrac, b: FeeFrac) -> int:
    """Compare a and b by increasing feerate, then by decreasing size."""
    c = a.compare(b)
    if c == 0:
        return b.size - a.size
    return c

class DepGraph:
    """A class representing a dependency graph of transactions."""
    __slots__ = ("_used", "_feefracs", "_ancestors", "_descendants")

    def __init__(self) -> None:
        """Initialize an empty graph."""
        self._used: set[int] = set()
        self._feefracs: list[FeeFrac | None] = []
        self._ancestors: list[set[int]] = []
        self._descendants: list[set[int]] = []

    def add_transaction(self, feefrac: FeeFrac) -> int:
        """Add a transaction at the end of this graph, with no parents or children."""
        idx = 0
        while idx in self._used:
            idx += 1
        if idx == len(self._feefracs):
            self._feefracs.append(FeeFrac(feefrac.fee, feefrac.size))
            self._ancestors.append(set([idx]))
            self._descendants.append(set([idx]))
        else:
            self._feefracs[idx] = FeeFrac(feefrac.fee, feefrac.size)
            self._ancestors[idx] = set([idx])
            self._descendants[idx] = set([idx])
        self._used.add(idx)
        return idx

    def remove_transactions(self, todel: set[int]) -> None:
        """Remove a subset of transactions from this graph."""
        self._used -= todel
        while self._feefracs and (len(self._feefracs) - 1) not in self._used:
            self._feefracs.pop()
            self._ancestors.pop()
            self._descendants.pop()
        for idx in range(len(self._feefracs)):
            self._ancestors[idx] &= self._used
            self._descendants[idx] &= self._used

    def tx_count(self) -> int:
        """Get the number of transactions in the graph."""
        return len(self._used)

    def position_range(self) -> int:
        """Get the highest used position in the graph plus 1."""
        return len(self._feefracs)

    def positions(self) -> set[int]:
        """Get a copy of the set of positions in used by this graph."""
        return set(self._used)

    def add_dependencies(self, parents: set[int], child: int) -> None:
        """Add a dependencies between a set of parents and a child in this graph."""
        assert parents.issubset(self._used)
        assert child in self._used
        par_anc: set[int] = set()
        for par in parents - self._ancestors[child]:
            par_anc |= self._ancestors[par]
        par_anc -= self._ancestors[child]
        if not par_anc:
            return
        chl_des = self._descendants[child]
        for anc_of_par in par_anc:
            self._descendants[anc_of_par] |= chl_des
        for dec_of_chl in chl_des:
            self._ancestors[dec_of_chl] |= par_anc

    def is_ancestor_of(self, parent: int, child: int) -> bool:
        """Determine if transaction parent is a (direct or indirect) ancestor of child."""
        assert parent in self._used
        assert child in self._used
        return parent in self._ancestors[child]

    def feerate(self, arg: int) -> FeeFrac:
        """Get a (mutable) FeeFrac object for the feerate of transaction arg."""
        ret = self._feefracs[arg]
        assert ret is not None
        return ret

    def ancestors(self, arg: int) -> set[int]:
        """Get a copy of the set of the ancestors of transaction arg."""
        assert arg in self._used
        return set(self._ancestors[arg])

    def descendants(self, arg: int) -> set[int]:
        """Get a copy of the set of the descendants of transaction arg."""
        assert arg in self._used
        return set(self._descendants[arg])

    def reduced_parents(self, arg: int) -> set[int]:
        """Get a set with the (reduced) parents of transaction arg."""
        ret = self.ancestors(arg)
        ret.remove(arg)
        for i in set(ret):
            if i in ret:
                ret -= self.ancestors(i)
                ret.add(i)
        return ret

    def reduced_children(self, arg: int) -> set[int]:
        """Get a set with the (reduced) children of transaction arg."""
        ret = self.descendants(arg)
        ret.remove(arg)
        for i in set(ret):
            if i in ret:
                ret -= self.descendants(i)
                ret.add(i)
        return ret

    @classmethod
    def from_reorder(cls, depgraph: DepGraph, mapping: list[int], pos_range: int) -> DepGraph:
        """Construct a graph by reordering an existing one."""
        assert len(mapping) == depgraph.position_range()
        assert (pos_range == 0) == (depgraph.tx_count() == 0)
        ret = cls()
        ret._used = set()
        ret._feefracs = [None for _ in range(pos_range)]
        ret._ancestors = [set() for _ in range(pos_range)]
        ret._descendants = [set() for _ in range(pos_range)]
        for i in depgraph.positions():
            new_idx = mapping[i]
            assert new_idx < pos_range
            ret._ancestors[new_idx] = set([new_idx])
            ret._descendants[new_idx] = set([new_idx])
            ret._used.add(new_idx)
            ret._feefracs[new_idx] = FeeFrac(depgraph.feerate(i).fee, depgraph.feerate(i).size)
        for i in depgraph.positions():
            parents: set[int] = set()
            for j in depgraph.reduced_parents(i):
                parents.add(mapping[j])
            ret.add_dependencies(parents, mapping[i])
        assert pos_range == 0 or (pos_range - 1) in ret._used
        return ret

@dataclass(slots=True)
class SetInfo:
    """A class representing a set of transactions, together with its feerate."""
    transactions: set[int]
    feerate: FeeFrac

    @classmethod
    def make_empty(cls) -> SetInfo:
        """Construct an empty SetInfo."""
        return cls(set(), FeeFrac.make_empty())

    @classmethod
    def make_singleton(cls, depgraph: DepGraph, tx: int) -> SetInfo:
        """Construct a SetInfo for a single transaction."""
        return cls({tx}, depgraph.feerate(tx).copy())

    @classmethod
    def make_set(cls, depgraph: DepGraph, txn: set[int]) -> SetInfo:
        """Construct a SetInfo for a set of transactions."""
        return cls(set(txn), sum((depgraph.feerate(i) for i in txn), FeeFrac.make_empty()))

    def __add__(self, other: SetInfo) -> SetInfo:
        """Construct the union of two SetInfos."""
        assert self.transactions.isdisjoint(other.transactions)
        return SetInfo(self.transactions | other.transactions, self.feerate + other.feerate)

    def __sub__(self, other: SetInfo) -> SetInfo:
        """Subtract two SetInfos."""
        assert other.transactions.issubset(self.transactions)
        return SetInfo(self.transactions - other.transactions, self.feerate - other.feerate)

    def __iadd__(self, other: SetInfo) -> SetInfo:
        """Merge another SetInfo into this one."""
        assert self.transactions.isdisjoint(other.transactions)
        self.transactions |= other.transactions
        self.feerate += other.feerate
        return self

    def __isub__(self, other: SetInfo) -> SetInfo:
        """Remove another SetInfo from this one."""
        assert other.transactions.issubset(self.transactions)
        self.transactions -= other.transactions
        self.feerate -= other.feerate
        return self

    def copy(self) -> SetInfo:
        """Construct a non-sharing copy of this SetInfo."""
        return SetInfo(set(self.transactions), self.feerate.copy())

    def compare(self, other: SetInfo) -> int:
        """Compare two SetInfos by feerate."""
        return self.feerate.compare(other.feerate)

def compute_chunking(depgraph: DepGraph, linearization: list[int]) -> list[SetInfo]:
    """Compute chunking for a given linearization, in [SetInfo] form."""
    ret: list[SetInfo] = []
    for pos in linearization:
        add = SetInfo.make_singleton(depgraph, pos)
        while len(ret) > 0 and add.compare(ret[-1]) > 0:
            add += ret[-1]
            ret.pop()
        ret.append(add)
    return ret

class DepGraphFormatter(Formatter[DepGraph]):
    """Formatter for acyclic DepGraph objects."""

    def __init__(self, max_range: int = 256):
        """Initialize a formatter. max_range sets the limit on deserialized positions."""
        super().__init__()
        self._max_range = max_range

    def encode(self, stream: BinaryIO, depgraph: DepGraph) -> None:
        """Append encoding of depgraph to stream."""
        topo_order = list(depgraph.positions())
        topo_order.sort(key=lambda i: (len(depgraph.ancestors(i)), i))
        done: set[int] = set()
        for topo_idx in range(depgraph.tx_count()):
            idx = topo_order[topo_idx]
            varint_formatter.encode(stream, depgraph.feerate(idx).size)
            signed_varint_formatter.encode(stream, depgraph.feerate(idx).fee)
            written_parents: set[int] = set()
            diff = 0
            for dep_dist in range(topo_idx):
                dep_idx = topo_order[topo_idx - 1 - dep_dist]
                if len(depgraph.descendants(dep_idx) & written_parents):
                    continue
                if depgraph.is_ancestor_of(dep_idx, idx):
                    varint_formatter.encode(stream, diff)
                    diff = 0
                    written_parents.add(dep_idx)
                else:
                    diff += 1
            add_holes = set(range(idx)) - done - depgraph.positions()
            if not add_holes:
                skips = sum(i >= idx for i in done)
                varint_formatter.encode(stream, diff + skips)
            else:
                varint_formatter.encode(stream, diff + len(done) + len(add_holes))
                done |= add_holes
            done.add(idx)
        varint_formatter.encode(stream, 0)

    def decode(self, stream: BinaryIO) -> DepGraph:
        """Pop and decode a DepGraph from reversed encoding rdata."""
        topo_depgraph = DepGraph()
        reordering: list[int] = []
        total_size = 0

        while True:
            size = varint_formatter.decode(stream)
            if size is None:
                break
            size &= 0x3fffff
            if size == 0 or topo_depgraph.tx_count() == self._max_range:
                break
            fee = signed_varint_formatter.decode(stream)
            if fee is None:
                break
            new_feerate = FeeFrac(fee, size)
            read_error = False
            new_ancestors: set[int] = set()
            try:
                topo_idx = len(reordering)
                diff = varint_formatter.decode(stream)
                if diff is None:
                    raise RuntimeError
                for dep_dist in range(topo_idx):
                    dep_topo_idx = topo_idx - 1 - dep_dist
                    if dep_topo_idx in new_ancestors:
                        continue
                    if diff == 0:
                        new_ancestors |= topo_depgraph.ancestors(dep_topo_idx)
                        diff = varint_formatter.decode(stream)
                        if diff is None:
                            raise RuntimeError
                    else:
                        diff -= 1
            except RuntimeError:
                read_error = True
                diff = 0
            topo_idx = topo_depgraph.add_transaction(new_feerate)
            topo_depgraph.add_dependencies(new_ancestors, topo_idx)
            if total_size < self._max_range:
                diff %= self._max_range
                if diff <= total_size:
                    for pos, val in enumerate(reordering):
                        reordering[pos] += (val >= total_size - diff)
                    reordering.append(total_size - diff)
                else:
                    total_size = diff
                    reordering.append(total_size)
                total_size += 1
            else:
                diff %= self._max_range - len(reordering)
                holes = set(range(self._max_range))
                for pos in reordering:
                    holes.remove(pos)
                for pos in sorted(holes):
                    if diff == 0:
                        reordering.append(pos)
                        break
                    diff -= 1
            if read_error:
                break

        return DepGraph.from_reorder(topo_depgraph, reordering, total_size)

class LinearizationFormatter(Formatter[list[int]]):
    """Formatter for topologically-valid linearizations of a DepGraph."""

    def __init__(self, depgraph: DepGraph):
        """Initialize a formatter for the specified DepGraph."""
        super().__init__()
        self._depgraph = depgraph

    def encode(self, stream: BinaryIO, arg: list[int]) -> None:
        """Append a serialization of arg to stream."""
        done: set[int] = set()
        for i in arg:
            diff = 0
            for j in sorted(self._depgraph.positions() - done):
                if len(self._depgraph.ancestors(j) - done) == 1:
                    if i == j:
                        varint_formatter.encode(stream, diff)
                        break
                    diff += 1
                else:
                    assert i != j
            done.add(i)

    def decode(self, stream: BinaryIO) -> list[int] | None:
        """Deserialize a linearization from stream."""
        done: set[int] = set()
        n = self._depgraph.tx_count()
        ret: list[int] = []
        for _ in range(n):
            diff = varint_formatter.decode(stream)
            if diff is None:
                diff = 0
            candidates: list[int] = []
            for j in sorted(self._depgraph.positions() - done):
                if len(self._depgraph.ancestors(j) - done) == 1:
                    candidates.append(j)
            assert len(candidates) > 0
            ret.append(candidates[diff % len(candidates)])
            done.add(ret[-1])
        return ret

class DepGraphListFormatter(Formatter[list[DepGraph]]):
    """Formatter for a list of DepGraphs."""
    def __init__(self, max_range: int = 256):
        self._depgraph_formatter = DepGraphFormatter(max_range)

    def decode(self, stream: BinaryIO) -> list[DepGraph] | None:
        ret: list[DepGraph] = []
        while True:
            dec = self._depgraph_formatter.decode(stream)
            if dec is None:
                return None
            if dec.tx_count() == 0:
                break
            ret.append(dec)
        return ret

def is_topological(depgraph: DepGraph, lin: list[int]) -> bool:
    """Test if the specified linearization is topological for the given DepGraph."""
    done: set[int] = set()
    for tx in lin:
        if tx in done:
            return False
        done.add(tx)
        if not depgraph.ancestors(tx).issubset(done):
            return False
    if done != depgraph.positions():
        return False
    return True
