"""Experiments with optimal subchunk linearization."""

from __future__ import annotations
from fractions import Fraction
from pathlib import Path
from functools import cmp_to_key
import math
import unittest
import json
from dataclasses import dataclass
import random
from depgraph import DepGraph, is_topological, DepGraphFormatter, compute_chunking

@dataclass(slots=True)
class LinearExpr:
    """A linear expression, consisting of 0 or more 1st degree terms and a 0th degree term."""

    var_terms: dict[str, Fraction]
    const_term: Fraction

    def __init__(self,
                 terms: list[tuple[str, Fraction]] | None=None,
                 var: str | None=None,
                 const: Fraction | None=None) -> None:
        if const is not None:
            self.const_term = const
        else:
            self.const_term = Fraction()
        if terms is not None:
            self.var_terms = dict(terms)
        else:
            self.var_terms = {}
        if var is not None:
            self.add_var(var, Fraction(1))

    def copy(self) -> LinearExpr:
        """Create a non-sharing copy of this expression."""
        ret = LinearExpr()
        ret.var_terms = dict(self.var_terms)
        ret.const_term = self.const_term
        return ret

    def add_const(self, arg: Fraction) -> None:
        """Add a constant term arg to this expression."""
        self.const_term += arg

    def sub_const(self, arg: Fraction) -> None:
        """Subtract a constant term arg from this expression."""
        self.const_term -= arg

    def add_var(self, var: str, coef: Fraction=Fraction(1)) -> None:
        """Add a variable term coef*var to this expression."""
        if (self.var_terms.setdefault(var, Fraction())) == -coef:
            del self.var_terms[var]
        else:
            self.var_terms[var] += coef

    def sub_var(self, var: str, coef: Fraction=Fraction(1)) -> None:
        """Subtract a variable term coef*var from this expression."""
        if (self.var_terms.setdefault(var, Fraction())) == coef:
            del self.var_terms[var]
        else:
            self.var_terms[var] -= coef

    def __iadd__(self, other: LinearExpr) -> LinearExpr:
        self.add_const(other.const_term)
        for var, coef in other.var_terms.items():
            self.add_var(var, coef)
        return self

    def __isub__(self, other: LinearExpr) -> LinearExpr:
        self.sub_const(other.const_term)
        for var, coef in other.var_terms.items():
            self.sub_var(var, coef)
        return self

    def __imul__(self, other: Fraction) -> LinearExpr:
        self.const_term *= other
        for var in self.var_terms:
            self.var_terms[var] *= other
        return self

    def __itruediv__(self, other: Fraction) -> LinearExpr:
        self.const_term /= other
        for var in self.var_terms:
            self.var_terms[var] /= other
        return self

    def __neg__(self) -> LinearExpr:
        return self * Fraction(-1)

    def __add__(self, other: LinearExpr) -> LinearExpr:
        ret = self.copy()
        ret += other
        return ret

    def __sub__(self, other: LinearExpr) -> LinearExpr:
        ret = self.copy()
        ret -= other
        return ret

    def __mul__(self, other: Fraction) -> LinearExpr:
        ret = self.copy()
        ret *= other
        return ret

    def __rmul__(self, other: Fraction) -> LinearExpr:
        ret = self.copy()
        ret *= other
        return ret

    def __truediv__(self, other: Fraction) -> LinearExpr:
        ret = self.copy()
        ret /= other
        return ret

    def substitute(self, var: str, expr: LinearExpr) -> None:
        """Modify this expression in-place, substituting var for expr."""
        if (coef := self.var_terms.pop(var, Fraction())) != 0:
            self.add_const(expr.const_term * coef)
            for evar, ecoef in expr.var_terms.items():
                self.add_var(evar, ecoef * coef)

    def get_coefficient(self, var: str) -> Fraction:
        """Get the coefficient of the specified variable."""
        return self.var_terms.get(var, Fraction())

    def get_constant(self) -> Fraction:
        """Get the constant term of this expression, or 0 if none."""
        return self.const_term

    def get_variables(self) -> set[str]:
        """Get the set of variables with non-zero coefficients in this expression."""
        return set(self.var_terms)

    def change_var(self, from_var: str | None, to_var: str) -> None:
        """Given the equation from_var=<old_self>, rewrite it to to_var=<new_self>."""
        if from_var is not None:
            self.add_var(from_var, Fraction(-1))
        coef = -self.var_terms.pop(to_var)
        self /= coef

    def __str__(self) -> str:
        """Convert this expression to a pretty string."""
        if not self.var_terms:
            return str(self.const_term)
        ret = ""
        for var in sorted(self.var_terms.keys()):
            coef = self.var_terms[var]
            if ret == "":
                if coef == 1:
                    ret += var
                elif coef == -1:
                    ret += f"-{var}"
                else:
                    ret += f"{coef}*{var}"
            elif coef == 1:
                ret += f" + {var}"
            elif coef == -1:
                ret += f" - {var}"
            elif coef > 0:
                ret += f" + {coef}*{var}"
            else:
                ret += f" - {-coef}*{var}"
        if self.const_term > 0:
            ret += f" + {self.const_term}"
        elif self.const_term < 0:
            ret += f" - {-self.const_term}"
        return ret


class Simplex:
    """A representation of the state of the Simplex algorithm."""

    def __init__(self) -> None:
        self._eqs: dict[str, LinearExpr] = {}
        self._free_vars: set[str] = set()

    def add_equation(self, left: LinearExpr, right: LinearExpr | None=None) -> bool:
        """Add an equation of the form <left>=<right, or <left>=0."""
        # Construct a single equation of the form <expr>=0.
        expr = left.copy() if right is None else left - right
        # Eliminate all existing basic variables from it.
        for v, e in self._eqs.items():
            expr.substitute(v, e)
        # Deal with 0=0 and 0=1 cases.
        if not (var_set := expr.get_variables()):
            assert expr.get_constant() == 0
            return False
        # Pick a new basic variable. If the expression includes one or more entirely novel
        # variables, pick one of those preferably, to minimize changes to existing equations.
        choices = new_vars if (new_vars := var_set - self._free_vars) else var_set
        new_basic = random.choice(list(choices))
        # Write the equation in the form <new_basic>=<expr>.
        expr.change_var(None, new_basic)
        # Add the new equation.
        self._eqs[new_basic] = expr
        # Eliminate the new basic variable from all other equations.
        for eq_expr in self._eqs.values():
            eq_expr.substitute(new_basic, expr)
        # Update the set of free variables.
        self._free_vars |= expr.get_variables()
        self._free_vars.discard(new_basic)
        return True

    def make_step(self, enter_var: str, leave_var: str) -> None:
        """Take a simplex step: enter_var goes free->basic, leave_var goes basic->free."""
        # Change the equation from defining leave_far to defining enter_var.
        expr = self._eqs.pop(leave_var)
        expr.change_var(leave_var, enter_var)
        self._eqs[enter_var] = expr
        # Eliminate enter_var from all other equations.
        for eq_expr in self._eqs.values():
            eq_expr.substitute(enter_var, expr)
        self._free_vars.remove(enter_var)
        self._free_vars.add(leave_var)

    def make_free(self, free_vars: set[str]) -> None:
        """Make the provided set of variables free."""
        assert len(free_vars) == len(self._free_vars)
        while not free_vars.isdisjoint(self._eqs.keys()):
            good = False
            for leave_var in free_vars & self._eqs.keys():
                if not (enter_candidates := self._eqs[leave_var].get_variables() - free_vars):
                    continue
                enter_var = random.choice(list(enter_candidates))
                self.make_step(enter_var, leave_var)
                good = True
                break
            assert good

    def get_basic_solution(self) -> dict[str, Fraction]:
        """Get the current solution."""
        return ({var: Fraction() for var in self._free_vars} |
                {var: expr.const_term for var, expr in self._eqs.items()})

    def possible_entering(self) -> dict[str, Fraction]:
        """Find the set of possible entering variables, with their goal derivative. """
        return {var: coef for var, coef in self._eqs["goal"].var_terms.items() if coef > 0}

    def possible_leaving(self, enter_var: str) -> set[str]:
        """Find the set of possible leaving variables for a given entering variable."""
        ret: set[str] = set()
        min_inc = Fraction()
        for var, expr in self._eqs.items():
            if (coef := expr.var_terms.get(enter_var)) is not None:
                if coef < 0:
                    inc = -expr.const_term / coef
                    if len(ret) == 0 or inc < min_inc:
                        ret = set([var])
                        min_inc = inc
                    elif inc == min_inc:
                        ret.add(var)
        return ret

    def __str__(self) -> str:
        ret = "Simplex:\n"
        ret += "* Free: " + ", ".join(sorted(self._free_vars)) + "\n"
        for var in sorted(self._eqs.keys()):
            ret += "* " + var + " = " + str(self._eqs[var]) + "\n"
        return ret

ONE = Fraction(1)

def linearize_subchunk_simplex(depgraph: DepGraph,
                               enforce_level: int=1,
                               avoid_level: int=0,
                               max_deriv: bool=False) -> tuple[list[int], int, int]:
    """The most naive simplex-based subchunk linearizer.

    @param depgraph: the DepGraph to linearize
    @param enforce_level: To what extent dependency violations should be prevented using equations:
                          * 0: not at all
                          * 1: only direct dependencies
                          * 2: direct and indirect dependencies
    @param avoid_level: To what extent dependency violations should be prevented by not considering
                        them as entering variables (0, 1, 2, with the same meaning as above).
    @param max_deriv: Whether entering variable selection should be done using max derivative.
    """
    init_lin = list(depgraph.positions())
    random.shuffle(init_lin)
    init_lin.sort(key=lambda pos: len(depgraph.ancestors(pos)))
    init_posmap = {}
    for i, pos in enumerate(init_lin):
        init_posmap[pos] = i

    assert is_topological(depgraph, init_lin)
    simplex = Simplex()
    goal_expr = LinearExpr()
    free_vars: set[str] = set()
    forced_free_vars: set[str] = set()
    steps = 0
    for p1 in depgraph.positions():
        fr1 = depgraph.feerate(p1)
        for p2 in depgraph.positions():
            if p1 < p2:
                # Add equation: v_{x,y} + v_{y,x} = 1.
                simplex.add_equation(LinearExpr([(f"v_{p1}_{p2}", ONE), (f"v_{p2}_{p1}", ONE)]),
                                     LinearExpr(const=ONE))
            if p1 != p2:
                fr2 = depgraph.feerate(p2)
                # Add term to goal: v_{x,y} * fee(x) * size(y).
                goal_expr += LinearExpr([(f"v_{p1}_{p2}", Fraction(fr1.fee * fr2.size))])
                if init_posmap[p1] > init_posmap[p2]:
                    # Make v_{x,y} free if y appears before x in init_lin.
                    free_vars.add(f"v_{p1}_{p2}")
                    if avoid_level == 1 and p1 in depgraph.reduced_children(p2):
                        forced_free_vars.add(f"v_{p1}_{p2}")
                    elif avoid_level == 2 and p1 in depgraph.descendants(p2):
                        forced_free_vars.add(f"v_{p1}_{p2}")
    for p1 in depgraph.positions():
        for p2 in depgraph.positions():
            if p2 <= p1:
                continue
            for p3 in depgraph.positions():
                if p3 <= p2:
                    continue
                # Add equation: w_{x,y,z} = v_{x,y} + v_{y,z} + v_{z,x} - 1.
                simplex.add_equation(LinearExpr(var=f"w_{p1}_{p2}_{p3}"),
                                     LinearExpr([(f"v_{p1}_{p2}", ONE),
                                                 (f"v_{p2}_{p3}", ONE),
                                                 (f"v_{p3}_{p1}", ONE)],
                                                const=-ONE))
                # Add equation w_{x,y,z} + w_{z,y,x} = 1.
                simplex.add_equation(LinearExpr([(f"w_{p1}_{p2}_{p3}", ONE),
                                                 (f"w_{p3}_{p2}_{p1}", ONE)]),
                                     LinearExpr(const=ONE))
    # Add goal equation.
    assert simplex.add_equation(LinearExpr(var="goal"), goal_expr)
    if enforce_level == 1:
        for p1 in depgraph.positions():
            for p2 in depgraph.reduced_children(p1):
                # Add equation: v_{x,y} = 0.
                simplex.add_equation(LinearExpr(var=f"v_{p2}_{p1}"))
                free_vars.remove(f"v_{p2}_{p1}")
                forced_free_vars.discard(f"v_{p2}_{p1}")
    elif enforce_level == 2:
        for p1 in depgraph.positions():
            for p2 in depgraph.descendants(p1):
                if p1 != p2:
                    # Add equation: v_{x,y} = 0.
                    simplex.add_equation(LinearExpr(var=f"v_{p2}_{p1}"))
                    free_vars.remove(f"v_{p2}_{p1}")
                    forced_free_vars.discard(f"v_{p2}_{p1}")

    simplex.make_free(free_vars)
    while (enter_cand := simplex.possible_entering()):
        for var in forced_free_vars:
            enter_cand.pop(var, None)
        if not enter_cand:
            break
        if max_deriv:
            max_coef = max(enter_cand.values())
            enter_cand = {var: coef for var, coef in enter_cand.items() if coef == max_coef}
        enter_var = random.choice(list(enter_cand.keys()))
        leave_cand = simplex.possible_leaving(enter_var)
        leave_var = random.choice(list(leave_cand))
        simplex.make_step(enter_var, leave_var)
        steps += 1
    sol = simplex.get_basic_solution()
    init_lin.sort(key=cmp_to_key(lambda a,b: 0 if a==b else 1-2*int(sol[f"v_{a}_{b}"])))
    return init_lin, steps, int(sol["goal"])

class TestSFL(unittest.TestCase):
    """Unit tests for the SFL algorithm."""

    def test_optimal(self) -> None:
        """Compare linearizations with known-optimal chunk feerate diagrams."""

        data_file = Path(__file__).resolve().parent / 'linearization_tests.json'
        with open(data_file, "r", encoding='utf-8') as input_file:
            data = json.load(fp=input_file)['optimal_linearization_chunkings']
            for ser_hex, expected_diagram in data:
                ser = bytes.fromhex(ser_hex)
                dg = DepGraphFormatter().deserialize(ser)
                assert dg is not None
                if dg.tx_count() > 12:
                    continue
                if dg.dep_count() <= dg.tx_count() + 1:
                    continue
                expected_diagram.sort()
                for config in ((0, 2, False), (0, 1, False), (0, 2, True), (0, 1, True)):
                    expect_area: int | None = None
                    steps_n = 0
                    steps_s = 0
                    steps_q = 0
                    for _ in range(100):
                        lin, steps, area = linearize_subchunk_simplex(depgraph=dg,
                                                                      enforce_level=config[0],
                                                                      avoid_level=config[1],
                                                                      max_deriv=config[2])
                        assert is_topological(dg, lin)
                        if expect_area is None:
                            chunking = compute_chunking(dg, lin)
                            diagram = [[si.feerate.fee, si.feerate.size] for si in chunking]
                            diagram.sort()
                            self.assertEqual(diagram, expected_diagram)
                            expected_area = area
                        else:
                            assert expected_area == area
                        steps_n += 1
                        steps_s += steps
                        steps_q += steps * steps
                    avg = steps_s / steps_n
                    stddev = math.sqrt((steps_q - steps_s**2 / steps_n) / (steps_n - 1))
                    print(f"ntx={dg.tx_count()} ndeps={dg.dep_count()} "
                          f"enforce={config[0]} avoid={config[1]} max_deriv={config[2]}: "
                          f"steps={avg:.2f}+-{stddev:.2f} area={expected_area}")

if __name__ == '__main__':
    unittest.main()
