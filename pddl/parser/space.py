#
# Copyright 2021-2023 WhiteMech
#
# ------------------------------
#
# This file is part of pddl.
#
# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
#

"""Implementation of the PDDL domain parser."""
import sys
from typing import Dict, Optional, Set, Tuple

from lark import Lark, ParseError, Transformer

from pddl.action import Action, Agents
from pddl.core import Space
from pddl.custom_types import name
from pddl.exceptions import PDDLMissingRequirementError, PDDLParsingError
from pddl.helpers.base import assert_
from pddl.logic.base import And, ExistsCondition, ForallCondition, Imply, Not, OneOf, Or, Intends, Step, Literal
from pddl.logic.effects import AndEffect, Forall, When
from pddl.logic.predicates import DerivedPredicate, EqualTo, Predicate
from pddl.logic.terms import Constant, Variable
from pddl.logic.axioms import Axiom
from pddl.parser import SPACE_GRAMMAR_FILE, PARSERS_DIRECTORY
from pddl.parser.symbols import Symbols
from pddl.parser.typed_list_parser import TypedListParser


class SpaceTransformer(Transformer):
    """Domain Transformer."""

    def __init__(self, *args, **kwargs):
        """Initialize the domain transformer."""
        super().__init__(*args, **kwargs)

        self._constants_by_name: Dict[str, Constant] = {}
        self._predicates_by_name: Dict[str, Predicate] = {}
        self._current_parameters_by_name: Dict[str, Variable] = {}
        self._requirements: Set[str] = set()
        self._extended_requirements: Set[str] = set()

    def start(self, args):
        """Entry point."""
        return args[0]

    def space(self, args):
        """Process the 'space' rule."""
        args = [arg for arg in args if arg is not None]
        kwargs = {}
        name = args[2]["name"]
        literals = args[3]
        steps = args[4]
        axioms = args[5]
        kwargs.update(name=name, literals=literals, steps=steps, axioms=axioms)
        return Space(**kwargs)

    def space_def(self, args):
        """Process the 'space_def' rule."""
        return dict(name=args[2])

    def axioms(self, args):
        return [a for a in args[2:-1]]

    def axiom_def(self, args):
        """Process the 'axiom_def' rule."""
        _, _axiom, children, _ = args

        # process action body
        _children = children.children
        body = {
            _children[i][1:]: _children[i + 1] for i in range(0, len(_children), 2)
        }
        return Axiom([], **body)

    def literals(self, args):
        return [a for a in args[2:-1]]

    def literal(self, args):
        if len(args) > 1:  #intends
            return Literal(Intends(args[2], args[3]))
        else:
            return Literal(args[0])

    def steps(self, args):
        return [x for x in args[2:-1]]

    def step(self, args):
        name = str(args[1])
        params = [str(a) for a in args[2:-1]]
        return Step(name, params)

    def emptyor_pregd(self, args):
        """Process the 'emptyor_pregd' rule."""
        if len(args) == 2:
            return Or()
        else:
            assert_(len(args) == 1)
            return args[0]

    def gd_not(self, args):
        """Process the 'gd' not rule."""
        return Not(args[2])

    def gd_and(self, args):
        """Process the 'gd_and' rule."""
        operands = args[2:-1]
        return And(*operands)

    def gd_or(self, args):
        """Process the 'gd' or rule."""
        operands = args[2:-1]
        return Or(*operands)

    def gd_imply(self, args):
        """Process the 'gd' imply rule."""
        return Imply(args[2], args[3])

    def gd_quantifiers(self, args):
        """Process the 'gd' quantifiers rule."""
        cond_class = {
            Symbols.FORALL.value: (
                ForallCondition,
            ),
            Symbols.EXISTS.value: (
                ExistsCondition,
            ),
        }[args[1]]
        variables = [Variable(var_name, tags) for var_name, tags in args[3]]
        condition = args[5]
        return cond_class(cond=condition, variables=variables)

    def gd(self, args):
        """Process the 'gd' rule."""
        if len(args) == 1:
            return args[0]
        elif args[1] == Symbols.NOT.value:
            return self.gd_not(args)
        elif args[1] == Symbols.AND.value:
            return self.gd_and(args)
        elif args[1] == Symbols.OR.value:
            return self.gd_or(args)
        elif args[1] == Symbols.IMPLY.value:
            return self.gd_imply(args)
        elif args[1] in [Symbols.FORALL.value, Symbols.EXISTS.value]:
            return self.gd_quantifiers(args)
        elif args[1] == Symbols.INTENDS.value:
            return self.gd_intends(args)

    def emptyor_effect(self, args):
        """Process the 'emptyor_effect' rule."""
        if len(args) == 2:
            return Or()
        else:
            return args[0]

    def effect(self, args):
        """Process the 'effect' rule."""
        if len(args) == 1:
            return args[0]
        if args[1] == Symbols.AND.value:
            return AndEffect(*args[2:-1])
        raise ValueError("case not recognized")

    def c_effect(self, args):
        """Process the 'c_effect' rule."""
        if len(args) == 1:
            return args[0]
        if args[1] == Symbols.FORALL.value:
            variables = [Variable(var_name, tags) for var_name, tags in args[3]]
            return Forall(effect=args[-2], variables=variables)
        if args[1] == Symbols.WHEN.value:
            return When(args[2], args[3])
        if args[1] == Symbols.ONEOF.value:
            return OneOf(*args[2:-1])
        if args[1] == Symbols.INTENDS.value:
            return Intends(args[2], args[3])
        raise ValueError()

    def p_effect(self, args):
        """Process the 'p_effect' rule."""
        if len(args) == 1:
            return args[0]
        else:
            return Not(args[2])

    def cond_effect(self, args):
        """Process the 'cond_effect' rule."""
        if len(args) >= 3 and args[1] == Symbols.AND.value:
            p_effects = args[2:-1]
            return And(*p_effects)
        assert_(len(args) == 1)
        return args[0]


    def atomic_formula_term(self, args):
        """Process the 'atomic_formula_term' rule."""

        def constant_or_variable(t):
            # Case where the term is a free variable (bug) or comes from a parent quantifier
            if (
                not isinstance(t, Constant)
                and t not in self._current_parameters_by_name
            ):
                return Variable(str(t), {})
            return t if isinstance(t, Constant) else self._current_parameters_by_name[t]

        if args[1] == Symbols.EQUAL.value:
            left = constant_or_variable(args[2])
            right = constant_or_variable(args[3])
            return EqualTo(left, right)
        else:
            predicate_name = args[1]
            terms = list(map(constant_or_variable, args[2:-1]))
            return Predicate(predicate_name, *terms)

    def constant(self, args):
        """Process the 'constant' rule."""
        assert_(len(args) == 1, "Unexpected parsing error.")
        # constant = self._constants_by_name.get(args[0], None)
        # if constant is None:
        #     raise ParseError(f"Constant '{args[0]}' not defined.")
        # return constant
        return Constant(args[0])

    def atomic_formula_skeleton(self, args):
        """Process the 'atomic_formula_skeleton' rule."""
        predicate_name = args[1]
        variable_data: Dict[str, Set[str]] = args[2]
        variables = [Variable(var_name, tags) for var_name, tags in variable_data]
        return Predicate(predicate_name, *variables)


    def typed_list_name(self, args) -> Dict[name, Optional[name]]:
        """Process the 'typed_list_name' rule."""
        try:
            types_index = TypedListParser.parse_typed_list(args)
            return types_index.get_typed_list_of_names()
        except ValueError as e:
            raise self._raise_typed_list_parsing_error(args, e) from e

    def typed_list_variable(self, args) -> Tuple[Tuple[name, Set[name]], ...]:
        """
        Process the 'typed_list_variable' rule.

        Return a dictionary with as keys the terms and as value a set of types for each name.

        :param args: the argument of this grammar rule
        :return: a typed list (variable), i.e. a mapping from variables to the supported types
        """
        try:
            types_index = TypedListParser.parse_typed_list(args, allow_duplicates=True)
            return types_index.get_typed_list_of_variables()
        except ValueError as e:
            raise self._raise_typed_list_parsing_error(args, e) from e

    def _raise_typed_list_parsing_error(self, args, exception) -> PDDLParsingError:
        string_list = [
            str(arg) if isinstance(arg, str) else list(map(str, arg)) for arg in args
        ]
        return PDDLParsingError(
            f"error while parsing tokens {string_list}: {str(exception)}"
        )

    def type_def(self, args):
        """Parse the 'type_def' rule."""
        assert_(len(args) != 0, "unexpected parser state: empty type_def")

        if len(args) == 1:
            # single-typed type-def, return
            return args

        # if we are here, type_def is of the form (either t1 ... tn)
        # ignore first and last tokens since they are brackets.
        either_keyword, types = args[1], args[2:-1]
        assert_(str(either_keyword) == Symbols.EITHER.value)
        return types

_space_parser_lark = SPACE_GRAMMAR_FILE.read_text()


class SpaceParser:
    """PDDL space (as generated by GLAIVE) parser class."""

    def __init__(self):
        """Initialize."""
        self._transformer = SpaceTransformer()
        self._parser = Lark(
            _space_parser_lark, parser="lalr", import_paths=[PARSERS_DIRECTORY]
        )

    def __call__(self, text):
        """Call."""
        sys.tracebacklimit = 0  # noqa
        tree = self._parser.parse(text)
        sys.tracebacklimit = None  # noqa
        formula = self._transformer.transform(tree)
        return formula
