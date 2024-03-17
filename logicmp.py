#!/usr/bin/python
#****************************************************************#
# ScriptName: logic.py
# Author: wdxu
# Create Date: 2023-02-25 13:03
# Modify Author: 
# Function: 
#***************************************************************#

#import re
import torch
from torch import nn
import opt_einsum as oe

from itertools import chain
from pyparsing import (printables, alphanums, alphas, delimitedList, Forward,
            Group, Keyword, Literal, opAssoc, infixNotation,
            ParserElement, ParseException, ParseSyntaxException, Suppress,
            Word)
ParserElement.enablePackrat()
"""
BNF

    FORMULA     ::= ('forall' | 'exists') SYMBOL ':' FORMULA
                 |  FORMULA '->' FORMULA      # right associative
                 |  FORMULA '|' FORMULA       # left associative
                 |  FORMULA '&' FORMULA       # left associative
                 |  '~' FORMULA
                 |  '(' FORMULA ')'
                 |  TERM '=' TERM
                 |  'true'
                 |  'false'
    TERM        ::= SYMBOL | SYMBOL '(' TERM_LIST ')'
    TERM_LIST   ::= TERM | TERM ',' TERM_LIST
    SYMBOL      ::= [0-9a-zA-Z.-]*
"""

FORALL = 'forall'
EXISTS = 'exists'
OR = '|'
AND = '&'
IMPLIES = '->'
NOT = '~'
COLON = ':'
LEFT = '('
RIGHT = ')'
AT = '@'
TRUE = 'true'
FALSE = 'false'

def onehot(x, n):
  r = [int(i in x) for i in range(n)]
  return r

def pyparsing_parse(text):
    """
    >>> formula = "f(a) & f(b)"
    >>> print(pyparsing_parse(formula))
    [['f', ['a']], '&', ['f', ['b']]]
    >>> formula = "exists y: f(a) -> f(b)"
    >>> print(pyparsing_parse(formula))
    ['exists', 'y', [['f', ['a']], '->', ['f', ['b']]]]
    >>> formula = "forall x: exists y: a | b"
    >>> print(pyparsing_parse(formula))
    ['forall', 'x', ['exists', 'y', ['a', '|', 'b']]]
    >>> formula = "(forall x: exists y: true) -> true & ~ true -> true"
    >>> print(pyparsing_parse(formula))
    [['forall', 'x', ['exists', 'y', 'true']], '->', [['true', '&', ['~', 'true']], '->', 'true']]
    >>> formula = "forall x: exists y: true -> true & true | ~ true"
    >>> print(pyparsing_parse(formula))
    ['forall', 'x', ['exists', 'y', ['true', '->', [['true', '&', 'true'], '|', ['~', 'true']]]]]
    >>> formula = "~ true | true & true -> forall x: exists y: true"
    >>> print(pyparsing_parse(formula))
    [[['~', 'true'], '|', ['true', '&', 'true']], '->', ['forall', 'x', ['exists', 'y', 'true']]]
    >>> formula = "forall x: = x & true"
    >>> print(pyparsing_parse(formula))
    Syntax error:
    forall x: = x & true
           ^
    []
    """
    left_parenthesis, right_parenthesis, colon, at = map(Suppress, LEFT+RIGHT+COLON+AT)
    forall = Keyword(FORALL)
    exists = Keyword(EXISTS)
    implies = Literal(IMPLIES)
    or_ = Literal(OR)
    and_ = Literal(AND)
    not_ = Literal(NOT)
    #equals = Literal("=")
    boolean = Keyword("false") | Keyword("true")
    symbol = Word(alphanums + '_.')
    term = Forward()
    #term << (Group(symbol + Group(left_parenthesis +
                   #delimitedList(symbol) + right_parenthesis)) | symbol)
    term << (Group(symbol + 
                   Group(left_parenthesis + delimitedList(symbol) + right_parenthesis) + 
                   at + 
                   Group(left_parenthesis + delimitedList(symbol) + right_parenthesis)) | 
             Group(symbol + 
                   Group(left_parenthesis + delimitedList(symbol) + right_parenthesis)) |
             symbol)
    #term << (Group(symbol + Group(left_parenthesis + delimitedList(symbol) + right_parenthesis) + at + symbol))
    formula = Forward()
    forall_expression = Group(forall + symbol + colon + formula)
    exists_expression = Group(exists + symbol + colon + formula)
    operand = forall_expression | exists_expression | boolean | term
    formula << infixNotation(operand, [
                                  #(equals, 2, opAssoc.LEFT),
                                  (not_, 1, opAssoc.RIGHT),
                                  (and_, 2, opAssoc.LEFT),
                                  (or_, 2, opAssoc.LEFT),
                                  (implies, 2, opAssoc.RIGHT)])
    try:
        result = formula.parseString(text, parseAll=True)
        assert len(result) == 1
        return result[0].asList()
    except (ParseException, ParseSyntaxException) as err:
        print("Syntax error:\n{0.line}\n{1}^".format(err,
              " " * (err.column - 1)))
        return []

def construct(ftree, name2predicate, name2variable={}):
  """
  >>> # step 1: define the fields
  >>> person = Field('person', ['Tom', 'Mike'])
  >>> paper = Field('paper', ['p0', 'p1'])
  >>> clazz = Field('class', ['c0', 'c1', 'c2'])
  >>> # step 2: define the predicates
  >>> smoke = Predicate('smoke', [person])
  >>> cancer = Predicate('cancer', [person])
  >>> friend = Predicate('friend', [person, person])
  >>> cite = Predicate('cite', [paper, paper])
  >>> ptype = Predicate('ptype', [paper], ['rl', 'ec', 'ml'])
  >>> name2predicate = {'smoke': smoke, 'friend': friend, 'ptype': ptype, 'cancer': cancer}
  >>> fstr = 'friend(Tom, Mike)@(false)'
  >>> ftree = pyparsing_parse(fstr)
  >>> formula = construct(ftree, name2predicate)
  >>> print(formula)
  friend(Tom, Mike)@(false)
  >>> fstr = 'forall x: forall y: smoke(x) & friend(x, y) -> smoke(y)'
  >>> ftree = pyparsing_parse(fstr)
  >>> print(ftree)
  ['forall', 'x', ['forall', 'y', [[['smoke', ['x']], '&', ['friend', ['x', 'y']]], '->', ['smoke', ['y']]]]]
  >>> formula = construct(ftree, name2predicate)
  >>> res = is_conjunctive_normal_form(formula)
  >>> print(res)
  False
  >>> formula = formula.apply()
  >>> print(formula)
  forall x: forall y: (~smoke(x) | ~friend(x, y) | smoke(y))
  >>> res = is_conjunctive_normal_form(formula)
  >>> print(res)
  True
  >>> print([str(x) for x in formula.clauses()])
  ['(~smoke(x) | ~friend(x, y) | smoke(y))']
  >>> f1 = formula
  >>> fstr = 'forall x: (smoke(x) -> cancer(x)) & (cancer(x) -> smoke(x))'
  >>> ftree = pyparsing_parse(fstr)
  >>> print(ftree)
  ['forall', 'x', [[['smoke', ['x']], '->', ['cancer', ['x']]], '&', [['cancer', ['x']], '->', ['smoke', ['x']]]]]
  >>> formula = construct(ftree, name2predicate)
  >>> print(formula)
  forall x: smoke(x) -> cancer(x) & cancer(x) -> smoke(x)
  >>> res = is_conjunctive_normal_form(formula)
  >>> print(res)
  False
  >>> formula = formula.apply()
  >>> print(formula)
  forall x: (~smoke(x) | cancer(x)) & (~cancer(x) | smoke(x))
  >>> res = is_conjunctive_normal_form(formula)
  >>> print(res)
  True
  >>> print([str(x) for x in formula.clauses()])
  ['(~smoke(x) | cancer(x))', '(~cancer(x) | smoke(x))']
  >>> f2 = formula
  """
  if isinstance(ftree, str):
    predicate = name2predicate[ftree]
    arguments = []
    cur = Term(predicate, arguments, name2predicate)
    return cur
  if ftree[0] == FORALL:
    name2variable[ftree[1]] = Variable(ftree[1], None)
    node = construct(ftree[2], name2predicate, name2variable)
    cur = Forall(ftree[1], node, name2predicate)
    name2variable.pop(ftree[1])
    return cur
  elif ftree[0] == EXISTS:
    name2variable[ftree[1]] = Variable(ftree[1], None)
    node = construct(ftree[2], name2predicate, name2variable)
    cur = Exists(ftree[1], node, name2predicate)
    name2variable.pop(ftee[1])
    return cur
  elif ftree[1] == IMPLIES:
    body = construct(ftree[0], name2predicate, name2variable)
    head = construct(ftree[2], name2predicate, name2variable)
    cur = Implies(body, head, name2variable)
    return cur
  elif AND in ftree:
    nodes = [construct(n, name2predicate, name2variable) for n in ftree if n != AND]
    cur = And(nodes, name2variable)
    return cur
  elif OR in ftree:
    nodes = [construct(n, name2predicate, name2variable) for n in ftree if n != OR]
    cur = Or(nodes, name2variable)
    return cur
  elif NOT == ftree[0]:
    assert len(ftree) == 2
    node = construct(ftree[1], name2predicate, name2variable)
    cur = Not(node, name2variable)
    return cur
  else:
    predicate = name2predicate[ftree[0]]
    arguments = []
    for i, argname in enumerate(ftree[1]):
      if argname in name2variable:
        variable = name2variable[argname]
        if variable.field is None:
          variable.field = predicate.fields[i]
        else:
          assert variable.field == predicate.fields[i], \
              "Argument {} was assigned to field {} rather than {} of predicate {}.".format(
                  argument, 
                  variable.field,
                  predicate.fields[i],
                  predicate.name
                  )
        argument = name2variable[argname]
      else:
        argument = Constant(argname, predicate.fields[i])
      arguments.append(argument)
    if len(ftree) > 2:
      values = ftree[2]
      for value in values:
        assert value in predicate.vocab, \
            "Value {} is not in the predicate {}.".format(
                value,
                predicate.name
                )
    else:
      values = {TRUE}

    cur = Term(predicate, arguments, values, name2variable)
    return cur

def is_atom(node):
  if isinstance(node, Not):
    node = node.node
    if isinstance(node, Term):
      return True
    return False
  if isinstance(node, Term):
    return True
  return False

def is_clause(node):
  if not isinstance(node, Or):
    return False
  for n in node.nodes:
    if not is_atom(n):
      return False
  return True

def is_conjunctive_normal_form(node):
  while node and isinstance(node, Forall):
    node = node.node

  if not node:
    return False

  if isinstance(node, And):
    for n in node.nodes:
      if not is_clause(n):
        return False
    return True
  elif isinstance(node, Or):
    if not is_clause(node):
      return False
    return True
  return False

class Formula:
  def __init__(self, name2variable=None):
    self.name2variable = name2variable if name2variable is not None else {}

  def clauses(self):
    if not is_conjunctive_normal_form(self):
      return []
    node = self
    while node and isinstance(node, Forall):
      node = node.node

    if isinstance(node, And):
      return node.nodes
    return [node]

class Forall(Formula):
  def __init__(self, name, node, name2variable=None):
    super().__init__(name2variable)
    self.name = name
    self.node = node

  def __str__(self):
    return ' '.join([FORALL, self.name + COLON, str(self.node)])

  def apply(self):
    return Forall(self.name, self.node.apply(), self.name2variable)

class Exists(Formula):
  def __init__(self, name, node, name2variable=None):
    super().__init__(name2variable)
    self.name = name
    self.node = node

  def __str__(self):
    return ' '.join([EXISTS, self.name + COLON, str(self.node)])

  def apply(self):
    return Exists(self.name, self.node.apply(), self.name2variable)

class Implies(Formula):
  def __init__(self, body, head, name2variable=None):
    super().__init__(name2variable)
    self.body = body
    self.head = head

  def __str__(self):
    return ' '.join([str(self.body), '->',  str(self.head)])

  def apply(self):
    n1 = Not(self.body.apply(), self.name2variable).apply()
    n2 = self.head.apply()
    return Or([n1, n2], self.name2variable).apply()

class And(Formula):
  def __init__(self, nodes, name2variable=None):
    super().__init__(name2variable)
    self.nodes = nodes
  
  def __str__(self):
    nodes = [str(n) for n in self.nodes]
    ret = list(chain(*[[n, AND] for n in nodes]))[:-1]
    return ' '.join(ret)

  def apply(self):
    nodes = []
    for n in self.nodes:
      n = n.apply()
      if isinstance(n, And):
        nodes.extend([n.apply() for n in n.nodes])
      else:
        nodes.append(n)
    return And(nodes, self.name2variable)

class Or(Formula):
  def __init__(self, nodes, name2variable=None):
    super().__init__(name2variable)
    self.nodes = nodes

  def __str__(self):
    nodes = [str(n) for n in self.nodes]
    ret = list(chain(*[[n, OR] for n in nodes]))[:-1]
    return LEFT + ' '.join(ret) + RIGHT

  def apply(self):
    nodes = []
    for n in self.nodes:
      n = n.apply()
      if isinstance(n, Or):
        nodes.extend([t.apply() for t in n.nodes])
      else:
        nodes.append(n)
    return Or(nodes, self.name2variable)

class Not(Formula):
  def __init__(self, node, name2variable=None):
    super().__init__(name2variable)
    self.node = node

  def __str__(self):
    return NOT + str(self.node)

  def apply(self):
    if isinstance(self.node, And):
      nodes = [Not(n, self.name2variable).apply() for n in self.node.nodes]
      return Or(nodes, self.name2variable)
    elif isinstance(self.node, Or):
      nodes = [Not(n, self.name2variable).apply() for n in self.node.nodes]
      return And(nodes, self.name2variable)
    elif isinstance(self.node, Implies):
      node = self.node.apply()
      nodes = [Not(n, self.name2variable).apply() for n in node.nodes]
      return And(nodes, self.name2variable)
    elif isinstance(self.node, Term):
      node = self.node
      predicate = node.predicate
      arguments = node.arguments
      values = predicate.labels - node.values
      name2variable = node.name2variable
      return Term(predicate, arguments, values, name2variable)
    return Not(self.node.apply(), self.name2variable)

class Term(Formula):
  def __init__(self, predicate, arguments, values, name2variable):
    super().__init__(name2variable)
    self.predicate = predicate
    self.arguments = arguments
    self.values = values
    self.index = onehot([predicate.vocab[v] for v in values], len(predicate.labels))

  def __str__(self):
    ret = self.predicate.name + LEFT + ', '.join([a.name for a in self.arguments]) + RIGHT
    if self.values == {TRUE}:
      return ret
    if self.values == {FALSE}:
      return NOT + ret
    values = LEFT + ', '.join(self.values) + RIGHT
    return ret + AT + values

  def apply(self):
    return Term(self.predicate, [a for a in self.arguments], self.values, self.name2variable)

class Field:
    def __init__(self, name: str, entities: list=None):
        self.name = name
        self.vocab = {v: i for i, v in enumerate(entities)}
    
    def __str__(self):
        return self.name

    def __getitem__(self, key):
        return self.vocab[key]
    
    def __setitem__(self, key):
        self.vocab[key] = len(self.vocab)

    def __len__(self):
        return len(self.vocab)

    def size(self):
        return len(self.vocab)

class Predicate:
    def __init__(self, name: str, fields: list, labels: list=None, size: int=2):
        self.name = name
        self.fields = fields
        if labels:
            self.vocab = {l: i for i, l in enumerate(labels)}
            self.labels = set(labels)
        else:
            assert size == 2
            self.vocab = {FALSE: 0, TRUE: 1}
            self.labels = set([FALSE, TRUE])
    
    def __str__(self):
        return self.name + '(' + ', '.join([str(a) for a in self.fields]) + ')'

    def size(self):
      return len(self.labels)

class Argument:
    def __init__(self, name: str, field: Field):
        self.name = name
        self.field = field
    
    def __str__(self):
        return str(self.field) + '/' + self.name
    
class Constant(Argument):
    def __init__(self, name: str, field: Field):
        super(Constant, self).__init__(name, field)
        self.index = self.field.vocab[name]

    def __str__(self):
        return super().__str__()
    
class Variable(Argument):
    def __init__(self, name: str, field: Field):
        super(Variable, self).__init__(name, field)

    def __str__(self):
        return super().__str__()

class LogicMP(nn.Module):
    def __init__(self, clauses, nstep=5, weights=None, batch_size=1):
        super(LogicMP, self).__init__()

        self.clauses = clauses
        self.nstep = nstep
        self.batch_size = batch_size
        self.build_einsum()

        if weights is None:
            weights = torch.ones(len(clauses))
        self.weights = nn.Parameter(weights)

    def build_einsum(self):
        self.message_infos = {}
        for ridx, clause in enumerate(self.clauses):
            vals = {}
            args_set = set()
            for i, node in enumerate(clause.nodes):
                valname = '{}_{}_vals'.format(ridx, i)
                self.register_buffer(valname, torch.tensor(node.index))
                vals[i] = self.__getattr__(valname)
                for argument in node.arguments:
                  if isinstance(argument, Variable):
                    args_set.add(argument.name)

            args_map = {}
            for arg in args_set:
              if len(arg) == 1 and arg.isalpha():
                args_map[arg] = arg
            for arg in args_set:
              if len(arg) != 1 or not arg.isalpha():
                for c in "xyzuvwacdefghijkopqrst":
                  if c not in args_map:
                    args_map[c] = arg
                    break
            vals_map = {}
            for i, _ in enumerate(clause.nodes):
              for c in "ijklrstbcdefghxyzuvwxys":
                if c not in args_map and c not in vals_map:
                  vals_map[c] = i
                  break
            args_map = {v: k for k, v in args_map.items()}
            vals_map = {v: k for k, v in vals_map.items()}

            for i, _ in enumerate(clause.nodes):
                s_pn_ls = [literal.predicate.name for j, literal in enumerate(clause.nodes) if j != i]
                t_pn = clause.nodes[i].predicate.name

                s_shape_ls, t_shape = [], [self.batch_size]
                s_consts_ls, t_consts = [], []
                s_args_ls, t_args = [], []
                for j, literal in enumerate(clause.nodes):
                    if j == i:
                        for argument in literal.arguments:
                            if isinstance(argument, Constant):
                                t_consts.append((argument.field.size(), argument.index))
                            else:
                                t_consts.append(None)
                                t_shape.append(argument.field.size())
                                t_args.append(argument.name)
                        t_shape.append(literal.predicate.size())
                    else:
                        s_shape, s_consts, s_args = [self.batch_size], [], []
                        for argument in literal.arguments:
                            if isinstance(argument, Constant):
                                s_consts.append((argument.field.size(), argument.index))
                            else:
                                s_consts.append(None)
                                s_shape.append(argument.field.size())
                                s_args.append(argument.name)
                        s_shape.append(literal.predicate.size())
                        s_shape_ls.append(s_shape)
                        s_consts_ls.append(s_consts)
                        s_args_ls.append(s_args)

                s_vals_ls = [vals[j] for j, _ in enumerate(clause.nodes) if j != i]
                t_vals = vals[i]

                einsum_s_term_ls = []
                einsum_s_shape_ls = []
                for j, args in enumerate(s_args_ls):
                  einsum_s_term_ls.append('b'+''.join([args_map[a] for a in args]) + vals_map[j])
                  einsum_s_shape_ls.append(s_shape_ls[j])
                  einsum_s_term_ls.append(vals_map[j])
                  einsum_s_shape_ls.append([s_shape_ls[j][-1]])
                einsum_s_term_ls.append(vals_map[len(clause.nodes)-1])
                einsum_s_shape_ls.append([t_shape[-1]])
                einsum_s_str = ','.join(einsum_s_term_ls)

                s_args_set = set(arg for args in s_args_ls for arg in args)
                einsum_t_str = 'b'+''.join([args_map[a] for a in t_args if a in s_args_set]) + vals_map[len(clause.nodes)-1]
                einsum_str = '->'.join([einsum_s_str, einsum_t_str])
                
                einsum = oe.contract_expression(einsum_str, *einsum_s_shape_ls)
                expand = [(idx, clause.name2variable[arg].field.size()) for idx, arg in enumerate(t_args) if arg not in s_args_set]
                export = [(idx, const) for idx, const in enumerate(t_consts) if const is not None]

                message_info = {}
                message_info['einsum'] = einsum
                message_info['expand'] = expand
                message_info['export'] = export
                message_info['s_pn_ls'] = s_pn_ls
                message_info['t_pn'] = t_pn
                message_info['s_args_ls'] = s_args_ls
                message_info['t_args'] = t_args
                message_info['s_vals_ls'] = s_vals_ls
                message_info['t_vals'] = t_vals
                message_info['s_consts_ls'] = s_consts_ls
                message_info['t_consts'] = t_consts
                message_info['s_shape_ls'] = s_shape_ls
                message_info['t_shape'] = t_shape
                message_info['rule_id'] = ridx
                print(message_info)
                self.message_infos[(ridx, i)] = message_info
        
    def forward(self, logits):
        device = list(logits.values())["0"].device

        def influence(message_info, q_values):
            einsum = message_info['einsum']
            expand = message_info['expand']
            export = message_info['export']
            s_pn_ls = message_info['s_pn_ls']
            t_pn = message_info['t_pn']
            s_vals_ls = message_info['s_vals_ls']
            t_vals = message_info['t_vals']
            s_consts_ls = message_info['s_consts_ls']
            t_consts = message_info['t_consts']
            rule_id = message_info['rule_id']

            s_q = []
            for pn, vals, consts in zip(s_pn_ls, s_vals_ls, s_consts_ls):
                q_val = q_values[pn]
                vals = 1 - vals

                for idx, const in enumerate(consts):
                    if const is not None:
                        q_val = torch.index_select(q_val, idx, torch.tensor(const).to(device))
                        q_val = q_val.squeeze(idx)
                s_q.append(q_val)
                s_q.append(vals)
            s_q.append(t_vals)
            
            msg = einsum(*s_q) * self.weights[rule_id]
            msg = msg if expand is None else self.expand(msg, expand)
            msg = msg if export is None else self.export(msg, export)

            return t_pn, msg

        msgs = []
        cur_logits = logits
        for step in range(self.nstep):
            q_values = {pn: F.softmax(cur_logits[pn], dim=-1) for pn in logits}
            
            msg = {pn: torch.zeros_like(logits[pn]) for pn in logits}
            for _, message_info in self.message_infos.items():
                _pn, _msg = influence(message_info, q_values)
                msg[_pn] += _msg
                
            cur_logits = {pn: logits[pn] + msg[pn] for pn in logits}
            msgs.append(msg)

        return cur_logits, msgs

    def expand(self, x, exclude_args):
        n = len(x.size())
        for idx, dim in exclude_args:
            x = torch.unsqueeze(x, idx)
            n = n + 1
            x = x.expand([dim if z == idx else -1 for z in range(n)])
        return x

    def export(self, x, consts):
        n = len(x.size())
        shape = x.size()
        device = x.device
        for idx, (dim, const) in consts:
            x = torch.unsqueeze(x, idx)
            n = n + 1
            x = x.repeat([dim if z == idx else 1 for z in range(n)])
            index = [i for i in range(dim) if i != const]
            if index:
                index = torch.tensor(index).to(device)
                x.index_fill_(idx, index, 0)
        return x

def main():

    # step 1: define the fields
    person = Field('person', ['Tom', 'Mike'])
    paper = Field('paper', ['p0', 'p1'])
    clazz = Field('class', ['c0', 'c1', 'c2'])
    

    # step 2: define the predicates
    smoke = Predicate('smoke', [person])
    cancer = Predicate('cancer', [person])
    friend = Predicate('friend', [person, person])
    cite = Predicate('cite', [paper, paper])
    ptype = Predicate('ptype', [paper], ['rl', 'ec', 'ml'])

    name2predicate = {'smoke': smoke, 'friend': friend, 'ptype': ptype, 'cancer': cancer}


    fstr = 'smoke(Tom)@(true)'
    fstr = 'ptype(p0)@(rl, ec)'
    fstr = 'friend(Tom, Mike)@(false)'
    ftree = pyparsing_parse(fstr)
    formula = construct(ftree, name2predicate)
    print(formula)

    fstr = "a(s)@0 -> b(s) -> c(sdf, ss)"
    fstr = "forall s: a(s) -> b(s) -> c(sdf, ss)"
    fstr = 'exists s: a(0.s)@(0,s_.0) -> b(s)'
    fstr = 'forall fdsfs: pss(fdsfs)'

    fstr = 'forall x: forall y: smoke(x) & friend(x, y) -> smoke(y)'
    ftree = pyparsing_parse(fstr)
    print(ftree)
    formula = construct(ftree, name2predicate)
    print(formula)
    res = is_conjunctive_normal_form(formula)
    print(res)
    formula = formula.apply()
    print(formula)
    res = is_conjunctive_normal_form(formula)
    print(res)
    print([str(x) for x in formula.clauses()])
    f1 = formula

    fstr = 'forall x: (smoke(x) -> cancer(x)) & (cancer(x) -> smoke(x))'
    ftree = pyparsing_parse(fstr)
    print(ftree)
    formula = construct(ftree, name2predicate)
    print(formula)
    res = is_conjunctive_normal_form(formula)
    print(res)
    formula = formula.apply()
    print(formula)
    res = is_conjunctive_normal_form(formula)
    print(res)
    print([str(x) for x in formula.clauses()])
    f2 = formula

    logicmp = LogicMP(f1.clauses() + f2.clauses(), 5)


main()
