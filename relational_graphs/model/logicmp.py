import math

import torch
from torch import nn
from torch.nn import functional as F


# all_grounding using einsum
class LogicMP(nn.Module):
    def __init__(self, rules, nstep=5, weights=None, nlabel=2):
        """
        Create a new instance of RuleCRF layer.

        Args:
            rules: disjunctive formula. [list of prednames, list of args, list of vals, rule_type],
                e.g., '(smoke(a) in [0]) v (friend(a, b) in [0]) v (friend(b) in [1])' can be represented by:
                list of prednames: [smoke, friend, smoke]
                list of args: [(a), (a, b), (b)]
                list of vals: [(1, 0), (1, 0), (1, 0)] for multi-class tasks
                list of types: any type to tag the rule (for conjunction of multiple disjunctive formulae)
            nstep: the number of steps for mean-field iteration.
            weights: a list of rule weights.
            nlabel: the number of classification labels, typically 2.
        """

        super(LogicMP, self).__init__()
        import opt_einsum as oe

        self.rules = rules
        self.nstep = nstep
        self.nlabel = nlabel

        self.rule_types = sorted(set(rule[-1] for rule in rules))
        self.nrule = len(self.rule_types)

        self.message_infos = {}

        for rule_idx, rule in enumerate(self.rules):
            rule, const2idx_dict, rule_type = rule

            vartype2dim = {vt: len(const2idx_dict[vt])
                           for vt in const2idx_dict}
            varname2dim = {}
            for atom in rule.atom_ls:
                for var_name, var_type in zip(atom.var_name_ls, atom.var_type_ls):
                    varname2dim[var_name] = vartype2dim[var_type]

            rlen = len(rule.atom_ls)
            for i in range(rlen):
                s_pn_ls = [atom.pred_name for j,
                           atom in enumerate(rule.atom_ls) if j != i]
                t_pn = rule.atom_ls[i].pred_name

                s_shape_ls, t_shape = [], []
                s_consts_ls, t_consts = [], []
                s_args_ls, t_args = [], []
                for j, atom in enumerate(rule.atom_ls):
                    if j == i:
                        for var_name, var_type in zip(atom.var_name_ls, atom.var_type_ls):
                            if var_name[0].isupper() and len(var_name) > 1:
                                t_consts.append(
                                    [vartype2dim[var_type], const2idx_dict[var_type][var_name]])
                            else:
                                t_consts.append(None)
                                t_shape.append(vartype2dim[var_type])
                                t_args.append(var_name)
                    else:
                        s_shape, s_consts, s_args = [], [], []
                        for var_name, var_type in zip(atom.var_name_ls, atom.var_type_ls):
                            if var_name[0].isupper() and len(var_name) > 1:
                                s_consts.append(
                                    const2idx_dict[var_type][var_name])
                            else:
                                s_consts.append(None)
                                s_shape.append(vartype2dim[var_type])
                                s_args.append(var_name)
                        s_shape_ls.append(s_shape)
                        s_consts_ls.append(s_consts)
                        s_args_ls.append(s_args)

                s_vals_ls = [[[1, 0], [0, 1]][1 - atom.neg]
                             for j, atom in enumerate(rule.atom_ls) if j != i]
                t_vals = [[1, 0], [0, 1]][1 - rule.atom_ls[i].neg]

                s_args_set = set(arg for args in s_args_ls for arg in args)
                einsum_s_str = ','.join(''.join(args) for args in s_args_ls)
                einsum_t_str = ''.join([a for a in t_args if a in s_args_set])
                einsum_str = '->'.join([einsum_s_str, einsum_t_str])

                einsum = oe.contract_expression(einsum_str, *s_shape_ls)
                expand = [(idx, varname2dim[arg])
                          for idx, arg in enumerate(t_args) if arg not in s_args_set]
                export = [(idx, const) for idx, const in enumerate(
                    t_consts) if const is not None]

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
                message_info['rule_id'] = self.rule_types.index(rule_type)
                self.message_infos[(rule_idx, i)] = message_info

        if weights is None:
            weights = torch.ones(self.nrule) * 5

        self.weights = nn.Parameter(weights)

    def forward(self, logits, observed=None, labels=None, masks=None):
        """
        Args:
            logit: dict, logits of size [nnodes, nlabel] for every predicates.
            observed: dict, if the logit is observed.
            labels: dict, the labels of observed data.

        Returns:
            logits,
            messages
        """

        device = list(logits.values())[0].device

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
                if observed:
                    o = observed[pn].unsqueeze(-1)
                    l = labels[pn]
                    q_val = q_val * (1 - o) + l * o
                q_val = q_val * torch.tensor(vals).to(device)
                q_val = torch.sum(q_val, dim=-1)
                q_val = 1 - q_val

                if masks:
                    m = masks[pn]
                    q_val = q_val * m
                for idx, const in enumerate(consts):
                    if const is not None:
                        q_val = torch.index_select(
                            q_val, idx, torch.tensor(const).to(device))
                        q_val = q_val.squeeze(idx)
                        # print(q_val)
                s_q.append(q_val)

            msg = einsum(*s_q) * self.weights[rule_id]
            msg = msg if expand is None else self.expand(msg, expand)
            msg = msg if export is None else self.export(msg, export)
            msg = msg.unsqueeze(-1) * torch.tensor(t_vals).to(device)

            return t_pn, msg

        msgs = []
        cur_logits = logits
        for step in range(self.nstep):
            q_values = {pn: F.softmax(cur_logits[pn], dim=-1) for pn in logits}

            msg = {pn: torch.zeros_like(logits[pn]) for pn in logits}
            for _, message_info in self.message_infos.items():
                # HACK !advisedBy(a, a) v !advisedBy(a, a)
                if message_info['t_args'] == ['a', 'a']:
                    _pn = message_info['t_pn']
                    _rule_id = message_info['rule_id']
                    _tmp = q_values[_pn][:, :, 1]
                    _device = _tmp.device
                    _eye = torch.eye(_tmp.shape[0]).to(_device)
                    _msg = torch.stack([_tmp, torch.zeros_like(_tmp).to(
                        _device)], dim=-1) * self.weights[_rule_id]
                    _msg = _msg * _eye[:, :, None]
                    _msg = _msg * masks[_pn][:, :, None]
                    msg[_pn] += _msg
                else:
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
