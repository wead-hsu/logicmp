import torch
from model.mean_field_posterior import FactorizedPosterior
from model.gcn import GCN, TrainableEmbedding
from data_process.dataset import Dataset
from common.cmd_args import cmd_args
from tqdm import tqdm
import torch.optim as optim
from model.graph import KnowledgeGraph
from common.predicate import PRED_DICT
from common.utils import EarlyStopMonitor, get_lr, count_parameters
from common.evaluate import gen_eval_query
from itertools import chain
import random
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from os.path import join as joinpath
import os
import math
from collections import Counter
from model.logicmp import LogicMP

import logging
logging.basicConfig(level=logging.DEBUG, datefmt='%m/%d/%Y %I:%M:%S %p',
                    format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s')
logger = logging.getLogger(__name__)


def train(cmd_args):
    if not os.path.exists(cmd_args.exp_path):
        os.makedirs(cmd_args.exp_path)

    with open(joinpath(cmd_args.exp_path, 'options.txt'), 'w') as f:
        param_dict = vars(cmd_args)
        for param in param_dict:
            f.write(param + ' = ' + str(param_dict[param]) + '\n')

    logpath = joinpath(cmd_args.exp_path, 'eval.result')
    param_cnt_path = joinpath(cmd_args.exp_path, 'param_count.txt')

    # dataset and KG
    dataset = Dataset(cmd_args.data_root, cmd_args.batchsize,
                      cmd_args.shuffle_sampling, load_method=cmd_args.load_method)
    kg = KnowledgeGraph(dataset.fact_dict, PRED_DICT, dataset)

    # model
    if cmd_args.use_gcn == 1:
        gcn = GCN(kg, cmd_args.embedding_size - cmd_args.gcn_free_size, cmd_args.gcn_free_size,
                  num_hops=cmd_args.num_hops, num_layers=cmd_args.num_mlp_layers,
                  transductive=cmd_args.trans == 1).to(cmd_args.device)
    else:
        gcn = TrainableEmbedding(
            kg, cmd_args.embedding_size).to(cmd_args.device)
    posterior_model = FactorizedPosterior(
        kg, cmd_args.embedding_size, cmd_args.slice_dim).to(cmd_args.device)

    if cmd_args.model_load_path is not None:
        gcn.load_state_dict(torch.load(
            joinpath(cmd_args.model_load_path, 'gcn.model')))
        posterior_model.load_state_dict(torch.load(
            joinpath(cmd_args.model_load_path, 'posterior.model')))

    # optimizers
    monitor = EarlyStopMonitor(cmd_args.patience)
    all_params = chain.from_iterable(
        [posterior_model.parameters(), gcn.parameters()])
    optimizer = optim.Adam(
        all_params, lr=cmd_args.learning_rate, weight_decay=cmd_args.l2_coef)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=cmd_args.lr_decay_factor,
                                                     patience=cmd_args.lr_decay_patience, min_lr=cmd_args.lr_decay_min)

    with open(param_cnt_path, 'w') as f:
        cnt_gcn_params = count_parameters(gcn)
        cnt_posterior_params = count_parameters(posterior_model)
        if cmd_args.use_gcn == 1:
            f.write('GCN params count: %d\n' % cnt_gcn_params)
        elif cmd_args.use_gcn == 0:
            f.write('plain params count: %d\n' % cnt_gcn_params)
        f.write('posterior params count: %d\n' % cnt_posterior_params)
        f.write('Total params count: %d\n' %
                (cnt_gcn_params + cnt_posterior_params))

    if cmd_args.no_train == 1:
        cmd_args.num_epochs = 0

    # for Kinship / UW-CSE / Cora data
    if cmd_args.load_method == 0:
        logger.info("Start training")
        for current_epoch in range(cmd_args.num_epochs):
            pbar = tqdm(range(cmd_args.num_batches))
            acc_loss = 0.0

            for k in pbar:
                import time
                cur = time.time()
                node_embeds = gcn(dataset)

                batch_neg_mask, flat_list, batch_latent_var_inds, observed_rule_cnts, batch_observed_vars, batch_sub = dataset.get_batch_rnd(
                    observed_prob=cmd_args.observed_prob,
                    filter_latent=cmd_args.filter_latent == 1,
                    closed_world=cmd_args.closed_world == 1,
                    filter_observed=1)

                inputs = dataset.generate_logiccrf_input_dense(
                    batch_sub, posterior_model, node_embeds)
                batch_const2idx_dict, batch_ifact_dict, batch_logits, batch_masks, batch_observed, batch_labels, flat_logits, _flat_list = inputs

                if cmd_args.no_entropy == 1:
                    entropy = 0
                else:
                    _keys = sorted(list(flat_logits.keys()))
                    _logits = torch.cat([flat_logits[k] for k in _keys], dim=0)
                    _flat_list = [(k, tuple(a))
                                  for k in _keys for a in _flat_list[k]]
                    posterior_prob = torch.sigmoid(_logits)
                    _mask, _seen = [], set()
                    for f in _flat_list:
                        if f not in _seen:
                            _mask.append(True)
                            _seen.add(f)
                        else:
                            _mask.append(False)
                    _posterior_prob = posterior_prob[_mask]
                    entropy = compute_entropy(
                        _posterior_prob) / cmd_args.entropy_temp

                entropy = entropy

                scores = torch.zeros(len(dataset.rule_ls), dtype=torch.float)

                _score = 0
                for ri, rule in enumerate(dataset.rule_ls):
                    with torch.no_grad():
                        if not batch_sub[ri]:
                            continue

                        const2idx_dict = batch_const2idx_dict[ri]
                        vartype2dim = {
                            vt: len(const2idx_dict[vt]) for vt in const2idx_dict}

                        pns = [atom.pred_name for atom in rule.atom_ls]

                        args_ls = []
                        for atom in rule.atom_ls:
                            _args = []
                            for var_name, var_type in zip(atom.var_name_ls, atom.var_type_ls):
                                if var_name[0].isupper() and len(var_name) > 1:
                                    _args.append(
                                        const2idx_dict[var_type][var_name])
                                else:
                                    _args.append(var_name)
                            args_ls.append(_args)

                        vals_ls = [[1, 0] if atom.neg else [0, 1]
                                   for atom in rule.atom_ls]
                        preds = [PRED_DICT[pn] for pn in pns]

                        rule_type = ri
                        _rule = [rule, const2idx_dict, rule_type]
                        logicmp = LogicMP([_rule], nstep=cmd_args.iterations, nlabel=2, weights=torch.ones(
                            1)).to(cmd_args.device)

                        masks = batch_masks[ri]
                        observed = batch_observed[ri]

                        logits = {}
                        labels = {}
                        for pn in batch_logits[ri]:
                            x = batch_logits[ri][pn]
                            d = x.device
                            logits[pn] = torch.stack(
                                [torch.zeros_like(x), x], dim=-1)
                            x = batch_labels[ri][pn]
                            d = x.device
                            labels[pn] = torch.stack(
                                [torch.zeros_like(x), x], dim=-1)
                        rlogits, msgs = logicmp.forward(logits=logits,
                                                        observed=observed,
                                                        labels=labels,
                                                        masks=masks)

                    msg = msgs[cmd_args.iterations-1]
                    for pn in msg:
                        _msg = msg[pn]
                        _msg = _msg.reshape(-1, 2)
                        _msg = _msg[:, 1] - _msg[:, 0]
                        _hid = masks[pn] - observed[pn]
                        _hid = _hid.reshape(-1)
                        _lgt = batch_logits[ri][pn].reshape(-1)
                        _prb = torch.sigmoid(_lgt)

                        _msg = _msg[_hid == 1]
                        _prb = _prb[_hid == 1]
                        _score += (_msg * _prb).sum(dim=0)

                potential = _score + sum(observed_rule_cnts)
                cur = time.time()

                optimizer.zero_grad()

                loss = - (potential + entropy) / cmd_args.batchsize
                acc_loss += loss.item()

                loss.backward()

                optimizer.step()

                pbar.set_description('train loss: %.4f, lr: %.4g' % (
                    acc_loss / (k + 1), get_lr(optimizer)))

            # test
            node_embeds = gcn(dataset)
            with torch.no_grad():

                posterior_prob = posterior_model(
                    [(e[1], e[2]) for e in dataset.test_fact_ls], node_embeds)
                posterior_prob = posterior_prob.to('cpu')

                label = np.array([e[0] for e in dataset.test_fact_ls])
                test_log_prob = float(np.sum(
                    np.log(np.clip(np.abs((1 - label) - posterior_prob.numpy()), 1e-6, 1 - 1e-6))))

                auc_roc = roc_auc_score(label, posterior_prob.numpy())
                auc_pr = average_precision_score(label, posterior_prob.numpy())

                tqdm.write('Epoch: %d, train loss: %.4f, test auc-roc: %.4f, test auc-pr: %.4f, test log prob: %.4f' % (
                    current_epoch, loss.item(), auc_roc, auc_pr, test_log_prob))
                # tqdm.write(str(posterior_prob[:10]))

            # validation for early stop
            valid_sample = []
            valid_label = []
            for pred_name in dataset.valid_dict_2:
                for val, consts in dataset.valid_dict_2[pred_name]:
                    valid_sample.append((pred_name, consts))
                    valid_label.append(val)
            valid_label = np.array(valid_label)

            valid_prob = posterior_model(valid_sample, node_embeds)
            valid_prob = valid_prob.to('cpu')

            valid_log_prob = float(np.sum(np.log(
                np.clip(np.abs((1 - valid_label) - valid_prob.numpy()), 1e-6, 1 - 1e-6))))

            # tqdm.write('epoch: %d, valid log prob: %.4f' % (current_epoch, valid_log_prob))
            #
            # should_stop = monitor.update(-valid_log_prob)
            # scheduler.step(valid_log_prob)
            #
            # is_current_best = monitor.cnt == 0
            # if is_current_best:
            #   savepath = joinpath(cmd_args.exp_path, 'saved_model')
            #   os.makedirs(savepath, exist_ok=True)
            #   torch.save(gcn.state_dict(), joinpath(savepath, 'gcn.model'))
            #   torch.save(posterior_model.state_dict(), joinpath(savepath, 'posterior.model'))
            #
            # should_stop = should_stop or (current_epoch + 1 == cmd_args.num_epochs)
            #
            # if should_stop:
            #   tqdm.write('Early stopping')
            #   break

        # evaluation after training
        node_embeds = gcn(dataset)
        with torch.no_grad():
            posterior_prob = posterior_model(
                [(e[1], e[2]) for e in dataset.test_fact_ls], node_embeds)
            posterior_prob = posterior_prob.to('cpu')

            label = np.array([e[0] for e in dataset.test_fact_ls])
            test_log_prob = float(np.sum(
                np.log(np.clip(np.abs((1 - label) - posterior_prob.numpy()), 1e-6, 1 - 1e-6))))

            auc_roc = roc_auc_score(label, posterior_prob.numpy())
            auc_pr = average_precision_score(label, posterior_prob.numpy())

            tqdm.write('test auc-roc: %.4f, test auc-pr: %.4f, test log prob: %.4f' %
                       (auc_roc, auc_pr, test_log_prob))


def compute_entropy(posterior_prob):
    eps = 1e-6
    posterior_prob.clamp_(eps, 1 - eps)
    compl_prob = 1 - posterior_prob
    entropy = -(posterior_prob * torch.log(posterior_prob) +
                compl_prob * torch.log(compl_prob)).sum()
    return entropy


def compute_MB_proba(rule_ls, ls_rule_idx):
    rule_idx_cnt = Counter(ls_rule_idx)
    numerator = 0
    for rule_idx in rule_idx_cnt:
        weight = rule_ls[rule_idx].weight
        cnt = rule_idx_cnt[rule_idx]
        numerator += math.exp(weight * cnt)
    return numerator / (numerator + 1.0)


if __name__ == '__main__':
    random.seed(cmd_args.seed)
    np.random.seed(cmd_args.seed)
    torch.manual_seed(cmd_args.seed)

    train(cmd_args)
