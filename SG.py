"""
This python file implement AEQUITAS.
"""

import numpy as np
import tensorflow as tf
import generation_utilities
# from lime import lime_tabular
from adf_baseline.lime import lime_tabular
from sklearn.tree import DecisionTreeClassifier
from z3 import Solver, sat, Int
import copy
from queue import PriorityQueue

def model_argmax(model, samples):
    """
    Given a model and data, return the argmax output.
    :param model: model
    :param samples: input data
    :return: argmax over model outputs
    """
    preds = model(samples)
    return (preds > 0.5).numpy().astype(int)

def getPath(X, model, input, conf):
    """
    Get the path from Local Interpretable Model-agnostic Explanation Tree
    :param X: the whole inputs
    :param model: TensorFlow 2 model
    :param input: instance to interpret
    :return: the path for the decision of given instance
    """
    explainer = lime_tabular.LimeTabularExplainer(X,
                                                  feature_names=conf['feature_name'],
                                                  class_names=conf['class_name'],
                                                  categorical_features=conf['categorical_features'],
                                                  discretize_continuous=True)
    g_data = explainer.generate_instance(input, num_samples=5000)
    g_labels = model_argmax(model, g_data)

    # build the interpretable tree
    tree = DecisionTreeClassifier(random_state=2019) #min_samples_split=0.05, min_samples_leaf =0.01
    tree.fit(g_data, g_labels)

    # get the path for decision
    path_index = tree.decision_path(np.array([input])).indices
    path = []
    for i in range(len(path_index)):
        node = path_index[i]
        i = i + 1
        f = tree.tree_.feature[node]
        if f != -2:
            left_count = tree.tree_.n_node_samples[tree.tree_.children_left[node]]
            right_count = tree.tree_.n_node_samples[tree.tree_.children_right[node]]
            left_confidence = 1.0 * left_count / (left_count + right_count)
            right_confidence = 1.0 - left_confidence
            if tree.tree_.children_left[node] == path_index[i]:
                path.append([f, "<=", tree.tree_.threshold[node], left_confidence])
            else:
                path.append([f, ">", tree.tree_.threshold[node], right_confidence])
    return path

def global_solve(path_constraint, arguments, t, constraint):
    """
    Solve the constraint for global generation
    :param path_constraint: the constraint of path
    :param arguments: the name of features in path_constraint
    :param t: test case
    :param conf: the configuration of dataset
    :return: new instance through global generation
    """
    s = Solver()
    for c in path_constraint:
        # s.add(arguments[c[0]] >= conf.input_bounds[c[0]][0])
        # s.add(arguments[c[0]] <= conf.input_bounds[c[0]][1])
        s.add(arguments[c[0]] >= constraint.tolist()[c[0]][0])
        s.add(arguments[c[0]] <= constraint.tolist()[c[0]][1])
        if c[1] == "<=":
            s.add(arguments[c[0]] <= c[2])
        else:
            s.add(arguments[c[0]] > c[2])

    if s.check() == sat:
        m = s.model()
    else:
        return None

    tnew = copy.deepcopy(t)
    # tnew = np.array(copy.deepcopy(t)) if isinstance(t, list) else np.copy(t.numpy())
    for i in range(len(arguments)):
        if m[arguments[i]] == None:
            continue
        else:
            tnew[i] = int(m[arguments[i]].as_long())
    return tnew.astype('int').tolist()

def local_solve(path_constraint, arguments, t, index, constraint):
    """
    Solve the constraint for local generation
    :param path_constraint: the constraint of path
    :param arguments: the name of features in path_constraint
    :param t: test case
    :param index: the index of constraint for local generation
    :param conf: the configuration of dataset
    :return: new instance through global generation
    """
    c = path_constraint[index]
    s = Solver()
    # s.add(arguments[c[0]] >= conf.input_bounds[c[0]][0])
    # s.add(arguments[c[0]] <= conf.input_bounds[c[0]][1])
    s.add(arguments[c[0]] >= constraint.tolist()[c[0]][0])
    s.add(arguments[c[0]] <= constraint.tolist()[c[0]][1])
    for i in range(len(path_constraint)):
        if path_constraint[i][0] == c[0]:
            if path_constraint[i][1] == "<=":
                s.add(arguments[path_constraint[i][0]] <= path_constraint[i][2])
            else:
                s.add(arguments[path_constraint[i][0]] > path_constraint[i][2])

    if s.check() == sat:
        m = s.model()
    else:
        return None

    tnew = copy.deepcopy(t)
    tnew[c[0]] = int(m[arguments[c[0]]].as_long())
    return tnew.astype('int').tolist()

def average_confidence(path_constraint):
    """
    The average confidence (probability) of path
    :param path_constraint: the constraint of path
    :return: the average confidence
    """
    r = np.mean(np.array(path_constraint)[:,3].astype(float))
    return r

def gen_arguments(conf):
    """
    Generate the argument for all the features
    :param conf: the configuration of dataset
    :return: a sequence of arguments
    """
    arguments = []
    for i in range(conf['num_attributes']):
        arguments.append(Int(conf['feature_name'][i]))
    return arguments

def symbolic_generation(X, seeds, protected_attribs, constraint, model, limit, conf):
    """
    The implementation of symbolic generation
    """
    # the rank for priority queue, rank1 is for seed inputs, rank2 for local, rank3 for global
    rank1 = 5
    rank2 = 1
    rank3 = 10
    T1 = 0.3

    num_attribs = len(X[0])
    arguments = gen_arguments(conf)

    g_id = np.empty(shape=(0, num_attribs))
    # all_gen_g = np.empty(shape=(0, num_attribs))

    l_id = np.empty(shape=(0, num_attribs))
    # all_gen_l = np.empty(shape=(0, num_attribs))

    try_times = 0
    all_gen_g_l = np.empty(shape=(0, num_attribs))

    tot_inputs = set()

    # select the seed input for fairness testing
    # inputs = seed_test_input(dataset, cluster_num, limit)
    inputs = seeds
    q = PriorityQueue() # low push first
    for inp in inputs[::-1]: # reverse order
        # q.put((rank1,X[inp].tolist()))
        q.put((rank1, inp.tolist()))

    visited_path = []
    l_count = 0
    g_count = 0
    while len(tot_inputs) < limit and q.qsize() != 0:
        t = q.get()
        t_rank = t[0]
        t = np.array(t[1])

        # found = check_for_error_condition(data_config[dataset], sess, x, preds, t, sensitive_param)
        similar_t = generation_utilities.similar_set(t, num_attribs, protected_attribs, constraint)
        found = generation_utilities.is_discriminatory(t, similar_t, model)

        # p = getPath(X, sess, x, preds, t, data_config[dataset])
        p = getPath(X, model, t, conf)
        temp = copy.deepcopy(t.tolist())
        # temp = temp[:sensitive_param - 1] + temp[sensitive_param:]

        tot_inputs.add(tuple(temp))
        all_gen_g_l = np.append(all_gen_g_l, [temp], axis=0)
        if found:
            if t_rank > 2:
                g_id = np.append(g_id, [temp], axis=0)
                # all_gen_g = np.append(all_gen_g, [temp], axis=0)
            else:
                l_id = np.append(l_id, [temp], axis=0)
                # all_gen_l = np.append(all_gen_l, [temp], axis=0)
            if len(tot_inputs) == limit:
                break

            # local search
            for i in range(len(p)):
                try_times += 1
                path_constraint = copy.deepcopy(p)
                c = path_constraint[i]
                # if c[0] == sensitive_param - 1:
                #     continue
                if c[0] in protected_attribs:
                    continue

                if c[1] == "<=":
                    c[1] = ">"
                    c[3] = 1.0 - c[3]
                else:
                    c[1] = "<="
                    c[3] = 1.0 - c[3]

                if path_constraint not in visited_path:
                    visited_path.append(path_constraint)
                    # input = local_solve(path_constraint, arguments, t, i, data_config[dataset])
                    input = local_solve(path_constraint, arguments, t, i, constraint)
                    l_count += 1
                    if input != None:
                        r = average_confidence(path_constraint)
                        q.put((rank2 + r, input))

        # global search
        prefix_pred = []
        for c in p:
            try_times += 1
            # if c[0] == sensitive_param - 1:
            #         continue
            if c[0] in protected_attribs:
                continue
            if c[3] < T1:
                break

            n_c = copy.deepcopy(c)
            if n_c[1] == "<=":
                n_c[1] = ">"
                n_c[3] = 1.0 - c[3]
            else:
                n_c[1] = "<="
                n_c[3] = 1.0 - c[3]
            path_constraint = prefix_pred + [n_c]

            # filter out the path_constraint already solved before
            if path_constraint not in visited_path:
                visited_path.append(path_constraint)
                input = global_solve(path_constraint, arguments, t, constraint)
                g_count += 1
                if input != None:
                    r = average_confidence(path_constraint)
                    q.put((rank3-r, input))

            prefix_pred = prefix_pred + [c]
    # l_id = np.array(list(set([tuple(id) for id in l_id])))
    # g_id = np.array(list(set([tuple(id) for id in g_id])))
    # g_l_id = np.vstack((g_id, l_id))
    l_id = np.unique(l_id, axis=0)
    g_id = np.unique(g_id, axis=0)
    if len(g_id) == 0:
        g_l_id = l_id
    elif len(l_id) == 0:
        g_l_id = g_id
    else:
        g_l_id = np.vstack((g_id, l_id))
    return g_l_id, all_gen_g_l, try_times

def individual_discrimination_generation(X, seeds, protected_attribs, constraint, model, dataset_configuration):
    all_id, all_gen, all_gen_num = symbolic_generation(X, seeds, protected_attribs, constraint, model, limit=len(seeds), conf=dataset_configuration)
    all_id_nondup = np.array(list(set([tuple(id) for id in all_id])))
    all_gen_nondup = np.array(list(set([tuple(gen) for gen in all_gen])))
    return all_id_nondup, all_gen_nondup, all_gen_num