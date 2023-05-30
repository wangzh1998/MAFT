'''
尝试更新全局生成方法，使用AdaGrad方法
'''

import tensorflow as tf
import numpy as np
import generation_utilities


def compute_grad(x, mode):
    pass

# v0 EIDIG Momentum版
def global_generation(X, seeds, num_attribs, protected_attribs, constraint, model, decay, max_iter, s_g):
    # global generation phase of EIDIG

    g_id = np.empty(shape=(0, num_attribs))
    all_gen_g = np.empty(shape=(0, num_attribs))
    try_times = 0
    g_num = len(seeds)
    for i in range(g_num):
        x1 = seeds[i].copy()
        grad1 = np.zeros_like(X[0]).astype(float)
        grad2 = np.zeros_like(X[0]).astype(float)
        for _ in range(max_iter):
            try_times += 1
            similar_x1 = generation_utilities.similar_set(x1, num_attribs, protected_attribs, constraint)
            if generation_utilities.is_discriminatory(x1, similar_x1, model):
                g_id = np.append(g_id, [x1], axis=0)
                break
            x2 = generation_utilities.max_diff(x1, similar_x1, model)
            # 3.2 use momentum
            grad1 = decay * grad1 + compute_grad(x1, model)
            grad2 = decay * grad2 + compute_grad(x2, model)
            direction = np.zeros_like(X[0])
            sign_grad1 = np.sign(grad1)
            sign_grad2 = np.sign(grad2)
            for attrib in range(num_attribs):
                if attrib not in protected_attribs and sign_grad1[attrib] == sign_grad2[attrib]:
                    direction[attrib] = (-1) * sign_grad1[attrib]
            x1 = x1 + s_g * direction
            x1 = generation_utilities.clip(x1, constraint)
            all_gen_g = np.append(all_gen_g, [x1], axis=0)
    g_id = np.array(list(set([tuple(id) for id in g_id])))
    return g_id, all_gen_g, try_times

'''
与Momentum方法比较，AdaGrad没有引入速度变量，而是记录每个参数方向上的梯度的值的平方和.
在该参数方向上步进时除以这个平方和的平方根，则对于原梯度较小学习进展较慢的方向相较于原梯度较大的方向rescale的程度较小，从而加速在该方向上的学习进程。
'''
# Ada-grad 还没改
def global_generation_v1(X, seeds, num_attribs, protected_attribs, constraint, model, decay, max_iter, s_g):
    # global generation phase of EIDIG

    g_id = np.empty(shape=(0, num_attribs))
    all_gen_g = np.empty(shape=(0, num_attribs))
    try_times = 0
    g_num = len(seeds)
    for i in range(g_num):
        x1 = seeds[i].copy()
        grad1 = np.zeros_like(X[0]).astype(float)
        grad2 = np.zeros_like(X[0]).astype(float)
        for _ in range(max_iter):
            try_times += 1
            similar_x1 = generation_utilities.similar_set(x1, num_attribs, protected_attribs, constraint)
            if generation_utilities.is_discriminatory(x1, similar_x1, model):
                g_id = np.append(g_id, [x1], axis=0)
                break
            x2 = generation_utilities.max_diff(x1, similar_x1, model)
            # 3.2 use momentum
            grad1 = decay * grad1 + compute_grad(x1, model)
            grad2 = decay * grad2 + compute_grad(x2, model)
            direction = np.zeros_like(X[0])
            sign_grad1 = np.sign(grad1)
            sign_grad2 = np.sign(grad2)
            for attrib in range(num_attribs):
                if attrib not in protected_attribs and sign_grad1[attrib] == sign_grad2[attrib]:
                    direction[attrib] = (-1) * sign_grad1[attrib]
            x1 = x1 + s_g * direction
            x1 = generation_utilities.clip(x1, constraint)
            all_gen_g = np.append(all_gen_g, [x1], axis=0)
    g_id = np.array(list(set([tuple(id) for id in g_id])))
    return g_id, all_gen_g, try_times