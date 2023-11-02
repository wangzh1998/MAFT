import numpy as np

# generate initial input for AEQUITAS
def generate_instance(constraints):
    """
    Given a set of constraints for categorical attributes,
    generate a random instance within those constraints.

    :param constraints: numpy ndarray of shape (n, 2) where n is the number of attributes.
                        Each row corresponds to an attribute and contains two elements:
                        the minimum and maximum categorical value (inclusive).
    :return: An instance as a numpy array where each element is a random value
             for the corresponding attribute within the specified constraints.
    """
    np.random.seed(0)  # 设置随机种子以保证结果的可复现性
    return np.array([np.random.randint(low, high + 1) for low, high in constraints])