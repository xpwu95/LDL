"""
    MAUCpy
    ~~~~~~

    Contains two equations from Hand and Till's 2001 paper on a multi-class
    approach to the AUC. The a_value() function is the probabilistic approximation
    of the AUC found in equation 3, while MAUC() is the pairwise averaging of this
    value for each of the classes. This is equation 7 in their paper.
"""
import itertools

def a_value(probabilities, zero_label=0, one_label=1):
    """
    Approximates the AUC by the method described in Hand and Till 2001,
    equation 3.

    NB: The class labels should be in the set [0,n-1] where n = # of classes.
    The class probability should be at the index of its label in the
    probability list.

    I.e. With 3 classes the labels should be 0, 1, 2. The class probability
    for class '1' will be found in index 1 in the class probability list
    wrapped inside the zipped list with the labels.

    Args:
        probabilities (list): A zipped list of the labels and the
            class probabilities in the form (m = # data instances):
             [(label1, [p(x1c1), p(x1c2), ... p(x1cn)]),
              (label2, [p(x2c1), p(x2c2), ... p(x2cn)])
                             ...
              (labelm, [p(xmc1), p(xmc2), ... (pxmcn)])
             ]
        zero_label (optional, int): The label to use as the class '0'.
            Must be an integer, see above for details.
        one_label (optional, int): The label to use as the class '1'.
            Must be an integer, see above for details.

    Returns:
        The A-value as a floating point.
    """
    # Obtain a list of the probabilities for the specified zero label class
    expanded_points = []
    for instance in probabilities:
        if instance[0] == zero_label or instance[0] == one_label:
            expanded_points.append((instance[0], instance[1][zero_label]))
    sorted_ranks = sorted(expanded_points, key=lambda x: x[1])

    n0, n1, sum_ranks = 0, 0, 0
    # Iterate through ranks and increment counters for overall count and ranks of class 0
    for index, point in enumerate(sorted_ranks):
        if point[0] == zero_label:
            n0 += 1
            sum_ranks += index + 1  # Add 1 as ranks are one-based
        elif point[0] == one_label:
            n1 += 1
        else:
            pass  # Not interested in this class

    return (sum_ranks - (n0*(n0+1)/2.0)) / float(n0 * n1)  # Eqn 3


def MAUC(data, num_classes):
    """
    Calculates the MAUC over a set of multi-class probabilities and
    their labels. This is equation 7 in Hand and Till's 2001 paper.

    NB: The class labels should be in the set [0,n-1] where n = # of classes.
    The class probability should be at the index of its label in the
    probability list.

    I.e. With 3 classes the labels should be 0, 1, 2. The class probability
    for class '1' will be found in index 1 in the class probability list
    wrapped inside the zipped list with the labels.

    Args:
        data (list): A zipped list (NOT A GENERATOR) of the labels and the
            class probabilities in the form (m = # data instances):
             [(label1, [p(x1c1), p(x1c2), ... p(x1cn)]),
              (label2, [p(x2c1), p(x2c2), ... p(x2cn)])
                             ...
              (labelm, [p(xmc1), p(xmc2), ... (pxmcn)])
             ]
        num_classes (int): The number of classes in the dataset.

    Returns:
        The MAUC as a floating point value.
    """
    # Find all pairwise comparisons of labels
    class_pairs = [x for x in itertools.combinations(range(num_classes), 2)]

    # Have to take average of A value with both classes acting as label 0 as this
    # gives different outputs for more than 2 classes
    sum_avals = 0
    for pairing in class_pairs:
        sum_avals += (a_value(data, zero_label=pairing[0], one_label=pairing[1]) +
                      a_value(data, zero_label=pairing[1], one_label=pairing[0]))# / 2.0

    #sum_avals = sum_avals * 2
    sum_avals = sum_avals / float(num_classes)
    sum_avals = sum_avals / float(num_classes - 1)
    return sum_avals
    #return sum_avals * (2 / float(num_classes * (num_classes-1)))  # Eqn 7