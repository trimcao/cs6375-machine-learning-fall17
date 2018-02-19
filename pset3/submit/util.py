"""
Utilities for ML algorithms
Author: Tri Minh Cao
Email: trimcao@gmail.com
Date: September 2017
"""

import numpy as np

def probability(y, weights=None):
    """
    Compute a probability table for a variable y.
    """
    prob_dict = {}
    choices = set(y)
    if weights is None:
        num_samples = y.shape[0]
        for each in choices:
            prob_dict[each] = np.sum(y==each) / num_samples
    else:
        for each in choices:
            prob_dict[each] = np.sum(weights[y==each])
    return prob_dict


def entropy(y, weights=None):
    """
    Compute entropy.
    """
    prob_dict = probability(y, weights)
    result = 0
    choices = set(y)
    for each in choices:
        result -= prob_dict[each]*log_zero(prob_dict[each])
    return result


def conditional_probability(y, x, weights=None):
    """
    conditional of y given x
    """
    # obtain probability table for x
    prob_x = probability(x, weights)
    # compute conditional probability
    choices_y = set(y)
    choices_x = set(x)
    num_samples = y.shape[0]
    prob_dict = {}
    if weights is None:
        for each_y in choices_y:
            prob_dict[each_y] = {}
            for each_x in choices_x:
                num_true = 0
                for y_cur, x_cur in zip(y, x):
                    # prob_dict[each_y][each_x] = (np.sum(y==each_y & x==each_x)/num_samples) / prob_x[each_x]
                    if y_cur == each_y and x_cur == each_x:
                        num_true += 1
                prob_dict[each_y][each_x] = (num_true/num_samples) / prob_x[each_x]
    else:
        for each_y in choices_y:
            prob_dict[each_y] = {}
            for each_x in choices_x:
                w = 0 # total weights of the current combination (y, x)
                for i, (y_cur, x_cur) in enumerate(zip(y, x)):
                    if y_cur == each_y and x_cur == each_x:
                        w += weights[i]
                prob_dict[each_y][each_x] = w / prob_x[each_x]
    return prob_dict


def log_zero(number):
    if number == 0:
        return 0
    else:
        return np.log(number)


def conditional_entropy(y, x, weights=None):
    """
    Compute conditional entropy.
    """
    prob_x = probability(x, weights)
    prob_y_given_x = conditional_probability(y, x, weights)
    choices_y = set(y)
    choices_x = set(x)
    total = 0
    for each_x in choices_x:
        for each_y in choices_y:
            cur_prob_y_given_x = prob_y_given_x[each_y][each_x]
            total -= prob_x[each_x] * cur_prob_y_given_x * log_zero(cur_prob_y_given_x)
    return total
