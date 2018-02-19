"""
CS 6375
Problem Set 5
Prof. Nick Ruozzi
Student: Tri M. Cao
Date: 25 November 2017
"""
import numpy as np
import itertools
from scipy.sparse.csgraph import minimum_spanning_tree

# CONSTANTS
parent_dict = {0: None, 4: 0, 11: 0, 14: 4, 5: 4, 15: 5, 6: 5, 8: 5, 12: 5,
            9: 5, 13: 5, 7: 8, 3: 8, 1: 12, 2: 13, 16: 7, 10: 2}
child_dict = {0: {4,11}, 4: {5,14}, 5: {15,6,8,12,9,13}, 8: {7,3}, 7: {16},
            13: {2}, 2: {10}, 12: {1}}
MISS = -99

def read_in(file_path):
    X = []
    y = []
    with open(file_path, 'r') as f:
        not_missing = []
        idx = 0
        for line in f:
            missing = False
            info = line.strip('\n').split(',')
            # label is the first column
            if info[0] == 'republican':
                y.append(1)
            else:
                y.append(0)
            feature = []
            for i in range(1, len(info)):
                if info[i] == 'y':
                    feature.append(1)
                elif info[i] == 'n':
                    feature.append(0)
                else:
                    feature.append(-99)
                    missing = True
            X.append(feature)
            if not missing:
                not_missing.append(idx)
            idx += 1
    X = np.array(X)
    y = np.array(y)
    # test using 0 and 1 in this assignment
    # y[y=='republican'] = 1
    # y[y=='democrat'] = 0
    return X, y, np.array(not_missing)

def mutual_info(X, i, j):
    """
    Compute the mutual information between variable i and j in dataset X.
    """
    values_i = set(X[:,i])
    values_j = set(X[:,j])
    info = 0
    N = X.shape[0] # number of samples
    for val_i in values_i:
        for val_j in values_j:
            idx_i = X[:,i] == val_i
            idx_j = X[:,j] == val_j
            idx_ij = idx_i & idx_j
            info += np.sum(idx_ij)/N * np.log((np.sum(idx_ij)/N) / ((np.sum(idx_i)/N)*(np.sum(idx_j)/N)))
            # print(info)
    return info

def chow_liu_tree(data):
    """
    Find the maximum spanning tree
    """
    num_vars = data.shape[1]
    graph_matrix = np.zeros((num_vars, num_vars))
    for i in range(num_vars):
        for j in range(num_vars):
            if i != j:
                info = mutual_info(data, i, j)
                graph_matrix[i,j] = -info
                graph_matrix[j,i] = -info
    tree = minimum_spanning_tree(graph_matrix).toarray()
    # print the edges of the Chow-Liu tree
    print('Printing the edges of the found tree.')
    for i in range(num_vars):
        for j in range(num_vars):
            if tree[i,j] != 0:
                print(i,j)


def em(data, num_iters=10, random_seed=0):
    """
    The main method for EM algorithm.
    """
    D = data.shape[1] # number of features
    N = data.shape[0] # number of samples
    def prob_non_missing(sample, missing_vars, theta):
        """
        Compute the probability for non missing variables
        """
        prob = theta[0]
        for var in range(1,D):
            parent_var = parent_dict[var]
            if var not in missing_vars and parent_var not in missing_vars:
                parent_val = sample[parent_dict[var]]
                if sample[var] == 1:
                    prob *= theta[var][parent_val]
                else:
                    prob *= (1 - theta[var][parent_val])
        return prob

    def gen_missing_var_dist(sample, theta):
        """
        Compute missing variables' probability distribution.
        Also return the log probability of the sample.
        """
        missing_vars = np.where(sample==MISS)[0]
        possible_assigns = list(itertools.product(*[range(2) for i in range(len(missing_vars))]))
        prob = prob_non_missing(sample, missing_vars, theta)
        q_dist = {var: {} for var in missing_vars}
        for var in q_dist:
            q_dist[var] = {i: 0 for i in range(2)}
        sum_prob = 0
        for each_assign in possible_assigns:
            current_sample = np.array(sample)
            current_sample[missing_vars] = each_assign
            #print(current_sample)
            # calculate the probability
            current_prob = prob
            for var in missing_vars:
                # add probability P(var|parent(var))
                parent_val = current_sample[parent_dict[var]]
                if current_sample[var] == 1:
                    current_prob *= theta[var][parent_val]
                else:
                    current_prob *= 1 - theta[var][parent_val]
                # add probability P(child(var)|var)
                if var in child_dict:
                    var_val = current_sample[var]
                    for each_child in child_dict[var]:
                        child_val = current_sample[each_child]
                        if child_val == 1:
                            current_prob *= theta[each_child][var_val]
                        else:
                            current_prob *= 1 - theta[each_child][var_val]
            # update the q distribution
            for i, var in enumerate(missing_vars):
                q_dist[var][each_assign[i]] += current_prob
            # add to sum_prob
            sum_prob += current_prob
        # normalize the q_dist
        for var in q_dist:
            total = q_dist[var][0] + q_dist[var][1]
            q_dist[var][0] /= total
            q_dist[var][1] /= total
        return q_dist, np.log(sum_prob)

    def maximization(data, missing_dist, theta):
        """
        Update the parameters of the model
        """
        # initialize a count dict to count the variables.
        count = {}
        count[0] = {}
        count[0][0] = np.mean(data[:,0]==0)
        count[0][1] = 1 - count[0][0]
        for var in range(1, D):
            count[var] = {}
            for val_parent in range(2):
                count[var][(0,val_parent)] = 0
                count[var][(1,val_parent)] = 0
        # print(count)
        # update the count using the distribution computed in the E-step
        for j, sample in enumerate(data):
            q_dist = missing_distribution[j]
            for var in range(1,D):
                parent_var = parent_dict[var]
                if var in q_dist and parent_var not in q_dist:
                    val_parent = sample[parent_var]
                    for val_child in range(2):
                        count[var][(val_child, val_parent)] += q_dist[var][val_child]
                elif var not in q_dist and parent_var in q_dist:
                    val_child = sample[var]
                    for val_parent in range(2):
                        count[var][(val_child, val_parent)] += q_dist[parent_var][val_parent]
                elif var in q_dist and parent_var in q_dist:
                    for val_child in range(2):
                        for val_parent in range(2):
                            count[var][(val_child, val_parent)] += q_dist[var][val_child]*q_dist[parent_var][val_parent]
                else:
                    val_child = sample[var]
                    val_parent = sample[parent_var]
                    count[var][(val_child,val_parent)] += 1
        # normalize the count to get probability distributions
        for var in range(1,D):
            for val_parent in range(2):
                total = count[var][(0,val_parent)] + count[var][(1,val_parent)]
                count[var][(0,val_parent)] /= total
                count[var][(1,val_parent)] /= total

        # update theta using theta_up
        theta[0] = count[0][1]
        for var in range(1,D):
            for val_parent in range(2):
                theta[var][val_parent] = count[var][(1,val_parent)]
        return theta
    # initialize model probabilities
    np.random.seed(random_seed)
    theta = {}
    theta[0] = np.random.random()
    for var in range(1,D):
        theta[var] = {}
        for val_parent in range(2):
            theta[var][val_parent] = np.random.random()
    # run EM
    for i in range(num_iters):
        print('iteration', i)
        log_likelihood = 0
        # E-step
        missing_distribution = []
        for i in range(N):
            q_dist, sample_prob = gen_missing_var_dist(data[i], theta)
            missing_distribution.append(q_dist)
            log_likelihood += sample_prob
        print('log likelihood:', log_likelihood)
        # M-step
        theta = maximization(data, missing_distribution, theta)
    # compute final log likelihood
    log_likelihood = 0
    for i in range(N):
        q_dist, sample_prob = gen_missing_var_dist(data[i], theta)
        log_likelihood += sample_prob
    print('final log likelihood:', log_likelihood)
    return theta

def main():
    # read in the data
    X, y, not_missing = read_in('hw5_data/congress.data')
    data = np.zeros((X.shape[0], X.shape[1]+1), dtype=np.int64)
    data[:,0] = y
    data[:,1:] = X
    full_data = data[not_missing]
    # Chow-Liu tree
    chow_liu_tree(full_data)
    # Expectation-Maximization
    print()
    D = data.shape[1] # number of features
    N = data.shape[0] # number of samples
    MISS = -99
    # missing probabilities
    b = np.zeros(data.shape[1])
    for i in range(D):
        b[i] = np.sum(data[:,i]==MISS) / N
    print('Start EM algorithm...')
    theta = em(data)
    print()
    print('The model parameters are:')
    for var in theta:
        print(var, theta[var])
    print('Missingness probabilities:')
    print(b)

if __name__ == '__main__':
    main()
