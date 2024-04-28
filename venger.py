import numpy as np
from queue import PriorityQueue
import time

start_time = time.time()

class Venger_solve2:
    def __init__(self, weights):
        weights = np.array(weights).astype(np.float32)
        self.weights = weights
        self.n, self.m = weights.shape
        assert self.n <= self.m

        self.label_x = np.max(weights, axis=1)
        self.label_y = np.zeros((self.m,), dtype=np.float32)

        self.max_match = 0
        self.xy = -np.ones((self.n,), dtype=np.int32)
        self.yx = -np.ones((self.m,), dtype=np.int32)

    def do_augment(self, x, y):
        self.max_match += 1
        while x != -2:
            self.yx[y] = x
            ty = self.xy[x]
            self.xy[x] = y
            x, y = self.prev[x], ty

    def find_augment_path(self):
        self.S = np.zeros((self.n,), np.bool_)
        self.T = np.zeros((self.m,), np.bool_)

        self.slack = np.zeros((self.m,), dtype=np.float32)
        self.slackyx = -np.ones((self.m,), dtype=np.int32)

        self.prev = -np.ones((self.n,), np.int32)

        root = -1

        for x in range(self.n):
            if self.xy[x] == -1:
                root = x
                self.prev[x] = -2
                self.S[x] = True
                break

        self.slack = self.label_y + self.label_x[root] - self.weights[root]
        self.slackyx[:] = root

        while True:
            queue = PriorityQueue()  # Создаем очередь с приоритетами
            queue.put((0, root))

            while not queue.empty():
                _, x = queue.get()

                is_in_graph = np.isclose(self.weights[x], self.label_x[x] + self.label_y)
                nonzero_inds = np.nonzero(np.logical_and(is_in_graph, np.logical_not(self.T)))[0]

                for y in nonzero_inds:
                    if self.yx[y] == -1:
                        return x, y
                    self.T[y] = True
                    queue.put((self.slack[y], self.yx[y]))
                    self.add_to_tree(self.yx[y], x)

            self.update_labels()
            is_in_graph = np.isclose(self.slack, 0)
            nonzero_inds = np.nonzero(np.logical_and(is_in_graph, np.logical_not(self.T)))[0]

            for y in nonzero_inds:
                x = self.slackyx[y]
                if self.yx[y] == -1:
                    return x, y
                self.T[y] = True
                if not self.S[self.yx[y]]:
                    self.add_to_tree(self.yx[y], x)


    def solve(self, verbose=False):

        history = []
        sum_hist = []

        while self.max_match < self.n:
            x, y = self.find_augment_path()
            self.do_augment(x, y)
            history.append([self.xy[x] for x in range(self.n)])

            sum = 0

            for x in range(self.n):
                sum += self.weights[x, self.xy[x]]

            sum_hist.append(sum)

        sum = 0.
        for x in range(self.n):
            sum += self.weights[x, self.xy[x]]

        if verbose:
            print(history[-1:])
            print(sum_hist[-1:])

        self.best = sum
        return history[-1:]

    def add_to_tree(self, x, prevx):
        self.S[x] = True
        self.prev[x] = prevx

        better_slack_idx = self.label_x[x] + self.label_y - self.weights[x] < self.slack
        self.slack[better_slack_idx] = self.label_x[x] + self.label_y[better_slack_idx] - self.weights[
            x, better_slack_idx]
        self.slackyx[better_slack_idx] = x

    def update_labels(self):
        delta = self.slack[np.logical_not(self.T)].min()
        self.label_x[self.S] -= delta
        self.label_y[self.T] += delta
        self.slack[np.logical_not(self.T)] -= delta


def solve_assignment_problem(cost_matrix):
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    assignment_matrix = np.zeros_like(cost_matrix, dtype=int)
    assignment_matrix[row_ind, col_ind] = 1
    return assignment_matrix


def probabilities_to_costs(probabilities):
    return 1 - probabilities


from scipy.optimize import linear_sum_assignment


def split(matrix):
  a = matrix[:-1]
  b = matrix[-1]
  return a, b

def read_probability_matrix(file_path):
    probability_matrix = np.loadtxt(file_path, delimiter=',', dtype=float)

    return probability_matrix

def sumColumn(m):
    return [sum(col) for col in zip(*m)]

def read_solve_and_write(file_input, file_output):

    read_matrix = read_probability_matrix(file_input)
    Compatibility_Matrix, vector_of_weights = split(read_matrix)

    from copy import copy, deepcopy
    Compatibility_Matrix[Compatibility_Matrix == 0] = -1
    Compatibility_Matrix[Compatibility_Matrix == 1] = 0
    Compatibility_Matrix[Compatibility_Matrix == -1] = 1
    CM = deepcopy(-1*Compatibility_Matrix)
    for i in range(CM.shape[0]):
        for j in range(CM.shape[1]):
            if j < i:
                CM[i, j] = 0
            else:
                if j == i:
                    CM[i, j] = 0
                else:
                    continue

    resulting_matrix = np.multiply(CM,vector_of_weights)

    resulting_matrix[resulting_matrix == 0] = -110

    matcher = Venger_solve2(resulting_matrix)
    assignment_cost = matcher.solve(verbose=True)

    vector = np.array(assignment_cost)

    for num_solv in range(len(vector)):

        num_classes = len(vector[num_solv])
        max_value = np.max(vector[num_solv])

        one_hot_matrix = np.zeros((num_classes, max_value + 1), dtype=float)

        for i, val in enumerate(vector[num_solv]):
            one_hot_matrix[i, val] = 1
        resulting_binar_matrix = np.multiply(CM, one_hot_matrix)

        s = sumColumn(resulting_binar_matrix)
        one = (1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1)

        result = [a + b for a, b in zip(s, one)]
        result_sum = sum (vector_of_weights * result)

        print(result)
        print(result_sum)
        result.append(result_sum)
        print(result)

        np.savetxt(file_output.replace('@', f'{num_solv}'), result, fmt='%1.3f, ')
    return resulting_matrix
file_path = "input.csv"
file_output = "output.csv"

cost_matrix = read_solve_and_write(file_path, file_output)

#import time

start_time1 = time.time()

matcher = Venger_solve2(cost_matrix)
assignment_cost = matcher.solve(verbose=True)

end_time = time.time()
end_time1 = time.time()

execution_time = end_time - start_time
execution_time1 = end_time1 - start_time1
print("Execution time:", execution_time, "seconds")
print("Execution time:", execution_time1, "seconds")

'''import time


start_time = time.time()

#print()
res = solve_assignment_problem(probabilities_to_costs(cost_matrix))
end_time = time.time()
summ = sumColumn(res)
execution_time = end_time - start_time
print("Execution time:", execution_time, "seconds")
print("res:", res)
print("summ:", summ)'''

