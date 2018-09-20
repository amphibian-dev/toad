import numpy as np
import pandas as pd

np.random.seed(1)

LENGTH = 500

A = np.random.rand(LENGTH)
A[np.random.choice(LENGTH, 20, replace = False)] = np.nan

B = np.random.randint(100, size = LENGTH)
C = A + np.random.normal(0, 0.2, LENGTH)
D = A + np.random.normal(0, 0.1, LENGTH)

E = np.random.rand(LENGTH)
E[np.random.choice(LENGTH, 480, replace = False)] = np.nan

F = B + np.random.normal(0, 10, LENGTH)

target = np.random.randint(2, size = LENGTH)

frame = pd.DataFrame({
    'A': A,
    'B': B,
    'C': C,
    'D': D,
    'E': E,
    'F': F,
})

frame['target'] = target


if __name__ == '__main__':
    frame.to_csv('test_data.csv', index = False)
