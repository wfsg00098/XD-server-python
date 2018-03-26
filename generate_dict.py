import pickle
import numpy as np

test = []

for i in range(20):
    test.append([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

for i in range(20):
    test.append([0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

for i in range(20):
    test.append([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

for i in range(1):
    test.append([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

for i in range(1):
    test.append([0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0])

for i in range(1):
    test.append([0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0])

for i in range(1):
    test.append([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0])

for i in range(1):
    test.append([0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0])

for i in range(1):
    test.append([0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0])

for i in range(1):
    test.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0])

for i in range(1):
    test.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0])

for i in range(1):
    test.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0])

for i in range(1):
    test.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0])

for i in range(1):
    test.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])


test = np.asarray(test, dtype='float64')

print(test.shape)

pickle.dump(test, open("data\\teY.pkl", 'wb'))
