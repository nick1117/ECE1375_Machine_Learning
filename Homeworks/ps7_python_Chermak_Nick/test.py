import numpy as np

test1 = 10
test2 = 20
test3 = 30

tests = []
results = []
lams = [0.1, 1, 2]


for i in range(2):
    costs = []
    for j in range(3):
        tests = np.hstack([tests,10])
        results = np.hstack([results, 20])

    total_costs = np.append(total_costs, costs, axis = 0)

print(tests.shape)
#lams = np.hstack([lams])
data = np.column_stack([lams,tests,results])
print(data)
print(data.shape)