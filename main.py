import rust_numpy_ext as RNE
import numpy as np
import matplotlib.pyplot as plt

# Test max/min
random_data = np.random.rand(100,100)
print(random_data.shape)
print("Max:", np.max(random_data))
print("Min:", np.min(random_data))
print("MaxMin:", RNE.max_min(random_data))

assert(np.min(random_data) == RNE.max_min(random_data)[1])
assert(np.max(random_data) == RNE.max_min(random_data)[0])

# Test eye
assert(np.array_equal(np.eye(3), RNE.eye(3)))
print("Eye test completed")

# Test random perturbation
for k in [0,1,10]:
    linear_data = np.linspace(0, 10, 100)
    RNE.double_and_random_perturbation(linear_data, k)
    plt.plot(linear_data)
plt.show()
