import numpy as np

output_1 = (64.00 - 5.00 - (-2.00)) / 2.00 + 1.00
print(np.ceil(output_1))
afterPool_1 = np.ceil(output_1) / 2.00
print(afterPool_1)
output_2 = (output_1 - 5.00 - (-2.00)) / 2.00 + 1.00
print(np.ceil(output_2))

afterPool_2 = np.ceil(output_2) / 2.00
print(afterPool_2)
