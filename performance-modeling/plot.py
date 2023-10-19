import matplotlib.pyplot as plt
import numpy as np


n = 1024
m = 768
k_labels = np.array(['3', '5', '7', '9', '11', '13', '15'])
k = np.array([3, 5, 7, 9, 11, 13, 15])
k_time = np.array( (2 * n * m * np.power(k, 2)) / (8.92079 * (np.power(10, 11))) )
k_bw = np.array( 4 * ((2 * n * m) + (np.power(k, 2))) / (34.13 * (np.power(10, 9))) )


print(k)
print(k_time)
print(k_bw)
print(k_labels)


a = plt.figure(1, figsize=(6, 6))
a_scatter = plt.scatter(x=k, y=k_time, s=50.0, marker='x', label='Flops')
plt.xticks(ticks=k, labels=k_labels)
# plt.xlabel("k")
# plt.ylabel("Time for flops (s)")
# plt.title("Expected time to perform flops in convolution (n=1024, m=768)")
plt.legend()
a.tight_layout()
a.savefig('perf-model-plt.png')
# a.show()


plt.xlabel("k")
plt.ylabel("Time (s)")
plt.title("Expected time to calculate convolution (n=1024, m=768)")


b = plt.figure(1, figsize=(6, 6))
b_line = plt.plot(k, k_bw, '-go', label='Bandwidth')
plt.xticks(ticks=k, labels=k_labels)
# plt.xlabel("k")
# plt.ylabel("Time for bandwidth (s)")
# plt.title("Expected time to move memory in convolution (n=1024, m=768)")
plt.legend()
b.tight_layout()
# b.savefig('perf-model-plt2.png')
b.savefig('perf-model-plt.png')
# b.show()