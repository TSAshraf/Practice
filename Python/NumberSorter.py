n = input("Input a list of numbers separated by spaces: ").split()
n = [float(x) for x in n] # Converts input strings into floats
x = len(n)

# Bubble sort algorithm
for i in range(x):
    for j in range(0, x - i - 1):
        if n[j] > n[j + 1]:
            n[j], n[j + 1] = n[j + 1], n[j]
print(f"Sorted List: {n}")
