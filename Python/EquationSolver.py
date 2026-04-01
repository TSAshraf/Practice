import math
print("A quadratic equation can be shown in the format ax^2+bx+c. Please input your equation in this format")
a = float(input("What is the value of a?"))
b = float(input("What is the value of b?"))
c = float(input("What is the value of c?"))

if a == 0:
    x1 = -c / b
    print(f"Equation has one solution: {x1}")
else:
    D = (b ** 2) - (4 * a * c)
    if D < 0:
        print("The equation has no solution")
    elif D == 0:
        x2 = -b / (2 * a)
        print(f"Equation has one real repeated root: {x2}")
    else:
        x3 = (-b - math.sqrt(D)) / (2 * a)
        x4 = (-b + math.sqrt(D)) / (2 * a)
        print(f"Equation has two solutions: {x3} and {x4}")
