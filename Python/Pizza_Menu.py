# Creating two lists for toppings and prices
toppings = ["pepperoni", "pineapple", "cheese", "sausage", "olives", "anchovies", "mushrooms"]
prices = [2, 6, 1, 3, 2, 7, 2]

num_two_dollar_slices = prices.count(2)
num_pizzas = len(toppings)
print(f"We sell {num_pizzas} different kinds of pizza!")

# Manual method for creating a combined list
pizza_and_prices = [[2, "pepperoni"], [6, "pineapple"], [1, "cheese"], [3, "sausage"], [2, "olives"], [7, "anchovies"], [2, "mushrooms"]]
pizza_and_prices.sort()
print(pizza_and_prices)

# Zip method for creating a combined list
pizza_and_prices_auto = list(zip(prices, toppings))
pizza_and_prices_auto.sort()
print(pizza_and_prices_auto)

# Finding the Cheapest and Most expensive pizza
cheapest_pizza = pizza_and_prices[0]
print(cheapest_pizza)
priciest_pizza = pizza_and_prices[-1]
print(priciest_pizza)

# Finding the Cheapest three pizzas
three_cheapest = pizza_and_prices[0:3]
print(three_cheapest)

# Removing options to the menu
pizza_and_prices.pop()
print(pizza_and_prices)

# Adding options to the menu
pizza_and_prices.insert(-1, [2.5, "peppers"])
pizza_and_prices.sort()
print(pizza_and_prices)

