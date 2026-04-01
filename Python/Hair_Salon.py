# List of hairstyles, with accompanying prices, and sales of each from last week
hairstyles = ["bouffant", "pixie", "dreadlocks", "crew", "bowl", "bob", "mohawk", "flattop"]
prices = [30, 25, 40, 20, 20, 35, 50, 35]
last_week = [2, 3, 5, 8, 4, 4, 6, 2]

# Checks for average price of haircuts
total_price = sum(prices)
average_price = total_price/len(prices)
print(f"Average Haircut Price: ${average_price}")

# Decreases cost of each haircut by $5
new_prices = [price - 5 for price in prices]
print(f"New Prices: ${new_prices}")

# Checks for weekly revenue and total daily revenue
total_revenue = 0
for i in range(len(hairstyles)):
  total_revenue += prices[i] * last_week[i]
print(f"Total Revenue: ${total_revenue}")
average_daily_revenue = total_revenue/7
print(f"Average Daily Revenue: ${average_daily_revenue}")

# Shows all haircuts now under $30
cuts_under_30 = [hairstyles[i] for i in range(len(new_prices)) if new_prices[i] < 30]
print(cuts_under_30)
