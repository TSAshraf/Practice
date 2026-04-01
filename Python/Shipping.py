weight = int(input("What is the weight of your package?"))

# Ground Shippping Rates
if weight < 2: # Lower than 2lbs
  GroundShipppingPrice = (1.5 * weight) + 20
elif weight > 2 and weight <= 6: # Between 2 and 6 lbs
  GroundShipppingPrice = (3.0 * weight) + 20 
elif weight > 6 and weight <= 10: # Between 6 and 10 lbs
  GroundShipppingPrice = (4.0 * weight) + 20
elif weight > 10: # Over 10 lbs
  GroundShipppingPrice = (4.75 * weight) + 20
print(f"Ground Shipping: ${GroundShipppingPrice}")

# Premium Shipping Rates - just a flat fee
PremiumGroundShipppingPrice = 125
print(f"Premium Ground Shipping: ${PremiumGroundShipppingPrice}")

# Drone Shipping Rates
if weight < 2: # Lower than 2lbs
  DroneShipppingPrice = (4.5 * weight)
elif weight > 2 and weight <= 6: # Between 2 and 6 lbs
  DroneShipppingPrice = (9.0 * weight)
elif weight > 6 and weight <= 10: # Between 6 and 10 lbs
  DroneShipppingPrice = (12.0 * weight)
elif weight > 10: # Over 10 lbs
  DroneShipppingPrice = (14.25 * weight)
print(f"Drone Shipping: ${DroneShipppingPrice}")
