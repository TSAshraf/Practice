train_mass = 22680
train_acceleration = 10
train_distance = 100
bomb_mass = 1

# Converts Fahrenheit to Celcius
def f_to_c(f_temp):
  c_temp = (f_temp - 32) * 5/9
  return c_temp
print(f"The temperature is {f_to_c(100)}°C")

# Converts Celcius to Fahrenheit
def c_to_f(c_temp):
  f_temp = c_temp * (9/5) + 32
  return f_temp
print(f"The temperature is {c_to_f(0)}°F")


def get_force(mass, acceleration):
  force = mass * acceleration
  return force
print(f"The GE train supplies {get_force(train_mass, train_acceleration)} Newtons of force")

# Calculates mass from energy
def get_energy(mass, c = 3 * 10 ** 8):
  energy = mass * (c ** 2) # E = MC ** 2
  return energy
bomb_energy = get_energy(bomb_mass)
print(f"A 1kg bomb supplies {bomb_energy} Joules")


def get_work(mass, acceleration, distance):
  work = get_force(mass, acceleration) * distance
  return work
train_work = get_work(train_mass, train_acceleration, train_distance)
print(f"The GE train does {train_work} Joules of work over {train_distance} metres")
