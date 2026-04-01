import random 

# Asks for name, if there is no name a default name is given
name = input("What is your name?")
if name == "":
  name = "The Nameless One"

# Asks for a question, if there is no name a default name is given
question = input("What is your question?")
if question == "":
  question = "Is there an afterlife?"

# Randomly picks an answer from a preselected range of answers
answer = ""
random_number = random.randint(1, 10)
if random_number == 1:
  answer = "Yes - definitely"
elif random_number == 2:
  answer = "It is decidedly so"
elif random_number == 3:
  answer = "Without a doubt"
elif random_number == 4:
  answer = "It is decidedly so"
elif random_number == 5:
  answer = "Reply hazy, try again"
elif random_number == 6:
  answer = "Better not tell you now"
elif random_number == 7:
  answer = "My sources say no"
elif random_number == 8:
  answer = "Outlook not so good"
elif random_number == 9:
  answer = "Very doubtful"
elif random_number == 10:
  answer = "Hopefully so"
else:
  answer = "Error"

# Prints response
print(f"{name} asks: {question}")
print(f"Magic 8-Ball's answer: {answer}")

