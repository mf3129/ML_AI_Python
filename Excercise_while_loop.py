import random
random_number = random. randint(0,9)

names = []


while len(names) < 9: 
    name = input("Enter 8 names: ")
    names.append(name)



print("Here is a random name", names[random_number-1])
