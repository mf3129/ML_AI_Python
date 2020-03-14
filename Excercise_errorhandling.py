import random

number = input("Type a number")



try:
    number = float(number)
    print("The number is:", number)
except:
    print("Invalid number")





list_of_countries = ("Mali", "France", "England", "Germany", "Spain", "Ethiopia", "Egypt", "Switzerland")

selected_grade = int(input("Choose a number between 0 and 8: "))


rand_num = random.randint(0,selected_grade)


if selected_grade >= 0 and selected_grade < 8:
    print("The random country is", list_of_countries[0:rand_num])
else:
    print("Type a valid entry", rand_num)
