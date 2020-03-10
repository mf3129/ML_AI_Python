months = ("January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December")
birthday = input("What is your birthday: (DD-MM-YYYY) ")

index = int(birthday[3:5])
bd_month = months[index-1]





print("You were born in", bd_month)

people = ["Mark", "Jamie", "Sammy", "Martin", "Samon"]
user_name = input("What is your name? ")

people.append(user_name)
print(people)



