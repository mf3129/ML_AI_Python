first_name = input("What is your first name ");
last_name = input("What is your last name ");

print("Your initials are", first_name[0] + last_name[0])


product_number = input("What is your product number? ")
country_code = product_number[0:3]
product_code = product_number[4:9]
batch_number = product_number[-5:]


print("Country code:", country_code)
print("Product code:", product_code)
print("Batch number:", batch_number)
