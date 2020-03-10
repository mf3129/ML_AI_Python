height = float(input("What is your height: "))
weight = float(input("What is your weight: "))

bmi = (weight/(height * height)) * 703

if (bmi <= 18.5):
    print("Less than or equal to 18.5: Underweight")
elif (bmi >= 18.5 or bmi <= 24.9):
    print("Normal Weight")
elif (bmi > 24.9 or bmi == 29.9):
    print("Overweight")
else:
    print("Obesity")
