fruits = ["apple", "orange", "mango",] #list of strings

for fruit in fruits:
    print(fruit + " juice")

nums = [1, 2, 3]

for num in nums:
    print(num)

for num in range(5):
    if num == 3:
        continue
    print(num)

for num in range(1, 5, 2):
    print(num)

x = 5

while x > 1:
    print(x)
    x -= 1