# def say_hi():
#     print ("HI")

# print(say_hi()) # it prints None. To avoid:

# say_hi()

# def say_hi():
#     return "HI"
# print(say_hi().lower()) # chain function execuit as we have return value.

# def say_hi():
#     for i in range(3):
#         print ("Hi")

# say_hi()
# print("catching breath")
# say_hi()

# def say_hi(person):
#     for i in range(3):
#         print ("Hi " + person)

# say_hi("you")
# print("catching breath")
# say_hi("me")

# def say_hi(person = "X"):
#     for i in range(3):
#         print ("Hi " + person)

# say_hi("you")
# print("catching breath")
# say_hi()

def say_hi(person, times):
    for i in range(times):
        print ("Hi " + person)

# say_hi("you", 2) 
#or 
say_hi(times= 2, person="you")
