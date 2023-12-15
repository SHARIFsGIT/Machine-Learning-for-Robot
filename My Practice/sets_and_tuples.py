# a = (1, 2, 3)
# a[0] = 4 # not posssible
# print(a)

a = {"a", "b", "c", "d", "a"}
print(a)
print("a" in a)
# print(a[0]) # not posssible
for i in a: 
    print(i)

a = ["a", "b", "c", "d", "a"]
print(list(set(a))) # it will give a unique element