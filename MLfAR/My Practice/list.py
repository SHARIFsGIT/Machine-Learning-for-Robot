from pyparsing import nums


fruits = ["a", "b", "c", "d"]
# print (fruits)

# fruits.append("e") # for single element add in list
# print (fruits)

# fruits.extend(["f", "g"]) # for multiple element add in list
# print (fruits)

# fruits.insert(1, "ab") # for insert an element at one position
# print (fruits)

# popped = fruits.pop() # remove the last element
# print (fruits)
# print (popped) # show the popped element

# popped = fruits.pop(0) # remove the first element
# print (fruits)
# print (popped)

# removed = fruits.remove("d") # remove without index number
# print (fruits)
# print (removed) # it shows none as output 

# element = fruits[0]
# print (element)
# element = fruits[-1]
# print (element)
# print (len(fruits))
# element = fruits[len(fruits)-1]
# print (element)

# for fruit in fruits:
#     print (fruit)

# for idx, fruit in enumerate(fruits):
#     print (idx, fruit)

# fruits[0] = "aa"
# print (fruits)

# fruits[0:2] = ["aa", "ab"]
# print (fruits)

nums = list(range(10,20))
print (nums)

# for idx, num in enumerate(nums):
#     nums[idx] = num + 2
# print(nums)

nums_plus_two = [num + 2 for num in nums]
print(nums_plus_two)


# nums = list(range(20, 40))
# print (nums)

# nums_sliced = nums[0:10]
# print (nums_sliced)

# nums_sliced = nums[0:10:2]
# print (nums_sliced)

# nums_sliced = nums[0:10:-2]
# print (nums_sliced)

# nums_sliced = nums[10:0:-2]
# print (nums_sliced)