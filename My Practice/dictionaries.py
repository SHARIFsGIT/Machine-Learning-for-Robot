red = [255, 0, 0]
print (red[1])

colors = [[255, 0, 0], [155, 150, 144], [215, 0, 100], [250, 100, 0]]
print (colors)
colors = { "blue": [255, 0, 0], "orange": [155, 150, 144], "black": [215, 0, 100], "red": red}
print (colors)
print (colors["orange"])
colors ["green"] = [0, 255, 0]
print (colors["green"])

for element in colors:
    print (element)

for element in colors:
    print (colors[element])

for element in colors.keys():
    print (element)

for element in colors.values():
    print (element)

for element in colors.items():
    print (element)

for key, value in colors.items():
    print (key, ":" , value)