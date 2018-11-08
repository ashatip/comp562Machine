# The original file 'shapemap1000000.txt' had a line that had an extra space

import csv

print("in progress...")
targetfile = open("clean.txt", "w")

with open('shapemap1000000.txt') as f:
	count = 1
	for line in f:
		if count % 3 == 0:
			targetfile.write(line[:-2]+'\n') # remove that extra space
		elif count % 3 == 2:
			targetfile.write(line)

		count += 1


print("done")