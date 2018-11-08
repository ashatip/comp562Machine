########################################
############## PYTHON 3 ################
########################################

import csv
import numpy as np

barbaricfile = 'shapemap1000000.txt'
targetfile = open("shapemap5thpoly.txt", "w")

print("counting lines...")
with open(barbaricfile) as f:
	row_count = sum(1 for row in f)
print('total rows: ' + str(row_count))
percent_interval = 1 # percent. CHANGE THIS TO CHANGE PROGRESS METER INTERVALS
progress_interval = int(round(row_count/100*percent_interval))

print("processing...")
with open(barbaricfile) as f:
	reader = csv.reader(f, delimiter=' ')
	for row in reader:
		if reader.line_num%progress_interval == 0:
			print(reader.line_num)

		if reader.line_num%3 == 1:
			targetfile.write(' '.join(row)+'\n')
		elif reader.line_num%3 == 0:
			coords = [float(s) for s in row[:-1]] # skip whitespace at the end
			x = coords[::3]
			y = coords[1::3]
			z = coords[2::3]
			fitx = np.polyfit(range(20), x, 5)
			fity = np.polyfit(range(20), y, 5)
			fitz = np.polyfit(range(20), z, 5)
			targetfile.write(str(fitx)[1:-1]+'\n')
			targetfile.write(str(fity)[1:-1]+'\n')
			targetfile.write(str(fitz)[1:-1]+'\n')


targetfile.close()
print("done")