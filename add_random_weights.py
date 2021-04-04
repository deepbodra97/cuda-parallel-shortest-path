import random

def addWeights(fileName, start=1, stop=100):
	file = open(fileName, "r")
	lines = file.readlines()

	for i, line in enumerate(lines):
		if i:	
			lines[i] = line[:-1] + " " + str(random.randint(start, stop)) + "\n"

	file = open(fileName, "w")
	file.writelines(lines)
	file.close()

fileName = "gnutella04.txt"
start, stop = 1, 100
addWeights(fileName, start, stop)