import sys
import random

def parseDIMACS(fileName):
	file = open(fileName, "r")
	lines = file.readlines()

	for i, line in enumerate(lines):
		if i==4:
			tokens = line.split()
			numVertex, numEdges = tokens[2], tokens[3]
			lines[i] = numVertex + " " + numEdges + "\n"
		elif i>=7:	
			_, src, dest, cost = line.split()
			lines[i] = str(int(src)-1) + " " + str(int(dest)-1) + " " + cost + "\n"
		else:
			lines[i] = ''

	file = open(fileName, "w")
	file.writelines(lines)
	file.close()

def addWeights(fileName, weightMin=1, weightMax=100):
	file = open(fileName, "r")
	lines = file.readlines()

	for i, line in enumerate(lines):
		if i==2:
			tokens = line.split()
			numVertex, numEdges = tokens[2], tokens[4]
			lines[i] = numVertex + " " + numEdges + "\n"
		elif i>=4:	
			src, dest = line.split()
			lines[i] = src + " " + dest + " " + str(random.randint(weightMin, weightMax)) + "\n"
		else:
			lines[i] = ''

	file = open(fileName, "w")
	file.writelines(lines)
	file.close()

def replaceWeights(fileName, weightMin=1, weightMax=100):
	file = open(fileName, "r")
	lines = file.readlines()

	for i, line in enumerate(lines):
		if i:
			src, dest, cost = line.split()
			lines[i] = src + " " + dest + " " + str(random.randint(weightMin, weightMax)) + "\n"

	file = open(fileName, "w")
	file.writelines(lines)
	file.close()

def createRandomGraph(numVertex, fileName):
	lines = [str(numVertex)+" "+str(numVertex*numVertex-numVertex)]
	for i in range(numVertex):
		for j in range(numVertex):
			if i != j:
				line = str(i) + " " + str(j) + " " + str(random.randint(1, 100)) + "\n"
				lines.append(line)
	file = open(fileName, "w")
	file.writelines(lines)
	file.close()


if __name__ == "__main__":
	_, action, *rest = sys.argv
	if action == 'parse':
		parseDIMACS(rest[0])
	elif action == 'random':
		createRandomGraph(int(rest[0]), rest[1])
	else:
		fileName, weightMin, weightMax = rest
		if action == 'add':
			addWeights(fileName, int(weightMin), int(weightMax))
		elif action == 'replace':
			replaceWeights(fileName, int(weightMin), int(weightMax))