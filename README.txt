Instructions to run

1. Project Structure
	1. Unzip the project
	2. You will see 2 folders in the project. "data" contains all the dataset files. sample1.txt represents a graph with 6 nodes and 9 edges. sample2.txt represents a graph with 100 nodes and 99000 edges. "output" stores the output files. Currently it has a blank dummy files. The output of all the algorithms on just one dataset itself is 20GB so they are not included due to size constraints. sample files are to be used to confirm the correctness of the algorithms because CPU could take hours to run for other files. The output files will be overwritten if you rerun the code.
	3. The source code files and CMakeLists.txt will be in the root of the project.

2. Load modules
	2.1 module load ufrc cmake/3.19.1 intel/2018.1.163 cuda/10.0.130
	
3. Compile
	3.1 mkdir Release
	3.2 cd Release
	3.3 cmake -DCMAKE_BUILD_TYPE=Release ..
	3.4 make
	
	3 executables named BellmanFord, Dijkstra and FloydWarshall will be created inside the Release folder

4. Bellman Ford

	Command Format
	srun -p gpu --nodes=1 --gpus=geforce:1 --time=00:05:00 --mem=1500 --pty -u BellmanFord algorithm inputFileName source validateOutput outputFormat -i

	algorithm=any integer in the range [0,3] both inclusive 0=CPU, 1=strided, 2=stride with flag
	inputFileName=name of input file with extension (this file should be in "data" folder) use any of these files sample1.txt, sample2.txt, nyc.txt, bay.txt, col.txt, fla.txt, ne.txt, e.txt

	source = source node in the graph. any number in the range [0, number of nodes-1] both inclusive
	
	validateOutput=true|false true=compares gpu's output against cpu's output

	outputFormat=none|print|write
	print=prints distance and path info on screen
	write=distance and path is written in a file named bf{algorithm}.txt in "output" folder
	none=doesnt print or write distance and path [use this to time the kernels]

	Run this sample command to make sure everything is set up correctly. It will print output on screen for the input file sample1.txt
	srun -p gpu --nodes=1 --gpus=geforce:1 --time=01:00:00 --mem=1600 --pty -u BellmanFord 1 sample1.txt 0 false print -i

5. Dijkstra
	
	Command Format
	srun -p gpu --nodes=1 --gpus=geforce:1 --time=01:00:00 --mem=1600 --pty -u Dijkstra algorithm inputFileName validateOutput outputFormat -i

	algorithm=any integer in the range [0,1] both inclusive. 0=CPU, 1=CPU
	
	inputFileName=name of input file with extension (this file should be in "data" folder) use any of these files sample1.txt, sample2.txt, gnutella04.txt, gnutella25.txt, gnutella30.txt
	
	validateOutput=true|false true=compares gpu's output against cpu's output

	outputFormat=none|print|write
	print=prints distance and path info on screen
	write=distance and path is written in a file named d{algorithm}.txt in "output" folder
	none=doesnt print or write distance and path [use this to time the kernels]


	Run this sample command to make sure everything is set up correctly. It will print output on screen for the input file sample1.txt
	srun -p gpu --nodes=1 --gpus=geforce:1 --time=01:00:00 --mem=1500 --pty -u Dijkstra 0 sample1.txt false print -i

6. Floyd Warshall
	Command Format
	srun -p gpu --nodes=1 --gpus=geforce:1 --time=01:00:00 --mem=1500 --pty -u FloydWarshall algorithm inputFileName validateOutput outputFormat -i
	
	algorithm=any integer in the range [0,5], 0=CPU, 1=Super Naive, 2=Naive, 3=Super Naive Shared, 4=Tiled Global, 5=Tiled Shared Memory
	
	inputFileName=name of input file with extension
	
	validateOutput=true|false true=compares cpu output with gpu output

	outputFormat=none|print|write
	print=prints distance and path info on screen
	write=distance and path is written in a file named fw{algorithm}.txt in "output" folder
	none=doesnt print or write distance and path [use this to time the kernels]

	sample command
	srun -p gpu --nodes=1 --gpus=geforce:1 --time=01:00:00 --mem=25000 --pty -u FloydWarshall 1 sample1.txt false print -i


7. utils.py
	The dataset files have already been parsed. There is no need to run these commands. They are provided just for documentation purpose.

	1. Create a random graph of 100 vertices and store it in a file named sample2.txt
	python utils.py random 100 ./data/sample2.txt

	2. Parse a file from DIMACS dataset
	python utils.py parse ./data/nyc.txt

	3. Add random weights in the range [1, 100] to a file from SNAP dataset
	python utils.py add ./data/gnutella04.txt 1 100

	4. Replace already added weights with new random weights in the range [1, 100] to a file generated from command 3
	python utils.py replace ./data/gnutella04.txt 1 100


