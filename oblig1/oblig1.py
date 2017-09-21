import itertools
import timeit
import random
import math
import statistics

#Load the data
f = open('european_cities.csv','r')
headers = f.readline().rstrip().split(';')

data = []
for line in f:
	row = []
	for p in line.rstrip().split(';'):
		row.append(float(p))
	data.append(row)

def getRouteDistance(r):
	#Start with return journey distance
	dist = data[r[-1]][r[0]]
	for p in r:
		dist = dist + data[r[0]][p]
		#Could do Djikstra type condition with additional arg to function
		#if dist > minDist:
		#	break
	
	return(dist)

def swapTwo(cityRoute, city1, city2):
	tmpRoute = list(cityRoute)
	t = tmpRoute[city1]
	tmpRoute[city1] = tmpRoute[city2]
	tmpRoute[city2] = t
	
	return(tmpRoute)

def climbHill(cityRoute, numSamp):
	minDist = getRouteDistance(cityRoute)
	bestRoute = cityRoute	
	
	cp = [(random.randint(0, len(cityRoute)-1), random.randint(0, len(cityRoute)-1)) for i in range(numSamp)]
	
	for c in cp:	
		c1 = c[0]
		c2 = c[1]
		if c1 != c2:
			tmpRoute = list(cityRoute) #copy
			t = tmpRoute[c1]
			tmpRoute[c1] = tmpRoute[c2]
			tmpRoute[c2] = t
			dist = getRouteDistance(tmpRoute)
			if dist < minDist:
				minDist = dist
				bestRoute = tmpRoute
	return((minDist, bestRoute)) #return tuple

def exhaustiveSearch(numCities):
	cityRoutes = itertools.permutations(list(range(numCities)))
	minDist = -1
	bestRoute = ();	

	for cityRoute in cityRoutes:
		dist = getRouteDistance(list(cityRoute))
			
		if minDist == -1 or minDist > dist:
			minDist = dist
			bestRoute = cityRoute
	
	namedRoute = []
	for r in bestRoute:
		namedRoute.append(headers[r])
	
	print("[ES](" + str(numCities) + ") Best route: " + str(namedRoute) + " in " + str(round(minDist, 2)) + "km")

def hillClimbing(numCities):
	minDist = 0
	bestRoute = ()
	solDists = []
	solRoutes = []
	
	numIter = 20
	numSamp = 1000
	
	initRoutes = [random.sample(range(0, numCities), numCities) for i in range(numIter)]	
	
	for cityRoute in initRoutes:
		#Pairs of random changes
		#cp = [(random.randint(0, len(initRoutes[0])-1), random.randint(0, len(initRoutes[0])-1)) for i in range(numSamp)]
		#minDist = getRouteDistance(cityRoute)
		#bestRoute = cityRoute	
		#
		#for c in cp:	
		#	c1 = c[0]
		#	c2 = c[1]
		#	if c1 != c2:
		#		tmpRoute = list(cityRoute) #copy
		#		t = tmpRoute[c1]
		#		tmpRoute[c1] = tmpRoute[c2]
		#		tmpRoute[c2] = t
		#		dist = getRouteDistance(tmpRoute)
		#		if dist < minDist:
		#			minDist = dist
		#			bestRoute = tmpRoute
		#solDists.append(dist)
		#solRoutes.append(bestRoute)
		r = climbHill(cityRoute, numSamp)
		solDists.append(r[0])
		solRoutes.append(r[1])
	
	print("Best: {:.2f}km, Worst: {:.2f}km, Avg: {:.2f}km, Stdev: {:.2f}km".format(min(solDists),max(solDists),statistics.mean(solDists), statistics.stdev(solDists)))

def getSamplePopulation(numCities, popSize):
	pop = []
	
	#First, calculate initial populate distances.
	for r in [random.sample(range(0, numCities), numCities) for i in range(popSize)]:
		pop.append((getRouteDistance(r),r))
	
	pop.sort()
	
	return(pop)
	
def geneticAlgorithm(numCities, popSize):
	numGens = 10
	pop = getSamplePopulation(numCities, popSize)
	#print(pop)
	
	#Generation loop
	for j in range(numGens):
		
		#Population breeding loop
		for i in range(0,int(popSize/2),2):
			#The children
			#c1 = list(pop[i]);
			#c2 = list(pop[i+1]);
			
			children = list(pop[i:i+2]) #end on i+2, but do not include it
			
			for c in children:
				#Crossover - for a random integer, move that city to that spot.
				#k1,k2 = random.sample(range(0, numCitites), 2)
				#Find k1 in both children. For c2 make k1 at the same postition as in c1
				#Find k2 in both children. For c1 make k2 at the same position as in c2. 
					
				#Mutation - swap two cities on each child, like a single step hillclimb.
				m = random.sample(range(0, numCities), 2)
				#route = c[1]
				#tmpCity = route[m[0]]
				#route[m[0]] = m[1]
				#route[m[1]] = tmpCity
				newRoute = swapTwo(c[1], m[0], m[1])	
				pop = pop + [(getRouteDistance(newRoute),newRoute)]	
		#Only the top popSize survives
		pop.sort()
		pop = pop[0:popSize+1]
	#After numGens generations, print stats
	dists = [p[0] for p in pop]
	print("Best: {:.2f}km, Worst: {:.2f}km, Avg: {:.2f}km, Stdev: {:.2f}km".format(min(dists),max(dists),statistics.mean(dists), statistics.stdev(dists)))
	print(pop[0])
	
		
			
	#steps.
	#1. Select which parents will "breed": Ordered by distance asc.


	#2. Make two children
	#3. Make one crossover 
	#3a. Select one city from parent a. Switch that city in parent b to the same location. Vise versa for parent a.
	#4. Mutation
	#5. Survivor
	#5a. all parents are killed, only children survive to next iteration
	#5b. children are added to population, the "fastest" popSize population survives.

def hybridLamarckian(numCities, popSize):
	pop = getSamplePopulation(numCities, popSize)
	print(pop)

def hydridBaldwinian(numCities, popSize):
	pop = getSamplePopulation(numCities, popSize)
	print(pop)

#Timing Exhaustive Search
#for n in [10]:
#	t = timeit.Timer("exhaustiveSearch(" + str(n) + ")", globals=globals())
#	print("[ES]({}) took {:.4f}s".format(n, t.timeit(1)))

#Hill climing
for n in [24]:
	t = timeit.Timer("hillClimbing(" + str(n) + ")", globals=globals())
	print("[HC]({}) took {:.4f}s".format(n, t.timeit(1)))

#Genetic algorithm, pop 10
for n in [24]:
	for p in [100]:
		t = timeit.Timer("geneticAlgorithm(" + str(n) + ", 10)", globals=globals())
		print("[GA]({},{}) took {:.4f}s".format(n, p, t.timeit(1)))


