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

def swap(cityRoute, city1, city2):
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

#Breeders 1 return top 50%
def getBreeders1(pop):
	pop.sort()
	cutoff = int(len(pop)/2)+1
	breeders = list(pop[0:cutoff])
	return(breeders)

#CrossOver1 selects a random mirror a position of in both children
def crossover1(route1, route2):
	numCities = len(route1)
	
	m = random.sample(range(0, numCities), 2)
	mPos0 = route1.index(m[0])
	mPos1 = route2.index(m[0])
	
	route1 = swap(route1, mPos0, mPos1)
	
	mPos0 = route1.index(m[1])
	mPos1 = route2.index(m[1])
	
	route2 = swap(route2, mPos0, mPos1)
	
	return((route1,route2))

def mutate1(routes):
	newRoutes = []
	for r in routes:
		m = random.sample(range(0, len(r)), 2)
		newRoutes.append(swap(r, m[0], m[1]))
	return newRoutes

def geneticAlgorithm(numCities, popSize):
	numGens = 10
	pop = getSamplePopulation(numCities, popSize)
	
	#Generation loop
	for j in range(numGens):
		breeders = getBreeders1(pop)
		
		#Population breeding loop
		for i in range(0,int(len(breeders)/2),2):
			children = list(breeders[i:i+2]) #end on < i+2
			
			c0 = list(children[0][1])
			c1 = list(children[1][1])
			
			#For each route, a positioned is mirrored.	
			c0, c1 = crossover1(c0, c1)
			
			#Mutate - switch to random cities
			c0, c1 = mutate1([c0, c1])
			
			for r in [c0, c1]:
				pop = pop + [(getRouteDistance(r),r)]	
		#Only the top popSize survives
		pop.sort()
		#Kill the worst two.
		pop = pop[0:popSize]
		print(len(pop))
	#After numGens generations, print stats
	dists = [p[0] for p in pop]
	print("Best: {:.2f}km, Worst: {:.2f}km, Avg: {:.2f}km, Stdev: {:.2f}km".format(min(dists),max(dists),statistics.mean(dists), statistics.stdev(dists)))

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
for n in [10]:
	t = timeit.Timer("hillClimbing(" + str(n) + ")", globals=globals())
	print("[HC]({}) took {:.4f}s".format(n, t.timeit(1)))

#Genetic algorithm, pop 10
for n in [10]:
	for p in [10]:
		t = timeit.Timer("geneticAlgorithm(" + str(n) + ", " + str(p) + ")", globals=globals())
		print("[GA]({},{}) took {:.4f}s".format(n, p, t.timeit(1)))


