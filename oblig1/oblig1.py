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
	dist = sum(data[r[i-1]][r[i]] for i in range(len(r)))
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

def hillClimbing(numCities, numSamp):
	minDist = 0
	bestRoute = ()
	solDists = []
	solRoutes = []
	
	numIter = 20
	
	initRoutes = [random.sample(range(0, numCities), numCities) for i in range(numIter)]	
	
	for cityRoute in initRoutes:
		r = climbHill(cityRoute, numSamp)
		solDists.append(r[0])
		solRoutes.append(r[1])
	
	print("[HC]({},{}) Best: {:.2f}km, Worst: {:.2f}km, Avg: {:.2f}km, Stdev: {:.2f}km".format(numCities, numSamp, min(solDists),max(solDists),statistics.mean(solDists), statistics.stdev(solDists)))

#--- Genetic algorithms ---
def getSamplePopulation(numCities, popSize):
	pop = []
	
	#First, calculate initial populate distances.
	for r in [random.sample(range(0, numCities), numCities) for i in range(popSize)]:
		pop.append((getRouteDistance(r),r))
	
	pop.sort()
	
	return(pop)

#Breeders 1 return top x%
#TODO: Doesn't have the s arg used in Rank-based Selection Linear ranking
def getBreeders1(pop):
	pop.sort()
	r = random.uniform(0,1)
	#Round down to nearest even number
	cutoff = int(len(pop)*r/2)*2+1 
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

#Mutation 1, swap two cities in each route
def mutate1(routes):
	newRoutes = []
	for r in routes:
		m = random.sample(range(0, len(r)), 2)
		newRoutes.append(swap(r, m[0], m[1]))
	return newRoutes

def runGA(numCities, popSize):
	sols = []
	for i in range(20):
		sol = geneticAlgorithm(numCities, popSize)
		sols.append(sol)
	dists = [s[0] for s in sols]
	print("[GA]({},{}) Best: {:.2f}km, Worst: {:.2f}km, Avg: {:.2f}km, Stdev: {:.2f}km".format(numCities, popSize, min(dists),max(dists),statistics.mean(dists), statistics.stdev(dists)))

def geneticAlgorithm(numCities, popSize):
	numGens = 0
	avgs = []
	pop = getSamplePopulation(numCities, popSize)
	currAvg = statistics.mean([p[0] for p in pop])
	avgs.insert(0, currAvg) 
	#Generation loop - continue producing generations as long as there is an improvement for the last 10
	while len(avgs) < 10 or (numGens < (5 * popSize) and (len(avgs) == 10 and statistics.stdev(avgs) != 0)):
		breeders = getBreeders1(pop)
		
		#Population breeding loop - pair two and two, on best fitness.
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
			numGens = numGens + 1
		#Only the top popSize survives
		pop.sort()
		pop = pop[0:popSize]
		currAvg = statistics.mean([p[0] for p in pop])
		avgs.insert(0, currAvg)
		avgs = avgs[0:10]
	#return best solution
	return(pop[0])

#This is a stub..
def hybridLamarckian(numCities, popSize):
	pop = getSamplePopulation(numCities, popSize)
	print(pop)

#This is a stub..
def hydridBaldwinian(numCities, popSize):
	pop = getSamplePopulation(numCities, popSize)
	print(pop)

#Timing Exhaustive Search
def timeExhaustiveSearch():
	for n in [10]:
		t = timeit.Timer("exhaustiveSearch(" + str(n) + ")", globals=globals())
		print("[ES]({}) took {:.4f}s".format(n, t.timeit(1)))

#Hill climing
def timeHillClimbing():
	for n in [10, 24]:
		for s in [100, 1000, 10000]:
			t = timeit.Timer("hillClimbing(" + str(n) + "," + str(s) + ")", globals=globals())
			print("[HC]({},{}) took {:.4f}s".format(n,s, t.timeit(1)))

#Genetic algorithm, pop 10
def timeGA():
	for n in [10, 24]:
		for p in [10, 100, 1000]:
			t = timeit.Timer("runGA(" + str(n) + ", " + str(p) + ")", globals=globals())
			print("[GA]({},{}) took {:.4f}s".format(n, p, t.timeit(1)))

timeExhaustiveSearch()
timeHillClimbing()
timeGA()
