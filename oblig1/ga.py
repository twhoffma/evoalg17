#Genetic algorithms

import common
import random
import statistics
from hillclimb import climbHill

avgs = []

def getSamplePopulation(data, numCities, popSize):
	pop = []
	
	#First, calculate initial populate distances.
	for r in [random.sample(range(0, numCities), numCities) for i in range(popSize)]:
		pop.append((common.getRouteDistance(data, r),r))
	
	pop.sort()
	
	return(pop)

#--- SELECTION methods ---
#Breeders 1 return top x%
#TODO: Doesn't have the s arg used in Rank-based Selection Linear ranking
def getBreeders1(pop):
	pop.sort()
	r = random.uniform(0,1)
	#Round down to nearest even number
	cutoff = int(len(pop)*r/2)*2+1 
	breeders = list(pop[0:cutoff])
	return(breeders)

#--- FITNESS functions ---
def fitness1(pop, i):
	return(len(pop) * (1 / pop[0][i]) /  sum([(1 / p[0]) for p in pop]))



#--- SAMPLING functions ---
#Selects a random mirror a position of in both children
def crossover1(route1, route2):
	numCities = len(route1)
	
	m = random.sample(range(0, numCities), 2)
	mPos0 = route1.index(m[0])
	mPos1 = route2.index(m[0])
	
	route1 = common.swap(route1, mPos0, mPos1)
	
	mPos0 = route1.index(m[1])
	mPos1 = route2.index(m[1])
	
	route2 = common.swap(route2, mPos0, mPos1)
	
	return((route1,route2))

#--- MUTATION functions ---
#Swap two cities in each route
def mutate1(routes):
	newRoutes = []
	for r in routes:
		m = random.sample(range(0, len(r)), 2)
		newRoutes.append(common.swap(r, m[0], m[1]))
	return newRoutes

#--- TERMINATION functions ---
def terminate1(pop):
	global avgs
	avgs.insert(0, statistics.mean([p[0] for p in pop]))
	avgs = avgs[0:10]
	if len(avgs) < 10:
		return(False)
	else:
		return(statistics.stdev(avgs) == 0)

#--- MAIN algorithms ---
#def runGA(numCities, popSize):
#	sols = []
#	fitnessByGen = []
#	for i in range(20):
#		sol = geneticAlgorithm(data, numCities, popSize)
#		sols.append(sol[0])
#		for j in range(len(sol[1])):
#			if j >= len(fitnessByGen):
#				fitnessByGen.append([])
#			fitnessByGen[j].append(sol[1][j])
#	dists = [s[0] for s in sols]
#	print(fitnessByGen)
#	print("[GA]({},{}) Best: {:.2f}km, Worst: {:.2f}km, Avg: {:.2f}km, Stdev: {:.2f}km".format(numCities, popSize, min(dists),max(dists),statistics.mean(dists), statistics.stdev(dists)))
#	return([statistics.mean(bf) for bf in fitnessByGen])

def geneticAlgorithm(data, numCities, popSize):
	bestOfGen = []
	numGens = 0
	global avgs
	pop = getSamplePopulation(data, numCities, popSize)
	#currAvg = statistics.mean([p[0] for p in pop])
	#avgs.insert(0, currAvg) 
	#Generation loop - continue producing generations as long as there is an improvement for the last 10
	while (numGens < (5 * popSize) and not terminate1(pop)):
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
				pop = pop + [(common.getRouteDistance(data, r),r)]
			numGens = numGens + 1
		#Only the top popSize survives
		pop.sort()
		pop = pop[0:popSize]
		#currAvg = statistics.mean([p[0] for p in pop])
		#avgs.insert(0, currAvg)
		#avgs = avgs[0:10]
		bestFitness = fitness1(pop, 0)
		bestOfGen.append(bestFitness)
	#return best solution - tuple of 3.
	return((*pop[0], bestOfGen))

#This is a stub..
def hybridLamarckian(numCities, popSize):
	pop = getSamplePopulation(numCities, popSize)
	print(pop)

#This is a stub..
def hydridBaldwinian(numCities, popSize):
	pop = getSamplePopulation(numCities, popSize)
	print(pop)
