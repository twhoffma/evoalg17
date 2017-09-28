import itertools
import random
import math
import statistics
import hillclimb
from datetime import datetime 
import common
import ga
import matplotlib.pyplot as plt

_OUTPUT_LATEX = False

#Load the data
f = open('european_cities.csv','r')
headers = f.readline().rstrip().split(';')

data = []
for line in f:
	row = []
	for p in line.rstrip().split(';'):
		row.append(float(p))
	data.append(row)

def printResultsTable(res):
		d = "\t"
		if _OUTPUT_LATEX:
			d = " & "
		print(d.join(res[0]))
		for r in res[1:]:
			print(d.join(["{}","{:.2f}","{:.2f}","{:.2f}","{:.2f}","{:.2f}", "{:.2f}"]).format(*r))

def exhaustiveSearch(data, numCities):
	cityRoutes = itertools.permutations(list(range(numCities)))
	minDist = -1
	bestRoute = ();	

	for cityRoute in cityRoutes:
		dist = common.getRouteDistance(data, list(cityRoute))
			
		if minDist == -1 or minDist > dist:
			minDist = dist
			bestRoute = cityRoute
	
	return(bestRoute)	
	#namedRoute = []
	#for r in bestRoute:
	#	namedRoute.append(headers[r])
	#
	#print("[ES](" + str(numCities) + ") Best route: " + str(namedRoute) + " in " + str(round(minDist, 2)) + "km")

#def hillClimbing(numCities, numSamp):
#	minDist = 0
#	bestRoute = ()
#	solDists = []
#	solRoutes = []
#	
#	numIter = 20
#	
#	initRoutes = [random.sample(range(0, numCities), numCities) for i in range(numIter)]	
#	
#	for cityRoute in initRoutes:
#		r = climbHill(cityRoute, numSamp)
#		solDists.append(r[0])
#		solRoutes.append(r[1])
#	
#	print("[HC]({},{}) Best: {:.2f}km, Worst: {:.2f}km, Avg: {:.2f}km, Stdev: {:.2f}km".format(numCities, numSamp, min(solDists),max(solDists),statistics.mean(solDists), statistics.stdev(solDists)))

#--- Genetic algorithms ---
#def getSamplePopulation(numCities, popSize):
#	pop = []
#	
#	#First, calculate initial populate distances.
#	for r in [random.sample(range(0, numCities), numCities) for i in range(popSize)]:
#		pop.append((getRouteDistance(r),r))
#	
#	pop.sort()
#	
#	return(pop)

#Breeders 1 return top x%
#TODO: Doesn't have the s arg used in Rank-based Selection Linear ranking
#def getBreeders1(pop):
#	pop.sort()
#	r = random.uniform(0,1)
#	#Round down to nearest even number
#	cutoff = int(len(pop)*r/2)*2+1 
#	breeders = list(pop[0:cutoff])
#	return(breeders)

#CrossOver1 selects a random mirror a position of in both children
#def crossover1(route1, route2):
#	numCities = len(route1)
#	
#	m = random.sample(range(0, numCities), 2)
#	mPos0 = route1.index(m[0])
#	mPos1 = route2.index(m[0])
#	
#	route1 = swap(route1, mPos0, mPos1)
#	
#	mPos0 = route1.index(m[1])
#	mPos1 = route2.index(m[1])
#	
#	route2 = swap(route2, mPos0, mPos1)
#	
#	return((route1,route2))

#Mutation 1, swap two cities in each route
#def mutate1(routes):
#	newRoutes = []
#	for r in routes:
#		m = random.sample(range(0, len(r)), 2)
#		newRoutes.append(swap(r, m[0], m[1]))
#	return newRoutes

#def runGA(numCities, popSize):
#	sols = []
#	fitnessByGen = []
#	for i in range(20):
#		sol = geneticAlgorithm(numCities, popSize)
#		sols.append(sol[0])
#		for j in range(len(sol[1])):
#			if j >= len(fitnessByGen):
#				fitnessByGen.append([])
#			fitnessByGen[j].append(sol[1][j])
#	dists = [s[0] for s in sols]
#	print(fitnessByGen)
#	print("[GA]({},{}) Best: {:.2f}km, Worst: {:.2f}km, Avg: {:.2f}km, Stdev: {:.2f}km".format(numCities, popSize, min(dists),max(dists),statistics.mean(dists), statistics.stdev(dists)))
#	return([statistics.mean(bf) for bf in fitnessByGen])

#def geneticAlgorithm(numCities, popSize):
#	bestOfGen = []
#	numGens = 0
#	avgs = []
#	pop = getSamplePopulation(numCities, popSize)
#	currAvg = statistics.mean([p[0] for p in pop])
#	avgs.insert(0, currAvg) 
#	#Generation loop - continue producing generations as long as there is an improvement for the last 10
#	while len(avgs) < 10 or (numGens < (5 * popSize) and (len(avgs) == 10 and statistics.stdev(avgs) != 0)):
#		breeders = getBreeders1(pop)
#		
#		#Population breeding loop - pair two and two, on best fitness.
#		for i in range(0,int(len(breeders)/2),2):
#			children = list(breeders[i:i+2]) #end on < i+2
#			
#			c0 = list(children[0][1])
#			c1 = list(children[1][1])
#			
#			#For each route, a positioned is mirrored.	
#			c0, c1 = crossover1(c0, c1)
#			
#			#Mutate - switch to random cities
#			c0, c1 = mutate1([c0, c1])
#			
#			for r in [c0, c1]:
#				pop = pop + [(getRouteDistance(r),r)]
#			numGens = numGens + 1
#		#Only the top popSize survives
#		pop.sort()
#		pop = pop[0:popSize]
#		currAvg = statistics.mean([p[0] for p in pop])
#		avgs.insert(0, currAvg)
#		avgs = avgs[0:10]
#		bestFitness = popSize * (1 / pop[0][0]) /  sum([(1 / p[0]) for p in pop])
#		bestOfGen.append(bestFitness)
#	#return best solution
#	return((pop[0], bestOfGen))

#This is a stub..
#def hybridLamarckian(numCities, popSize):
#	pop = getSamplePopulation(numCities, popSize)
#	print(pop)

#This is a stub..
#def hydridBaldwinian(numCities, popSize):
#	pop = getSamplePopulation(numCities, popSize)
#	print(pop)

#Timing Exhaustive Search
def timeExhaustiveSearch():
	for n in [10]:
		starttime = datetime.now()
		soln = exhaustiveSearch(data, n)
		endtime = datetime.now()
		
		print("[ES](" + str(n) + ") Best route: " + common.routeToCityNames(headers, soln) + " in " + str(round(common.getRouteDistance(data, soln), 2)) + "km")
		print("[ES]({}) took {}s".format(n, str(endtime - starttime)))

#Hill climing
def timeHillClimbing():
	print("Hill climbing") 
	for numCities in [10, 24]:
		print("\n" + str(numCities) + " cities:")
		res = []
		res.append(['samples','best(km)','worst(km)','avg(km)','std(km), avg(time, s), std(time, s)'])
		
		for numSamps in [100, 1000, 10000]:
			rt = []
			solns = []
			for t in range(20):	
				starttime = datetime.now()
				solns.append(hillclimb.hillClimbing(data, numCities, numSamps))
				endtime = datetime.now()
				rt.append(endtime - starttime)
			d = [s[0] for s in solns]
			rtts = [ts.total_seconds() for ts in rt]
			res.append([numSamps, min(d),max(d),statistics.mean(d), statistics.stdev(d), statistics.mean(rtts),statistics.stdev(rtts)])
		printResultsTable(res)

#Genetic algorithm, pop 10
def timeGA():
	bogs = []
	print("Genetic Algorithm")
	#for numCities in [10]:
	for numCities in [10, 24]:
		print(str(numCities) + " cities:")
		bog = [] #best of generations
		res = [["pop.size","best(km)","worst(km)","avg(km)","stdev(km)","avg(time, s)","num.gen"]]
		for popSize in [10, 100, 1000]:
			rt = []
			solns = []
			for t in range(20):
				starttime = datetime.now()
				solns.append(ga.geneticAlgorithm(data, numCities, popSize))
				endtime = datetime.now()
				rt.append(endtime - starttime)
			#After t runs, gather resulting data
			d = [s[0] for s in solns]
			g = [len(s[2]) for s in solns]
			sbs = [s[2] for s in solns]
			for sb in sbs:
				for b in range(len(sb)):
					if len(bog) <= b:
						bog.append([])
					bog[b].append(sb[b])
			bogs.append([statistics.mean(sb) for sb in bog])
			rtts = [ts.total_seconds() for ts in rt]
			res.append([popSize, min(d),max(d),statistics.mean(d), statistics.stdev(d), statistics.mean(rtts), statistics.mean(g)])
		printResultsTable(res)
		#pltseries,pltgens = zip(*[(b, list(range(len(b)))) for b in bogs])
		#print(pltseries)
		#print(pltgens)
		#pltdata = [None]*(len(pltseries)*len(pltgens))
		#pltdata[::2] = pltseries
		#pltdata[1::2] = pltgens
		#plt.plot(*pltdata)
		plt.plot(list(range(len(bogs[0]))), bogs[0], list(range(len(bogs[1]))),bogs[1], list(range(len(bogs[2]))), bogs[2])
		plt.savefig('ga-' + str(numCities) + '-cities.eps', format='eps', dpi=1000)
		#plt.show()
#timeExhaustiveSearch()
#timeHillClimbing()
timeGA()
