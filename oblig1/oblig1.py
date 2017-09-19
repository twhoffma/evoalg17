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
	
	#Select initial solutions
	#cityRoutes = itertools.permutations(list(range(numCities)))
	#x = sorted([random.randint(0, math.factorial(numCities)) for i in range(numIter)])
	
	##Sorting through these takes too much time
	#i = 0
	#j = 0
	#initRoutes = []
	#for r in cityRoutes:
	#	if i == x[j]:
	#		initRoutes.append(list(r))
	#		if len(initRoutes) == len(x):
	#			break
	#		j = j +1
	#	i = i +1
	
	initRoutes = [random.sample(range(0, numCities), numCities) for i in range(numIter)]	
	
	print("init route length:" + str(len(initRoutes[0])))
	
	#Pairs of random changes
	cp = [(random.randint(0, len(initRoutes[0])-1), random.randint(0, len(initRoutes[0])-1)) for i in range(numSamp)]
	
	for cityRoute in initRoutes:
		minDist = getRouteDistance(cityRoute)
		bestRoute = cityRoute	
		
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
		solDists.append(dist)
		solRoutes.append(bestRoute)
	
	print("Best: {:.2f}km, Worst: {:.2f}km, Avg: {:.2f}km, Stdev: {:.2f}km".format(min(solDists),max(solDists),statistics.mean(solDists), statistics.stdev(solDists)))
		

#Timing Exhaustive Search
for n in [6,7,8,9,10]:
	t = timeit.Timer("exhaustiveSearch(" + str(n) + ")", globals=globals())
	print("[ES]({}) took {:.4f}s".format(n, t.timeit(1)))

for n in [10,24]:
	t = timeit.Timer("hillClimbing(" + str(n) + ")", globals=globals())
	print("[HC]({}) took {:.4f}s".format(n, t.timeit(1)))
	
