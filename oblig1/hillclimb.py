import common
import random 
import statistics

def climbHill(data, cityRoute, numSamp):
	minDist = common.getRouteDistance(data, cityRoute)
	bestRoute = cityRoute	
	
	cp = [(random.randint(0, len(cityRoute)-1), random.randint(0, len(cityRoute)-1)) for i in range(numSamp)]
	
	for c in cp:	
		c1 = c[0]
		c2 = c[1]
		if c1 != c2:
			#tmpRoute = list(cityRoute) #copy
			#t = tmpRoute[c1]
			#tmpRoute[c1] = tmpRoute[c2]
			#tmpRoute[c2] = t
			tmpRoute = common.swap(list(cityRoute), c1, c2)
			dist = common.getRouteDistance(data, tmpRoute)
			if dist < minDist:
				minDist = dist
				bestRoute = tmpRoute
	return((minDist, bestRoute)) #return tuple

def hillClimbing(data, numCities, numSamp):
	minDist = 0
	bestRoute = ()
	solDists = []
	solRoutes = []
	
	numIter = 1
	
	initRoutes = [random.sample(range(0, numCities), numCities) for i in range(numIter)]	
	
	for cityRoute in initRoutes:
		r = climbHill(data, cityRoute, numSamp)
		solDists.append(r[0])
		solRoutes.append(r[1])
	
	return(r)
