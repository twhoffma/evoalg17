import itertools
import timeit

#Load the data
f = open('european_cities.csv','r')
headers = f.readline().rstrip().split(';')

data = []
for line in f:
	row = []
	for p in line.rstrip().split(';'):
		row.append(float(p))
	data.append(row)


def exhaustiveSearch(numCities):
	cityRoutes = list(itertools.permutations(list(range(numCities))))
	minDist = -1
	bestRoute = ();	

	for r in cityRoutes:
		#start with the distance needed to travel back.
		dist = data[r[-1]][r[0]]
		for p in r:
			dist = dist + data[r[0]][p]
		
		if minDist == -1 or minDist > dist:
			minDist = dist
			bestRoute = r
	
	cityRoute = []
	for r in bestRoute:
		cityRoute.append(headers[r])
	
	print("[ES](" + str(numCities) + ") Best route: " + str(cityRoute) + " in " + str(round(minDist, 2)) + "km")

#Timing Exhaustive Search
for n in [1,2,3,4,5,6,7,8,9,10,11]:
	t = timeit.Timer("exhaustiveSearch(" + str(n) + ")", globals=globals())
	print("[ES](" + str(n) + ") took " + str(round(t.timeit(1), 4)) + "s")
