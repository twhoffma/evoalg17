import timeit
import itertools

f = open('european_cities.csv', 'r')

#header = f[0]
#print header

#Load the data
data = []
headers  = f.readline().split(";")
for l in f:
	r = []
	for i in l.split(";"):
		r.append(float(i))
	
	data.append(r)

print(headers)
print(data)

def exhaustiveSearch(numCities):
	cityOrder = list(itertools.permutations(list(range(numCities))))
	minDistance = -1
	shortestRoute = ()
	
	for route in cityOrder:	
		distance = data[route[-1]][route[0]]
		for c in route:
			distance = distance + data[route[0]][c]
		
		if minDistance == -1 or distance < minDistance:
			minDistance = distance
			shortestRoute = route
	
	routeCityNames = []
	for r in shortestRoute:
		routeCityNames.append(headers[r])
	print("[ES](" + str(numCities) + ")" + str(routeCityNames) + " in " + str(round(minDistance,2)) + "km")



#Run and measure Exhaustive Search
for n in [1,2,3,4,5,6,7,8,9,10]:
	t = timeit.Timer("exhaustiveSearch(" + str(n) +")", globals=globals())
	print("[ES](" + str(n) + ") took " + str(round(t.timeit(1),5)) + "s")
