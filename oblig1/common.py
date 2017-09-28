def getRouteDistance(data, r):
	dist = sum(data[r[i-1]][r[i]] for i in range(len(r)))
	return(dist)

def swap(cityRoute, city1, city2):
	tmpRoute = list(cityRoute)
	t = tmpRoute[city1]
	tmpRoute[city1] = tmpRoute[city2]
	tmpRoute[city2] = t
	
	return(tmpRoute)

def routeToCityNames(headers, route):
	namedRoute = []
	for r in route:
		namedRoute.append(headers[r])
	return(str(namedRoute))
