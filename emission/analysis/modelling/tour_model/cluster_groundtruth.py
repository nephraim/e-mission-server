# standard imports
from sklearn.metrics.cluster import homogeneity_score, completeness_score
import numpy 
import matplotlib.pyplot as plt
import uuid as uu
import sys
import math
import copy

# our imports
import emission.analysis.modelling.tour_model.cluster_pipeline as cp
import emission.analysis.modelling.tour_model.similarity as similarity
import emission.analysis.modelling.tour_model.evaluation as evaluation

"""
Functions to evaluate filtering based on groundtruth. To use these functions, 
save the groundtruth of the trips as a color field in the database, or import 
them manually. If they are saved to the database, run this file with a uuid on the 
command line. 

This file displays the similarity histogram of the data, and a graph that represents 
the evaluation of the histogram with different cutoff points. 
"""

#turns color array into an array of integers
def get_colors(data, colors):
    if len(data) != len(colors):
        raise ValueError('Data and groundtruth must have the same number of elements')
    indices = [] * len(set(colors))
    for n in colors:
        if n not in indices:
            indices.append(n)
    for i in range(len(colors)):
        colors[i] = indices.index(colors[i])
    return colors

#update the ground truth after binning
def update_colors(bins, colors):
    newcolors = []
    for bin in bins:
        for b in bin:
            newcolors.append(colors[b])
    #indices = [] * len(set(newcolors))
    #for n in newcolors:
    #    if n not in indices:
    #        indices.append(n)
    #for i in range(len(newcolors)):
    #    newcolors[i] = indices.index(newcolors[i])
    return newcolors

#evaluates the cluster labels against the groundtruth colors
def evaluate(colors, labels):
    b = homogeneity_score(colors, labels)
    c = completeness_score(colors, labels)
    print 'homogeneity is ' + str(b)
    print 'completeness is ' + str(c)

#maps the clusters, colored by the groundtruth
#creates a map for each groundtruthed cluster and 
#a map showing all the clusters. 
def map_clusters_by_groundtruth(data, labels, colors, map_individuals=False):
    import pygmaps
    from matplotlib import colors as matcol
    colormap = plt.cm.get_cmap()
    import random 
    r = random.sample(range(len(set(labels))), len(set(labels)))
    rand = []
    clusters = len(set(labels))
    for i in range(len(labels)):
        rand.append(r[labels[i]]/float(clusters))
    if map_individuals:
        for color in set(colors):
            first = True
            num_paths = 0
            for i in range(len(colors)):
                if colors[i] == color:
                    num_paths += 1
                    start_lat = data[i].trip_start_location.lat
                    start_lon = data[i].trip_start_location.lon
                    end_lat = data[i].trip_end_location.lat
                    end_lon = data[i].trip_end_location.lon
                    if first:
                        mymap = pygmaps.maps(start_lat, start_lon, 10)
                        first = False
                    path = [(start_lat, start_lon), (end_lat, end_lon)]
                    mymap.addpath(path, matcol.rgb2hex(colormap(rand[i])))
            mymap.draw('./mycluster' + str(color) + '.html')

    mymap = pygmaps.maps(37.5, -122.32, 10)
    for i in range(len(data)):
        start_lat = data[i].trip_start_location.lat
        start_lon = data[i].trip_start_location.lon
        end_lat = data[i].trip_end_location.lat
        end_lon = data[i].trip_end_location.lon
        path = [(start_lat, start_lon), (end_lat, end_lon)]
        mymap.addpath(path, matcol.rgb2hex(colormap(float(colors[i])/len(set(colors)))))
    mymap.draw('./mymap.html')

#calculates the cut-off point of a similarity histogram based on 
#keeping 20% of the trips
def calculate_cutoff(firstbins, colors):
    num = .2 * float(len(colors))
    num = int(math.ceil(num))
    sum = 0
    for i in range(len(firstbins)):
        bin = firstbins[i]
        sum += len(bin)
        if sum > num:
            num = len(bin)
            break
    return num

#evaluates the filtering
def evaluate(firstbins, bins, oldcolors, counts):
    falsePos = 0
    falseNeg = 0
    truePos = 0
    trueNeg = 0
    num = calculate_cutoff(firstbins, oldcolors)
    for i in range(len(oldcolors)):
        color = oldcolors[i]
        if any(i in bin for bin in bins):
            if counts[color] >= num:
                truePos += 1
            else:
                falsePos += 1
        else:
            if counts[color] >= num:
                falseNeg += 1
            else:
                trueNeg += 1
    return [truePos, falsePos, trueNeg, falseNeg]

#creates a graph to show the evaluation on the filtering
def graph(x,y, cutoff, elbow):
    N = len(y)
    width = .16
    index = numpy.arange(N)
    total = sum(y[0])
    a = numpy.array(y)[:,0]/float(total)
    b = numpy.array(y)[:,1]/float(total)
    c = numpy.array(y)[:,2]/float(total)
    d = numpy.array(y)[:,3]/float(total)
    e = numpy.true_divide(a+c,a+b+c+d)
    plt.clf()
    rects1 = plt.bar(index, a, width, color='b', label='TruePos')
    rects2 = plt.bar(index + width, b, width, color='g', label='FalsePos')
    rects3 = plt.bar(index + 2*width, c, width, color='r', label='TrueNeg')
    rects4 = plt.bar(index + 3*width, d, width, color='m', label='FalseNeg')
    rects4 = plt.bar(index + 4*width, e, width, color='c', label='True/Total')
    plt.xlabel('Frequency cutoff bin index')
    plt.ylabel('Totals')
    plt.title('Filtering Metrics for different cutoff points')
    plt.xticks(index + 2*width, tuple(x))
    plt.ylim(0,1)
    plt.legend(loc=2)
    plt.show()
    maxy = list(e).index(max(e))
    print 'The best cut-off point with ' + str(max(e)) + ' true/total is at bin index ' + str(x[maxy]) + ' and the index chosen is ' + str(elbow) + ' with a true/total score of ' + str(e[list(x).index(elbow)])

#gets the heights of the bins
def get_jumps(firstbins):
    jumps = []
    h = -1
    for i in range(len(firstbins)):
        bin = firstbins[i]
        if len(bin) != h:
            h = len(bin)
            jumps.append(i)
    return jumps

def main():
    uuid = None
    if len(sys.argv) == 2:
        uuid = sys.argv[1]
        uuid = uu.UUID(uuid)
    data, colors = cp.read_data(uuid=uuid) #read in data and colors
    colors = get_colors(data, colors) #fix the colors to be integers
    sim = similarity.similarity(data, 300, colors=colors) #create a similarity object
    colors = sim.colors #actual colors
    data = sim.data #actual data
    sim.bin_data() #bin the data
    elbow = sim.elbow_distance() #calculate the cut-off point that the algorithm would calculate
    firstbins = sim.bins #saved bins before removing bins
    colors = get_colors(data, colors) #fixed colors
    counts = [0] * len(set(colors)) #get the frequency of each color
    for c in colors:
        counts[c] += 1
    sim.graph() #graph the similarity histogram
    jumps = get_jumps(firstbins) #get the different heights of the bins
    y = []
    x = []
    for n in jumps:
        d, bins = cp.remove_noise(data, 300, numy=n) #remove noise from data    
        x.append(n)
        y.append(evaluate(firstbins, bins, colors, counts)) #evaluate filtering with n as cut-off point
    graph(x,y,n,elbow) #graph the results

if __name__=='__main__':
    main()
