import sys
import math
import random
import csv
from collections import defaultdict
import itertools


class KMeans:
    
    def __init__(self,clusters,iterations, k):    
        self.clusters=clusters
        self.iterations=iterations
        self.k=k

    def computeDistances(self,clusters,centroids,sets):
    #Computing eucledian distances from each point to its centroid
        sets = [ [] for c in centroids]
        for p in self.clusters:
            min_dist=30000
            dist=0.0
            for i in xrange(0,len(centroids)):
                temp=0.0
                for j in xrange(0,len(centroids[0])):
                    
                    temp+=(pow(float(p[j])-float(centroids[i][j]),2))

                dist=math.sqrt(temp)    
                if dist<min_dist:
                    min_dist=dist
                    centroid_pos=i
    #assigning the point to the cluster of its nearest centroid        
            sets[centroid_pos].append(p)
        return sets


    def kmeans(self,centroids):
    #initially assuming k points as random centroids

        centroids=random.sample(self.clusters,self.k)        
        count=0
        old_centroid=[]
    
    #providing a stopping condition in case it exceeds more than a certain no of iterations                
        while self.iterations>0:
        
            if self.iterations==200:
                #first time around our dataset hasn't been split
                sets=self.computeDistances(self.clusters,centroids,sets=None)                      
            else:
            
                sets=self.computeDistances(self.clusters,recomputed_centroids,sets)
                old_centroid.append(recomputed_centroids)

            
            recomputed_centroids=[[] for c in centroids]
            iterate=0
            #computation of centroids
            for s in sets:
                x_temp=[]
            
    		
                for j in xrange(0,len(s[0])):
                    temp=0.0
                    for i in xrange(0,len(s)):
                        temp+=float(s[i][j])
                    x_temp.append(temp)
                
        
                for x in x_temp:
                    recomputed_centroids[iterate].append(x/len(s))
        
                iterate+=1                
            
            #checking for stopping criteria if centroid values do not change
            if(count>0):
        	   if(old_centroid[count-1] == recomputed_centroids):
                    print "Centroids:"
                    print recomputed_centroids
                    exit()		            
            count+=1
               
            self.iterations-=1
    

def main():
    clusters=[]
    centroids=[]
    iterations=200
    k=3 #no of clusters
    try:
        f=open(sys.argv[1])
        cluster=csv.reader(f)
        for c in cluster:
            clusters.append(c)
        clus=KMeans(clusters,200,3)
        clus.kmeans(centroids)
    
    except Exception,e:
        print e
        print("Syntax: python <program_name.py> <data_file.txt>")
        exit()
    

if __name__ == "__main__":
    main()