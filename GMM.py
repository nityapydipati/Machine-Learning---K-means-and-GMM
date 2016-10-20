import sys
import math
import numpy as np
import csv
import random
import itertools


class GMM:

	def __init__(self,data,iterations,k,threshold):
		self.data=data
		self.iterations=iterations
		self.k=k
		self.threshold=threshold


	def E_step(self,Pdfs,prior_prob):
	
		post_prob = np.ones((len(self.data),3))
		new_cluster=[[] for x in xrange(0,self.k)]
		for i in xrange(0,len(self.data)):
			post_prob_max=0.001
			for j in xrange(0,self.k):
				post_prob[i][j]=prior_prob[i][j]/sum(prior_prob[i])
			
				if(post_prob_max<post_prob[i][j]):
					post_prob_max=post_prob[i][j]
					pos=j
		
		
		
			new_cluster[pos].append(self.data[i])
	
	
		return np.array(new_cluster), np.array(post_prob)	

	def M_step(self, new_cluster,post_prob):
		data=self.data
		recalculated_mean = [ [] for i in xrange(0,self.k)]
		recalculated_cov = [ [] for i in xrange(0,self.k)]
		recalculated_amplitudes = [ [] for i in xrange(0,self.k)]
	
		denominator=np.ones((len(data),3))	
		
		for i in xrange(0,self.k):
			denominator=np.sum(post_prob, axis=0)
			post_proba=(post_prob.T)[i]
			recalculated_mean[i]=(np.sum(np.multiply(post_proba.reshape(len(data),1),data),axis=0))/denominator[i]
			difference = data - np.tile(recalculated_mean[i], (len(data), 1))
			recalculated_cov[i]=(np.dot(np.multiply(post_proba.reshape(len(data),1), difference).T,difference)/denominator[i])
			recalculated_amplitudes[i]=denominator[i]/len(data)
		
		#print recalculated_mixing_coefficiants
		#print recalculated_mean
		#print recalculated_cov
		#raw_input()	
		return recalculated_mean,recalculated_cov,recalculated_amplitudes
			
	def check(self,oldmean,new_mean,new_covariance,new_amplitudes):
		oldmean=np.array(oldmean) 
		new_mean=np.array(new_mean)
		
		if (oldmean-new_mean<self.threshold).all():
			print "Mean:"
			print new_mean
			print "Covariance:"
			print new_covariance
			print "Amplitudes:"
			print new_amplitudes
			exit()
			
	
	def estimatepdf(self,cluster,cluster_mean,cluster_covs,amplitudes):
		
		data=self.data
		Pdfs = np.ones((len(data),1))
		prior_prob = np.ones((len(data),3))
	
		for i in range(0,self.k):
			mean_data=cluster_mean[i]
		
			cov_det=np.linalg.inv(cluster_covs[i])
			denominator=1/np.sqrt((2*np.pi)**2 * np.linalg.det(cluster_covs[i]))
			for j in xrange(0,len(data)):
				temp=data[j,:]-mean_data
				temp = np.dot(-0.5*temp, cov_det)
				temp = np.dot(temp, np.transpose(data[j,:] - mean_data))
				Pdfs[j]=denominator*np.exp(temp)

				prior_prob[j][i]=Pdfs[j]*amplitudes[i]						
			
		oldmean=cluster_mean
		oldcov=cluster_covs
		oldmixingcoeff=amplitudes
	
		new_cluster,post_prob=self.E_step(Pdfs,prior_prob)
		new_mean, new_covariance, new_amplitudes=self.M_step(new_cluster,post_prob)
		#print self.iterations
		if(0<self.iterations<200):
			
			self.check(oldmean,new_mean,new_covariance,new_amplitudes)
		self.iterations=self.iterations-1
		self.estimatepdf(new_cluster,new_mean,new_covariance,new_amplitudes)


def kmeans(data,k):
	centroids=random.sample(data,k)
	sets = [ [] for c in centroids]
        for p in data:
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
        return np.array(sets)
    



def main():
	data=np.loadtxt(sys.argv[1],dtype='float', delimiter=",")
	k=3
	iterations=200
	threshold=0.001
	cluster_mean = [ [] for i in xrange(0,k)]
	cluster_cov = [ [] for i in xrange(0,k)]
	
	
	#initial=np.array()
	intial_amplitudes=np.random.dirichlet(np.ones(3)*150,size=1)
	intial_amplitudes=list(itertools.chain.from_iterable(intial_amplitudes))
	#intial_mixing_coefficiants=[0.33,0.33,0.33]
	

	cluster=kmeans(data,k)
	#cluster=np.array(cluster[0])
	cluster[0]= np.array(cluster[0])
	cluster[1]= np.array(cluster[1])
	cluster[2]=np.array(cluster[2])

	
	
	
	for i in xrange(0,k):
		cluster_mean[i]=np.mean((cluster[i]),axis=0)
		
		
		cluster_cov[i]= np.cov(cluster[i].T)
	g=GMM(data,iterations,k,threshold)
			
			
	g.estimatepdf(cluster,cluster_mean,cluster_cov,intial_amplitudes)
	


if __name__ == "__main__":
    main()