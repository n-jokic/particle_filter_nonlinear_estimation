# -*- coding: utf-8 -*-
"""
Created on Mon May 29 10:14:51 2023

@author: Milosh Yokich
"""

import numpy as np

from operator import itemgetter


class particle():
    def __init__(self, **kwargs):
        if ('mu' in kwargs):
            self.mu = kwargs['mu']
        else:
            self.mu = np.random.rand()*4-2
        
        if ('x' in kwargs):
            self.x =  kwargs['x']
        else:
            self.x = 0
            
        if ('v' in kwargs):
            self.v =  kwargs['v']
        else:
            self.v = np.random.randn()
        
    def prediction(self):
        x_next = self.x + self.v
        
        v_next = self.v + 0.2*(self.mu - self.v) + 0.32*np.random.randn()
        
        self.x = x_next
        self.v = v_next

    def likelihood(self, e):
        b = np.sqrt(np.abs(self.x))
        
        return (0.05*np.exp(-np.abs(e+self.x)/b) + 0.95*np.exp(-np.abs(e-self.x)/b))/2/b
    def getValues(self):
        return self.x, self.v, self.mu
    
    def getEstimation(self):
        return np.array([self.x, self.v]).reshape((2, 1))
    
class particle_filter():
    
    def __init__(self, **kwargs):
        
        if('sensor_data' in kwargs):
            self.sensor_data = kwargs['sensor_data']
        else:
            self.sensor_data = np.array([])
        
        if('N_particles' in kwargs):
            self.Np = kwargs['N_particles']
        else:
            self.Np = 10
        
        if ('mu' in kwargs):
            self.particles = [particle(mu = kwargs['mu']) for i in range(self.Np)] 
        else:
            self.particles = [particle() for i in range(self.Np)] 
            
        if('algorithm' in kwargs):
            self.algorithm = kwargs['algorithm']
        else:
            self.algorithm = "no_resampling"
            
        self.weights = np.ones((self.Np, 1))/self.Np
        
        self.t = 1
            
        self.old_particles = self.particles
        self.old_weights = self.weights
        
         
    def setSensorData(self, data):
        self.data = data
    
    def setNumberParticles(self, Np):
        self.Np = Np
        self.particles = [particle() for i in range(self.Np)]
        self.weights = np.ones((self.Np, 1))/Np
        
    def setCurrentStep(self, t):
        self.t = t
    def setAlgorithm(self, alg):
        self.algorithm = alg
        
    def resetParticles(self, **kwargs):
        if ('mu' in kwargs):
            self.particles = [particle(mu = kwargs['mu']) for i in range(self.Np)]
        else:
            self.particles = [particle() for i in range(self.Np)]
            
        self.old_particles = self.particles
        
        self.weights = np.ones((self.Np, 1))/self.Np
        self.old_weights = self.weights
        self.t = 1
        
    def particle_filtering(self):
        
        if self.t > len(self.sensor_data) - 1:
            print('No more sensor data')
            return None
        
        e = self.sensor_data[self.t]
        self.t += 1
        
        for i in range(self.Np):
            self.particles[i].prediction()
            
            self.weights[i] *= self.particles[i].likelihood(e)
            
        self.weights = self.weights/np.sum(self.weights)
        
            
        estimation = np.zeros((2, 1))
        
        for idx, particle in enumerate(self.particles):
            estimation += particle.getEstimation()*self.weights[idx]   
        
        self.old_particles = self.particles 
        self.old_weights = self.weights
        
        self.particles, self.weights = self.__update_particles(self.algorithm)
        
        return estimation
        
    def __update_particles(self, algorithm):
        if (algorithm == "no_resampling"):
            return self.__no_resampling()
        if (algorithm == "iterative_resampling"):
            return self.__iterative_resampling()
        if (algorithm == "dynamic_resampling"):
            return self.__dynamic_resampling()
        
        print('Wrong algorithm specified')
        return self.particles, self.weights
    
    def __no_resampling(self):

        
        return self.particles, self.weights
        
    def __iterative_resampling(self):
        new_particles = [0]*self.Np
        new_weights =  np.ones((self.Np, 1))/self.Np
      

        idx = 0
        indices = np.random.choice(self.Np, self.Np, p=self.weights.flatten())
       
        for idx in range(self.Np):
            idy = indices[idx]
            x, v, mu = self.particles[idy].getValues()
            new_particles[idx] = particle(x = x, v = v,  mu = mu)
           
        return new_particles, new_weights
    
    def __dynamic_resampling(self):
        
        effective_size =1/np.sum(self.weights**2)
        if(effective_size < self.Np/2):
            return self.__iterative_resampling()
        else: 
            return self.__no_resampling()