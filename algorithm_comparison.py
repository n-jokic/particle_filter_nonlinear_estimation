# -*- coding: utf-8 -*-
"""
Created on Mon May 29 17:15:01 2023

@author: Milosh Yokich
"""
import numpy as np    
import particle_filter
import matplotlib.pyplot as plt
        
plt.close('all')
def model(N):
    
    mu = np.random.rand()*4-2
    x = np.zeros((N + 1, ))
    
    v = np.zeros((N+1, ))
    v[0] = np.random.randn()
    e = np.random.randn(N)
    
    for t in range(N):
        x[t+1] = x[t] + v[t]
        v[t+1] = v[t] + 0.2*(mu - v[t]) + 0.32*e[t]
        
    return x, v, mu

def measurment(x):
    N = len(x)
    teta = np.random.binomial(size=N, n=1, p= 0.05)
    e = np.zeros((N, ))
    
    for t in range(N):
        b = np.sqrt(np.abs(x[t]))/5
        e[t] = ((-1)**teta[t])*x[t] + np.random.laplace(loc = 0, scale = b)
    return e    
    

N = 50
x, v, mu = model(N)
e = measurment(x)
plt.rcParams['text.usetex'] = True

t = np.linspace(0, N, N+1)

# Plotting the graph
#%%
fig, ax1 = plt.subplots()

# Plotting the position-time graph
ax1.plot(t, x, 'b.', label=r'$x[t]$')
ax1.plot(t, x, 'b--', alpha = 0.5)
ax1.set_xlabel(r'$t$')
ax1.set_ylabel(r'$x[t]$')
ax1.tick_params('y')
ax1.legend(loc='upper left')
ax1.set_xlim([0, N])
ax1.set_ylim([np.min(x) - np.abs(np.min(x)*0.1), np.max(x) + np.abs(np.min(x)*0.1)])

# Creating a twin y-axis for velocity
ax2 = ax1.twinx()

# Plotting the velocity-time graph as a dashed line
ax2.plot(t, v, 'r.', label=r'$v[t]$')
ax2.plot(t, v, 'r--', alpha = 0.5)
ax2.set_ylabel(r'$v[t]$')
ax2.tick_params('y')
ax2.legend(loc='upper right')


# Adding a horizontal line for the parameter μ
ax2.axhline(mu, color='g', linestyle=':', label=r'$\mu$')
ax2.legend(loc='upper right')
ax2.set_xlim([0, N])
ax2.set_ylim([np.min(v) - np.abs(np.min(v)*0.1), np.max(v) + np.abs(np.min(v)*0.1)])

# Display the graph

ax1.set_rasterized(True)
ax2.set_rasterized(True)
FULL_PATH = 'C:/Users/milos/OneDrive/VIII semestar/VI/domaci2/izvestaj/slike/zad2/pos_vel.png'
plt.savefig(FULL_PATH, format='png', dpi = 600)
plt.show()



# Plotting the graph
fig, ax1 = plt.subplots()

# Plotting the position-time graph
ax1.plot(t, x, 'b.', label=r'$x[t]$')
ax1.plot(t, x, 'b--', alpha = 0.5)
ax1.plot(t, e, 'r.', label = r'$e[t]$', markersize = 2)
ax1.plot(t, e, 'r--', alpha = 0.5)
ax1.set_xlabel(r'$t$')
ax1.set_ylabel(r'$x[t]$')
ax1.tick_params('y')
ax1.legend(loc='upper left')
ax1.set_xlim([0, N])
ax1.set_rasterized(True)
ax1.set_ylim([np.min(e) - np.abs(np.min(e)*0.1), np.max(e) + np.abs(np.min(e)*0.1)])
FULL_PATH = 'C:/Users/milos/OneDrive/VIII semestar/VI/domaci2/izvestaj/slike/zad2/measurment.png'
plt.savefig(FULL_PATH, format='png', dpi = 600)


plt.show()



# %%
N = len(e)
N_particles = 30
T = np.round(np.linspace(0, N - 1, N)).astype(int)
particle_filt = particle_filter.particle_filter(sensor_data = e, N_particles = N_particles)

particle_filt.setAlgorithm("no_resampling")
part = None
weights = [0]*(N)
weights[0] = np.array(particle_filt.old_weights)
part = particle_filt.old_particles
t = 0
positions = np.zeros((N_particles, len(T)))
for num, particle in enumerate(part):
    x_val, _, _ = particle.getValues()
    positions[num, t] = x_val


x_est = np.zeros((N, ))
v_est = np.zeros((N, ))

for t in T:
    if t == 50:
        break
    estimation = particle_filt.particle_filtering()
    
    weights[t+1] =  np.array(particle_filt.old_weights)
    part = particle_filt.old_particles
    
    for num, particle in enumerate(part):
        x_val, _, _ = particle.getValues()
        positions[num, t+1] = x_val
    
    x_est[t + 1] = estimation[0]
    v_est[t + 1] = estimation[1]
    
# Plotting the graph
fig, ax1 = plt.subplots()

# Plotting the position-time graph
ax1.plot(T, x, 'b.', label=r'$x[t]$')
ax1.plot(T, x, 'b--', alpha = 0.5)
ax1.plot(T, e, 'r.', label = r'$e[t]$', markersize = 2)
ax1.plot(T, e, 'r--', alpha = 0.5)
ax1.plot(T, x_est, 'k*', label=r'$\hat{x}[t]$')
ax1.plot(T, x_est, 'k', alpha = 0.5)
ax1.set_xlabel(r'$t$')
ax1.set_ylabel(r'$x[t]$')
ax1.tick_params('y')
ax1.set_ylim([np.min(e) - np.abs(np.min(e)*0.1), np.max(e) + np.abs(np.min(e)*0.1)])
ax1.legend(loc='upper left')
ax1.set_xlim([0, N])
ax1.set_rasterized(True)
ax1.set_xlim([0, N-1])
FULL_PATH = 'C:/Users/milos/OneDrive/VIII semestar/VI/domaci2/izvestaj/slike/zad2/x_no_res.png'
plt.savefig(FULL_PATH, format='png', dpi = 600)

plt.show()




fig, ax1 = plt.subplots()

# Plotting the velocity-time graph as a dashed line
ax1.plot(T, v, 'r.', label=r'$v[t]$')
ax1.plot(T, v_est, 'g*', label=r'$\hat{v}[t]$')
ax1.plot(T, v_est, 'g', alpha = 0.5)
ax1.plot(T, v, 'r--', alpha = 0.5)
ax1.set_ylim([np.min(v) - np.abs(np.min(v)*0.1), np.max(v) + np.abs(np.min(v)*0.1)])
ax1.set_ylabel(r'$v[t]$')
ax1.tick_params('y')
ax1.legend(loc='upper right')

ax1.set_xlim([0, N-1])
ax1.set_rasterized(True)
FULL_PATH = 'C:/Users/milos/OneDrive/VIII semestar/VI/domaci2/izvestaj/slike/zad2/v_no_res.png'

plt.savefig(FULL_PATH, format='png', dpi = 600)
plt.show()


weights = (np.array(weights).T)
weights = weights[0, :, :]





# Plotting the graph
fig, ax1 = plt.subplots()

for i in range(N_particles):
    # Extract position and weight data for the i-th particle
    position = positions[i]
    weight = weights[i]

    # Plotting the positions with scaled dot sizes
    ax1.scatter(T, position, s=100 * weight, alpha = 0.3)
#fig, ax1 = plt.subplots()
# Plotting the position-time graph
ax1.plot(T, x, 'b.', label=r'$x[t]$')
ax1.plot(T, x, 'b--', alpha = 0.5)
ax1.plot(T, e, 'r.', label = r'$e[t]$', markersize = 2)
ax1.plot(T, e, 'r--', alpha = 0.5)
ax1.set_ylim([np.min(e) - np.abs(np.min(e)*0.1), np.max(e) + np.abs(np.min(e)*0.1)])
ax1.set_xlabel(r'$t$')
ax1.set_ylabel(r'$x[t]$')
ax1.tick_params('y')
ax1.legend(loc='upper left')
ax1.set_xlim([0, N])
ax1.set_rasterized(True)

ax1.set_xlim([0, N-1])
FULL_PATH = 'C:/Users/milos/OneDrive/VIII semestar/VI/domaci2/izvestaj/slike/zad2/w_no_res.png'
plt.savefig(FULL_PATH, format='png', dpi = 600)
plt.show()
#%%  iterative_resampling
particle_filt.setAlgorithm("iterative_resampling")
particle_filt.resetParticles()

part = None
weights = [0]*(N)
weights[0] = np.array(particle_filt.old_weights)
part = particle_filt.old_particles
t = 0
positions = np.zeros((N_particles, len(T)))
for num, particle in enumerate(part):
    x_val, _, _ = particle.getValues()
    positions[num, t] = x_val
    
x_est = np.zeros((N, ))
v_est = np.zeros((N, ))
for t in T:
    if t == 50:
        break
    estimation = particle_filt.particle_filtering()
    
    weights[t+1] =  np.array(particle_filt.old_weights)
    part = particle_filt.old_particles
    
    for num, particle in enumerate(part):
        x_val, _, _ = particle.getValues()
        positions[num, t+1] = x_val
    
    x_est[t + 1] = estimation[0]
    v_est[t + 1] = estimation[1]
    
# Plotting the graph
fig, ax1 = plt.subplots()

# Plotting the position-time graph
ax1.plot(T, x, 'b.', label=r'$x[t]$')
ax1.plot(T, x, 'b--', alpha = 0.5)
ax1.plot(T, e, 'r.', label = r'$e[t]$', markersize = 2)
ax1.plot(T, e, 'r--', alpha = 0.5)
ax1.plot(T, x_est, 'k*', label=r'$\hat{x}[t]$')
ax1.plot(T, x_est, 'k', alpha = 0.5)
ax1.set_xlabel(r'$t$')
ax1.set_ylabel(r'$x[t]$')
ax1.tick_params('y')
ax1.set_ylim([np.min(e) - np.abs(np.min(e)*0.1), np.max(e) + np.abs(np.min(e)*0.1)])
ax1.legend(loc='upper left')
ax1.set_xlim([0, N])

ax1.set_xlim([0, N-1])
ax1.set_rasterized(True)
FULL_PATH = 'C:/Users/milos/OneDrive/VIII semestar/VI/domaci2/izvestaj/slike/zad2/x_it_res.png'
plt.savefig(FULL_PATH, format='png', dpi = 600)
plt.show()
fig, ax1 = plt.subplots()

# Plotting the velocity-time graph as a dashed line
ax1.plot(T, v, 'r.', label=r'$v[t]$')
ax1.plot(T, v_est, 'g*', label=r'$\hat{v}[t]$')
ax1.plot(T, v_est, 'g', alpha = 0.5)
ax1.plot(T, v, 'r--', alpha = 0.5)
ax1.set_ylim([np.min(v) - np.abs(np.min(v)*0.1), np.max(v) + np.abs(np.min(v)*0.1)])
ax1.set_ylabel(r'$v[t]$')
ax1.tick_params('y')
ax1.legend(loc='upper right')
ax1.set_rasterized(True)
ax1.set_xlim([0, N-1])
FULL_PATH = 'C:/Users/milos/OneDrive/VIII semestar/VI/domaci2/izvestaj/slike/zad2/v_it_res.png'
plt.savefig(FULL_PATH, format='png', dpi = 600)
plt.show()

weights = (np.array(weights).T)
weights = weights[0, :, :]





# Plotting the graph
fig, ax1 = plt.subplots()

for i in range(N_particles):
    # Extract position and weight data for the i-th particle
    position = positions[i]
    weight = weights[i]

    # Plotting the positions with scaled dot sizes
    ax1.scatter(T, position, s=100 * weight, alpha = 0.3)
#fig, ax1 = plt.subplots()
# Plotting the position-time graph
ax1.plot(T, x, 'b.', label=r'$x[t]$')
ax1.plot(T, x, 'b--', alpha = 0.5)
ax1.plot(T, e, 'r.', label = r'$e[t]$', markersize = 2)
ax1.plot(T, e, 'r--', alpha = 0.5)
ax1.set_ylim([np.min(e) - np.abs(np.min(e)*0.1), np.max(e) + np.abs(np.min(e)*0.1)])
ax1.set_xlabel(r'$t$')
ax1.set_ylabel(r'$x[t]$')
ax1.tick_params('y')
ax1.legend(loc='upper left')
ax1.set_xlim([0, N])


ax1.set_xlim([0, N-1])
ax1.set_rasterized(True)

FULL_PATH = 'C:/Users/milos/OneDrive/VIII semestar/VI/domaci2/izvestaj/slike/zad2/w_it_res.png'
plt.savefig(FULL_PATH, format='png', dpi = 600)
plt.show()
#%%  dynamic_resampling
particle_filt.setAlgorithm("dynamic_resampling")
particle_filt.resetParticles()

part = None
weights = [0]*(N)
weights[0] = np.array(particle_filt.old_weights)
part = particle_filt.old_particles
t = 0
positions = np.zeros((N_particles, len(T)))
for num, particle in enumerate(part):
    x_val, _, _ = particle.getValues()
    positions[num, t] = x_val
    
x_est = np.zeros((N, ))
v_est = np.zeros((N, ))
for t in T:
    if t == 50:
        break
    estimation = particle_filt.particle_filtering()
    
    weights[t+1] =  np.array(particle_filt.old_weights)
    part = particle_filt.old_particles
    
    for num, particle in enumerate(part):
        x_val, _, _ = particle.getValues()
        positions[num, t+1] = x_val
    
    x_est[t + 1] = estimation[0]
    v_est[t + 1] = estimation[1]
    
# Plotting the graph
fig, ax1 = plt.subplots()

# Plotting the position-time graph
ax1.plot(T, x, 'b.', label=r'$x[t]$')
ax1.plot(T, x, 'b--', alpha = 0.5)
ax1.plot(T, e, 'r.', label = r'$e[t]$', markersize = 2)
ax1.plot(T, e, 'r--', alpha = 0.5)
ax1.plot(T, x_est, 'k*', label=r'$\hat{x}[t]$')
ax1.plot(T, x_est, 'k', alpha = 0.5)
ax1.set_ylim([np.min(e) - np.abs(np.min(e)*0.1), np.max(e) + np.abs(np.min(e)*0.1)])
ax1.set_xlabel(r'$t$')
ax1.set_ylabel(r'$x[t]$')
ax1.tick_params('y')
ax1.legend(loc='upper left')
ax1.set_xlim([0, N])


ax1.set_xlim([0, N-1])
ax1.set_rasterized(True)
FULL_PATH = 'C:/Users/milos/OneDrive/VIII semestar/VI/domaci2/izvestaj/slike/zad2/x_dy_res.png'
plt.savefig(FULL_PATH, format='png', dpi = 600)
plt.show()

fig, ax1 = plt.subplots()

# Plotting the velocity-time graph as a dashed line
ax1.plot(T, v, 'r.', label=r'$v[t]$')
ax1.plot(T, v_est, 'g*', label=r'$\hat{v}[t]$')
ax1.plot(T, v_est, 'g', alpha = 0.5)
ax1.plot(T, v, 'r--', alpha = 0.5)
ax1.set_ylim([np.min(v) - np.abs(np.min(v)*0.1), np.max(v) + np.abs(np.min(v)*0.1)])
ax1.set_ylabel(r'$v[t]$')
ax1.tick_params('y')
ax1.legend(loc='upper right')
plt.title('Prava i estimirana vrednost brzine')
ax1.set_xlim([0, N-1])

ax1.set_rasterized(True)
FULL_PATH = 'C:/Users/milos/OneDrive/VIII semestar/VI/domaci2/izvestaj/slike/zad2/v_dy_res.png'
plt.savefig(FULL_PATH, format='png', dpi = 600)
plt.show()
    

weights = (np.array(weights).T)
weights = weights[0, :, :]

    


# Plotting the graph
fig, ax1 = plt.subplots()

for i in range(N_particles):
    # Extract position and weight data for the i-th particle
    position = positions[i]
    weight = weights[i]

    # Plotting the positions with scaled dot sizes
    ax1.scatter(T, position, s=100 * weight, alpha = 0.3)
#fig, ax1 = plt.subplots()
# Plotting the position-time graph
ax1.plot(T, x, 'b.', label=r'$x[t]$')
ax1.plot(T, x, 'b--', alpha = 0.5)
ax1.plot(T, e, 'r.', label = r'$e[t]$', markersize = 2)
ax1.plot(T, e, 'r--', alpha = 0.5)
ax1.set_ylim([np.min(e) - np.abs(np.min(e)*0.1), np.max(e) + np.abs(np.min(e)*0.1)])
ax1.set_xlabel(r'$t$')
ax1.set_ylabel(r'$x[t]$')
ax1.tick_params('y')
ax1.legend(loc='upper left')
ax1.set_xlim([0, N])
ax1.set_rasterized(True)

ax1.set_xlim([0, N-1])

FULL_PATH = 'C:/Users/milos/OneDrive/VIII semestar/VI/domaci2/izvestaj/slike/zad2/w_dy_res.png'
plt.savefig(FULL_PATH, format='png', dpi = 600)
plt.show()
#%%
Nr = 100
mse = np.zeros((3,Nr));

for i in range(Nr):
    N = 50
    x, v, mu = model(N)
    e = measurment(x)
    
    for j, algorithm in enumerate(["no_resampling", "iterative_resampling", "dynamic_resampling"]):
        particle_filt.setAlgorithm(algorithm)
        particle_filt.resetParticles()
        particle_filt.setSensorData(e)
        x_est = np.zeros((N + 1, ))
        for t in T:
            if t == 50:
                break
            estimation = particle_filt.particle_filtering()
            x_est[t + 1] = estimation[0]
            
        mse[j, i] = np.sqrt(np.mean(np.square(x-x_est)))


# fig, ax1 = plt.subplots()
# for i in range(3):
#     # Extract position and weight data for the i-th particle
#     mse_t= mse[i]

#     # Plotting the positions with scaled dot sizes
#     ax1.plot(range(Nr),  mse_t)

import pandas as pd
table = {'mean mse': np.mean(mse, axis = 1), 'median mse': np.median(mse, axis = 1), 'std mse': np.std(mse, axis = 1)}
table = pd.DataFrame(table);
table.index = ['no resampling','iter resampling','dynamic resampling']
print(table)
  
#%%
#backup_table = table
#latex_table = table.to_latex(index=False)
#Save the LaTeX table to a .tex file
#file_path = 'C:/Users/milos/OneDrive/VIII semestar/VI/domaci2/izvestaj/table.tex'
#with open(file_path, 'w') as f:
#    f.write(latex_table)


