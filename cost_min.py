import gurobipy as gp
from gurobipy import GRB
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
import random
import sys

m = gp.Model("cost_min")

# read coordinate data
orin_data = np.loadtxt("C:\\Users\\mamen\OneDrive\\桌面\\data.txt")
coord = orin_data[:,1:3]
t = cdist(coord, coord, 'euclidean')

# parameter
station_num = len(coord)
truck_num = 10

g = 1.5e5 #先设定为每个j都是一样的价格
h = 200
Q = -50
sigma = 0.0001 # a small positive number
M = 1e6 # a large positive number
phi = 0.4 # maximum waiting time at a swapping station
d = 100

# create paths
paths = list()
for i in range(truck_num):
    path_points_num = np.random.randint(6,9)
    path = random.sample(range(station_num), k=path_points_num)
    paths.append(path)

# caculate distance matrix(speed=1)
t = cdist(coord, coord, 'euclidean')

# precursor node
c = np.zeros((truck_num, station_num, station_num))
for i in range(truck_num):
    for point_i in range(1,len(paths[i])):
        c[i, paths[i][point_i-1], paths[i][point_i]] = 1

# caclulate a
a = np.zeros((truck_num, station_num))
for i in range(truck_num):
    for point_i in range(1,len(paths[i])):
        a[i,paths[i][point_i]] = a[i,paths[i][point_i-1]] + t[paths[i][point_i-1], paths[i][point_i]]

#把a第二个索引改成所有的站点

Gamma = list()
for j in range(station_num):
    Gamma_J = list()
    for i in range(truck_num):
        if j in paths[i]:
            Gamma_J.append(i)
    Gamma.append(Gamma_J)

upsilon = paths

# create decision variables
X = m.addVars(station_num, lb=0, ub=GRB.INFINITY, vtype=GRB.INTEGER, name="X")
Y = m.addVars(truck_num, station_num, vtype=GRB.BINARY, name="Y")
V = m.addVars(station_num, lb=0, ub=GRB.INFINITY, vtype=GRB.INTEGER, name="V")
Z = m.addVars(station_num, truck_num, truck_num, vtype=GRB.BINARY, name="Z")
w = m.addVars(truck_num, station_num, lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="w")
r = m.addVars(truck_num, station_num, lb=0, ub=100, vtype=GRB.CONTINUOUS, name="r")
r_bar = m.addVars(truck_num, station_num, lb=0, ub=100, vtype=GRB.CONTINUOUS, name="r_bar")
p = m.addVars(truck_num, station_num, lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="p")
u = m.addVars(truck_num, station_num, vtype=GRB.BINARY, name="u")
v = m.addVars(truck_num, station_num, vtype=GRB.BINARY, name="v")


# m.modelSense = GRB.MINIMIZE
# m.setObjective(gp.quicksum(x[i,j]*c[i,j] for i,j in A))


m.setObjective(gp.quicksum(g* X[j] + h * V[j] for j in range(station_num)) + gp.quicksum(Q * w[i, j] for i in range(truck_num) for j in range(station_num)), gp.GRB.MINIMIZE)
print(m.getObjective())



# Add the constraints

for j in range(station_num):
    for i in Gamma[j]:
        for k in Gamma[j]:
            m.addConstr(Z[j, i, k] <= Y[i, j], name="cons1")
            m.addConstr(Z[j, i, k] <= Y[k, j], name="cons2")
            m.addConstr(Z[j, i, k] <= M * (a[k,j] - a[i,j] - (d - r[i, j])/4.2 + sigma), name="cons3")

for j in range(station_num):
        for k in Gamma[j]:
            m.addConstr(gp.quicksum(Z[j, i, k] for i in Gamma[j]) <= 1, name="cons4")
            m.addConstr(w[k,j] >= (p[i,j])-p[k,j]+(d-r[i,j])/4.2-M*(Y[k,j]-gp.quicksum(Z[j, i, k] for i in Gamma[j])), name="cons5")
            m.addConstr(w[k,j] >= 0, name="cons6")
            

for j in range(station_num):
    for i in range(truck_num):
        m.addConstr(gp.quicksum(Z[j, i, k] for k in Gamma[j]) <= 1, name="cons7")
        m.addConstr(r[i,j]>=0, name="cons8")
        # m.addGenConstrMin(r_bar[i,j],d*Y[i,j]+r[i,j]*(1-Y[i,j]),d*Y[i,j]+r[i,j])
        m.addConstr(r[i,j] <= d*Y[i,j]+r[i,j]*(1-Y[i,j]), name="cons9.1")
        m.addConstr(r[i,j] <= d*Y[i,j]+r[i,j], name="cons9.2")
        m.addConstr(d*Y[i,j]+r[i,j]*(1-Y[i,j]) <= r[i,j]-M*(1-u[i,j]), name="cons9.3")
        m.addConstr(d*Y[i,j]+r[i,j] <= r[i,j]-M*(1-v[i,j]), name="cons9.4")
        m.addConstr(u[i,j]+v[i,j] >= 1, name="cons9.5")
                    
        m.addConstr(r[i,j] == gp.quicksum(c[i,j,k]*(r_bar[i,j]-t[k,j]) for k in upsilon[i]), name="cons10")
        m.addConstr(Y[i, j] <= X[j], name="cons11")

for i in Gamma[j]:
    for j in range(station_num):
        m.addConstr(w[k,j] <= gp.quicksum(Z[j, i, k]*phi for k in Gamma[j]), name="cons12")
        m.addConstr(w[k,j] <= gp.quicksum(Y[k,j]*phi for k in Gamma[j]), name="cons13")

for j in range(station_num):
    m.addConstr(V[j] <= 10*X[j], name="cons14")
    m.addConstr(gp.quicksum(Y[i, j] for i in range(truck_num)) >= 1, name="cons15")
    m.addConstr((gp.quicksum(Y[i, j] for i in range(truck_num))-gp.quicksum(Z[j, i, k] for k in Gamma[j] for i in Gamma_J)) <= V[j], name="cons16")


m.write('cost_min.lp')

m.setParam(GRB.Param.DualReductions, 0)
m.optimize()


if m.status == GRB.Status.INFEASIBLE:
        print('Optimization was stopped with status %d' % m.status)
        # do IIS, find infeasible constraints
        m.computeIIS()
        for c in m.getConstrs():
            if c.IISConstr:
                print('%s' % c.constrName)






# plt.scatter(coord[:,0], coord[:,1])
