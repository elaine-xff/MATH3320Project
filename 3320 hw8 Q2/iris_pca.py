import numpy as np
import pandas as pd
import plotly.express as px

df =  pd.read_csv('iris.data', sep=",", header=None, names=['A', 'B', 'C', 'D', 'E'])
df = df.drop(columns=['E'], axis=1)
# (a)(b) download and preprocess the data by subtracting the mean and dividing by the sd of each attribute value
pc_df = (df - df.mean())/df.std()
print(pc_df)
print(pc_df.mean())
print(pc_df.std())

# (c) compute the covariance matrix
m = pc_df.shape[0] 
cov_m = 1/m * (pc_df.transpose().dot(pc_df))
print('covariance matrix is')
print(cov_m)

# (d) SVD of covariance matrix
u, s, v = np.linalg.svd(cov_m, full_matrices=True)
print('u is') 
print(u)
print('s is')
print(s)
print('v is')
print(v)

# (e) project the data onto its first two principal components and plot the results
phi = []
y = []
for i in range (0, 4):
    phi.append(np.array([v[i]]).T)
    y.append(pc_df.dot(phi[i]))
    if i == 2 or i == 3:
        y[i] = y[i].mean() * pd.DataFrame(np.ones((pc_df.shape[0], 1)))

name_list = ['A', 'B', 'C', 'D']
projection = 0
for i in range (0, 4):
    projection = projection + y[i].dot(phi[i].transpose())
projection.columns = name_list

fig = px.scatter_matrix(
    projection,
    dimensions=name_list,
)
fig.update_traces(diagonal_visible=False)
fig.show()
