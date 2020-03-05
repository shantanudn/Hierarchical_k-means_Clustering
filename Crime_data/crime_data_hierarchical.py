import pandas as pd
import matplotlib.pylab as plt 
Crime = pd.read_csv("C:/Training/Analytics/Clustering/crime_data.csv")

# Normalization function 
#def norm_func(i):
 #   x = (i-i.min())	/	(i.max()	-	i.min())
  #  return (x)

# alternative normalization function 

def norm_func(i):
    x = (i-i.mean())/(i.std())
    return (x)

# Normalized data frame (considering the numerical part of data)
df_norm = norm_func(Crime.iloc[:,1:])

from scipy.cluster.hierarchy import linkage 

import scipy.cluster.hierarchy as sch # for creating dendrogram 

type(df_norm)

#p = np.array(df_norm) # converting into numpy array format 

z = linkage(df_norm, method="complete",metric="euclidean")

plt.figure(figsize=(15, 5));plt.title('Hierarchical Clustering Dendrogram');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(
    z,
    leaf_rotation=0.,  # rotates the x axis labels
    leaf_font_size=8.,  # font size for the x axis labels
)
plt.show()

help(linkage)

# Now applying AgglomerativeClustering choosing 3 as clusters from the dendrogram
from	sklearn.cluster	import	AgglomerativeClustering 
h_complete	=	AgglomerativeClustering(n_clusters=4,	linkage='complete',affinity = "euclidean").fit(df_norm) 


cluster_labels=pd.Series(h_complete.labels_)

Crime['clust']=cluster_labels # creating a  new column and assigning it to new column 
Crime = Crime.iloc[:,[5,0,1,2,3,4]]
Crime.head()

# getting aggregate mean of each cluster
Crime.iloc[:,2:].groupby(Crime.clust).median()

# creating a csv file 
Crime.to_csv("Crime_grouped.csv",encoding="utf-8")
