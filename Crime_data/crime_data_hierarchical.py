import pandas as pd
import matplotlib.pylab as plt 
Crime = pd.read_csv("C:/Training/Analytics/Clustering/Crime_data/crime_data.csv")

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

# =============================================================================
# With method single Linkage and euclidean metric
# =============================================================================

z = linkage(df_norm, method="single",metric="euclidean")

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
h_single	=	AgglomerativeClustering(n_clusters=3,	linkage='single',affinity = "euclidean").fit(df_norm) 


cluster_labels=pd.Series(h_single.labels_)

Crime_single = Crime
Crime_single['clust']=cluster_labels # creating a  new column and assigning it to new column 
Crime_single = Crime.iloc[:,[5,0,1,2,3,4]]
Crime_single.head()

# getting aggregate mean of each cluster
Crime_single.iloc[:,2:].groupby(Crime_single.clust).median()

# =============================================================================
# With method complete linkage and euclidean metric
# =============================================================================

#p = np.array(df_norm) # converting into numpy array format 

z = linkage(df_norm, method="complete",metric="euclidean")

plt.figure(figsize=(15, 5));plt.title('Hierarchical Clustering Dendrogram');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(
    z,
    leaf_rotation=0.,  # rotates the x axis labels
    leaf_font_size=8.,  # font size for the x axis labels
)
plt.show()

# Now applying AgglomerativeClustering choosing 3 as clusters from the dendrogram
from	sklearn.cluster	import	AgglomerativeClustering 
h_complete	=	AgglomerativeClustering(n_clusters=4,	linkage='complete',affinity = "euclidean").fit(df_norm) 


cluster_labels=pd.Series(h_complete.labels_)
Crime_complete = Crime
Crime_complete['clust']=cluster_labels # creating a  new column and assigning it to new column 
Crime_complete = Crime_complete.iloc[:,[5,0,1,2,3,4]]
Crime_complete.head()

# getting aggregate mean of each cluster
Crime_complete.iloc[:,2:].groupby(Crime_complete.clust).median()

# =============================================================================
# With method average linkage and euclidean metric
# =============================================================================

#p = np.array(df_norm) # converting into numpy array format 

z = linkage(df_norm, method="average",metric="euclidean")

plt.figure(figsize=(15, 5));plt.title('Hierarchical Clustering Dendrogram');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(
    z,
    leaf_rotation=0.,  # rotates the x axis labels
    leaf_font_size=8.,  # font size for the x axis labels
)
plt.show()

# Now applying AgglomerativeClustering choosing 3 as clusters from the dendrogram
from	sklearn.cluster	import	AgglomerativeClustering 
h_complete	=	AgglomerativeClustering(n_clusters=5,	linkage='average',affinity = "euclidean").fit(df_norm) 


cluster_labels=pd.Series(h_complete.labels_)
Crime_average = Crime
Crime_average['clust']=cluster_labels # creating a  new column and assigning it to new column 
Crime_average = Crime_average.iloc[:,[5,0,1,2,3,4]]
Crime_average.head()

# getting aggregate mean of each cluster
Crime_average.iloc[:,2:].groupby(Crime_average.clust).median()

# =============================================================================
# With method ward linkage and euclidean metric
# =============================================================================

#p = np.array(df_norm) # converting into numpy array format 

z = linkage(df_norm, method="ward",metric="euclidean")

plt.figure(figsize=(15, 5));plt.title('Hierarchical Clustering Dendrogram');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(
    z,
    leaf_rotation=0.,  # rotates the x axis labels
    leaf_font_size=8.,  # font size for the x axis labels
)
plt.show()

# Now applying AgglomerativeClustering choosing 3 as clusters from the dendrogram
from	sklearn.cluster	import	AgglomerativeClustering 
h_complete	=	AgglomerativeClustering(n_clusters=2,	linkage='ward',affinity = "euclidean").fit(df_norm) 


cluster_labels=pd.Series(h_complete.labels_)
Crime_ward = Crime
Crime_ward['clust']=cluster_labels # creating a  new column and assigning it to new column 
Crime_ward = Crime_ward.iloc[:,[5,0,1,2,3,4]]
Crime_ward.head()

# getting aggregate mean of each cluster
Crime_ward.iloc[:,2:].groupby(Crime_ward.clust).median()


# =============================================================================
# Distance
# =============================================================================

# =============================================================================
# With method complete linkage and Manhattan metric
# =============================================================================

#p = np.array(df_norm) # converting into numpy array format 

z = linkage(df_norm, method="complete",metric="cityblock")

plt.figure(figsize=(15, 5));plt.title('Hierarchical Clustering Dendrogram');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(
    z,
    leaf_rotation=0.,  # rotates the x axis labels
    leaf_font_size=8.,  # font size for the x axis labels
)
plt.show()

# Now applying AgglomerativeClustering choosing 3 as clusters from the dendrogram
from	sklearn.cluster	import	AgglomerativeClustering 
h_complete	=	AgglomerativeClustering(n_clusters=2,	linkage='complete',affinity = "cityblock").fit(df_norm) 


cluster_labels=pd.Series(h_complete.labels_)
Crime_complete = Crime
Crime_complete['clust']=cluster_labels # creating a  new column and assigning it to new column 
Crime_complete = Crime_complete.iloc[:,[5,0,1,2,3,4]]
Crime_complete.head()

# getting aggregate mean of each cluster
Crime_complete.iloc[:,2:].groupby(Crime_complete.clust).median()