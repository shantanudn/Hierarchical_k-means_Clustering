import pandas as pd
import matplotlib.pylab as plt 
EastWestAirlines_ori = pd.read_excel("C:/Training/Analytics/Clustering/EastWestAirlines/EastWestAirlines.xlsx", sheet_name='data')
EastWestAirlines=EastWestAirlines_ori.drop(['ID#'],axis=1)
#EastWestAirlines.isnull()
# Normalization function 
#def norm_func(i):
 #   x = (i-i.min())	/	(i.max()	-	i.min())
  #  return (x)

# alternative normalization function 

def norm_func(i):
    x = (i-i.mean())/(i.std())
    return (x)

# Normalized data frame (considering the numerical part of data)
df_norm = norm_func(EastWestAirlines.iloc[:,0:])

from scipy.cluster.hierarchy import linkage 

import scipy.cluster.hierarchy as sch # for creating dendrogram 

type(df_norm)

#p = np.array(df_norm) # converting into numpy array format 
help(linkage)
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
h_complete	=	AgglomerativeClustering(n_clusters=10,	linkage='complete',affinity = "euclidean").fit(df_norm) 


cluster_labels=pd.Series(h_complete.labels_)

EastWestAirlines['clust']=cluster_labels # creating a  new column and assigning it to new column 
EastWestAirlines = EastWestAirlines.iloc[:,[11,0,1,2,3,4,5,6,7,8,9,10]]
EastWestAirlines.head()

# getting aggregate mean of each cluster
median = EastWestAirlines.iloc[:,2:].groupby(EastWestAirlines.clust).median()

# creating a csv file 
EastWestAirlines.to_csv("EastWestAirlines.csv",encoding="utf-8")
