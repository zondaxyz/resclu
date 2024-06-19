import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from joblib import Parallel, delayed
import warnings
from sklearn.decomposition import PCA
from kneed import KneeLocator



warnings.filterwarnings('ignore')

# Elbow method to determine the optimal number of clusters
def find_optimal_clusters(data, max_k):
    if max_k < 2:
        return 1
    iters = range(2, max_k + 1)
    sse = []
    for k in iters:
        kmeans = KMeans(n_clusters=k, random_state=123, n_init='auto')
        kmeans.fit(data)
        sse.append(kmeans.inertia_)

    # Find the elbow point, i.e., the point of maximum curvature
    kneedle = KneeLocator(iters, sse, curve="convex", direction="decreasing")
    return kneedle.elbow if kneedle.elbow else 2

def reduce_dimensions(data, target_dim=1):
    try:
        #print('PCA')
        pca = PCA(n_components=target_dim)
        reduced_data = pca.fit_transform(data)
        return pd.DataFrame(reduced_data)
    except:
        return data

def bstpfdt(data, size,inu=False):
    rows, cols = data.shape
    #print(rows, cols)
    indices = np.random.randint(0, rows, (size, cols))
    ps = data.to_numpy()[indices, np.arange(cols)]
    if inu:
        data=pd.DataFrame(ps.T)
        rows, cols = data.shape
    #print(rows, cols)
        indices = np.random.randint(0, rows, (rows, cols))
        ps = data.to_numpy()[indices, np.arange(cols)]
        return ps.T
    return ps

def predcoh(data,stitime=20,prt=False):
    varcoh=[]
    for i in range(stitime):
        y_pred1 = KMeans(n_clusters=2, random_state=np.random.randint(0,1000)).fit_predict(data)
        y_pred2 = KMeans(n_clusters=2, random_state=np.random.randint(0,1000)).fit_predict(data)
        varcoh.append(abs(np.corrcoef(y_pred1,y_pred2)))
    if prt:
        print(np.average(varcoh))
    return np.average(varcoh)

def pcavar(data,stitime=20, prt=False):
    stitime='''This is no longer needed in current version of this function,
    but I don't want to drop it because I need to modify a lot other functions if I
    delete this parameter.'''
    pcasti=PCA(n_components=data.shape[1],svd_solver='full')
    evr=pcasti.fit(data).explained_variance_ratio_
    stdpca=np.std(evr)
    if prt:
        print(evr,stdpca)
    return stdpca

def parallel_predcoh(data, stitime,prt=False):
    return predcoh(data, stitime, prt=prt)

def parallel_pcavar(data, stitime,prt=False):
    return pcavar(data, stitime, prt=prt)

def PCPTest(data, sti1=100, sti2=20,inu=False,prt=False,fun='predcoh'):
    if fun=='predcoh':
        pctr = predcoh(data, sti2,prt=prt)
        size = min(20000, len(data))
        pcbs = Parallel(n_jobs=-1)(delayed(parallel_predcoh)(bstpfdt(data, size, inu=inu), sti2) for _ in range(sti1))
        #print(pcbs)
        return (np.array(pcbs) > pctr).sum() / sti1
    elif fun=='pcavar':
        pctr = pcavar(data, sti2,prt=prt)
        size = min(20000, len(data))
        pcbs = Parallel(n_jobs=-1)(delayed(parallel_pcavar)(bstpfdt(data, size, inu=inu), sti2) for _ in range(sti1))
        #print(pcbs)
        return (np.array(pcbs) > pctr).sum() / sti1

def cluster(data,sti1=100,sti2=30,pth=0.05,inu=True,prt=False,fun='predcoh'):
    if len(data) < 3:
        return None
    #if len(data) < 3:  # Adjusting the minimum size for making at least 2 clusters
     #   return np.zeros(len(data), dtype=int)
    p=PCPTest(data,sti1=sti1,sti2=sti2,inu=inu,prt=prt,fun=fun)
    if  p > pth:
        print(f'p={p},没有通过PCP检验')
        return None
    # Use the elbow method to find the optimal number of clusters
    best_k = find_optimal_clusters(data, max_k=min(10, len(data)))
    if best_k > len(data):
        best_k = len(data)
    labels = KMeans(n_clusters=best_k, n_init='auto')
    labels.fit(data)
    return labels.labels_

def labels_to_dataframe(label_path):
    """
    Converts hierarchical label paths dictionary to a pandas DataFrame.
    Args:
        label_path (dict): Dictionary with hierarchical label paths.
    Returns:
        pd.DataFrame: DataFrame with hierarchical labels, filling empty places with NaN.
    """
    max_depth = max(len(path) for path in label_path.values())
    data = {idx: path + [np.nan] * (max_depth - len(path)) for idx, path in label_path.items()}
    df_labels = pd.DataFrame.from_dict(data, orient='index')
    df_labels.columns = [f'Level_{i}' for i in range(max_depth)]
    return df_labels

def recursive_cluster(df, label_path=None, max_depth=10, current_depth=0,
                      sti1=100,sti2=30,pth=0.05,
                      PCAtoN=True,n_pca=None,hitvar=0.95,inu=False,fun='predcoh',faster=False):
    """
    Recursively performs dynamic clustering on a given dataframe.
    Args:
        df (pd.DataFrame): Input dataframe.
        label_path (dict, optional): Dictionary to store hierarchical label paths for each index. Defaults to None.
        max_depth (int, optional): Maximum depth of recursion. Defaults to 10.
        current_depth (int, optional): Current depth of recursion. Defaults to 0.
    """
    inu_clu=False
    #print(current_depth)
    if not PCAtoN:
        df_pca=df.copy()
    if PCAtoN and current_depth!=0 and faster:
        df_pca=df.copy()
    if PCAtoN and current_depth==0 and faster:
        inu=True
        fun='pcavar'
        if n_pca is not None:
            df=reduce_dimensions(df,target_dim=n_pca)
        else:
            pca_=PCA(n_components=df.shape[1])
            pca_.fit(df)
            varsra=pca_.explained_variance_ratio_
            n_pca=0
            while sum(varsra[0:n_pca])<hitvar:
                n_pca+=1
            df=reduce_dimensions(df,target_dim=n_pca)
        df_pca=df.copy()
    if PCAtoN and (not faster):
        inu_clu=True
        #print('inu is True now!')
        if n_pca is not None:
            df_pca=reduce_dimensions(df,target_dim=n_pca)
        else:
            pca_=PCA(n_components=df.shape[1])
            pca_.fit(df)
            varsra=pca_.explained_variance_ratio_
            n_pca=0
            while sum(varsra[0:n_pca])<hitvar:
                n_pca+=1
            df_pca=reduce_dimensions(df,target_dim=n_pca)

    if label_path is None:
        label_path = {idx: [] for idx in df.index}

    # Base case: Check if the dataframe is suitable for clustering
    if len(df) < df.shape[1] or current_depth >= max_depth:  # Ensure there are enough samples and depth limit
        return label_path

    labels = cluster(df_pca,sti1=sti1,sti2=sti2,pth=pth,inu=(inu or inu_clu),fun=fun)
    if labels is None:
        return label_path

    # Recursive case: Split dataframe into clusters and apply clustering recursively
    unique_labels = np.unique(labels)
    for label in unique_labels:
        cluster_df = df[labels == label]
        if len(cluster_df) < 2:  # Ensure each subset has sufficient samples to proceed with clustering
            continue
        for idx in cluster_df.index:
            label_path[idx].append(label)
        label_path = recursive_cluster(cluster_df, label_path, max_depth, current_depth + 1,
                                       sti1=sti1,sti2=sti2,pth=pth,inu=inu,fun=fun,
                                       PCAtoN=PCAtoN,faster=faster)
    return label_path

def cutlevel(dflabel,t=0.1):
    for i in dflabel.columns.tolist():
        rnan=1-(dflabel[i].isnull()).sum()/len(dflabel)
        print(rnan)
        if rnan<t:
            dflabel=dflabel.drop([i],axis=1)
    return dflabel
