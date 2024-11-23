import time
import numpy as np
import torch
import pycuda.driver as cuda

from sklearn.utils.extmath import stable_cumsum
from tslearn.metrics import cdist_soft_dtw
from tslearn.clustering import TimeSeriesKMeans
from tslearn.clustering.kmeans_torch import TimeSeriesKMeansTorch

from SOFTDTW.soft_dtw_cuda import PairwiseSoftDTW

def _k_init_metric(X, n_clusters, cdist_metric, random_state, n_local_trials=None):
    """Init n_clusters seeds according to k-means++ with a custom distance
    metric.

    Parameters
    ----------
    X : array, shape (n_samples, n_timestamps, n_features)
        The data to pick seeds for.

    n_clusters : integer
        The number of seeds to choose

    cdist_metric : function
        Function to be called for cross-distance computations

    random_state : RandomState instance
        Generator used to initialize the centers.

    n_local_trials : integer, optional
        The number of seeding trials for each center (except the first),
        of which the one reducing inertia the most is greedily chosen.
        Set to None to make the number of trials depend logarithmically
        on the number of seeds (2+log(k)); this is the default.

    Notes
    -----
    Selects initial cluster centers for k-mean clustering in a smart way
    to speed up convergence. see: Arthur, D. and Vassilvitskii, S.
    "k-means++: the advantages of careful seeding". ACM-SIAM symposium
    on Discrete algorithms. 2007

    Version adapted from scikit-learn for use with a custom metric in place of
    Euclidean distance.
    """
    n_samples, n_timestamps, n_features = X.shape

    centers = np.empty((n_clusters, n_timestamps, n_features), dtype=X.dtype)

    # Set the number of local seeding trials if none is given
    if n_local_trials is None:
        # This is what Arthur/Vassilvitskii tried, but did not report
        # specific results for other than mentioning in the conclusion
        # that it helped.
        n_local_trials = 2 + int(np.log(n_clusters))

    # Pick first center randomly
    center_id = random_state.randint(0, n_samples)
    centers[0] = X[center_id]

    # Initialize list of closest distances and calculate current potential
    closest_dist_sq = cdist_metric(centers[0, np.newaxis], X) ** 2
    current_pot = closest_dist_sq.sum()

    # Pick the remaining n_clusters-1 points
    for c in range(1, n_clusters):
        # Choose center candidates by sampling with probability proportional
        # to the squared distance to the closest existing center
        rand_vals = random_state.random_sample(n_local_trials) * current_pot
        scumsum = stable_cumsum(closest_dist_sq)
        candidate_ids = np.searchsorted(scumsum, rand_vals)
        # XXX: numerical imprecision can result in a candidate_id out of range
        np.clip(candidate_ids, None, closest_dist_sq.size - 1, out=candidate_ids)

        # Compute distances to center candidates
        distance_to_candidates = cdist_metric(X[candidate_ids], X) ** 2

        # update closest distances squared and potential for each candidate
        np.minimum(
            closest_dist_sq, distance_to_candidates, out=distance_to_candidates
        )
        candidates_pot = distance_to_candidates.sum(axis=1)

        # Decide which candidate is the best
        best_candidate = np.argmin(candidates_pot)
        current_pot = candidates_pot[best_candidate]
        closest_dist_sq = distance_to_candidates[best_candidate]
        best_candidate = candidate_ids[best_candidate]

        # Permanently add best center candidate found in local tries
        centers[c] = X[best_candidate]

    return centers

def test_distance_function():
    n_samples, n_clusters, sz, d = 4000, 10, 128, 1

    X = np.random.randn(n_samples, sz, d)   
    Y = np.random.rand(n_clusters, sz, d)

    cdist_soft_gpu = PairwiseSoftDTW(gamma=1, precision=torch.float32)
    st = time.time() 
    distance_gpu = cdist_soft_gpu(X, Y) 
    print(f"gpu-time:{time.time() -st}")

    st = time.time()
    distance_cpu = cdist_soft_dtw(X, Y, gamma=1) 
    print(f"cpu-time: {time.time() - st}")

    diff = np.linalg.norm(distance_cpu- distance_gpu.cpu().numpy())
    tol = 1e-4
    # assert diff < tol, "no, no, no"


def test_k_means_init():
    # Generate synthetic time series data
    n_samples, sz, d = 50, 32, 1
    X = np.random.randn(n_samples, sz, d)
    seed = 0

    # Set fixed random state for reproducibility
    random_state = np.random.RandomState(seed)
    
    # Create metric parameters for SoftDTW
    metric_params = {"gamma": 0.5}
    
    # Initialize TimeSeriesKMeans with SoftDTW metric
    # Use 'k-means++' initialization
    kmeans = TimeSeriesKMeans(
        n_clusters=3,
        metric="softdtw",
        metric_params=metric_params,
        random_state=random_state,
        init="k-means++"
    )
    
    # Since _k_init_metric is a private method, we'll access it directly for testing
    # Define the distance metric function using cdist_soft_dtw
    
    def softdtw_distance(x, y):
        return cdist_soft_dtw(x, y, **metric_params)
    
    # Perform initialization using the original implementation
    centers_tslearn = _k_init_metric(
        X,
        kmeans.n_clusters,
        cdist_metric=softdtw_distance,
        random_state=random_state
    )
    
    random_state = np.random.RandomState(seed)
    # Initialize TimeSeriesKMeansTorch
    kmeans_torch = TimeSeriesKMeansTorch(
        n_clusters=3,
        gamma=metric_params["gamma"],
        random_state=random_state
    )
    
    # Perform initialization using the PyTorch implementation
    centers_torch = kmeans_torch._kmeans_init(torch.from_numpy(X).to(device='cuda'), random_state=random_state)
    
    # Compare the initialized centers
    # Compute the difference between centers
    diff = np.linalg.norm(centers_tslearn - centers_torch.cpu().numpy())
    print(f"Difference between initialized centers: {diff}")
    
    # Optionally, assert that the difference is within an acceptable tolerance
    tolerance = 1e-6
    assert diff < tolerance, "The initialized centers differ more than the acceptable tolerance."
    
    print("k-means++ initialization test passed!")

def test_fit_one_init(X_ntd):
    kmeans = TimeSeriesKMeans(n_clusters=3, metric='softdtw')
    kmeans_toch = TimeSeriesKMeansTorch(n_clusters=3)

    x_squared_norms = None 
    random_state =np.random.mtrand._rand

    # Run one init on cpu 
    kmeans.labels_ = None
    kmeans.inertia_ = np.inf
    kmeans.cluster_centers_ = None
    kmeans._X_fit = None
    kmeans._squared_inertia = True
    kmeans._fit_one_init(X_ntd, x_squared_norms=x_squared_norms, rs=random_state)
    labels, clusters, inertia = kmeans._get_one_init_results()

    # Run one init on cpu 
    labels_t, clusters_t, inertia_t = kmeans_toch._fit_one_init(X_ntd, x_squared_norms, rs=random_state)

    print("done") 

def test_centroid_update():
    pass

def test_predict():
    pass

def k_means_cpu(S, n_clusters):
    # Initialize CUDA context properly
    cuda.init()
    cpu_time_start = time.time()
    k_means = TimeSeriesKMeans(n_clusters=n_clusters, metric="softdtw").fit(S)
    print(f"CPU time: {time.time()- cpu_time_start}")
    return k_means

def k_means_gpu(S, n_clusters):
    gpu_time_start = time.time()
    print(f"Available GPU: {cuda.Device(0).name()}")
    k_means = TimeSeriesKMeans(n_clusters=n_clusters, metric="gpudtw").fit(S)
    print(f"GPU time: {time.time() - gpu_time_start}")
    return k_means

if __name__ == '__main__':
    time_series_length = 1024
    n_series = 12 
    n_clusters = 4
    dim = 1
    S_ntd = np.random.random((n_series, time_series_length, dim))

    #test_distance_function()
    test_k_means_init()