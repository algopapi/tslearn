import torch
import numpy as np
from torch.optim import LBFGS

from SOFTDTW.soft_dtw_cuda import PairwiseSoftDTW
from SOFTDTW.soft_dtw_barycenter import SoftDTWBarycenter

class TimeSeriesKMeansTorch:
    """TimeSeries K-Means clustering using SoftDTW and PyTorch.

    Parameters
    ----------
    n_clusters : int, default=3
        The number of clusters to form.

    max_iter : int, default=50
        Maximum number of iterations of the k-means algorithm.

    tol : float, default=1e-6
        Relative tolerance with regards to inertia to declare convergence.

    gamma : float, default=1.0
        SoftDTW gamma parameter.

    device : str, default='cuda'
        Device to use for computations ('cuda' or 'cpu').

    n_init : int, default=1
        Number of time the k-means algorithm will be run with different centroid seeds.

    Attributes
    ----------
    cluster_centers_ : torch.Tensor
        Cluster centers (barycenters), of shape (n_clusters, seq_len, n_features).

    labels_ : numpy.ndarray
        Labels of each time series.
    """
    def __init__(
            self, 
            n_clusters=3, 
            max_iter=50, 
            tol=1e-6, 
            gamma=1.0, 
            device='cuda', 
            n_init=1,
            random_state=None,
        ):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state
        self.tol = tol
        self.gamma = gamma
        self.device = device
        self.n_init = n_init
        self.distance = PairwiseSoftDTW(gamma=self.gamma)
        self.barycenter = None

    def fit(self, X):
        """
        Compute k-means clustering.

        Parameters
        ----------
        X : array-like of shape (n_samples, seq_len, n_features)
            Training data.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        # Convert data to torch tensor
        X = torch.tensor(X, dtype=torch.float32).to(self.device)
        n_samples = X.shape[0]

        best_inertia = float('inf')
        best_labels = None
        best_centers = None

        for init_no in range(self.n_init):
            random_state = None if self.n_init == 1 else torch.randint(0, 10000, (1,)).item()
            labels, inertia, centers = self.fit_one_init(X, random_state=random_state)

            if inertia < best_inertia:
                best_inertia = inertia
                best_labels = labels
                best_centers = centers

        self.labels_ = best_labels.cpu().numpy()
        self.cluster_centers_ = best_centers.detach()
        return self

    def predict(self, X):
        """
        Predict the closest cluster each time series in X belongs to.

        Parameters
        ----------
        X : array-like of shape (n_samples, seq_len, n_features)
            New data to predict.

        Returns
        -------
        labels : numpy.ndarray
            Index of the cluster each sample belongs to.
        """
        X = torch.tensor(X, dtype=torch.float32).to(self.device)
        distances = self._compute_soft_dtw_distances(X, self.cluster_centers_)
        labels = torch.argmin(distances, dim=1)
        return labels.cpu().numpy()

    def fit_one_init(self, X, random_state=None):
        """
        Initialize cluster centers and perform clustering with one initialization.

        Parameters
        ----------
        X : torch.Tensor of shape (n_samples, seq_len, n_features)
            Training data.

        random_state : int or None
            Seed for random number generator.

        Returns
        -------
        labels : torch.Tensor
            Labels of each time series.
        inertia : float
            Sum of distances (inertia) for assigned clusters.
        centers : torch.Tensor
            Cluster centers.
        """
        n_samples = X.shape[0]

        # Initialize cluster centers using k-means++
        cluster_centers = self._kmeans_init(X, random_state=random_state).clone().requires_grad_(True)

        for i in range(self.max_iter):
            # Compute distances between each time series and each cluster center
            distances = self.distance(X, cluster_centers)

            # Assign each time series to the nearest cluster center
            labels = torch.argmin(distances, dim=1)

            # Compute inertia (sum of distances for assigned clusters)
            inertia = distances[torch.arange(n_samples), labels].sum()

            # Update cluster centers
            new_centers = []
            for k in range(self.n_clusters):
                cluster_members = X[labels == k]
                if cluster_members.nelement() == 0:
                    # If a cluster has no members, re-initialize its center
                    new_center = X[torch.randint(0, n_samples, (1,))].squeeze(0)
                else:
                    # Compute the barycenter for the cluster
                    new_center = self.barycenter(cluster_members)
                new_centers.append(new_center)
            new_centers = torch.stack(new_centers)

            # Check for convergence
            center_shift = torch.norm(cluster_centers - new_centers)
            if center_shift < self.tol:
                break

            cluster_centers = new_centers.clone().detach().requires_grad_(True)

        return labels, inertia.item(), cluster_centers

    def _kmeans_init(self, X, random_state):
        """
        Initialize cluster centers using the k-means++ algorithm.

        Parameters
        ----------
        X : torch.Tensor of shape (n_samples, seq_len, n_features)
            Training data.

        random_state : numpy.RandomState
            Random number generator.

        Returns
        -------
        centers : torch.Tensor of shape (n_clusters, seq_len, n_features)
            Initialized cluster centers.
        """
        n, t, d = X.shape
        n_clusters = self.n_clusters

        n_local_trials = 2 + int(np.log(n_clusters))
        centers = torch.empty((n_clusters, t, d), dtype=X.dtype, device=X.device)

        # Choose the first center using NumPy's random_state
        c_id = random_state.randint(0, n)
        centers[0] = X[c_id]

        # Initialize list of squared distances to closest center
        closest_dist_sq = self.distance(centers[0].unsqueeze(0), X) ** 2
        current_pot = closest_dist_sq.sum().item()

        for c in range(1, n_clusters):
            # Generate rand_vals using NumPy's random_state
            rand_vals_np = random_state.random_sample(n_local_trials) * current_pot

            # Convert rand_vals to PyTorch tensor on GPU with appropriate dtype
            rand_vals = torch.from_numpy(rand_vals_np).to(
                device=X.device, 
                dtype=closest_dist_sq.dtype
            )

            # Compute cumulative sum of distances
            c_ids = torch.searchsorted(torch.cumsum(closest_dist_sq.flatten(), dim=0), rand_vals)
            max = closest_dist_sq.size(1) -1
            c_ids = torch.clamp(c_ids, min=None, max=max)

            # Compute distances to center candidates
            distance_to_candidates = self.distance(X[c_ids], X) ** 2

            # Update closest distances squared and potential for each candidate
            # shape (3,)
            closest_dist_sq_candidate = torch.minimum(closest_dist_sq, distance_to_candidates)
            candidates_pot = closest_dist_sq_candidate.sum(dim=1)

            # Decide which candidate is the best
            best_candidate = torch.argmin(candidates_pot)
            current_pot = candidates_pot[best_candidate].item()
            closest_dist_sq = closest_dist_sq_candidate[best_candidate].unsqueeze(0)
            best_candidate_id = c_ids[best_candidate]

            # Permanently add best center candidate found in local tries
            centers[c] = X[best_candidate_id]

        return centers 

    def _compute_softdtw_barycenter(self, X):
        """
        Compute the SoftDTW barycenter of a set of time series using PyTorch.

        Parameters
        ----------
        X : torch.Tensor of shape (n_samples, seq_len, n_features)
            Time series in the cluster.

        Returns
        -------
        Z : torch.Tensor of shape (seq_len, n_features)
            Barycenter of the cluster.
        """
        n_samples, seq_len, n_features = X.shape
        Z = X.mean(dim=0, keepdim=True).requires_grad_(True)  # Shape: (1, seq_len, n_features)

        optimizer = torch.optim.LBFGS([Z], max_iter=5, line_search_fn='strong_wolfe')

        def closure():
            optimizer.zero_grad()
            D = (X.unsqueeze(2) - Z.unsqueeze(1)).pow(2).sum(dim=3)  # Shape: (n_samples, seq_len, seq_len)
            loss = self.distance(D).sum() / n_samples
            loss.backward()
            return loss

        optimizer.step(closure)
        return Z.detach().squeeze(0)