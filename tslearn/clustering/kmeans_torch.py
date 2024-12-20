import torch
import numpy as np

from SOFTDTW.soft_dtw_cuda import PairwiseSoftDTW, SoftDTW

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
            optimizer = 'lbfgs',
            optimizer_kwargs={'lr': 1.0}
        ):
        self.n_clusters = n_clusters
        self.max_iter = max_iter

        if random_state is None:
            self.random_state = np.random.RandomState()

        self.random_state = random_state
        self.tol = tol
        self.gamma = gamma
        self.device = device
        self.n_init = n_init
        self.max_iter_barycenter = 10
        self.barycenter = SoftDTW(use_cuda=True, gamma=self.gamma)
        self.distance = PairwiseSoftDTW(gamma=self.gamma)
        self.optimizer = optimizer
        self.optimizer_kwargs = optimizer_kwargs 

    def fit(self, X):
        """
        Compute k-means clustering.

        Parameters
        ----------
        X : array-like of shape (n_samples, seq_len, n_features)
            Training data.
        """
        # Convert data to torch tensor
        X = torch.tensor(X, dtype=torch.float32).to(self.device)

        best_inertia = float('inf')
        best_labels = None
        best_centers = None

        for _ in range(self.n_init):
            labels, inertia, centers = self._fit_one_init(X, rs=self.random_state)

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
        distances = self.distance(X, self.cluster_centers_)
        labels = torch.argmin(distances, dim=1)
        return labels.cpu().numpy()

    def _fit_one_init(self, X, rs=None):
        """
        Initialize cluster centers and perform clustering with one initialization.

        Parameters
        ----------
        X : torch.Tensor of shape (n_samples, seq_len, n_features)
            Training data.

        rs : numpy Randomstate 

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
        cluster_centers = self._k_means_init(X, rs=rs).clone().requires_grad_(True)

        for _ in range(self.max_iter):
            # Compute distances between each time series and each cluster center
            print(" memory allcoated before" , torch.cuda.memory_allocated())
            distances = self.distance(X, cluster_centers)
            print(" memory allocated after", torch.cuda.memory_allocated())
            # Assign each time series to the nearest cluster center
            labels = torch.argmin(distances, dim=1)

            # Compute inertia (sum of distances for assigned clusters)
            inertia = torch.sum(
                distances[torch.arange(n_samples), labels].pow(2)
            ) / n_samples

            # Update cluster centers
            new_centers = []
            for k in range(self.n_clusters):
                X_k = X[labels == k] # shape  = (t, d)
                init_center = cluster_centers[k]
                if X_k.nelement() == 0:
                    new_center = X[torch.randint(0, n_samples, (1,))].squeeze(0)
                else:
                    new_center = self._update_centroid(X_k, init_center=init_center)
                new_centers.append(new_center.detach())

            new_centers = torch.stack(new_centers)
            center_shift = torch.norm(cluster_centers - new_centers)

            # Convergence
            if center_shift < self.tol:
                break

            cluster_centers = new_centers.clone().detach().requires_grad_(True)

        return labels, inertia.item(), cluster_centers

    def _k_means_init(self, X, rs):
        """
        Initialize cluster centers using the k-means++ algorithm.

        Parameters
        ----------
        X : torch.Tensor of shape (n_samples, seq_len, n_features)
            Training data.

        rs: numpy.RandomState
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
        c_id = rs.randint(0, n)
        centers[0] = X[c_id]

        # Initialize list of squared distances to closest center
        closest_dist_sq = self.distance(centers[0].unsqueeze(0), X) ** 2
        current_pot = closest_dist_sq.sum().item()

        for c in range(1, n_clusters):
            # Generate rand_vals using NumPy's random_state
            rand_vals_np = rs.random_sample(n_local_trials) * current_pot

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
            print(f"kmeans init memory before", torch.cuda.mem_get_info())
            distance_to_candidates = self.distance(X[c_ids], X) ** 2
            print(f"kmeans init memory after", torch.cuda.mem_get_info())
            
            # Update closest distances squared and potential for each candidate
            # shape (3,)
            closest_dist_sq_candidate = torch.minimum(
                closest_dist_sq, 
                distance_to_candidates
            )

            candidates_pot = closest_dist_sq_candidate.sum(dim=1)

            # Decide which candidate is the best
            best_candidate = torch.argmin(candidates_pot)
            current_pot = candidates_pot[best_candidate].item()
            closest_dist_sq = closest_dist_sq_candidate[best_candidate].unsqueeze(0)
            best_candidate_id = c_ids[best_candidate]

            # Permanently add best center candidate found in local tries
            centers[c] = X[best_candidate_id]

        return centers 

    def _update_centroid(self, X_cluster, init_center):
        num_iters = 10
        lr = 0.1
        centroid = init_center.clone().detach().requires_grad_(True).to(self.device)
        if self.optimizer.lower() == 'lbfgs':
            optimizer = torch.optim.LBFGS(
                [centroid], 
                lr=self.optimizer_kwargs.get('lr', 1.0), 
                max_iter=self.max_iter_barycenter,
                line_search_fn="strong_wolfe"
            )

            def closure():
                optimizer.zero_grad()
                centroid_expanded = centroid.unsqueeze(0).expand(X_cluster.shape[0], -1, -1)
                sdtw_values = self.barycenter(centroid_expanded, X_cluster)
                loss = sdtw_values.mean()
                loss.backward()
                return loss
            optimizer.step(closure)
        else:
            optimizer = torch.optim.Adam([centroid], lr=lr)
            for _ in range(num_iters):
                optimizer.zero_grad()
                centroid_expanded = centroid.unsqueeze(0).expand(
                    X_cluster.shape[0], -1, -1
                )

                sdtw_values = self.barycenter(centroid_expanded, X_cluster)
                loss = sdtw_values.mean()
                loss.backward()
                optimizer.step()

        return centroid.data