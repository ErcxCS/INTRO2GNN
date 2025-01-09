import numpy as np
from utils import *
from concurrent.futures import ThreadPoolExecutor
from torch_geometric.data import Data

class NBP:

    def __init__(self, X_true: np.ndarray, n_anchors: int, n_iter: int):
        self.X_true = X_true
        self.n_nodes = self.X_true.shape[0]
        self.n_anchors = int(n_anchors)
        self.n_iter = n_iter
        self.n_batches = 4

    def iterative_NBP(
            self,
            graph: Data
    ):
        
        
        messages = graph.messages
        weights = graph.weights
        all_particles = graph.particles
        self.n_samples = all_particles.shape[1]
        self.C = graph.B
        self.D = graph.D

        self.anchor_list = list(range(self.n_anchors))
        self.radius = graph.radius.numpy()

        self._rmse = []
        for iter in range(self.n_iter):
            messages, weights, all_particles = self.NBP_iteration(all_messages=messages, all_weights=weights, all_particles=all_particles)
            estimates = np.einsum('ijk,ij->ik', all_particles[self.n_anchors:], weights[self.n_anchors:])
            
            t_rmse, _ = RMSE(self.X_true[self.n_anchors:], estimates)
            self._rmse.append(t_rmse)

        return self._rmse, estimates
    
    def NBP_iteration(self, all_messages: np.ndarray, all_weights: np.ndarray, all_particles: np.ndarray):
            estimates = np.einsum('ijk,ij->ik', all_particles[self.n_anchors:], all_weights[self.n_anchors:])
            messages_ru = dict()
            sampled_particles = [[] for _ in range(self.n_nodes)]
            batches_remaining = np.array([self.n_batches * self.n_samples for _ in range(self.n_nodes)])
            neighbor_count = np.count_nonzero(self.C, axis=1)

            def message_approximation(node_r):
                nonlocal batches_remaining, neighbor_count
                particles_r = all_particles[node_r]
                for node_u in range(self.n_nodes):
                    if node_u in self.anchor_list or node_r == node_u or self.D[node_r, node_u] == 0 or self.C[node_r, node_u] != 1:
                        continue

                    if len(self._rmse) == 0:
                        d_xy, W_xy = random_spread(particles_r=particles_r, d_ru=self.D[node_r, node_u])
                    else:
                        particles_u = all_particles[node_u]
                        d_xy, W_xy = relative_spread(particles_u=particles_u, particles_r=particles_r, d_ru=self.D[node_r, node_u])

                    X_ru = particles_r + d_xy
                    difference_sq = np.sum((X_ru - estimates[node_u - self.n_anchors]) ** 2, axis=1)
                    detection_probabilities = np.exp(-(difference_sq / (2 * self.radius ** 2)))
                    W_ru = detection_probabilities * (all_weights[node_r] / all_messages[node_r, node_u]) * (1/W_xy)
                    W_ru /= W_ru.sum()
                    
                    proposal_ru = gaussian_kde(dataset=X_ru.T, weights=W_ru, bw_method='silverman')
                    messages_ru[node_r, node_u] = proposal_ru

                    n_particles = batches_remaining[node_u] // neighbor_count[node_u]
                    batches_remaining[node_u] -= n_particles
                    neighbor_count[node_u] -= 1

                    particles = proposal_ru.resample(n_particles).T
                    sampled_particles[node_u].append(particles)

            with ThreadPoolExecutor() as executor:
                executor.map(message_approximation, range(self.n_nodes))

            for node in range(self.n_nodes):
                if sampled_particles[node]:
                    sampled_particles[node] = np.concatenate(sampled_particles[node])

            temp_all_particles = all_particles.copy()
            temp_all_weights = all_weights.copy()

            def belief_update(node_u):
                if node_u in self.anchor_list:
                    return
                
                particles_u = sampled_particles[node_u]
                incoming_message_u = dict()
                one_hop_messages = []
                all_messages_u = []
                
                for node_r in range(self.n_nodes):
                    if self.D[node_u, node_r] != 0:
                        if self.C[node_u, node_r] == 1:
                            message_ru = messages_ru[node_r, node_u](particles_u.T)
                            one_hop_messages.append(message_ru)
                        else:
                            difference_sq = np.sum((particles_u[:, None, :] - temp_all_particles[node_r]) ** 2, axis=2)
                            detection_probabilities = np.exp(-(difference_sq / (2 * self.radius ** 2)))
                            received_message_r = 1 - np.sum(temp_all_weights[node_r] * detection_probabilities, axis=1)
                            message_ru = received_message_r
                            
                        all_messages_u.append(message_ru)
                        incoming_message_u[node_r] = message_ru
                       
                proposal_product = np.prod(all_messages_u, axis=0)
                proposal_sum = np.sum(one_hop_messages, axis=0)

                W_u = proposal_product / proposal_sum
                W_u /= W_u.sum()
                
                indexes = np.random.choice(np.arange(W_u.size), size=self.n_samples, replace=True, p=W_u)
                
                all_particles[node_u] = particles_u[indexes]
                all_weights[node_u] = W_u[indexes]
                all_weights[node_u] /= all_weights[node_u].sum()

                for neighbor, message in incoming_message_u.items():
                    all_messages[node_u, neighbor] = message[indexes]
                
            with ThreadPoolExecutor() as executor:
                executor.map(belief_update, range(self.n_nodes))
            
            return all_messages, all_weights, all_particles