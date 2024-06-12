class OCSort(object):
    def __init__(
        self,
        det_thresh,
        max_age=30,
        min_hits=5, #3
        giou_threshold=0.2, #0.3
        delta_t=3,
        asso_func="giou",
        inertia=0.15,  #0.2
        w_association_emb=0.5, #.75
        alpha_fixed_emb=0.98, #0.95
        aw_param=0.5, # 0.5
        embedding_off=False,
        cmc_off=False,
        aw_off=False,
        new_kf_off=False,
        grid_off=False,
        **kwargs,
    ):
        """
        Sets key parameters for SORT
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.giou_threshold = giou_threshold
        self.trackers = []
        self.memory = {}
        self.long_memory = {}
        self.memory_capacity = 3  # Capacity for short-term memory 25
        self.long_memory_capacity = 20  # Capacity for long-term memory 64
        self.frame_count = 0
        self.det_thresh = det_thresh
        self.delta_t = delta_t
        self.asso_func = ASSO_FUNCS[asso_func]
        self.inertia = inertia
        self.w_association_emb = w_association_emb
        self.alpha_fixed_emb = alpha_fixed_emb
        self.aw_param = aw_param
        KalmanBoxTracker.count = 0

        self.embedder = EmbeddingComputer(kwargs["args"].dataset, kwargs["args"].test_dataset, grid_off)
        self.cmc = CMCComputer()
        self.embedding_off = embedding_off
        self.cmc_off = cmc_off
        self.aw_off = aw_off
        self.new_kf_off = new_kf_off
        self.grid_off = grid_off

        self.prev_embedding = None

    def update(self, output_results, img_tensor, img_numpy, tag):
        """
        Params:
          dets - a numpy array of detections in the format [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]
        Requires: this method must be called once for each frame even with empty detections (use np.empty((0, 5)) for frames without detections).
        Returns the a similar array, where the last column is the object ID.
        NOTE: The number of objects returned may differ from the number of detections provided.
        """
        if output_results is None:
            return np.empty((0, 5))
        if not isinstance(output_results, np.ndarray):
            output_results = output_results.cpu().numpy()
        self.frame_count += 1
        if output_results.shape[1] == 5:
            scores = output_results[:, 4]
            bboxes = output_results[:, :4]
        else:
            output_results = output_results
            scores = output_results[:, 4] * output_results[:, 5]
            bboxes = output_results[:, :4]  # x1y1x2y2
        dets = np.concatenate((bboxes, np.expand_dims(scores, axis=-1)), axis=1)
        remain_inds = scores > self.det_thresh
        dets = dets[remain_inds]

        # Rescale
        scale = min(img_tensor.shape[2] / img_numpy.shape[0], img_tensor.shape[3] / img_numpy.shape[1])
        dets[:, :4] /= scale

        # Generate embeddings
        dets_embs = np.ones((dets.shape[0], 1))
        if not self.embedding_off and dets.shape[0] != 0:
            # Shape = (num detections, 3, 512) if grid
            dets_embs = self.embedder.compute_embedding(img_numpy, dets[:, :4], tag)

        # CMC
        if not self.cmc_off:
            transform = self.cmc.compute_affine(img_numpy, dets[:, :4], tag)
            for trk in self.trackers:
                trk.apply_affine_correction(transform)

        trust = (dets[:, 4] - self.det_thresh) / (1 - self.det_thresh)
        af = self.alpha_fixed_emb
        # From [self.alpha_fixed_emb, 1], goes to 1 as detector is less confident
        dets_alpha = af + (1 - af) * (1 - trust)

        # get predicted locations from existing trackers.
        trks = np.zeros((len(self.trackers), 5))
        trk_embs = []
        to_del = []
        ret = []
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()[0]
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)
            else:
                trk_embs.append(self.trackers[t].get_emb())
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        # Shape = (num_trackers, 3, 512) if grid
        trk_embs = np.array(trk_embs)
        for t in reversed(to_del):
            self.trackers.pop(t)

        velocities = np.array([trk.velocity if trk.velocity is not None else np.array((0, 0)) for trk in self.trackers])
        last_boxes = np.array([trk.last_observation for trk in self.trackers])
        k_observations = np.array([k_previous_obs(trk.observations, trk.age, self.delta_t) for trk in self.trackers])

        """
            First round of association
        """
        matched, unmatched_dets, unmatched_trks = associate(
            dets,
            trks,
            dets_embs,
            trk_embs,
            self.memory,
            self.long_memory,
            self.giou_threshold,
            velocities,
            k_observations,
            self.inertia,
            self.w_association_emb,
            self.aw_off,
            self.aw_param,
            self.embedding_off,
            self.grid_off,
        )

        # Handling ID switches and new objects
        for m in matched:
            # If matched, update the tracker with the corresponding detection
            self.trackers[m[1]].update(dets[m[0], :])
            self.trackers[m[1]].update_emb(dets_embs[m[0]], alpha=dets_alpha[m[0]])
            # Handle ID switch or update existing memory
            if m[1] in self.memory.keys():
                # Check if the difference between previous and current embeddings is greater than threshold
                if self.prev_embedding is None or np.linalg.norm(self.prev_embedding - dets_embs[m[0]]) > 0.4:  # threshold=0.25
                    self.memory[m[1]].append(dets_embs[m[0]])
                    # Ensure short-term memory capacity is maintained
                    if len(self.memory[m[1]]) > self.memory_capacity:
                        # Move the oldest frame to long-term memory
                        oldest_frame = self.memory[m[1]].pop(0)
                        if m[1] in self.long_memory:
                            self.long_memory[m[1]].append(oldest_frame)
                            # Ensure long-term memory capacity is maintained
                            if len(self.long_memory[m[1]]) > self.long_memory_capacity:
                                # Remove frames with less information from long-term memory
                                self._remove_low_info_frames(self.long_memory[m[1]])
                        else:
                            self.long_memory[m[1]] = [oldest_frame]
            else:
                self.memory[m[1]] = [dets_embs[m[0]]]

            # Long-term memory update
            if m[1] in self.long_memory.keys():
                # Check if the difference between previous and current embeddings is greater than threshold
                if self.prev_embedding is None or np.linalg.norm(self.prev_embedding - dets_embs[m[0]]) > 0.4:  # threshold=0.25
                    self.long_memory[m[1]].append(dets_embs[m[0]])
                    # Ensure long-term memory capacity is maintained
                    if len(self.long_memory[m[1]]) > self.long_memory_capacity:
                        # Remove frames with less information from long-term memory
                        self._remove_low_info_frames(self.long_memory[m[1]])
            else:
                self.long_memory[m[1]] = [dets_embs[m[0]]]

        """
            Second round of association by OCR
        """
        if unmatched_dets.shape[0] > 0 and unmatched_trks.shape[0] > 0:
            left_dets = dets[unmatched_dets]
            left_dets_embs = dets_embs[unmatched_dets]
            left_trks = last_boxes[unmatched_trks]
            left_trks_embs = trk_embs[unmatched_trks]

            # TODO: maybe use embeddings here
            giou_left = self.asso_func(left_dets, left_trks)
            giou_left = np.array(giou_left)
            if giou_left.max() > self.giou_threshold:
                """
                NOTE: by using a lower threshold, e.g., self.iou_threshold - 0.1, you may
                get a higher performance especially on MOT17/MOT20 datasets. But we keep it
                uniform here for simplicity
                """
                rematched_indices = linear_assignment(-giou_left)

                to_remove_det_indices = []
                to_remove_trk_indices = []
                for m in rematched_indices:
                    det_ind, trk_ind = unmatched_dets[m[0]], unmatched_trks[m[1]]
                    if giou_left[m[0], m[1]] < self.giou_threshold:
                        continue
                    self.trackers[trk_ind].update(dets[det_ind, :])
                    self.trackers[trk_ind].update_emb(dets_embs[det_ind], alpha=dets_alpha[det_ind])
                    to_remove_det_indices.append(det_ind)
                    to_remove_trk_indices.append(trk_ind)
                unmatched_dets = np.setdiff1d(unmatched_dets, np.array(to_remove_det_indices))
                unmatched_trks = np.setdiff1d(unmatched_trks, np.array(to_remove_trk_indices))

        for m in unmatched_trks:
            self.trackers[m].update(None)

        # create and initialise new trackers for unmatched detections
        for i in unmatched_dets:
            trk = KalmanBoxTracker(
                dets[i, :], delta_t=self.delta_t, emb=dets_embs[i], alpha=dets_alpha[i], new_kf=not self.new_kf_off
            )
            self.trackers.append(trk)
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            if trk.last_observation.sum() < 0:
                d = trk.get_state()[0]
            else:
                """
                this is optional to use the recent observation or the kalman filter prediction,
                we didn't notice significant difference here
                """
                d = trk.last_observation[:4]
            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                # +1 as MOT benchmark requires positive
                ret.append(np.concatenate((d, [trk.id + 1])).reshape(1, -1))
            i -= 1
            # remove dead tracklet
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)
        if len(ret) > 0:
            return np.concatenate(ret)
        return np.empty((0, 5))

    def dump_cache(self):
        self.cmc.dump_cache()
        self.embedder.dump_cache()

    def _remove_low_info_frames(self, memory):
        """
        Removes frames from memory based on the amount of information they contain.
        """
        # Calculate information measure for each frame
        info_measures = []
        for emb in memory:
            # Calculate information measure for the embedding
            info_measures.append(self._calculate_info_measure(emb))
        # Sort frames by their information measures
        sorted_indices = np.argsort(info_measures)
        # Remove frames with the lowest information measures
        for idx in sorted_indices[:len(memory) - self.long_memory_capacity]:
            memory.pop(idx)

    def _calculate_info_measure(self, embedding):
        """
        Calculate an information measure for the given embedding.
        This can be any method that assesses the amount of information contained in the embedding.
        """
        # Example: calculate L2 norm of the embedding
        return np.linalg.norm(embedding)






####### new.........
# Handling ID switches and new objects
for m in matched:
    # If matched, update the tracker with the corresponding detection
    self.trackers[m[1]].update(dets[m[0], :])
    self.trackers[m[1]].update_emb(dets_embs[m[0]], alpha=dets_alpha[m[0]])
    # Handle ID switch or update existing memory
    if m[1] in self.memory.keys():
        # Check if the difference between previous and current embeddings is greater than threshold
        if self.prev_embedding is None or np.linalg.norm(self.prev_embedding - dets_embs[m[0]]) > 0.4:  # threshold=0.25
            self.memory[m[1]].append(dets_embs[m[0]])
            # Ensure short-term memory capacity is maintained
            if len(self.memory[m[1]]) > self.memory_capacity:
                # Move the oldest frame to long-term memory
                oldest_frame = self.memory[m[1]].pop(0)
                if m[1] in self.long_memory:
                    self.long_memory[m[1]].append(oldest_frame)
                    # Ensure long-term memory capacity is maintained
                    if len(self.long_memory[m[1]]) > self.long_memory_capacity:
                        # Remove frames with less information from long-term memory
                        self._remove_low_info_frames(self.long_memory[m[1]])
                else:
                    self.long_memory[m[1]] = [oldest_frame]
    else:
        self.memory[m[1]] = [dets_embs[m[0]]]

    # Long-term memory update
    if m[1] in self.long_memory.keys():
        # Check if the difference between previous and current embeddings is greater than threshold
        if self.prev_embedding is None or np.linalg.norm(self.prev_embedding - dets_embs[m[0]]) > 0.4:  # threshold=0.25
            self.long_memory[m[1]].append(dets_embs[m[0]])
            # Ensure long-term memory capacity is maintained
            if len(self.long_memory[m[1]]) > self.long_memory_capacity:
                # Remove frames with less information from long-term memory
                self._remove_low_info_frames(self.long_memory[m[1]])
    else:
        self.long_memory[m[1]] = [dets_embs[m[0]]]

# Second round of association by OCR
if unmatched_dets.shape[0] > 0 and unmatched_trks.shape[0] > 0:
    left_dets = dets[unmatched_dets]
    left_dets_embs = dets_embs[unmatched_dets]
    left_trks = last_boxes[unmatched_trks]
    left_trks_embs = trk_embs[unmatched_trks]

    # Check for objects that were previously assigned new IDs
    for i, trk in enumerate(self.trackers):
        if trk.id < 0:  # Check if the tracker was assigned a new ID
            # Calculate the embedding of the tracker
            trk_emb = trk.get_emb()
            # Compare the embedding with unmatched detections
            dists = np.linalg.norm(left_dets_embs - trk_emb, axis=1)
            min_dist_idx = np.argmin(dists)
            min_dist = dists[min_dist_idx]
            # If the minimum distance is below a certain threshold, assign the previous ID
            if min_dist < threshold:
                trk.id = trk.prev_id
                # Update the memory with the new embedding
                self.memory[trk.id].append(left_dets_embs[min_dist_idx])
                # Remove the assigned ID from unmatched detections
                left_dets = np.delete(left_dets, min_dist_idx, axis=0)
                left_dets_embs = np.delete(left_dets_embs, min_dist_idx, axis=0)

    # Create and initialise new trackers for unmatched detections
    for i in range(left_dets.shape[0]):
        trk = KalmanBoxTracker(
            left_dets[i, :], delta_t=self.delta_t, emb=left_dets_embs[i], alpha=dets_alpha[i], new_kf=not self.new_kf_off
        )
        self.trackers.append(trk)
        # Add new IDs to memory
        self.memory[trk.id] = [left_dets_embs[i]]

# Construct the return array
ret = []
for trk in reversed(self.trackers):
    if trk.last_observation.sum() < 0:
        d = trk.get_state()[0]
    else:
        """
        This is optional to use the recent observation or the Kalman filter prediction,
        we didn't notice significant difference here
        """
        d = trk.last_observation[:4]
    if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
        # +1 as MOT benchmark requires positive
        ret.append(np.concatenate((d, [trk.id + 1])).reshape(1, -1))
    # Remove dead tracklets
    if trk.time_since_update > self.max_age:
        self.trackers.remove(trk)

# Return the constructed array
if len(ret) > 0:
    return np.concatenate(ret)
return np.empty((0, 5))


"""
# use this function while calculation of the mutual information
from sklearn.metrics import mutual_info_score

class EmbeddingMutualInformationCalculator:
    def calculate_mutual_information(self, current_embeddings, previous_embeddings):
        """
        #Calculate the mutual information between the current embeddings and the previous embeddings.
        """
        # Compute the mutual information using the scikit-learn library
        mutual_info = mutual_info_score(current_embeddings.flatten(), previous_embeddings.flatten())
        return mutual_info




# Use This one
import numpy as np
from scipy.stats import entropy

class EmbeddingMutualInformationCalculator:
    def calculate_mutual_information(self, current_embeddings, previous_embeddings):
        """
        #Calculate mutual information between the current embeddings and previous saved embeddings.
        """
        # Calculate the joint distribution of the current and previous embeddings
        joint_distribution = self.compute_joint_distribution(current_embeddings, previous_embeddings)
        
        # Calculate the marginal distributions of the current and previous embeddings
        marginal_current = self.compute_marginal_distribution(current_embeddings)
        marginal_previous = self.compute_marginal_distribution(previous_embeddings)
        
        # Calculate the mutual information using Shannon entropy and Kullback-Leibler divergence
        mutual_information = entropy(joint_distribution.flatten()) - entropy(marginal_current.flatten()) - entropy(marginal_previous.flatten())
        
        return mutual_information
    
    def compute_joint_distribution(self, current_embeddings, previous_embeddings):
        """
        #Compute the joint distribution of the current and previous embeddings.
        """
        # Combine the current and previous embeddings into a single array
        combined_embeddings = np.concatenate((current_embeddings, previous_embeddings), axis=1)
        
        # Calculate the joint distribution
        joint_distribution, _ = np.histogramdd(combined_embeddings.T, bins=10)  # Adjust the number of bins as needed
        
        # Normalize the joint distribution
        joint_distribution /= np.sum(joint_distribution)
        
        return joint_distribution
    
    def compute_marginal_distribution(self, embeddings):
        """
        #Compute the marginal distribution of the embeddings.
        """
        # Calculate the marginal distribution
        marginal_distribution, _ = np.histogramdd(embeddings.T, bins=10)  # Adjust the number of bins as needed
        
        # Normalize the marginal distribution
        marginal_distribution /= np.sum(marginal_distribution)
        
        return marginal_distribution

# latex code for this
\begin{equation}
\text{Mutual Information} = H(X,Y) - H(X) - H(Y)
\end{equation}

Where:
\begin{itemize}
    \item \(H(X,Y)\) is the joint entropy of the current embeddings \(X\) and previous saved embeddings \(Y\).
    \item \(H(X)\) is the marginal entropy of the current embeddings \(X\).
    \item \(H(Y)\) is the marginal entropy of the previous saved embeddings \(Y\).
\end{itemize}

To calculate the joint entropy \(H(X,Y)\), we first compute the joint distribution \(P(X,Y)\) of \(X\) and \(Y\). Similarly, we calculate the marginal distributions \(P(X)\) and \(P(Y)\). Then, we use the following formulas:

\begin{equation}
H(X,Y) = -\sum_{i,j} P(X_i,Y_j) \log_2(P(X_i,Y_j))
\end{equation}

\begin{equation}
H(X) = -\sum_{i} P(X_i) \log_2(P(X_i))
\end{equation}

\begin{equation}
H(Y) = -\sum_{j} P(Y_j) \log_2(P(Y_j))
\end{equation}

Finally, the mutual information is computed by subtracting the marginal entropies from the joint entropy.

"""
# Normalized mutual information
"""
import numpy as np
from scipy.stats import entropy

class EmbeddingMutualInformationCalculator:
    def calculate_normalized_mutual_information(self, current_embeddings, previous_embeddings):
        """
        Calculate normalized mutual information between the current embeddings and previous saved embeddings.
        """
        # Calculate the mutual information
        mutual_information = self.calculate_mutual_information(current_embeddings, previous_embeddings)
        
        # Calculate the entropies of the current and previous embeddings
        entropy_current = entropy(current_embeddings.flatten())
        entropy_previous = entropy(previous_embeddings.flatten())
        
        # Calculate the normalized mutual information
        normalized_mutual_information = mutual_information / np.sqrt(entropy_current * entropy_previous)
        
        return normalized_mutual_information
    
    def calculate_mutual_information(self, current_embeddings, previous_embeddings):
        """
        Calculate mutual information between the current embeddings and previous saved embeddings.
        """
        # Calculate the joint distribution of the current and previous embeddings
        joint_distribution = self.compute_joint_distribution(current_embeddings, previous_embeddings)
        
        # Calculate the marginal distributions of the current and previous embeddings
        marginal_current = self.compute_marginal_distribution(current_embeddings)
        marginal_previous = self.compute_marginal_distribution(previous_embeddings)
        
        # Calculate the mutual information using Shannon entropy
        mutual_information = entropy(joint_distribution.flatten()) - entropy(marginal_current.flatten()) - entropy(marginal_previous.flatten())
        
        return mutual_information
    
    def compute_joint_distribution(self, current_embeddings, previous_embeddings):
        """
        Compute the joint distribution of the current and previous embeddings.
        """
        # Combine the current and previous embeddings into a single array
        combined_embeddings = np.concatenate((current_embeddings, previous_embeddings), axis=1)
        
        # Calculate the joint distribution
        joint_distribution, _ = np.histogramdd(combined_embeddings.T, bins=10)  # Adjust the number of bins as needed
        
        # Normalize the joint distribution
        joint_distribution /= np.sum(joint_distribution)
        
        return joint_distribution
    
    def compute_marginal_distribution(self, embeddings):
        """
        Compute the marginal distribution of the embeddings.
        """
        # Calculate the marginal distribution
        marginal_distribution, _ = np.histogramdd(embeddings.T, bins=10)  # Adjust the number of bins as needed
        
        # Normalize the marginal distribution
        marginal_distribution /= np.sum(marginal_distribution)
        
        return marginal_distribution

# Example usage
if __name__ == "__main__":
    # Instantiate the EmbeddingMutualInformationCalculator
    calculator = EmbeddingMutualInformationCalculator()
    
    # Example current embeddings and previous saved embeddings (replace with your actual embeddings)
    current_embeddings = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
    previous_embeddings = np.array([[0.2, 0.3], [0.4, 0.5], [0.6, 0.7]])
    
    # Calculate normalized mutual information
    normalized_mutual_information = calculator.calculate_normalized_mutual_information(current_embeddings, previous_embeddings)
    
    # Print the result
    print("Normalized Mutual Information between Current and Previous Embeddings:", normalized_mutual_information)



"""

# without scipy
import numpy as np

class EmbeddingMutualInformationCalculator:
    def calculate_normalized_mutual_information(self, current_embeddings, previous_embeddings):
        """
        Calculate normalized mutual information between the current embeddings and previous saved embeddings.
        """
        # Calculate the mutual information
        mutual_information = self.calculate_mutual_information(current_embeddings, previous_embeddings)
        
        # Calculate the entropies of the current and previous embeddings
        entropy_current = self.calculate_entropy(current_embeddings)
        entropy_previous = self.calculate_entropy(previous_embeddings)
        
        # Calculate the normalized mutual information
        normalized_mutual_information = mutual_information / np.sqrt(entropy_current * entropy_previous)
        
        # Take the absolute value to ensure positivity
        normalized_mutual_information = np.abs(normalized_mutual_information)
        
        return normalized_mutual_information
    
    def calculate_mutual_information(self, current_embeddings, previous_embeddings):
        """
        Calculate mutual information between the current embeddings and previous saved embeddings.
        """
        # Calculate the joint distribution of the current and previous embeddings
        joint_distribution = self.compute_joint_distribution(current_embeddings, previous_embeddings)
        
        # Calculate the marginal distributions of the current and previous embeddings
        marginal_current = self.compute_marginal_distribution(current_embeddings)
        marginal_previous = self.compute_marginal_distribution(previous_embeddings)
        
        # Calculate the mutual information using Shannon entropy
        mutual_information = self.calculate_entropy(joint_distribution.flatten()) - \
                              self.calculate_entropy(marginal_current.flatten()) - \
                              self.calculate_entropy(marginal_previous.flatten())
        
        return mutual_information
    
    def calculate_entropy(self, distribution):
        """
        Calculate the entropy of a probability distribution.
        """
        # Ensure distribution sums to 1
        distribution = distribution / np.sum(distribution)
        
        # Calculate entropy
        entropy = -np.sum(np.where(distribution > 0, distribution * np.log2(distribution), 0))
        
        return entropy
    
    def compute_joint_distribution(self, current_embeddings, previous_embeddings):
        """
        Compute the joint distribution of the current and previous embeddings.
        """
        # Combine the current and previous embeddings into a single array
        combined_embeddings = np.concatenate((current_embeddings, previous_embeddings), axis=1)
        
        # Calculate the joint distribution
        joint_distribution, _ = np.histogramdd(combined_embeddings.T, bins=10)  # Adjust the number of bins as needed
        
        # Normalize the joint distribution
        joint_distribution /= np.sum(joint_distribution)
        
        return joint_distribution
    
    def compute_marginal_distribution(self, embeddings):
        """
        Compute the marginal distribution of the embeddings.
        """
        # Calculate the marginal distribution
        marginal_distribution, _ = np.histogramdd(embeddings.T, bins=10)  # Adjust the number of bins as needed
        
        # Normalize the marginal distribution
        marginal_distribution /= np.sum(marginal_distribution)
        
        return marginal_distribution

# Example usage
if __name__ == "__main__":
    # Instantiate the EmbeddingMutualInformationCalculator
    calculator = EmbeddingMutualInformationCalculator()
    
    # Example current embeddings and previous saved embeddings (replace with your actual embeddings)
    current_embeddings = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
    previous_embeddings = np.array([[0.2, 0.3], [0.4, 0.5], [0.6, 0.7]])
    
    # Calculate normalized mutual information
    normalized_mutual_information = calculator.calculate_normalized_mutual_information(current_embeddings, previous_embeddings)
    
    # Print the result
    print("Normalized Mutual Information between Current and Previous Embeddings:", normalized_mutual_information)

