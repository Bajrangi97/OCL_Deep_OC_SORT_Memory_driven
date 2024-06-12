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

    # If target ID is present in the frame
    if self.target_id is not None:
        target_found = False
        for i, det_emb in enumerate(dets_embs):
            # Calculate distance between target embedding and detection embeddings
            distance = np.linalg.norm(det_emb - self.memory[self.target_id][-1][1])
            # If distance is below a threshold, assign the target ID to the detection
            if distance < 0.4:  # Adjust threshold as needed
                self.target_id = i
                target_found = True
                break
        if not target_found:
            print("Target missing")

    # If target ID is not present in the frame
    else:
        target_reidentified = False
        for i, det_emb in enumerate(dets_embs):
            # Check if any ID switching happens in the frames
            for mem_id, mem_data in self.memory.items():
                distance = np.linalg.norm(det_emb - mem_data[-1][1])
                if distance < 0.4:  # Adjust threshold as needed
                    # Reidentify target and print message
                    self.target_id = mem_id
                    target_reidentified = True
                    print("Target reidentified")
                    break
            if target_reidentified:
                break

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
    for m in matched:
        self.trackers[m[1]].update(dets[m[0], :])
        self.trackers[m[1]].update_emb(dets_embs[m[0]], alpha=dets_alpha[m[0]])
        
        # Store patches and embeddings of the target person
        if self.target_id is not None and m[1] == self.target_id:
            # Short-term memory update
            if m[1] in self.memory.keys():
                # Check if the difference between previous and current embeddings is greater than threshold
                if self.prev_embedding is None or np.linalg.norm(self.prev_embedding - dets_embs[m[0]]) > 0.4: # threshold=0.25
                    self.memory[m[1]].append((dets[m[0], :], dets_embs[m[0]]))
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

def set_target_id(self, target_id):
    """
    Set the ID of the target person.
    """
    self.target_id = target_id

