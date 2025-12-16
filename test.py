"""Small demo for pytorch-metric-learning miners and samplers.

Demonstrates:
 - TripletMarginMiner: mining triplets from embeddings and labels
 - MPerClassSampler: producing batches with M samples per class
 - FixedSetOfTriplets: sampler that yields previously mined triplets

Run this file with a Python env that has `torch` and `pytorch-metric-learning` installed.
"""

import torch
from torch.utils.data import Dataset, DataLoader

from pytorch_metric_learning.miners import TripletMarginMiner
from pytorch_metric_learning.samplers import MPerClassSampler, FixedSetOfTriplets


class DummyDataset(Dataset):
	def __init__(self, embeddings, labels):
		self.embeddings = embeddings
		self.labels = labels

	def __len__(self):
		return len(self.labels)

	def __getitem__(self, idx):
		# Return a tuple like an image and label; here we just return the embedding tensor
		return self.embeddings[idx], int(self.labels[idx])


def _generate_triplets_from_labels(labels, num_triplets, seed: int | None = None):
	"""Generate a list of (a,p,n) triplets from labels when miner found none.

	This picks random anchors, positives (different index same label), and negatives
	(index with different label) until `num_triplets` are collected or attempts
	are exhausted.
	"""
	import random

	if seed is not None:
		random.seed(seed)

	if isinstance(labels, torch.Tensor):
		labels_list = labels.tolist()
	else:
		labels_list = list(labels)

	label_to_indices = {}
	for idx, lab in enumerate(labels_list):
		label_to_indices.setdefault(lab, []).append(idx)

	classes = list(label_to_indices.keys())
	if len(classes) < 2:
		return []

	triplets = []
	N = len(labels_list)
	attempts = 0
	max_attempts = max(1000, num_triplets * 20)
	while len(triplets) < num_triplets and attempts < max_attempts:
		a = random.randrange(N)
		pos_candidates = label_to_indices[labels_list[a]]
		if len(pos_candidates) < 2:
			attempts += 1
			continue
		p = a
		while p == a:
			p = random.choice(pos_candidates)
		neg_class = random.choice([c for c in classes if c != labels_list[a]])
		n = random.choice(label_to_indices[neg_class])
		triplets.append((a, p, n))
		attempts += 1

	return triplets


def demo():
	# Create synthetic embeddings and labels
	torch.manual_seed(0)
	num_classes = 5
	samples_per_class = 6
	dim = 16
	print(f"num_classes: {num_classes}, samples_per_class: {samples_per_class}, dim: {dim}")
	embeddings = []
	labels = []
	for c in range(num_classes):
		center = torch.randn(dim) * 5.0
		for i in range(samples_per_class):
			embeddings.append(center + 0.5 * torch.randn(dim))
			labels.append(c)

	embeddings = torch.stack(embeddings)  # shape (N, dim)
	labels = torch.tensor(labels)

	print("Embeddings shape:", embeddings.shape)
	print("Labels shape:", labels.shape)

	# 1) Use TripletMarginMiner to mine triplets from embeddings
	# TripletMarginMiner supports different distance metrics via the `distance` parameter:
	#   - LpDistance(p=2): Euclidean distance (default)
	#   - LpDistance(p=1): Manhattan distance
	#   - CosineSimilarity(): Cosine similarity/distance
	
	# Example with Euclidean distance (default)
	miner = TripletMarginMiner(margin=0.9, type_of_triplets="all")
	
	# Alternative: use CosineSimilarity
	# from pytorch_metric_learning.distances import CosineSimilarity
	# miner = TripletMarginMiner(margin=0.5, type_of_triplets="all", distance=CosineSimilarity())
	
	# Alternative: use Manhattan distance
	# from pytorch_metric_learning.distances import LpDistance
	# miner = TripletMarginMiner(margin=0.9, type_of_triplets="all", distance=LpDistance(p=1))
	
	# TripletMarginMiner expects embeddings and labels, returns indices (a, p, n)
	a_idx, p_idx, n_idx = miner(embeddings, labels)
	print(f"Mined {a_idx.numel()} triplets")
	# Show first 5 mined triplets as (label_a, label_p, label_n)
	for i in range(min(5, a_idx.numel())):
		print((labels[a_idx[i]].item(), labels[p_idx[i]].item(), labels[n_idx[i]].item()))

	# 2) MPerClassSampler: create batches with M samples per class
	dataset = DummyDataset(embeddings, labels)
	m = 3  # samples per class per batch
	sampler = MPerClassSampler(labels, m=m, length_before_new_iter=100)
	loader = DataLoader(dataset, batch_size=m * 4, sampler=sampler, num_workers=0)

	print("\nUsing MPerClassSampler: show one batch of labels")
	for batch in loader:
		batch_embeddings, batch_labels = batch
		print(batch_labels)
		break

	# 3) FixedSetOfTriplets: try the API form that takes `labels` and `num_triplets`.
	# Create triplets as a list of tuples (a, p, n) from the miner results for use as a fallback.
	triplets = list(zip(a_idx.tolist(), p_idx.tolist(), n_idx.tolist()))

	# If miner returned no triplets, generate a small set from labels so we can
	# still demonstrate FixedSetOfTriplets(labels, num_triplets).
	if len(triplets) == 0:
		print("Miner produced 0 triplets — generating fallback triplets from labels")
		# generate a reasonable number of triplets
		fallback_num = max(32, num_classes * 4)
		triplets = _generate_triplets_from_labels(labels, fallback_num, seed=0)

	print("\nUsing FixedSetOfTriplets: attempting `FixedSetOfTriplets(labels, num_triplets)` usage")
	try:
		fixed_sampler = FixedSetOfTriplets(labels, len(triplets))

		# Try to use the sampler with a simple index dataset. Different pml versions
		# may return either (a,p,n) batches or indices into the range(len(labels)).
		dl = DataLoader(list(range(len(labels))), batch_size=32, sampler=fixed_sampler)
		for batch in dl:
			# batch can be a tensor of shape (B,3) or a 1D tensor of indices
			batch = batch.numpy()
			if batch.ndim == 2 and batch.shape[1] == 3:
				selected = [tuple(map(int, b)) for b in batch]
			else:
				# treat as indices into the `triplets` list
				selected = [triplets[int(i)] for i in batch.tolist()]

			print("Triplet batch size:", len(selected))
			for t in selected[:4]:
				a, p, n = t
				print((labels[a].item(), labels[p].item(), labels[n].item()))
			break
	except Exception as e:
		print("FixedSetOfTriplets(labels, num_triplets) failed with:", e)
		print("Falling back to FixedSetOfTriplets(triplets, batch_size=32)")
		fixed_sampler = FixedSetOfTriplets(triplets, batch_size=32)
		for triplet_batch in DataLoader(list(range(len(triplets))), batch_size=32, sampler=fixed_sampler):
			selected = [triplets[i] for i in triplet_batch]
			print("Triplet batch size:", len(selected))
			for t in selected[:4]:
				a, p, n = t
				print((labels[a].item(), labels[p].item(), labels[n].item()))
			break


if __name__ == "__main__":
	demo()

