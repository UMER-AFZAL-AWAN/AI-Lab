# =============================================================================
# 16_classical_ml / 02 — DECISION TREES AND RANDOM FORESTS
# =============================================================================
#
# LIKE YOU ARE 5:
# Decision tree = a game of 20 questions about the data.
#   "Is the animal bigger than a dog?"
#     Yes → "Does it have a trunk?" → Elephant / Rhino
#     No  → "Does it have feathers?" → Bird / Cat
# Each question (split) divides the data to reduce confusion.
#
# Random Forest = ask 100 different people to each play their own
# version of 20 questions on a random subset of the data.
# Then vote on the answer. The crowd wisdom beats any single person.
#
# C# ANALOGY:
# Decision tree = nested if/else chain decided by data, not you
# Random forest = run 100 parallel if/else chains, majority vote wins
#
# WHY THESE MATTER:
# - Fast to train, interpretable
# - Random forests are competitive with neural nets for tabular data
# - Gradient Boosted Trees (XGBoost, LightGBM) beat NNs on structured data
# - Decision trees explain what a neural net "learned" (interpretability)
#
# =============================================================================

import torch
import math
from collections import Counter

torch.manual_seed(42)

print("=" * 60)
print("PART 1: THE SPLIT CRITERION — GINI IMPURITY")
print("=" * 60)
print("""
When building a decision tree, at each node we choose the
feature and threshold that best SEPARATES the classes.

Measurement: Gini Impurity
  G = 1 - Σ p_i²   (where p_i = proportion of class i)

  G = 0.0: perfectly pure (all one class)  → great split
  G = 0.5: maximally impure (50/50 split)  → useless split

We pick the split that reduces Gini the most.
""")

def gini(labels):
    """Gini impurity of a list of class labels."""
    if len(labels) == 0:
        return 0.0
    counts = Counter(labels)
    n = len(labels)
    return 1.0 - sum((c/n)**2 for c in counts.values())

# Examples
all_one_class  = [0, 0, 0, 0, 0]
mixed          = [0, 0, 0, 1, 1]
perfectly_half = [0, 0, 0, 1, 1, 1]

print(f"All class 0: Gini = {gini(all_one_class):.4f}  (pure — perfect)")
print(f"3×0 + 2×1:   Gini = {gini(mixed):.4f}  (some impurity)")
print(f"3×0 + 3×1:   Gini = {gini(perfectly_half):.4f}  (worst case — 50/50)")

def information_gain(parent_labels, left_labels, right_labels):
    """How much does this split reduce impurity?"""
    n  = len(parent_labels)
    nl = len(left_labels)
    nr = len(right_labels)
    return gini(parent_labels) - (nl/n)*gini(left_labels) - (nr/n)*gini(right_labels)

# Evaluate two candidate splits
labels = [0, 0, 0, 0, 1, 1, 1, 1]

split_good_left  = [0, 0, 0, 0]
split_good_right = [1, 1, 1, 1]

split_bad_left   = [0, 0, 1, 1]
split_bad_right  = [0, 0, 1, 1]

print()
print(f"Good split IG = {information_gain(labels, split_good_left, split_good_right):.4f} (high — separates classes)")
print(f"Bad  split IG = {information_gain(labels, split_bad_left,  split_bad_right ):.4f} (zero — useless split)")

print()
print("=" * 60)
print("PART 2: BUILDING A DECISION TREE FROM SCRATCH")
print("=" * 60)

class DecisionNode:
    """One node in the decision tree."""
    def __init__(self):
        self.feature_idx = None   # which feature to split on
        self.threshold   = None   # split value: left if x[feat] <= threshold
        self.left        = None   # left child node
        self.right       = None   # right child node
        self.prediction  = None   # only set for leaf nodes

def find_best_split(X, y):
    """Find the feature and threshold that give the best Gini gain."""
    best_gain    = -1
    best_feature = 0
    best_thresh  = 0.0

    n_features = X.shape[1]
    for feat in range(n_features):
        vals = sorted(X[:, feat].tolist())
        thresholds = [(vals[i] + vals[i+1]) / 2 for i in range(len(vals)-1)]
        for thresh in thresholds:
            left_mask  = X[:, feat] <= thresh
            right_mask = ~left_mask
            if left_mask.sum() == 0 or right_mask.sum() == 0:
                continue
            gain = information_gain(
                y.tolist(),
                y[left_mask].tolist(),
                y[right_mask].tolist()
            )
            if gain > best_gain:
                best_gain, best_feature, best_thresh = gain, feat, thresh

    return best_feature, best_thresh

def build_tree(X, y, depth=0, max_depth=3):
    """Recursively build a decision tree."""
    node = DecisionNode()

    # Stop: leaf conditions
    if depth >= max_depth or len(set(y.tolist())) == 1:
        node.prediction = Counter(y.tolist()).most_common(1)[0][0]
        return node

    # Find the best split
    feat, thresh = find_best_split(X, y)
    left_mask    = X[:, feat] <= thresh
    right_mask   = ~left_mask

    if left_mask.sum() == 0 or right_mask.sum() == 0:
        node.prediction = Counter(y.tolist()).most_common(1)[0][0]
        return node

    node.feature_idx = feat
    node.threshold   = thresh
    node.left        = build_tree(X[left_mask],  y[left_mask],  depth+1, max_depth)
    node.right       = build_tree(X[right_mask], y[right_mask], depth+1, max_depth)
    return node

def predict_one(node, x):
    """Walk the tree for a single sample."""
    if node.prediction is not None:
        return node.prediction
    if x[node.feature_idx] <= node.threshold:
        return predict_one(node.left, x)
    else:
        return predict_one(node.right, x)

# Generate XOR-like dataset (can't be solved by a single linear split)
X_xor = torch.tensor([
    [-1, -1], [-1,  1], [1, -1], [1,  1],
    [-2, -2], [-2,  2], [2, -2], [2,  2],
], dtype=torch.float32)
y_xor = torch.tensor([0, 1, 1, 0, 0, 1, 1, 0])

tree = build_tree(X_xor, y_xor, max_depth=3)
preds = torch.tensor([predict_one(tree, X_xor[i]) for i in range(len(X_xor))])
acc   = (preds == y_xor).float().mean().item()
print(f"Decision tree on XOR data: accuracy = {acc*100:.0f}%  (expects 100%)")

print()
print("=" * 60)
print("PART 3: RANDOM FOREST — ENSEMBLE OF TREES")
print("=" * 60)
print("""
Decision tree problems:
  1. Overfits: a deep enough tree memorises training data
  2. Unstable: small data changes → completely different tree

Random Forest fixes both with two tricks:
  1. Bootstrap sampling: each tree trains on a random 60-80% subset of data
     → each tree sees different data → trees make different errors
  2. Feature randomness: each split considers only √(n_features) features
     → trees are less correlated → errors cancel out

Prediction: majority vote across all trees (classification)
            average across all trees (regression)
""")

class RandomForest:
    """Minimal random forest (uses our DecisionTree above)."""
    def __init__(self, n_trees=10, max_depth=3, sample_ratio=0.7):
        self.n_trees     = n_trees
        self.max_depth   = max_depth
        self.sample_ratio = sample_ratio
        self.trees       = []

    def fit(self, X, y):
        self.trees = []
        n = len(X)
        for _ in range(self.n_trees):
            # Bootstrap: sample with replacement
            idx  = torch.randint(0, n, (int(n * self.sample_ratio),))
            Xb   = X[idx]
            yb   = y[idx]
            tree = build_tree(Xb, yb, max_depth=self.max_depth)
            self.trees.append(tree)

    def predict(self, X):
        all_preds = torch.stack([
            torch.tensor([predict_one(tree, X[i]) for i in range(len(X))])
            for tree in self.trees
        ])  # shape: (n_trees, n_samples)
        # Majority vote
        return torch.tensor([
            Counter(all_preds[:, i].tolist()).most_common(1)[0][0]
            for i in range(len(X))
        ])

# Test on a bigger classification problem
n = 200
X_rf = torch.randn(n, 4)
y_rf = ((X_rf[:, 0] > 0) & (X_rf[:, 1] > 0)).long() | \
       ((X_rf[:, 0] < 0) & (X_rf[:, 1] < 0)).long()

rf = RandomForest(n_trees=10, max_depth=4)
rf.fit(X_rf, y_rf)
preds_rf = rf.predict(X_rf)
acc_rf = (preds_rf == y_rf).float().mean().item()
print(f"Random Forest (10 trees) accuracy: {acc_rf*100:.1f}%")

single_tree = build_tree(X_rf, y_rf, max_depth=4)
preds_1 = torch.tensor([predict_one(single_tree, X_rf[i]) for i in range(n)])
acc_1 = (preds_1 == y_rf).float().mean().item()
print(f"Single decision tree accuracy:    {acc_1*100:.1f}%  (usually lower)")
print()
print("Random forest beats single tree because:")
print("  diversity of trees → errors cancel out → lower variance")

print()
print("=" * 60)
print("PART 4: THE REAL TOOLS — SCIKIT-LEARN")
print("=" * 60)
print("""
In practice, use scikit-learn (not our hand-rolled version):

  from sklearn.tree import DecisionTreeClassifier
  from sklearn.ensemble import RandomForestClassifier
  import xgboost as xgb   # gradient boosted trees — often best on tabular

  # Train
  clf = RandomForestClassifier(n_estimators=100, max_depth=10)
  clf.fit(X_train, y_train)

  # Predict
  predictions = clf.predict(X_test)
  probs       = clf.predict_proba(X_test)   # (N, n_classes)

  # Feature importance (which features matter most)
  print(clf.feature_importances_)

When to use decision trees / random forests vs neural networks:
  Tabular data, structured data  → Random Forest / XGBoost often wins
  Images, text, audio, video     → Neural networks always win
  Interpretability needed        → Decision trees
  Leaderboard on Kaggle tabular  → XGBoost / LightGBM
""")

print("=" * 60)
print("GRADIENT BOOSTING — the other ensemble method")
print("=" * 60)
print("""
Random Forest: trains trees IN PARALLEL, votes at the end.
Gradient Boosting: trains trees SEQUENTIALLY.
  Each new tree corrects the mistakes of all previous trees.

  model_1 predicts badly → error_1 = y - pred_1
  model_2 predicts error_1 → now prediction = pred_1 + pred_2
  model_3 predicts remaining error → etc.

  Final: prediction = sum of all tree outputs × learning_rate

XGBoost / LightGBM are optimised gradient boosting.
They regularly beat neural networks on structured data (Kaggle competitions).
""")

# =============================================================================
# EXPECTED OUTPUT:
# Gini pure = 0.0, mixed = 0.48 range, 50/50 = 0.5
# Decision tree XOR accuracy: 100%
# Random forest accuracy: typically > single tree
# =============================================================================
