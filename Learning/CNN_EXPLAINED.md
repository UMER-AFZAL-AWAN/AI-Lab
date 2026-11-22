# How a CNN Works — Complete Explanation
### Every keyword defined in brackets the moment it appears. No looking things up separately.

---

## Step 1: What is an image to a computer?

You look at a photo of a cat and see a cat.

The computer sees a grid of numbers.

A 6×6 pixel image is just a 6×6 table of numbers. Each number is a brightness
value from 0 (black) to 255 (white). A colour image has three of these grids
stacked — one for Red, one for Green, one for Blue. This stack of three grids
is called a **tensor** *(an N-dimensional grid of numbers — here it is 3D:
3 grids × height × width)*.

We describe its size as **shape** *(the dimensions of a tensor, written as a
tuple)*: **shape (3, 6, 6)** — meaning 3 **channels** *(one grid per colour:
Red=1, Green=1, Blue=1; after a conv layer, channels become filter outputs)*,
6 pixels tall, 6 pixels wide.

```
One channel (grayscale) of a 6×6 image:
  0   0   0   0   0   0
  0   0 200 200   0   0
  0 200 255 255 200   0
  0 200 255 255 200   0
  0   0 200 200   0   0
  0   0   0   0   0   0
```

This particular grid of numbers looks like a bright square in the centre.
That's a shape to you. To the computer it's just numbers.

---

## Step 2: What is a filter?

A **filter** *(also called a kernel — a small grid of numbers, usually 3×3,
that slides across an image asking "is my pattern here?")* is a tiny 3×3 grid
of numbers.

```
A filter that detects vertical edges:
 -1   0   1
 -1   0   1
 -1   0   1
```

It is a question: "Is there a vertical edge here?"

- Left column is -1 (negative): expects dark on the left
- Right column is +1 (positive): expects bright on the right
- That pattern = a vertical edge

The numbers in each filter are **weights** *(learnable numbers inside a layer
— what the model adjusts during training)*. You do not set them by hand.
The network **learns** *(adjusts weights to reduce mistakes)* them during
**training** *(the process of showing the model data and correcting it repeatedly)*.

---

## Step 3: Convolution — the filter slides across the image

This sliding operation is called a **convolution** *(sliding a filter across
an image; at each position multiply filter numbers by image numbers and sum
them — detects where the filter's pattern appears)*.

Place the 3×3 filter on the top-left corner of the image.
Multiply each filter number by the image number it sits on.
Add all 9 results together.
That sum becomes ONE number in the output.

Slide the filter one step right. Repeat.
Reach the right edge, drop down one row. Repeat.

```
Image patch (3×3):        Filter:          Multiply & sum:
  0   0   0              -1   0   1        (0×-1) + (0×0) + (0×1)
  0 200 255    ×         -1   0   1    =   (0×-1) + (200×0) + (255×1)
  0 200 255              -1   0   1        (0×-1) + (200×0) + (255×1)
                                           = 0 + 255 + 255 = 510  ← strong edge detected
```

Do this for every position → you get a new grid called a **feature map**
*(the output grid from one filter — a heatmap showing where the filter's
pattern was found in the image)*.

A HIGH number in the feature map = "YES, the pattern is strongly here."
A LOW number = "NO, not here."

This whole operation — filter + slide + sum — is a **Conv2d layer**
*(a PyTorch layer that runs one or more filters across a 2D image)*.

```python
# In PyTorch:
nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
# 3 input channels (RGB), 32 filters, each 3×3
```

---

## Step 4: Many filters running at once

One filter asks one question. A Conv2d layer has MANY filters in parallel.

After a layer with 32 filters, you have 32 feature maps stacked.
The shape becomes **(32, H, W)** — 32 channels, one per filter's output.

Each filter asks something different:
- Filter 1: Is there a vertical edge?
- Filter 2: Is there a horizontal edge?
- Filter 3: Is there a bright blob?
- Filter 4: Is there a dark circle?
- ... 28 more questions

These filters are not set by you. They are **learned** through
**backpropagation** *(the algorithm that works backwards from the mistake
to figure out how much each weight contributed — then adjusts them)*.

---

## Step 5: ReLU — throwing away negatives

After the convolution, we apply an **activation function** *(a rule applied to
each number in the output — without it, stacking layers collapses into one
useless layer)*.

The most common one is **ReLU** *(Rectified Linear Unit — if the number > 0,
keep it; if ≤ 0, set it to 0)*.

```
Before ReLU:  [-3,  0,  5, -1,  8, -4,  2]
After ReLU:   [ 0,  0,  5,  0,  8,  0,  2]
```

Why? Negative values mean "this filter's pattern was NOT found here."
We don't need negatives — just zero them out.
This also introduces **non-linearity** *(allows the network to learn curved,
complex decision boundaries — not just straight lines)*.

---

## Step 6: Pooling — shrinking the image

After conv + ReLU, we apply **pooling** *(shrinking a feature map by keeping
only one summary number per small region — reduces size and memory)*.

**Max pooling** *(keep only the largest value in each 2×2 block)* is most common:

```
Feature map (4×4):        After Max Pool 2×2:
  1   3 | 2   4
  5   2 | 1   0    →    5   4
  ------+------
  3   1 | 8   2         3   8
  0   2 | 4   8

  Top-left block:    max(1,3,5,2) = 5
  Top-right block:   max(2,4,1,0) = 4
  Bottom-left block: max(3,1,0,2) = 3
  Bottom-right block:max(8,2,4,8) = 8
```

Each 2×2 block → 1 number (the largest). Image goes from 4×4 → 2×2.

Why? We don't need to know exactly WHERE a pattern was — just ROUGHLY where.
"An eye was in the top-left region" is enough.
Smaller = less memory, less computation.

---

## Step 7: Stacking layers — building up complexity

One Conv2d + ReLU + Pool is called a **layer** *(one processing step in a
network — transforms input to output)*. We stack many of them.

```
Input image: (3, 512, 512)   ← 3 channels (RGB), 512×512 pixels

Layer 1 (Conv + ReLU + Pool):
  sees: 3×3 pixel patches
  detects: edges, colour blobs
  output: (32, 256, 256)   ← 32 filters, image halved by pooling

Layer 2 (Conv + ReLU + Pool):
  sees: effectively 6×6 pixel regions (3×3 on the already-pooled features)
  detects: corners, curves, textures
  output: (64, 128, 128)

Layer 3 (Conv + ReLU + Pool):
  sees: effectively 12×12 pixel regions
  detects: shapes, parts of objects
  output: (128, 64, 64)

Layer 4 (Conv + ReLU + Pool):
  sees: effectively 24×24 pixel regions
  detects: ears, wheels, eyes
  output: (256, 32, 32)

Layer 5 (Conv + ReLU + Pool):
  sees: effectively 48×48 pixel regions
  detects: faces, cats, cars
  output: (512, 16, 16)
```

Simple patterns combine into complex ones. This is what "deep" in
**deep learning** means — many layers stacked.

---

## Step 8: Flattening and classifying

After all conv layers (following Step 7's 5-layer example), we have a small
tensor of abstract features. Shape: **(512, 16, 16)**.

**Flatten** *(collapse all dimensions into one long list)*:
512 × 16 × 16 = 131,072 numbers in a single list.

Feed that into a **Linear layer** *(also called fully-connected or dense —
every input number connects to every output neuron, like a lookup table with
learned weights)*:

```
131,072 numbers → Linear(131072, 1000) → 1,000 numbers
```

The final layer outputs one number per class — these are called **logits**
*(raw, unnormalised scores — not yet probabilities; can be any value positive
or negative)*.

Apply **Softmax** *(converts a list of scores to probabilities that sum to 1.0
— the largest score gets the largest probability)* to turn logits into
**probabilities** *(numbers between 0 and 1 that sum to 1.0 — confidence per class)*:

```
Logits:        [8.3,   2.1,  -1.4,   0.5]
After Softmax: [0.97,  0.02,  0.00,  0.01]  ← sums to 1.0

→ 97% confident this is class 0 (cat)
```

---

## Step 9: How does it learn?

At the start, all filter numbers (**weights**) are random. The network is useless.

**Training loop:**

1. Show the network a photo of a cat.
2. It runs the **forward pass** *(input data flowing through every layer to
   produce a prediction)* and predicts "dog: 73%."
3. We measure how wrong it was with a **loss function** *(a formula that outputs
   a single number measuring how wrong the prediction was — lower = better)*.
   Here we use **CrossEntropyLoss** *(the standard loss for classification —
   measures how far predicted probabilities are from the correct class)*.
4. **Backpropagation** *(works backwards from the loss through every layer,
   computing how much each weight contributed to the mistake)* runs.
5. The **optimizer** *(the algorithm that uses gradients to update weights —
   common choice: AdamW)* uses the **gradients** *(a number per weight saying
   "if you increase this weight, the loss goes up/down by this much")* to
   nudge every filter number slightly in the direction that reduces the loss.
   This is **gradient descent** *(move weights downhill on the loss surface)*.
6. Before the next step, call **zero_grad** *(reset all gradients to zero —
   if you forget this, gradients from multiple steps add up and corrupt training)*.
7. Repeat for the next image. And the next. And a million more.

After enough repetitions across the full dataset (one complete pass = one
**epoch** *(one full loop through the entire training dataset)*), the filters
have slowly adjusted into patterns that reliably detect cats, dogs, cars.

The network learned to see — by being wrong millions of times and slowly
correcting itself each time.

---

## Step 10: The batch — processing many images at once

We don't process one image at a time (slow). We group them into a
**batch** *(a small group of training examples processed together — typically
32 or 64 images at once)*.

**Batch size** *(how many images per batch)* of 32 means:
- Input shape: **(32, 3, 512, 512)** — B × C × H × W
  - **B** = batch (32 images at once)
  - **C** = channels (3 for RGB)
  - **H** = height, **W** = width

A **DataLoader** *(PyTorch helper that takes your data and automatically
groups it into batches, shuffles order, and loads in parallel)* handles this.
You write a **Dataset** *(a Python class that knows how to load one sample
by index — you write \_\_len\_\_ and \_\_getitem\_\_)*. DataLoader uses your
Dataset automatically.

---

## Step 11: Overfitting vs underfitting

**Overfitting** *(model memorises training data but fails on new data —
train loss low, val loss high)* happens when the model is too powerful for
the amount of data.

Fixes:
- **Dropout** *(randomly zero out some neurons during training — forces the
  network to not rely on any single neuron)*
- **Weight decay** *(penalise large weights — keeps the model simpler)*
- More training data

**Underfitting** *(model too simple to learn the patterns — both train and val
loss stay high)* happens when the model is too small.

Fix: use a bigger model, train longer.

We monitor this with a **validation set** *(data held out during training,
never used to update weights — used only to measure how well the model
generalises to new data)*.

---

## The whole thing in one paragraph

An image is a grid of numbers split into **channels** (3 for RGB). A
**Conv2d layer** slides many small grids of learned numbers called **filters**
(or **kernels**) across the image — this is a **convolution**. At each
position, it multiplies and sums, producing one number. Doing this across all
positions gives a **feature map** — a heatmap of where that filter's pattern
appeared. Many filters run in parallel, one per output **channel**. After each
conv layer, **ReLU** zeros out negatives. **Pooling** shrinks the spatial size.
Stacking many conv + ReLU + pool **layers** builds up from simple patterns
(edges) to complex ones (faces). Finally, **flatten** + **Linear layers**
convert the features into class **logits** → **Softmax** → probabilities.
Training works by: forward pass → compute **loss** (CrossEntropyLoss) →
**backpropagation** → **optimizer** nudges **weights** via **gradient descent**
→ **zero_grad** → repeat for each **batch** in each **epoch**.

---

## Shape summary for one image through a CNN

*(5-layer example matching Step 7)*

```
Input:              (1,   3, 512, 512)  — 1 image, 3 channels (RGB), 512×512
After Conv2d(32):   (1,  32, 512, 512)  — 32 feature maps (one per filter)
After MaxPool2d:    (1,  32, 256, 256)  — spatial size halved
After Conv2d(64):   (1,  64, 256, 256)
After MaxPool2d:    (1,  64, 128, 128)
After Conv2d(128):  (1, 128, 128, 128)
After MaxPool2d:    (1, 128,  64,  64)
After Conv2d(256):  (1, 256,  64,  64)
After MaxPool2d:    (1, 256,  32,  32)
After Conv2d(512):  (1, 512,  32,  32)
After MaxPool2d:    (1, 512,  16,  16)
After Flatten:      (1, 131072)         — 512 × 16 × 16 = 131,072 numbers
After Linear(1000): (1, 1000)           — one score per class
After Softmax:      (1, 1000)           — one probability per class (sums to 1.0)
```
