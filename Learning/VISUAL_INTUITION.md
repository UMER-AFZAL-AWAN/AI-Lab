# How a CNN Actually Works
## From zero — no code, no jargon, just plain words.

---

## Step 1: What is an image to a computer?

You look at a photo of a cat and you see a cat.

The computer sees a grid of numbers.

A 6×6 pixel image is just a 6×6 table of numbers. Each number is a brightness
value from 0 (black) to 255 (white). A colour image has three of these grids
stacked — one for Red, one for Green, one for Blue.

That's it. The computer never "sees" anything. It only ever works with numbers.

```
A tiny 6×6 grayscale image:
  0   0   0   0   0   0
  0   0 200 200   0   0
  0 200 255 255 200   0
  0 200 255 255 200   0
  0   0 200 200   0   0
  0   0   0   0   0   0
```

This particular grid of numbers happens to look like a bright square in the
centre. That's a "shape" to you. To the computer it's just numbers.

---

## Step 2: What is a filter (kernel)?

A filter is a tiny grid of numbers — usually 3×3.

```
A filter that detects vertical edges:
 -1   0   1
 -1   0   1
 -1   0   1
```

It is a question. Specifically, the question: "Is there a vertical edge here?"

The numbers in the filter encode WHAT to look for.
- Left column is -1 (negative): dark on the left
- Right column is +1 (positive): bright on the right
- That pattern = a vertical edge

You don't design these filters by hand. The neural network LEARNS them during
training. It figures out which filters are useful for the task.

---

## Step 3: What happens when the filter slides over the image?

This is the core operation. It's called a convolution.

Take your 3×3 filter. Place it on the top-left corner of the image. 
Now multiply each filter number by the image number it's sitting on.
Add all 9 results together.
That sum becomes ONE number in the output.

Then slide the filter one step to the right. Repeat.
Slide again. Repeat.
When you reach the right edge, drop down one row. Repeat the whole thing.

You end up with a new grid of numbers — the same size as the image.
This new grid is called a FEATURE MAP.

```
Image patch (3×3):        Filter:          Multiply & sum:
  0   0   0              -1   0   1        0×(-1) + 0×0 + 0×1
  0 200 255    ×         -1   0   1    =   0×(-1) + 200×0 + 255×1
  0 200 255              -1   0   1        0×(-1) + 200×0 + 255×1
                                           = 0 + 0 + 0 + 0 + 0 + 255 + 0 + 0 + 255
                                           = 510
```

A HIGH number in the feature map means: "YES, the thing this filter looks for
is strongly present at this location."

A LOW (or negative) number means: "NO, it's not here."

The feature map is a HEATMAP of where the filter's pattern was found in the image.

---

## Step 4: Many filters = many questions asked at once

One filter asks one question ("is there a vertical edge here?").

A CNN layer has MANY filters running in parallel — often 32, 64, or 512.

Each filter asks a different question:
- Filter 1: "Is there a vertical edge?"
- Filter 2: "Is there a horizontal edge?"
- Filter 3: "Is there a diagonal line?"
- Filter 4: "Is there a bright blob?"
- ... and so on

After the layer, you have 32 (or 64, or 512) feature maps stacked on top of
each other. Each one is a heatmap for a different question.

This stack is called a TENSOR. Shape: (num_filters, height, width).

---

## Step 5: Why does the image get smaller as it goes deeper?

Two reasons:

**Pooling:**
After each conv layer, we apply a "max pool" operation.
Take every 2×2 block of the feature map. Keep only the LARGEST number. Discard the rest.
Result: the image shrinks to half its size.

Why? We don't need to know exactly WHERE a feature was — just roughly WHERE.
If an eye was detected somewhere in the top-left region, we don't need to know
it was at pixel (47, 23). "Top-left" is enough.
Smaller = less memory, less computation.

**Stacking makes big patterns from small ones:**
- Layer 1 (sees 3×3 pixels): detects edges, blobs
- Layer 2 (sees 6×6 pixels effectively): detects corners, curves
- Layer 3 (sees 12×12 pixels effectively): detects textures, parts
- Layer 4 (sees 24×24 pixels effectively): detects eyes, ears, wheels
- Layer 5 (sees 48×48 pixels effectively): detects faces, cats, cars

Each layer builds on the previous one. Simple patterns combine into complex ones.
This is what "deep" in "deep learning" means — many layers stacked.

---

## Step 6: How does it go from a heatmap to "this is a cat"?

After all the conv layers, you have a small tensor of abstract features.
Something like shape (512, 7, 7) — 512 feature maps, each 7×7 pixels.

Flatten that into a single long list of numbers: 512 × 7 × 7 = 25,088 numbers.

Feed that list into a regular classifier (a few linear layers).
The last layer outputs ONE number per class:
  - Score for "cat": 8.3
  - Score for "dog": 2.1
  - Score for "car": -1.4
  - ...

Apply softmax to convert scores to probabilities:
  - cat: 97%
  - dog: 2%
  - car: 0%

That's the prediction.

---

## Step 7: How does it LEARN the right filters?

This is the magic part.

At the start, all filters are random numbers. The network is useless — it
outputs random predictions.

You show it a photo of a cat and tell it "this is a cat."
It predicts "dog" with 73% confidence.
That's wrong. How wrong? We measure it with a LOSS function.
Loss = how far the prediction was from the correct answer.

Now BACKPROPAGATION runs:
- It asks: "which filter numbers caused this mistake?"
- It calculates: "if I nudge this filter number up by 0.001, does the loss go
  up or down?"
- It nudges every filter number slightly in the direction that reduces loss.

You repeat this for a million cat photos. And dog photos. And car photos.

After enough repetitions, the filters have slowly adjusted themselves into
patterns that reliably separate cats from dogs from cars.

The network learned to see — by being wrong millions of times and slowly
correcting itself each time.

---

## The whole picture in one paragraph

An image is a grid of numbers. A CNN slides small grids of learned numbers
(filters) across the image, asking "is my pattern here?" at every position.
The answer at each position becomes a new grid (feature map) — a heatmap of
where that pattern was found. Many filters run in parallel, each finding a
different pattern. Deeper layers combine simple patterns (edges) into complex
ones (faces). At the end, the feature maps are flattened and classified. The
filter values are not designed by hand — they are learned by being wrong and
correcting, millions of times.

---

## What to look at now

Go back to: https://poloclub.github.io/cnn-explainer

You will see:
- The tiny image on the left
- Filters sliding across it (the 3×3 grids moving)
- Feature maps appearing on the right of each layer
- Hovering on a feature map shows you which filter produced it

Now it should make sense. Each column of coloured boxes is one layer.
Each box in that column is one feature map (one filter's heatmap).
The image shrinks as you go right because of pooling.
The final layer feeds into the class scores on the far right.

---

## Now go run this file

`Learning/02_cnn/01_convolution.py`

You now know what it's doing. The code is just the maths above, written in Python.
