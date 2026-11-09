# Glossary — Every Keyword Explained Like You Are 5

No code. No maths. Just plain words.
When you hit a word you don't know, look it up here first.

---

## A

**Activation**
The output number a neuron produces after doing its calculation.
Like the answer a calculator shows after you press =.

**Activation Function**
A rule applied to a neuron's output to squash it into a useful range.
Without it, stacking layers does nothing useful (they'd just collapse into one layer).
Common ones: ReLU, Sigmoid, Softmax, GELU.

**AdamW**
The most popular optimizer in modern deep learning.
Smarter than SGD — it remembers how much each weight has been moving and
adjusts step sizes individually per weight. Almost always use this.
The W means weight decay is applied correctly (a bug fix over plain Adam).

**Attention**
A mechanism that lets the model decide WHICH parts of the input to focus on.
Like reading a paragraph and highlighting the words most relevant to your question.
The model learns what to highlight — you don't tell it.

**Autograd**
PyTorch's system that automatically tracks every calculation you do so it can
compute gradients (run backpropagation) for you.
You just write the forward pass. PyTorch handles the backward pass.

**Autocast**
A PyTorch tool that automatically runs your forward pass in FP16 (half precision)
to save memory and speed up training, while keeping the important parts in FP32.

---

## B

**Backpropagation (Backprop)**
The algorithm that figures out how much each weight contributed to the mistake.
Works backwards from the loss through the network using the chain rule.
Result: a gradient for every weight, telling it which direction to move.

**Batch**
A small group of training examples processed together in one forward pass.
Instead of showing the model one image at a time (slow) or all images at once
(too much memory), we process 32 or 64 at a time.

**Batch Size**
How many examples are in one batch. Typical values: 8, 16, 32, 64.
Larger = more stable gradients, more memory. Smaller = noisier, less memory.

**BatchNorm (Batch Normalisation)**
After a layer, rescale the outputs so they have mean≈0 and std≈1.
Prevents values from exploding or shrinking as they pass through many layers.
Makes training much more stable and faster.
Used in CNNs. Transformers use LayerNorm instead.

**BF16 (BFloat16)**
A 16-bit number format with the same range as FP32 but less precision.
Preferred over FP16 for training because it doesn't overflow.
Your RTX 5060 supports BF16.

**Bias**
An extra learnable number added to a neuron's output.
Like the y-intercept in y = mx + b. Lets the neuron shift its output up or down.
Every Linear layer has weights AND biases.

**Broadcasting**
When PyTorch automatically stretches a smaller tensor to match a larger one.
Like adding a single number to every element of a list without writing a loop.

---

## C

**Channel**
One "layer" of a grid of numbers.

For images:
- A grayscale image has 1 channel (brightness)
- A colour image has 3 channels (Red, Green, Blue)

After a conv layer with 32 filters:
- You have 32 channels — one per filter's output

Think of channels as asking: "how many different questions am I tracking at each pixel?"

**Checkpoint**
A saved copy of a model's weights at a specific point in training.
Like a save file in a game. If training crashes, you reload from the last checkpoint.

**Classification**
Deciding which category something belongs to.
Input: image. Output: "cat" or "dog" or "car".
The final layer outputs one score per class.

**CNN (Convolutional Neural Network)**
A neural network designed for images.
Uses convolutions instead of fully-connected layers, so it looks at local patches
rather than the entire image at once. Much more efficient for spatial data.

**Contrastive Learning**
A training technique: push similar things close together in embedding space,
push different things far apart.
How CLIP learns to match images to text captions.

**Convolution**
Sliding a small grid of numbers (a filter) across an image.
At each position: multiply filter numbers by image numbers, sum them all up.
Result: one number per position → a new grid (feature map).
Detects whether the filter's pattern is present at each location.

**Cross-Attention**
Attention where the Query comes from one sequence and the Keys/Values come
from a different sequence.
In diffusion models: Query = image features, Keys/Values = text embeddings.
This is how text ("a red cat") steers what gets generated.

**CrossEntropyLoss**
The standard loss function for classification tasks.
Measures how wrong the predicted class probabilities are.
Lower = model is more confident and correct. Higher = model is wrong or uncertain.
Always pass RAW logits to it (not softmax output).

---

## D

**DataLoader**
A PyTorch helper that takes your Dataset and automatically:
- Groups samples into batches
- Shuffles the order each epoch
- Loads data in parallel (multiple workers)
You write the Dataset. DataLoader handles the rest.

**Dataset**
A Python class that knows how to load one sample of your data by index.
You write two methods: __len__ (how many total?) and __getitem__ (give me #42).

**Dead Neuron**
A neuron stuck at outputting 0 forever.
Happens with ReLU if the input is always negative — gradient becomes 0,
no updates ever happen, neuron never recovers.
Fix: use LeakyReLU, or careful weight initialisation.

**Decoder**
The part of a network that takes a compressed representation and expands it
back to full size.
In a VAE: takes a latent vector → reconstructs the image.
In a Transformer: generates output tokens one at a time.

**Dense Layer**
Another name for a fully-connected layer (Linear layer).
Every input connects to every output. Called "dense" because there are no gaps.

**Diffusion Model**
A type of generative model that learns to remove noise.
Training: add random noise to an image in steps.
Inference: start from pure noise, remove noise step by step → generates an image.
Stable Diffusion, AnimateDiff, Wan are all diffusion models.

**Dropout**
During training: randomly set some neuron outputs to 0.
Forces the network to not rely on any single neuron → more robust.
During inference: turned off (all neurons active).
Controlled by a probability p (e.g. p=0.1 means 10% of neurons zeroed).

**dtype**
The number format a tensor uses. Common ones:
- float32 (FP32): 4 bytes, full precision, default
- float16 (FP16): 2 bytes, half precision, faster on GPU
- bfloat16 (BF16): 2 bytes, safer range than FP16
- int8: 1 byte, used for quantisation (inference only)

---

## E

**Embedding**
Converting something discrete (a word, a pixel patch, a user ID) into a dense
vector of numbers.
Similar things get similar vectors. "cat" and "kitten" end up close together.
Learned during training.

**Encoder**
The part of a network that compresses input into a smaller representation.
In a VAE: takes an image → compresses to a latent vector.
In a Transformer: takes tokens → produces context-aware representations.

**Epoch**
One complete pass through the entire training dataset.
If you have 1,000 images and batch size 32: one epoch = ~31 batches.
Training usually runs for 10-100+ epochs.

---

## F

**Feature**
A meaningful piece of information the model has learned to detect.
Early layers: edges, colours, blobs.
Middle layers: textures, corners, curves.
Late layers: eyes, ears, wheels, faces.
You don't define features — the model discovers them.

**Feature Map**
The output grid produced by one filter sliding across an image.
A heatmap showing WHERE in the image that filter's pattern was found.
One conv layer with 32 filters produces 32 feature maps.

**Fine-tuning**
Taking a pretrained model and continuing to train it on your specific task/data.
Like hiring an expert (pretrained) and teaching them your company's specific process.
Much faster and cheaper than training from scratch.

**Filter (Kernel)**
A small grid of numbers (usually 3×3) that slides across an image.
Each filter asks one question: "is my pattern here?"
The numbers in the filter determine what pattern it detects.
Learned during training — not set by hand.

**FP16 (Float16)**
16-bit floating point. Half the size of FP32.
Faster on GPU, uses half the memory.
Limited range — can overflow to infinity for large values.
Use GradScaler when training with FP16.

**FP32 (Float32)**
32-bit floating point. The standard precision.
4 bytes per number. Full precision. The default.

**Forward Pass**
Running input data through the network to get a prediction.
Input → Layer 1 → Layer 2 → ... → Output.
Like pressing play on a function.

---

## G

**Gaussian**
The bell-curve distribution. Most values near the mean, fewer far away.
Random noise in diffusion models is Gaussian.
When we say "sample from a Gaussian" we mean: pick a random number from this curve.

**GELU**
An activation function used in Transformers (GPT, BERT, etc.).
Similar to ReLU but smoother — doesn't have a hard cutoff at 0.
You rarely need to choose this — it's the default in most modern architectures.

**Gradient**
A number (or vector) that says: "if you increase this weight by a tiny amount,
the loss goes UP by this much."
Positive gradient → weight is making things worse → decrease it.
Negative gradient → weight is making things better → increase it.
Gradients point uphill. We walk downhill (subtract them).

**Gradient Clipping**
Before the optimizer step: if any gradient is too large, scale them all down.
Prevents "gradient explosion" where one bad batch sends all weights to infinity.
Standard in Transformer training: max_norm=1.0.

**Gradient Descent**
The core training algorithm.
Compute gradients → move weights slightly in the direction that reduces loss → repeat.
"Descent" because we're going downhill on the loss surface.

**GradScaler**
A PyTorch tool used with FP16 training.
Multiplies the loss by a large number before backward pass so gradients don't
underflow to zero. Automatically adjusts the scale factor.

---

## H

**Hidden Layer**
Any layer that is not the input or output.
"Hidden" because you don't directly observe what it does.
More hidden layers = deeper network = can learn more complex patterns.

**Hyperparameter**
A setting YOU choose before training (not learned by the model).
Examples: learning rate, batch size, number of layers, dropout rate.
Contrast with parameters (weights) which the model learns.

---

## I

**Inference**
Using a trained model to make predictions on new data.
No training, no gradient computation. Just forward pass.
Always call model.eval() before inference.

**IoU (Intersection over Union)**
A metric for how well two bounding boxes overlap.
0.0 = no overlap at all. 1.0 = identical boxes.
Used in object detection to evaluate prediction quality.

---

## K

**Kernel**
Same as Filter. A small grid of numbers that slides across an image.
In PyTorch: kernel_size=3 means a 3×3 filter.

**KL Divergence**
A measure of how different two probability distributions are.
0 = identical. Higher = more different.
Used in VAEs: measures how far the learned distribution is from a standard Gaussian.

---

## L

**L2 Loss**
See MSELoss.

**Latent Space**
A compressed, lower-dimensional representation of your data.
In a VAE: an image (512×512×3) is compressed to a small vector (e.g. 64×64×4).
The diffusion model works in latent space — much cheaper than pixel space.

**Layer**
One processing step in a neural network.
Input → Layer → output.
Types: Linear (fully connected), Conv2d, BatchNorm, Dropout, Attention, etc.
Layers are stacked to build deep networks.

**LayerNorm (Layer Normalisation)**
Like BatchNorm but normalises across features instead of across the batch.
Works for any batch size, including batch=1.
Used in every Transformer (BERT, GPT, CLIP, AnimateDiff UNet).

**Learning Rate (lr)**
How big a step the optimizer takes when adjusting weights.
Too high: overshoots the minimum, loss explodes or oscillates.
Too low: takes forever to converge.
Typical starting point: 1e-4 (that's 0.0001).

**Linear Layer**
The basic building block. Computes: output = input × weights + bias.
Every input connects to every output — hence "fully connected."
In PyTorch: nn.Linear(in_features, out_features).

**Logits**
Raw, unnormalised scores output by the final layer.
Not probabilities — they can be any value, positive or negative.
Apply softmax to convert logits to probabilities.
Always pass logits (not probabilities) to CrossEntropyLoss.

**LoRA (Low-Rank Adaptation)**
A technique to fine-tune large models cheaply.
Instead of updating all weights, add two tiny matrices (A and B) to each layer.
Only train A and B — the original weights stay frozen.
Result: same quality as full fine-tuning, 100× fewer trainable parameters.

**Loss**
A single number measuring how wrong the model's prediction was.
Training goal: make this number as small as possible.
Higher loss = more wrong. Loss = 0 = perfect (almost never happens).

**Loss Function**
The formula used to calculate the loss.
Different tasks use different loss functions:
- Regression: MSELoss
- Classification: CrossEntropyLoss
- Binary: BCELoss

---

## M

**Matrix**
A 2D grid of numbers. Like a spreadsheet.
Shape: (rows, columns). E.g. a (4, 3) matrix has 4 rows and 3 columns.

**Matrix Multiply (@)**
The fundamental operation in neural networks.
(m, K) @ (K, n) → (m, n).
The inner dimensions must match (both K). Result is (m, n).
Every Linear layer is a matrix multiply.

**Model**
In ML: the neural network with all its learned weights.
"The model" = the network architecture + the trained parameter values.

**MSELoss (Mean Squared Error)**
Loss function for regression (predicting numbers, not categories).
Average of (prediction - truth)² across all examples.
Sensitive to large errors (squaring magnifies them).

**Multi-Head Attention**
Running the attention operation H times in parallel, each with different
learned projections. Each "head" looks at different aspects.
Then concatenate all heads' outputs.
H=8 or H=12 is common. More heads = more things to look at simultaneously.

---

## N

**NaN (Not a Number)**
What you get when maths goes wrong: dividing by zero, log(0), 0/0.
In training: NaN in the loss means something is broken.
Once one NaN appears, it spreads through the whole model — training is ruined.

**Neural Network**
A stack of layers, each doing: output = activation(input × weights + bias).
Loosely inspired by neurons in a brain. Learns by adjusting weights.

**Neuron**
One unit in a layer. Takes inputs, multiplies by weights, adds bias, applies activation.
Outputs one number.
A layer with 512 neurons has 512 of these units, all running in parallel.

**Normalisation**
Rescaling input values so they're in a predictable range.
For images: subtract mean, divide by std → values roughly in [-2, 2].
Important: use the same normalisation at training AND inference.
Wrong normalisation = model sees out-of-distribution inputs = bad results.

**NMS (Non-Maximum Suppression)**
In object detection: removes duplicate bounding boxes for the same object.
Keep the highest-confidence box. Remove all boxes that overlap with it too much.
Repeat until no overlapping boxes remain.

---

## O

**Optimizer**
The algorithm that updates the weights using gradients.
Common ones: SGD (simple), Adam (adaptive), AdamW (Adam with correct weight decay).
Takes gradients → adjusts weights → hopefully reduces loss.

**Overfitting**
When the model memorises the training data instead of learning general patterns.
Signs: very low training loss, much higher validation loss.
Fix: more data, dropout, weight decay, smaller model.

---

## P

**Parameter**
A learnable number inside the model — a weight or bias.
Adjusted during training by the optimizer.
A ResNet-50 has 25 million parameters.

**Patch**
A small square region of an image.
ViT (Vision Transformer) splits images into 16×16 patches and treats each as a token.

**Pooling**
Shrinking a feature map by summarising each small region with one number.
Max pooling: take the largest value in each 2×2 block.
Average pooling: take the average value in each 2×2 block.
Halves the spatial size. Reduces memory and computation.

**Pretrained Model**
A model already trained on a large dataset (e.g. ImageNet, internet text).
Has already learned useful features. Fine-tune it for your specific task.
Much better starting point than random initialisation.

**Probability Distribution**
A description of how likely each possible outcome is.
Sum of all probabilities = 1.0.
Softmax converts logits (any numbers) to a probability distribution.

---

## Q

**Quantisation**
Reducing weights from 32-bit or 16-bit to 8-bit or 4-bit.
4-bit quantisation: a 7B model shrinks from ~14GB to ~4GB VRAM.
Slight quality loss. Used heavily for running LLMs locally.

---

## R

**ReLU (Rectified Linear Unit)**
The most common activation function.
Rule: if input > 0, output = input. If input ≤ 0, output = 0.
Simple, fast, works well. Used in almost all CNNs.
Problem: can cause dead neurons (see Dead Neuron).

**Residual Connection (Skip Connection)**
A shortcut that adds the input directly to the output of a layer.
output = layer(input) + input
Prevents gradients from vanishing in very deep networks.
Used in ResNet, U-Net, Transformers — almost every modern architecture.

**RGB**
The three colour channels of an image: Red, Green, Blue.
Each channel is a grid of values 0-255.
A colour image tensor has shape (3, H, W).

---

## S

**Scheduler (LR Scheduler)**
Changes the learning rate during training.
Common pattern: start high, end low (cosine decay).
Or: warm up slowly, then decay (warmup + cosine).
Helps the model converge better.

**Self-Attention**
Attention where Q, K, V all come from the same sequence.
Each token can attend to every other token.
Lets the model understand relationships across the whole sequence.
Core of every Transformer.

**Sigmoid**
An activation function that squashes output to [0, 1].
Used for binary classification (is this spam? yes/no).
Formula: σ(x) = 1 / (1 + e^(-x))

**Softmax**
Converts a list of scores (logits) to probabilities that sum to 1.
Input: [2.0, 1.0, -1.0] → Output: [0.71, 0.26, 0.03]
The largest input gets the largest probability.
Temperature controls how "peaked" the distribution is.

**Step (Training Step)**
One forward pass + backward pass + optimizer update on one batch.
1 step = process 1 batch = call optimizer.step() once.
Different from epoch (which covers the whole dataset).

**State Dict**
A dictionary mapping parameter names to their values.
How you save and load a model's weights:
  torch.save(model.state_dict(), "model.pth")
  model.load_state_dict(torch.load("model.pth"))

---

## T

**Temperature**
A number that controls how random a model's output is.
Low temperature (0.1): model is very confident, picks the most likely token.
High temperature (2.0): model is more random, explores more options.
Temperature = 1.0: use the raw probabilities unchanged.

**Tensor**
The fundamental data structure in PyTorch.
An N-dimensional grid of numbers.
- 0D: a single number (scalar)
- 1D: a list of numbers (vector)
- 2D: a table of numbers (matrix)
- 3D: a stack of matrices (like an RGB image: 3 × H × W)
- 4D: a batch of images (B × C × H × W)

**Token**
A piece of text that the model processes as one unit.
Words, parts of words, or characters — depends on the tokeniser.
"unforgettable" might become ["un", "forget", "table"] — 3 tokens.

**Training**
The process of adjusting a model's weights to reduce loss on a dataset.
Forward pass → compute loss → backprop → optimizer step → repeat.

**Transfer Learning**
Using a model trained on one task as a starting point for a different task.
Pretrain on ImageNet → fine-tune on your medical images.
Works because low-level features (edges, textures) transfer across tasks.

**Transformer**
A neural network architecture based entirely on attention.
No convolutions. No recurrence.
Used in: GPT, BERT, CLIP, the UNet inside Stable Diffusion, Wan, everything modern.

---

## U

**Underfitting**
When the model is too simple to learn the patterns in the data.
Signs: high training loss AND high validation loss.
Fix: larger model, more epochs, lower weight decay.

**UNet**
An encoder-decoder architecture with skip connections.
Encoder: compress image to small representation.
Decoder: expand back to full size.
Skip connections: pass encoder features directly to decoder at each scale.
Used in: image segmentation, AND the denoising network in every diffusion model.

---

## V

**VAE (Variational Autoencoder)**
A network that learns to compress images to a small "latent" vector, then
reconstruct them.
In Stable Diffusion: VAE compresses a 512×512 image to 64×64 latent.
The diffusion model works on the 64×64 latent — 64× cheaper than pixel space.
To see the final image: pass latent through VAE decoder.

**Validation Set**
A portion of your data held out during training.
Used to measure how well the model generalises.
Never used to update weights — only for measuring performance.
If train loss goes down but val loss goes up → overfitting.

**Vector**
A list of numbers with a direction and magnitude in multi-dimensional space.
An embedding is a vector. A word "cat" might be a 768-dimensional vector.
Similarity between vectors = cosine similarity.

**ViT (Vision Transformer)**
A Transformer applied to images by splitting them into patches.
Each 16×16 patch → one token → Transformer processes the sequence of patches.
Competes with (and often beats) CNNs on image tasks.

---

## W

**Weight**
A learnable number inside a layer. The values the model adjusts during training.
A Linear layer with in=4, out=3 has 4×3=12 weights (plus 3 biases).
Weights encode what the model has learned.

**Weight Decay**
A regularisation technique: add a small penalty to the loss for large weights.
Pushes all weights toward zero → simpler model → less overfitting.
Controlled by weight_decay parameter in the optimizer (e.g. 0.01).

---

## Z

**Zero Grad**
Before each backward pass you must zero out the gradients from the previous step.
Otherwise gradients accumulate (add up) across steps — a very common bug.
optimizer.zero_grad() — always call this before loss.backward().

---

## Shape notation cheat sheet

```
B  = batch size (how many samples in this batch)
C  = channels (RGB=3, or number of filters after a conv layer)
H  = height (pixels)
W  = width (pixels)
T  = time (number of frames in a video)
L  = sequence length (number of tokens)
D  = embedding dimension (size of each token's vector)
N  = any generic dimension

Common shapes:
  Image:          (B, C, H, W)       e.g. (32, 3, 512, 512)
  Video:          (B, C, T, H, W)    e.g. (4, 3, 16, 512, 512)
  Text tokens:    (B, L)             e.g. (8, 77)
  Text embeddings:(B, L, D)          e.g. (8, 77, 768)
  Latent (SD):    (B, 4, H/8, W/8)  e.g. (1, 4, 64, 64) for a 512×512 image
```
