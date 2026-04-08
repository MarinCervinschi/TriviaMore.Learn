# Discrete AI

## Introduction

**DISCRETIZATION**

While most deep learning models operate in **continuous vector spaces**, many real-world processes are naturally represented by **discrete symbols**. To better model these domains, we aim to build architectures that can learn **discrete representations**.

The **Discretization** is the process of **partitioning continuous phenomena into distinct parts,** this aligns closely with the way humans interpret and interact with the world.

Many real-world domains are inherently discrete, and modeling them with continuous representations may miss important structural properties. Common discrete domains include:

- **Language** consists of sequences of discrete symbols (words, subwords, or phonemes).
- **Images** are stored as discrete grids of pixels with quantized color values.
- **Music** and **speech** are often represented symbolically (e.g., notes, phonemes).

**DISCRETE AI SYSTEMS**

In the context of artificial learning systems, **discrete models** and **representations enable**:

- **Efficient compression** The discrete latents are easier to encode compactly, so ****by utilizing a discrete domain, we can deploy highly efficient algorithms to compress complex data (such as images).
- **Symbolic reasoning** classic logical reasoning (such as Boolean algebra) operates on discrete states.

- **Modular design** Instead of processing every input through every layer (as in continuous projection), the model can use discrete signals to "route" inputs to specific sub-modules or transformations. This allows the network to dynamically choose which parts of the model to activate based on the specific input.

![image.png](Discrete%20AI/image.png)

**PROBLEMS**

Neural networks are optimized using **Gradient Descent**, which relies on the chain rule of calculus to **backpropagate error gradients**. This requires every operation in the computational graph to be differentiable.

However, discrete operations break this pipeline because they are **not differentiable**. This creates several fundamental issues:

<aside>

**Case Study 1 — Quantization**

Consider a function that maps a continuous value to the nearest integer:

$r' =quantize(r \in \mathcal{R})$

Neural networks learn through gradients: during the backward pass we use gradients to update the weights. But quantization is **not a differentiable function**:

- **Forward pass:** Continuous inputs jump to discrete values (e.g., 1.4 → 1.0). Small changes in the input do **not** change the output, producing flat regions in the function.
- **Backward pass:** The gradient is **0** (because the curve is flat between steps). With a zero gradient, the network cannot update its weights, effectively **blocking learning** at this layer.
</aside>

<aside>

**Case Study 2 — Hard Selection (Argmax)**

A common discrete operation is selecting the "best" option from a set. This is often done using **`argmax`** which again ****is a non-differentiable operation. It selects an index, breaking the computational graph. 

When backpropagation reaches an argmax, gradients cannot flow through it. To train such systems, we need strategies that produce a **differentiable approximation of the maximum** (e.g., softmax relaxations, Gumbel-Softmax, etc.).

</aside>

Another challenge is that **we do not always know which loss function is appropriate** for training models with discrete components. Classical losses (like MSE or cross-entropy) assume differentiability, so new formulations are required when discrete choices are part of the model.

**SOLUTIONS**

 We will see some solutions:

- **VQ-VAEs involve straight-through estimators**
- **Pixel-RNN employ autoregressive training** (teacher forcing, likelihood maximization)
- **Surrogate gradients** (e.g., Gumbel Softmax)

## Vector Quantized-Variational Autoencoders

**VARIATIONAL AUTOENCODER: OVERALL ARCHITECTURE**

![image.png](Discrete%20AI/image%201.png)

**VAE: KNOWN ISSUES**

- **Blurry, low-quality outputs:**
    
    VAEs often produce outputs that look blurry or unrealistic. In many samples, the background tends to dominate and the generated images lack fine details.
    
- **Mode collapse:**
    
    The model may generate only a **small subset of the possible outputs**  (i.e., a few “modes” of the data distribution), failing to capture the full diversity present in the training set.
    

![image.png](Discrete%20AI/image%202.png)

**VAE: PROBLEMS**

The standard VAE formulation uses a **Kullback-Leibler (KL) divergence** term that forces the latent representations to approximate a **standard Gaussian prior.**

This pushes different inputs toward a single **unimodal Gaussian distribution**, reducing the expressiveness of the latent space and leading to overly smooth, blurry reconstructions. Ideally, the model should employ a **more flexible, multimodal prior** that better reflects the structure and variability of real-world data.

This strict regularization also leads to **posterior collapse**, where the encoder’s variational posterior quickly becomes identical to the prior during the early stages of training. When the KL penalty is too strong, the network learns that the “safest” strategy is to ignore the input and simply output the prior distribution. As a result, the latent variables stop carrying meaningful information, and the decoder learns to ignore them.

This collapse is often difficult to reverse: once the encoder converges to an uninformative prior, it rarely recovers without explicit intervention.

**Solution(s):**

1. **β-VAE**  Introduces a hyperparameter **$β$** that controls the weight of the KL divergence term in the ELBO objective:
    
    $E_{z \sim Q(z|X)}[\log P (X|z)] - \beta D_{KL}[Q(z|X) \| P(z)]$
    
2. **Vector Quantised-Variational AutoEncoder (VQ-VAE)** 

**VECTOR QUANTISED-VARIATIONAL AUTOENCODER (VQ-VAE)**

VQ-VAE was the **first successful generative model** to use **discrete latent variables**. It **utilizes the full latent space** and completely **avoids posterior collapse**, because it does *not* use a KL term that forces the posterior toward a fixed prior.

It resembles a standard autoencoder with an **encoder** and **decoder**, but unlike VAEs, the latent embedding space is **not Gaussian**. Instead, it is composed of **discrete learnable embeddings**.

![image.png](Discrete%20AI/image%203.png)

**How does it works?**

1. **Encoding**
    
    The model takes an input image $x$, which is passed through the encoder. This produces a **continuous output** $z_e(x) \in \mathbb{R}^{H \times W \times D}$ where:
    
    - $W$ width
    - $H$ height
    - $D$ channels

![image.png](Discrete%20AI/image%204.png)

- **Codebook**
    
    The encoder output $z_e(x)$ is mapped into a **grid of discrete latent variables,** through a learnable **codebook** (lookup table)  $e \in R^{K \times D}$ , where:
    
    - $K$ is the size of the discrete latent space (number of code vectors)
    - $D$ is the dimensionality of each latent embedding vector.
    
    The codebook consists of $K$ embedding vectors $e_i \in R^D, i = 1, 2, ..., K$ which can be learned through gradient descent.
    
    <aside>
    ⚠️
    
    The **discrete indices** used in the latent grid range from **$0$ to $K−1$**.
    
    </aside>
    
- **Nearest-Neighbor Assignment (Quantization)**
    
    For each location in $z_e(x)$, the corresponding **discrete latent index** is obtained via nearest-neighbor lookup:
    
    $q(z = k|x) =
    \begin{cases}
    1 & \text{if } k = \arg \min_j \|z_e(x) - e_j \|_2, \\
    0 & \text{otherwise}
    \end{cases}$
    The proposal distribution $q(z=k|x)$ is deterministic.
    
- **Decoder**
    
    The input to the decoder is the corresponding embedding vector $e_k$:
    
    $z_q(x) = e_k, \quad \text{where } k = \arg \min_j \|z_e(x) - e_j \|_2$
    
    The decoder takes this grid of discrete tokens and attempts to reconstruct the original image.
    
    <aside>
    ⚠️
    
    **This quantization step is not differentiable.**
    
    VQ-VAE uses the **straight-through estimator** to allow gradients to flow.
    
    </aside>
    

The complete set of parameters are union of parameters of the encoder, decoder, and the embedding space $e$.

![image.png](Discrete%20AI/image%205.png)

**`Figure 1:`** *Left:* A figure describing the VQ-VAE *Right:* Visualisation of the embedding space. The output of the encoder $z(x)$ is mapped to the nearest point $e_2$. The gradient $∇_zL$ (in red) will push the encoder to change its output, which could alter the configuration in the next forward pass.

**VQ-VAE: LEARNING**

The forward pass is like a standard autoencoder, but with a non-linearity that maps latents to a 1-of-K embedding vector. This operation is non-differentiable, as discrete functions like **`argmax`**, **`round`**, or **`quantize`** break gradient flow.

**Solution: straight-through gradient estimation.** To address this, gradients are approximated by copying them from the decoder input $z_q(x)$ to the encoder output $z_e(x)$.

**STRAIGHT-THROUGH GRADIENT ESTIMATION**

The Straight-Through Estimator enables training with non-differentiable operations by:

- Using the discrete output during the forward pass.
- Backpropagating gradients as if the operation were the identity.

Formally:

- **Forward:** $z_q = \text{quantize}(z_e)$ (e.g., nearest codebook entry)
- **Backward:** $\frac{\partial L}{\partial z_e} \approx \frac{\partial L}{\partial z_q}$ (treat as identity)

This approximation prevents gradient flow from breaking and is commonly used whenever a layer is non-differentiable.

**Applications:**

- **Binary neural gates**: forward as step, backward as sigmoid
- **Neural architecture search**, **routing**, and **pruning**

**VQ-VAE: TRAINING OBJECTIVE**

To learn the embedding space, VQ-VAE utilizes Vector Quantization (VQ), a dictionary learning approach. Due to the non-differentiable nature of quantization, the architecture employs **Straight-Through Gradient Estimation** during backpropagation.

The overall training objective is defined as:

$L = \underbrace{\log p(x|z_q(x))}_{\text{Reconstruction error}} + \underbrace{\| \text{stopgrad}[z_e(x)] - z_q(x) \|_2^2}_{\text{Vector Quantisation (VQ)}} + \beta \underbrace{\| z_e(x) - \text{stopgrad}[z_q(x)] \|_2^2}_{\text{Commitment term}}$

<aside>

**Note:** $\text{stopgrad}[\cdot]$ denotes the detach operator. It treats its operand as a non-updated constant during backpropagation.

</aside>

Where:

1.  **Reconstruction term**
    
    Compare the **decoder output** with the **original input** and minimize their difference (e.g., L2 or cross-entropy). This term updates **only the decoder** parameters.
    
2. **Vector Quantization (VQ) term**
    
    This term aligns the **codebook embedding vectors** with the **encoder outputs**, so with the corresponding real value.
    
    - We take the encoder output $z_e(x)$.
    - We perform a nearest-neighbor lookup in the codebook and obtain $z_q(x)$.
    - The VQ loss minimizes:
        
        $\| \text{stopgrad}[z_e(x)] - z_q(x) \|_2^2$
        
        So it moves the **Codebook** vectors closer to the encoder outputs. Since the encoder output is detached (stopgrad), only the codebook is updated.
        
- The **third term (commitment)** encourages the encoder to  be committed to the selected codeword $z_q(x)$.
    
    This term  treats the **codebook vector** as the **ground truth** (target) and pulls the **Encoder** output closer to it. This ensures the encoder produces values that translate easily into discrete symbols.
    

| Loss term | Trains | Does NOT train |
| --- | --- | --- |
| Reconstruction | Decoder | Codebook, Encoder |
| VQ term | Codebook | Encoder, Decoder |
| Commitment term | Encoder | Codebook, Decoder |

The VQ-VAE differs significantly from the standard Variational Autoencoder (VAE).

- **Standard VAE:** Relies on a fixed Gaussian prior (fixed before, during, and after training) and continuous latent space.
- **VQ-VAE:** Replaces the Gaussian distribution with a discrete **codebook** that describes the latent space. Unlike the fixed static prior of a standard VAE, the VQ-VAE learns the distribution of the discrete latent space (the prior) during training.

**VQ-VAE: TRAINING STEP**

![image.png](Discrete%20AI/image%206.png)

In PyTorch, the Straight-Through Estimator used in VQ-VAE can be implemented in a single line:

$z_q^{st} ← z_e + \text{detach} (z_q - z_e)$

**`detach()`** breaks the computational graph, which means that during backpropagation 
the term inside **`detach()`** is treated as a constant.

So:

- During **forward pass,** you get the **real quantized vector**.
    
    $z_q^{\text{st}} = z_e + (z_q - z_e) = z_q$
    
- During **backward pass**:
    
    Gradients stop at **`detach$\mathbf{(z_q - z_e)}$`**, so only **$z_e$** is considered and ****receives gradients.
    
    Effectively the model pretends that:
    
    $z_q^{\text{st}} = z_e$
    
    → quantization is treated as an identity function.
    

**VQ-VAE FOR IMAGE COMPRESSION**

VQ-VAE can act as a learned image compressor: 

- The encoder converts an image into a grid of discrete latent codes, which are far smaller than the original image,
- The decoder reconstructs the image from these codes.
- The resulting codebook indices can then be efficiently compressed using entropy coding methods such as **Huffman** or **arithmetic coding**.

This approach learns **domain-specific representations**, enabling higher-quality reconstructions at low bitrates compared to traditional handcrafted codecs. 

**SAMPLING FROM THE VQ-VAE LATENT SPACE**

A VQ-VAE can **reconstruct images**, but it cannot generate new ones because it does **not** learn a prior over its discrete latent codes. Sampling random codebook entries would usually produce meaningless images, since the latent structure is unknown.

To solve this, we **learn a prior over the discrete latent space (Learnable Prior).** So instead of sampling directly from the discrete latent space, we first learn this distribution with another generative model. 

**The process:**

1. **Prepare Data:** Encode the original training data to obtain sequences of discrete latent codes. These sequences serve as the "ground truth" training set for the new model.
2. **Train Prior:** Train a generative model (e.g., PixelCNN ) on these latent code sequences.
3. **Sample:** After training, use the learned prior to sample a new sequence of latent codes. Each latent code corresponds to a specific spatial "patch" of the image.
4. **Decode:** Pass the sampled codes through the VQ-VAE decoder to generate the final new image.

## Autoregressive Models (PixelCNN)

Given a VQ-VAE codebook, our objective is to **learn a distribution** **over sequences of discrete latent codes** (symbols):

$\{z^{(i)} = (z_1^{(i)}, z_2^{(i)}, \ldots, z_T^{(i)})\}_{i=1}^N$

where:

- $z_t^{(i)} \in \{1, \ldots, K\}$

![image.png](Discrete%20AI/image%207.png)

In order to do so, we parameterize a generative model (a **learned prior**) using parameters $\theta$. This defines a parametric distribution over sequences of discrete latent variables:

$p_{\theta_z}(z), \text{ with } z = (z_1, z_2, \ldots, z_T), z_t \in \{1, \ldots, K\}$

Where:

- $T$ denotes the **total sequence length**
- $K$ represents the size of the **discrete vocabulary** (the number of potential values for each symbol).

**AUTOREGRESSIVE MODELLING OF DISCRETE LATENTS**

Estimating the joint probability distribution of a set of random variables $p(z) = p(z_1, z_2, \ldots, z_T)$ directly is often more complex due to the high dimensionality of the data.

To make this problem manageable, we apply the **chain rule of probability**, which allows us to rewrite the *joint distribution* as a *product of conditional distributions*. In this way, the prior  $p_{\theta_z}(z)$ over discrete latent codes can be learned using an **autoregressive factorization**:

**$p_{\theta_z}(z) = \prod_{t=1}^T p_{\theta_z}(z_t | z_{<t})$**

This means that each latent code $z_t$ is predicted conditioned on all previous ones in the sequence.

<aside>
⚙

For a sequence $z = (z_1, z_2, z_3)$, this implies:

- $p(z_1)$ depends on no prior context.
- $p(z_2)$ is conditioned on $z_1$.
- $p(z_3)$ is conditioned on $z_1, z_2$.

So we can rewrite the joint distribution as the product of conditional distribution:

$p(z_1, z_2, z_3) = p(z_1) \cdot p(z_2 | z_1) \cdot p(z_3 | z_1, z_2)$ 

</aside>

**CHAIN RULE OF PROBABILITY (BAYESIAN FACTORIZATION)**

To justify the autoregressive factorization, we start from the fundamental **product rule**, which states that a joint distribution can be decomposed into a **prior** and a **conditional probability**:

$p(A | B) = \frac{p(A, B)}{p(B)} \Rightarrow p(A, B) = p(A | B) \cdot p(B)$

Our goal is to apply this rule to the joint distribution of a sequence of $T$ random variables, $z = (z_1, z_2, \ldots, z_T)$. We can achieve the autoregressive factorization by recursively applying the product rule:

$\begin{aligned}
p(z_1, z_2, \ldots, z_T) &= p(z_T \mid z_1, \ldots, z_{T-1}) \cdot p(z_1, \ldots, z_{T-1}) \\
&= p(z_T \mid z_{<T}) \cdot \underbrace{p(z_{T-1} \mid z_{<T-1}) \cdot p(z_1, \ldots, z_{T-2})}_{\text{expanding the remaining joint}} \\
&= p(z_T \mid z_{<T}) \cdot p(z_{T-1} \mid z_{<T-1}) \cdot \ldots \cdot p(z_2 \mid z_1) \cdot p(z_1) \\
&= \prod_{t=1}^T p(z_t \mid z_{<t})
\end{aligned}$

The derivation follows three logical steps:

1. **Isolation:** We isolate the last variable, $z_T$, and condition it on all preceding variables.
2. **Recursive Step:** We repeat this process for the remaining joint term $z_1 \dots z_{T-1}$, "peeling off" one variable at a time.
3. **Termination:** We repeat this exactly $T$ times until we reach the initial prior, $p(z_1)$.

This mathematical result is the definition of an **Autoregressive Generative Model**. It implies that the joint distribution can be written as a product of conditionals, where every random variable is conditioned **only on its predecessors**, not on future values.

Each latent variable $z_t$ is a discrete symbol from a codebook of size $K$.  Therefore, each conditional probability can be modeled as a **categorical distribution** over the $K$ possible codebook entries:

$p_{\theta_z}(z_t | z_{<t}) = \text{Cat}(z_t; \pi_{\theta_z}(z_{<t}))$

where:

- $z_t \in \{1, \ldots, K\}$ is the index of the selected codebook entry.
- $\pi_{\theta_z}(z_{<t}) \in \Delta^{K-1}$ is the predicted probability vector over each entry of the codebook, which can be parameterized by a neural network.

![image.png](Discrete%20AI/image%208.png)

We can visualize this distribution as a **histogram** with $K$ bins. By sampling from this histogram, we determine the specific index for the next latent code $z_t$.

<aside>
⚠️

Common choices for $p_{\theta_z}$:

- **Pixel Recurrent Neural Networks** a recurrent neural network adapted to image data, where each pixel (or latent symbol) is predicted sequentially based on all previously generated ones.
- **Transformer**  sequence models based on self-attention, which take the sequence of previous latent codes as input and output a categorical distribution over the $K$ possible symbols at each position.
- **PixelCNN**
</aside>

### PixelCNN

The general idea of Pixel CNN is that we can model the prior over discrete latent variables $z \in \{1, \ldots, K\}^{H \times W}$ using a **2D autoregressive model** implemented with a modified Convolutional Neural Network.

The joint distribution is factorized as:

$$
p_{\theta_z}(z) = \prod_{i=1}^H \prod_{j=1}^W p_{\theta_z}(z_{i,j} | z_{<i,j})
$$

where:

- $z_{i,j}$ is the pixel at row $i$ and column $j$.

![image.png](Discrete%20AI/image%209.png)

- $z_{<i,j}$ denotes all pixels that come **before** position $(i,j)$ in raster-scan order (row by row, left to right).

Each conditional term $p(z_{i,j} | z_{<i,j})$ is modeled as a **categorical distribution** predicted by a convolutional neural network.

**`Problem`**  The model must respect the autoregressive ordering: when predicting pixel $(i,j)$, it must only access **past** pixels, never **future** ones. However, a standard convolutional kernel naturally looks at both past and future pixels in its receptive field, which breaks causality.

**`Solution`** PixelCNN enforces autoregressive structure using **masked convolutional kernels**. As we will see later, a binary mask is applied to each convolutional filter so that all weights corresponding to **future** pixels are set to zero. This guarantees that the network is strictly causal during training and sampling.

**INFERENCE**

During inference, an autoregressive model such as PixelCNN generates an image *sequentially*, sampling one value at a time from the learned discrete probability distribution:

1. **Initialization:** Start with an empty image (or latent grid).
2. **Sequential Generation:** For each position $(i, j)$ in the image grid the model perform a forward pass through the PixelCNN to obtain the conditional distribution:
    
    $p_{\theta_z}(z_{i,j} | z_{<i,j}) = \text{Cat}(z_{i,j}; \pi_{\theta_z}(z_{<i,j})), \text{ where } \pi_{\theta_z}(z_{<i,j}) = \text{PixelCNN}(z_{<i,j}; \theta_z)$
    
    A new value is then sampled from this distribution:
    
    $z_{i,j} \sim p_{\theta_z}(z_{i,j} | z_{<i,j})$
    
3. **Update** The newly sampled value is inserted into the grid, and the process continues in raster-scan order (row by row and left to right).

<aside>
⚙

**Illustrative Example and Computational Cost:**

- For the **first pixel ($z_1$)**, the model learns a probability distribution that is not conditional on any previous data. Let us assume we sample a value, **128**.
- This value (128) is fed back into the model to compute the distribution for the **second pixel**, conditional on the first. Suppose we sample **100**.
- For the **third pixel**, the network evaluates the probability distribution conditional on both previous values (128 and 100).

This sequential dependency means that to generate an image of size $10 \times 10$, we must perform **100 separate forward passes** during inference.

In this example: *Each red square represents a forward pass and a sampling step.*

![Screenshot 2025-12-04 165406.png](Discrete%20AI/Screenshot_2025-12-04_165406.png)

![Screenshot 2025-12-04 165619.png](Discrete%20AI/Screenshot_2025-12-04_165619.png)

![Screenshot 2025-12-04 165626.png](Discrete%20AI/Screenshot_2025-12-04_165626.png)

![Screenshot 2025-12-04 165633.png](Discrete%20AI/Screenshot_2025-12-04_165633.png)

The process is repeated **until the entire grid has been filled**, meaning that all positions have been generated (all squares turn red).

</aside>

**`Note`** This sequential sampling procedure is inherently slow, since each pixel depends on all previously generated ones. As a consequence, sampling time scales linearly with the number of pixels, and pixel generation cannot be parallelized.

**`Conclusion`** This method represents a clear trade-off: it provides a robust way to model discrete sequential distributions, but it is computationally expensive. It is a viable solution for short sequences or small grids, but it becomes impractical for high-resolution data due to the prohibitive inference time.

**CAUSAL CONVOLUTIONS**

So we said that in autoregressive models, we must ensure that the prediction at position $(i,j)$ depends **only** on previously generated positions. This prevents information leakage from future pixels (i.e., the model must not “see the future”).

To enforce this constraint, PixelCNN uses **masked convolutions**, which block access to all pixels that come after $(i, j)$ in the chosen ordering.

This is implemented by **element-wise multiplying** the convolutional kernel with a **hard-coded binary mask**. All weights that correspond to future positions are set to zero before each forward pass.

In practice, PixelCNN does not rely on a single mask, for 2D data (images), we have two masking strategies:

![image.png](Discrete%20AI/image%2010.png)

- **Mask type A:** is applied **only in the first convolutional layer**. It strictly excludes the **current pixel** $(i, j)$ from the calculation. This is necessary because $(i, j)$ is the target value we are trying to predict; if the model were allowed to see it in the first layer, the task would become trivial (identity mapping) rather than predictive.
- **Mask type B:** is used in **all subsequent layers**. Using the center pixel here is safe because the input to these layers is not the raw image, but a feature map. Therefore, allowing connection to $(i,j)$ allows the network to process and propagate this contextual information without violating the autoregressive constraint.

![Screenshot 2025-12-04 174359.png](Discrete%20AI/Screenshot_2025-12-04_174359.png)

**TRAINING**

PixelCNN is trained to maximize the **Log-Likelihood** of the training data, which is equivalent to minimizing the **Negative Log-Likelihood (NLL)**:

$L(\theta_z) = - \sum_{i,j} \log p_{\theta_z}(z_{i,j} | z_{<i,j})$

Since the model predicts a **categorical distribution** for each position (e.g., a softmax over $K$ categories, one for each line in the codebook), the training loss corresponds to the **standard cross-entropy** between the **predicted distribution** and the **ground-truth class**.

In practice, for each pixel $(i, j)$, the model outputs a probability vector of size $K$, whose entries sum to 1.  Cross-entropy is applied independently at each location, encouraging the network to assign high probability to the *correct* class. 

**`Problem`** Sampling from PixelCNN is slow, as it requires a **sequential forward pass** for each pixel due to the autoregressive dependency. If we were to train the model in the same way—using its own generated samples to predict the next step—training would be computationally infeasible.

**`Solution` Teacher Forcing**

 During training, we already possess the ground-truth image. Instead of feeding the model its own sampled predictions from the previous step (as done in inference ), we feed the **actual ground-truth pixels** from the training data in the network. 

This simple shift changes the computational paradigm:

- **Parallelization:** Since the input (the ground-truth image) is fully known, we can predict the distributions for *all* pixels simultaneously.
- **Single Pass:** The entire grid is processed in a **single forward pass**, rather than one pixel at a time.

<aside>
⚠️

This works even though the input contains “future” pixels, because the **masked convolutions automatically prevent the model from accessing invalid information**.
As the convolutional filters slide across the image, the masks zero out all entries that correspond to future positions, ensuring the autoregressive constraint is respected.

</aside>

**PSEUDOCODE**

**Training (single forward pass)**

**Sampling (autoregressive)**

![image.png](Discrete%20AI/image%2011.png)

**LIKELIHOOD AND ANOMALY DETECTION**

PixelCNN models the exact data likelihood:

$\log p(x) = \sum_{i,j} \log p(x_{i,j} | x_{<i,j})$

This allows us to assign a **score probability** (or log-likelihood) to any image:

- If the input truly comes from the training distribution (digits 1–4), this likelihood will be relatively **high**.
- If the input is unusual or unseen (e.g., a digit 5), the pixel-level probabilities will be small, and the overall likelihood will be **very low**.

This means PixelCNN can serve as a method for **measuring how likely an input is with respect to the training dataset**. By thresholding this likelihood, we obtain a simple but effective decision rule for identifying unexpected or out-of-distribution samples.

**Applications** 

- **Anomaly detection:** Unusual images tend to have low likelihood under the model.
- **Novelty detection:** Detect out-of-distribution (OOD) samples by thresholding likelihood scores.

<aside>
🧠

**SUMMARY: VQ-VAE AND PIXELCNN**

**VQ-VAE: Vector Quantized Variational Autoencoder**

- Learns a discrete latent space using vector quantization.
- Encoder maps inputs to discrete latent codes (via a codebook).
- Decoder reconstructs the input from latent codes.
- Enables efficient and compressible representations.

But VQ-VAE doesn’t have a mechanism for sampling new content. We cannot simply sample random codes from the codebook because we do not know their valid arrangement (the prior). To solve this, we combine VQ-VAE with an autoregressive model.

**PixelCNN: Autoregressive Prior over Latents**

- Models the joint distribution over latent codes.
- Trained on the discrete latent space from the VQ-VAE encoder.
- Allows generation of new latent code grids → decoded into images.

**Combined Pipeline**

1. Train VQ-VAE ⇒ Learn discrete latent space.
2. Train PixelCNN on latent codes ⇒ Learn generative prior.
3. Sample from PixelCNN and decode via VQ-VAE decoder.
</aside>

## Gumbel Softmax

We still face the same issue that appeared at the **beginning of the lesson**. When we sample from a distribution—e.g., by taking an **argmax** over logits during generation—the backward pass **fails,** because **argmax is non-differentiable** therefore gradients cannot flow through the sampling step.

PixelCNN suffers from exactly this problem during sampling: once we pick the most likely index from the histogram (the categorical distribution), this discrete choice **breaks backpropagation**.

**STRAIGHT-THROUGH ESTIMATOR (STE)**

In VQ-VAE, the trick used to handle the non-differentiable quantization step is the Straight-Through Estimator (STE). It works as follows:

- **Forward pass:** use the discrete codebook index selected by nearest neighbor search.
- **Backward pass:** ignore the discrete choice and **copy the gradient** from $z_q$ directly to $z_e$, as if the quantization step were the identity function.

This works for VQ-VAE, but it is **not suitable for sampling problems** like those in PixelCNN, where the output is a **categorical distribution**, not a single codebook vector.

**GUMBEL-SOFTMAX**

The **Gumbel–Softmax** trick provides a differentiable way to sample from a categorical distribution **inside a neural network**, enabling backpropagation through the sampling operation.

Instead of performing a non-differentiable **`argmax`** or standard categorical sampling, the encoder produces **logits** $\pi$ over the $K$ possible categories. These logits are transformed into a *soft* sample using the Gumbel–Softmax relaxation:

$y_k = \frac{\exp((\log \pi_k + g_k)/\lambda)}{\sum_j \exp((\log \pi_j + g_j)/\lambda)}$

Where:

- $\pi_k$ is the predicted probability of category $k$,
- $g_k$ are i.i.d. samples from a **Gumbel(0,1)** distribution,
- $\lambda$ is a **temperature parameter:**
    - $\lambda \to 0$, the samples approach a hard categorical distribution (one-hot vector).
    - $\lambda \to \infty$, the samples approach a uniform distribution.
    
    <aside>
    ⚙
    
    Consider a scenario with Logits $\pi = [2.0, 0.5, 1.0]$ and Gumbel noise $g$. The impact of reducing $\lambda$ is evident:
    
    - **$\lambda = 1.0$ (Smooth):** Output $\approx [0.62, 0.10, 0.28]$. The winner is distinct but the vector is "soft".
    - **$\lambda = 0.5$ (Peaked):** Output $\approx [0.71, 0.22, 0.07]$. The confidence in the first class increases.
    - **$\lambda = 0.1$ (Hard):** Output $\approx [0.98, 0.003, 0.017]$. The vector is virtually identical to a one-hot encoding $[1, 0, 0]$.
    </aside>
    

 This mechanism closely resembles the **reparameterization trick** used in VAEs: the network provides logits (analogous to the mean in a Gaussian), noise is sampled from a known distribution (Gumbel instead of Gaussian), and the two are combined into a differentiable transformation.

**DIFFERENTIABLE APPROXIMATION: FROM ARGMAX TO GUMBEL SOFTMAX**

We can distinguish three approaches to sampling from a categorical distribution:

- **Hard categorical choice**
    
    This involves selecting the class with the highest probability:
    
    $y_k = \text{one-hot}(\arg \max_j \pi_j)$
    
    **`Problem`**  It is **non-differentiable**.
    
- **Softmax:**
    
    This creates a continuous probability distribution, which is differentiable.
    
    $y_k = \frac{\exp(\pi_k/\lambda)}{\sum_j \exp(\pi_j/\lambda)}$
    
    **`Problem`** it's not **sampleable**, it only gives probabilities.
    
- **Gumbel-Softmax (differentiable sampling):**
    
    To achieve **differentiable sampling**, we combine the Softmax function with **Gumbel noise:**
    
    $y_k = \frac{\exp((\log \pi_k + g_k)/\lambda)}{\sum_j \exp((\log \pi_j + g_j)/\lambda)}$
    
    where:
    
    - $g_k \sim \text{Gumbel}(0,1) = -\log(-\log(U)), \quad U \sim \text{Uniform}(0,1)$
        
        This introduces stochasticity while keeping the operation differentiable with respect to the class probabilities $\pi$
        

**APPLICATIONS**

**Conditional computation** refers to architectures where only a subset of the model is activated for each input, enabling sparse and efficient inference.

- It operates by selectively activating only parts of the network at a time.
- A common example is **Mixture of Experts** in large language model architectures.

**Challenge:** Selecting which computation path (e.g., which module or expert) to activate is typically **non-differentiable**:

$\text{select}(i) = \arg\max(\text{scores}_i)$

**Solution: Gumbel-Softmax Trick**

- Provides a **differentiable routing mechanism** by producing a soft (or annealed-hard) one-hot vector.
- Allows the network to learn routing decisions via **gradient descent**, even though the final selection is discrete at inference.