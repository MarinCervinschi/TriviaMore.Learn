# Generative models

## Variational Autoencoders

### **Generative modeling**

**Generative modeling** is a class of machine learning whose goal is to learn the probability distribution $p(x)$ of data points within a high-dimensional space $\mathcal{X}$. By modeling this distribution, generative models can capture the underlying structural patterns of a training dataset $X$.

This enables them to:

- **Sample new, realistic data points** that resemble the original dataset.
- **Support inference**, that is, computing the likelihood of a data point, allowing the model to assess whether it belongs to the learned distribution. This represents a critical feature for **anomaly and novelty detection**.

<aside>
⚙

In practical terms, such as with image data, a successful model assigns **high probability** to data that resembles **real images** and low probability to random noise.

</aside>

**GENERATIVE MODELING AND AUTOENCODERS**

Standard autoencoders, including **denoising** and **contractive autoencoders**, are primarily **discriminative**, ****meaning they focus on learning a mapping that is useful for distinguishing between classes. Although these models can **implicitly capture the structure of the data distribution** $p(x)$ by learning the geometry of the data manifold, they do not explicitly model this distribution. As a result, **sampling new data points from them is difficult**.

**`Solution`** modeling the true distribution $p(x)$ using simpler distribution that we can easily sample from.

### Latent Variable Models

To model a complex distribution $p(x)$ using a simpler one, we  often map the data into a new space $\mathcal{Z}$, known as the **latent space**.

The assumption is that the observed data $x$ (e.g., image pixels) are generated from unobserved variables $z$ that exist implicitly, called **latent variables**. These variables capture **high-level factors** **that explain differences between data points**, providing a simplified, abstract representation of the data. 

<aside>
⚙

**Analogy:** Instead of memorizing every pixel in an image of a face, the model summarizes it using key characteristics like identity, facial expression, head angle, and lighting. These characteristics act as **latent variables**, capturing the essential differences between images in a compact form.

</aside>

<aside>
⚠️

The latent variables $z$ are not manually specified; the model **learns them directly** from the observed data to find the most efficient representation. 

</aside>

Rather than modeling $p(x)$ directly, generative models introduce latent variables $z$ and model the **joint distribution** $p(x, z)$. By applying the **law of total probability** and the **definition of conditional probability**, we can express the marginal distribution of the data as:

$\underbrace{p(x)}_{\text{marginal}} = \underbrace{\int p(x, z) dz}_{\text{law of total probability}} = \underbrace{\int p(x|z) p(z) dz}_{\text{definition of conditional probability}}$

**INTRACTABILITY PROBLEM**

While this provides the foundation for Variational Autoencoders (VAEs), calculating this integral is typically **computationally intractable**, especially in high-dimensional spaces or when the conditional distribution $p(x|z)$ is complex.

To approximate this integral, a common approach is **Monte Carlo estimation.** Instead of calculating the full integral, we approximate it by averaging over a finite set of $K$ samples drawn from a proposal distribution (the prior) $p(z)$:

$p(x) \approx \frac{1}{K} \sum_{k=1}^{K} p(x|z^{(k)}), \quad z^{(k)} \sim p(z)$

**SAMPLING PROCEDURE**

This formulation can be represented using a **graphical model**, where $z$ is the parent node and $x$ is the child node. The generative sampling procedure consists of:

1. **Sampling** $z \sim p(z)$
2. **Generating** $x$ from the conditional distribution $p(x \mid z)$
3. Apply **Monte Carlo approximation**

![image.png](Generative%20models/image.png)

**VARIATIONAL INFERENCE**

Using **Monte Carlo methods** provides some significant challenges because the quality of the approximation depends critically on the choice of latent samples. Most sampled values of $z$ are irrelevant to a specific observation $x$; consequently, random sampling often leads to **poor estimates**.

**Variational Inference** addresses this issue by learning a **proposal distribution $q(z|x)$** such that it approximates well the original $p(z|x)$. Instead of sampling blindly, the model learns a distribution that, given an observation $x$, focuses on the **most plausible latent variables $z$**—that is, those with high probability under the true posterior $p(z \mid x)$.

This is formulated as an **optimization problem** that minimizes the Kullback–Leibler (KL) divergence between the two distributions:

$q^*(z|x) = \arg \min_q D_{KL}[q(z|x) \parallel p(z|x)]$

Minimizing this divergence ensures that the learned distribution $q(z \mid x)$ can effectively replace the true posterior for inference. However, direct minimization is intractable because computing $D_{KL}$ requires knowing the evidence $\log p(x)$, which is generally impossible for complex generative models.

**DERIVING VARIATIONAL OBJECTIVE**

To make the problem tractable, we start from the definition of the KL divergence:

$$
D_{KL}[q(z|x) \parallel p(z|x)] = \int_x q(z|x) \left[ \log \frac{q(z|x)}{p(z|x)} \right] dz
$$

<aside>
⚠️

**`Note`**  The expectation operator $\mathbb{E}$ corresponds to an integral when the variable $z$ is continuous:

$\mathbb{E}_{z \sim q(z|x)}[f(z)] = \int f(z)q(z|x)dz$

</aside>

Expanding the logarithm:

$$
\begin{aligned}
D_{KL}[q(z|x) \parallel p(z|x)]
&= \mathbb{E}_{z \sim q(z|x)}\!\left[ \log \frac{q(z|x)}{p(z|x)} \right] \\
&= \mathbb{E}_{z \sim q(z|x)} \left[ \log q(z|x) - \log p(z|x) \right]
\end{aligned}
$$

By applying **Bayes’ rule**  $p(z|x) = \frac{p(x|z)p(z)}{p(x)}$, we can write:

$\begin{aligned}
D_{KL}[q(z|x) \parallel p(z|x)]
&= \mathbb{E}_{z \sim q(z|x)}\!\left[
    \log q(z|x) - \log \frac{p(x|z)p(z)}{p(x)}
  \right] \\
&= \mathbb{E}_{z \sim q(z|x)}\!\left[
    \log q(z|x) - \log p(x|z) - \log p(z) + \log p(x)
  \right]
\end{aligned}$

Importantly, the term $\log p(X)$ does not depend on $z$, and therefore can be moved **outside the expectation**:

$= E_{z \sim q(z|x)} [\log q(z|x) - \log p(x|z) - \log p(z) ] + \log p(x)$

Rearranging the terms:

$\begin{aligned}
&= \mathbb{E}_{z \sim q(z|x)} \left[
    \log q(z|x) - \log p(z)
  \right]
  - \mathbb{E}_{z \sim q(z|x)} \left[
    \log p(x|z)
  \right]
  + \log p(x) \\
&= \mathbb{E}_{z \sim q(z|x)} \left[
    \log \frac{q(z|x)}{p(z)}
  \right]
  - \mathbb{E}_{z \sim q(z|x)} \left[
    \log p(x|z)
  \right]
  + \log p(x)
\end{aligned}$

Since $E_{z \sim q(z|x)} [\log \frac{q(z|x)}{p(z)} ] = D_{KL}[q(z|x) || p(z)]$, we obtain:

$$
D_{KL}[q(z|x) \parallel p(z|x)] = D_{KL}[q(z|x) \parallel p(z)] - \mathbb{E}_{z \sim q(z|x)}[ \log p(x|z) ] + \log p(x)
$$

**ELBO: THE EVIDENCE LOWER BOUND**

By rearranging the previous equation, we obtain:

$\log p(x) - \underbrace{D_{KL} [q(z|x) \parallel p(z|x)]}_{\substack{\text{KL divergence between the proxy} \\ \text{and the real posterior distributions.}}} = \underbrace{E_{z \sim q(z|x)} [\log p(x|z)] - D_{KL} [q(z|x) \parallel p(z)]}_{\text{ELBO: Evidence Lower BOund}}.$

This is important because It provides a relationship between $p(x)$ (first term of the left side) and the posterior distribution $q(z|x)$. 

- **Right side**
    
    The right-hand side (the ELBO) is something we can optimize via **stochastic gradient descent** given the right choice of $q(\cdot)$. This optimization is the core principle behind **Variational Autoencoders (VAEs)**.
    
- **Left side:**
    
    Since the KL divergence is always non-negative, the ELBO serves as a lower bound on the true log-likelihood: 
    
    $\log p(x) \geq \text{ELBO}$ 
    

**Why Maximizing the ELBO Works?**

Maximizing the ELBO is equivalent to minimizing the KL divergence between the approximate posterior $q(z|x)$ and the true posterior $p(z|x)$.

This is because the $\log p(x)$ depends only on the data and the model parameters — it is **constant** with respect to the variational distribution $q$. Therefore, as we push the ELBO "up" toward $\log p(x)$, the "gap" (the KL divergence) must necessarily shrink. 

![image.png](Generative%20models/image%201.png)

### Variational Autoencoder (VAE)

A Variational Autoencoder (VAE) is an autoencoder trained to **maximize the Evidence Lower Bound (ELBO):**

$$

\underbrace{E_{z \sim q_{\phi}(z|x)}[\log p_{\theta}(x | z)]}_{\text{Reconstruction Term}} - \underbrace{D_{KL} \left( q_{\phi}(z | x) \parallel p(z) \right)}_{\text{Regularization Term}}

$$

where:

- **Reconstruction term**, encourages accurate reconstruction of the input data,
- **Regularization term**, forces the approximate posterior to remain close to a chosen prior distribution $p(z)$, which is typically a Gaussian distribution.

**ARCHITECTURE**

The VAE architecture maintains the **encoder–decoder** structure of standard autoencoders. However, instead of mapping the input to a single deterministic latent vector, the encoder outputs the parameters — typically the **mean and variance** — of a **latent distribution**.

![image.png](Generative%20models/image%202.png)

**Components:**

- **Encoder network** $q(z | x)$: Given an input $x$, the encoder estimates the posterior distribution $p(z|x)$ over latent variables.  In practice, this is usually modeled as a multivariate Gaussian:
    
    $*x \to f_{ENC}(x; \theta) = [\mu(x), \sigma^2(x)] \quad \text{s.t.} \quad \mathcal{N} (z|\mu(x), \sigma^2(x))*$
    
    The network outputs two vectors, $\mu(x)$ and $\sigma^2(x)$, representing the parameters of the approximate posterior.
    
    <aside>
    
    **`In practice`** the encoder projects the data into the space of latent variables.
    
    </aside>
    
- **Decoder network** $p(x | z)$: The decoder reconstructs the input data from a latent sample $z$. A latent vector is drawn from the approximate posterior $z \sim q(z \mid x)$ and the decoder produce a reconstruction $\hat{x}$ that aims to match the original input $x$:
    
    $\hat{x} = g_{\text{DEC}}(z; \theta) \quad \text{where} \quad z \sim \mathcal{N} (z|\mu(x), \sigma^2(x))$
    
    <aside>
    
    **`In practice`** the decoder network **learns the conditional distribution** $p_\theta(x \mid z)$ and tries to reconstruct the input as accurately as possible.
    
    </aside>
    
    To maximize the log-likelihood $\log p(x \mid z)$, the model typically minimizes the **Mean Squared Error (MSE)** between **input** and **reconstruction**.
    

**REGULARIZATION**

The **Regularization term** in the ELBO objective is defined as:

$D_{KL}[q(z|x) \parallel p(z)]$

his term measures how much the approximate posterior $q(z|x)$ produced by the encoder diverges from the **prior distribution** $p(z)$. 

The prior represents our prior belief about the distribution of the latent variables $z$ before observing any data. It is typically "hand-crafted" as a **Standard Multivariate Gaussian**:

$p(Z) = \mathcal{N}(0, I)$

<aside>
⚠️

**Why use a Gaussian Prior?**
Choosing a Gaussian distribution for $p(z)$ is a standard practice in VAEs for several reasons:

- **Computational Efficiency:** If both the approximate posterior $q(z \mid x)$ and the prior $p(z)$ are Gaussian, the KL divergence has a **closed-form solution**. This allows it to be computed exactly using a fixed formula, without numerical approximation or sampling, making training efficient.
- **Well-Behaved Latent Space:** A standard Gaussian encourages the latent space to be **centered, compact, smooth, and connected**, preventing the model from spreading latent codes infinitely apart.
- **Simplicity:** A Gaussian prior provides a simple, **data-independent** reference distribution, allowing the model to learn complex data structure through the encoder rather than through the prior.
</aside>

**REPARAMETRIZATION TRICK**

A critical implementation challenge in VAEs arises from the sampling operation:

$z \sim \mathcal{N}(\mu(x), \sigma^2(x))$

Sampling is stochastic and therefore **non-differentiable**, which prevents gradients from flowing through the latent layer to the encoder.

To address this problem, we use the **reparameterization trick**. Instead of sampling directly from the approximate posterior:

![image.png](Generative%20models/image%203.png)

1. **Sample Noise:** Draw $\epsilon$ from a standard normal distribution of the same dimension as $z$:
    
    $\epsilon \sim \mathcal{N} (z|0, I_n)$.
    
2. **Reparametrize**: We then compute $z$ using the parameters provided by the encoder:
    
    $z = \mu(x) + \epsilon * \sigma(x)$
    

This transformation produces a sample that is statistically equivalent to sampling from the original distribution $\mathcal{N}(\mu, \sigma^2)$, but now **gradients can flow through $\mu(x)$ and $\sigma(x)$** during backpropagation.

**Advantages:**

- The stochasticity is isolated in $\epsilon$, whose distribution does not depend on the encoder parameters, allowing standard gradient-based optimization to train the model efficiently.

**SAMPLING IMAGES AFTER TRAINING**

Once a Variational Autoencoder has been successfully trained, the learned posterior $q(z \mid x)$ closely matches the prior $p(z)$. This allows new data points to be generated by **sampling directly from the prior**. 

A sampled latent vector $z \sim p(z)$ is then fed into the decoder, which transforms it into a generated image. This process works because, during training, the decoder has learned to reconstruct inputs from latent variables that were explicitly encouraged to follow the prior distribution. As a result, any latent sample drawn from $p(z)$ can be decoded into a realistic new data point, enabling the VAE to generate entirely novel images that resemble the training data.

## Generative Adversarial Networks

**WHY GANs?**

**Generative Adversarial Networks (GANs)** were introduced to address limitations of likelihood-based generative models such as VAEs.

**Limitations of VAEs / VQ-VAEs**

VAEs often generate **blurry images** mainly because:

- **Likelihood-based training (MSE ):** The VAE decoder is typically trained by minimizing pixel-wise MSE. If the model is uncertain about the exact position of details, it averages the possibilities, resulting in a blurry line.
- **Strong regularization (KL term):** The latent posterior is forced to match a simple prior (e.g., standard normal). If this regularization is too strong, the latent representation may lose important information (posterior collapse).

**GANs: A Different Approach**

Unlike VAEs,  GANs do **not require an explicit density function.** Instead of modeling the data distribution directly, they learn a generator $G(z)$ that maps noise $z \sim p(z)$ directly to real data distribution via adversarial training.

**ADVERSARIAL TRAINING NETWORKS: ARCHITECTURE**

Generative Adversarial Networks (GANs) are another generative approach, based on **two networks** that **compete with each other**, in a game theoretic scenario:

- The **generator network $G$**
    
    The generator’s goal is to produce realistic samples that can "fool" the discriminator. During training, its objective is to maximize the probability of the discriminator **making a mistake.** In order to do so:
    
    - It takes random noise $z$ from a simple distribution (e.g., Uniform or Gaussian).
    - Transform this noise $z$ into a data sample $G(z)$ in the target domain (such as an image).
- The **discriminator network** $D$
    
    The discriminator acts as a binary classifier that evaluates the **authenticity of the data**. It receives samples from both the **training set** and the **generator**, then it emits a probability value $D(x) \in [0, 1]$ indicating whether the input image is "real" (from the data distribution) or "fake" (from the generator).
    

![image.png](Generative%20models/image%204.png)

**LOSS**

The Generator ($G$) and Discriminator ($D$) are trained jointly using a **Minimax Objective Function:**

$$
\min_{\theta_G} \max_{\theta_D}
\left[
\underbrace{E_{x \sim p_{data}} \log D_{\theta_D} (x)}_{\substack{\text{log prob of D predicting that} \\ \text{real-world data is genuine}}}
+
\underbrace{E_{z \sim p(z)} \log(1 - D_{\theta_D} (G_{\theta_G} (z)))}_{\substack{\text{log prob of D predicting that} \\ \text{G’s generated data is not genuine}}}
\right] 
$$

- The **Discriminator** ($\theta_d$) wants to maximize the objective such that $D(x)$ is close to 1 (real) and $D(G(z))$ is close to 0 (fake):
    - The **first term** is the expectation over real data samples $x \sim p_{\text{data}}$:
        
        $\mathbb{E}_{x \sim p_{\text{data}}} [\log D_{\theta_d}(x)]$
        
        Maximizing this term encourages the discriminator to assign high probability to real data.
        
    - The **second term** is the expectation over generated samples $z \sim p(z)$:
        
        $\mathbb{E}_{z \sim p(z)} [\log(1 - D_{\theta_D}(G_{\theta_G}(z)))]$
        
        Maximizing this term encourages the discriminator to assign low probability to generated (fake) samples.
        
    
    $D$ is trained to maximize its classification accuracy across both real and synthetic datasets.
    
- The **Generator** ($\theta_g$) aims to minimize the objective such that $D(G(z))$ is close to 1 as possible (effectively "fooling" the discriminator).
    
    Since the generator only appears in the **second term**, it seeks to minimize $\log(1 - D_{\theta_D}(G_{\theta_G}(z)))$. Like that it is forced to produce increasingly realistic samples.
    

**TRAINING DYNAMICS**
The two optimizers operate on separate sets of parameters, even though the overall loss depends on both. Training typically **alternates** between two steps:

1. **Gradient ascent on discriminator:**
    
    $\max_{\theta_d} \ E_{x \sim p_{data}} \log D_{\theta_d} (x) + E_{z \sim p(z)} \log(1 - D_{\theta_d} (G_{\theta_g} (z)))$
    
    **Updating the Discriminator** to improve its classification.
    
2. **Gradient descent on generator:**
    
    $\min_{\theta_g} \ E_{z \sim p(z)} \log(1 - D_{\theta_d} (G_{\theta_g} (z)))$
    
    **Updating the Generator** to produce samples that better evade the discriminator’s detection.
    

Convergence is reached when the generator produces data indistinguishable from real data, and the discriminator cannot reliably differentiate between the two.

**TRAINING GANs: VANISHING GRADIENTS PROBLEM**

**`Problem`** If the discriminator is too good and the generator is not yet producing realistic samples, the generator will receive very small gradients, leading to **vanishing gradients**.

In the original GAN formulation, the generator minimizes:

$\min_{\theta_g} \ E_{z \sim p(z)} \log(1 - D_{\theta_d} (G_{\theta_g} (z)))$

If the discriminator correctly classifies fake samples, then: 

- $D_{\theta_d} (G_{\theta_g} (z))$ is close to 0
- $\log(1 - D_{\theta_d} (G_{\theta_g} (z)))$ is close to 0,

and the gradient of this term becomes **very small**. As a result the generator fails to learn because it receives almost no useful feedback to update its parameters.

![image.png](Generative%20models/image%205.png)

**`Solution`** To address this, we modify the generator’s objective. Instead of minimizing the probability of the discriminator being correct, we **maximize the probability of the discriminator being wrong**.

We switch the generator's objective to:

$\max_{\theta_g} \ E_{z \sim p(z)} \log(D_{\theta_d} (G_{\theta_g} (z)))$

This alternative formulation provides **stronger gradients** even when the discriminator performs well, improving training stability:

By receiving stronger feedback in the early stages of training, the generator can more quickly learn to produce samples that resemble the real data distribution.

![image.png](Generative%20models/image%206.png)

**TRAINING ALGORITHM**

***Algorithm 1*** Minibatch stochastic gradient descent training of generative adversarial nets. The number of steps to apply to the discriminator, $k$, is a **hyperparameter**. 

![image.png](Generative%20models/image%207.png)

It is common to perform multiple discriminator updates per generator update (e.g., $k$ discriminator steps and 1 generator step). 

**PROS AND CONS**

- **Pros:**
    - Can utilize power of backpropagation
    - The loss function is learned instead of being hand selected
    - No MCMC needed
- **Cons:**
    - Hard to train Trickier / more unstable to train
    - Need to manually babysit during training
    - No evaluation metric, so it’s hard to compare with other models

## Deep Convolutional GANs

Adversarial training is inherently unstable as it requires maintaining a delicate balance between two competing networks. **DCGANs** were the first architecture to replace fully connected layers with **Convolutional Neural Networks (CNNs)** in both the generator and discriminator. This significantly improved image quality and training stability.

A key property of DCGANs is the **smooth interpolation** in the latent space: moving between random latent vectors produces gradual and realistic transitions in the generated output.

![image.png](Generative%20models/fb9f3b31-08be-4fa9-88f5-589126177369.png)

**SEMANTIC LATENT SPACE ARITHMETIC**

Latent vectors in DCGANs exhibit **semantic properties**. Moving along specific directions in the latent space corresponds to meaningful transformations, such as changing a subject's gender, adding glasses, or modifying facial attributes (e.g., "Man with glasses" - "Man" + "Woman" = "Woman with glasses").

![image.png](Generative%20models/image%208.png)

**IMPROVING STABILITY: LSGAN AND WGAN**

To prevent one network from dominating the other and to mitigate gradient saturation, alternative loss functions were proposed:

1. **Least Squares GAN (LSGAN)** Replaces the binary cross-entropy loss with a least squares objective, which provides a smoother gradient and penalizes samples that are far from the decision boundary.
2. **Wasserstein GAN (WGAN):** Uses the **Wasserstein distance** (Earth Mover's distance) instead of the original Jensen-Shannon divergence. This provides more stable gradients and requires enforcing **Lipschitz continuity** (often via weight clipping or gradient penalty).

**SPECIALIZED ARCHITECTURES**

**CycleGAN: unpaired image-to-image translation**

CycleGAN is designed for domains where paired training data (e.g., the exact same photo in "summer" and "winter" versions) is unavailable. It employs two generators ($G_{A \to B}$, $G_{B \to A}$) and two discriminators.

The core principle is **Cycle Consistency:**

$x \to G_{A \to B}(x) \to G_{B \to A}(G_{A \to B}(x)) \approx x$

This ensures that an image translated from domain A to B and back to A reconstructs the original input, preventing the model from losing the structural content of the image.

![image.png](Generative%20models/image%209.png)

**Progressive GAN**

This architecture starts by training on very low-resolution images (e.g., 4x4) and **progressively adds layers** to both the generator and discriminator to handle higher resolutions. This approach is essential for generating very high-resolution outputs (e.g. 1024x1024).

![image.png](Generative%20models/image%2010.png)

**StyleGANs** 

StyleGAN introduces a generator architecture based on "style" to allow fine-grained control over image attributes.

Instead of using the latent vector $z$ directly, it is transformed into a style vector $w$ via an MLP. This **disentangles** the latent space, ensuring that similar vectors don't overwrite unrelated features.

The style vector $w$ is translated into scale ($y_s$) and bias ($y_b$) parameters that normalize the feature maps:

$\text{AdaIN}(x, y) = y_s \frac{x - \mu(x)}{\sigma(x)} + y_b$

Uncorrelated Gaussian noise is added to each convolutional layer to generate fine details like pores, hair follicles, or freckles without affecting the global structure.

![image.png](Generative%20models/image%2011.png)