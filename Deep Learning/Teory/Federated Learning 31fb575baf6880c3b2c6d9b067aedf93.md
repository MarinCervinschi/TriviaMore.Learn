# Federated Learning

## Introduction

Previously, we explored scenarios where **tasks are distributed over time,** such as Task 1, Task 2, and Task 3 appearing sequentially. In that setting, knowledge is acquired progressively as time passes.

We now shift our focus to a **spatial distribution**. Rather than appearing sequentially in time, tasks are distributed across **different physical locations** or **devices** simultaneously.

**CENTRALIZED MACHINE LEARNING**

In traditional machine learning, we usually assume a **centralized setting**, where:

- **Unified Data Storage:** All data is stored in a single location, giving the model access to the entire dataset.
- **Single Model, Combined Dataset:** A single global model is trained on this unified dataset. The training algorithm, the model, and its **weights** are all located on the same machine.
- **Iterative Optimization (SGD):** The model’s parameters are updated iteratively using Stochastic Gradient Descent, epoch after epoch, until convergence.

In this architecture, **data, algorithms, and model parameters** are consolidated at a single point in the network. While this setup is relatively simple to manage and avoids the complexities of distributed systems, it presents several limitations in real-world applications.

**PROBLEMS**

- **Data Privacy and Confidentiality**
    
    Handling sensitive data (e.g., medical records or personal communications) creates some privacy concerns. In many cases, regulations require that:
    
    - **Local Residency:** Data must remain on the device where it was collected.
    - **Local Processing:** Computations, including model training, must occur locally.
    
    Because of legal regulations (such as **GDPR** or **HIPAA**), sending raw sensitive data to a central server is often restricted.
    
- **Infrastructure and Reliability**
    - **Data Transfer Costs:** Transferring large datasets from edge devices to a central server requires high bandwidth and can be expensive or impractical.
    - **Single Point of Failure:** Centralized systems depend on a single central node. If it fails or the network connection is lost, the whole system may stop working.
- **Underutilization of Edge Resources**
    
    Centralized training relies only on the computational power of the central server, ignoring the capabilities of modern **edge devices** (such as smartphones or embedded systems) that could perform part of the computation locally.
    

**SOLUTION: FEDERATED LEARNING**

**Federated Learning** is a machine learning paradigm designed to address the limitations of centralized systems. Instead of collecting data in a single location, a **central server** coordinates a set of **clients**, such as mobile devices or organizations, to collaboratively train a model.

Each client contributes to the global model by sending **learned knowledge**, typically in the form of **model parameters or weights**, back to the server. In this way, sensitive raw data remains on the original device and is never shared.

By shifting computation to the **edge of the network**, Federated Learning improves both **data privacy** and **operational efficiency**.

**FORMULATION**

- **Client-side** Each client $i$ holds a private dataset $D_i$. The objective is to obtain the set of weights $\theta_i$ to minimize the loss using only its own dataset:
    
     $\min_{\theta_i} E_{(x,y) \sim D_i} L(f_{\theta_i}(x), y)$ 
    
- **Server-side** The server does not have access to the data; it only receives **model parameters** from the clients. Its objective is to obtain a global model that minimizes the average loss across the datasets of all $N$ clients:
    
     $\min_{\theta} \frac{1}{N} \sum_{i=1}^N E_{(x,y) \sim D_i} L(f_{\theta}(x), y)$ 
    

**APPLICATION SCENARIOS**

- **Cross-device Federated Learning:**
    
    A large number of user devices (e.g., smartphones or IoT devices) each hold small amounts of data and collaboratively contribute to training a shared model.
    
    - **Example:** training a mobile keyboard prediction model directly on users’ phones without uploading their keystrokes.
- **Cross-silo Federated Learning:**
    
    A smaller number of organizations (or *silos*) each possess richer local datasets and collaborate to train a common model.
    
    - **Example:** multiple hospitals jointly train a diagnostic model without sharing patient records.

**WEIGHT AVERAGING & MODEL MERGING**

1. The server maintains an **initialized global model**, while the data remains distributed across the clients.
2. The server sends the global model to a subset of selected clients.
3. Each client performs **local training** on its private dataset for one or more epochs.

![image.png](Federated%20Learning/image.png)

1. After local training, clients send their **model updates** (e.g., gradients or updated weights) back to the server.
2. The server **aggregates** these updates, typically by averaging them, to improve the global model. The goal is to obtain a model that performs well **across all client datasets**, even though the server never accesses the raw data.
3. The updated global model is then sent back to the clients and used as the **initialization for the next training round**.

These steps are repeated until the model converges.

**KEY CHALLENGES in FL**

- **Data heterogeneity (non-IID data)**: Clients often have data drawn from different distributions. This heterogeneity can cause **client drift** and may reduce the accuracy of the global model.
- **Partial participation**: In each training round, only a subset of clients participates. The set of active clients can change over time due to **stragglers** or **device dropouts**.
- **Communication overhead**: Network capacity is limited, so reducing the **size and frequency of model updates** is essential for maintaining efficiency.
- **Privacy and security**: Although raw data stays local, shared model updates may still leak information.

**FAMILIES OF FL APPROACHES**

- **Baseline Method:** *Federated Averaging (FedAvg)* – simple periodic averaging of local models to update the global model.
- **Client-side Regularization:** introduces constraints or adjustments during local training to reduce divergence across clients (e.g. **FedProx**, **SCAFFOLD**)
- **Server-side Modifications:** Adjustments applied at the server to improve convergence of the global model (e.g. **Server Momentum**, **GradMA**, **Fisher-Averaging**)
- **Knowledge Sharing:** Exchanges intermediate knowledge instead of raw model parameters (e.g. distillation-based methods, **FedProto**)
- **Personalization:** Tailors models to individual clients through fine-tuning or client-specific updates.

## Federated Averaging (FedAvg)

**Federated Averaging (FedAvg)** is the foundational algorithm for Federated Learning. In this iterative process, a central server coordinates **multiple rounds of training** across a subset of clients. Each round, clients receive the global model, train it locally, and send only the updated parameters back to the server.

**ALGORITHM**

The process follows these steps:

1. **Initialize**: Server sets initial global model parameters $w_0$.
2. **Iterative Rounds:** For each round $t = 1, 2, \ldots, T$:
    - Server selects a subset $C_t$ of available clients.
    - Server sends the current model $w_{t-1}$ to all clients in $C_t$.
    - Each client $i \in C_t$ initializes the model $w_{t-1}$ and performs local training (e.g., $E$ epochs of SGD on its local dataset) to obtain an updated model $w_i^t$.
    - Clients send its model’s weights $w_i^t$ (or the model difference) back to the server.
    - Server aggregates the weights to form the new global model:
        
         $w^t \leftarrow \sum_{i \in C_t} \frac{n_i}{\sum_{j \in C_t} n_j} w_i^t$ 
        
        - $w_i^t$ Model parameters from client $i$ at round $t$.
        - $n_i$: Number of training samples on client $i$.
        - $\sum_{j \in C_t} n_j$ Total number of samples across all participating clients in the current round.
        
        This aggregation corresponds to a **weighted average**, where clients with **larger datasets contribute more** to the global model.
        

**STRENGTHS AND LIMITATIONS**

**FedAvg** has become the de facto baseline for federated learning experiments due to its practical advantages. It is simple to implement and reduces communication overhead by performing multiple local SGD steps per round instead of requiring frequent synchronization.

FedAvg performs well when client data is **Independently and Identically Distributed (IID).** However this assumption is frequently violated in real-world federated settings. When data is **non-IID**, local models can diverge significantly, a phenomenon known as **client drift**. This divergence can degrade both the **convergence speed** and the **final accuracy** of the global model.

![image.png](Federated%20Learning/image%201.png)

<aside>
⚙

One common form of non-IID data is **label distribution skew**, where different clients observe entirely different subsets of classes:

- **Client 1:** Observes only classes 1 and 2.
- **Client 2:** Observes only classes 3 and 4.
- **Client 3:** Observes only classes 5 and 6.
</aside>

**POSSIBLE IMPROVEMENTS**

To mitigate these issues, researchers have developed various enhancements to the original FedAvg algorithm. These approaches can be roughly categorized into 3 families:

- **Server-Side Approach**: The server employs more powerful ways, instead of simply using the average, to aggregate the parameters.
- **Client-Side Approach**: Each client has some additional terms, e.g. regularization terms, that constrain their objective or their weights to force consistency w.r.t. the server.
- **Prototype-based Approach**: Constraints are forced in feature space instead of weight space, in order to obtain a good trade-off between consistency among clients and specialization on local-datasets.

## Server-side Approach

Standard parameter averaging (such as **FedAvg**) treats every weight dimension as equally informative. While this is effective for **i.i.d. data**, it can be sub-optimal under **data heterogeneity** (non-i.i.d. settings).

An improved server-side strategy **weights the contribution of each model parameter** according to its importance for each client. Instead of relying only on dataset size, the server considers the model’s **“confidence”** in each parameter, quantified using the **Fisher Information Matrix (FIM)**.

**FISHER INFORMATION MATRIX**

There is a theoretical relationship between the empirical FIM and the second derivative of the loss near a minimum. This reflects how sensitive the loss is to changes in a specific parameter:

- **High Fisher Value (High Importance):** A high value indicates a steep curvature. If parameter $i$ is pivotal, even minor modifications are likely to significantly increase the loss. Therefore, its value should be preserved more strictly in the global model to maintain client performance.
- **Low Fisher Value (Low Importance):** A low value indicates the parameter lies in a "flat" region of the loss landscape. If parameter $j$ is less vital, the server can modify or average it more freely, as changes will not drastically impact the loss.

![image.png](Federated%20Learning/image%202.png)

By capturing the curvature of local objective functions, this approach allows **more precise global updates**, protecting parameters that are vital for local accuracy while permitting flexibility in less sensitive areas of the model.

**FISHER-WEIGHTED AVERAGING**

To aggregate models intelligently, we can treat each client’s weights as a probability distribution. Given $M$ client models $\{\theta_i\}_{i=1}^M$ with identical initialization, we approximate each model’s posterior as a Gaussian-distributed posterior:

$p(\theta|\theta_i, F_i)$

where:

- $F_i$ is the Fisher Information Matrix, acting as a proxy for the model's confidence.

**The Optimization Objective**

The objective is to find the set of weights $\theta^*$that maximizes the joint posterior across all clients:

 $*\theta^* = \arg \max_\theta \prod_{i=1}^M \lambda_i p(\theta | \theta_i, F_i)*$ 

To simplify the computation, we apply a log transformation, converting the product into a summation:

 $\theta^* = \arg \max_\theta \sum_{i=1}^M \lambda_i \log p(\theta | \theta_i, F_i)$ 

In this context, $\lambda_i$ are scalars (where $\sum \lambda_i = 1$) that represent additional importance weighting, such as $\lambda_i = 1/C$ or proportional to dataset size.

**The Closed-Form Solution**

Solving this optimization problem leads to a specific closed-form solution for the global model parameters:

 $\theta^* = \left(\sum_i \lambda_i F_i\right)^{-1}\left(\sum_i \lambda_i F_i \theta_i\right)$ 

This approach mirrors the logic of **Bayesian Neural Networks (BNNs)**. By modeling weights and their importance, the FIM effectively dictates the "certainty" of a parameter. 

**Addressing Computational Complexity**

**`Problem`** Estimating and storing a full FIM is often unfeasible for modern, over-parameterized neural networks due to the massive number of parameters.

**`Solution`** We can use a **diagonal approximation** of the FIM. Representing the matrix as a vector makes the computation tractable.

For each individual parameter $j$, the solution simplifies to:

$\theta^*_j = \frac{\sum_{i=1}^M \lambda_i F_{i,j} \theta_{i,j}}{\sum_{i=1}^M \lambda_i F_{i,j}}$

<aside>
⚠️

If all importance scores are equal ($F_1 = F_2 = \dots = F_K$), this formula reduces exactly to standard **Federated Averaging**.

</aside>

**PRACTICAL CONSIDERATION IN FL**

Each client must transmit two primary components to the server:

- The updated local model weights $\theta^{(k)}$
- The diagonal Fisher vector $\text{diag}(F^{(k)})$ 
calculated after local training using the **squared gradient of the log-likelihood** on a local mini-batch.

Since both components have the same size as the model, the total payload per communication round is **doubled**. Although this increases bandwidth requirements, it remains practical in many Federated Learning settings where the model size is manageable.

From a privacy perspective, the Fisher values reveal only the **sensitivity of the model parameters**, not the raw data. However, they may still leak some information, so they are often combined with **secure aggregation** or **differential privacy (DP)** mechanisms.

**STRENGHTS AND LIMITATIONS**

**Strengths**

- Improves accuracy over **FedAvg** on heterogeneous data by weighting parameters according to their importance.
- Often performs even better when applied to **pre-trained models**, where clients share a common feature representation.
- Naturally supports **one-shot federated learning** or **sporadic client participation**, allowing models to be merged without multiple training rounds.
- Compatible with other **server-side techniques**, such as momentum or quantization.

**Weaknesses**

- Requires **additional client-side computation**, since multiple forward and backward passes are needed to estimate the FIM accurately.
- Can suffer from **numerical instability**, as the FIM may produce very large or very small values.
- May provide limited benefit when **client models are too far apart in parameter space**, because the FIM estimated on local data may no longer be meaningful at a very different point in the weight space.

**DERIVATION OF FISHER-WEIGHTED AVERAGING**

**Setup**

After local training, each client $i = 1, \ldots, M$ provides:

- $\theta_i \in \mathbb{R}^{n_p}$ the vector of model parameters, where $n_p$ is the number of parameters.
- $F_i \in \mathbb{R}^{n_p \times n_p}$ the **Fisher Information Matrix**, which is symmetric and positive semi-definite.

Assume a Gaussian posterior $N(\theta_i, F_i^{-1})$ around each $\theta_i$. The goal is to find the parameter vector $\theta$ that maximizes the product of these $M$ independent posteriors:

$\theta^* = \arg \max_\theta \prod_{i=1}^{M} \lambda_i  p(\theta \mid \theta_i, F_i)$

This is equivalent to maximizing the sum of the logarithms of the posteriors:

$\theta^* = \arg \max_\theta \sum_{i=1}^{M} \lambda_i \log p(\theta \mid \theta_i, F_i)$

***dim***

Assuming a Gaussian posterior with precision matrix $F_i$:

$p(\theta \mid \theta_i, F_i) =\frac{|F_i|^{1/2}}{(2\pi)^{n_p/2}}\exp\left\{-\frac{1}{2}(\theta-\theta_i)^\top F_i(\theta-\theta_i)\right\}$

The normalization constant does not depend on 
$\theta$, so it can be ignored in the optimization:

$p(\theta \mid \theta_i, F_i) \approx\exp\left\{-\frac{1}{2}(\theta-\theta_i)^\top F_i(\theta-\theta_i)\right\}$

Taking the logarithm and substituting into the objective:

$\theta^\star =\arg \max_\theta-\frac{1}{2}\sum_{i=1}^{M}\lambda_i(\theta-\theta_i)^\top F_i (\theta-\theta_i)$

Multiplying by $-2$ yields an equivalent minimization problem:

$\theta^\star =\arg \min_\theta\sum_{i=1}^{M}\lambda_i(\theta-\theta_i)^\top F_i (\theta-\theta_i)$

**Expand the quadratic forms:**

 $\theta^\star = \arg \min_\theta \sum_{i=1}^M \lambda_i \left( \theta^\top F_i \theta - 2 \theta^\top F_i \theta_i + \theta_i^\top F_i \theta_i \right)$ 

The last term is independent of $\theta$, so it can be removed from the optimization problem.

**Collect terms**

Collect $\theta$ as it does not depend on the summation over clients models:

$\theta^\star =\arg \min_\theta\theta^\top\left(\sum_{i=1}^{M}\lambda_i F_i\right)\theta-2\theta^\top\left(\sum_{i=1}^{M}\lambda_i F_i \theta_i\right)$

**Take the gradient and set to zero**

Since all $F_i$ are symmetric, we can write it as:

 $2 \left( \sum_{i=1}^M \lambda_i F_i \right) \theta - 2 \left( \sum_{i=1}^M \lambda_i F_i \theta_i \right) = 0$ 

**Solve for $\theta^\star$**

$\theta^\star =\left(\sum_{i=1}^{M}\lambda_i F_i\right)^{-1}\left(\sum_{i=1}^{M}\lambda_i F_i \theta_i\right)$

If $F_i$ is the diagonal of the Fisher Information Matrix, we can treat it as a vector, that is $F_i \in \mathbb{R}^{n_p}$. So the matrix inversion becomes a simple fraction (element-wise for each component $j$):

$\theta^* = \frac{\sum_{i=1}^M \lambda_i F_{i}^{(j)} \theta_{i}^{(j)}}{\sum_{i=1}^M \lambda_i F_{i}^{(j)}}$

If all importance scores are equal (i.e., $F_i$ is the identity matrix for all clients) and $\lambda_i$ represents the sample fraction per client, the solution reduces exactly to **Federated Averaging (FedAvg)**.

## Client-Side Approach

**THE CLIENT DRIFT PROBLEM**

In non-IID settings, **FedAvg’s** local updates often move in inconsistent directions because each client optimizes toward its own local optimum. Simply averaging models, even with importance weighting, may produce a **global model that drifts away from the true optimum.** The Key Drivers of this Drift are:

- **Data Heterogeneity:** the more dissimilar the clients’ data are, the more divergent the updates.
- **Partial Participation:** When only a subset of clients updates the model in each round, the global model's trajectory becomes even more unstable, worsening the drift effect.

![image.png](Federated%20Learning/image%203.png)

The primary result of client drift is either **slower convergence** or a **suboptimal global model** that fails to generalize effectively across all clients. This highlights a critical need: finding mechanisms to **align local updates with the global objective**.

**SIMULATING THE IDEAL UPDATE**

To mitigate client drift, the goal is to ensure each client follows the **ideal update direction**—the trajectory the model would take if all data were consolidated in a single central location.

This ideal direction could theoretically be achieved if clients transmitted both their model parameters and gradients to a central server. The server would then:

- Aggregate all local gradients.
- Compute a unified global update.
- Broadcast this correction back to the clients.

By using this global information, clients could adjust their local update direction, avoiding drift caused by non-IID data.

**`Challenge`** While this method would align local updates with the global objective, **directly exchanging gradients is communication-expensive**. The primary challenge, therefore, is to **approximate this global correction efficiently.**

### Scaffold

SCAFFOLD addresses the **client drift** problem by modifying the local update step. It incorporates a correction term—known as a **control variate**—to align local gradients with the global objective.

The algorithm introduces two auxiliary variables to track training directions:

- **$c$ (Server Control Variate):** An estimate of the global update direction (the path the model would take if all data were centralized).
- **$c_i$  (Client Control Variate):** A term that captures how client $i$’s local updates systematically differ from the global direction due to its specific local data distribution.

**THE TRAINING PROCESS**

**Round 1: Initialization**

1. **Initial Training:** In the first round, no correction is applied; clients train normally on the initial global model $w$.
2. **Local Estimation:** At the end of the round, each client computes its initial $c_i$ (the average gradient experienced during local training) and sends it to the server.
3. **Global Aggregation:** The server aggregates these values to form the global control variate:
    
     $c = \frac{1}{N}\sum_{i=1}^N c_i$ 
    

**Round 2 and Beyond: Corrected Training**

When client $i$ receives the current global model $w$, it no longer uses the raw local gradient  $\nabla L_i(w)$. Instead, it applies an **adjusted gradient**:

 $g_i(w) = \nabla L_i(w) - c_i + c$ 

![image.png](Federated%20Learning/1111c486-f43e-4ce7-9e5e-a7478284a0b1.png)

This formula works by:

- **Removing local bias ($-c_i$):** Subtracting the client’s own typical drift.
- **Injecting global direction ($+c$):** Adding the estimate of the ideal global path.

By using the signal $(c - c_i)$, each client biases its optimization trajectory toward areas where there is stronger agreement among all participants.

![image.png](Federated%20Learning/image%204.png)

**Strengths and Benefits**

- **Mitigation of Client Drift:** By constraining local updates to align with the global trajectory, SCAFFOLD keeps the model on a **stable path toward the true optimum**.
- **Faster Convergence:** Reduces variance between client updates, enabling **provably faster convergence** than FedAvg, especially under **partial participation**.
- **Robust Regularization:** Particularly effective in **non-IID or challenging scenarios**, where FedAvg might struggle or converge to a suboptimal model.

**Limitations**

- **Scope of Impact:** Client-side regularization alone may not suffice in complex settings. **Server-side strategies**, such as Fisher-weighted aggregation, may still be necessary to further improve performance.

## Prototype-based Approach

Model or gradient averaging (like **FedAvg**) often proves suboptimal when client data distributions are highly divergent—for instance, when each client possesses data for entirely unique classes. 

The **Prototype-based approach** addresses this by sharing **interpretable intermediate representations** rather than raw model parameters.

<aside>
🧠

**What is a Prototype?**

A **class prototype** is a feature vector that captures the essence of a specific class from a client’s data. It is generated as follows:

- **Feature Extraction:** A neural network maps input examples into a high-dimensional embedding space.
- **Clustering:** Examples from the same class form a cluster in this space.
- **Centroid Calculation:** The **prototype** is defined as the centroid of these points. This is typically computed as:
    - The **mean** of all feature vectors for that class.
    - The **medoid** i.e., the point minimizing total distance to others in the cluster.

Prototypes can be used for **classification** (nearest prototype) and **semantic alignment** between clients.

</aside>

The core **hypothesis** is that by sharing and aligning these prototypes, clients can build a **consistent global understanding** of each category.

- **Improved Generalization:** The global model learns what each class “looks like” by observing the **average representation** across clients, without accessing raw data.
- **Handling Heterogeneity:** Even if clients see completely different classes (e.g., Client A sees only dogs, Client B sees only cats), sharing prototypes allows all clients to build a **consistent understanding of the full feature space**.

**FEDPROTO ALGORITHM**

**FedProto** is a specialized federated learning framework that communicates through prototypes rather than model parameters. 

**The Training Process**

For each client $i$, the model is composed of a feature extractor $f_i$ and a classifier $g_i$. The algorithm proceeds as follows:

1. **Local Prototype Computation**
At the end of each local training round, client $i$ calculates a prototype $p_{i,c}$ for every class $c$ present in its local dataset $D_i$. The prototype is the **mean feature embedding** for each class in its local data:
    
     $p_{i,c} = E_{(x,y) \sim D_i} [f_i(x) : y = c]$ 
    
2.  **Communication and Aggregation**
    
    Instead of full model weights, clients send these local prototypes to the central server. The server then averages them to form **global prototypes** ($\bar{p}_c$) for each class.
    
3. **Prototype Regularization**
    
    The server sends the global prototypes back to the clients. To ensure that **local feature** representations **align with the global one,** each client adds a regularization term to its local loss function:
    
     $L^{\text{FedProto}}_i = L^{\text{local}}_i + \lambda \|p_{i,c} - \bar{p}c\|^2$ 
    
    effectively addressing the challenges of heterogeneous label distributions.
    

**Inference: Nearest-Mean Classification**

FedProto supports standard classifiers (e.g., softmax or linear heads trained with the local loss + regularization). Alternatively, a **nearest-mean classifier** can be used:

 $\hat{y} = \arg \min_c \|f(x) - \bar{p}_c\|^2$ 

The input is assigned to the class whose prototype is **closest in feature space**, implementing a **nearest-prototype classification** strategy.

**BENEFITS AND RESULTS**

- **Efficiency:** Prototype vectors are much smaller than full model updates, reducing **communication costs** since only prototypes—not entire models—are shared.
- **Effectiveness:** Regularizing local models toward **global prototypes** allows FedProto to outperform FedAvg and similar methods on heterogeneous benchmarks, achieving higher accuracy.

Experiments demonstrate **faster convergence** and improved handling of clients with **non-overlapping class distributions**, making FedProto a simple yet powerful method for sharing knowledge beyond raw parameter averaging.

**APPLICATIONS**

- **Standard Federated Learning:**
    
    By optionally combining prototype sharing with parameter merging techniques (e.g., FedAvg), FedProto can learn a **single global model** like other FL methods.
    
- **Personalized Federated Learning:**
    
    Class prototypes allow each client to **fine-tune its model locally** while still benefiting from knowledge transferred through the prototypes.
    
- **Model-Heterogeneous Federated Learning:**
    
    Since prototypes are architecture-agnostic, FedProto works even when clients have **different model architectures**, unlike standard parameter-wise methods.
    

## Summary and Conclusions

Federated Learning (FL) provides a robust framework for collaborative model training on decentralized data. By shifting the training process to the network's edge, it addresses critical privacy, regulatory, and bandwidth constraints that typically make data centralization impractical.

While **FedAvg** serves as the simple and effective baseline for the field, its performance often degrades when client data is heterogeneous. This has led to the development of several sophisticated categories of enhancement:

- **Client-side tweaks**: e.g. FedProx, SCAFFOLD introduce regularization or control variates to reduce client drift.
- **Server-side tweaks**: e.g. Fisher-merge, introduce per-parameter weights to have a more fine-grained aggregation.
- **Knowledge sharing**: e.g. FedProto shares prototypes (or other knowledge) instead of raw model parameters to align learning.

These approaches greatly improve convergence speed and final accuracy in challenging FL settings (non-iid data, few clients per round, etc.), narrowing the gap between federated and centralized training.

**FUTURE DIRECTIONS**

As Federated Learning continues to evolve, research is shifting toward making these systems more robust and adaptable for real-world deployment. Key areas of focus include:

- **Personalization**: Moving beyond a single global model — allowing client-specific models or fine-tuning — is a key research direction to handle diverse user needs.
- **Privacy enhancements**: Incorporating stronger privacy guarantees (differential privacy, secure aggregation, etc.) without sacrificing too much accuracy remains an ongoing challenge.
- **Scalability**: Future FL systems must scale to millions of devices; this entails reducing communication (e.g., model compression) and handling unreliable devices and network conditions.

Overall, FL is key to enabling collaborative AI on distributed data. Ongoing research aims to make it more robust, fair, and applicable to a wider range of real-world scenarios.