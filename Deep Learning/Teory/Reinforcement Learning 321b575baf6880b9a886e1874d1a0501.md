# Reinforcement Learning

## Introduction

Reinforcement learning (RL) is a third paradigm in machine learning, alongside **supervised learning** and **unsupervised learning**.

- **Supervised Learning:** Relies on labeled data where each input is paired with a correct output. It is generally the most effective approach when labels are readily available.
- **Unsupervised Learning:** Operates on raw, unlabeled data to extract patterns or representations.
- **Reinforcement Learning:** Differs from both paradigms, because it operates without labels or prior knowledge of the "correct" action. Instead, an agent learns through **trial and error** by **interacting with an environment** to maximize cumulative rewards.

![image.png](Reinforcement%20Learning/55c67e79-daa8-45f4-a6a6-47996e22bf44.png)

**KEY CHARACTERISTICS**

RL is defined by several unique attributes that distinguish it from other paradigms:

- **No Labeled Supervision:** The agent does not know the optimal action in advance and receives no explicit instructions on right or wrong moves.
- **Reward Signal:** The agent receives a scalar feedback signal, $R_t$, indicating how well it is performing at step $t$. Actions leading to high rewards are reinforced, while those leading to low or negative rewards are discouraged.
    
    <aside>
    ⚙
    
    **Helicopter Stunt Maneuvers:** A positive reward is granted for following a desired trajectory, while a negative reward is issued for crashing.
    
    </aside>
    
- **Delayed Feedback:** Consequences may not appear immediately, making it difficult to determine which specific action was beneficial.
- **Sequential Data:** data is non-i.i.d. (independent and identically distributed), meaning that the current actions influence future states.
- **Active Agency:** The agent is an active participant whose actions directly affect the environment he lives in.

## Inside a RL agent

**SEQUENTIAL DECISION MAKING**

The objective of the agent is to select actions that maximize the **cumulative reward** over the course of an episode. This requires considering the long-term effects of decisions, since:

- **Actions** may have long-term consequences.
- **Rewards** may be delayed.
- It may be beneficial to sacrifice an immediate reward in order to obtain a greater reward later (e.g., refuelling a helicopter now to prevent a crash several hours later).

**AGENT-ENVIRONMENT INTERACTION**

The interaction is a continuous cycle defined by discrete time steps. At each step $t$:

**The Agent:**

1. **Receives** an observation $O_t$ (the current state of the environment).
2. **Receives** a scalar reward $R_t$
3. **Executes** an action $A_t$ based on that information.

**The Environment:**

1. **Receives** the action $A_t$
2. **Transitions** to a new state and emits the next observation $O_{t+1}$
3. **Emits** the next scalar reward $R_{t+1}$

![image.png](Reinforcement%20Learning/image.png)

An **episode** represents a complete sequence of these interactions, from the initial state to a terminal state (such as the end of a game or the completion of a flight).

### State

**HISTORY**

The history, $H_t$, is the complete record of all interactions from the start of an episode up to the current time step $t$. It includes the entire sequence of observations, actions, and rewards:

$H_t = O_1, R_1, A_1, \ldots, A_{t-1}, O_t, R_t$

**STATE**

The state, $S_t$, represents the specific information required for the agent to determine its next action. Formally, the state is a **function of the history**:

$S_t = f(H_t)$

Although the current state depends on the entire sequence of previous interactions, using the full history is usually impractical. In practice, the state is a simplified representation that captures only the essential information needed for decision-making.

**AGENT AND ENVIRONMENT STATES**

In reinforcement learning, it is critical to distinguish between the internal state of the agent and the external state of the world:

- **Agent state** $S^a_t$ This is the internal representation of information that the agent uses to select its next action. In most RL contexts, the term "state" refers specifically to this agent-side information.
- **Environment state** $S^e_t$ this represents the full configuration of the environment, the actual data the world uses to determine the next observation or reward. This state is typically invisible to the agent.

**OBSERVABILITY**

The relationship between these two states defines the nature of the learning task:

- **Full observability**: agent directly observes environment state, meaning $S^a_t=S^e_t$
- **Partial observability**: agent only indirectly or partially observes the environment state ***(**e.g. ****A poker agent sees only public cards, not opponents’ private cards)*.

<aside>
⚠️

RL systems generally do not attempt to explicitly model the complete environment state ($S^e_t$) for several practical reasons:

- **Complexity:** the environment may be too large or intricate to represent completely.
- **Uncertainty:** some underlying rules or variables may be unknown.
- **Infeasibility:** precise modeling can be computationally expensive and often unnecessary.

Instead, the agent learns through **interaction and experience**.

</aside>

**MAJOR COMPONENTS OF RL AGENT**

An RL agent may include one or more of the following components:

- **Policy:** the function that defines the agent’s behaviour.
- **Value function:** estimates how good a state and/or action is.
- **Model:** a representation of the environment.

### Policy

The **policy** defines an agent’s behavior. It acts as a mapping from state to action, essentially It determines **which action the agent chooses in each state**. Policies generally fall into two categories:

- **Deterministic policy**
    
    A deterministic policy maps each state directly to a single, specific action:
    
    $a = \pi(s)$
    
    *Example: In a specific state, the agent is programmed to always move left.*
    
    A purely deterministic policies can be impractical in complex environments, as they may require the agent to have pre-memorized the optimal action for every possible state it might encounter.
    
- **Stochastic policy**
    
    A stochastic policy defines a probability distribution over the available actions given a specific state:
    
     ****$\pi(a|s) = P[A_t = a|S_t = s]$ 
    
    Rather than choosing one fixed action, the agent assigns probabilities to all possible moves. 
    

### Return

The **Return** ($G_t$) is the **total discounted reward** starting from time-step $t$:

$G_t = R_{t+1} + \gamma R_{t+2} + \ldots = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}$

The discount factor $\gamma \in [0, 1]$ determines the present value of future rewards. Specifically, the value of receiving a reward $R$ after $k + 1$ time-steps is $\gamma^k R$. This mechanism prioritizes immediate rewards over delayed ones.

In reinforcement learning, future rewards are usually **discounted**, meaning rewards received further in the future contribute less to the total value. If the discount factor is close to:

- **`0`** makes the agent **myopic**, valuing immediate rewards much more than future rewards,
- **`1`** makes the agent **far-sighted**, giving nearly equal importance to immediate and long-term rewards.

### Value function

The **value function** tells us **how good it is to be in a certain state under a given policy.** There are two primary types of value functions:

- **State-Value function $v_\pi(s)$** is the expected return starting from state $s$ and then following policy $\pi$:
    
    $v_\pi(s) = E_\pi[G_t|S_t = s]$
    
    The agent evaluates the quality of its current state.
    
- The **action-value function** $q_\pi(s, a)$ is the expected return starting from state $s$, taking action $a$, and then following policy $\pi$:
    
    $q_\pi(s, a) = E_\pi[G_t|S_t = s, A_t = a]$
    
    Evaluates the quality of taking a specific action in a state.
    

While the state-value function evaluates the state itself, the action-value function allows the agent to compare the long-term impact of different individual actions within that state.

**BELLMAN EXPECTATION EQUATION - STATE VALUE FUNCTION**

The **state-value function** $v_\pi(s)$ can be decomposed into two parts:

- The **immediate reward $R_{t+1}$** received after taking an action in state $s$,
- The **discounted expected value** of the successor state $\gamma v_\pi(S_{t+1})$

Formally, the equation is expressed as:

$v_{\pi}(s)=\mathbb{E}_{\pi}[R_{t+1}+\gamma v_{\pi}(S_{t+1})\mid S_t=s]$

***Derivation***

We begin with the definition of the state-value function:

$v_\pi(s) = E_\pi[G_t|S_t = s]$

Substituting the definition of the return $G_t$ as the sum of discounted rewards:

$v_\pi(s) = \mathbb{E}_\pi[R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \ldots \mid S_t = s]$

Factoring out the discount factor $\gamma$:

$v_\pi(s) = \mathbb{E}_\pi[R_{t+1} + \gamma (R_{t+2} + \gamma R_{t+3} + \ldots) \mid S_t = s]$

The term inside the parentheses is the definition of the return at the next time step, $G_{t+1}$:

$v_\pi(s) = \mathbb{E}_\pi[R_{t+1} + \gamma G_{t+1} \mid S_t = s]$

Using the **linearity of expectation**:

$v_\pi(s) =\mathbb{E}_\pi[R_{t+1} \mid S_t=s] +\gamma \, \mathbb{E}_\pi[G_{t+1} \mid S_t=s]$

Applying the **law of iterated expectations**:

$v_\pi(s) =\mathbb{E}_\pi[R_{t+1} \mid S_t=s]+\gamma\mathbb{E}_\pi\big[\mathbb{E}_\pi[G_{t+1} \mid S_{t+1}=s'] \mid S_t=s\big]$

Since $\mathbb{E}_\pi[G_{t+1} \mid S_{t+1}] = v_\pi(S_{t+1})$ we obtain:

$v_{\pi}(s)=\mathbb{E}_{\pi}[R_{t+1}+\gamma v_{\pi}(S_{t+1}) \mid S_t=s]$

This is the **Bellman expectation equation** for the state-value function.

**BELLMAN EXPECTATION EQUATION - ACTION VALUE FUNCTION**

The action-value function can similarly be decomposed:

$q_\pi(s, a) = E_\pi[R_{t+1} + \gamma q_\pi(S_{t+1}, A_{t+1})|S_t = s, A_t = a]$

***Derivation***

We begin with the definition of the **action-value function**:

$q_\pi(s,a) = \mathbb{E}_\pi[G_t \mid S_t = s, A_t = a]$

Substituting the definition of the return $G_t$ as the sum of discounted rewards:

$q_\pi(s,a) = \mathbb{E}_\pi[R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \ldots \mid S_t=s, A_t=a]$

Factoring out the discount factor $\gamma$:

$q_\pi(s,a) =\mathbb{E}_\pi[R_{t+1} + \gamma (R_{t+2} + \gamma R_{t+3} + \ldots) \mid S_t=s, A_t=a]$

The term inside the parentheses corresponds to the **return at the next time step**, $G_{t+1}$:

$q_\pi(s,a) = \mathbb{E}_\pi[R_{t+1} + \gamma G_{t+1} \mid S_t=s, A_t=a]$

Using the **linearity of expectation**:

$q_\pi(s,a) = \mathbb{E}_\pi[R_{t+1} \mid S_t=s, A_t=a] + \gamma \mathbb{E}_\pi[G_{t+1} \mid S_t=s, A_t=a]$

Applying the **law of iterated expectations**:

$q_\pi(s,a) =
\mathbb{E}_\pi[R_{t+1} \mid S_t=s, A_t=a]
+
\gamma \,
\mathbb{E}_\pi
\big[
\mathbb{E}_\pi[G_{t+1} \mid S_{t+1}=s', A_{t+1}=a']
\mid S_t=s, A_t=a
\big]$

Since:

$\mathbb{E}_\pi[G_{t+1} \mid S_{t+1}=s', A_{t+1}=a'] = q_\pi(S_{t+1},A_{t+1})$

we obtain:

$q_\pi(s,a)=\mathbb{E}_\pi\big[R_{t+1}+\gamma q_\pi(S_{t+1},A_{t+1})\mid S_t=s, A_t=a\big]$

This expresses the **Bellman expectation equation for the action-value function** in terms of the **state-value of the next state**.

### Model

A **model** is the agent's internal representation of the environment. It is used to predict how the environment will respond to specific actions, allowing the agent to plan ahead. 

- **State Transition Probability** ($P$) predicts the next state
    
    $P^a_{ss'} = P[S_{t+1} = s'|S_t = s, A_t = a]$
    
- **Reward Function** ($R$) predicts the next (immediate) reward.
    
    $R^a_s = E[R_{t+1}|S_t = s, A_t = a]$
    

<aside>
⚙

**Look at the Maze example in the slides**

</aside>

## Model-free prediction

Model-free prediction aims to **estimate the value function for a given policy** when the **environment model is unknown**.

Without knowledge of the transition probabilities and reward function, the agent cannot compute the value function analytically. Instead, it must estimate the value function by interacting with the environment and learning directly from **experience**.

Two primary methods are used for this estimation:

- **Monte-Carlo Learning**
- **Temporal-Difference Learning**

### Monte-Carlo Learning

Monte Carlo (MC) methods enable an agent to learn a value function directly from **episodes of experience**.

**Core Principles**

- **Model-Free:** MC does not require knowledge of the environment’s transition probabilities or reward function.
- **Complete Episodes**: MC updates occur only after an episode ends, so it can be applied only to **episodic environments** where all episodes terminate.

The agent estimates the value of a state by **averaging the actual returns** ($G_t$) observed after visiting that state across many episodes.

**POLICY EVALUATION**

The goal of policy evaluation is to learn the state-value function $v_\pi$ from episodes of experience generated by a specific policy $\pi$:

$S_1, A_1, R_2, \ldots, S_k \sim \pi$

While the true value function relies on an expected return, **Monte-Carlo Policy Evaluation** uses the **empirical mean return** observed during actual play.

**EVERY-VISIT MONTE-CARLO POLICY EVALUATION**

The goal si to estimate the value of a state under a policy $\pi$. To achieve this, the agent tracks all visits to a state across multiple episodes and averages the returns observed after each visit. 
According to the **Law of Large Numbers**, this empirical average converges to the true state value as the number of visits approaches infinity:

$V(s) \rightarrow v_\pi(s)$ as $N(s) \rightarrow \infty$.

**The Algorithm:**

1. **Initialization**
    
    For every state $s$ in the state space:
    
    - Set the visit counter to zero: $N(s) = 0$
    - Set the cumulative return sum to zero: $S(s) = 0$
2. **Data Collection**
    
    Run multiple episodes following the fixed policy $\pi$
    
3. **State Evaluation (After each episode)**
    
    For every time-step $t$ where state $s$ is visited:
    
    - **Increment Counter:** $N(s) \leftarrow N(s) + 1$
    - **Update Total Return:** $S(s) \leftarrow S(s) + G_t$ (where $G_t$ is the total discounted reward from time $t$ until the end of the episode).
4. **Value Estimation**
After a sufficient number of episodes, calculate the state value as the mean return:
    
    $V(s) = \frac{S(s)}{N(s)}$
    

**INCREMENTAL MONTE-CARLO UPDATES**

Rather than storing a massive sum of returns, we can **update the value function** $V(s)$ **incrementally** after each episode ($S_1, A_1, R_2, \ldots, S_T$). For each state $S_t$ visited with a corresponding return $G_t$, the update is performed as follows:

1. **Update Visit Count:** $N(S_t) \leftarrow N(S_t) + 1$
2. **Update Value Estimate:**
    
    $V(S_t) \leftarrow V(S_t) + \frac{1}{N(S_t)} (G_t - V(S_t))$
    

In many practical reinforcement learning problems, especially non-stationary ones where the environment may change over time, it is useful to "forget" older episodes. By replacing the $1/N(S_t)$ term with a constant learning rate $\alpha$, we create a **running mean**:

$V(S_t) \leftarrow V(S_t) + \alpha(G_t - V(S_t))$

where:

- **$V(s)$:** The current value estimate for state $s$.
- **$G_t$:** The actual return observed in the current episode.
- **$\alpha$:** The learning rate (step size), which determines how much the new observation influences the existing estimate.
- **$(G_t - V(S_t))$:** The **estimation error**, representing the difference between the observed return and the current prediction.

This approach ensures the value function converges toward the expected return as more samples are collected.

**LIMITATIONS OF MONTE-CARLO METHODS**

The main drawback of Monte Carlo methods is **slow convergence**. Since the agent must wait until the end of an episode to obtain each return sample, many episodes are often required for accurate estimates.

- **Example:** In Blackjack, after **10,000 episodes** the value function remains noisy; after **500,000 episodes**, the estimates become smooth and reliable.

![image.png](Reinforcement%20Learning/image%201.png)

<aside>
⚙

Look at the **blackjack example** in the slides

</aside>

### Temporal-Difference learning

The goal of TD learning is to learn the state-value function $v_\pi$ online and incrementally. TD improves upon Monte-Carlo methods through two key mechanisms:

- **Step-by-step Updates:** The value function is updated after every interaction, rather than waiting for the episode's conclusion.
- **Bootstrapping:** The agent updates its current value estimate using its own estimate of the next state’s value.

**THE TD(0) ALGORITHM**

TD(0) is the simplest form of **temporal-difference learning**. It updates the value $V(S_t)$ toward a **TD Target**, which consists of the immediate reward $R_{t+1}$ plus the discounted value of the next state $\gamma V(S_{t+1})$.

Therefore the Update Rule:

$V(S_t) \leftarrow V(S_t) + \alpha(R_{t+1} + \gamma V(S_{t+1}) - V(S_t))$

where:

- **TD Target $R_{t+1} + \gamma V(S_{t+1})$:** is the estimated total return from the perspective of the next time step.
- **TD Error $\delta_t$:** The difference between the TD Target and the current estimate $V(S_t)$.

<aside>
⚠️

At the start of the learning process, value estimates ($V(S_t)$ and $V(S_{t+1})$) are inaccurate. However, each update combines a **certain reward** (real feedback) with an **uncertain estimate** (the next state's value). Because the reward is a ground-truth signal, each update incrementally pulls the value function toward reality. Over time, these improved estimates propagate backward through the state space, making the entire value function increasingly reliable.

</aside>

**MC VS. TD**

The fundamental difference between **Monte-Carlo** (MC) and **Temporal-Difference** (TD) learning lies in **when** and **how** they update their knowledge:

- **TD is "Online":** it updates the value function after every step. If an episode lasts $3000$ steps, TD performs $3000$ updates, enabling continuous and incremental learning.
- **MC is "Offline":** It must wait until the end of an episode to observe the total return $G_t$ before performing a single update.

**In practice:** For long episodes, an MC agent stays "blind" to its performance until the very end, while a TD agent adjusts its behavior continuously.

Another key difference concerns **sequence requirements**:

- **TD handles incomplete sequences:** it can learn from partial interactions without knowing the final outcome. This is useful in tasks with long horizons where reaching the end is rare or time-consuming.
- **MC requires complete sequences:** ecause it relies on the actual total return, learning can occur only after the episode terminates.

As a result, **TD can operate in continuing (non-terminating) environments**, while **MC is limited to episodic (terminating) environments**.

**BIAS/VARIANCE TRADE-OFF**

In reinforcement learning, the choice between **Monte Carlo** (MC) and **Temporal Difference** (TD) methods reflects a trade-off between **bias** and **variance**.

- The **Return** $G_t = R_{t+1} + R_{t+2} + \ldots + \gamma^{T-1} R_T$ is an **unbiased estimate** of the true value $v_\pi(S_t)$ because it reflects the actual rewards observed until the end of the episode.
- The **TD Target $R_{t+1} + \gamma V(S_{t+1})$** is a **biased** estimate of $v_\pi(S_t)$ bbecause it relies on the agent’s current estimate of the next state value rather than the true value.
- The **True TD target** $R_{t+1} + v_\pi(S_{t+1})$ would be an **unbiased estimate** of $v_\pi(S_t)$, but the true value function is unknown.

**Comparison of Performance:**

- **Temporal Difference (Low Variance, High Bias):**
The TD target depends on only a single transition (one action, one reward, and the resulting state). Because fewer random variables are involved in each update, it has **much lower variance**. Although bootstrapping introduces bias, the reduced noise allows the agent to learn **faster**.
- **Monte Carlo (High Variance, Zero Bias):**
    
    While MC is unbiased, it relies on the **final return**, which is the sum of many actions, transitions, and rewards. This accumulation of random events leads to **high variance**, requiring a large number of episodes to obtain accurate estimates.
    

In practice, it is often preferable to accept a **slightly biased estimate** if it can be obtained more efficiently. This reflects a common situation in optimization, where **approximate solutions** are preferred over exact ones because they are computationally feasible and faster to compute.

## Model-free control

- **Model-free prediction** focuses on **estimating the value function for a given policy** when the environment model is unknown.
- **Model-free control** instead focuses on **finding a good policy** without knowledge of the environment model.

Common approaches to model-free control include:

- **On-Policy Monte Carlo Control**
- **On-Policy Temporal-Difference Learning**
- **Off-Policy Learning**

**IMPROVING A POLICY**

Given a policy $\pi$ the general approach is:

- **Policy Evaluaton** The agent evaluates the current policy $\pi$ by estimating its value functions:
    - $v_\pi(s) = E[R_{t+1} + \gamma R_{t+2} + \ldots |S_t = s]$
    - $q_\pi(s, a) = E[R_{t+1} + \gamma R_{t+2} + \ldots |S_t = s, A_t = a]$
- **Policy Improvement** The agent uses these value estimates to build a **better policy**. 
A common method is to act **greedily** with respect to the current estimates:
    - $\pi' = \text{greedy}(v_\pi)$
    - $\pi' = \text{greedy}(q_\pi)$

In model-free contexts, we must rely on the **action-value function $Q(s, a)$** for policy improvement. Being greedy with respect to the state-value function $V(s)$ requires a model of the environment to know which action leads to which state (transition probabilities). Without a model, only $Q(s, a)$ directly tells the agent which action yields the highest expected return from the current state.

**THE EXPLORATION DILEMMA**

A greedy policy always selects the action that **maximizes the current estimated return**.

<aside>
⚙

If an agent is in front of a door and **evaluates** that moving "Left" yields a reward of 0 while moving "Right" yields a reward of 1, a greedy policy will always choose "Right" from that point forward.

</aside>

While acting greedily maximizes immediate performance based on current knowledge (**exploitation**), it has a significant flaw: **it does not explore**. If the agent only selects the action with the highest current estimate:

- It never tests alternative actions.
- It may miss superior actions that haven't been explored yet.

This creates the fundamental **Exploration vs. Exploitation** dilemma: the agent must balance choosing what it knows to be good with trying new things to discover what might be better.

**$\epsilon$-GREEDY EXPLORATION**

The **$\epsilon$-greedy policy** is a standard solution to the exploration-exploitation dilemma. It ensures that all $m$ available actions are sampled with a non-zero probability, preventing the agent from prematurely converging on a suboptimal strategy.

At each decision point the agent:

- With **probability** $1 - \epsilon$  chooses the **"greedy" action**, the one with the highest current value estimate.
- With **probability** $\epsilon$ selects an action at random from all $m$ possible options (including the greedy one).

$\pi(a|s) = \begin{cases} \epsilon/m + 1 - \epsilon, & \text{if } a^* = \arg \max_{a \in \mathcal{A}} Q(s, a) \\ \epsilon/m, & \text{otherwise} \end{cases}$

**ON AND OFF POLICY LEARNING**

Another important distinction in reinforcement learning is between:

- **On-policy:** the agent evaluates and improves the exact policy it is currently using to interact with the environment. Learn about policy $\pi$ using experience sampled directly from $\pi$.
    
    <aside>
    ⚙
    
    Person learns that a specific move is dangerous because they attempt it themselves and experience the immediate consequence.
    
    </aside>
    
- **Off-Policy:** The agent learns about a **target policy** $\pi$ while following a different **behavior policy** $\mu$ that could be more exploratory.
    
    <aside>
    ⚙
    
    An athlete observes how a professional moves, extracts useful dynamics from those observations, and then applies that knowledge to their own performance
    
    </aside>
    

Off-policy learning is often advantageous because it allows the agent to **reuse experience generated by old policies $\pi_1, \pi_2, \ldots, \pi_{t-1}$**, even if those policies are not optimal.

### Monte-Carlo Control

**MONTE-CARLO POLICY ITERATION**

In Monte Carlo (MC) control, the agent seeks the **optimal policy** by alternating between two fundamental phases:

- **Policy evaluation**: The agent estimates the action-value function ($Q = q_\pi$) for the current policy $\pi$. This is typically initialized to zero, representing a lack of prior knowledge.
- **Policy improvement**: The agent refines its behavior using an **$\epsilon$-greedy policy** based on the current $Q$ estimates.

![image.png](Reinforcement%20Learning/image%202.png)

By repeatedly cycling through evaluation and improvement, the agent's behavior and value estimates gradually converge toward the optimal action-value function ($q^*$) and the optimal policy ($\pi^*$).

**ON-POLICY MONTE-CARLO CONTROL**

A practical and efficient implementation of this cycle is **On-Policy MC Control**. While traditional policy iteration theoretically requires the evaluation phase to fully converge (infinite episodes) before improving the policy, this is inefficient in practice.

Instead of waiting for full convergence, we perform **incomplete evaluation**. The agent updates the value function and improves the policy simultaneously after every single episode.

![image.png](Reinforcement%20Learning/image%203.png)

The term "On-Policy" specifically means the agent is evaluating and improving the same policy it uses to make decisions.

### TD control

Temporal-Difference (TD) learning offers several advantages over Monte-Carlo (MC) methods, including **lower variance** and the ability to learn **online** from **incomplete sequences**.

By applying these **TD** principles to the control loop, we can estimate the **action-value function** $Q(S, A)$ and improve the policy at every time-step using $\epsilon$-greedy selection.

**SARSA**

**SARSA** is a primary TD control algorithm named after the sequence of variables it processes:

$(S_t, A_t, R_{t+1}, S_{t+1}, A_{t+1})$

The agent updates its action-value estimate using the following TD update rule:

$Q(S, A) \leftarrow Q(S, A) + \alpha(R + \gamma Q(S', A') - Q(S, A))$

![image.png](Reinforcement%20Learning/image%204.png)

**ON-POLICY CONTROL WITH SARSA**

SARSA is an **on-policy** algorithm. This means the action $A'$ used to calculate the TD target in the update rule is selected using the **same policy** ($\pi$) that the agent uses to interact with the environment.

**Every time-step:**

- **Policy evaluation** with Sarsa, the agent updates $Q$ to better approximate $q_\pi$→ $Q \approx q_\pi$
- **Policy improvement** with $\epsilon$-greedy policy improvement.

![image.png](Reinforcement%20Learning/image%205.png)

**Algorithm**

![image.png](Reinforcement%20Learning/image%206.png)

1. **Initialization**
    
    The agent initializes the **Q-table** for every state-action pair, with zero-values representing a lack of prior knowledge.
    
2. **Interaction Loop**
    
    For each step of an episode, the agent follows this sequence:
    
    - **Initial State & Action:** The agent observes the current state $S$ and selects an action $A$ using its $\epsilon$-greedy policy.
    - **Execution:** The agent executes action $A$, receiving a reward $R$ and transitioning to a new state $S'$
    - **Next Action Selection:** From the new state $S'$, the agent selects the next action $A'$ using the **same $\epsilon$-greedy policy**.
    - **Update:** The agent uses $A'$  to update the value of the previous state-action pair $Q(S, A)$.
    - **Transition:** The agent sets $S \leftarrow S'$ and $A \leftarrow A'$, repeating the process until a terminal state is reached.

<aside>
⚙

**Example of Windy gridworld**

</aside>

### Q-learning

Q-learning is the most prominent off-policy Temporal Difference (TD) control algorithm. It allows an agent to evaluate and improve a **target policy** $\pi(a, s)$ while following a different **behavior policy** $\mu(a|s)$.

By decoupling the learning process from the behavior, Q-learning enables several powerful techniques:

- **Imitation Learning:** Learning by observing a human or another agent.
- **Experience Replay:** Reusing data generated by previous policies.
- **Safe Exploration:** Converging on a purely greedy optimal policy while using a highly exploratory behavior policy to navigate the environment.

**THE UPDATE RULE**

The fundamental difference between SARSA and Q-learning lies in how the next action is used in the update:

- In **SARSA**, the update uses the action $A_{t+1}$ actually chosen by the behavior policy.
- In **Q-learning**, the update assumes the agent will take the **best possible action** in the next state, regardless of what the behavior policy actually chooses.

During training, the agent selects the next action using the behavior policy $A_{t+1} \sim \mu(\cdot|S_t)$, 
but the update is performed **as if the agent followed the target policy** $\pi$. 
he Q-value of the current state–action pair is therefore updated toward the value of a **successor action** chosen according to the target policy $A' \sim \pi(\cdot|S_{t+1})$:

$Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha \left[ R_{t+1} + \gamma Q(S_{t+1}, A') - Q(S_t, A_t) \right]$

**OFF-POLICY CONTROL WITH Q-LEARNING**

In **off-policy Q-learning**, both the **behavior policy** $\mu$ and the **target policy** $\pi$ are allowed to improve over time:

- The target policy $\pi$ is defined as **greedy with respect to the current Q-values:**
    
    $\pi(S_{t+1}) = \arg\max_{a'} Q(S_{t+1}, a')$
    
- The behavior policy $\mu$ remains **exploratory (e.g., ε-greedy)**, to ensure sufficient exploration.

Because the target policy is greedy, the Q-learning target simplifies:
$R_{t+1} + \gamma Q(S_{t+1}, A')$

$= R_{t+1} + \gamma Q(S_{t+1}, \arg \max_{a'} Q(S_{t+1}, a'))$

$= R_{t+1} + \gamma \max_{a'} Q(S_{t+1}, a')$

**ALGORITHM**

![image.png](Reinforcement%20Learning/image%207.png)

1. **Initialization**
    
    The Q-table $Q(s, a)$ is initialized for all state-action pairs (typically to 0), and the terminal state is set to 0.
    
2. **Interaction and Learning**
    
    
    - Starting from a **state** $S$, the agent selects an **action** $A$ using the **$\epsilon$-greedy** **behavior policy** $\mu$.
    - The agent then observes:
        - The **immediate reward** $R$
        - The **next state** $S'$
    
    ![image.png](Reinforcement%20Learning/image%208.png)
    
    - At each step, the $Q$-value is updated as:
        
        $Q(S, A) \leftarrow Q(S, A) + \alpha \Big[ R + \gamma \max_{a'} Q(S', a') - Q(S, A) \Big]$
        
    - The agent moves to the new state $S'$ and repeats the process until the end of the episode.

**Theoretical guarantee:** if every state–action pair is visited infinitely often and the learning rate decays appropriately, Q-learning **converges to the optimal action-value function**:

$Q(s, a) \rightarrow q_*(s, a)$

## Value Function Approximation

So far, we have represented value functions using **lookup tables**, where:

- Every **state** $s$ has an entry $V(s)$
- Every **state-action pair** $s, a$ has an entry $Q(s, a)$

While this approach is effective for simple environments, it becomes **impractical for complex problems**. In environments with high-dimensional or continuous state spaces, there are too many states/or actions to store in memory, and learning each value individually becomes too slow.

**Solution:**

Instead of storing every value explicitly, we learn a **parameterized function** that approximates the value using a set of weights:

- $\hat{v}(s, \mathbf{w}) \approx v_{\pi}(s)$
- $\hat{q}(s, a, \mathbf{w}) \approx q_{\pi}(s, a)$

This approach allows the agent to **generalize** its knowledge from seen states to unseen states.

Rather than updating a single table entry, we update the parameter vector $w$ using standard reinforcement learning methods like **Monte Carlo (MC)** or **Temporal Difference (TD)** learning.

**TYPES OF VALUE FUNCTION APPROXIMATION**

![image.png](Reinforcement%20Learning/image%209.png)

There are many possible **function approximators**, for example:

- **Linear combinations of features**
- **Neural network**
- Decision tree
- Nearest neighbour
- Fourier / wavelet bases
- …

In reinforcement learning, the training method must be suitable for **non-stationary and non-iid data**, since the data distribution changes as the agent interacts with the environment and updates its policy.

## Incremental Methods

**GRADIENT DESCENT**

If $J(w)$ is a differentiable function of a parameter vector $w$, its gradient $\nabla_w J(w)$ represents the direction of the steepest increase:

 $\nabla_w J(w) = \begin{pmatrix} \frac{\delta J(w)}{\delta w_1} \\ \vdots \\ \frac{\delta J(w)}{\delta w_n} \end{pmatrix}$ 

To find a local minimum of $J(w)$, we adjust $w$ in the direction of the negative gradient. The update rule is defined as:

$\Delta w = - \frac{1}{2} \alpha \nabla_w J(w)$

where $\alpha$ is a step-size (learning rate) parameter.

![image.png](Reinforcement%20Learning/image%2010.png)

**VALUE FUNCTION APPROXIMATION BY SGD**

In Reinforcement Learning, the goal is to find a parameter vector $w$ that minimizes the **Mean-Squared Error (MSE)** between the approximate value function $\hat{v}(S, w)$ and the true value function $v_{\pi}(S)$:

 $J(w) = E_{\pi}[(v_{\pi}(S) - \hat{v}(S, w))^2]$ 

Using gradient descent to find a local minimum results in the following update:

 $\Delta w = - \frac{1}{2} \alpha \nabla_w J(w)$ 

 $= \alpha E_{\pi}[(v_{\pi}(S) - \hat{v}(S, w))\nabla_w \hat{v}(S, w)]$ 

**Stochastic Gradient Descent (SGD)** simplifies this by sampling the gradient at each step rather than calculating the full expectation. The expected update remains equal to the full gradient update:

 $\Delta w = \alpha(v_{\pi}(S) - \hat{v}(S, w))\nabla_w \hat{v}(S, w)$ 

**FEATURE VECTORS**

To apply function approximation effectively, we represent a state $S$ as a **feature vector** $x(S)$:

 $x(S) = \begin{pmatrix} x_1(S) \\ \vdots \\ x_n(S) \end{pmatrix}$ 

<aside>
⚙

**Examples of Features**

- **Robotics:** Distances from specific landmarks.
- **Finance:** Moving averages or trends in the stock market.
- **Games (Chess):** Specific configurations of pieces and pawns.
</aside>

A well-designed feature vector acts as the bridge between perception and decision-making. It is fundamental for:

- **Efficiency:** Avoiding redundant or inefficient state representations.
- **Dimensionality Reduction:** Compressing a vast state space into a manageable set of inputs.
- **Generalization:** Allowing the agent to estimate the value of states it has never seen by identifying similar features.

Feature representations allow the model to **capture relevant structure in the environment**, avoiding the need to encode every possible state explicitly. Without accurate feature encoding, an agent cannot reliably learn the value of its states or actions in complex environments.

### Linear value function approximation

In linear function approximation, the value function is represented as a linear combination of features. Given a feature vector $x(S)$  and a weight vector $w$, the approximate value is defined as:

$\hat{v}(S, w) = x(S)^T w = \sum_{j=1}^{n} x_j(S)w_j$

The objective function $J(w)$ is quadratic in terms of the parameters $w$:

 $J(w) = E_{\pi}[(v_{\pi}(S) - x(S)^T w)^2]$ 

Because the objective is quadratic, **Stochastic Gradient Descent (SGD)** is guaranteed to converge to the global optimum. The gradient is simply the feature vector itself, $\nabla_w \hat{v}(S, w) = x(S)$, leading to a highly efficient update rule:

 $\Delta w = \alpha(v_{\pi}(S) - \hat{v}(S, w))x(S)$ 

This formula states that the update is the product of:

**Update = stepsize $\times$ prediction error $\times$ feature value**

Structurally, this model can be viewed as a single-layer neural network without an activation function.

**TABLE LOOKUP FEATURES**

Interestingly, traditional tabular reinforcement learning (like Q-learning and SARSA) is a special case of linear approximation. This occurs when we use **Table Lookup Features**, where the state space is represented by a "one-hot" encoded vector.

For a state space with $n$ states, the feature vector $x_{table}(S)$ contains a $1$ at the index corresponding to the current state and $0$ elsewhere:

 $x_{table}(S) = \begin{pmatrix} 1(S = s_1) \\ \vdots \\ 1(S = s_n) \end{pmatrix}$ 
In this specific case, the parameter vector $w$ directly provide the value of each individual state:

 $\hat{v}(S, w) = \begin{pmatrix} 1(S = s_1) \\ \vdots \\ 1(S = s_n) \end{pmatrix} \cdot \begin{pmatrix} w_1 \\ \vdots \\ w_n \end{pmatrix}$ 

### Incremental prediction algorithms

In supervised learning, we assume a supervisor provides the true value $v_{\pi}(s)$. In reinforcement learning, we must substitute this missing supervisor with a **target** derived from experience:

- **For Monte Carlo (MC):** The target is the actual return $G_t$:
    
     $\Delta w = \alpha(G_t - \hat{v}(S_t, w))\nabla_w \hat{v}(S_t, w)$ 
    
- **For TD(0):** The target is the TD target $R_{t+1} + \gamma \hat{v}(S_{t+1}, w)$
    
     $\Delta w = \alpha(R_{t+1} + \gamma \hat{v}(S_{t+1}, w) - \hat{v}(S_t, w))\nabla_w \hat{v}(S_t, w)$ 
    

**MONTE-CARLO WITH VALUE FUNCTION APPROXIMATION**

The return $G_t$ is an unbiased but noisy sample of the true value $v_{\pi}(S_t)$. This allows us to treat the episode data as a supervised training set:

 $\langle S_1, G_1 \rangle, \langle S_2, G_2 \rangle, \dots, \langle S_T, G_T \rangle$ 

**Linear MC Policy Evaluation:**

When using linear approximation, the update simplifies using the feature vector $x(S_t)$:

$\Delta w
= \alpha (G_t - \hat{v}(S_t, w)) \nabla_w \hat{v}(S_t, w) \\ 
= \alpha (G_t - \hat{v}(S_t, w)) x(S_t)$

**Procedure & Properties:**

1. **Storage:** Record all states visited during an episode.
2. **Calculation:** At the episode's end, compute the total return $G_t$ for each state.
3. **Update:** Use the (state, return) pairs as training data to update weights.

MC evaluation converges to a **local optimum**, even with non-linear function approximators (like neural networks). However, the primary drawback remains that updates can only occur after an episode terminates.

**TD WITH VALUE FUNCTION APPROXIMATION**

The TD target $R_{t+1} + \gamma \hat{v}(S_{t+1}, w)$ serves as a **biased** sample of the true value $v_{\pi}(S_t)$. We generate training data by pairing the **current state** with a **target** composed of the immediate reward and the estimated value of the successor state:

 $\langle S_1, R_2 + \gamma \hat{v}(S_2, w) \rangle, \langle S_2, R_3 + \gamma \hat{v}(S_3, w) \rangle, \dots, \langle S_{T-1}, R_T \rangle$ 

The update uses the TD error ($\delta$) and the feature vector $x(S)$:

$\Delta \mathbf{w} = \alpha (R + \gamma \hat{v}(S', \mathbf{w}) - \hat{v}(S, \mathbf{w})) \nabla_{\mathbf{w}} \hat{v}(S, \mathbf{w}) \\
= \alpha \delta \mathbf{x}(S)$

**Advantages & Convergence:**

- **Online Learning:** Updates are performed at every step, allowing the agent to learn during the episode without waiting for termination.
- **Convergence:** Linear TD(0) is mathematically guaranteed to converge near the global optimum.

### Action-value function approximation

The principles used for state-value approximation extend directly to the action-value function. The goal is to approximate $q_{\pi}(S, A)$ using a parameterized function $\hat{q}(S, A, w)$ that accounts for both the **current state** and the **action taken**.

**objective Function**

We aim to minimize the **Mean-Squared Error (MSE)** between the approximate action-value function $\hat{q}(S, A, w)$ and true action-value function $q_{\pi}(S, A)$:

 $J(w) = E_{\pi}[(q_{\pi}(S, A) - \hat{q}(S, A, w))^2]$ 

To find a local minimum, we apply SGD. The weight update is proportional to the prediction error and the gradient of the approximation function:

 $- \frac{1}{2} \nabla_w J(w) = (q_{\pi}(S, A) - \hat{q}(S, A, w))\nabla_w \hat{q}(S, A, w)$ 

 $\Delta w = \alpha(q_{\pi}(S, A) - \hat{q}(S, A, w))\nabla_w \hat{q}(S, A, w)$ 

The learning procedure is identical to the state-value approach, with the input space extended to incorporate the action.

**LINEAR ACTION-VALUE FUNCTION APPROXIMATION**

In the linear case, the relationship between features and values is straightforward. The state-action pair is represented by a **feature vector**:

 $x(S, A) = \begin{pmatrix} x_1(S, A) \\ \vdots \\ x_n(S, A) \end{pmatrix}$ 

The action-value function is the dot product of the feature vector and the parameter vector:

 $\hat{q}(S, A, w) = x(S, A)^T w = \sum_{j=1}^{n} x_j(S, A)w_j$ 

Since the gradient of a linear function is simply the feature vector itself $\nabla_w \hat{q}(S, A, w) = x(S, A)$, the update rule becomes:

 $\Delta w = \alpha(q_{\pi}(S, A) - \hat{q}(S, A, w))x(S, A)$ 

### Incremental control algorithms

As in prediction methods, we replace the unknown value $q_{\pi}(S, A)$ with a **target**:

- For **Monte Carlo (MC)**, the target is the **return** $G_t$
    
     $\Delta w = \alpha(G_t - \hat{q}(S_t, A_t, w))\nabla_w \hat{q}(S_t, A_t, w)$ 
    
- For **TD(0)** (**SARSA**), the target is the **TD target** $R_{t+1} + \gamma \hat{q}(S_{t+1}, A_{t+1}, w)$
    
     $\Delta w = \alpha(R_{t+1} + \gamma \hat{q}(S_{t+1}, A_{t+1}, w) - \hat{q}(S_t, A_t, w))\nabla_w \hat{q}(S_t, A_t, w)$ 
    

These updates adjust the parameter vector $w$ incrementally to reduce the difference between the predicted value and the chosen target.

### Convergence of prediction algorithms

Convergence guarantees vary significantly depending on the representation (Tabular, Linear, or Non-Linear) and the learning approach (On-Policy vs. Off-Policy).

![image.png](Reinforcement%20Learning/image%2011.png)

<aside>
📌

**Key Takeaways**

- **Monte Carlo (MC) prediction** is **theoretically robust**. It converges under standard assumptions regardless of whether a **tabular, linear, or non-linear (e.g., neural network)** representation is used. This stability applies to both **on-policy** and **off-policy** learning.
- **Temporal-Difference (TD) methods** have weaker theoretical guarantees. While convergence is guaranteed in the **tabular case**, it is not always mathematically guaranteed when using **linear or non-linear function approximation**.

Despite this limitation, TD methods are widely used in practice because they **learn significantly faster** than Monte Carlo methods. Convergence is generally achieved by using a **sufficiently small learning rate**.

Monte Carlo methods, although theoretically sound, can be **extremely slow**, making them impractical for many real-world problems.

</aside>

### Convergence of control algorithms

Control algorithms—where the policy is updated alongside the value function—face even stricter challenges, particularly with function approximation.

![image.png](Reinforcement%20Learning/image%2012.png)

**Note on "Chattering":** In linear cases, the value function may not settle on a single point but will "chatter" (oscillate) around a near-optimal solution.

## Batch Methods

While standard Gradient Descent is simple, it is often **sample inefficient** because it processes each experience once and then discards it. Batch methods address this by seeking the best-fitting value function based on the agent's entire history of experience ("training data").

**LEAST SQUARES PREDICTION**

Given a value function approximation $\hat{v}(s, w)$ and a dataset of experience $D$ containing $\langle \text{state, value} \rangle$ pairs:

 $D = \{ \langle s_1, v^{\pi}_1 \rangle, \langle s_2, v^{\pi}_2 \rangle, \dots, \langle s_T, v^{\pi}_T \rangle \}$ 

The goal is to find the parameter vector $w$ that minimizes the **Sum-Squared Error (LS)** between the approximate values  $\hat{v}(s_t, w)$ and the target values $v^{\pi}_t$ across the entire dataset:

 $*LS(w) = \sum_{t=1}^{T} (v^{\pi}_t - \hat{v}(s_t, w))^2*$ 

 $= E_D[(v^{\pi} - \hat{v}(s, w))^2]$ 

**STOCHASTIC GRADIENT DESCENT WITH EXPERIENCE REPLAY**

Experience Replay is a practical implementation of batch reinforcement learning. Instead of performing a single update per transition, the agent stores its experiences and repeatedly samples from them to train the model.

**The Process**

Given the experience dataset $D$:

1. **Sample:** Randomly select a $\langle \text{state, value} \rangle$ pair from the history: 
    
    $\langle s, v^{\pi} \rangle \sim D$
    
2. **Update:** Apply the stochastic gradient descent (SGD) update:
    
    $\Delta w = \alpha(v^{\pi} - \hat{v}(s, w))\nabla_w \hat{v}(s, w)$
    
3. **Repeat:** Iterate this process multiple times until it converges to least squares solution:
    
    $\mathbf{w} = \argmin_\mathbf{w} LS(\mathbf{w})$
    

### Experience replay in deep Q-networks (DQN)

DQN combines **Q-learning** with **deep neural networks**. Using non-linear function approximators with TD learning can be unstable, so DQN introduces two key stabilization techniques:

- **Experience Replay**
- **Fixed Q-Targets**.

Two neural networks are used:

- A **learnable network**, updated via gradient descent
- A **frozen (target) network**, providing stable Q-value targets

**The Learning Loop:**

1. **Action Selection:** The agent selects an action $a_t$ using an **$\epsilon$-greedy policy** based on the current network weights.
2. **Storage:** The resulting transition $(s_t, a_t, r_{t+1}, s_{t+1})$ is stored in a **Replay Memory ($D$)**.
3. **Sampling:** Instead of learning only from the most recent step, the agent samples a **random mini-batch** of transitions from $D$. This breaks the temporal correlation between consecutive samples.
4. **Target Calculation:** The agent computes the Q-learning targets using a separate set of **old, fixed parameters ($w^-$)**. This "Target Network" prevents the target from moving too quickly, which stabilizes the update.
5. **Optimization:** The agent minimizes the Mean-Squared Error (MSE) Loss function using a variant of Stochastic Gradient Descent:
    
     $L_i(w_i) = E_{s,a,r,s' \sim D_i} \left[ \left( r + \gamma \max_{a'} Q(s', a'; w^-_i) - Q(s, a, w_i) \right)^2 \right]$ 
    

After each parameter update, a new loss is computed using the latest mini-batch, and the agent may select new actions accordingly. Convergence occurs when Q-values stabilize.

<aside>
🧠

DQN is off-policy because it uses a **Behavior Policy** ($\epsilon$-greedy) to collect experiences into a buffer, but it updates its knowledge based on a **Target Policy** (purely greedy $\max Q$). Since the data being used for training can come from any previous policy stored in the Replay Buffer, the agent is effectively learning the optimal strategy by 'observing' its own past mistakes and successes.

</aside>

**DQN IN ATARI**

**DQN** is a **model-free**, **off-policy** reinforcement learning algorithm that learns the **action-value function** end-to-end directly from raw pixels, without the need for hand-engineered features.

It was used in **Atari games**:

- To provide the agent with a sense of motion and temporal context (such as the speed and direction of a ball), the **input state $s$** is not a single image. Instead, it is a **stack of raw pixels from the four most recent frames.**
- The input stack is processed through a Convolutional Neural Network (CNN) that automatically extracts spatial features (like the positions of paddles or enemies):
    - **Output Layer:** The final layer is a fully connected layer with **18 outputs**, each corresponding to a specific joystick/button configuration.
    - **Q-values:** Each output represents the estimated $Q(s, a)$ for that action, allowing the agent to select the one with the highest expected return.
- **Reward**: defined as the **change in game score** at each step.

![image.png](Reinforcement%20Learning/image%2013.png)