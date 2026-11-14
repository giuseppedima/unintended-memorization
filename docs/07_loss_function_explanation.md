# Understanding Autoregressive Loss & Cross-Entropy in LLMs

## 1. The Objective Function

In causal language modeling (and continual pre-training), models are trained to minimize the **autoregressive Negative Log-Likelihood (NLL)** across a sequence of $T$ tokens:

$$
\mathcal{L}(\theta) = -\frac{1}{T}\sum_{t=1}^{T}\log \mathbb{P}_{\theta}(x_t \mid x_{<t})
$$

### Why This Function is Minimized

* **Chain Rule of Probability:** A language model aims to maximize the joint probability of observing a real sequence of text, which factorizes into conditional next-token probabilities:
  

  $$
  \mathbb{P}_\theta(x_1, \dots, x_T) = \prod_{t=1}^{T} \mathbb{P}_\theta(x_t \mid x_{<t})
  $$

* **Numerical Stability:** Taking the logarithm transforms products of small probabilities into sums, preventing floating-point underflow while preserving the optimal parameters.

* **Optimization & Normalization:** Multiplying by $-1$ turns maximization into minimization (for gradient descent), and dividing by $T$ averages the loss per token so gradients remain invariant to sequence length.

* **Context on Unintended Memorization:** When evaluating *unintended* memorization during continual pre-training, the loss is computed uniformly across **all** tokens ($t = 1 \dots T$) rather than masking exclusively for sensitive tokens ("secrets"). Computing loss only on secrets would explicitly force the model to memorize them, distorting the natural domain-adaptation setup.

## 2. Equivalence to Multi-Class Cross-Entropy

While often referred to as Negative Log-Likelihood in NLP, this objective is mathematically identical to standard **Multi-Class Cross-Entropy**.

### Step-by-Step Derivation

1. **General Multi-Class Cross-Entropy:**
   For a single prediction step $t$ over a vocabulary of $V$ classes, Cross-Entropy compares the ground-truth distribution $\mathbf{t}^{(t)}$ and the predicted distribution $\mathbf{y}^{(t)}$:
   

   $$
   E^{(t)} = -\sum_{k=1}^{V} t^{(t)}_k \log y^{(t)}_k
   $$

2. **Mapping to Next-Token Prediction:**

   * **Predicted probabilities (**$y^{(t)}_k$**):** The softmax output over the vocabulary given prior context, $\mathbb{P}_{\theta}(w_k \mid x_{<t})$.

   * **One-hot target (**$t^{(t)}_k$**):** Because there is only one true next token $x_t$ at position $t$, $t^{(t)}_k = 1$ for the true token ($w_k = x_t$) and $t^{(t)}_k = 0$ for all other $V - 1$ vocabulary tokens.

3. **Collapsing the Vocabulary Sum:**
   Multiplying by zero eliminates every incorrect token from the sum, leaving only the log-probability of the true token:
   

   $$
   E^{(t)} = -\log \mathbb{P}_{\theta}(x_t \mid x_{<t})
   $$

4. **Averaging Over the Sequence:**
   Because Cross-Entropy is an additive error function across data points, averaging $E^{(t)}$ across all $T$ positions yields the sequence loss:
   

   $$
   \mathcal{L}(\theta) = \frac{1}{T} \sum_{t=1}^{T} E^{(t)} = -\frac{1}{T}\sum_{t=1}^{T}\log \mathbb{P}_{\theta}(x_t \mid x_{<t})
   $$

## Summary Takeaway

* **Cross-Entropy** is the general distance measure between two probability distributions.

* **Negative Log-Likelihood (NLL)** is the simplified form of Cross-Entropy when the ground-truth distribution is **one-hot encoded** (a single deterministic target class).
