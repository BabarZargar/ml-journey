# Week 1 — Machine Learning Specialization Notes

## What is Machine Learning?
Teaching a computer to learn from data instead of explicitly programming every rule.
Linear regression is the simplest form — find a straight line that best fits your data.

---

## The 3 Core Concepts

### 1. Hypothesis (Prediction)
The model predicts output using a straight line:

```
ŷ = w * x + b
```

- `w` = weight (slope) — how much price changes per unit of size
- `b` = bias (intercept) — base value when x = 0
- `x` = input feature (e.g. house size)
- `ŷ` = predicted output (e.g. house price)

Goal of training: find the best `w` and `b`.

---

### 2. Cost Function (How Wrong Are We?)
Measures the average error of our predictions across all training examples:

```
J(w,b) = (1/2m) * Σ (ŷᵢ - yᵢ)²
```

- Squaring makes all errors positive and punishes big mistakes more
- The `1/2` cancels cleanly when we take the derivative (just convenience)
- Lower J = better model. Goal: minimize J.

---

### 3. Gradient Descent (How We Improve)
Repeatedly nudges `w` and `b` in the direction that reduces cost:

```
w := w - α * (∂J/∂w)
b := b - α * (∂J/∂b)
```

Where the gradients are:
```
∂J/∂w = (1/m) * Σ (ŷᵢ - yᵢ) * xᵢ
∂J/∂b = (1/m) * Σ (ŷᵢ - yᵢ)
```

- `α` (alpha) = learning rate — size of each step
- **Simultaneous update is critical** — compute both gradients first, then update both parameters. Never update w first and use the new w to compute db.

---

## The Learning Rate Problem

Alpha controls everything:

| Alpha | Result |
|-------|--------|
| Too large | Cost explodes → `nan` |
| Too small | Converges but extremely slowly |
| Just right | Cost decreases smoothly and flattens ✓ |

If you ever see `nan` after iteration 0 — your alpha is too large. Divide by 10 and try again.

---

## Feature Scaling (Most Important Lesson)

Raw house sizes (1000–3000 sqft) make gradients huge.
Huge gradients × any reasonable alpha = explosion.

**Fix: divide x by 1000** so values become 1.0–3.0.
Now gradients are small and a normal alpha works fine.

```python
x_train = np.array([1.0, 1.5, 2.0, 2.5, 3.0])  # already scaled
```

This is why almost every real ML dataset is normalized before training.

---

## How to Know if Your Model Converged

Plot cost vs iteration — two plots:
- **Start (first 100 iterations):** should show a sharp drop
- **End (last iterations):** should be completely flat

If the end plot is still a diagonal line going down → not converged yet.
Increase iterations or carefully raise alpha.

---

## The House Price Predictor

Trained a model from scratch on 5 data points:

```python
x_train = np.array([1.0, 1.5, 2.0, 2.5, 3.0])   # size (1000 sqft)
y_train = np.array([200, 270, 350, 430, 500])      # price ($1000s)
```

Final parameters after gradient descent:
- `w ≈ 150` — price increases ~$150k per 1000 sqft
- `b ≈ 50`  — base price ~$50k

Prediction: `price = w * size + b`

---

