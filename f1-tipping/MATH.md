# The Mathematics of the F1 Tipping Optimiser

This document specifies exactly what the pipeline computes, derives the
formulas the code uses, and states precisely where approximations enter.
It is written to match the implementation line-for-line; where the model
is wrong or simplified, that is called out in an **Approximation** note
rather than glossed over.

Notation used throughout:

- $n$ — number of drivers in the fitted field.
- Finishing positions are **0-indexed**: position $0$ is the winner,
  position $9$ is tenth. A "top-$k$" finish means position $< k$.
- $\theta \in \mathbb{R}^n$ — driver strength (log-scale) parameters;
  $w_i = e^{\theta_i}$ is driver $i$'s Plackett–Luce weight.
- $\sigma(x) = 1/(1+e^{-x})$ — the logistic sigmoid.
- $d_i \in [0,1]$ — driver $i$'s probability of retirement (DNF).
- Market-implied (de-vigged) probabilities are written $\hat p$.

The pipeline is: **odds $\to$ de-vig $\to$ fit $\theta$ $\to$ Monte
Carlo $\to$ score tickets $\to$ optimise.** Each stage is a section
below.

Section 7 specifies the alternative **Bayesian** path (`--bayes`): the
point fit of Section 3 is replaced by a posterior
$P(\theta \mid \text{odds})$ and the single-$\theta$ Monte Carlo of
Section 4 by posterior-predictive simulation. Sections 5–6 (scoring,
optimisation) apply unchanged to either path; Section 7 defines its
additional notation in its opening table.

---

## 1. De-vigging: from prices to probabilities

A decimal price $o_i$ (stake returns $o_i$ including stake) implies a
gross probability

$$q_i = \frac{1}{o_i}.$$

For any complete market the $q_i$ sum to more than the true total
probability; the excess

$$\text{overround} = \sum_i q_i - T$$

(where $T=1$ for a winner market, $T=N$ for a "top-$N$" market, $T=1$ per
side of a two-way market) is the bookmaker/exchange margin, the "vig".
De-vigging removes it to recover probabilities summing to $T$. The
correct *allocation* of the margin across outcomes is not identifiable
from a single market — it depends on an unobservable bias model — so the
method is a modelling choice, offered two ways.

### 1.1 Winner market

**Proportional** (`devig_win(..., "proportional")`): assume the margin is
a uniform multiplicative inflation of every probability. Then

$$\hat p_i = \frac{q_i}{\sum_j q_j}.$$

This preserves all odds *ratios* $q_i/q_j$.

**Power** (`devig_win(..., "power")`): assume a favourite–longshot
distortion of the form $\hat p_i = q_i^{\,k}$, and choose $k$ so the
probabilities normalise:

$$\text{find } k \text{ s.t. } \sum_i q_i^{\,k} = 1.$$

$\sum_i q_i^{k}$ is strictly decreasing in $k$ (each $q_i<1$), so the
root is unique and found by bisection (`scipy.optimize.brentq`). Because
$q_i<1$, raising to a power $k>1$ shrinks small $q_i$ (longshots)
proportionally more than large ones, which is the empirically observed
direction of the favourite–longshot bias.

> **Approximation.** Neither method is "true". Proportional assumes the
> margin is scale-free in probability; power assumes a specific
> parametric bias. A single market cannot distinguish them — they are two
> priors. Default is proportional for the winner market (exchange
> midpoints carry little bias); it is configurable.

### 1.2 Top-$N$ markets (e.g. points = top 10)

Here exactly $N$ drivers finish in the top $N$, so the marginal
probabilities satisfy the **hard count constraint**

$$\sum_{i} \Pr(i \in \text{top }N) = N.$$

This is exact, not modelled: it is a deterministic property of any
ranking of the field. De-vigging must therefore scale the $q_i$ to sum to
$N$, not $1$. Default is the power method (`devig_topn(..., "power")`),
solving

$$\sum_i q_i^{\,k} = N,$$

for the same reason as above but with a right-hand side of $N$. The
alternative (`"proportional"`) scales linearly but must then **cap**:
a near-certain favourite priced at, say, $1.04$ has $q_i\approx 0.96$,
and linear upscaling can push it above $1$. The code pins any capped
runner at $\hat p = 0.999$ and rescales the remainder to absorb the
deficit, iterating until no runner exceeds the cap.

> **Approximation / rationale.** For top-$N$ markets the vig sits
> overwhelmingly in the longshots; proportional scaling drags
> near-certain favourites *below* the probability their winner and
> top-6 prices already imply, which no coherent race model can
> reproduce. Power keeps favourites pinned and removes the margin where
> it lives. This is why the default differs from the winner market.

### 1.3 Two-way markets: head-to-head and "To be Classified"

A two-runner market on "$a$ finishes ahead of $b$" is normalised to one:

$$\hat p(a \succ b) = \frac{q_a}{q_a + q_b}.$$

**"To be Classified"** is, per driver, the same two-outcome event; the
exchange lists it as two driver-runner markets ("Yes To be Classified"
and, when offered, "No To be Classified"). Driver $i$'s pair of prices
normalises exactly as above:

$$\hat p_i(\text{classified})
  = \frac{q^{(i)}_{\text{yes}}}{q^{(i)}_{\text{yes}} + q^{(i)}_{\text{no}}},$$

and supplies the DNF layer's marginal (Section 2.3) as
$d_i = 1 - \hat p_i(\text{classified})$. Note there is **no count
constraint across the field** here — $\sum_i \hat p_i(\text{classified})$
is the *expected* number of classified cars, not a fixed total — so
de-vigging is per driver, never field-wide. If only the Yes side is
quoted for a driver, its raw implied probability is used (capped at
$0.999$); that side's share of the vig stays in, which is still far
closer to truth than a flat prior. These markets are *inputs to the DNF
layer*, not fit targets: a driver priced only here does not join the
fitted field.

### 1.4 Liquidity weighting

Each market is given a weight for the fitting objective (Section 3) based
on matched volume $V$:

$$\omega = \min\!\Big(1.5,\ \max\big(0.2,\ \tfrac{1}{4}\log_{10}(1+V)\big)\Big),$$

with $\omega = 1$ when volume is unknown (e.g. manual CSV). A £100k
market is trusted more than a £500 one, but the clamp stops any single
deep market from dominating.

---

## 2. The race model: Plackett–Luce with a DNF layer

### 2.1 Plackett–Luce

The Plackett–Luce (PL) model defines a distribution over full orderings.
Given weights $w_i = e^{\theta_i}$, a ranking is generated by sampling
**without replacement**, drawing the next position from the softmax over
the remaining drivers. The probability of a full ordering
$\pi = (\pi_1, \pi_2, \dots, \pi_n)$ (position 1 first) is

$$\Pr(\pi) = \prod_{r=1}^{n} \frac{w_{\pi_r}}{\sum_{s \ge r} w_{\pi_s}}.$$

The first factor is the standard softmax winner probability; each
subsequent factor is the softmax over whoever is left. PL is a
**rank-1** model: one scalar $\theta_i$ per driver controls its behaviour
in *every* market simultaneously.

### 2.2 Sampling by the Gumbel-max trick

Sequential softmax sampling is $O(n^2)$ and not vectorisable. Instead the
engine uses the **Gumbel-max** representation (`sample_finish_positions`):

> **Theorem (Gumbel / Luce).** Let $G_1,\dots,G_n$ be i.i.d. standard
> Gumbel. Define perturbed scores $X_i = \theta_i + G_i$. Then sorting
> the drivers by $X_i$ in **decreasing** order yields a permutation
> distributed exactly according to Plackett–Luce with weights
> $w_i = e^{\theta_i}$.

*Why it holds.* The key fact is $\Pr(\arg\max_i X_i = j) = w_j/\sum_i w_i$
— the max of Gumbel-perturbed log-weights is softmax-distributed (Gumbel
1954; Yellott 1977). Because the Gumbel is memoryless in the right sense,
conditioning on the arg-max and removing it leaves the remaining $X$'s as
Gumbel-perturbed log-weights over the remaining drivers, whose max is
again their softmax — reproducing exactly the sequential-softmax
construction of Section 2.1. Sorting the $X_i$ once therefore samples a
full PL ranking. This is **exact**, not an approximation, and reduces a
race to one `argsort` over an $(\text{sims} \times n)$ array of
Gumbel-perturbed strengths.

The code computes, for $n_{\text{sims}}$ races at once,
$X_{s,i} = \theta_i + G_{s,i}$, then `argsort(-X)` to get the ordering and
an inverse permutation to get each driver's 0-indexed finishing position.

### 2.3 The DNF layer

Before ranking, each driver independently retires with probability $d_i$:

$$B_{s,i} \sim \text{Bernoulli}(d_i), \qquad
X_{s,i} \;\leftarrow\; X_{s,i} - D\cdot B_{s,i},$$

with $D = 10^6$ a demotion far larger than the span of the Gumbel noise.
A retired driver's score is pushed below every survivor's, so all DNFs
are classified behind all finishers, while **relative order among the
DNFs is preserved** (they all shift by the same $D$, keeping their PL
order). Survivors are then ranked normally.

This is the mechanism that matters for the underdog bonus: on a
high-attrition sample several front/mid runners retire together, and the
`argsort` promotes several midfield/underdog drivers into the points at
once. That co-movement is real structure the model produces, not an
add-on.

> **Approximation (important).** Retirements are modelled as
> **independent** Bernoulli events. Real DNFs are correlated — a
> first-corner pile-up, a safety car, weather, or a spec engine failure
> take out several cars together. The model captures *one* channel of
> attrition correlation (independent removals still jointly reshape the
> order) but **not** common-cause correlation.
>
> The *marginals* $d_i$, by contrast, are market-informed where the
> **"To be Classified"** markets price the driver:
> $d_i = 1 - \hat p_i(\text{classified})$ after two-way de-vigging
> (Section 1.3). The event even matches exactly — FIA classification is
> what the comp scores against, and the DNF demotion *is* the event
> "unclassified: ranked behind every classified driver", so a driver who
> retires late enough to still be classified is correctly not a DNF in
> either the market or the model. Precedence per driver: an explicit
> `dnf.per_driver` config override, then the market, then the flat
> season prior (0.10) for unpriced drivers. What the market does **not**
> fix is independence: it pins each marginal $d_i$ but says nothing
> about the joint distribution of retirements.

### 2.4 Marginal probabilities under the model

**Winner (closed form).** With $X_i \sim \text{Gumbel}(\theta_i, 1)$
independent, the winner marginal is the softmax, exactly:

$$\Pr(i \text{ wins}) = \frac{w_i}{\sum_j w_j}
= \frac{e^{\theta_i}}{\sum_j e^{\theta_j}}.$$

**Top-$k$ for $k>1$ (no closed form).** Driver $i$ finishes in the top
$k$ iff at most $k-1$ others exceed $X_i$. Conditional on $X_i = x$, the
count $\#\{j\ne i : X_j > x\}$ is a sum of independent (non-identical)
Bernoulli variables with success probabilities
$p_j(x) = \Pr(X_j > x) = 1 - \exp(-e^{-(x-\theta_j)})$ — a
**Poisson-binomial**. Hence

$$\Pr(i \in \text{top }k)
= \int_{-\infty}^{\infty} f_{X_i}(x)\,
  \Pr\!\Big(\textstyle\sum_{j\ne i}\mathbf{1}[X_j>x] \le k-1\Big)\, dx.$$

For $k=1$ the inner probability is $\prod_{j\ne i}(1-p_j(x))$ and the
integral evaluates to the softmax above. For $k>1$ there is **no
elementary closed form** (the Poisson-binomial tail has none, and the
integral is non-elementary), and adding the DNF layer only makes it
worse. This intractability is precisely why fitting uses a simulated
soft-rank surrogate (Section 3) and validation uses the hard Monte Carlo
(Section 4).

**Head-to-head (closed form, DNF-adjusted).** Pairwise, PL gives the
exact logistic marginal $\Pr(X_a > X_b) = \sigma(\theta_a - \theta_b)$
(the two-driver softmax). Layering independent DNFs and conditioning on
the four retirement cases:

| case | probability | $\Pr(a \succ b)$ |
|---|---|---|
| both finish | $(1-d_a)(1-d_b)$ | $\sigma(\theta_a-\theta_b)$ |
| only $b$ retires | $d_b(1-d_a)$ | $1$ |
| only $a$ retires | $d_a(1-d_b)$ | $0$ |
| both retire | $d_a d_b$ | $\sigma(\theta_a-\theta_b)$ |

(The last row: both are demoted equally, so their PL order stands.)
Summing, with $u_a = d_a(1-d_b)$ and $u_b = d_b(1-d_a)$ the "exactly one
retires" probabilities,

$$\Pr(a \succ b) = \big(1 - u_a - u_b\big)\,\sigma(\theta_a-\theta_b)
                  + u_b,$$

which is exactly `_analytic_h2h`. This is used directly in the fit
objective (no simulation needed for H2H).

---

## 3. Fitting the strengths $\theta$

### 3.1 Objective

$\theta$ is chosen to reproduce the de-vigged market probabilities. Let
$\mathcal{M}$ index the available top-$k$ markets (win $=$ top-1, plus
top-3/6/10 where present) and $\mathcal{H}$ the H2H markets. The loss is
a liquidity-weighted sum of squared errors plus an $L_2$ anchor:

$$
L(\theta) = \sum_{k\in\mathcal M} \omega_k \!\!
  \sum_{i \in \text{market }k}\!\! \big(P^{\text{model}}_{k}(i;\theta)
     - \hat p_{k}(i)\big)^2
\;+\; \sum_{(a,b)\in\mathcal H} \omega_{ab}\,
  \big(\Pr(a\succ b;\theta) - \hat p_{ab}\big)^2
\;+\; \lambda \lVert\theta\rVert_2^2 .
$$

H2H terms use the closed form of Section 2.4. The top-$k$ model
probabilities $P^{\text{model}}_k$ have no closed form, so they are
replaced by a **differentiable simulated surrogate**.

### 3.2 The soft-rank surrogate

Fix $M$ Monte Carlo draws (`fit_sims`, default 4000) of Gumbel noise and
DNF masks — fixed, so the objective is a deterministic, smooth function
of $\theta$ (common random numbers, Section 4.2). On draw $m$ the
perturbed, DNF-demoted scores are

$$s_{m,i} = \theta_i + G_{m,i} - D\,B_{m,i}.$$

The hard finishing rank of driver $i$ is
$\sum_{j} \mathbf 1[s_{m,j} > s_{m,i}]$ (number ahead of $i$), a
step function of $\theta$ with zero gradient almost everywhere. Replace
each indicator by a logistic of the score gap at temperature $\tau$:

$$\widetilde{\text{rank}}_{m,i}(\theta)
  = \sum_{j} \sigma\!\Big(\frac{s_{m,j}-s_{m,i}}{\tau}\Big) - \tfrac12 ,$$

the $-\tfrac12$ removing the $j=i$ self-term $\sigma(0)=\tfrac12$. Driver
$i$ is in the top $k$ iff its rank is $\le k-1$, i.e. below the threshold
$k-\tfrac12$; soften that indicator too and average over draws:

$$P^{\text{model}}_k(i;\theta)
  = \frac{1}{M}\sum_{m=1}^{M}
    \sigma\!\Big(\frac{(k-\tfrac12) - \widetilde{\text{rank}}_{m,i}(\theta)}{\tau}\Big).$$

This is exactly `objective` in `fit.py`. As $\tau \to 0$ both sigmoids
approach the hard step functions and $P^{\text{model}}_k$ approaches the
true simulated top-$k$ frequency; but the gradient also vanishes and the
surrogate becomes the non-smooth quantity we were avoiding.

> **Approximation (two distinct biases).** (1) $\widetilde{\text{rank}}$
> is a sum of sigmoids of score gaps, which equals the true count of
> drivers-ahead only in the $\tau\to0$ limit; for $\tau>0$ it smears
> near-ties. (2) The outer sigmoid smooths the top-$k$ threshold. Both
> biases are $O(\tau)$ and shrink the fitted probabilities toward $1/2$
> for borderline drivers. $\tau$ (default $0.15$) trades this bias
> against optimisability. **The bias is not assumed away — it is
> measured:** `validate.py` re-simulates the fitted $\theta$ with the
> exact hard engine (Section 4) and prints market-vs-simulated marginals,
> flagging any gap $>2$ percentage points.

### 3.3 Identifiability, regularisation, optimisation

PL is invariant to a constant shift, $\theta_i \mapsto \theta_i + c$
(it cancels in every softmax), so $\theta$ is identified only up to an
additive constant. The objective mean-centres $\theta$ on every
evaluation and returns a mean-centred solution; the $L_2$ term
($\lambda = 10^{-4}$) pins the remaining gauge and discourages runaway
strengths for near-zero-probability back-markers whose $\theta$ is weakly
constrained.

Optimisation is L-BFGS-B (`scipy.optimize.minimize`), initialised at
$\theta^{(0)}_i = \log \hat p_{\text{win}}(i)$ (clipped below), which is
the exact optimum if only the winner market existed. Gradients are
finite-difference with step $10^{-3}$: the objective runs in
`float32` for speed, where the default $10^{-8}$ step underflows the
dtype resolution and stalls the optimiser with zero gradients — the
surrogate is smooth, so a coarse step is safe.

> **Approximation (the load-bearing one).** PL is rank-1: a single
> $\theta_i$ per driver cannot in general match win, top-3, top-6 and
> top-10 *simultaneously*, because real markets encode more than a pure
> strength ordering (variance, track-specific pace, correlated team
> form). The fit therefore returns a **weighted-least-squares
> compromise**, not an exact match, and the validation table will
> typically show residual deviations of a few percentage points on the
> intermediate markets. This is a limitation of the model class, not a
> bug in the fit, and is the main reason to treat the output as a
> well-calibrated prior rather than ground truth.

---

## 4. Monte Carlo and common random numbers

### 4.1 The estimator

With fitted $(\theta, d)$, the engine simulates $N$ full races (default
$N = 10^5$) via Section 2. Any expectation — a marginal $\Pr(i\in\text{top
}k)$, a ticket's expected score, a score percentile — is estimated by its
sample analogue over the $N$ races. For a probability $p$ the standard
error is $\sqrt{p(1-p)/N}$; at $N=10^5$ that is $\le 1.6\times10^{-3}$,
i.e. sub-0.2-percentage-point noise, which is why validation flags real
model misfit ($>2$pp) rather than sampling noise.

### 4.2 Common random numbers (CRN)

Two uses of the *same fixed* simulation set:

1. **Fitting** holds its $M$ draws fixed across all $\theta$ evaluations
   so $L(\theta)$ is smooth (a fresh draw each call would inject noise and
   destroy the finite-difference gradient).
2. **Optimisation** scores every candidate ticket against the *same* $N$
   races. For comparing two tickets, what matters is
   $\mathrm{Var}(\text{score}_A - \text{score}_B)$; with CRN the two
   scores are positively correlated (a high-scoring race tends to lift
   both), and

$$\mathrm{Var}(\text{score}_A - \text{score}_B)
 = \mathrm{Var}_A + \mathrm{Var}_B - 2\,\mathrm{Cov}(A,B)$$

   is far smaller than under independent draws. Ticket *rankings* are
   therefore low-variance even where absolute EVs still carry sampling
   error.

---

## 5. Scoring

Scoring is delegated to the competition's own engine
(`src/scorer.py::Scorer`); this section states the maths that engine
implements, which the vectorised optimiser path reproduces exactly (and a
test asserts equality on random inputs).

### 5.1 Per-pick points

A ticket is an ordered list of 10 drivers; the pick in slot $s\in\{0,
\dots,9\}$ names the punter's predicted position-$s$ finisher. If that
driver's actual 0-indexed finishing position is $r$ (or "outside the top
10"), the base points are

$$
\text{base}(r, s) =
\begin{cases}
5 & r < 10 \text{ and } r = s & (\textbf{exact}) \\
3 & r < 10 \text{ and } |r-s| = 1 & (\textbf{close}) \\
1 & r < 10 \text{ and otherwise} & (\textbf{top-10}) \\
0 & r \ge 10 & (\textbf{miss})
\end{cases}
$$

### 5.2 Underdog multiplier

From round 2 onward, a pick of a driver outside the championship top-10
*before the race* scores double. With $\mathcal C$ the pre-race
championship top-10 and $\text{round}>1$,

$$m_d = \begin{cases} 2 & d \notin \mathcal C \\ 1 & d \in \mathcal C
\end{cases}, \qquad \text{points}(d, r, s) = m_d \cdot \text{base}(r, s).$$

(In round 1, or with no championship set, $m_d \equiv 1$.) A ticket's
score against one race is the sum over its 10 slots.

### 5.3 Additivity and the expected-points matrix

The total score is a **sum of independent per-pick terms** — there is no
interaction between picks. Therefore a ticket's expected score decomposes:

$$\mathbb E[\text{score}] = \sum_{s=0}^{9} m_{d_s}\,
   \mathbb E\big[\text{base}(R_{d_s}, s)\big],$$

where $R_d$ is driver $d$'s (random) finishing position. Define the
**expected-points matrix**

$$E[d, s] = m_d \cdot \frac{1}{N}\sum_{\text{races}} \text{base}(R_d, s),$$

an $(n \times 10)$ table estimated once from the simulation set
(`expected_points_matrix`). A ticket assigning driver $d_s$ to slot $s$
has expected score $\sum_s E[d_s, s]$.

> **Consequence, stated honestly.** Under the *current* additive rules,
> the expected score of a ticket depends **only on each chosen driver's
> marginal finishing-position distribution** — the joint distribution
> over finishing orders does **not** affect the EV. The elaborate joint
> simulation is justified for three other reasons: (i) it is how we
> obtain those marginals under PL+DNF, which have no closed form
> (Section 2.4); (ii) the DNF layer *shapes* the marginals (attrition
> mass in the tail promotes underdogs), so the marginals already encode
> the effect that matters; and (iii) the **score distribution**
> (p10/p50/p90) and any future "probability of beating the field"
> objective are genuinely joint and cannot be read off marginals. So the
> claim "we need the joint distribution" is true for variance and for the
> future objective, but **not** for the mean under today's rules — and
> the optimiser exploits exactly that (Section 6).

---

## 6. Ticket optimisation

### 6.1 Search space

A ticket is an ordered selection of 10 of the ~20 drivers, so the space
has

$$\frac{20!}{10!} = 20 \cdot 19 \cdots 11 \approx 6.7\times10^{11}$$

elements — far beyond brute force.

### 6.2 The additive objective is a linear assignment problem

By Section 5.3, maximising expected score is

$$\max_{\substack{d_0,\dots,d_9 \\ \text{distinct}}}
  \sum_{s=0}^{9} E[d_s, s],$$

i.e. choose an injective assignment of drivers to the 10 slots maximising
a sum of cell values in the matrix $E$. This is exactly the **linear
assignment problem**, solved to global optimality in polynomial time by
the Hungarian algorithm (`scipy.optimize.linear_sum_assignment` on
$-E$). So despite the $6.7\times10^{11}$ count, **the EV-optimal ticket
under the current rules is found exactly, not heuristically.**

### 6.3 Local search, restarts, runner-ups

The engine still runs steepest-ascent local search (swap two slots;
substitute an outside driver into a slot) from a probability-greedy seed,
the Hungarian seed, and random restarts, all under CRN. With additive
scoring this is a **verification and enumeration** layer: it confirms the
Hungarian optimum and supplies ranked runner-up tickets (the best
neighbours), rather than doing the primary optimisation. Its real purpose
is forward-compatibility:

> **When the assumption breaks.** If the competition rules are ever
> rewritten with cross-pick interactions (a bonus for a fully-correct
> podium, diminishing returns, a beat-a-rival objective), scoring is no
> longer additive, $E$ no longer summarises a ticket's value, and the
> assignment reduction fails. Setting `optimise.assume_additive: false`
> switches evaluation to the full per-race score of each candidate ticket
> (`ticket_scores`) over the CRN set, and local-search-with-restarts
> becomes the actual optimiser — now a genuine heuristic with no global
> guarantee. The code path exists; today's rules simply don't need it.

### 6.4 What is exact and what is estimated

The assignment is globally optimal **with respect to the estimated
matrix $\hat E$**. $\hat E$ itself carries Monte Carlo error $O(1/\sqrt
N)$, so the returned ticket is optimal for the simulated model, not
provably optimal for the true (unknown) race distribution. Reported
percentiles (p10/p50/p90) are exact order statistics of the chosen
ticket's scores over the CRN set — again, of the *model's* distribution.

---

## 7. Bayesian inference: the `--bayes` path

Section 3 returns a single $\hat\theta$ — a weighted-least-squares
compromise — and Section 4 simulates as though it were the truth. That
plug-in step quietly asserts that the markets pin the parameters down
exactly. They do not: prices are noisy, mutually inconsistent (the
rank-1 misfit of Section 3.3), and nearly uninformative at the back of
the field. The `--bayes` path makes the uncertainty explicit: the
de-vigged probabilities are treated as **noisy observations** of the
race model's parameters, Bayes' rule gives a posterior over those
parameters, and simulation **mixes over posterior draws** instead of
plugging in one point. This section defines the latent state, the
observation space, every distribution used and why, the two inference
engines, and what changes downstream. (Code: `model/bayes.py` builds
the posterior; `model/mcmc.py` and `model/importance.py` are the two
engines.)

Additional notation for this section:

| symbol | meaning |
|---|---|
| $x = (\theta, \eta, \gamma)$ | the latent state (Section 7.1) |
| $D_x = n + m + 1$ | state dimension ($D$ is taken by the DNF demotion, Section 2.3) |
| $\eta \in \mathbb R^m$ | logit-scale DNF parameters for the $m$ classified-priced drivers; $d_i = \sigma(\eta_i)$ |
| $\gamma = \log \sigma_{\text{obs}}$ | log of the global observation-noise scale $\sigma_{\text{obs}}$ |
| $\operatorname{logit}(p) = \log\frac{p}{1-p}$ | the inverse of the sigmoid $\sigma$ |
| $y_j,\ \mu_j(x),\ \omega_j$ | observation $j$: logit-scale value, model mean, precision multiplier |
| $\tau_p,\ \eta_0,\ s_d,\ \sigma_{\text{sc}}$ | prior hyperparameters (Section 7.3) |
| $M$ | fixed CRN draws behind the surrogate likelihood (`bayes.fit_sims`) |
| $K,\ S,\ R$ | ensemble walkers; retained posterior draws; races per draw |

### 7.1 The state: what is unknown

The latent state gathers everything the markets are informative about
but do not fix:

$$x = (\theta,\ \eta,\ \gamma) \in \mathbb R^{D_x}, \qquad
D_x = n + m + 1.$$

**$\theta \in \mathbb R^n$ — driver strengths.** Exactly the PL
parameters of Section 2.1; their meaning is unchanged, only their
epistemic status: a distribution over strength vectors instead of one
fitted vector.

**$\eta \in \mathbb R^m$ — retirement parameters, logit scale.** For
the $m$ drivers whose "To be Classified" markets are priced (Section
1.3), the DNF probability becomes latent: $d_i = \sigma(\eta_i)$.
Parameterising on the logit scale rather than sampling $d_i$ directly
has two payoffs: (i) $\eta_i$ is unconstrained, so the samplers never
hit the $[0,1]$ boundary; (ii) the classified observation becomes
*linear* in the state (Section 7.2) — the best-behaved likelihood
geometry available. Drivers **without** a classified price get no
$\eta$: their only likelihood leverage would be the weak, indirect
effect of $d_i$ on the top-$k$ marginals, so their posterior would
reproduce the prior while inflating $D_x$ — pure cost for both engines
(mixing for MCMC, proposal quality for IS). They stay fixed at their
Section 2.3 values (config override, else the season prior). With no
classified market at all, $m = 0$ and the state is $(\theta, \gamma)$
— the current pipeline's unknowns exactly.

**$\gamma = \log\sigma_{\text{obs}}$ — the observation-noise scale.**
One global scalar measuring how far, on the logit scale, de-vigged
market probabilities scatter around the model's marginals. It is
latent because its dominant component is not price noise but *model
misfit* (rank-1 PL, Section 3.3), whose size is unknown a priori. The
log transform enforces $\sigma_{\text{obs}} > 0$ by construction; the
Jacobian is carried in the prior (Section 7.3).

Everything else is fixed and known at inference time: the de-vig
outputs $\hat p$ and weights $\omega$ (Section 1), the CRN draws
behind the surrogate (Section 7.2), and the unpriced drivers' $d_i$.

**The generative story** (the model read forwards): draw $\theta$,
$\eta$, $\sigma_{\text{obs}}$ from the priors; the race model of
Section 2 then implies a marginal probability for every market
outcome; each market publishes that marginal corrupted by Gaussian
noise on the logit scale, with variance $\sigma^2_{\text{obs}}/\omega$.
Inference inverts this story.

### 7.2 The observation space: what the market says

The data vector stacks one observation per (market, runner) pair, all
mapped to logits. Three observation types:

| $j$ ranges over | observed $y_j$ | model mean $\mu_j(x)$ | precision $\omega_j$ |
|---|---|---|---|
| top-$k$ markets, $k \in \{1,3,6,10\}$, priced driver $i$ | $\operatorname{logit}\hat p_k(i)$ | $\operatorname{logit}\tilde P_k(i;\, \bar\theta, d)$ | $\omega_k$ (Section 1.4) |
| H2H pairs $(a, b)$ | $\operatorname{logit}\hat p_{ab}$ | $\operatorname{logit}\Pr(a \succ b;\, \bar\theta, d)$ — Section 2.4 closed form | $\omega_{ab}$ |
| classified, priced driver $i$ | $\operatorname{logit}\hat p_i(\text{classified})$ | $-\eta_i$ | $\omega_{\text{cls},i}$ |

with $\bar\theta = \theta - \tfrac1n\sum_i\theta_i$ the centred
strengths (gauge: Section 7.3) and all probabilities clipped to
$[\varepsilon, 1-\varepsilon]$, $\varepsilon = \max(10^{-4}, 1/(2M))$,
before the logit — the simulated surrogate cannot resolve
probabilities finer than $\sim 1/M$, and the clip keeps saturated
longshot prices from producing infinite logits.

Three remarks:

- **Top-$k$ means are the soft-rank surrogate $\tilde P_k$ of Section
  3.2**, over $M$ *fixed* CRN Gumbel draws and DNF uniforms
  ($B_{mi} = \mathbf 1[U_{mi} < d_i]$, recomputed against the sampled
  $d$ when $\eta$ moves). Fixing the draws makes the whole
  log-posterior a **deterministic** function of $x$ — the property
  both engines require. The same intractability that forced the
  surrogate in Section 3 forces it here: the exact top-$k$ marginal
  has no closed form (Section 2.4). Note $\tilde P_k$ is
  piecewise-constant in $\eta$ (the masks $B$ flip in steps of $1/M$);
  neither engine needs gradients, so this staircase is admissible —
  $d$ is identified chiefly through the smooth classified and H2H
  terms anyway.
- **The classified mean is linear in the state**:
  $\Pr(\text{classified}) = 1 - d_i = 1 - \sigma(\eta_i) =
  \sigma(-\eta_i)$, so $\operatorname{logit}\Pr(\text{classified}) =
  -\eta_i$. This upgrades the classified markets from the plug-in
  inputs of Section 1.3 to genuine observations: the market pins $d_i$
  *with a precision*, rather than fixing it outright.
- **Why logits and not probabilities?** (i) Probability-scale
  residuals are heteroscedastic: a 1 pp error against
  $\hat p = 0.5$ is noise, against $\hat p = 0.99$ it is a
  factor-of-two error in the implied longshot price. By the delta
  method a constant logit-scale noise corresponds to price noise
  proportional to $p(1-p)$ — shrinking toward the ends of the unit
  interval, roughly how price uncertainty actually behaves. (ii) The
  support is all of $\mathbb R$, so Gaussian noise never leaks outside
  $[0,1]$. (iii) $p$ and $1-p$ are treated symmetrically — backing and
  laying the same outcome give consistent likelihoods.

### 7.3 Priors, likelihood, posterior

**Likelihood.** Observations are conditionally independent given the
state:

$$y_j \mid x \;\sim\; \mathcal N\!\Big(\mu_j(x),\
\frac{\sigma^2_{\text{obs}}}{\omega_j}\Big)
\qquad\Longrightarrow\qquad
\ell(x) = \sum_j \Big[-\tfrac{\omega_j}{2}\,
\big(y_j - \mu_j(x)\big)^2\, e^{-2\gamma} \;-\; \gamma\Big]$$

up to constants. The $-\gamma$ per observation is the
$\log(1/\sigma_{\text{obs}})$ normaliser; it cannot be dropped because
$\sigma_{\text{obs}}$ is inferred — it is what stops the sampler
explaining every residual by inflating the noise.

> **Rationale (why a Gaussian pseudo-likelihood).** An exchange price
> is not a binomial count — it is the equilibrium of anonymous order
> flow, and it has no "true" sampling distribution. Any likelihood is
> therefore a modelling choice. Gaussian-on-logits is chosen for three
> reasons. (1) It is the maximum-entropy distribution given a mean and
> a variance — the weakest additional assumption beyond "the market is
> right up to some scatter". (2) The liquidity weights of Section 1.4
> slot in exactly as inverse-variance multipliers,
> $\mathrm{Var} \propto 1/\omega$: a £100k market is a *tighter*
> observation, formalising what the point fit did heuristically.
> (3) It nests the existing fit: with $\sigma_{\text{obs}}$ fixed and
> flat priors, the MAP of this model minimises the same
> liquidity-weighted squared error as Section 3.1, measured on the
> logit rather than the probability scale — the Bayesian path is a
> strict generalisation of the point fit, not a rival model.

**Why one global $\sigma_{\text{obs}}$, and why inferred?** Inferred,
because the residual scale is dominated by rank-1 misfit whose size is
not known in advance: markets that a single $\theta$ cannot jointly
satisfy leave irreducible residuals, the posterior of
$\sigma_{\text{obs}}$ rises to cover them, and every credible interval
downstream honestly widens. Global (one scale, not per-market
$\sigma_k$), because roughly a hundred observations identify one scale
well, whereas per-market scales are weakly identified and can
degenerate (a market's own $\sigma_k$ shrinking onto its residuals);
*relative* precision between markets is already carried by $\omega$.

**Priors.**

$$\theta_i \overset{\text{iid}}{\sim} \mathcal N(0, \tau_p^2), \qquad
\eta_i \overset{\text{iid}}{\sim} \mathcal N(\eta_0, s_d^2), \qquad
\sigma_{\text{obs}} \sim \text{HalfNormal}(\sigma_{\text{sc}}).$$

- **Strengths**, $\tau_p = 2$ (`bayes.tau_prior`). Weakly informative:
  fitted strengths span roughly $\pm 4$ after centring (front-runner
  to pinned backmarker), i.e. within two prior standard deviations, so
  the data dominate wherever markets are liquid. The prior's two real
  jobs: **(a) the gauge.** The likelihood depends on $\theta$ only
  through $\bar\theta$ (every $\mu_j$ is shift-invariant, Section
  3.3), so without a prior the posterior would be flat — improper —
  along the direction $\theta \mapsto \theta + c\mathbf 1$. A proper
  prior on *raw* $\theta$ fixes this: the mean component is a
  posteriori exactly its prior, $\mathcal N(0, \tau_p^2/n)$ — pure
  noise carrying no information — and every retained draw is
  mean-centred at save time, the same convention as the point fit.
  **(b) containment of longshots.** For a driver priced $1000.0$
  everywhere, the likelihood is nearly flat in $\theta_i$ below some
  threshold — the data only say "small enough". The prior supplies the
  missing curvature; it is the probabilistic counterpart of the $L_2$
  anchor of Section 3.1.
- **Retirements**, $\eta_0 = \operatorname{logit}(0.10)$ (the season
  prior `dnf.default_prob`), $s_d = 0.7$ (`bayes.dnf_prior_sd`). A
  logit-normal on $d_i$ centred on season attrition; $s_d = 0.7$ puts
  the central 95% at $d_i \in [0.03, 0.30]$ — generous around
  historical per-driver rates. Why logit-normal rather than the
  seemingly natural Beta: the state is $\eta$, and a normal on $\eta$
  keeps every prior term quadratic — the same geometry the samplers
  see everywhere else. A Beta on $d_i$ is an equivalent choice
  re-expressed through a Jacobian, with no identifiability gain: the
  classified *market*, not the prior, is meant to carry the
  information.
- **Noise scale**, $\sigma_{\text{sc}} = 0.5$ (`bayes.sigma_scale`).
  Half-normal: weakly informative, full mass at small scales (a
  near-perfect fit is not penalised), thin tail that resists "explain
  everything as noise". Scale intuition: $\sigma_{\text{obs}} = 0.5$
  on the logit scale is $\approx \pm 12$ pp around a 50% price at unit
  weight. Sampled as $\gamma = \log\sigma_{\text{obs}}$; the change of
  variables gives
  $\log p(\gamma) = -e^{2\gamma}/(2\sigma_{\text{sc}}^2) + \gamma +
  \text{const}$ (the $+\gamma$ is the Jacobian).

**Posterior.** Bayes' rule, up to an additive constant:

$$\log p(x \mid \text{odds}) \;=\;
-\frac{\lVert\theta\rVert^2}{2\tau_p^2}
\;-\; \sum_{i=1}^{m} \frac{(\eta_i - \eta_0)^2}{2 s_d^2}
\;-\; \frac{e^{2\gamma}}{2\sigma_{\text{sc}}^2} + \gamma
\;+\; \sum_j \Big[-\tfrac{\omega_j}{2}\big(y_j - \mu_j(x)\big)^2
e^{-2\gamma} - \gamma\Big] \;+\; \text{const}.$$

It is evaluated batched over many states at once; the surrogate runs
in `float32` (Section 3.3's reasons), residual accumulation in
`float64`.

> **Approximation (what "likelihood" quietly assumes).** Four caveats,
> stated rather than hidden. (i) *Within-market correlation*:
> de-vigged top-$k$ probabilities sum to $k$ by construction (Section
> 1.2), so one market's $n$ observations carry at most $n-1$ degrees
> of freedom; treating them as independent double-counts information
> and narrows the posterior somewhat. (ii) *Shared CRN*: every top-$k$
> mean is computed from the same $M$ draws, so surrogate errors are
> correlated across observations rather than independent. (iii)
> *Surrogate bias*: the $O(\tau)$ soft-rank bias of Section 3.2 sits
> in the likelihood **mean**, so the posterior concentrates on
> "$\theta$ that makes the *surrogate* match the market"; as with the
> point fit, the hard-engine validation (Section 7.6) is the check.
> (iv) *De-vig as data*: the proportional-vs-power choice of Section 1
> happens before inference and its uncertainty is not propagated. None
> of these is unique to the Bayesian path — the point fit makes
> (i)–(iv) implicitly — but a posterior invites reading its width
> literally, so the width's limitations are stated.

### 7.4 Engine I (default): affine-invariant ensemble MCMC

Nothing about this posterior is differentiable in a usable way (the
surrogate is `float32`, and $\tilde P_k$ is a $1/M$-staircase in
$\eta$), which rules out gradient-based samplers (HMC/NUTS). And with
$D_x \approx 23$–$45$ and per-coordinate scales that differ wildly — a
favourite's $\theta_i$ pinned to a few hundredths by four deep
markets, a backmarker's bound only by the prior — a hand-tuned
random-walk proposal is fragile. The **stretch move** (Goodman & Weare
2010; the `emcee` algorithm) needs only deterministic log-density
evaluations and is invariant under every affine transformation of the
state, so it self-adapts to the posterior's scales and correlations
with a single tuning constant.

An ensemble of $K$ walkers $\{x_1, \dots, x_K\}$ is split into two
halves, updated alternately. To move walker $x_j$: draw a partner
$x_k$ uniformly from the other half, draw the stretch factor

$$z = \frac{\big((a-1)u + 1\big)^2}{a}, \qquad u \sim U(0,1)
\quad\big(\text{density} \propto 1/\sqrt z \text{ on } [1/a, a],\
a = 2\big),$$

propose $y = x_k + z\,(x_j - x_k)$ — a stretch along the line through
the pair — and accept with probability

$$\min\!\Big(1,\ z^{\,D_x - 1}\, \frac{p(y)}{p(x_j)}\Big).$$

The $z^{D_x-1}$ factor is the volume correction that gives the move
detailed balance in $D_x$ dimensions. Walkers within one half move
conditionally independently given the other half, so each half-step is
a single batched evaluation of the log-posterior on a $(K/2, D_x)$
block.

Defaults (config `bayes:`): $K = \max(64,\ 2D_x)$ walkers (even, and
at least $2D_x$ so the ensemble spans the state space), $1200$ steps,
burn-in $400$, thinning $50$ — retaining $S \approx 1024$ draws.
Initialisation: $\theta$ at the Section 3 MAP plus small independent
jitter per coordinate (full-rank, so the ensemble is not trapped in an
affine subspace — the stretch move's known failure mode), $\eta$ at
the market-implied logits, $\gamma$ at $\log 0.15$.

Convergence is *diagnosed, not assumed*: the acceptance fraction
(healthy $\approx 0.2$–$0.5$), split-$\hat R$ across walker
half-chains (flag $> 1.05$ — with the honest caveat that walkers
interact, so this understates disagreement relative to truly
independent chains), and the integrated autocorrelation time
$\tau_{\text{int}}$ (flag when the post-burn run is shorter than
$50\,\tau_{\text{int}}$).

### 7.5 Engine II: importance sampling (`bayes.method: is`)

MCMC pays for robustness with *sequential* evaluations ($\sim$77k at
the defaults, $\approx$ 10–20 min). When odds move an hour before
lock-in, a cheaper answer helps. Importance sampling replaces the
chain with a **one-shot, fully batchable** estimate ($\sim$8–13k
evaluations, $\approx$ 2–4 min).

**The estimator.** For a proposal density $q$ covering the posterior's
support and the unnormalised posterior $p^*$ (the exponential of the
Section 7.3 expression):

$$\mathbb E\big[h(x) \mid \text{odds}\big]
\;\approx\; \sum_{r=1}^{N_{\text{IS}}} \tilde w_r\, h(x_r),
\qquad x_r \sim q, \quad
w_r = \frac{p^*(x_r)}{q(x_r)}, \quad
\tilde w_r = \frac{w_r}{\sum_{r'} w_{r'}}.$$

Self-normalisation cancels the unknown evidence; the estimator is
consistent, with $O(1/N_{\text{IS}})$ bias.

**The proposal** is a Laplace approximation with fattened tails:
maximise $\log p^*$ (L-BFGS-B, the same finite-difference discipline
as Section 3.3) to get the mode $x^\ast$; take a finite-difference
Hessian $H$ of $-\log p^*$ there; **floor its eigenvalues at the prior
curvatures** ($1/\tau_p^2$, $1/s_d^2$, $1/\sigma_{\text{sc}}^2$) so
that directions the data leave flat (pinned longshots) default to the
prior's scale rather than exploding; then

$$q = t_\nu\big(x^\ast,\ c^2 H^{-1}\big), \qquad \nu = 7,\ c = 1.3.$$

> **Rationale (why Student-$t$, not the Gaussian Laplace itself).**
> Importance weights have finite variance only if $q$'s tails are at
> least as heavy as the posterior's. A Gaussian proposal even slightly
> too narrow anywhere gives infinite-variance weights — a handful of
> draws dominate and the estimate silently degrades. The $t_7$
> dominates every Gaussian in the tails, and the inflation $c$ buys
> margin against skew; both are cheap insurance priced in effective
> sample size, not bias.

**Diagnosed, not assumed**, in the same spirit as Section 7.4: the
effective sample size $\mathrm{ESS} = 1/\sum_r \tilde w_r^2$ (how many
unweighted draws the weighted set is worth) and the **Pareto-$\hat k$**
of PSIS (Vehtari et al.): fit a generalised Pareto distribution to the
largest $\min(0.2\,N_{\text{IS}},\ 3\sqrt{N_{\text{IS}}})$ weights;
$\hat k$ estimates the tail index of the weight distribution.
$\hat k < 0.5$: weight variance finite, estimates reliable;
$0.5 < \hat k < 0.7$: usable, slower convergence; $\hat k > 0.7$: the
proposal misses the posterior — the run **warns and directs to
`bayes.method: mcmc`** rather than returning a quietly wrong answer.
The largest weights are replaced by their fitted Pareto quantiles
(PSIS smoothing), trading a little bias for much variance.

**Resampling to draws.** Downstream plumbing wants unweighted draws,
so the weighted set is reduced by **sampling-importance-resampling**:
systematic resampling of $S$ indices with probabilities $\tilde w_r$
(systematic rather than multinomial: strictly lower resampling
variance). Duplicated draws are harmless — each simulated race
re-draws its own Gumbel noise (Section 7.6).

**Sequential reweighting when odds move.** Because draws and their
$\log p^*$ values are stored, a *new* snapshot needs no new sampling:
reweight the existing draws by

$$\log w_s = \log p^*_{\text{new}}(x_s) - \log p^*_{\text{old}}(x_s),$$

apply the same PSIS diagnostic and resampling — seconds of work. This
is one step of sequential importance sampling with the old posterior
as the proposal; it is trustworthy exactly when the odds moved little,
and $\hat k$ says when they moved too much (then re-fit fully).

> **Approximation.** IS is exact only as $N_{\text{IS}} \to \infty$
> and only as good as its proposal. In $D_x \approx 45$ a single
> Laplace-$t$ can be a poor global fit to a correlated posterior, and
> IS cannot see modes the proposal misses (the least-squares structure
> makes multimodality unlikely, but that is an assumption). $\hat k$
> measures the failure per run instead of assuming it away.

### 7.6 Posterior-predictive simulation and validation

The object the rest of the pipeline needs is the posterior-predictive
distribution over finishing orders:

$$P(\text{order} \mid \text{odds})
= \int P(\text{order} \mid \theta, d)\;
  dP(\theta, d \mid \text{odds}),$$

approximated by simulating $R = \lfloor N/S \rfloor$ races from each
retained draw with the exact Section 2 engine (one batched Gumbel-max
call, one seed) and concatenating into a single `SimSet` of $S \cdot R$
races. Downstream, nothing changes: scoring and optimisation consume
`SimSet` exactly as in Sections 5–6; the expected-points matrix
estimated on this set makes the Hungarian ticket the exact maximiser
of *posterior-predictive* EV under additivity; CRN comparisons across
tickets hold because it is one fixed set of races.

The races are exchangeable but **not** i.i.d. — blocks of $R$ share a
parameter draw. Marginals and EVs remain unbiased; what changes is the
*spread*. By the law of total variance,

$$\mathrm{Var}(\text{score})
= \underbrace{\mathbb E_{\theta,d}\big[\mathrm{Var}(\text{score}
  \mid \theta, d)\big]}_{\text{race luck (Section 4)}}
+ \underbrace{\mathrm{Var}_{\theta,d}\big(\mathbb E[\text{score}
  \mid \theta, d]\big)}_{\text{parameter uncertainty (new)}},$$

so the reported p10/p50/p90 now include what the markets genuinely do
not know.

Validation upgrades from a threshold to a calibration test: for each
market probability the table reports the posterior-predictive marginal
and a 90% credible interval (per-draw hard-engine marginals across a
subsample of draws), and flags a market whose de-vigged probability
falls **outside its interval** — replacing the fixed 2 pp rule of
Section 3.2, which cannot distinguish misfit from honest uncertainty.
The posterior median of $\sigma_{\text{obs}}$ is printed as the
one-number summary of model-vs-market misfit.

### 7.7 What the posterior buys, honestly

Under today's additive rules the EV-optimal ticket depends only on the
expected-points matrix (Section 5.3), and averaging that matrix over
posterior draws typically moves it little — **the Bayesian path will
often pick the same ticket as the point fit.** The genuine gains are
elsewhere: score percentiles that include parameter uncertainty rather
than overstating confidence; validation that asks "is the market
inside the model's own uncertainty?" instead of an arbitrary
threshold; DNF probabilities identified from the classified markets
*with* a precision attached, not plugged in; near-free re-fits when
odds move (Section 7.5); and the piece a future "probability of
beating the field" objective needs, since that objective is exactly a
posterior-predictive tail probability.

---

## 8. Summary of approximations

| Stage | Exact | Approximated | Controlled / checked by |
|---|---|---|---|
| De-vig | count constraint ($\sum=N$), two-way normalisation (H2H, classified) | how margin is split across outcomes (proportional vs power) | configurable; exchange midpoints carry little vig |
| PL sampling | Gumbel-max gives exact PL orderings | — | — |
| DNF layer | demotion preserves PL order among DNFs; $d_i = 1-\hat p(\text{classified})$ where the market is priced | retirements independent; flat $0.10$ prior for unpriced drivers | classified markets pin the marginals; joint correlation still unmodelled |
| Marginals | win (softmax), H2H (logistic, DNF-adjusted) | top-$k$, $k>1$: no closed form | estimated by Monte Carlo |
| Fit | H2H terms; identifiability handled | soft-rank surrogate, bias $O(\tau)$; rank-1 PL can't match all markets | `validate.py` hard-sim table, 2pp flags |
| Monte Carlo | unbiased estimator | sampling error $O(1/\sqrt N)$ | $N=10^5 \Rightarrow <0.2$pp |
| Scoring | matches `Scorer` exactly (tested) | — | equality test on random inputs |
| Optimise | Hungarian = global optimum **under additivity** | additivity assumption; $\hat E$ has MC error | exact today; `assume_additive: false` for future rules |
| Bayes likelihood (§7) | H2H and classified means; gauge pinned by the prior | Gaussian pseudo-likelihood on logits; within-market independence; surrogate bias in the mean | inferred $\sigma_{\text{obs}}$; CI-based validation |
| Bayes sampling (§7) | stretch move satisfies detailed balance; log-posterior deterministic under CRN | finite chain (mixing); IS proposal mismatch | acceptance, $\hat R$, $\tau_{\text{int}}$; ESS, Pareto $\hat k$ — printed with thresholds |
| Posterior predictive (§7) | unbiased mixture over draws | $S \times R$ granularity (races share draws in blocks) | $S \approx 1024$ draws; validation CIs from dedicated per-draw sims |

The single most consequential approximation is **rank-1 Plackett–Luce**
(Section 3.3): it is why the fitted marginals do not exactly reproduce
every market, and why the output is best read as a disciplined,
odds-anchored prior over finishing orders — not a precise truth. Every
other approximation is either exact in the relevant limit, measured by
validation, or driven below the noise floor by simulation count. The
Bayesian path (Section 7) does not remove that misfit — it **measures**
it: the inferred noise scale $\sigma_{\text{obs}}$ is precisely the
realised market-vs-model discrepancy, and the posterior (hence every
credible interval downstream) widens with it.
