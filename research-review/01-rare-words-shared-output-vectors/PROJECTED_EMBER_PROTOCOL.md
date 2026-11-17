# Paper 01 — new candidate: class-projected Ember (pre-registered)

**Status:** protocol fixed before trial. This is a **new proposed variant**, not the upstream author’s `R=own+C[group]` method. It combines two known ingredients (learned output-head grouping and gradient preconditioning) and must demonstrate added benefit over both original grouping and official Ember before claiming algorithmic significance.

## New operator and distinguishing property

Let `R∈R^(V×d)` be a standard, **single-parameter** untied full-softmax head and `G=∇_R L`. Every 50 steps, cluster the current rows of `R` into `K=64` groups and store assignment matrix `A∈{0,1}^(V×K)`. Define `D=AᵀA` (per-group counts) and

`P=A D⁻¹ Aᵀ`,

the orthogonal projector onto vectors constant within each group (one independent projector per feature). The next backward-pass head gradient is replaced with

`G̃=(I+αP)G`

*before* sending it to the official Ember head optimizer (only head is Ember; other parameters use the unchanged original optimizers).

- `P=Pᵀ=P²`; its eigenvalues are 0 or 1, so `I+αP` has eigenvalues in `{1,1+α}` for α≥0, independent of vocabulary size and group population.
- `P1=1` for a true partition. Under an **untied full-softmax** head, `1ᵀG=0`; hence `1ᵀG̃=0`. This applies **to the gradient before Ember**, not necessarily to the nonlinear factored-optimizer parameter update.
- No extra trainable centroids, no `R=U+AC` parameterization, and no regroup-induced parameter or optimizer-state transportation. Only a moving group assignment and its gradient pooling are added.
- It changes the output-head **training rule**, but at inference it remains exactly the ordinary single-matrix head. It is mathematically distinct from the original split-parameter Adam method.

## Controlled V6k pilot

- Original `grid.py` scientific model, reconstructed public FineWeb, `V=6000,width32,layers2,heads2,L64,batch32,400steps`.
- New model seeds **211,307,401**, not used in V6k Ember selection or upstream published runs. CPU one thread.
- Group refresh 50 steps, 64 clusters. `alpha∈{0.5,1.0,2.0}` frozen from normalized-projector results that were **originally measured under Adam, not Ember**. This is a transfer, not a new Ember hyperparameter sweep.
- Full-softmax head Ember LR=.05/beta2=.95/epsilon1e−8 (best combined Ember configuration selected on earlier V6k seeds0/1/2). Unchanged body optimizers and original Adam settings for the split-centroid group control.
- Five arms per seed: `ember_plain`, `ember_original_groups` (`R=U+AC` with Ember on U and Adam on C), and `ember_projector_α` for each of three α. **15 full trainings**.

## Fixed analysis

- Primary: paired `Ember plain − Ember projected` and `Ember original groups − Ember projected` CE, all new seeds and all α.
- Report target exposure bins (0,1–9,10–31,32–99,100+), overall CE, group sizes, wall train time.
- Select **one** α for confirmatory V16k/V50k comparison only after seeing these 15 outcomes. This is a small-model pilot/selection study; its winning score must not be advertised as independent validation or state of the art.
- If direct projector is substantially weaker than original grouped Ember, record the negative result: this isolates the value of parameterized shared centroids and their distinct optimizer histories from pooled gradient geometry alone.

## Prior art/credit

The original output-head KMeans+reassignment method belongs to the [upstream author](../../01-rare-words-shared-output-vectors/PAPER.md). Cluster gradient pooling is already conceptually described there. Our prior [direct normalized projector](PROJECTOR_RESULTS.md) was tested only with Adam; the **new scientific question** here is whether a bounded, zero-sum-preserving, parameter-free group-gradient rule produces an independent performance/computation tradeoff **on top of a modern factored optimizer**. [Ember v4](https://arxiv.org/html/2607.01455v4) already establishes strong token-table second-moment optimization; do not claim the latter as our contribution.
