# Paper 01 — A genuinely new question: can a train-gradient coherence gate select context versus learned group structure?

**Status: pre-registered before any of these runs.** This is a new algorithm candidate, not the original author's KMeans centroid sharing and not a proof of originality from the formula alone. Strong motivation from independently completed evidence:

- At V6k/400step batch32 and at batch128/100step, a **fixed train-only co-occurrence group projector** wins vs early noisy online KMeans, while at V16k/V50k/400–800step batch32, dynamic online KMeans wins.
- The measured group-gradient energy ratio c=||PG||²/||G||² is ~3.4% for context at training step0 versus ~1% for untrained random groups. By step100–200 dynamic KMeans can reach ~5%, compared with shuffled controls ~1%. But high c by itself does not necessarily imply better final CE; this study tests whether a **lagged** energy gate can select the right representation at useful times.

## Rule fixed in advance

Maintain (1) fixed train-context partition P_context (64 groups); (2) online KMeans partition P_dynamic refreshed from current head rows every50 optimizer steps. Both are orthogonal class-mean projectors on *the same ordinary output head*.

At step t and unmodified head gradient G_t, compute c_context(t)=||P_context G_t||²/||G_t||², c_dynamic(t)=||P_dynamic G_t||²/||G_t||². Maintain EMA scores z_j(t)=0.9 z_j(t−1)+0.1 c_j(t), initialize with the first observation. **Choose the partition using *previous step's EMA***:

- At step0, use context.
- At t>0 choose dynamic iff z_dynamic(t−1) > **1.5 × z_context(t−1)**, otherwise choose context.
- Update the head gradient using `G_t'=(I+2 P_selected)G_t` before official Ember. Update both z scores **after** choosing so current-batch noise does not select itself.

The gate uses **only ordinary training gradients and train-only context preprocessing**, no validation/test values, no extra backward pass, no centroid params. K64 and α2 fixed; EMA .9 and ratio1.5 are a preregistered hypothesis, not tuned on this trial. Selection trace at every step is saved as aggregated dynamic-use fraction per 50-step window and the two observed energies at steps0/25/50/100/200/last.

## First falsification: V6k equal-token batch change

Original untied V6000/width32/2layers/2heads, FineWeb original 819,200 train tokens and fixed 600-window val, Ember head-only LR=.05 beta2=.95 eps1e−8; unchanged other optimizers.

**Fresh seeds 431/433/439** (not previously selected); batch32/400steps and batch128/100steps with gradient accumulation4; same training prefix in each regime. Four arms × two batches × three seeds = **24 trainings**:
1. Plain Ember.
2. Static train-context class projector α2.
3. Dynamic KMeans class projector α2, refresh50.
4. **New lagged-gradient-coherence-gated class projector**, as specified above.

Report per-seed val CE, target exposure bins, train wall time and fraction of steps using dynamic. Primary question is whether the gate matches or beats the *best* of the two fixed strategies **at both batch settings** on the same seeds; mere improvement over plain Ember is insufficient as static context already does that. If it fails, retain the negative result without threshold retuning on the same set.

A positive result motivates independent V16k/V50k and previously untouched heldout corpus confirmation, as well as direct Tree Loss reparam, semantic smoothing and concept-supervision comparisons. The mathematical mechanism of group sharing is old; the potential new contribution is **learning which group subspace is reliable for future gradients**.
