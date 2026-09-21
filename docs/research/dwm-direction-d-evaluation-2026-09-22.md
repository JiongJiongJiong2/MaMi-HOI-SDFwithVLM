# DWM Direction D Evaluation: Stop D1, Test One Decision-Focused Probe Design

Date: 2026-09-22

Status: research decision report. This report does not reopen the frozen
D1 gate, train a new model, or authorize MaMi integration.

## Abstract

The current D1 formulation should stop. Its regression objective does not
identify object response well enough on held-out object configurations, and
the failure is not explained by a single missing layer or by a small loss
weight. A post-hoc server audit shows that the D0 branches contain real
action-dependent outcomes: a nonlinear branch model improves centered
prediction on held-out resets of seen objects, but still fails on unseen
mass/friction configurations. The problem is therefore better described as
unidentifiable hidden response under the current observation model than as
the complete absence of action information.

The broad Direction D question remains worth pursuing, but the generic
design "add a probe, infer physical parameters, and predict a world model"
is already occupied by active tactile perception, interactive system
identification, context adaptation, and counterfactual quotient learning.
The only defensible next formulation is narrower: use a short interaction
probe to make hidden object response identifiable, then learn a
decision-sufficient model for ranking candidate action chunks. The first
experiment must test decision utility without an active probe policy. Active
probe selection is a second experiment and should be removed if it does not
beat random or fixed probing at the same budget.

This report gives a qualified **Accept with Revisions** for that new design,
conditional on a decisive ranking pilot. It gives **Reject and Pivot** for
the frozen D1 initial-state-to-action regression.

## 1. Research Brief

The investigation answers three questions.

**RQ1. Is the D1 NO-GO a true identifiability failure, a training defect, or
an evaluation-design problem?** The frozen D1 result is a real failure of
the current formulation. The new audit narrows the cause: action effects
exist and are partly learnable on seen response regimes, but they do not
transfer to unseen mass/friction when only one initial state is observed.
That is a hidden-response generalization failure, not proof that action has
no physical effect.

**RQ2. How do other fields solve the same problem?** They do not use one
general solution. World-model work diagnoses action consistency or predicts
action differences; system-identification work chooses informative inputs
and enforces excitation; robotics perception estimates physical properties
from deliberate interaction; context-conditioned dynamics models infer a
latent environment factor from history. Each branch solves one part of D,
but none retrieved here combines all parts under a decision-utility gate.

**RQ3. Should Direction D continue, and if so in what form?** Continue only
with a narrow decision-focused pilot. Stop the initial-state-to-action
regression. Do not start with a large world model, explicit mass/friction
labels, active probe learning, and MaMi integration at the same time.

## 2. Method and Evidence Standard

The literature search used arXiv API queries across four perspectives:
action identifiability and controlled world models; decision utility and
counterfactual ranking; active or interactive system identification; and
contact-rich HOI or manipulation representations. Crossref was used to
verify older journal and conference metadata, including the Science Robotics
work on manipulation for self-identification [25]. The search produced 141
papers in the initial broad corpus, 151 in a targeted probe corpus, and 69
in a final identifiability-and-experiment-design query. Candidate papers were
deduplicated and checked against title, abstract, and metadata.

Claims about mechanism are limited to each paper's title and abstract unless
otherwise stated. Reported numerical gains belong to the original authors'
settings and are not treated as gains for this project. This is not a
priority proof: "no fully overlapping work retrieved" does not mean
"first".

The local evidence base is the frozen D0-D2 protocol and D1 report:

```text
docs/experiments/dwm-d0-d2-protocol-2026-09-22.md
docs/experiments/dwm-d1-identifiability-result-2026-09-22.md
docs/experiments/dwm_d1_identifiability_summary_v1_20260922.json
```

A post-hoc read-only audit of the full D0-B arrays was added in:

```text
scripts/audit_dwm_action_effects.py
docs/experiments/dwm_action_effect_audit_v1_20260922.json
```

The audit did not alter the D0 data or the frozen gate. Its purpose is
diagnostic: to separate the absence of action effect from the failure to
use or generalize an existing effect.

The audit can be rerun with:

```bash
python scripts/audit_dwm_action_effects.py \
  --train-npz /root/autodl-tmp/dwm_d0b_full_v1_20260922/train.npz \
  --val-npz /root/autodl-tmp/dwm_d0b_full_v1_20260922/val.npz \
  --test-npz /root/autodl-tmp/dwm_d0b_full_v1_20260922/test.npz \
  --output docs/experiments/dwm_action_effect_audit_v1_20260922.json
```

## 3. What the Local Evidence Now Shows

### 3.1 D1 failed on the original promotion gate

The object-held-out and reset-held-out runs did not achieve the required
30% relative reduction over shuffled or zero actions. At h8, true actions
did not beat the controls on object translation or rotation. Contact-mode
macro-F1 improved by only about 0.03 rather than the required 0.10. The
state-centered target, corrected loss weighting, and seen-object
reset-held-out split did not change the direction.

This result is sufficient to reject the frozen formulation. Re-running the
same model with a larger MLP or another seed would not change the scientific
question; it would only search for a favorable point in a formulation whose
controls are already stronger.

### 3.2 Action effect exists, but hidden-response generalization is poor

The post-hoc audit computed branch-dependent object deltas after removing
the reset-level common component. In training, the mean centered response
norm across the 13 branches was about 5.84 mm translation and 0.106 in the
6D rotation representation. In test, the corresponding values were about
4.24 mm translation and 0.110 rotation. The branches therefore do not
produce identical outcomes.

A nonlinear `ExtraTreesRegressor` using selected initial-state features and
a one-hot encoding of the fixed branch improved centered prediction on a
held-out reset split of the seen objects:

| Split | Branch input | Translation L1 | Rotation L1 | All-9 L1 |
|---|---|---:|---:|---:|
| reset-held-out, seen objects | zero | 0.003366 m | 0.04207 | 0.02917 |
| reset-held-out, seen objects | true branch | 0.002574 m | 0.03327 | 0.02303 |
| object-held-out test | zero | 0.001845 m | 0.03523 | 0.02410 |
| object-held-out test | true branch | 0.002148 m | 0.03947 | 0.02703 |

The audit therefore splits the interpretation. Action-dependent structure
is learnable when the response regime has been seen. It does not transfer
to an unseen mass/friction configuration from one initial state. The
initial-state-to-action MLP did not expose this distinction because its
absolute and centered metrics were already dominated by state variation,
common settling, and contact noise.

This is consistent with the identifiability condition in [1]: the
counterfactual error can be amplified when the weakest action-excitation
margin is small. It is also consistent with the online latent-property
results in [13], [14], and [15]: histories or interaction evidence are used
to infer a context that the current state alone does not contain.

The audit is exploratory, uses a branch one-hot rather than the original
action tensor, and is not a replacement for D2. It establishes a direction
for the next test, not a positive D result.

## 4. Literature Taxonomy

### 4.1 Branch A: Diagnose and learn action-dependent effects

The closest work to the original D1 question is **On the Identifiability of
Controlled World Models** [1]. It proves conditions under which a controlled
world model can recover latent representation and controlled dynamics, and
relates counterfactual prediction error to non-degenerate action variation.
This directly supports the conclusion that D1 must be judged by the
excitation and conditional action margin, not only by average prediction
loss.

**ATM** [2] compares action semantics in real and predicted latent
transitions using post-hoc probes. It turns action identifiability into a
fast diagnostic and a training signal. It does not infer an unseen object
response from a new interaction, so it addresses diagnosis and improvement
of a learned latent model rather than the hidden-response problem here.

**Counterfactual Quotient Models** [3] remove the component of a future that
is shared across candidate actions and learn the remaining action-dependent
effect directly from synchronized counterfactual rollouts. The paper reports
that this representation supports unseen reward queries and improves action
ranking relative to absolute-future prediction. This is a direct warning
against presenting "learn centered deltas and rank actions" as a new method.
Its limitation for this project is that the latent dynamics regime is
assumed to be available during training; it does not explain how to acquire
the response context of an unseen object at test time.

**Decision-Metric Alignment in Latent World Models** [4] shows that strong
decoding of task variables does not guarantee that a latent cost ranks
candidate plans by real progress. It introduces rank-agreement diagnostics
and action-conditioned objectives. This is important for D2: object-pose
regression error is not the same endpoint as choosing the better action
chunk.

The causal benchmarks **What-If World** [5] and **ACWM-Phys** [6] reinforce
the same conclusion from evaluation. What-If World pairs prompts that change
one physical variable and finds that current systems fail many causal
interventions. ACWM-Phys reports larger generalization drops for deformable
contacts, high-dimensional actions, and complex articulated motion, with
evidence that appearance remains a strong shortcut. These are benchmarks,
not solutions, but they define the failure mode that a new D method must
beat.

| Work | Mechanism | Strongest relevant claim | What it does not supply |
|---|---|---|---|
| On the Identifiability of Controlled World Models [1] | Theoretical identifiability conditions | Counterfactual error depends on action-excitation margin | An interaction policy for hidden object response |
| ATM [2] | Post-hoc action-consistency probe and AITS training signal | Action identifiability diagnoses and improves latent world models | Probe acquisition or hidden physical response inference |
| Counterfactual Quotient Models [3] | Centered action-effect representation | Removes action-shared variation and improves action ranking | Test-time adaptation to a new response regime |
| Decision-Metric Alignment [4] | Plan-rank diagnostics and inverse-dynamics objectives | Prediction quality does not imply ranking utility | Active acquisition of missing dynamics |
| What-If World [5] | Paired causal interventions | Current world models fail one-variable interventions | A corrective model |
| ACWM-Phys [6] | OOD physical benchmark | Physical generalization is weak under contact and high-dimensional action | A probe-based identification method |

### 4.2 Branch B: Acquire information through interaction

System identification already treats input choice as part of the problem.
**Active Learning for Nonlinear System Identification with Guarantees** [7]
chooses trajectories to explore feature directions and shows parametric
rates for a class of nonlinear systems. **High Effort, Low Gain** [8]
characterizes when excitation design can help and proposes an active
algorithm for sequentially exciting a system. These works make clear that
"actively choosing informative actions" is an established principle rather
than a new technical idea.

Robotics and tactile perception provide the closest implemented versions.
**Learning active tactile perception through belief-space control** [9]
combines a generative world model, differentiable Bayesian filtering, and
an information-gathering controller to estimate mass, height, or toppling
height. **Push to know!** [10] uses active dual differentiable filtering to
select where and how to push and estimates shape, friction, mass, center of
mass, and inertia. **Predictive Visuo-Tactile Interactive Perception** [11]
selects exploratory push or pull actions by information gain and explicitly
uses the inferred properties to anticipate outcomes of manipulation.
**Interactive Learning of Physical Object Properties** [12] selects actions
by expected information gain and stops when further measurement is not
useful.

These works are a strong negative constraint on novelty. A new paper cannot
claim the general mechanism of active probing for physical-parameter
inference. The remaining defensible difference must be the output and
decision criterion: whether the probe is selected to improve the ranking of
future candidate actions under hidden response, rather than to estimate an
interpretable parameter as accurately as possible.

**Phys2Real** [13] conditions a policy on VLM physical priors and refines
them with interaction data and uncertainty-aware fusion. **EDO-Net** [14]
learns a latent elastic-property representation from a pulling interaction
without property labels and uses it for forward dynamics. **Context-Aware
Deep Lagrangian Networks** [15] combine recurrent online system
identification with a context-aware physical model and MPC. **ProtoCAD**
[16] learns a temporally consistent latent context for dynamics
generalization in model-based RL. **Shake to Learn** [17] uses a fixed
dynamic interrogation to infer hidden center of mass, and **PAC-NeRF** [18]
estimates geometry and physical parameters from video through
differentiable physics. Together they cover history-conditioned response
inference, label-free physical latents, and probe-based hidden-property
estimation.

| Work | Interaction or context | Inferred object state | Downstream endpoint | Remaining opening for D |
|---|---|---|---|---|
| Active tactile belief-space control [9] | Information-gathering tactile policy | Mass, height, toppling height | Property estimation | Does not optimize counterfactual action ranking |
| Push to know! [10] | N-step active push | Shape, friction, mass, CoM, inertia | Parameter inference | Not decision-focused ranking of action chunks |
| Predictive visuo-tactile perception [11] | Active push or pull | Multiple physical properties | Outcome anticipation | Robot objects and observations, not dexterous hand counterfactuals |
| Phys2Real [13] | Online interaction update | CoM-like parameters | Policy success | Parameter-conditioned control, not action-effect quotient |
| EDO-Net [14] | Pulling history | Latent elastic properties | Forward graph dynamics | Deformable objects, no active identifiability objective |
| Context-aware DeLaN [15] | Recurrent online system identification | Dynamics context | MPC tracking | Continuous model control, not candidate ranking under contact |
| ProtoCAD [16] | Trajectory context and prototypes | Latent dynamics context | Model-based RL | No deliberate probe or counterfactual quotient |
| Shake to Learn [17] | Fixed shaking excitation | Hidden center of mass | Regrasp feasibility | Fixed probe, not learned probe selection for ranking |

### 4.3 Branch C: Represent contact and object response

Contact representation remains a separate constraint. **ContactWorld** [19]
finds that preserving spatial structure and temporal continuity improves
contact-rich prediction and planning, and that tactile data help when the
representation is compatible with vision. **FORCE** [20] explicitly models
the relation between human force and object resistance using mass and
friction attributes. **ContactNets** [21] represents signed distance and
contact-frame Jacobians with complementarity and maximum dissipation, and
reports realistic impact, non-penetration, and stiction from limited real
data. **PhysHOI** [22] makes a contact graph central to physics-based HOI
imitation. **InterDreamer** [23] separates high-level interaction semantics
from low-level object dynamics and learns how human actions affect object
motion.

These works do not solve hidden response inference by themselves, but they
set a representation floor. A new method that flattens forces, ignores
spatial contact structure, or treats all contact modes as one continuous
state will be judged against stronger contact-specific baselines. In
addition, **Physion++** [24] shows that standard prediction models do not
spontaneously infer latent mass, friction, elasticity, or deformability
when prediction depends on them.

## 5. Cross-Branch Synthesis

The literature does not contain a retrieved work with the exact joint claim
"select a short probe to infer hidden object response, then rank candidate
dexterous hand action chunks under unseen physics using a counterfactual
effect model." This joint gap is real but narrow.

Each individual component is already occupied:

| Component | Established precedent | Consequence |
|---|---|---|
| Action-difference or quotient prediction | Counterfactual Quotient Models [3] | Centering and shared-mode cancellation cannot be the main novelty |
| Action-identifiability diagnosis | ATM [2], controlled-world-model theory [1] | A diagnostic probe alone is not enough |
| Active information gathering | Tactile belief-space control [9], Push to know! [10], EIG action selection [12] | Probe-based parameter estimation is not new |
| History-conditioned context | EDO-Net [14], Context-aware DeLaN [15], ProtoCAD [16] | A recurrent response latent alone is not new |
| Contact-aware representation | ContactWorld [19], ContactNets [21], FORCE [20], PhysHOI [22] | The model must respect contact structure |
| Ranking rather than prediction | Counterfactual Quotient [3], Decision-Metric Alignment [4] | D2 must measure ranking and regret, not only L1 |

The defensible research question is therefore not "can we build a
probe-conditioned world model" but:

> Under unseen object response, can an identifiability-aware short probe
> produce a context that improves counterfactual action ranking beyond
> no-probe, random-probe, and absolute-prediction baselines at the same
> interaction budget?

This is an incremental method question with a disruptive seed. The seed is
the claim that the right probe objective is not parameter accuracy but the
weakest counterfactual action contrast needed by the downstream decision.

## 6. Proposed New Design

Working name: **Probe-Adaptive Counterfactual Effect Model (PACEM)**.

### 6.1 Scope

The first version should use the existing MuJoCo rigid-object simulator,
right hand, 13 fixed branches, and unseen mass/friction split. It should not
claim MaMi improvement, human tactile transfer, deformable dynamics, or
general physical fidelity.

The model should produce a candidate-action ranking or pairwise action
effect. It should not be presented as a general video world model and should
not require explicit mass or friction labels.

### 6.2 Model structure

```text
initial state
    + short probe action sequence
    + observed probe outcome
    -> response context z

initial state after probe
    + response context z
    + candidate action chunks A_i, A_j
    -> pairwise counterfactual effect or ranking
```

The response context can be probabilistic, but the first version should be
small. A recurrent encoder for `z` and a shared score for candidate actions
are sufficient to test the hypothesis. A large transformer or diffusion
model would obscure attribution.

The training objective should combine:

1. a pairwise or listwise ranking loss over simulator outcomes;
2. a small response-prediction or calibration auxiliary loss;
3. optional information-gain regularization only in the probe-selection
   extension.

Absolute object-pose regression can remain a diagnostic, not the primary
promotion metric.

### 6.3 Probe design

The pilot must separate two questions.

**P0, passive probe-conditioned ranking.** Give the model a fixed or random
short probe and test whether the response context improves D2 ranking on
unseen object configurations. This is the first and decisive experiment.

**P1, active probe selection.** Only if P0 passes, compare an
identifiability-aware probe policy against random and fixed probes under the
same number of simulator steps. A useful selector should maximize the
weakest relevant action contrast or expected ranking improvement, not
merely parameter entropy.

If P1 does not beat matched random or fixed probes, remove active-probe
learning from the paper. A passive probe-conditioned model without a strong
novelty argument should remain an internal diagnostic rather than a headline
method.

### 6.4 Minimum promotion experiment

The pilot should be frozen before implementation. A reasonable first protocol
is:

| Stage | Comparison | Primary endpoint | Stop condition |
|---|---|---|---|
| P0a | no probe, fixed probe, random probe, oracle response | Unseen-variant action ranking | No probe gain over zero probe |
| P0b | quotient score, absolute prediction, decision-focused score | Top-1 and regret | No gain over absolute or geometry baseline |
| P1 | active probe, random probe, fixed probe | Regret per interaction step | Active does not beat matched controls |
| P2 | branch shuffle, zero action, wrong response context | Causal contribution | Performance survives controls |
| P3 | add small noise or contact perturbation | Robustness | Gain disappears under trivial perturbation |

Promotion should use the original D2 directions: top-1 improvement over a
geometry-only baseline, lower regret, and a pairwise bootstrap interval
above zero. The exact thresholds should be re-frozen before the run. D2
should not be replaced by an easier absolute-pose L1 threshold.

### 6.5 What would count as a contribution

A publishable D contribution would need at least three ingredients:

1. a controlled result showing that a single initial state is insufficient
   for unseen response ranking;
2. a probe-conditioned or decision-focused mechanism that improves ranking
   at a matched interaction budget;
3. evidence that the gain is attributable to the response context and
   counterfactual objective rather than to extra parameters, extra data, or
   a branch-specific shortcut.

If the first condition is demonstrated but the second fails, the work is a
useful benchmark or diagnostic paper, not a method paper. If both fail, stop
Direction D rather than relabeling the model.

## 7. Idea Evaluation

### 1. First impression

Paper type: **Novel Method**, but an incremental one with a disruptive seed.

One-sentence story: a short probe can identify the hidden response regime of
an object, allowing a decision-focused model to rank hand action chunks
better than models that predict absolute futures from one initial state.

### 2. Fatal-flaws audit

| # | Flaw | Severity | Defense |
|---|---|---|---|
| 1 | F1: generic active probing and counterfactual quotient learning already exist | MAJOR | Position on the joint objective: probe for downstream counterfactual ranking under unseen response; include CQM, ATM, active tactile perception, Phys2Real, and ProtoCAD as baselines or explicit positioning |
| 2 | F8: probe selection, response inference, quotient learning, contact modeling, and MaMi integration are too many contributions | MAJOR | Cut MaMi integration and explicit physical-parameter accuracy; first run P0 passive probe ranking only |
| 3 | F6: a persuasive claim requires a ranking experiment that is currently outside D1 | MINOR | Freeze D2 as the only promotion endpoint and do not promote on L1 regression |
| 4 | Data-refuted original D1 mechanism | CRITICAL for the old version | Stop D1; do not present the old initial-state-to-action regression as repairable |

The CRITICAL flaw applies to the frozen D1 version only. The revised PACEM
proposal is untested, so it does not inherit that data-refuted mechanism.

### 3. Lifecycle and capability match

| Aspect | User's input | Assessment |
|---|---|---|
| Idea category | Innovative technique plus data-intensive simulation | Medium-to-long development, not a short benchmark-only task |
| Lifecycle | 6-9 months for a defensible method paper | Yellow |
| Compute | RTX 4090 D, 16 CPU, 62 GB RAM server available | Green for rigid-object MuJoCo and small models |
| Engineering | Prior D0 MuJoCo, counterfactual dataset, training scripts already exist | Green |
| Main risk | Adding active probing before proving passive response adaptation | Mitigate by P0 gate |

### 4. Five-dimension radar

| Dimension | Score | Evidence | Lift suggestion |
|---|---:|---|---|
| Higher | 7 | Mechanism-based: response context addresses the measured unseen-response failure; no new D2 result yet | Make P0b a direct ranking comparison |
| Faster | 4 | Active probing adds interaction cost; no evidence of speedup | Keep the active policy out until passive ranking works |
| Stronger | 8 | Mechanism-based: explicitly targets unseen mass/friction and matched probe controls | Add contact perturbation and response-context mismatch controls |
| Cheaper | 7 | Mechanism-based: can use a short interaction history and no property labels | Compare probe lengths and include oracle-physics upper bound |
| Broader | 6 | Could transfer to robot manipulation, but current scope is one simulator and one hand | Do not claim cross-domain before one cross-dynamics replication |

### 5. Paradigm-shift probe

| Probe | Yes / Partial / No | Rationale |
|---|---|---|
| First Principles | Partial | Challenges single-observation sufficiency, but identifiability and probing are established principles |
| Elephant in the Room | Partial | Hidden response and ranking mismatch are recognized, but existing benches and active perception already attack parts |
| Technology Cycle | No | The core idea could have been attempted with earlier simulation and system-identification tools |
| Hamming's Rule | Partial | Strong decision-utility results would matter to world-model planning, but the current scope is narrow |

Disruptive potential: **possible**, not strong.

### 6. Feasibility

| Risk | Level | Mitigation |
|---|---|---|
| Compute | Low | Existing rigid-object data and 4090 are sufficient |
| Data | Medium | Probe counterfactual data must be regenerated with frozen budgets and split separation |
| Engineering | Medium | Reuse D0 simulator; keep the first model small |
| Timeline | Medium | One P0 pilot is feasible; full active-probe plus MaMi integration is not justified yet |

### 7. Verdict

**Accept with Revisions** for the revised decision-focused probe design,
worth pursuing pending the P0 ranking validation experiment.

**Reject and Pivot** for the frozen D1 initial-state-to-action regression.

Top three actions:

1. Close D1 permanently and freeze the exact P0 protocol before coding.
2. Build the passive probe-conditioned ranking pilot with matched
   no-probe, random-probe, quotient, absolute-prediction, and oracle-physics
   controls.
3. Only after P0 passes, test whether active probe selection beats random or
   fixed probes at the same interaction budget; otherwise stop the active
   branch.

## 8. Open Problems

The strongest unresolved issue is whether the hidden response parameter is
actually identifiable from the available hand-object state and short probe.
The audit shows that branch effects exist, but not that a fixed probe
separates mass from friction for every shape and reset. P0 should report
identifiability and ranking utility separately.

A second issue is whether quotient or ranking objectives exceed an absolute
model after controlling for model capacity and data. Counterfactual
Quotient Models [3] and Decision-Metric Alignment [4] make this a required
ablation, not an optional one.

A third issue is transfer. A positive result on three rigid shapes with
four mass/friction variants would not establish deformable, articulated, or
real tactile generalization. Those claims require separate data and
separate experiments.

## 9. Conclusion

There is no evidence that the broad question behind Direction D is useless.
There is clear evidence that the current D1 formulation is the wrong way to
study it. The most valuable next step is not a larger world model and not an
immediate active-probe system. It is a small, decision-focused pilot that
asks whether a short interaction history makes unseen object response
rankable. If that pilot fails, Direction D should be stopped despite the
existence of attractive terminology.

## References

1. Xiangteng Zhang et al. *On the Identifiability of Controlled World Models.* arXiv:2607.22430v2. [https://arxiv.org/abs/2607.22430](https://arxiv.org/abs/2607.22430)
2. Jiaheng Chen. *ATM: Action-Consistency Transfer Matrix for Diagnosing and Improving Latent World Models.* arXiv:2606.09028v1. [https://arxiv.org/abs/2606.09028](https://arxiv.org/abs/2606.09028)
3. Junlin Chen, Ruijie Wang, Jianxin Li. *Counterfactual Quotient Models: Learning What Actions Change, Not What the World Does.* arXiv:2608.22092v1. [https://arxiv.org/abs/2608.22092](https://arxiv.org/abs/2608.22092)
4. Jiawei Wang et al. *Decision-Metric Alignment in Latent World Models: Diagnostics and Action-Conditioned Objectives for MPC Planning.* arXiv:2608.18746v1. [https://arxiv.org/abs/2608.18746](https://arxiv.org/abs/2608.18746)
5. Kunlin Cai et al. *What-If World: A Causal Benchmark for General World Models in Embodied Scenarios.* arXiv:2605.27589v1. [https://arxiv.org/abs/2605.27589](https://arxiv.org/abs/2605.27589)
6. Haotian Xue et al. *ACWM-Phys: Investigating Generalized Physical Interaction in Action-Conditioned Video World Models.* arXiv:2605.08567v2. [https://arxiv.org/abs/2605.08567](https://arxiv.org/abs/2605.08567)
7. Horia Mania, Michael I. Jordan, Benjamin Recht. *Active Learning for Nonlinear System Identification with Guarantees.* arXiv:2006.10277v1. [https://arxiv.org/abs/2006.10277](https://arxiv.org/abs/2006.10277)
8. Nicolas Chatzikiriakos, Kevin Jamieson, Andrea Iannelli. *High Effort, Low Gain: Fundamental Limits of Active Learning for Linear Dynamical Systems.* arXiv:2509.11907v2. [https://arxiv.org/abs/2509.11907](https://arxiv.org/abs/2509.11907)
9. Jean-Francois Tremblay et al. *Learning active tactile perception through belief-space control.* arXiv:2312.00215v1. [https://arxiv.org/abs/2312.00215](https://arxiv.org/abs/2312.00215)
10. Anirvan Dutta, Etienne Burdet, Mohsen Kaboli. *Push to know! -- Visuo-Tactile based Active Object Parameter Inference with Dual Differentiable Filtering.* arXiv:2308.01001v1. [https://arxiv.org/abs/2308.01001](https://arxiv.org/abs/2308.01001)
11. Anirvan Dutta, Etienne Burdet, Mohsen Kaboli. *Predictive Visuo-Tactile Interactive Perception Framework for Object Properties Inference.* arXiv:2411.09020v1. [https://arxiv.org/abs/2411.09020](https://arxiv.org/abs/2411.09020)
12. Andrej Kruzliak et al. *Interactive Learning of Physical Object Properties Through Robot Manipulation and Database of Object Measurements.* arXiv:2404.07344v2. [https://arxiv.org/abs/2404.07344](https://arxiv.org/abs/2404.07344)
13. Maggie Wang et al. *Phys2Real: Fusing VLM Priors with Interactive Online Adaptation for Uncertainty-Aware Sim-to-Real Manipulation.* arXiv:2510.11689v2. [https://arxiv.org/abs/2510.11689](https://arxiv.org/abs/2510.11689)
14. Alberta Longhini et al. *EDO-Net: Learning Elastic Properties of Deformable Objects from Graph Dynamics.* arXiv:2209.08996v4. [https://arxiv.org/abs/2209.08996](https://arxiv.org/abs/2209.08996)
15. Lucas Schulze, Jan Peters, Oleg Arenz. *Context-Aware Deep Lagrangian Networks for Model Predictive Control.* arXiv:2506.15249v3. [https://arxiv.org/abs/2506.15249](https://arxiv.org/abs/2506.15249)
16. Junjie Wang et al. *Prototypical context-aware dynamics generalization for high-dimensional model-based reinforcement learning.* arXiv:2211.12774v1. [https://arxiv.org/abs/2211.12774](https://arxiv.org/abs/2211.12774)
17. Wen Sin Lor, Jun Wang, Suyi Li. *Shake to Learn: Dynamic Interrogation of Hidden Object Physics for Robotic Manipulation with Physical Reservoir Computing.* arXiv:2609.20970v1. [https://arxiv.org/abs/2609.20970](https://arxiv.org/abs/2609.20970)
18. Xuan Li et al. *PAC-NeRF: Physics Augmented Continuum Neural Radiance Fields for Geometry-Agnostic System Identification.* arXiv:2303.05512v1. [https://arxiv.org/abs/2303.05512](https://arxiv.org/abs/2303.05512)
19. Zhiyuan Zhang et al. *ContactWorld: What Representations Matter in Vision-Tactile World Models for Contact-Rich Manipulation.* arXiv:2606.13877v2. [https://arxiv.org/abs/2606.13877](https://arxiv.org/abs/2606.13877)
20. Xiaohan Zhang et al. *FORCE: Physics-aware Human-object Interaction.* arXiv:2403.11237v2. [https://arxiv.org/abs/2403.11237](https://arxiv.org/abs/2403.11237)
21. Samuel Pfrommer, Mathew Halm, Michael Posa. *ContactNets: Learning Discontinuous Contact Dynamics with Smooth, Implicit Representations.* arXiv:2009.11193v2. [https://arxiv.org/abs/2009.11193](https://arxiv.org/abs/2009.11193)
22. Yinhuai Wang et al. *PhysHOI: Physics-Based Imitation of Dynamic Human-Object Interaction.* arXiv:2312.04393v1. [https://arxiv.org/abs/2312.04393](https://arxiv.org/abs/2312.04393)
23. Sirui Xu et al. *InterDreamer: Zero-Shot Text to 3D Dynamic Human-Object Interaction.* arXiv:2403.19652v2. [https://arxiv.org/abs/2403.19652](https://arxiv.org/abs/2403.19652)
24. Hsiao-Yu Tung et al. *Physion++: Evaluating Physical Scene Understanding that Requires Online Inference of Different Physical Properties.* arXiv:2306.15668v2. [https://arxiv.org/abs/2306.15668](https://arxiv.org/abs/2306.15668)
25. Kaiyu Hang et al. *Manipulation for self-Identification, and self-Identification for better manipulation.* Science Robotics, 2021. [https://doi.org/10.1126/scirobotics.abe1321](https://doi.org/10.1126/scirobotics.abe1321)
