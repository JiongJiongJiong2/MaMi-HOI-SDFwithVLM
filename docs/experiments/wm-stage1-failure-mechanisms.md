# Why the WM Stage 1 Learned Residual Failed: A Mechanism-Level Review

Date: 2026-09-16

## Abstract

The Stage 1 learned-residual arm failed a hard gate: H8 contact F1 decreased
slightly, while H8 palm L1 increased from `0.00922` to `0.08876`. The most
important observation is that the error ratio is already about `9.0x` at
H1, before recurrent rollout can compound anything. This makes one-step
residual bias or parameterization the primary cause, not exposure bias
alone. Teacher forcing and free-running rollout remain important
amplifiers: for seed 21, teacher-forced H8 palm L1 is `0.01562`, while
free-running H8 palm L1 is `0.09196`. Contact discontinuity supplies a third
failure mode because small clearance changes can flip contact labels,
penetration signs, and normals. Objective mismatch supplies the fourth:
the model is trained and checkpoint-selected using smooth state and event
losses, but evaluated through free-running contact F1, penetration, and
candidate ranking. The literature does not support fixing this by increasing
network size or tuning the residual scale alone. The residual must be
placed at the missing contact-physics layer, trained on the learner's
rollout distribution, evaluated against decision regret, and constrained
with a rollout-level stability or trust-region mechanism.

## 1. Introduction

This review asks three questions.

RQ1: Why did the learned residual make H8 palm geometry about `9.6x` worse
than the analytic integrator while contact F1 did not improve?

RQ2: Which literature-supported mechanisms explain this failure, and which
mechanism has the strongest support from the project's own numbers?

RQ3: What do those mechanisms imply for the next model, rather than merely
suggesting more training?

The review uses four independent evidence branches: residual learning and
stability, train-inference distribution mismatch, contact discontinuity,
and mismatch between prediction objectives and planning objectives.

## 2. Project Evidence

The formal run used three seeds and 30 epochs per seed. The learned model
kept the explicit action integrator and added a bounded residual on palm
position, palm velocity, clearance, and normal. The residual was computed
with `tanh` and then multiplied by `residual_scale=0.05`.

| Metric | Analytic rollout | Learned rollout |
|---|---:|---:|
| H1 palm L1 | 0.00133 | 0.01205 |
| H8 palm L1 | 0.00922 | 0.08876 |
| H1 contact F1 | 0.92922 | 0.92922 |
| H8 contact F1 | 0.71946 | 0.71488 |

The bounded residual had mean absolute value `0.1105-0.1186` and maximum
absolute value `0.9990-0.9999` across seeds. After multiplying by the
residual scale, the mean applied residual was roughly `0.0055-0.0059` per
dimension per step, while the maximum applied residual was `0.05`. The
reported residual statistics therefore describe the pre-scale bounded
residual; raw, bounded, and applied residual should be logged separately in
the next diagnostic.

For seed 21, teacher-forced H8 palm L1 was `0.01562`, while free-running H8
palm L1 was `0.09196`. The free-running path was about `5.9x` worse than the
teacher-forced path for the same checkpoint.

The downstream scorer reinforced the negative result. Relative to the
analytic geometry arm, the learned-state arm reduced dense contact F1 by
`10.17` percentage points with a paired 95 percent interval of
`[-18.09, -2.44]` points. The hybrid event arm was nearly identical at
`-10.11` points with an interval of `[-18.25, -2.47]`.

The scorer cohort contained 10 sequences and 19 events per seed. It is
therefore a valid kill test for the current branch, but not a precise final
estimate of the analytic geometry baseline.

## 3. Taxonomy of Failure Mechanisms

The four explanations are not mutually exclusive. They operate at different
levels and should be ranked rather than replaced by one umbrella story.

| Mechanism | Level | Expected signature | Evidence in this run |
|---|---|---|---|
| One-step residual bias | model parameterization | error already large before rollout | H1 is about `9.0x` worse than analytic |
| Train-inference distribution mismatch | rollout training | free-running much worse than teacher-forced | H8 free is `5.9x` worse than teacher-forced |
| Contact discontinuity and smoothing bias | target physics | event metrics do not track continuous error | contact F1 stays flat or worsens |
| Objective and evaluation mismatch | training and selection | lower loss can select a worse controller | checkpoint and scorer optimize different quantities |

## 4. One-Step Residual Bias

The analytic integrator is already a strong one-step model for palm motion:
its H1 palm L1 is `0.00133`, and it applies the candidate action directly to
the palm state. A useful residual therefore has to learn an almost-zero
correction on most examples. The learned branch instead applies an average
correction of roughly `0.0055` per normalized dimension per step, which is
already larger than the analytic H1 error.

Residual RL and residual policy learning work when the base controller is
good but has a systematic residual and the residual is trained in the
closed-loop task distribution [1][2]. In contrast, the current residual is
trained offline against window transitions and has freedom to add unrelated
state offsets. NeuralSim and stochastic simulator augmentation place the
learned component inside the physics equation, where it represents missing
friction or contact effects, rather than adding an unrestricted state
increment [3][4]. This distinction explains why the learned component can
lower an aggregate training loss while making the known palm trajectory
worse.

The failure is not evidence that model capacity is insufficient. The loss
decreased, and the residual stayed finite. The residual simply had the
wrong authority. A raw or bounded scalar bound is not a physical
identifiability guarantee.

## 5. Train-Inference Distribution Mismatch

The training loop used a fixed `teacher_forcing_ratio=0.5`, while evaluation
used free-running rollout. If each future state is chosen independently,
the probability that all eight rollout inputs come from predictions is
`0.5^8`, about `0.39` percent. The model is therefore trained mostly on
states near ground truth and evaluated on states generated by itself.

Scheduled Sampling describes exactly this mismatch between training with
ground-truth inputs and inference with generated inputs [5]. Professor
Forcing regularizes the hidden-state dynamics to be similar in teacher and
sampling modes [6]. DAgger goes further, collecting the learner's induced
states and requesting expert labels on those states [7]. MBPO responds to
the same problem by using short rollouts branched from real states instead
of trusting long imagined trajectories [8].

The project's teacher-forced versus free-running gap matches this
literature. However, the H1 result rules out exposure bias as the primary
cause: there is no previous predicted palm input at H1, yet the learned
model is already about `9x` worse than the analytic model. Exposure bias is
therefore an amplifier, not the root cause.

## 6. Contact Discontinuity and Smoothing Bias

Contact is not a single smooth trajectory family. Free motion, impact,
sticking, sliding, and release can induce velocity jumps, impulse, and
friction-cone constraints. A smooth state residual tends to learn a
conditional average across contact modes, while a smooth L1 loss treats the
rare large collision residual as an outlier.

ContactNets learns signed distance and contact-frame Jacobians but adds
complementarity and maximum-dissipation constraints to model impact,
stiction, and non-penetration [9]. Simultaneous learning of contact and
continuous dynamics treats unmeasured contact force and mode as variables
to infer rather than as an unconstrained state correction [10].
Contact-implicit MPC keeps contact timing, force, and mode variables
explicit in the optimization [11][12].

This evidence helps explain why Contact BCE can move without improving the
SDF scorer. The event head predicts whether contact occurs, but the scoring
pipeline evaluates dense mesh contact and penetration. A continuous palm
offset can improve a coarse event probability while moving many hand
vertices across the contact threshold in the wrong direction. The event
head and the geometry scorer are not optimizing the same physical object.

## 7. Objective and Evaluation Mismatch

The transition model is trained with smooth state regression and onset or
release BCE. The downstream objective is a candidate ranking based on
predicted dense contact fraction and penetration. Objective mismatch work
shows that lower one-step likelihood does not necessarily improve
downstream control or planning [13][14]. Value equivalence goes further:
two models need not predict every state equally well if they produce
equivalent Bellman updates or action values [15].

This is directly visible in the implementation. Checkpoint selection uses
object translation L1 plus contact BCE, not free-running palm error,
penetration, or candidate regret. A checkpoint can therefore look better
under the training objective while being worse under the scorer objective.
The decision-focused learning and SPO literature argues for optimizing a
regret or decision loss when the final consumer is an optimization or
selection step [16][17]. PETS and MOPO show the complementary need to model
or penalize uncertainty before using a learned rollout for planning
[18][19].

The hybrid event scorer attempted to add event information, but it could
not compensate for a learned geometry rollout that was already
substantially worse than the analytic one. The event term should be
calibrated against candidate ranking utility, not only classification AUC.

## 8. Cross-Branch Synthesis

The evidence supports the following attribution order.

First, the residual parameterization introduced a biased one-step model.
This is the strongest conclusion because H1 error is already about nine
times the analytic baseline and because the residual offset is larger than
the analytic error it was supposed to correct.

Second, teacher forcing and free-running rollout amplified the bias. The
free-running H8 error is about six times the teacher-forced H8 error for
the same checkpoint. Scheduled Sampling, Professor Forcing, DAgger, and
MBPO support treating this as a real training-distribution problem.

Third, contact discontinuity made the consequence non-monotonic.
ContactNets, simultaneous contact-continuous dynamics, and contact-implicit
MPC all indicate that contact requires mode, force, impulse, or
complementarity-aware variables. Smooth state regression does not
automatically preserve event timing or non-penetration.

Fourth, objective mismatch explains why the training pipeline did not
reject the bad rollout. The checkpoint and training loss did not directly
measure candidate ranking regret, and the event head was only loosely
coupled to the final geometry score.

Network size is not supported as the primary explanation. It remains
possible that the state representation is insufficient, but the current
experiment does not isolate capacity from parameterization, distribution
shift, and objective mismatch.

## 9. Diagnostic Tests Required Before a New Residual Model

The next experiment should first run diagnostics on the existing checkpoint
without training a new model.

1. Log raw, bounded, and applied residual separately by state dimension
   and horizon.
2. Compare teacher-forced and free-running error for the same checkpoint at
   H=1, 2, 4, and 8.
3. Measure action sensitivity for inward, outward, hold, and random chunks
   under both analytic and learned transition.
4. Measure candidate ranking regret separately from state L1 and event AUC.
5. Measure calibration error for clearance, contact probability, and the
   final selection score.

Only after these diagnostics should the model be changed. A revised model
should use scheduled sampling or DAgger-style state collection, short
branches from real states, a structured contact-frame or impulse residual,
and a checkpoint objective based on candidate ranking utility rather than
one-step state loss.

## 10. Conclusion

RQ1: The learned rollout worsened palm geometry because it introduced a
biased one-step residual that was already larger than the analytic error,
and free-running evaluation amplified that bias across the rollout.

RQ2: The literature supports four mechanisms. One-step residual bias is the
primary cause; train-inference mismatch and free-running compounding are
amplifiers; contact discontinuity makes the geometry and event metrics
non-monotonic; objective mismatch allows the training and checkpoint
selection process to prefer the wrong model.

RQ3: Increasing model size or tuning the residual scale is not the right
next step. The residual must be moved to the missing contact-physics
layer, trained on learner-induced states, constrained by a rollout-level
stability or trust-region criterion, and selected using the same decision
objective used for correction.

## References

[1] T. Johannink, S. Bahl, A. Nair, et al., "Residual Reinforcement Learning for Robot Control," ICRA, 2019.

[2] T. Silver, K. Allen, J. Tenenbaum, and L. Kaelbling, "Residual Policy Learning," arXiv:1812.06298, 2018.

[3] E. Heiden, D. Millard, E. Coumans, Y. Sheng, and G. S. Sukhatme, "NeuralSim: Augmenting Differentiable Simulators with Neural Networks," ICRA, 2021.

[4] A. Ajay, J. Wu, N. Fazeli, et al., "Augmenting Physical Simulators with Stochastic Neural Networks: Case Study of Planar Pushing and Bouncing," IROS, 2018.

[5] S. Bengio, O. Vinyals, N. Jaitly, and N. Shazeer, "Scheduled Sampling for Sequence Prediction with Recurrent Neural Networks," NeurIPS, 2015.

[6] A. Lamb, A. Goyal, Y. Zhang, et al., "Professor Forcing: A New Algorithm for Training Recurrent Networks," NeurIPS, 2016.

[7] S. Ross, G. J. Gordon, and J. A. Bagnell, "A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning," AISTATS, 2011.

[8] M. Janner, J. Fu, M. Zhang, and S. Levine, "When to Trust Your Model: Model-Based Policy Optimization," NeurIPS, 2019.

[9] S. Pfrommer, M. Halm, and M. Posa, "ContactNets: Learning Discontinuous Contact Dynamics with Smooth, Implicit Representations," CoRL, 2020.

[10] B. Bianchini, M. Halm, and M. Posa, "Simultaneous Learning of Contact and Continuous Dynamics," CoRL, 2023.

[11] S. Le Cleac'h, H. Suh, et al., "Fast Contact-Implicit Model Predictive Control," IEEE Transactions on Robotics, 2024.

[12] W.-C. Huang, A. Aydinoglu, W. Jin, and M. Posa, "Adaptive Contact-Implicit Model Predictive Control with Online Residual Learning," ICRA, 2024.

[13] N. Lambert, B. Amos, O. Yadan, and R. Calandra, "Objective Mismatch in Model-based Reinforcement Learning," L4DC, 2020.

[14] A.-M. Farahmand, A. Barreto, and D. Nikovski, "Value-Aware Loss Function for Model-based Reinforcement Learning," AISTATS, 2017.

[15] C. Grimm, A. Barreto, S. Singh, and D. Silver, "The Value Equivalence Principle for Model-Based Reinforcement Learning," NeurIPS, 2020.

[16] B. Wilder, B. Dilkina, and M. Tambe, "Melding the Data-Decisions Pipeline: Decision-Focused Learning for Combinatorial Optimization," AAAI, 2019.

[17] A. N. Elmachtoub and P. Grigas, "Smart Predict, then Optimize," Management Science, 2022.

[18] K. Chua, R. Calandra, R. McAllister, and S. Levine, "Deep Reinforcement Learning in a Handful of Trials using Probabilistic Dynamics Models," NeurIPS, 2018.

[19] T. Yu, G. Thomas, L. Yu, et al., "MOPO: Model-based Offline Policy Optimization," NeurIPS, 2020.
