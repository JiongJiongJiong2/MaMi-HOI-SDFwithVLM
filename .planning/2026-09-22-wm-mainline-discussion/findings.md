# Evidence

Read task finals: 评估ASTRA-920实验进展 (01a0c3c1-23b8-78f1-81f8-3fec91a189c7); ASTRA-920-评估实验结果与研究准备-后续方向 (01a0ba2d-20b5-7942-8be1-f99cbc53897e).

Primary reports read: wm-stage1-learned-residual.md; forward-inverse-consistency-2026-09-18.md; astra-920-progress-audit-2026-09-21.md.

Learned residual: H8 palm L1 0.00922 analytic vs 0.08876 learned; contact F1 0.7195 vs 0.7149. This implemented branch is negative, not evidence against all world models. Error accumulation is a report interpretation, not experimentally isolated cause.

E2: H8 forward contact F1 true/shuffled action 0.7099/0.7102; object translation 0.012963/0.012966. History shuffle matters. No demonstrated useful action dependence; need causal/representation diagnosis before causal claims.

E5: 343/376 vs five-point smoothing 312/376; train/dev overlapping windows, limited hand scope. Latest event detection is conditional offline localization, not policy improvement. OakInk data availability does not yield same-state multiple-action response labels. Existing test was exposed during postprocessing design.

Original motivation recovered from idea-revisit-20260919/history-user-excerpts.json: advisor requested avoiding floating/interpenetration and improving naturalness; user proposed local surface querying and paired contact. Early section-based hypotheses were at times evaluated using contact-state classification instead of hand-placement ranking. Later matched-candidate HPS/DMS negatives are still meaningful for their tested representations.

Code inspection: features.py explicitly uses 34D state / 15D action and no articulated finger poses. model.py analytically applies object and palm increments before learned residuals. E2 is a separate protocol excluding future object response from inputs; do not accuse E2 of the earlier action/target redundancy.

H1 residual palm error is already 0.01205 vs analytic 0.00133; long-rollout error accumulation cannot be the sole explanation. Input sufficiency, training distribution, scale/objective mismatch remain hypotheses, not proven causes.

Primary literature verified on 2026-09-22: InterDreamer (Xu et al. 2024, 2403.19652v1) learns object response from HOI mocap and local body contact vertices; no blanket requirement for robot-control or tactile datasets. HOI-Dyn (Wu et al. 2025, 2507.01737v3) trains dynamics 150 epochs and generator 100,000 steps; local 5-epoch/300-step migration screening is not a full reproduction. LYRIC (Han et al. 2026, 2609.19688v1) already relaxes finger/wrist and supporting-arm reference tracking near contact. CHOIR (Xu et al. 2026, 2605.20992v1) already dynamically updates contact correspondences with temporal memory. WorldContact (Wang et al. 2026, 2609.19600v1) predicts deformable-object states and expands policy data. DexTouch-WM (Qin et al. 2026, 2609.20649v1) aligns human/robot action spaces and tactile layouts, evaluates policies and generates data. TD-MPC2 official project verifies latent planning without image reconstruction.

Recommendation: keep advisor's visible contact-quality problem as the mainline; test short-horizon anticipatory correction through onset/hold/release on an executable hand-arm chain. Start with a same-budget downstream diagnostic, candidate oracle and audited phase oracle where appropriate, not more detection-only models. Choose learning target only after identifying a missing predictive quantity or a measurable speed/quality tradeoff. If future geometry is fully specified, use exact computation; a learned surrogate needs an efficiency advantage. Physical response claims need independently verified response evaluation. Same-state action branches are a useful diagnostic, not a universal training-data requirement.
