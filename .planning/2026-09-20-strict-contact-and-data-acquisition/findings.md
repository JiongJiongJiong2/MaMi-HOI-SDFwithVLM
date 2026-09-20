# Findings and Decisions

## Requirements

- Keep EPIC-Contact when strict contact produces enough lifecycle events.
- Do not post-hoc relabel the official `3 mm` result as successful.
- Acquire external data only for fields EPIC lacks: release and handover.

## Research Findings

### EPIC-Contact

Official `3 mm`:

```text
train onset transitions:      37
train release transitions:    33
train complete episodes:      17
test complete episodes:        0
overlapping train/test videos: 14
```

Strict `1 mm` sensitivity:

```text
train onset transitions:      2,100
train release transitions:    2,132
train complete episodes:      1,160
test complete episodes:         100
```

This supports a new strict-contact pilot but requires a new protocol and
video-disjoint split.

### Candidate External Sources

| Source | Official access | Main value | Limit |
|---|---|---|---|
| GRAB | https://grab.is.tue.mpg.de/ | MANO contact maps and long grasp/motion sequences | May be single-hand; license/registration |
| ARCTIC | https://arctic.is.tue.mpg.de/ | Bimanual articulated-object manipulation and dynamic contact | Needs MANO/SMPL-X topology gate; not explicit handover |
| OakInk | https://oakink.net/ | Large hand-object interaction repository | Must verify contact/release labels and license |
| H2O | https://taeinkwon.com/projects/h2o/ | Two-hand object interactions in egocentric recordings | Must verify release annotations and availability |
| DexYCB | https://dex-ycb.github.io/ | Large single-hand grasp/object sequence data | Not bimanual; dense release labels uncertain |

No public dataset should be described as an explicit handover benchmark
until receiver identity and role change are verified in the files.

## Technical Decisions

| Decision | Rationale |
|---|---|
| Use minimum-distance threshold sensitivity | Official threshold changes event availability by two orders of magnitude |
| Require video-disjoint split | Official split shares videos |
| Keep handover separate from bimanual contact | Simultaneous support is not role transfer |
