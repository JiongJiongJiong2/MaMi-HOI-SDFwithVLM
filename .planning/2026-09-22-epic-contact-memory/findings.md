# Findings

- The raw EPIC sample contains `idx.ro`, `idx.lo`, `idx.or`, `idx.ol`,
  `object.v.cam`, `object.f`, and MANO vertices/poses.
- `object.v.cam` uses a stable per-object vertex indexing within a clip; the
  object topology and vertex count were verified equal between first and last
  sampled frames.
- The strict EPIC split is participant-disjoint with train/dev/test counts
  25/3/3 participants and 28,287/13,159/16,240 frames.
- Supported object-held-out classes with at least 20 complete train episodes:
  bowl, plate, pan, bottle.
- The existing test has already been viewed by the EPIC event baseline and
  must be disclosed as reused.
