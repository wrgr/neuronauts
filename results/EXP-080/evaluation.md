# EXP-080 — Does SegCLR select the true continuation? No.

## Result: as a selector among geometry's candidates, SegCLR scores top-1 on 0 of 44 panels where geometry scores 22. Below chance, with 94% embedding coverage, so this is not a coverage artifact.

The unmerged July branch (`claude/segclr-fuser-grammar-x8ba3x`, 50 commits never
on this branch) reported: *"endpoint proximity generates candidates and already
disambiguates most stitches; SegCLR selects among the candidates. On the hard
contested cases SegCLR is 12/12 while colinearity-geometry is 9/12, and overall
150/150."* That is precisely our failure mode — geometry ranks well but cannot
break ties — so it was worth testing before anything else.

Our own cache already held a contradicting measurement
(`data/external/segclr/auc_result.json`): AUC 0.445 on 34 v117 atoms, with
different-owner pairs *more* similar (0.842) than same-owner pairs (0.823).
Both samples were small. This settles it on 66 soma-seeded panels.

## Setup

89,127 embedded points inside the 100 µm cube, from 2,795 m343 segments,
extracted by `scripts/build_segclr_cube_index.py`. SegCLR is keyed to the m343
segmentation and no identifier maps onto v117, so assignment is spatial — which
is what the July code did as well.

Geometry (`along × collin × proximity × caliber`) proposes the top 20
candidates; the embedding then re-ranks them. 44 panels have their true partner
inside geometry's top 20.

## Result

| | median rank | top-1 | top-3 |
|---|---:|---:|---:|
| geometry alone | 0.5 | **22/44** | 30/44 |
| SegCLR selects | 11.0 | **0/44** | 3/44 |
| geometry × SegCLR | 0.5 | 22/44 | 30/44 |

Chance for a selector over 20 candidates is a median rank near 9.5 and about
2 of 44 at top-1. SegCLR reaches 11.0 and 0. Coverage is not the explanation:
**44 of 44 true partners** carry an embedding, as do 1,205 of 1,276 distractors.

Multiplying geometry by the embedding changes nothing, because the embedding
term is essentially constant across candidates — which is the same thing the
0.445 measurement says.

## Why this does not contradict July, and where it might

July's own headline was **"SegCLR = type not identity"**. An embedding that
encodes cell type cannot separate one axon from another axon beside it, and a
soma-seeded panel is dense with same-type neighbours. On that reading both
results agree, and the July stitching success came from geometry with the
embedding adding little.

One difference I did not control: July laid embeddings **along a skeleton** and
compared rolling-averaged local traces either side of a gap. This test used a
whole-object mean embedding. A local trace could carry signal a mean washes out,
so this is a negative for object-level embedding similarity, not for every use
of SegCLR. That variant is the one worth trying if anyone revisits this.

## Consequence

Embeddings are not the tie-breaker. The tie is still the open problem: many
candidates touch the seed at a single voxel and distance cannot order them.
