// TCG-Net — results summary
// Compile:  typst compile TCG-Net_results_summary.typ
// Self-contained: uses only Typst's bundled fonts, no @preview packages.

#let accent   = rgb("#1b3a6b")
#let accent2  = rgb("#2f5c96")
#let rulec    = rgb("#c3d2e8")
#let band     = rgb("#eef3fa")
#let muted    = rgb("#5b6470")
#let callout  = rgb("#f7f4e9")
#let calrule  = rgb("#c9b98a")

#set document(
  title: "TCG-Net — Results Summary",
  author: "broccolize",
)

#set page(
  paper: "a4",
  margin: (x: 21mm, top: 20mm, bottom: 18mm),
  footer: context [
    #set text(size: 7.8pt, fill: muted)
    #line(length: 100%, stroke: 0.4pt + rulec)
    #v(-4pt)
    #grid(
      columns: (1fr, auto),
      align: (left, right),
      [TCG-Net — Results Summary · 1 August 2026],
      [#counter(page).display("1 / 1", both: true)],
    )
  ],
)

#set text(
  font: ("Libertinus Serif", "New Computer Modern", "Georgia", "Times New Roman"),
  size: 9.8pt,
  lang: "en",
  hyphenate: true,
)
#set par(justify: true, leading: 0.62em, spacing: 0.95em, first-line-indent: 0pt)

#let sans = ("Segoe UI", "Calibri", "Arial")

#set heading(numbering: none)
#show heading.where(level: 1): it => block(above: 15pt, below: 7pt)[
  #set text(font: sans, size: 12.5pt, weight: "bold", fill: accent)
  #it.body
  #v(-5pt)
  #line(length: 100%, stroke: 0.6pt + rulec)
]
#show heading.where(level: 2): it => block(above: 11pt, below: 5.5pt)[
  #set text(font: sans, size: 10pt, weight: "semibold", fill: accent2)
  #it.body
]

#show strong: set text(fill: rgb("#12233d"))

// ── table helper: booktabs-style rules, banded header ─────────────────────
#let tbl(cols, aligns, header, ..body) = block(above: 6pt, below: 9pt)[
  #set text(size: 8.8pt)
  #table(
    columns: cols,
    align: aligns,
    stroke: none,
    inset: (x: 6pt, y: 4.6pt),
    fill: (x, y) => if y == 0 { band },
    table.hline(stroke: 0.9pt + accent),
    table.header(..header.map(h => text(font: sans, weight: "semibold", fill: accent, size: 8.4pt, h))),
    table.hline(stroke: 0.5pt + rulec),
    ..body.pos(),
    table.hline(stroke: 0.9pt + accent),
  )
]

#let note(body) = block(
  above: 6pt, below: 8pt, inset: (left: 8pt),
  stroke: (left: 1.5pt + rulec),
)[#set text(size: 8.6pt, fill: muted); #body]

#let keybox(body) = block(
  above: 8pt, below: 10pt, width: 100%,
  inset: (x: 10pt, y: 8pt),
  fill: callout,
  stroke: (left: 2pt + calrule),
  radius: (right: 2pt),
)[#set text(size: 9.2pt); #body]

// ── title block ───────────────────────────────────────────────────────────
#block[
  #set text(font: sans, fill: accent)
  #text(size: 19pt, weight: "bold")[TCG-Net — Results Summary]
  #v(3pt)
  #set text(size: 9pt, fill: muted, weight: "regular")
  Two-branch cooperative game-theoretic fusion of resting-state functional
  connectivity and plasma proteomics · UK Biobank
  #v(1pt)
  1 August 2026
]
#v(-2pt)
#line(length: 100%, stroke: 1.1pt + accent)
#v(6pt)

All results below are held-out test-set numbers. Joint-model figures are
aggregated over *11 independent training runs* differing only in random seed,
each trained to convergence under its own early-stopping criterion. Participants
used to evaluate the joint model are excluded from every pretraining and
baseline fit, so no evaluation subject influences any model it is scored on.

= 1. Fluid intelligence — current results

The joint model is trained on 3,471 participants and evaluated on 747 held-out
participants. Alongside the fused prediction it exposes three internal
predictions: an fMRI-only head, a proteomics-only head, and a plain MLP on the
concatenated embeddings. Scoring all four on the same subjects shows where the
performance originates.

#tbl(
  (1fr, 30mm, 30mm, 22mm),
  (left, right, right, right),
  ([Output], [$R^2$], [$R^2$ calibrated], [Pearson $r$]),
  [*Fused prediction (full model)*], [*0.0725 ± 0.0056*], [*0.0768*], [*0.279*],
  [fMRI branch only],               [0.0652 ± 0.0028],  [0.0665],   [0.260],
  [Proteomics branch only],         [−0.0010 ± 0.0022], [0.0285],   [0.172],
  [Concatenation head (naive fusion)], [0.0665 ± 0.0060], [0.0706], [0.267],
)

*The calibrated column matters.* An affine rescaling fitted on the validation
fold and applied to the test fold corrects a systematic amplitude error in the
model outputs. It changes the reading of the proteomics branch entirely: a raw
$R^2$ of approximately zero becomes 0.0285, essentially the theoretical maximum
for a predictor correlating at $r$ = 0.17. The proteomics branch is
therefore *extracting all of the signal available to it* — its raw score of zero
reflected a collapsed output scale, not an absence of information.

== Comparisons between the model's own outputs

#tbl(
  (1fr, 26mm, 30mm, 23mm),
  (left, right, right, right),
  ([Contrast], [Mean #sym.Delta#h(1pt)$R^2$], [Seeds in favour], [$t$ (df = 10)]),
  [Fused − fMRI branch], [+0.0073], [9 / 11], [+3.82],
  [*Fused − fMRI branch (calibrated)*], [*+0.0103*], [*11 / 11*], [—],
  [*Concatenation head − fused*], [*−0.0060*], [*1 / 11*], [*−4.19*],
)

The cooperative fusion output exceeds its own fMRI branch on 11 of 11 seeds once
calibrated, and exceeds the naive concatenation head on 10 of 11. Within the
architecture, the Shapley-guided gate outperforms both the single-modality path
and the standard concatenation alternative.

== Why the proteomic contribution is small

The proteomics-to-cognition association decomposes cleanly, and the loss is
attributable rather than anomalous.

#tbl(
  (62mm, 22mm, 1fr),
  (left, right, left),
  ([Setting], [Pearson $r$], [Explanation]),
  [Full proteomics cohort, baseline visit], [0.345], [reference],
  [Imaging subcohort, baseline visit],      [0.254], [cohort restriction],
  [Imaging subcohort, imaging visit],       [0.171], [#sym.tilde.op#h(1pt)9-year blood-to-scan interval],
)

The final step is fully accounted for by measurement attenuation: the
test–retest correlation of the cognitive measure between the two visits is 0.636
on these participants, and 0.254 × 0.636 = 0.162 against 0.171 observed. A
predictor correlating at $r$ = 0.17 is capped at $R^2 = 0.029$
however it is modelled. *This is a property of the cohort design* — plasma was
sampled roughly nine years before the scan — *and not of the model.*

#pagebreak()

= 2. Ablation and baseline testing — fluid intelligence

Every model below is evaluated on the same 747 held-out participants. Two
families of linear baseline were run. The first is fitted on the joint training
fold only (3,471 participants). The second is given the same advantage the deep
encoders receive: fitted on the full single-modality cohort with all joint
validation and test participants excluded, then adapted to the joint training
fold by a two-parameter affine map. The second family is the fair comparison.

#tbl(
  (1fr, 24mm, 24mm, 22mm),
  (left, right, right, right),
  ([Model], [Fitted on], [$R^2$], [Pearson $r$]),
  [Predict the training mean],                 [—],      [−0.0007], [—],
  [Ridge, proteomics (2,920 features)],        [3,471],  [0.0242],  [0.165],
  [Ridge, connectivity (2,145 edges)],         [3,471],  [0.0409],  [0.213],
  [Ridge, concatenated features],              [3,471],  [0.0317],  [0.183],
  [Ridge, proteomics — pretrained],            [11,528], [0.0363],  [0.195],
  [Ridge, connectivity — pretrained],          [30,511], [0.0818],  [0.290],
  [*Late fusion of the two pretrained ridges*], [both],  [*0.1035*], [*0.325*],
  [TCG-Net, fused output],                     [3,471],  [0.0725],  [0.279],
  [*TCG-Net, fused output (calibrated)*],      [3,471],  [*0.0768*], [—],
)

== What the baselines establish

*(a) Multimodal integration genuinely helps.* Late fusion of the two pretrained
linear models reaches 0.1035 against 0.0818 for connectivity alone — a gain of
+0.0217. Plasma proteomics does add information beyond the connectome for
cognitive prediction.

*(b) Feature-level concatenation is not the way to combine them.* Fitted on the
joint training fold, concatenating raw features (0.0317) performs *worse* than
connectivity alone (0.0409). Naive early fusion destroys signal at this sample
size, which motivates a learned fusion mechanism.

*(c) The current network does not yet beat the strongest linear baseline.* Late
fusion of pretrained ridges reaches 0.1035 against 0.0768 for the calibrated
fused output — a shortfall of 0.027 $R^2$, several times the seed-to-seed
standard deviation of 0.006.

#keybox[
  The likely explanation is *representational rather than architectural*. The
  linear model uses all 2,145 connectivity edges with weights estimated from
  30,511 participants, whereas the graph encoder compresses the connectome to
  128 dimensions and its downstream heads are refitted on 3,471. The fusion
  mechanism itself performs well in every internal comparison (Section 1); it is
  the encoded representations that trail the raw features.
]

#note[
  *Ablations still outstanding.* Single-modality, early-fusion and late-fusion
  comparisons are covered above. Not yet run: attention-based fusion, a learned
  non-Shapley modality gate, and the model without the interaction term, the
  squeeze-and-excitation block, or the graph encoder. The learned non-Shapley
  gate is the most informative of these, since it isolates the contribution of
  the game-theoretic weighting specifically.
]

= 3. Ablation and baseline testing — age

Age was assessed as a second target because both modalities carry substantial
signal for it, which fluid intelligence does not permit.

== Single-modality proteomics encoder

#tbl(
  (1fr, 24mm, 26mm, 22mm),
  (left, right, right, right),
  ([Model], [$R^2$], [MAE (years)], [Pearson $r$]),
  [Predict the training mean],       [−0.0000], [6.99], [—],
  [*Ridge, all 2,920 proteins*],     [*0.8690*], [*2.35*], [*0.932*],
  [Ridge, 200 strongest proteins],   [0.7677],  [3.14], [0.876],
  [*Proteomics encoder (Stage 1)*],  [*0.868*], [—],    [—],
)

Evaluated on 7,904 held-out participants, *the proteomics encoder and ridge
regression are indistinguishable* (0.868 against 0.8690). Age is linearly
decodable from an Olink panel, so this figure reflects the information content of
the assay rather than the encoder architecture. This was the specific comparison
requested, and the answer is that the proteomics branch does not require a deep
architecture.

== Joint cohort — the multimodal baselines

Evaluated on 799 held-out participants from the joint cohort, using the same two
baseline families as in Section 2.

#tbl(
  (1fr, 24mm, 24mm, 22mm),
  (left, right, right, right),
  ([Model], [Fitted on], [$R^2$], [Pearson $r$]),
  [Predict the training mean],          [—],      [−0.0001], [—],
  [Ridge, connectivity],                [3,730],  [0.2469],  [0.498],
  [Ridge, proteomics],                  [3,730],  [0.7742],  [0.880],
  [Ridge, concatenated features],       [3,730],  [0.7828],  [0.885],
  [Ridge, connectivity — pretrained],   [33,081], [0.3526],  [0.595],
  [Ridge, proteomics — pretrained],     [35,754], [0.8106],  [0.901],
  [*Late fusion of the two pretrained ridges*], [both], [*0.8228*], [*0.908*],
)

*The multimodal gain is positive for age.* Late fusion improves on proteomics
alone by +0.0122 $R^2$, and unlike fluid intelligence, concatenating raw features
also helps rather than hurts (0.7828 against 0.7742). Both modalities carry real
age information and are partially complementary — although the connectome's
contribution is largely redundant with the proteome's, since only about 3.5% of
its standalone signal survives as independent.

Pretraining is worth a great deal on the imaging side specifically: connectivity
ridge improves from 0.2469 to 0.3526 when fitted on 33,081 rather than 3,730
participants. This is why the pretrained baseline family was constructed —
comparisons against joint-fold-only baselines systematically overstate any deep
model's advantage.

== Status of the joint age model

The two-branch model has not yet been run on age. It requires a connectome
encoder pretrained against age, which is pending on the imaging side. The tables
above therefore establish the threshold rather than a comparison:

#keybox[
  The joint age model must exceed $R^2 = 0.8228$ ($r$ = 0.908) to
  demonstrate value over a purely linear approach.
]

= Summary

The pipeline now produces properly held-out, multi-seed, leakage-controlled
results on both targets. Three findings stand out.

#block(inset: (left: 6pt))[
  #set enum(numbering: n => text(fill: accent, weight: "bold")[#n.])
  + The cooperative fusion mechanism outperforms both single-modality prediction
    and naive concatenation within the architecture, consistently across seeds.

  + Multimodal integration is genuinely informative for both targets. The
    proteomic contribution to cognition is small for a quantifiable reason — a
    nine-year interval between blood sampling and scanning that attenuates the
    association to a ceiling of $R^2 = 0.029$.

  + Neither deep encoder currently outperforms a regularised linear model given
    the same training data, which locates the remaining work in the learned
    representations rather than in the fusion mechanism.
]

*Immediate priorities.* Complete the remaining ablations, particularly the
learned non-Shapley gate; obtain an age-trained connectome encoder so the joint
age model can be evaluated against the 0.8228 threshold; and test the branch
comparisons at the participant level by bootstrap, since the figures reported
here quantify variation across training runs rather than across participants.
