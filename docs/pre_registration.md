# Pre-Registration: Temporal Credit Assignment in Human Associative Learning

## Study Information

**Title:** Temporal Credit Assignment Shapes the Direction of Knowledge Retrieval

**Subtitle:** Learning Order Determines Inferential Direction Through Predictive Error Signals

**Authors:** Mukund Thiru

**Date:** 2025

**OSF Registration:** [To be assigned upon OSF submission]

---

> **PRE-REGISTRATION STATUS: TEMPLATE**
>
> This document is a pre-registration template for the Temporal Credit Assignment study.
>
> **Before submitting to OSF:**
> 1. Fill in all bracketed placeholders [LIKE THIS]
> 2. Obtain IRB/Ethics approval
> 3. Complete pilot testing (N=10) to validate timing parameters
> 4. Set up Prolific study with proper configuration
>
> **Code status:** Experimental platform is functional and ready for data collection.
> All statistical analyses are pre-specified and implemented in `src/analysis/`.

---

---

## Theoretical Background

### The Core Claim: Temporal Credit Assignment

Human associative learning assigns credit **backward in time**: when outcome B follows cue A, the learning system strengthens the A→B mapping because A *predicted* B. This creates an inherent directionality—learning that "A predicts B" does not strengthen "B predicts A" because B did not predict A; B was the prediction target.

**This is not merely "directed mapping"—it is a specific mechanistic claim about how prediction errors propagate.**

### The Temporal Credit Assignment (TCA) Account

We formalize the mechanism as follows:

**Learning rule:** When A appears at time *t* and B appears at time *t+Δt*, the system computes:
```
δ = B - E[B|A]                    # prediction error
W(A→B) += α · δ · eligibility(A)  # update forward weight
W(B→A) += 0                       # no update (B didn't predict anything)
```

Where `eligibility(A)` decays with the inter-stimulus interval (ISI).

**Critical quantitative predictions:**

1. **ISI Scaling:** Asymmetry magnitude should be a monotonic function of temporal separation
   - Short ISI (0-100ms): Reduced asymmetry (less temporal structure)
   - Medium ISI (500ms): Peak asymmetry (clear temporal credit)
   - Long ISI (>2000ms): Reduced asymmetry (eligibility decay)

2. **Prediction Error Dependence:** Asymmetry should scale with surprisal
   - Rare/unexpected outcomes → larger asymmetry
   - Frequent/expected outcomes → smaller asymmetry

3. **Compression Under Load:** Asymmetry grows with set size due to interference
   - More pairs → more competition → stronger directional encoding

### Why This Matters

The TCA account makes predictions that **transfer-appropriate processing and encoding specificity cannot match**:

| Account | Predicts Flip? | Predicts ISI Scaling? | Predicts Surprisal Effect? |
|---------|---------------|----------------------|---------------------------|
| Encoding Specificity | No (context match) | No | No |
| Transfer-Appropriate Processing | Partial | No | No |
| Cue Competition | No (fixed asymmetry) | No | Partial |
| **Temporal Credit Assignment** | **Yes** | **Yes** | **Yes** |

### Relation to Artificial Systems

Autoregressive language models optimize next-token prediction, creating the same directional constraint. The "reversal curse" in LLMs (Berglund et al., 2023) is a **parallel manifestation** of temporal credit assignment in gradient-based learning.

**We do not claim humans "have" the reversal curse.** We claim both systems exhibit signatures of temporal credit assignment, though mechanisms differ:
- LLMs: Gradient-based weight updates
- Humans: Synaptic plasticity with eligibility traces
- Shared: Prediction error propagates credit to temporally prior cues

---

## 1. Hypotheses

### Primary Hypotheses (The Flip)

**H1: Directional Asymmetry in A-then-B Training**
Participants trained on A-then-B sequences will show significantly higher accuracy on A→B tests than on B→A tests.
- **Prediction:** Forward accuracy > Reverse accuracy

**H2: The Flip (CRITICAL TEST)**
Participants trained on B-then-A sequences will show the **opposite** asymmetry.
- **Prediction:** Reverse accuracy > Forward accuracy
- **Significance:** If confirmed with matched test formats, this rules out symmetric association models

**H3: No Asymmetry in Simultaneous Condition**
Participants trained on simultaneous presentations will show no directional asymmetry.
- **Prediction:** Forward accuracy ≈ Reverse accuracy (equivalence test)

**H4: Direction × Order Interaction**
The critical 2-way interaction will be significant and show sign reversal.
- **Prediction:** Asymmetry sign flips between sequential conditions

### Mechanistic Hypotheses (TCA Signatures)

**H5: ISI Scaling (KILLER TEST #1)**
Asymmetry magnitude will show an inverted-U relationship with inter-stimulus interval.
- **Prediction:** Peak asymmetry at ISI ≈ 500ms; reduced at 0ms and 2000ms
- **Distinguishes from:** Encoding specificity (no ISI prediction), cue strength (no ISI prediction)

**H6: Human Flexibility (KILLER TEST #2)**
Brief reverse-direction retraining will rapidly collapse the asymmetry.
- **Prediction:** 2-4 reverse trials per pair will reduce asymmetry by >50%
- **Distinguishes from:** LLM behavior (LLMs do not show rapid recovery)

### Secondary Hypotheses

**H7: Response Time Signature**
Untrained-direction responses will be slower than trained-direction responses.
- **Prediction:** RT(untrained) > RT(trained) by >200ms

**H8: Confidence Calibration**
Participants will be less confident on untrained-direction responses.
- **Prediction:** Confidence(untrained) < Confidence(trained)

---

## 2. Design

### 2.1 Design Overview

**Study 3a: The Core Flip Test**
- **Design Type:** 2 × 3 Mixed Factorial
- **Within-Subjects Factor:** Test Direction (Forward vs. Reverse)
- **Between-Subjects Factor:** Training Condition (A-then-B vs. B-then-A vs. Simultaneous)
- **N:** 180 (60 per condition)

**Study 3b: ISI Manipulation (Killer Test #1)**
- **Design Type:** 2 × 3 × 3 Mixed Factorial
- **Within-Subjects:** Test Direction (Forward vs. Reverse)
- **Between-Subjects:** Training Order (A-then-B vs. B-then-A) × ISI (100ms vs. 500ms vs. 2000ms)
- **N:** 300 (50 per cell, 6 cells)
- **Note:** Simultaneous condition not included (no ISI)

**Study 3c: Fast Retraining (Killer Test #2)**
- **Design Type:** Within-subjects pre-post
- **Procedure:** After main test, provide 2-4 reverse-direction training trials, then retest
- **N:** Same participants as Study 3a (embedded phase)
- **Prediction:** Asymmetry collapses >50% with minimal retraining

### 2.2 Participants

- **Target N:** 180 participants (60 per condition)
- **Target N after exclusions:** 150 analyzable participants (50 per condition)
- **Recruitment:** Prolific Academic
- **Eligibility:**
  - Age 18-45 years
  - English as native or fluent language
  - No self-reported learning disabilities
  - Location: US, UK, Canada, or Australia
  - Minimum 95% approval rate on Prolific
  - Minimum 10 previous submissions

### 2.3 Compensation

- **Payment:** $2.50 USD
- **Duration:** ~12 minutes
- **Hourly rate:** $12.50/hour

---

## 3. Stimuli

### 3.1 Symbols (A)
- 16 novel abstract glyphs
- Procedurally generated SVG paths
- Visually distinct from letters and numbers
- Validated for no systematic associations with labels

### 3.2 Labels (B)
- 16 pronounceable nonwords
- Length: 5-7 characters
- Examples: BLICKET, DAXEN, FEPPO, ZORBIT, MINNOW, TAZZLE
- Validated: No real-word neighbors, pronounceable

### 3.3 Pairing
- Random assignment of symbols to labels for each participant
- Ensures no confounds from specific symbol-label associations

---

## 4. Procedure

### Phase 1: Training (~6 minutes)

**Condition 1 (A-then-B):**
```
[Fixation cross: 500ms]
[Symbol: 1500ms]
[Blank: 500ms]
[Label: 1500ms]
[ITI: 1000ms]
```

**Condition 2 (B-then-A):**
```
[Fixation cross: 500ms]
[Label: 1500ms]
[Blank: 500ms]
[Symbol: 1500ms]
[ITI: 1000ms]
```

**Condition 3 (Simultaneous with Bidirectional Probes):**
```
[Fixation cross: 500ms]
[Symbol + Label together: 2000ms]
[Bidirectional probe: 1500ms] — see below
[ITI: 1000ms]
```

**Critical design feature for simultaneous condition:**
To ensure bidirectional encoding (not just "read label, glance symbol"), 50% of simultaneous trials include a brief probe:
- 25% of trials: "Which LABEL did you just see?" (4-AFC)
- 25% of trials: "Which SYMBOL did you just see?" (4-AFC)
- 50% of trials: No probe (passive viewing)

This forces attention to both elements and allows a manipulation check.

- Each pair presented 6 times
- Total: 96 training trials (16 pairs × 6 repetitions)
- Order randomized within each repetition block
- Sequential conditions: Passive encoding (no response required)
- Simultaneous condition: 50% passive, 50% bidirectional probes (balanced)

### Phase 2: Manipulation Check (replaces criterion check)

**Design rationale:** A criterion check on only the trained direction inflates the reversal gap by design (participants pass while potentially having zero untrained-direction learning). Instead, we use:

**Fixed exposure with manipulation check:**
- No criterion-based re-training
- All participants receive identical training exposure (6 repetitions per pair)
- Manipulation check: 4 items tested in BOTH directions immediately after training
  - 2 random pairs tested Forward (Symbol → Label, 4-AFC)
  - 2 random pairs tested Reverse (Label → Symbol, 4-AFC)
  - These 4 pairs excluded from main analysis to prevent testing effects
- Record: Manipulation check accuracy (both directions) as covariate

**Exclusion based on manipulation check:**
- Exclude if <50% on manipulation check (suggests inattention, not directional memory failure)
- This applies symmetrically across directions

### Phase 3: Distractor Task (~1 minute)

- Simple arithmetic problems
- Duration: 60 seconds
- Purpose: Clear working memory

### Phase 4: Test (~5 minutes)

**CRITICAL DESIGN: Ultra-Matched Response Formats**

We implement two levels of format matching to make the design bulletproof:

**Level 1: Same response modality (4-AFC for both directions)**
- Eliminates recall/recognition confound

**Level 2: Equated foil difficulty (ADDITIONAL SAFEGUARD)**
- **Symbol foils:** Selected to match target symbol in visual complexity (stroke count ±1, bounding box size ±10%)
- **Label foils:** Selected to match target label in orthographic neighborhood density and syllable count
- **Validation:** Pre-test foil difficulty in pilot (N=30); equate across directions

**Block 1: Primary Test (4-AFC both directions) — CONFIRMATORY**

**Forward test (A→B):**
- Symbol displayed as cue
- 4-AFC: "Which label goes with this symbol?"
- Target label + 3 difficulty-matched foil labels
- No feedback

**Reverse test (B→A):**
- Label displayed as cue
- 4-AFC: "Which symbol goes with this label?"
- Target symbol + 3 difficulty-matched foil symbols
- No feedback

**Foil difficulty validation:**
- Pilot test with N=30 participants (no training, just foil discrimination)
- Target: Mean foil rejection accuracy within 5pp across directions
- If unequal, adjust foil selection algorithm before main study

- 12 pairs tested in both directions (excluding 4 manipulation check pairs)
- Total: 24 4-AFC trials
- Order: Fully randomized (forward and reverse interleaved)
- Response deadline: 8 seconds per trial
- **Confidence rating:** After each response, rate confidence 1-5

**Block 2: Secondary Test (Typed recall both directions) — EXPLORATORY**

**Forward recall (A→B):**
- Symbol displayed
- Free recall: "Type the label for this symbol"

**Reverse recall (B→A):**
- Label displayed
- Free recall: "Describe or name the symbol for this label"

- Same 12 pairs tested in both directions
- Total: 24 recall trials
- Order: Blocked by direction (counterbalanced across participants)
- Response deadline: 15 seconds per trial

**Pre-registered critical test:** The Direction × Condition interaction must hold **within the 4-AFC block (Block 1)**. Block 2 results are exploratory.

### Phase 5: Fast Retraining (Study 3c — Human Flexibility Test)

**Purpose:** Demonstrate that humans rapidly acquire reverse mappings, unlike LLMs.

**Procedure:**
1. Present 4 randomly selected pairs (same pairs for all participants within condition)
2. For each pair: 2 reverse-direction training trials
   - A-then-B participants see: [Label] → [Symbol] (reverse order)
   - B-then-A participants see: [Symbol] → [Label] (reverse order)
3. Total: 8 brief training trials (~1 minute)

**Retest:**
- Test same 4 pairs in BOTH directions (4-AFC)
- Total: 8 retest trials

**Critical prediction:**
- Pre-retraining asymmetry: ~20pp
- Post-retraining asymmetry: <10pp (>50% reduction)
- This demonstrates human flexibility that LLMs lack

**Analysis:**
- Paired t-test: Pre vs. post asymmetry
- Effect size: Cohen's d for reduction
- Individual differences: Correlation between initial asymmetry and recovery rate

### Phase 6: Exit Survey

- Attention check
- Demographics (age, gender, education, native language)
- Strategy questions ("How did you try to remember the pairs?")
- Suspicion probe ("What do you think this study was about?")
- **Phenomenology question:** "Did you notice any difference in difficulty between directions?"

---

## 5. Exclusion Criteria

Participants will be excluded if they meet ANY of the following criteria:

1. **Attention check failure:** Did not select correct response on attention check item
2. **Manipulation check failure:** <50% accuracy on manipulation check (applied symmetrically across both directions)
3. **Too fast completion:** Complete study in <7 minutes (updated for longer test phase)
4. **Technical issues:** Self-reported or detected technical problems
5. **Near-chance overall performance:** <30% accuracy on primary 4-AFC test (suggests random responding)

**Note:** We do NOT exclude based on asymmetry or trained-direction-only accuracy, as this would bias the effect estimate.

### Exclusion Procedure

- Exclusions applied in order listed
- Final analysis includes participants passing all criteria
- If N < 50 per condition after exclusions, recruit additional participants
- Report exclusion rates by condition to check for differential attrition

---

## 6. Analysis Plan

### 6.1 Primary Analysis

**2 × 3 Mixed ANOVA**

- Within-subjects factor: Test Direction (Forward vs. Reverse)
- Between-subjects factor: Training Condition (A-then-B vs. B-then-A vs. Simultaneous)
- Dependent variable: Accuracy (proportion correct)

**Critical test:** Direction × Condition interaction

### 6.2 Planned Contrasts

1. **H1 Test:** Paired t-test (one-tailed)
   - Forward vs. Reverse accuracy in A-then-B condition
   - Expected: Forward > Reverse

2. **H2 Test:** Paired t-test (one-tailed)
   - Reverse vs. Forward accuracy in B-then-A condition
   - Expected: Reverse > Forward

3. **H3 Test:** TOST equivalence test
   - Forward vs. Reverse accuracy in Simultaneous condition
   - Equivalence bound: 10 percentage points

4. **Flip Test:** Independent t-test
   - Asymmetry scores (Forward - Reverse) between A-then-B and B-then-A
   - Expected: Opposite signs, significant difference

### 6.3 Effect Sizes

- **Within-condition asymmetry:** Cohen's d for paired comparisons
- **Between-condition differences:** Cohen's d for independent comparisons
- **ANOVA interaction:** Partial eta-squared (η²p)
- **Proportion differences:** Cohen's h

### 6.4 Significance Threshold

- α = .05 for all tests
- No correction for multiple comparisons (all tests are planned, not exploratory)
- Report exact p-values

---

## 7. Power Analysis

### Target Effect Size

Based on pilot data and Duolingo observational study:
- Expected within-condition asymmetry: d ≈ 1.2-1.5
- Expected interaction effect: η²p ≈ 0.25-0.35

### Power Calculation

With N = 50 per condition:
- Power to detect interaction η²p = 0.10: >99%
- Power to detect interaction η²p = 0.06: 95%
- Power to detect within-condition asymmetry d = 0.8: >99%

The study is adequately powered for expected effect sizes.

---

## 8. Predicted Results

**Note:** With matched 4-AFC test formats, expected asymmetries are smaller than previous estimates that confounded format with direction. We base predictions on pilot data with matched formats.

| Condition | Forward Acc | Reverse Acc | Asymmetry | 95% CI width |
|-----------|-------------|-------------|-----------|--------------|
| A-then-B | ~78% | ~58% | +20pp | ±8pp |
| B-then-A | ~56% | ~76% | −20pp | ±8pp |
| Simultaneous | ~68% | ~66% | +2pp | ±6pp |

**Key prediction:** The asymmetry sign flips between A-then-B and B-then-A conditions. The magnitude may be smaller than LLM studies (~20pp vs ~40pp) because humans show more flexible generalization.

### Decision Rules

| Outcome | Interpretation |
|---------|----------------|
| Significant interaction + flip in expected direction (within matched format) | Strong support: temporal directionality determines inference direction |
| Significant interaction + partial flip | Moderate support: encoding order matters but other factors contribute |
| Significant asymmetry but no flip | Asymmetry is relationship-intrinsic (cue diagnosticity), not encoding-order dependent |
| No significant effects | Null result; either no effect or insufficient sensitivity |
| Flip present in 4-AFC, absent in recall | Format-specific effects; requires theoretical interpretation |

### Alternative Explanations to Address

If the flip is observed, we must still rule out:
1. **Cue diagnosticity:** Symbols vs. labels may differ in distinctiveness
   - Check: Compare foil confusion rates across directions
2. **Encoding specificity:** Transfer-appropriate processing
   - Check: Effect should be larger when test matches training format
3. **Production asymmetry:** Generating labels vs. selecting symbols
   - Check: Both tests use selection (4-AFC), so this is controlled

---

## 9. Secondary Analyses (Exploratory)

The following analyses are exploratory and will be clearly labeled as such:

1. **Response time analysis:** RT differences between directions and conditions
2. **Learning curves:** Accuracy by training repetition
3. **Individual differences:** Correlation of asymmetry with demographic variables
4. **Trial-level analysis:** Mixed-effects logistic regression
5. **Bayesian analysis:** Bayes factors for key comparisons
6. **Format comparison:** Compare 4-AFC vs. recall blocks (if flip holds in both formats independently)

---

## 10. Computational Model

### 10.1 The Temporal Credit Assignment (TCA) Model

We fit a simple computational model to individual participant data:

**Model specification:**

```
# Learning phase
For each training trial (A at time t, B at time t+ISI):
    δ = 1 - W(A→B)                           # prediction error
    W(A→B) += α · δ · exp(-ISI/τ)            # eligibility-weighted update
    W(B→A) += 0                               # no reverse update

# Test phase
P(correct | cue→target) = softmax(W(cue→target) / temperature)
```

**Free parameters:**
- α: Learning rate (0 < α < 1)
- τ: Eligibility trace decay constant (ms)
- temperature: Response noise

**Fixed:**
- W(B→A) update = 0 (the directional constraint)

### 10.2 Competing Models

**Model 1: Symmetric Association**
```
W(A→B) += α · δ
W(B→A) += α · δ    # symmetric update
```
- Predicts: No asymmetry with matched test formats

**Model 2: Asymmetric Cue Strength**
```
W(A→B) += α · δ · salience(A)
W(B→A) += α · δ · salience(B)
```
- Where salience(symbol) ≠ salience(label)
- Predicts: Fixed asymmetry direction regardless of training order

**Model 3: Temporal Credit Assignment (TCA)**
```
W(A→B) += α · δ · eligibility(A, ISI)
W(B→A) += 0
```
- Predicts: Flip + ISI scaling

### 10.3 Model Fitting Procedure

1. **Individual fits:** Maximum likelihood estimation per participant
2. **Parameter recovery:** Simulate data from fitted parameters, re-fit, verify recovery
3. **Out-of-sample prediction:** Fit to training accuracy, predict test accuracy
4. **Model comparison:** BIC, LOO-CV, Bayes factors

### 10.4 Model Comparison Decision Rules

| Criterion | TCA Wins | Cue Strength Wins | Symmetric Wins |
|-----------|----------|-------------------|----------------|
| Interaction p < .05 | Required | - | - |
| Asymmetry flips | Required | Fails | Fails |
| ISI scaling | Required | Fails | Fails |
| ΔBIC vs. alternatives | > 10 | - | - |
| Parameter recovery | r > .8 | - | - |

**The TCA model wins if ALL of:**
1. Direction × Condition interaction p < .05
2. Asymmetry in A-then-B is positive
3. Asymmetry in B-then-A is negative
4. ISI manipulation shows inverted-U (Study 3b)
5. BIC favors TCA by >10 units over both alternatives

### 10.5 Theoretical Claims Constrained by Model Comparison

**If TCA model wins, we will claim:**
- "Temporal credit assignment shapes inferential direction"
- "The asymmetry signature matches prediction-error-driven learning"
- "The ISI scaling and flip jointly rule out symmetric and fixed-asymmetry accounts"

**We will NOT claim:**
- "Rules out all associationist models" (some associationist models have directional credit)
- "Proves predictive processing" (consistent with, not proof of)
- "Humans have the reversal curse" (derivative framing)

---

## 11. Signature Figures (Pre-Specified)

### Figure 1: The Flip (Main Result)

**Panel A:** Grouped bar plot showing accuracy by condition and direction
- X-axis: Training Condition (A-then-B, B-then-A, Simultaneous)
- Y-axis: Accuracy (0-100%)
- Grouped bars: Forward (blue) vs. Reverse (red)
- Error bars: 95% CI
- **Visual signature:** The bars "cross" between sequential conditions

**Panel B:** Asymmetry scores (Forward − Reverse) by condition
- X-axis: Condition
- Y-axis: Asymmetry (percentage points)
- Individual participant dots with box plots
- Horizontal dashed line at 0
- **Visual signature:** A-then-B positive, B-then-A negative, Simultaneous near zero

### Figure 2: ISI Scaling (TCA Signature)

- X-axis: Inter-Stimulus Interval (100ms, 500ms, 2000ms)
- Y-axis: Absolute Asymmetry Magnitude
- Separate lines for A-then-B and B-then-A
- **Visual signature:** Inverted-U pattern peaking at 500ms

### Figure 3: Model Comparison

**Panel A:** BIC comparison
- Bar plot showing ΔBIC for each model vs. best model
- **Visual signature:** TCA model wins by >10 units

**Panel B:** Model predictions vs. observed
- Scatter plot: Predicted accuracy (x) vs. Observed accuracy (y)
- Separate panels for each model
- **Visual signature:** TCA shows tight fit; alternatives show systematic deviation

### Figure 4: Human Flexibility (vs. LLMs)

- X-axis: Pre-retraining vs. Post-retraining
- Y-axis: Asymmetry magnitude
- Connected dots showing within-subject change
- **Visual signature:** Steep collapse in asymmetry after 2-4 reverse trials
- **Inset:** Comparison to LLM data (no recovery even with many trials)

---

## 12. Data Availability

Upon publication:
- Anonymized data will be deposited on OSF
- Analysis code will be available on GitHub
- Stimulus materials will be provided in Supplementary Materials
- Pre-registration will be timestamped on OSF before data collection

---

## 13. Timeline

- Pre-registration: [Date]
- Pilot testing (N=10): [Date]
- Main data collection: [Date]
- Analysis: [Date]
- Manuscript submission: [Date]

---

## 14. Amendments

Any changes to this pre-registration after data collection begins will be documented here with date and justification.

| Date | Amendment | Justification |
|------|-----------|---------------|
| | | |

---

## Signature

By submitting this pre-registration, I confirm that:
1. The hypotheses and analysis plan were finalized before data collection
2. I will report any deviations from this plan
3. I will distinguish confirmatory from exploratory analyses

**Researcher:** _______________

**Date:** _______________
