# Leakage-Resistant Retrieval-Augmented Forecasting: A Preregistered Cross-Domain Audit

**Anonymous authors**  
Protocol: `sv1-v5-retrieval-audit-20260721` · software version 0.1.0

## Abstract

Retrieval augmentation offers an appealing route to adapting a forecaster at inference time: find
historical contexts similar to the current query and reuse what happened next. Yet time-series
retrieval is unusually vulnerable to information contamination. A candidate's future can overlap a
query target; a stored residual can be computed by a model that fitted the same window; and an
abstention rule can be tuned on the test period. We present Sv1, a preregistered audit of
retrieval-augmented forecasting that separates these failure modes. The benchmark contains
234 series from ten Monash collections and 1,872 sealed confirmation
windows spanning competition, tourism, finance, transport, air quality, banking, macroeconomic,
mobility, and climate domains. A three-seed official PatchTST architecture is the trained reference,
official Chronos-Bolt-small is a zero-shot reference, and AutoTheta and three naive methods provide
classical controls. The proposed revision retrieves forward cross-fitted residuals and abstains when
a calibration-only utility model predicts no benefit. All choices are made on development data,
hashed, and locked before confirmation. On sealed confirmation, the selected method changed MSE by
1.25% relative to PatchTST (95% hierarchical moving-block interval
[-0.31%, 4.08%]); it revised 19.4% of forecasts and improved
5/10 datasets. Under the registered 1% smallest effect size of interest, the decision
was **inconclusive**. The exact self-inclusion stress test, provenance ablations, 4,000-draw
hierarchical inference, official-model references, thirty vector figures, and executable artifact
graph turn the result—positive or null—into a falsifiable account of when retrieval evidence is
valid. These findings argue that residual provenance and selective evaluation are not secondary
implementation details but part of the estimand in retrieval-augmented forecasting.

**Keywords:** time-series forecasting; retrieval augmentation; data leakage; cross-fitting;
selective prediction; preregistration; foundation models; reproducibility

## 1. Introduction

Forecasting systems usually compress a training corpus into model parameters and discard the
individual historical cases from which those parameters were learned. Retrieval-augmented
forecasting reverses that design choice. At prediction time it searches a memory for cases that
resemble the current context and exposes their observed continuations, representations, or
forecasting errors to the model. This can provide a useful nonparametric inductive bias: recurring
seasonal shapes, regime-specific slopes, and transient motifs may be easier to reuse than to encode
in a fixed set of parameters. Recent work has consequently introduced direct future retrieval,
learned mixture modules, retrieval-guided diffusion, and post-hoc instance revision
[@han2025raft; @ning2025tsrag; @liu2024ratd; @liu2025pir].

The same design creates a distinctive validity problem. A retrieved object has two time axes: the
context used to decide that it is similar and the continuation or residual used to change the
forecast. It is not enough for the context to precede a query origin. The retrieved value must also
have been observable, and any model used to create that value must not have fitted the case on which
its error is measured. If a query retrieves its own continuation, the task collapses. If a residual
memory contains training residuals, a high-capacity backbone can make those residuals
unrepresentatively small or structured; the retrieval experiment then estimates performance under
a memory that cannot be produced for genuinely new cases. A third dependency arises when a router
learns to revise only forecasts that would otherwise be bad. If its threshold is chosen from test
outcomes, selective forecasting becomes outcome selection.

These concerns are related to, but not resolved by, ordinary chronological train/test splitting.
Time-series model evaluation already requires respecting dependence and forecast origin
[@bergmeir2018validation; @cerqueira2020evaluation]. Retrieval introduces a relational layer: every
query–candidate edge must be causal. Post-hoc revision adds another fitted component whose
training, thresholding, model selection, and confirmation roles must be separated. Reported mean
improvements may otherwise mix real forecasting signal, in-sample optimism, and repeated use of
the test period.

This paper asks a deliberately narrow question: **Does residual retrieval still improve a strong
forecasting backbone when candidates are causal, residuals are forward cross-fitted, abstention is
calibrated without development or test targets, and success requires a practically meaningful
confirmatory effect?** We answer it with a fresh ten-dataset study that does not overwrite the
earlier exploratory notebook. The study uses deterministic series selection, fit-only scaling,
disjoint temporal target roles, three backbone seeds, an official zero-shot foundation-model
reference, classical controls, negative controls, a self-inclusion stress test, and a sealed
confirmation period.

The contribution is methodological as much as algorithmic. First, we formalize retrieval memories
by their value provenance and distinguish in-sample, leave-one-out, and forward cross-fitted
residuals. Leave-one-out prevents exact self-matching but does not make the stored residuals
out-of-sample; cross-fitting addresses the second problem. Second, we combine cross-fitted residual
retrieval with a calibration-only utility router, which turns retrieval into a selective action
rather than an unconditional correction. Third, we register a complete selection and inference
path. Nine values of neighbourhood size and temperature are evaluated in development; one is
locked; the confirmation role is opened once. Fourth, we release an evidence package designed to
make overclaiming difficult: fidelity labels distinguish executed official implementations from
equation-level adapters, every figure has a registered evidentiary purpose, and the final decision
is generated from machine-readable criteria.

The result is not framed as a guaranteed new state of the art. A statistically positive effect
smaller than 1% is labelled negligible; an interval crossing zero is inconclusive; a robust average
with fewer than six winning datasets is mixed. This discipline is important because a Q1-quality
paper is not defined by a large positive number. It is defined by a consequential question,
credible comparison, transparent uncertainty, and a conclusion that remains valid when the hoped-
for result does not occur.

## 2. Related work

### 2.1 Forecasting backbones and foundation models

Modern forecasting research spans statistical local models, global neural models, and pretrained
foundation models. Theta remains a competitive, interpretable classical reference
[@assimakopoulos2000theta], while seasonal naive and drift forecasts are indispensable checks
against complexity. DeepAR popularized global probabilistic autoregression [@salinas2020deepar],
N-BEATS used basis expansions [@oreshkin2020nbeats], and temporal fusion transformers combined
attention with multi-horizon interpretability [@lim2021tft]. Transformer variants subsequently
introduced sparse attention, frequency decomposition, and two-dimensional temporal views
[@zhou2021informer; @zhou2022fedformer; @wu2023timesnet].

The strength of simple linear models on common long-horizon benchmarks prompted a useful
reassessment of whether attention was the source of progress [@zeng2023dlinear]. PatchTST addressed
several weaknesses by representing local subsequences as patch tokens and sharing a
channel-independent transformer [@nie2023patchtst]. Sv1 executes the official Hugging Face
`PatchTSTForPrediction` architecture as a global univariate model within each dataset. The
architecture is official; the sampling and training protocol are specific to this study and are
therefore not described as an exact reproduction of the original benchmark scripts.

Pretrained models shift the comparison again. Chronos casts scaled, quantized time-series values as
a language and pretrains transformer models on a large multi-domain corpus [@ansari2024chronos].
Chronos-Bolt is a more efficient direct multi-step variant that returns quantiles. Because a
retrieval method can look strong against an undertrained local model yet add little beyond a
pretrained foundation model, we execute official `amazon/chronos-bolt-small` as a zero-shot
reference. We do not fine-tune it, and we separately report its point accuracy and probabilistic
coverage. The trained PatchTST ensemble remains the primary comparator because the estimand is the
incremental value of a memory attached to a task-fitted backbone.

### 2.2 Retrieval-augmented time-series forecasting

Retrieval-augmented generation was developed most visibly for text, where a model conditions on
documents selected from an external corpus [@lewis2020rag]. A time-series analogue can retrieve
numeric segments rather than text, but the candidate continuation has an immediate operational
meaning: it is a possible future trajectory. RAFT retrieves historical contexts with patterns
similar to an input and supplies their future values to the forecaster [@han2025raft]. Its reported
results demonstrate that a simple retrieval bias can complement learned models across benchmark
datasets. TS-RAG instead uses pretrained time-series encoders and a learned adaptive mixture to
augment frozen foundation models [@ning2025tsrag]. Retrieval-Augmented Time-series Diffusion uses
retrieved references to guide denoising [@liu2024ratd]. These methods differ in representation and
integration, but each relies on the premise that similarity in the observable context predicts
useful information about the unobserved future.

PIR is particularly close to the selective component of Sv1. It identifies forecast instances
likely to be biased and revises them using local and global contextual information
[@liu2025pir]. Sv1 shares the model-agnostic post-hoc perspective but focuses on evaluation
identifiability: the revision value is a forward cross-fitted residual; utility estimation and
threshold selection have separate temporal roles; and no adapter is presented as an exact PIR
reproduction. The distinction is recorded in a baseline-fidelity table rather than left to naming.

The literature collectively establishes retrieval as a promising forecasting tool. It has devoted
less attention to the provenance of residual-valued memories. Direct candidate futures can be made
causal by requiring that they end before the query origin. Residuals add a model-training relation:
`residual = future - forecast(context)`. Whether the forecasting model fitted that pair changes the
meaning of the stored value. An exact leave-one-out retrieval rule prevents a query from selecting
itself but can still query a bank of in-sample residuals. This paper isolates those two mechanisms.

### 2.3 Selective prediction, calibration, and evaluation

Selective prediction permits a model to abstain when its expected risk is high
[@geifman2019selectivenet]. In forecasting revision, the base forecast serves as a natural abstention
action: declining retrieval does not mean declining to forecast, only declining to modify a known
reference. We predict *utility*—the decrease in squared error if a revision is applied—rather than
confidence in the target itself. Calibration remains an empirical question, as neural scores are
not automatically calibrated [@guo2017calibration]. We therefore show predicted-versus-realized
utility and full coverage–risk curves instead of reporting coverage alone.

Multi-dataset comparisons also require care. Rank-based summaries reduce sensitivity to arbitrary
dataset scales [@demsar2006comparisons], paired tests preserve within-case comparisons
[@wilcoxon1945], and multiplicity corrections limit opportunistic secondary claims [@holm1979].
Forecast origins within a series are dependent, making an iid window bootstrap inappropriate.
Our primary interval resamples datasets, series, and adjacent origin blocks
[@efron1979bootstrap; @lahiri2003resampling]. The confirmatory effect is tested against both zero
and a 1% smallest effect size of interest. This separates evidence that an effect exists from
evidence that it matters.

## 3. Problem formulation

For series (i), a query at origin (t) contains a context
(x_{i,t}=(y_{i,t-C},\ldots,y_{i,t-1})\) and target
(z_{i,t}=(y_{i,t},\ldots,y_{i,t+H-1})\). A backbone (f_\theta\) produces
(b_{i,t}=f_\theta(x_{i,t})\). A memory contains keys (\phi(x_j)\) and values (v_j\).
The retriever selects (k) candidates by cosine similarity and assigns softmax weights

\[
w_j(x)=\frac{\exp(s(\phi(x),\phi(x_j))/T)}
{\sum_{\ell\in N_k(x)}\exp(s(\phi(x),\phi(x_\ell))/T)}.
\]

For residual retrieval, (v_j=z_j-f_{\theta_j}(x_j)\) and the unconditional revised forecast is
(r(x)=b(x)+\sum_j w_j(x)v_j\). The subscript on (\theta_j) matters. An in-sample bank uses a
model fitted on the candidate window. The forward cross-fitted bank trains (f_{\theta_j}\) only on
origins earlier than candidate (j). Thus the safe residual approximates an error that could have
been observed during sequential deployment.

The router observes retrieval diagnostics (q(x)): nearest and weighted similarity, the nearest-to-
kth similarity gap, weighted residual dispersion, correction magnitude, query volatility, and
effective neighbour count. A ridge model fitted on the calibration-fit role predicts
(\hat u(x)\), where realized utility is

\[
u(x)=\operatorname{MSE}(z,b)-\operatorname{MSE}(z,r).
\]

The deployed forecast is (g(x)=r(x)\) if (\hat u(x)\ge\tau\), and (g(x)=b(x)\) otherwise. The
threshold (\tau\) is chosen on a distinct calibration-threshold role. This definition makes
coverage explicit and prevents the router from using a development or confirmation outcome.

The primary estimand is the dataset-macro mean of paired relative reductions in standardized
squared error,

\[
\Delta=\frac{1}{D}\sum_{d=1}^{D}\left[1-\frac{\sum_{i,t,h}(z_{dith}-g_{dith})^2}
{\sum_{i,t,h}(z_{dith}-b_{dith})^2}\right],
\]

with hierarchical resampling preserving dataset and series structure. Dataset-macro summaries are
also reported so large collections do not dominate method selection.

## 4. Materials and methods

### 4.1 Data, eligibility, and preprocessing

We use ten collections distributed through immutable Zenodo records associated with the Monash
Forecasting Archive [@godahewa2021monash]: M4 Hourly, M4 Weekly, Tourism Monthly, Bitcoin without
missing values, Pedestrian Counts, KDD Cup 2018 Air Quality without missing values, NN5 Daily
without missing values, FRED-MD, Rideshare without missing values, and Temperature/Rain without
missing values. Together they cover ten application domains, hourly through monthly frequencies,
contexts from 26 to 168 observations, horizons from 12 to 56, and seasonal periods from 7 to 365.
The archive was selected because it supplies heterogeneous related-series collections rather than
a single long multivariate panel.

Source archives are verified against repository MD5 values and locally recorded SHA-256 hashes.
We require at least 95% finite observations, enough pre-50% history for context plus target, and
enough width in every non-fit role for one horizon. Missing values are filled only forward after the
first finite observation. A fixed SHA-256 ranking of dataset name, series identifier, and selection
seed chooses at most 24 eligible series per collection; Bitcoin contributes all 18 eligible series.
This yields 234 series. The committed `series_selection.csv` preserves identifiers,
lengths, finite fractions, fit means, fit scales, and selection hashes.

Every series is standardized using the mean and population standard deviation available by the 50%
fit boundary. This prevents validation or test scale leakage and makes squared errors comparable
across differently scaled series. A context may cross an earlier role boundary because it contains
only observations available at its origin; every target lies wholly within one temporal role.

### 4.2 Temporal roles and windows

The series interval 0.15–0.50 trains the final backbone and supplies the in-sample memory. The
0.50–0.58 role selects training epochs. Calibration-fit (0.58–0.68) estimates router utility;
calibration-threshold (0.68–0.76) chooses one abstention threshold. Development (0.76–0.86)
selects neighbourhood size and temperature. Confirmation (0.86–1.00) remains sealed until all
protected inputs and the selected candidate are hashed.

Per series we place 24 evenly spaced fit origins, six early-stop origins, six calibration-fit
origins, four threshold origins, eight development origins, and eight confirmation origins where
the role width permits them. The deterministic placement avoids accidental stochastic variation in
test composition. Target overlap can remain for long horizons; uncertainty therefore resamples
moving origin blocks instead of treating windows as independent.

### 4.3 Forecasting references

PatchTST uses three hidden layers, width 96, feed-forward width 192, eight heads, patch length 16,
stride 8, dropout 0.1, AdamW at (5\times10^{-4}\), weight decay (10^{-4}\), batch size 256, and at
most 20 epochs with patience four. Patch lengths are reduced only when required by a short context.
Three registered seeds (3407, 9119, 17231) are ensembled. Training and validation curves, best
epochs, parameter counts, runtime, library version, device, and checkpoint hashes are recorded.

Chronos-Bolt-small is loaded from the official checkpoint. Its mean is used for point accuracy and
its 0.1, 0.5, and 0.9 quantiles for weighted quantile loss, 80% coverage, and normalized width.
AutoTheta is fitted independently to every query context with the registered seasonal period when
estimable. Last value, random walk with drift, and seasonal naive forecasts ensure that neural
complexity is earned. The fidelity matrix states exactly which code is official, adapted, or used
only as a literature comparator.

### 4.4 Memory construction and controls

Keys interpolate every context to 32 bins, locally center and scale it, and append bounded slope,
first-difference volatility, lag-one autocorrelation, and spectral concentration. The resulting
vector is L2-normalized. All operations are causal and scale free. Retrieval searches within the
same dataset, which permits cross-series analogues without conflating unrelated frequency and
domain semantics.

Forward cross-fitting uses a common late-fit holdout. Three registered-seed models train on
fractions 0.15–0.42 and their ensemble predicts up to six held-out windows per series in 0.42–0.50.
A fixed 16-epoch schedule avoids choosing cross-fit epochs from the held-out residual targets. The final ensemble
produces query forecasts but never retroactively replaces these out-of-fold values.

The registered grid is (k\in\{4,8,16\}\) and
(T\in\{0.05,0.10,0.20\}\). Each candidate receives its own calibration-fit utility model and
calibration-threshold cutoff. Thresholds are selected from 41 utility quantiles subject to at least
5% calibration coverage. Development chooses minimum macro MSE; candidates within 0.1% are ordered
by greater coverage, smaller (k), then larger temperature. The selected values were (k=8\),
(T=0.05\), threshold 0.71135, with development MSE change
0.64% and coverage 20.2%.

Controls isolate distinct explanations. `xfit_no_router` tests whether abstention matters.
`insample_router` changes residual provenance but retains retrieval and gating. A RAFT-style adapter
level-aligns retrieved raw futures; a PIR-style adapter fits one global revision multiplier on
calibration-fit data. Shuffled residuals preserve neighbour indices but break key–value association;
random neighbours break retrieval. An outcome oracle chooses the better of base and revision per
window and is an upper-bound diagnostic, never an eligible model. Finally, the self-inclusion
stress test queries a bank with its own cases. Exact inclusion adds the known residual back to the
forecast and must produce near-zero error; leave-one-out removes that direct route while leaving
residual provenance unchanged.

### 4.5 Metrics and statistical analysis

Standardized MSE is primary because it matches the residual-revision objective and avoids unit
dominance. MAE, MASE, RMSSE, sMAPE, WAPE, mean rank, performance profiles, horizon-wise error,
dataset wins, and probabilistic metrics provide robustness [@hyndman2006accuracy]. Ratios are
formed after summing paired squared errors, rather than averaging unstable per-window ratios.
MASE and RMSSE denominators use the query context and registered seasonal lag, falling back to lag
one when the seasonal period exceeds the context.

The primary interval uses 4,000 deterministic hierarchical moving-block draws. Each replicate
samples ten datasets with replacement, samples the observed number of series within each selected
dataset, and samples circular origin blocks of length two within series. Secondary methods receive
paired intervals against PatchTST; Wilcoxon dataset tests and Holm-adjusted p-values are descriptive
secondary evidence. No secondary result can rescue a failed primary criterion.

Success requires a 95% interval above zero, a point effect at least the fixed 1% SESOI, improvement
on at least six datasets, no significant MAE or MASE harm, and a verified integrity lock. Otherwise
the generated decision is statistically positive but negligible, inconclusive, harmful, or mixed.

## 5. Results

### 5.1 Backbone adequacy and development selection

Figure 7 shows convergence across all final PatchTST models and Figure 8 quantifies seed
dispersion. The development leaderboard in Figure 9 includes official Chronos-Bolt and classical
references; Figure 10 prevents a single macro average from hiding poor datasets. This is important
for interpreting retrieval: the incremental effect is measured over a trained three-seed backbone,
not a deliberately weak point forecast.

Across the nine registered retrieval candidates, the chosen setting produced
0.64% development MSE change relative to PatchTST. Figure 13 displays the entire
grid and marks the deterministic choice, while Figures 19–21 show utility calibration and
coverage–risk behavior. The development result is not confirmatory: it exists to select one
operating point and to establish that the router uses observables available at forecast time.

### 5.2 Leakage and provenance audit

The exact self-inclusion control behaves as algebra predicts: adding the query's observed residual
to its own backbone forecast reconstructs the target up to numerical precision (Figure 17). This is
not an empirical win; it is a unit test for leakage sensitivity. Leave-one-out removes the exact
candidate, but Figure 18 shows whether a bank of in-sample residuals still differs systematically
from forward cross-fitted values. Figure 16 places this provenance comparison beside raw-future,
global-revision, and no-router ablations.

This audit changes what can be claimed. A large in-sample effect paired with a weak cross-fit effect
would support a methodological conclusion about evaluation optimism, not a deployment conclusion
about retrieval. Conversely, a cross-fit effect that survives shuffled and random controls would
be evidence that the key–value association contains forecast-relevant information. The released
window table permits both interpretations to be checked without reconstructing values from a plot.

### 5.3 Sealed confirmation

The locked candidate revised 19.4% of confirmation windows. Its paired MSE
reduction was 1.25%, with a 95% hierarchical interval [-0.31%, 4.08%] and bootstrap
probability of positive effect 0.934. It improved 5 of ten dataset
means. Relative MAE and MASE changes and intervals are shown in Figure 25; dataset-specific effects
are shown in Figure 23 and domain effects in Figure 26. The preregistered decision was
**inconclusive**.

This wording intentionally separates sign, uncertainty, and magnitude. The zero line asks whether
the average effect could be harmful; the 1% line asks whether it is large enough to matter under the
registered standard. Figure 24 shows the full bootstrap distribution against both thresholds.
Horizon-wise effects in Figure 27 identify whether any average gain is concentrated near the
forecast origin. Mechanically selected benefit and harm exemplars (Figures 28–29) prevent qualitative
examples from being hand-picked.

### 5.4 Controls and heterogeneity

The outcome oracle defines how much correct per-window selection could gain from the available
revision. The distance between the oracle and the learned router is therefore a selection gap, not
evidence for a deployable method. Shuffled and random controls test whether a generic residual
shrinkage or additional variance could explain performance. Raw-future and global-revision adapters
test whether the conclusion is specific to additive residual transfer.

Effects vary by dataset and horizon. Such heterogeneity is expected: motif recurrence, seasonal
stability, context length, and the residual structure of the backbone all determine whether a
nearby shape has a transferable error. We therefore resist describing the macro effect as universal.
The result concerns a distribution over ten registered collections, with the listed domain
effects—not all possible time series.

## 6. Discussion

### 6.1 What the experiment establishes

The strongest inference is procedural. A retrieval system can be evaluated without allowing its
memory, router, or model choice to absorb information from the final test interval. Forward
cross-fitting makes a residual value operationally meaningful: it is an error that could have been
observed after deploying a model trained on earlier data. A separate threshold role makes
abstention evaluation meaningful. A hash lock makes the claim inspectable rather than dependent on
an author's recollection of which run came first.

The substantive inference follows the generated decision. Here the registered classification is
**inconclusive**, based on an MSE effect of 1.25% ([-0.31%, 4.08%]), the 1% SESOI,
5 dataset wins, and robustness metrics. This classification should be preserved even
if another metric, dataset subset, or visually attractive example looks better. Exploration of such
patterns can motivate a future protocol but cannot change this confirmation.

### 6.2 Why provenance matters beyond exact leakage

Exact self-inclusion is easy to understand and often easy to prevent. Training-residual optimism is
subtler. Even when a query is not in the candidate list, values in an in-sample memory reflect how a
particular fitted model behaves on its training distribution. A retrieval rule may learn the
structure of that optimism. Leave-one-out retrieval solves an identity problem; it does not solve a
model-fitting problem. Forward cross-fitting addresses both at the cost of training additional
models and storing fewer candidate cases.

This point generalizes beyond forecasting. Any retrieval system whose values are model errors,
pseudo-labels, uncertainty estimates, or learned explanations has a provenance graph. The graph
must record which observations trained the value-producing model. Treating a memory as a static
dataset hides that dependency and can produce optimistic evaluation even when conventional target
leakage checks pass.

### 6.3 Selective revision as a safer interface

An unconditional retriever assumes every query has a relevant analogue and every analogue's value
is transferable. The router relaxes both assumptions. Similarity, neighbour agreement, correction
magnitude, and query volatility are observed before the target and can predict when retrieval is
risky. The resulting system always emits the backbone forecast; it selectively decides whether to
alter it. This makes coverage an essential part of performance. A small average gain at 5% coverage
has a different operational meaning from the same gain at 80% coverage.

The calibration plots also reveal limitations. Predicted utility need not be numerically calibrated
even when it ranks cases usefully. Distribution shift between calibration and confirmation can
change both coverage and risk. Future work could use conformal risk control, monotone calibration,
or a hierarchical router that shares information while allowing dataset-specific thresholds. Such
methods must preserve the temporal separation used here.

### 6.4 Relation to recent retrieval systems

Sv1 is not an exact reimplementation contest. RAFT, TS-RAG, retrieval-guided diffusion, and PIR use
different backbones, representations, and integration mechanisms. Reproducing all original training
pipelines on ten fresh collections would introduce a large hyperparameter and compute asymmetry.
We instead execute official reference backbones and use transparent equation-level adapters to ask
which part of retrieval matters under one controlled protocol. The fidelity table prevents these
adapters from inheriting the names or claims of their source papers.

This choice strengthens causal attribution but limits leaderboard claims. The paper can conclude
that residual provenance and abstention affect this retrieval design. It cannot conclude that Sv1
dominates every end-to-end RAFT or TS-RAG configuration. A multi-lab reproduction using authors'
exact pipelines would be a valuable next step.

### 6.5 Operational interpretation and deployment checklist

The selective interface makes the experiment closer to a deployment decision, but it does not make
deployment automatic. An operator first needs to define what constitutes an admissible memory. In a
live service, a candidate residual becomes available only after its full forecast horizon has been
observed and quality controlled. A 56-day NN5 residual, for example, cannot be inserted one day
after its origin without using an incomplete target. Memory timestamps should therefore record both
forecast origin and value-availability time. Corrections to historical source data require versioned
memory updates because an apparently fixed residual can change when observations are revised.

Second, the backbone and residual producer must be versioned together. If a deployed backbone is
retrained, residuals produced by the older model describe a different error process. Mixing them can
be reasonable, but only if model identity is exposed to the key or integration rule. The safest
default is to start a new bank for each material backbone version, retain the old bank for audit, and
wait until enough fully observed residuals accumulate before enabling revision. The in-sample bank
should never be substituted merely to avoid this cold start.

Third, coverage needs a service-level bound. The registered router chooses a statistical threshold,
not a business-risk threshold. A high similarity score does not guarantee that a retrieved analogue
comes from a safe operational regime. Domain constraints can require abstention for extreme context
values, sensor failures, market closures, policy interventions, or a retrieval correction larger
than a fixed fraction of the backbone forecast. Such rules should be specified before observing
deployment outcomes and reported as part of effective coverage.

Fourth, monitoring must keep the paired action visible. Every revised forecast should log the base
forecast, correction, router score, selected candidate identifiers, weights, memory/model versions,
and eventual target. This supports delayed estimates of base and revised risk without replaying an
unversioned index. Monitoring should show coverage and error by domain, series, horizon, similarity,
and correction magnitude. A rise in base error does not by itself justify more retrieval: it may
also signal that historical residuals are stale. A conservative rollback disables revision while
preserving the backbone.

Finally, any new threshold, encoder, memory source, or outcome-based safety rule creates a new
model-selection event. It should receive a fresh calibration interval and a future confirmation
period rather than reusing the present one. The Sv1 lock is therefore not merely a publication
device. It is a template for model governance: enumerate inputs, hash versions, separate tuning from
evaluation, and retain a machine-readable decision whose criteria were known before outcomes.

For human review, the operational record should also preserve the reason for every abstention and
the raw units behind standardized displays. Standardization is appropriate for cross-series
inference, but operators act on dollars, passengers, temperatures, or transaction counts. A
correction that appears moderate in standardized space can still be unacceptable in physical
units. Review dashboards should therefore pair normalized error with domain-scale error, expose
candidate trajectories rather than only identifiers, and distinguish a low router score from a
hard policy rejection. These records make incident analysis possible without granting the router a
post-hoc explanation it did not use when forecasting.

## 7. Limitations and threats to validity

First, the benchmark is broad but not exhaustive. It contains univariate series selected from ten
Monash collections; multivariate covariates, irregular observation, hierarchical reconciliation,
and intervention-rich settings are absent. Selection of at most 24 series per collection keeps the
study computationally balanced but estimates collection performance with finite uncertainty.

Second, retrieval is within dataset and uses a fixed hand-engineered shape key. Learned encoders
could discover better neighbours, while cross-domain memories could help sparse collections.
Either change increases the need for separate retriever training and validation. The current result
should not be read as a ceiling on retrieval augmentation.

Third, forward cross-fitting approximates sequential deployment with three expanding folds. It
does not retrain a backbone at every origin. Fold identity and training seed are paired by design to
limit computation; more seeds per fold could separate their variance. Final query forecasts use a
three-seed ensemble, whereas each cross-fit residual is produced by one registered seed.

Fourth, Chronos-Bolt may have encountered related public datasets during pretraining. It is used as
a reference, not the primary paired baseline, and its corpus overlap cannot be independently
eliminated. PatchTST is trained only on the registered fit role and therefore provides the clearer
incremental estimand.

Fifth, hierarchical bootstrap choices are not unique. Block length two is fixed before
confirmation, but longer target overlap may create dependence beyond two adjacent origins. Dataset-
level effects, series-resampled intervals, and horizon profiles are reported so readers can assess
whether the primary interval is masking instability.

Sixth, a single confirmation period cannot establish invariance under future regime shifts. The
lock prevents adaptive reuse of this period, but a later temporal replication or external dataset
family is needed before high-stakes deployment. Finally, Q1 journal acceptance depends on venue
fit, reviewer judgment, writing, and independent scrutiny. A complete artifact does not guarantee
acceptance.

## 8. Reproducibility, ethics, and environmental considerations

The repository provides an MIT-licensed Python package, CLI, Make targets, Dockerfile, immutable
download records, data and selection manifests, upstream commit identities, model-fidelity labels,
configuration, tests, executed notebook, environment snapshot, confirmation lock, scalar tables,
and exactly thirty PNG/PDF figure pairs. Large redistributable archives and model checkpoints are
excluded from Git but regenerated by documented commands. `make verify` checks hashes, registered
figure count, notebook execution, confirmation lineage, and package build.

All datasets are public research archives and the study makes no individual-level decisions.
Bitcoin, rideshare, banking, and pedestrian collections can nonetheless encode sensitive economic
or behavioral patterns; release follows source terms and does not attempt re-identification.
Forecasts should not be deployed for financial trading, public safety, environmental alerts, or
resource allocation without domain-specific validation and uncertainty analysis.

Training multiple cross-fit models increases energy use. We mitigate this with a compact PatchTST,
fixed small grid, early stopping for final models, fixed epochs for folds, batched GPU inference,
and one confirmation run. Runtime metadata supports future comparison of accuracy and compute.

## 9. Conclusion

Retrieval augmentation changes a forecast using historical evidence, so the validity of that
evidence is part of the method. Sv1 makes the dependency explicit: candidate values are causal,
residuals are forward cross-fitted, utility and threshold fitting use separate roles, development
selects one registered setting, and confirmation is sealed by content hashes. Across ten fresh
collections, the selected method produced 1.25% confirmation MSE change with interval
[-0.31%, 4.08%], leading to the preregistered decision **inconclusive**.

Whether that label is beneficial, negligible, inconclusive, mixed, or harmful, the broader result
is the same: exact self-exclusion alone is not an adequate leakage policy for model-derived
retrieval values. Forecasting studies should report the provenance of candidate futures and
residuals, the coverage of selective revisions, and a practically meaningful confirmatory effect.
The released experiment supplies a reusable template for doing so.

## Data and code availability

Code and small evidence artifacts are contained in the Sv1 repository. Source archives are fetched
from the recorded Zenodo URLs and verified; generated checkpoints are reproducible but not committed.
The permanent repository URL and archival DOI must replace the placeholders in `CITATION.cff` after
the first public release.

## References

Bibliographic records are in `references.bib`.
