[0] Introduction
[1] CuraNex is an intelligent demand forecasting system for pharmaceutical distributors. It predicts how much of each medicine each retail pharmacy is likely to order in future weeks, using historical order data enriched with retailer characteristics, product attributes, calendar context, and health-related signals. The goal is to support better inventory planning, fewer stockouts (especially for critical medicines), and less waste from overstock and expiry.
[2] This document explains what the project delivers, how the methodology works (in depth), and how technology and business outcomes fit together. The current codebase implements an MVP (proof-of-concept) using synthetic data shaped like real distributor operations; production deployment would connect the same pipeline to live ERP or order-management data.
[4] 2. Background and problem statement
[5] 2.1 Context
[6] Pharmaceutical distribution in markets such as Sri Lanka depends heavily on imports, long lead times, and regulatory processes. Demand is not stable: it moves with seasons (e.g. dengue, respiratory illness), holidays, economic conditions, and differences between pharmacy types (hospital-attached vs standalone vs chain). At network scale—thousands of retailers and thousands of SKUs—manual averages or spreadsheet rules struggle to capture this complexity.
[7] 2.2 Business problems
[8] Reactive fulfillment — Orders are often filled only after pharmacies place them. Sudden demand spikes cause shortages and expensive emergency replenishment.
[9] Inventory inefficiency — Holding high safety stock everywhere ties up capital; yet fast movers can still stock out, while slow movers expire.
[10] 2.3 Technical challenge
[11] The core task is multi-retailer, multi-SKU time-series forecasting at weekly granularity:
[12] Scale — Very large numbers of retailer–SKU series.
[13] Sparsity — Many pairs order irregularly (intermittent demand).
[14] Heterogeneity — A large Colombo hospital pharmacy behaves differently from a small rural outlet.
[15] Cold start — New pharmacies or new products have little history.
[16] Shocks — Outbreaks and policy changes shift patterns (non-stationarity).
[17] The distributor’s position is advantageous: centralized order history across the network allows models to learn shared patterns that no single pharmacy could see alone.
[19] 3. Solution overview
[20] CuraNex is built as three logical layers:
[21] Data intelligence — Ingest and harmonize transactions, retailer and SKU masters, calendars, and contextual signals; build ML-ready features with strict time ordering so future information does not leak into past rows.
[22] Hybrid forecasting core — Train several model families (statistical baselines, gradient-boosted trees, and optional deep sequence models), then combine them with a stacking ensemble that learns which model to trust in which segment (e.g. by retailer tier and therapeutic category).
[23] Decision support — Turn predictions into recommendations (suggested order quantities, safety-stock style buffers) and expose them through a dashboard and REST API for planning workflows.
[24] A cold-start module clusters retailers with similar profiles so new or data-poor outlets can borrow strength from peers until enough history exists for personalized blending.
[26] 4. Methodology
[27] This section explains end-to-end how CuraNex turns raw orders into forecasts and recommendations. The methodology follows standard machine-learning practice for demand forecasting: clear problem formulation, clean temporal splits, diverse models, and rigorous evaluation—while staying aligned with how pharmaceutical planners actually work (weekly buckets, critical SKUs, regional context).
[28] 4.1 Problem formulation
[29] We frame demand forecasting as a supervised learning problem on a panel of observations. Each row corresponds to one retailer, one SKU, and one week (a fixed date or week index).
[30] Target variable — The quantity ordered in that week for that retailer–SKU pair (quantity_ordered). This is what the distributor needs to anticipate for replenishment and pre-staging.
[31] Inputs — For that same row, we use only information that would have been known at forecast time: past demand (lags, rolling statistics), static retailer and product attributes, calendar flags, and district-level indices (e.g. health/rainfall proxies in the synthetic pipeline).
[32] The business asks for visibility over multiple planning horizons (e.g. 1, 2, 4, 8, 13 weeks ahead). The system configuration encodes these horizons for planning and for neural forecasters; the gradient-boosted models in the MVP operate on the weekly panel—each row’s features summarize the past, and the model learns to predict the demand for that week. Extending to explicit multi-step targets per horizon is a natural production enhancement without changing the overall methodology.
[33] Why weekly? Pharmaceutical wholesale ordering is typically batched weekly or bi-weekly; weekly aggregation reduces noise, aligns with operational meetings, and matches how safety stock and reorder points are often reviewed.
[34] 4.2 Data sources and preprocessing
[35] Inputs used in the MVP pipeline:
[36] Preprocessing principles:
[37] Temporal ordering — All splits and features respect time: training rows only use past values to predict the current week’s demand.
[38] Aggregation — Data are aligned to a weekly grain before feature construction (in the synthetic generator, transactions are already weekly-style).
[39] Missing history — Early rows in each retailer–SKU series lack long lags; those rows are dropped or filled according to pipeline rules (e.g. drop until the shortest required lag exists; fill remaining feature gaps with neutral values where appropriate).
[40] Harmonization — Retailer and SKU identifiers are normalized (string IDs) so joins across transactions, masters, calendar, and health tables are consistent.
[41] In production, the same methodology applies to real ERP extracts, with additional steps for code changes, pack-size normalization, and data-quality monitoring.
[42] 4.3 Feature engineering
[43] Feature engineering translates raw history into signals a model can learn from. CuraNex uses four complementary families of features:
[44] A. Temporal and momentum (per retailer–SKU)
[45] Lags — Demand 1, 2, 3, 4 weeks ago capture short-term momentum; 13 weeks captures quarterly pattern; 52 weeks captures same week last year (annual seasonality).
[46] Rolling statistics — Mean and standard deviation over the last 4 and 13 weeks (computed from shifted series so the current week’s target is not included) measure level and volatility.
[47] Order-pattern features — Days since last order, order frequency over a recent window, and growth of average demand over a six-month window capture reorder rhythm and trend.
[48] Seasonal index — A normalized index by week of year summarizes how high or low this series tends to be in a given calendar week compared to its overall average.
[49] Together, these features help both tree models and the dashboard’s simple heuristics distinguish steady chronic demand from spiky or seasonal demand.
[50] B. Retailer profile
[51] Static attributes (type, district, tier) segment the network. RFM-style aggregates (recency, frequency, monetary volume, breadth of SKUs ordered) summarize relationship strength. Encodings map categories to integers for tree algorithms.
[52] C. SKU metadata
[53] Therapeutic category, product group, dosage form group, price band, country of supply, schedule, shelf life, supplier lead time, and criticality (e.g. essential medicines) help the model generalize: antibiotics and insulin can be treated differently from low-risk OTC lines.
[54] D. External context
[55] Calendar flags (holidays, festivals, month-end) capture predictable demand shifts. District-level dengue, respiratory, and rainfall indices proxy epidemiological and seasonal pressure — especially relevant in tropical climates where vector-borne and seasonal illness drive pharmacy purchases.
[56] Temporal train / validation / test split
[57] The full panel is split by calendar time (e.g. first 70% of weeks for training, next 15% for validation, final 15% for test). This mimics real deployment: the model is never trained on future weeks when we evaluate it on the past. No random shuffling of rows across time is used for evaluation, which would inflate accuracy unrealistically.
[58] 4.4 Modeling strategy
[59] Different algorithms capture different aspects of demand. CuraNex uses a portfolio of models rather than a single method.
[60] Statistical baselines
[61] A seasonal naïve baseline (e.g. same week last year) sets a minimum bar: any ML model should beat simple seasonal logic on key metrics, or the extra complexity is hard to justify.
[62] Gradient-boosted trees (LightGBM and XGBoost)
[63] Tree ensembles excel at heterogeneous tabular data: interactions between tier, category, lags, and calendar flags are learned automatically. LightGBM handles categorical features efficiently; XGBoost adds a different inductive bias (e.g. regularization behavior), which improves ensemble diversity. Both can be tuned with Optuna using time-series–aware validation. Quantile versions (e.g. 10th, 50th, 90th percentiles) support uncertainty-aware planning: planners can target a median forecast or a higher percentile for critical SKUs.
[64] Deep sequence models (TFT and N-BEATS)
[65] Temporal Fusion Transformer (TFT) — Designed for forecasting with multiple inputs and long dependencies; attention helps focus on relevant past weeks and known future inputs (e.g. calendar). Suitable for volatile series where pure lags are insufficient.
[66] N-BEATS — Emphasizes intrinsic time-series structure (trend and seasonality) with less reliance on hand-built cross-sectional features; acts as a complementary “pure time-series” expert.
[67] Neural models are optional in the training pipeline (they need more compute and stable software versions); if training fails, the rest of the system still runs.
[68] Why several models?
[69] No single method wins on all retailer–SKU types: regular high-volume pairs often favor rich tabular models; irregular or seasonal series may benefit from sequence or decomposition approaches. The methodology explicitly embraces diversity before combination.
[70] 4.5 Ensemble: stacked generalization
[71] Instead of averaging predictions by hand, CuraNex uses stacking:
[72] Level 0 — Each base model produces predictions on the validation period (out-of-sample in time).
[73] Level 1 — A meta-learner (e.g. Ridge regression with non-negative weights, or a small LightGBM model) is trained to map those predictions to the actual demand, optionally augmented with segment features (retailer tier, therapeutic category encoded).
[74] The meta-learner learns when to trust LightGBM vs XGBoost vs TFT vs N-BEATS—for example, higher weight on tree models for dense, feature-rich segments and more weight on sequence models where temporal structure dominates. If a base model is missing or misaligned, the pipeline can degrade gracefully (e.g. fewer models in the stack or a simple average fallback).
[75] For uncertainty, quantile outputs from GBMs (and probabilistic heads where available in deep models) can be combined so planners see not only a point forecast but lower and upper bands for safety stock decisions.
[76] 4.6 Cold-start handling
[77] New retailers or SKUs have too few weeks of history for the full model to personalize reliably. The methodology addresses this in three steps:
[78] Similarity — Existing retailers are described by a weekly demand profile (demand by week-of-year, aggregated across SKUs) and by static attributes (type, tier, district).
[79] Clustering — K-means (and related profile features) groups retailers into clusters of similar behavior and context.
[80] Inheritance and blending — A cold-start outlet is assigned to a cluster and can inherit cluster-level behavior. As weeks of history accumulate (e.g. beyond 12 weeks), predictions blend from cluster-dominant toward individual models using a weight that increases with data volume (e.g. full personalization toward 24 weeks), matching the proposal’s progressive personalization idea.
[81] This reduces the “zero history” failure mode and stabilizes early recommendations.
[82] 4.7 Evaluation protocol
[83] Models are compared on the held-out test window using metrics that match both statistics and operations:
[84] wMAPE (volume-weighted MAPE) — Puts more weight on high-volume pairs, aligning error with business impact.
[85] MAPE, RMSE, MAE — Standard accuracy and penalty on large errors.
[86] Bias — Detects systematic over- or under-forecasting (inventory vs stockout risk).
[87] Stratified analysis — Accuracy broken down by tier, category, or order pattern reveals where the system is safe to deploy first.
[88] Diebold–Mariano-style tests (where implemented) help check whether ensemble gains are meaningful rather than noise. Business-facing KPIs (fill rate, implied inventory holding) can be derived from predictions versus actuals for stakeholder reporting.
[89] 4.8 From forecasts to actions
[90] Methodology does not stop at error metrics:
[91] Recommendations — Suggested order quantities combine recent demand with a safety buffer (e.g. based on variability); critical SKUs can be filtered for priority lists.
[92] Dashboard — Supports exploration by retailer, SKU, region, and category and surfaces anomaly-style deviations from typical patterns.
[93] API — Allows other systems (planning tools, warehouse systems) to request forecasts and recommendation lists programmatically.
[94] Together, this closes the loop from model output to operational decision.
[96] 5. System architecture (summary)
[97] Batch pipeline — Generates or loads data → builds features → trains models → saves pickles and evaluation CSVs.
[98] Inference pipeline — Loads artifacts and reproduces predictions on train/validation/test for monitoring or export.
[99] Serving — FastAPI loads models and tabular data; Streamlit reads featured data and evaluation outputs for visualization.
[100] Modularity allows swapping model implementations or adding new feature blocks without rewriting the entire stack.
[102] 6. Technology stack
[104] 7. Business impact and expected outcomes
[105] Quantitative targets (e.g. target wMAPE ranges) should be calibrated on real data after go-live; the MVP establishes methodology and measurement, not final contractual SLAs.
[107] 8. Current implementation status
