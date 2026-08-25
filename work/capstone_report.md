# Capstone Report — Next-Month Search Impression Decline Risk Ranking

- **Author:** Azam Hussain
- **Lane:** Machine Learning — Content Risk Ranking / Decision Support
- **Repo:** https://github.com/azam-hussain-ml/Starter-Notebook-flyrank-ml-internship
- **Deployed paper:** https://azam-hussain-ml.github.io/Starter-Notebook-flyrank-ml-internship/
- **Date:** 25 August 2026

## 1. Problem framing

### Research question

Can search-performance information available in March 2026 be used to rank content items by their risk of a substantial decline in impressions during April 2026?

### Decision supported

A content team may manage more content than it can manually inspect each month. The practical decision is therefore which content items should enter a limited human-review queue first.

The unit of analysis is one anonymized client-content item. The model produces a risk score and ranking rather than an automatic content decision.

The intended human action is to review the highest-ranked items first and investigate possible issues such as low CTR relative to search position, relevance, freshness, search intent alignment, or competitive changes.

A false positive mainly costs reviewer time and may lead to unnecessary investigation. Because the model is not treated as an automatic editing system, a false positive should not directly trigger publishing, deletion, pruning, or rewriting.

Machine learning helps because it can rank a large eligible content population using multiple measured signals while keeping the final action under human control.

The primary evaluation metric is **Precision@20**, because review capacity is limited and the quality of the first 20 recommendations is more important than classifying every content item.

---

## 2. Data safety

The analysis uses the **FlyRank ML Internship warehouse dataset**, primarily the `fact_content_daily_performance` table.

### Analysis windows

- **March 1–31, 2026:** feature and decision window.
- **April 1–30, 2026:** future outcome window only.

The raw March slice contains approximately **9.84 million daily records**, covering **55 anonymized clients** and **331,437 content items**.

After joining March and April observations and applying the eligibility rules, the final analytical frame contains:

- **100,893 eligible content items**
- **43 anonymized clients**

### Model features

Only March information available at the decision point is used:

- `impressions_31d`
- `ctr_31d`
- `avg_position_31d`
- `active_gsc_days`

### Fields deliberately excluded

`client_hash_id` is used only to keep clients separated during grouped validation. It is **not** used as a predictive feature.

`content_hash_id` is retained only as an anonymized identifier for the review queue. It is **not** used as a predictive feature.

April performance is used only to construct the future label and is never supplied to the model as an input feature.

Private URLs, raw search queries, client names, and other client-identifying information are not included in the public research artifact.

GA4-derived metrics were not included in the final feature set because their data coverage was lower for the analysis window.

### Eligibility and exclusions

Content is excluded when:

- March impressions are below 100.
- March GSC observations are unavailable.
- April GSC observations required to construct the outcome are unavailable.
- Required March impression or position values are missing or invalid.
- The content item cannot be matched across the March and April windows.

The public report therefore describes an eligibility-filtered content population rather than every page in the original warehouse.

---

## 3. Baseline

A transparent rule-based score was built before evaluating the learned model.

The baseline uses interpretable search-performance conditions such as:

- CTR opportunity relative to search position.
- Higher impression volume.
- Search position within a practically reviewable range.

The rule creates a ranking score so that it can be evaluated with the same ranking metrics as the machine-learning model.

The baseline and Logistic Regression model are evaluated on **exactly the same held-out client-grouped test rows**.

### Baseline results

| Metric | Rule baseline |
|---|---:|
| Precision@20 | 0.700 |
| Precision@50 | 0.500 |

At Precision@20, the rule baseline correctly identified **14 of its top 20** ranked items as future declines.

The held-out decline base rate was **0.412**, so the baseline performs substantially above simply reflecting the overall outcome rate.

---

## 4. Model / analysis

The final learned model is **Logistic Regression**.

Logistic Regression was selected because:

- The target is binary.
- The model is relatively interpretable.
- It produces probabilities that can be used directly for ranking.
- It provides a simple learned comparison against the transparent baseline.

### Target definition

The target is `declined_next_month`.

An eligible content item is labeled as a future decline when:

**April impressions < 80% of March impressions**

This represents a decline of more than 20% between the March decision window and April outcome window.

### Exact model feature list

The model uses:

1. `impressions_31d`
2. `ctr_31d`
3. `avg_position_31d`
4. `active_gsc_days`

No April variables, client IDs, content IDs, private URLs, or raw search-query fields are included as predictive features.

The four numerical features are standardized before fitting Logistic Regression.

The model is used as a **ranking and decision-support tool**, not as an automatic action system.

---

## 5. Evaluation

### Main validation split

The primary evaluation uses an **80/20 grouped train-test split by client** with random seed **42**.

This means all content belonging to the same client stays entirely in either the training set or the test set.

The resulting split contains:

- **Training rows:** 85,702
- **Test rows:** 15,191
- **Training clients:** 34
- **Test clients:** 9
- **Client overlap:** 0

This design reduces the risk of evaluating the model on content from clients it has already seen during training.

### Main results

| Method | Precision@20 | Precision@50 |
|---|---:|---:|
| Held-out base rate | 0.412 | 0.412 |
| Rule baseline | 0.700 | 0.500 |
| Logistic Regression | **0.800** | **0.720** |

On the main grouped holdout:

- The baseline identified **14 of 20** future declines.
- Logistic Regression identified **16 of 20** future declines.

The learned model therefore produced a better ranking than the rule baseline on this held-out client group.

### Cross-client robustness

A stronger validation audit uses **5-fold GroupKFold by client**.

| Fold | Precision@20 |
|---|---:|
| 1 | 0.700 |
| 2 | 0.900 |
| 3 | 0.500 |
| 4 | 0.550 |
| 5 | 0.900 |
| **Mean** | **0.710** |

Client overlap is zero in every fold.

The mean Precision@20 of **0.710** is a more cautious estimate than the single held-out result of 0.800.

### Leakage challenge

Several safeguards were applied:

1. April information is used only to construct the future outcome.
2. Client and content identifiers are not predictive features.
3. Training and test clients do not overlap.
4. A separate leakage challenge removes `impressions_31d`, the feature most directly related to the outcome definition.

The leakage challenge still produced useful ranking performance, supporting the conclusion that the ranking signal is not explained only by one feature.

### Error analysis

The main held-out model achieved Precision@20 of 0.800, meaning **4 of the top 20 recommendations were false positives**.

This is important operationally: the ranking is useful for deciding what to inspect first, but it is not accurate enough to justify automatic content changes.

---

## 6. Interpretation

The main measured result is that the learned Logistic Regression ranking outperformed the transparent rule baseline on the primary held-out client group:

- Baseline Precision@20: **0.700**
- Logistic Regression Precision@20: **0.800**

However, performance was not uniform across unseen client groups. Five-fold validation ranged from **0.500 to 0.900**, with a mean of **0.710**.

This variation is an important negative result because it shows that the single 0.800 score should not be interpreted as guaranteed future performance.

The model therefore provides useful **directional ranking signal**, but the strength of that signal depends on the client group being evaluated.

The action layer converts the highest-risk items into transparent review reason codes. In the primary top-20 queue, the items were classified as `LOW_CTR_FOR_POSITION`, indicating that their CTR appeared weak relative to their observed search position under the review rules.

This reason code should not be interpreted as proof of the cause of decline. It is a practical review hypothesis that helps a human strategist decide what to inspect.

Freshness is handled similarly: it can be investigated during review, but the study does not establish that refreshing content causes better search performance.

---

## 7. Recommendation

The model output should be used as a prioritized human-review queue.

### Ranked action playbook

**1. Review the highest-risk content first.**

Use the learned ranking to allocate limited review capacity to content with the strongest measured next-month decline-risk signal.

**2. Investigate low CTR relative to search position.**

For content with reasonable search position but unusually weak CTR, inspect:

- Title
- Search snippet
- Search intent alignment
- Relevance

**3. Investigate high-visibility content at risk.**

For higher-impression pages, review:

- Freshness
- Relevance
- Competitive changes
- Recent search-performance patterns

These items may deserve earlier attention because more visibility is potentially at stake.

**4. Use a general diagnostic review when the available features do not support a specific explanation.**

Keep uncertain high-risk items on a manual watchlist rather than assuming a particular fix.

### Human-review requirement

Every recommendation requires human review before:

- Editing content
- Publishing changes
- Removing content
- Pruning content
- Making other production changes

### Confidence

Confidence is **moderate and directional**.

The model improves the primary held-out ranking over the rule baseline, but cross-client validation shows meaningful variation.

The system should therefore support editorial prioritization rather than replace editorial judgment.

---

## 8. Reproducibility

The complete experiment is stored in this repository:

https://github.com/azam-hussain-ml/Starter-Notebook-flyrank-ml-internship

The main capstone notebook is:

`work/notebooks/capstone.ipynb`

The deployed public paper is:

https://azam-hussain-ml.github.io/Starter-Notebook-flyrank-ml-internship/

The exact deployed URL is also stored in:

`submission/paper_url.txt`

### Primary reproducibility route

1. Clone the repository:

```bash
git clone https://github.com/azam-hussain-ml/Starter-Notebook-flyrank-ml-internship.git
cd Starter-Notebook-flyrank-ml-internship
