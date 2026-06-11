# Discovering User Behavior Types from Session Sequences: A GRU-Autoencoder Approach

### Dabin Lee
Presentation Link: [Watch on YouTube](https://youtu.be/ZBjhhUdjs1c)

## Introduction

As of 2024, approximately 23% of domestic smartphone users fall into the at-risk group for overdependence [6]. A common technical approach to this problem is to forcibly block specific apps, but such methods provoke evasion and lack sustainability, often leading users to ultimately delete the app altogether.

Dopamedi, an mobile application that help to reduce smartphone screen-time, starts from a different hypothesis. Instead of forcibly blocking, it keeps addictive apps locked by default while allowing users to unlock them at any time, focusing on interrupting continuous use and helping users gradually reduce their screen time through a personalization-based nudge mechanism. Through Dopamedi, users experience a flow in which, within a defined session, they allocate "charge" time—the period during which app locks are released—and "relax" time, which aids calming through white noise. Each user's behavioral tendencies are revealed within this pattern of voluntary choices.

The problem is that these users are by no means homogeneous. Some users adhere to their planned charge time to the end, some rely more on Relax, and some open apps primarily at night. This researcher aims to classify users by behavioral pattern and then connect this to retention, in order to define the behavioral characteristics of the groups that exhibit higher retention. Moreover, Dopamedi's session data contains no ground-truth labels for "what type this user is." Therefore, user types must not be predefined but rather discovered from behavioral data in an unsupervised manner.

This project compresses each user's session sequence using a GRU [1] Autoencoder to learn embeddings, then clusters these via K-means to automatically discover behavioral types. The key result lies not in performance metrics but in a single observation: although the model divided users based solely on behavioral sequences—with no knowledge whatsoever of their retention outcomes—the retention indicators attached post-hoc diverged distinctly across clusters. In other words, the types discovered from behavioral patterns alone were aligned with actual retention.

## 2. Task & Method

### 2.1 Problem Definition

The goal of this project is to categorize Dopamedi users according to their behavioral patterns. However, since no ground-truth labels for user types exist, this is not a problem of classification into predefined categories, but rather an **unsupervised clustering** problem of discovering latent structure from the data. The input is each user's session sequence, and the output is behaviorally distinct user clusters.

### 2.2 Input Structure

A single user has multiple sessions ordered chronologically. Each session is represented by several attributes such as charge/relax time allocation, achievement rate, and time of day (see 2.4), so a user is represented as a **two-dimensional sequence of (sessions × features)**.

### 2.3 Method

**GRU as a sequence encoder:** Since the input is a session sequence, we need a model that can capture order and patterns of change rather than treating each session independently. This project uses a **GRU** as the sequence encoder, as it can handle variable-length sequences and has fewer parameters than an LSTM, making it well-suited to small datasets.

**Autoencoder as a learning structure:** If the GRU determines "how to read the sequence," then "what to use as the learning signal" in the absence of labels is a separate problem. In an environment with no ground-truth labels (unsupervised) and a small selected sample of 135 users, the autoencoder learns to compress the input into a low-dimensional representation and then reconstruct it [2]—a simple and robust representation-learning structure that takes the input itself as the signal (self-supervised), without labels.

In summary, our model is an **autoencoder that uses a GRU as its encoder (GRU Autoencoder)**. The GRU is the component that reads the sequence, and the autoencoder is the structure that learns representations without labels.

**Pipeline:** The overall process is as follows: session data → construction of per-user sequences → GRU autoencoder training → embedding extraction via the encoder → K-means clustering → cluster profiling and interpretation. Once training is complete, the decoder is discarded and only the encoder is used to obtain each user's embedding.

### 2.4 Feature Design

| # | Feature | Meaning | Preprocessing / Range |
| --- | --- | --- | --- |
| 1 | `f_charge_sec` | Actual charge time | log1p → min-max (0–1) |
| 2 | `f_relax_sec` | Actual relax time | log1p → min-max (0–1) |
| 3 | `f_total_sec` | Total usage time (charge + relax) | log1p → min-max (0–1) |
| 4 | `f_charge_ratio` | Charge proportion (0 = relax only, 1 = charge only) | 0–1 |
| 5 | `f_achieve` | Combined charge + relax achievement rate | 0–1 |
| 6 | `f_planned_sec` | Planned time | log1p → min-max (0–1) |
| 7 | `f_hour_sin` | sin encoding of start hour | −1–1 |
| 8 | `f_hour_cos` | cos encoding of start hour | −1–1 |
| 9 | `f_is_night` | Whether nighttime (22–6h KST) | 0 / 1 |
| 10 | `f_is_weekend` | Whether weekend (Sat/Sun KST) | 0 / 1 |

Each session is represented by 10 features (total charge/relax time, charge proportion, charge/relax achievement rate, planned time, sin/cos of the time of day, nighttime indicator, and weekend indicator). Time-unit features were skewed in distribution, so they were log-transformed and then min-max normalized; the hour was encoded with sin/cos so that 23:00 and 00:00 are not treated as far apart.

One deliberate decision in the feature design was to **explicitly add ratio-type features (achievement rate, charge proportion)**. In principle, these can be derived from values already present in the input (e.g., actual charge time and planned charge time). However, a ratio is a **division relationship** between two values, and the basic operation of a neural network is the weighted sum (addition), so division is not directly representable. Considering further the strong nonlinearity by which the value changes sharply as the denominator approaches zero, leaving the model to approximate this on its own from the data is inefficient; especially with the small number of users in this dataset, that capacity is better spent elsewhere. Therefore, ratios that are important for behavioral interpretation were computed directly and provided as input.

The input sequences are zero-padded at the front to standardize their length to 50, and actual sessions make up only about 59.8% of the total. Computing the standard MSE loss including the padding cells causes two problems. First, padding cells (0) are trivially easy to reconstruct as 0, so the model expends its limited representational capacity on reconstructing padding rather than actual sessions, and the loss value drops in a way unrelated to actual reconstruction quality. Second, since the padding ratio differs across users, the denominator of the loss (the total number of cells) is not constant, so users with fewer sessions have their loss filled in with the trivial 0-reconstruction score, distorting comparisons.

To prevent this, this project uses a **masked MSE loss**. The squared error at each position is multiplied by a mask (actual session = 1, padding = 0) to remove the contribution of padding cells, and the loss is averaged by dividing **only by the number of actual session cells**, not the total number of cells. As a result, the model is evaluated solely on the reconstruction quality of actual sessions, and the learned embeddings are guided to preserve actual behavioral information.

## 3. Experiments

### 3.1 Data Preparation

The raw data was collected firsthand and consists of 10,100 sessions from 224 of the total users. A session is a record of how a user allocated charge and Relax time within a defined rest period, and sessions in which one of the two is 0 also exist.

The following cleaning procedure was applied: (1) excluding sessions with a planned time of 0; (2) min-clipping the actual time to the planned time to prevent runaway durations from sessions that the user left unattended without pressing stop (since the session's actual end time is measured when stop is pressed); (3) converting the start time to Korea Standard Time (KST, +9) to extract the time of day; and (4) excluding users with fewer than 5 sessions. In the end, 135 users remained as clustering targets.

Each user was converted into a sequence of (50 sessions × 10 features). If a user had more than 50 sessions, the most recent 50 were used; if fewer, the front was zero-padded, and the padding positions were recorded with a separate mask. The proportion of actual sessions is 59.8% of the total. The final input tensors are X: (135, 50, 10) and mask: (135, 50), with all feature values normalized to [−1, 1].

### 3.2 Model Configuration

The encoder is a single-layer GRU with a hidden dimension of 64, whose output is mapped to the embedding dimension by a linear layer. The decoder is configured to reconstruct the original sequence from the embedding. The loss uses the masked MSE described in 2.3. Training was performed in PyTorch on a MacBook M3 (MPS), and K-means from scikit-learn was used for clustering. For reproducibility, the seed was fixed at 42. The embedding dimension (8/16/32), teacher forcing (on/off) [5], and the number of clusters k (2–6) were set as experimental variables, and ablation was performed (3.4).

### 3.3 Training
<img width="690" height="390" alt="image" src="https://github.com/user-attachments/assets/5e5270bc-0532-4a02-b434-87cdd9af343b" />

The masked MSE loss decreased monotonically and then converged during training (final ≈ 0.105 with teacher forcing on, ≈ 0.170 with it off). The learned embeddings had a per-dimension standard deviation of about 0.27, and no representational collapse—in which all users cluster into a single point—was observed.

Meanwhile, the reconstruction loss and the clustering quality moved independently of each other. Here, clustering quality was measured by the **silhouette coefficient [3]**, which expresses how close each data point is to its own cluster versus how far it is from the neighboring cluster as a value between −1 and 1, where values closer to 1 indicate better-separated clusters. The experiments showed that a lower reconstruction loss did not lead to a higher silhouette (reconstructing well and embedding being well-separated are distinct matters). Accordingly, representation quality was evaluated not by the loss value but by the silhouette of the downstream clusters and by cluster profiling.

### 3.4 Ablation

For the six combinations of embedding dimension {8, 16, 32} × teacher forcing (TF) {on, off}, we measured the silhouette and the final reconstruction loss for each number of clusters k {2–6}.

| EMB | TF | loss | k=2 | k=3 | k=4 | k=5 | k=6 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 8 | on | 0.1075 | 0.446 | 0.351 | 0.265 | 0.283 | 0.252 |
| 8 | off | 0.1705 | 0.327 | 0.313 | 0.313 | 0.306 | 0.303 |
| 16 | on | 0.1070 | 0.294 | 0.257 | 0.276 | 0.249 | 0.259 |
| 16 | off | 0.1706 | 0.319 | 0.229 | 0.237 | 0.230 | 0.238 |
| 32 | on | 0.1061 | 0.519 | 0.317 | 0.329 | 0.322 | 0.294 |
| 32 | off | 0.1703 | 0.327 | 0.316 | 0.287 | 0.284 | 0.260 |

The reconstruction loss was divided almost solely by whether teacher forcing was used, nearly independent of dimension (on ≈ 0.106–0.108, off ≈ 0.170). The silhouette, on the other hand, had no single high-performing combination, and the highest value (EMB 32, TF on, k=2 = 0.519) did not necessarily signify the most meaningful clusters. The rationale for this model selection, which is not determined by quantitative metrics alone, is addressed in Chapter 4 together with cluster profiling.

## 4. Result Analysis

### 4.1 Embedding Dimension: 8D

By silhouette alone, an embedding dimension of 8 is not the best. In the table in 3.4, the highest silhouette is 0.519 at EMB 32, TF on, k=2, while the 8-dimensional values remain at a moderate level. However, this highest value did not necessarily signify the most meaningful clusters.

| Dimension | Cluster | n | charge_ratio | charge_ach |
| --- | --- | --- | --- | --- |
| **8D** | c0 | 46 | 0.97 | 0.85 |
|  | c2 | 32 | 0.96 | 0.84 |
|  | c1 | 45 | 0.88 | 0.72 |
|  | c3 | **12** | **0.59** | **0.39** |
| **32D** | c1 | 38 | 0.96 | 0.87 |
|  | c3 | 41 | 0.91 | 0.74 |
|  | c0 | 40 | 0.90 | 0.81 |
|  | c2 | 16 | 0.77 | 0.51 |

Comparing the two dimensions under the final condition (k=4, TF off), the silhouette shows little difference: 0.313 for 8D and 0.287 for 32D. The cluster profiles, however, differ clearly. In both 32 and 8 dimensions, clusters with similar charge behavior were divided by time of day, but 8 dimensions additionally separated out a group of users who fail to fill their charge to completion (charge_ratio 0.59, achievement rate 0.39) as a qualitatively distinct cluster. In 32 dimensions, the lowest charge_ratio only reached 0.77, so the separation along this behavioral axis was weak. We therefore chose 8 dimensions—which clearly separates an interpretable behavioral axis—rather than the dimension that maximizes silhouette.

### 4.2 Teacher Forcing: off

Teacher forcing is a training technique in which, when the decoder reconstructs the sequence, it receives the ground truth (original) of the immediately preceding cell as input (on). If the ground truth is fed in at every cell, the decoder can reconstruct while relying less on the embedding, leaving room for the embedding to "grow lazy" without sufficiently capturing behavioral information. Conversely, with it off, the decoder must rely on the embedding z to reconstruct, so the embedding is pressured to hold more information.

<img width="1389" height="590" alt="image" src="https://github.com/user-attachments/assets/a681b9a6-26cd-446c-b0d8-6acbe6b5dda7" />

The experimental results also supported off. Quantitatively, at k=4 the silhouette was 0.313 for off, higher than the 0.265 for on, and off stably maintained a level around 0.30 across the k=2–6 range, whereas on declined as k grew. Decisively, when the embeddings were projected into two dimensions with t-SNE [4] for visual comparison, the four clusters under off separated their regions more distinctly than under on. Under on, the boundary between the charge-only group and the charge-plus-relax group was visually blurred and the distinction was ambiguous, whereas under off this boundary was clear. Since all three axes—quantitative, qualitative, and visual—pointed to off, we chose to turn teacher forcing off.

### 4.3 Number of Clusters: k=4

The number of clusters k was not self-evidently determined by silhouette alone. With teacher forcing on, the silhouette was higher at k=3 than at k=4; with it off, it was flat across the k=2–6 range with little difference. In other words, the quantitative metric did not strongly indicate any particular k; rather, the optimal k changed depending on the teacher forcing setting.

The basis for choosing k=4 was not the silhouette but the reproducibility of the clusters. Dividing into k=4 produces a middle group that combines charge and relax as a single cluster, and this cluster was reproduced identically in both embeddings made by the two different training methods—teacher forcing on and off. A cluster that appears only under a particular training setting may be an incidental result, but a cluster that is reproduced independently across different representations is likely a structure that genuinely exists in the data. By contrast, k=3 failed to separate this combined middle group on its own and absorbed it into an adjacent cluster. Therefore, even though k=3 was higher on silhouette under some conditions, we made the final choice of k=4, which preserves the behavioral structure that genuinely exists.

With this, the final model is determined as embedding dimension 8, teacher forcing off, and 4 clusters. The next section profiles the four behavioral types discovered by this model.

### 4.4 Introducing Separated Achievement Rates as an Interpretive Tool

Before interpreting the clusters, it is worth noting how the achievement rate (time achieved / time planned) is handled. The input feature `f_achieve` is the achievement rate summing charge and relax. However, the combined achievement rate cannot distinguish *what* a user fails to fill—because failing to fill charge and failing to fill relax are blended into the same value. To interpret the clusters' behavior, we need separated charge and relax achievement rates that pull these two apart.

The separated achievement rates were therefore computed post-hoc from the raw data. The median across all users was 0.822 for charge and 0.562 for relax, showing that users tended to adhere to their charge plans better than their relax plans.

An important design decision was that these separated achievement rates were not fed as input to the clustering but used only for post-hoc interpretation. There are three reasons. First, given the cross-sectional structure of the data, many sessions plan only one of charge or relax, so splitting the achievement rate in two would create many missing values, which is disadvantageous for training on this small-sample dataset. Second, the purpose of this project is not to "divide" users per se but to "explain" the discovered clusters. If clusters that the model divided without any separated-achievement-rate information also differed distinctly in separated achievement rates after the fact, this becomes a stronger finding.

### 4.5 Four Behavioral Types

The final model divided users into four clusters. Profiling each cluster with post-hoc behavioral metrics yielded the following. The clusters were named solely by behavioral axes (charge, relax allocation, time of day).

| cluster | n | charge_ratio | charge_ach | relax_ach | avg_relax_sec | night_ratio | Label |
| --- | --- | --- | --- | --- | --- | --- | --- |
| c0 | 46 | 0.97 | 0.85 | 0.62 | 38 | 0.26 | Charge-concentrated, daytime |
| c2 | 32 | 0.96 | 0.84 | 0.68 | 40 | 0.48 | Charge-concentrated, nighttime |
| c1 | 45 | 0.88 | 0.72 | 0.60 | 151 | 0.18 | Charge + relax combined |
| c3 | 12 | 0.59 | 0.39 | 0.69 | 224 | 0.16 | Relax-oriented, charge-stopping |

These four types are organized along two behavioral axes. The first axis is whether usage is concentrated on charging. c0 and c2 spend nearly all their time on charging, with a charge proportion of 0.96–0.97; these two are almost identical in behavior and are divided only by time of day—c0 is daytime-centered (nighttime ratio 0.26), while c2 has a nighttime ratio of 0.48, opening the app at night nearly half the time. The second axis is an increase in relax allocation. c1's charge proportion drops to 0.88 and its average relax time rises sharply to 151 seconds, combining charge and relax; by c3, the charge proportion falls to 0.59 and the charge achievement rate is the lowest at 0.39, differentiating into a type that fails to fill its charge plan to completion.

In summary, the charge-only groups (c0/c2) are divided by time of day, while the flow of progressively failing to fill charge (c1→c3) is divided by an increase in relax allocation.

### 4.6 Key Finding: Behavioral Types Align with Retention

The clustering up to this point used no outcome metrics such as retention or recency at all. The model divided users based solely on behavioral sequences. Yet once the clusters were finalized and retention and recency were attached to each user after the fact, these metrics diverged distinctly across clusters.

Here, `retention_days` is the period between the first and last session (how long the user used the app), and `recency_days` is the period from the data extraction date to the last session, where a smaller value means active until more recently and a larger value means closer to churn.

| cluster | Label | retention avg. | recency avg. | recency med. | Analysis |
| --- | --- | --- | --- | --- | --- |
| c0 | Charge-concentrated, daytime | 19.86 | 19.66 | 13.9 | Longest retention, recently active |
| c2 | Charge-concentrated, nighttime | 19.91 | 33.40 | 37.4 | Same charge behavior as c0 but less active |
| c1 | Charge + relax combined | 15.67 | 29.19 | 26.5 | Intermediate |
| c3 | Relax-oriented, charge-stopping | 7.98 | 53.46 | 54.3 | Shortest retention, highest churn risk |

Two things can be observed from this alignment.

**Observation 1. Behavior that fills charge to completion appears together with high retention.** c0, which has the highest charge achievement rate (charge_ach 0.85), has the longest retention at 19.9 days and is active until recently, whereas c3, which fills charge the least (charge_ach 0.39), has the shortest retention at 8.0 days and a recency of 54.3 days, closest to churn. That said, this is a co-observation of behavior and retention and cannot be interpreted causally as "filling charge raises retention."

**Observation 2. Even with identical charge behavior, activity level diverges by time of day.** c0 and c2 are behaviorally near-identical charge-only groups with almost the same charge proportion and charge achievement rate. Even so, daytime-centered c0 had a median recency of 13.9 days, indicating it was active, whereas c2, with its high nighttime ratio, had a recency of 37.4 days, relatively less active. Even within the same charge behavior, nighttime usage was co-observed with lower activity.

The reason this alignment is meaningful is that retention and recency were not inputs to the clustering. Since the model divided users based on behavior alone, without knowing these outcome metrics, the fact that the clusters diverged in retention after the fact is not a tautology where input and output are the same, but a substantive finding that behavioral patterns are connected to actual retention.

## 5. Conclusion

This project applied representation learning to Dopamedi users' session sequences with a GRU autoencoder and then clustered them with K-means, discovering four behavioral types without labels: charge-concentrated groups (daytime/nighttime), a charge-plus-relax combined group, and a relax-oriented charge-stopping group. Although the model used no retention metrics, the discovered behavioral types aligned with post-hoc retention and recency.

### 5.1 Applications

The discovered types connect directly to product interventions. Since a high charge achievement rate was observed together with high retention, a nudge that raises the charge achievement rate—by guiding users to choose a moderate amount of charge time—can be made a priority candidate for experimentation. In addition, c3, which has the highest risk of stopping charge and churning, is a target for early intervention, and a strategy of sending intervention notifications to users whose charge stoppages accumulate can be considered. The fact that retention and recency aligned with the behavioral types despite not being inputs to the clustering suggests that such behavior-based segmentation can serve as a clue for retention management.

### 5.2 Limitations

The conclusions of this study should be interpreted within the following limitations. First, all associations between clusters and retention are correlational (co-observations), not causal. Second, since the data is a cross-section cut off at the extraction point (censoring), it is difficult to fully distinguish recent sign-ups from genuine churners by recency alone. Third, very short (about 1-second) sessions in the onboarding process may lower some users' achievement rates below their true values. Fourth, the input feature `f_achieve` is an achievement rate summing charge and relax, so the model could not distinguish the two during the training stage.

### 5.3 Future Work

We propose three directions for future work. First, verifying whether type discovery changes when the separated achievement rates (charge/relax), used only for post-hoc interpretation, are fed as clustering input. Second, comparing representation quality by applying autoencoder variants such as denoising, sparse, and contractive. Third, analyzing the common demographic characteristics of the four discovered types to inform the definition of marketing targets.

## References

[1] Cho, K., van Merriënboer, B., Gulcehre, C., Bahdanau, D., Bougares, F., Schwenk, H., & Bengio, Y. (2014). Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation. *Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP)*, 1724–1734. https://doi.org/10.3115/v1/D14-1179

[2] Hinton, G. E., & Salakhutdinov, R. R. (2006). Reducing the Dimensionality of Data with Neural Networks. *Science*, 313(5786), 504–507. https://doi.org/10.1126/science.1127647

[3] Rousseeuw, P. J. (1987). Silhouettes: A Graphical Aid to the Interpretation and Validation of Cluster Analysis. *Journal of Computational and Applied Mathematics*, 20(1), 53–65. https://doi.org/10.1016/0377-0427(87)90125-790125-7)

[4] van der Maaten, L., & Hinton, G. (2008). Visualizing Data using t-SNE. *Journal of Machine Learning Research*, 9(86), 2579–2605.

[5] Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press. http://www.deeplearningbook.org

[6] 과학기술정보통신부, 한국지능정보사회진흥원. (2025). *2024년 스마트폰 과의존 실태조사*. 과학기술정보통신부.
