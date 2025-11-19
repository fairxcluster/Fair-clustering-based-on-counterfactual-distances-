UniFair: A Unified fair clustering approach based on separation
and compactness.

This repository provides implementations of multiple fairness-aware 
clustering algorithms for high-dimensional tabular data. These methods 
extend classical k-means by incorporating fairness constraints derived 
from demographic groups and counterfactual reasoning.

The repository includes:

1. Separation Fair k-Means
2. Social Fair k-Means
3. Separation–Social Fair k-Means (combined method)
4. Deep Social and Deep Separation Fair k-Means

Each method addresses fairness in clustering from a different 
perspective, and all are implemented in a modular and reproducible form.

--------------------------------------------------------------------
1. Separation Fair k-Means
--------------------------------------------------------------------

Separation Fair k-means enforces fairness by requiring that demographic 
groups remain equally well separated from cluster decision boundaries.

The method measures fairness using *counterfactual hyperplane distance*:
it computes, for each point, the squared signed distance to the 
mid-hyperplane between its closest two centroids. If the two demographic 
groups have significantly different average distances, the clustering 
is considered unfair.

The algorithm penalizes large differences between groups, ensuring that 
both subgroups are positioned at similar levels of separation from 
cluster boundaries. This reduces the risk of systematically placing one 
group near ambiguous regions of the space.

--------------------------------------------------------------------
2. Social Fair k-Means
--------------------------------------------------------------------

Social Fair k-means enforces fairness by encouraging each cluster to 
represent demographic groups proportionally to how they appear in the 
dataset.

The method computes how each demographic group is distributed *inside* 
every cluster and penalizes cases where one group is consistently placed 
farther from its centroid than the other. Intuitively, this prevents 
clusters from being biased toward one demographic group at the expense 
of another.

This method ensures that cluster centers faithfully represent both 
groups, promoting balanced and socially fair cluster assignments.

--------------------------------------------------------------------
3. Separation–Social Fair k-Means (Combined Method)
--------------------------------------------------------------------

This algorithm combines the two fairness notions above.

The goal is to simultaneously enforce:
- **Separation fairness** (equal distance from decision boundaries)
- **Social fairness** (equal representation within clusters)

The method optimizes a joint objective that integrates:
- k-means clustering quality,
- fairness based on boundary separation, and
- fairness based on demographic representation.

This leads to clusterings that are both structurally fair (in decision
geometry) and socially fair (in demographic distribution).

--------------------------------------------------------------------
4. Deep Fair Clustering (Deep Social & Deep Separation Fair k-Means)
--------------------------------------------------------------------

The deep versions extend the fairness models by incorporating a neural 
network encoder (autoencoder). The encoder learns a nonlinear latent 
representation of the data, and the fairness-aware clustering objective 
is applied inside the learned latent space.

These models allow:
- better handling of nonlinear cluster structures,
- improved robustness on high-dimensional datasets,
- an end-to-end differentiable fairness-aware clustering pipeline.

Both fairness mechanisms (separation fairness and social fairness) can 
be activated in the deep setting.

--------------------------------------------------------------------
5. Included Datasets
--------------------------------------------------------------------

The repository supports the following datasets:

- **Adult Income**
- **Credit Card Clients**
- **Student Performance**
- **Bank Marketing**

Each dataset includes a binary sensitive attribute (e.g., gender).  
The code automatically handles preprocessing and scaling.

--------------------------------------------------------------------
6. How to Run
--------------------------------------------------------------------

1. Choose a dataset when prompted (e.g., "adult", "credit", "student", "bank").
2. Select which fairness-aware k-means variant to run:
   - separation
   - social
   - separation-social
   - deep-social
   - deep-separation
3. The script trains the clustering model across multiple seeds and λ values.
4. Plots and metrics are automatically saved in the `results/` directory.

