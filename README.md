# Retail-Customer-Segmentation

Retail Customer Segmentation Using Machine Learning (KMeans + Behavioral Features)

## 1. Overview

This project performs customer segmentation using transaction data (downloaded Kaggle.com), combining:

- Basket behavior

- Product diversity

- Purchase frequency

- Recency (RFM-style)

- Temporal patterns (day-of-week + entropy)

Clustering is used to group customers into interpretable segments to support marketing, personalization, retention strategies, and product targeting.

## 2. Project structure

```
Retail-Customer-Segmentation/
├── data/
│   ├── raw/
│   │   └── products.csv
│   └── processed/
│       └── customer_features.csv
│
├── src/
│   ├── preprocess_transactions.py      # Build customer_features
│   └── customer_segmentation.py        # KMeans + cluster profiling + PCA
│
├── outputs/
│   ├── figures/
│   │   └── clusters_pca.png
│   └── tables/
│       └── cluster_profile.csv
│
├── requirements.txt
└── README.md
````
## 3. Data process
### 3.1. Data preprocessing

The raw dataset (products.csv) contains:

- CustomerID

- TransactionID

- Timestamp

- Products (list of purchased items in one basket)

Extracted features:

| Feature            | Description                                 |
| ------------------ | ------------------------------------------- |
| `frequency`        | Number of distinct transactions             |
| `avg_basket_size`  | Avg. number of products per basket          |
| `diversity`        | Number of unique products purchased         |
| `Recency`          | Days since last purchase                    |
| `dow_mode`         | Preferred day of purchase (0–6)             |
| `dow_share_*`      | Purchase distribution by day                |
| `weekend_share`    | % purchases occurring Sat–Sun               |
| `dow_entropy`      | Spread across weekdays (high = more random) |
| `dow_mode_cos/sin` | Circular encoding of preferred day          |


Run preprocessing: python src/preprocess_transactions.py

Output: data/processed/customer_features.csv

### 3.2. Customer Segmentation (KMeans)

Steps:

- Log-transform skewed variables (frequency, avg_basket_size, diversity)

- Standardize features

- Test K from 2 → 8 using silhouette score

- Fit final KMeans model

- Inverse-transform cluster centers to original business scale

Run segmentation: python src/customer_segmentation.py

Outputs: outputs/tables/cluster_profile.csv

### 3.3. PCA Visualization

PCA is used to project customers into 2D space, showing cluster separation and density.

## 4. Cluster Insights (K=2)
````
### Cluster 0 — High-Value Loyal Customers
- High purchase frequency
- Larger basket size
- High product diversity
- Low recency (recent purchases)
- Clear weekly buying pattern

**→ Strategy:** 
Business actions:

- Premium loyalty program

- Early access to promos

- Personalized bundles

- Retention-focused communication
````
````
### Cluster 1 — Low-Engagement Customers
- Low purchase frequency
- Small baskets
- Low diversity
- High recency (long time since last purchase)
- Irregular buying habits
**→ Strategy:**
Business actions:

- Onboarding push (email flows)

- Welcome offers

- Cross-sell recommendations

- Make first 60 days attractive
````
## 5. Skills Demonstrated

This project demonstrates full-stack Data Science for Retail:

✔ Data engineering

Complex preprocessing: explode product lists, temporal engineering

Aggregation of customer-level behavioral metrics

✔ Machine learning

KMeans clustering

Silhouette evaluation

PCA dimensionality reduction

✔ Analytics & business insight

Cluster interpretation

Segmentation for marketing/CRM

Behavior-based profiling

✔ Good engineering practices

Clean folder structure

Modular scripts

Reproducible pipeline

Prepared for productionization / dashboards

## 6. Requirements

````
pandas
numpy
scikit-learn
matplotlib
````
📬 Author

Linh Nguyen - Data Scientist

Retail Analytics / Customer Segmentation

GitHub: https://github.com/linhnguyenlethuy1611
