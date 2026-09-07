# Gross Margin Decomposition: A Hybrid SQL & SHAP Driver Analysis
## Problem Statement
Between 2024 and 2025, the business delivered strong top-line revenue growth, with revenue increasing from $16.52M to $19.00M (+15%). However, this growth did not translate proportionally into profitability: gross profit increased by only 1.7% (from $6.00M to $6.11M), and gross margin declined from 36.34% to 32.14%, a 420 basis point compression.

This divergence creates a fundamental profitability question:
> **Why did substantial revenue growth produce almost no corresponding growth in gross profit?**

These headline metrics reveal the scale of margin compression, but not its underlying causes. 
Understanding what absorbed the benefit of revenue growth and where those pressures originated, is therefore the central focus of this analysis.


## Analytical Questions
The analysis is guided by three questions:
1. **Financial Drivers - what changed?**
- What underlying economic factors explain the change in gross profit despite strong revenue growth?

2. **Product Attribution - Where did it change?**
- Which products and categories contributed most to the deterioration or improvement in profitability?

3. **Relationship Dynamics - How did the drivers behave?**
- Do key operating variables exhibit nonlinear relationships or critical tipping points that could inform commercial decision making?



## Analytical Approach
To answer these questions, the project uses a two-tiered analytical framework:

1. **Deterministic Financial Analysis:**
A SQL-based Price-Volume-Mix-Cost (PVMC) decomposition quantiifies the contribution of price, volume, product mix, and unit cost changes to explain the movement in gross profit underlying the margin compression.

2. **Machine Learning Analysis:**
An XGBoost regression model, interpreted using SHAP dependence analysis, tests whether changes in pricing, discounts, or unit costs exhibit nonlinear thresholds or critical tipping points.


## Performance Dashboard
![Gross Margin Dashboard](assets/1_Gross_Margin_Dashboard.png)
> Strong volume growth was largely offset by higher costs and adverse product mix, limiting gross profit growth.


## Analysis & Findings
### 1. Financial Baseline Analysis

SQL aggregation established the scale of margin compression.

```sql
SELECT
    EXTRACT(YEAR FROM transaction_date::DATE) AS year,
    ROUND(SUM(net_revenue)) AS total_revenue,
    ROUND(SUM(net_revenue - cogs)) AS gross_profit,
    ROUND(100 * (SUM(net_revenue - cogs) / SUM(net_revenue))) AS gross_margin
FROM
    sales_ledger
GROUP BY
    1
ORDER BY
    year;
```
| Metric | 2024 | 2025 |
|---|---:|---:|
| Revenue | $16.52M | $19.00M |
| Gross Profit | $6.00M | $6.11M |
| Gross Margin | 36.34% | 32.14% |

Revenue increased 15.0% from $16.52M to $19.00M, while gross profit grew by only 1.7%, from $6.00M to $6.11M.
Over the same period, gross margin declined from 36.34% to 32.14% representing a 420 basis point compression.

These movements prompted a deeper examination of the economic forces underlying the margin compression.

### 2. Price–Volume–Mix–Cost Decomposition

The PVMC decomposition isolates the year-over-year movement in gross profit across four economic drivers: price, volume, product mix, and unit cost.


```sql
SELECT
    SUM(r.q_25 * (r.p_25 - r.p_24)) AS price_effect,
    SUM((r.q_24 * (t.vol_factor - 1)) * r.m_24) AS volume_effect,
    SUM((r.q_25 - (r.q_24 * t.vol_factor)) * r.m_24) AS mix_effect,
    SUM(r.q_25 * (r.c_24 - r.c_25)) AS cost_effect
FROM sku_rates r
CROSS JOIN totals t;
```


| Driver | Gross Profit Impact | Key Insight |
|---|---:|---|
| Volume | +$936.6K | Strong demand; volume expanded substantially YoY. |
| Realized Price | +$135.1K | Modest realized price improvement offered minor support. |
| Mix | -$110.8K | Shift in customer purchases towards lower-margin SKUs. |
| Cost | -$859.1K | Unit cost inflation absorbed 91.7% of volume profit gains. |

Overall, margin compression was driven primarily by higher unit costs, with adverse product mix adding further pressure.

### 3. Product-Level Contribution & SKU Mechanics

Extending the PVMC bridge to individual SKUs revealed key operational patterns:
- Electronics (Concentration of Drag):
SKUs P011 (-$18.7K), P004 (−$18.4K), P001 (−$15.5K), P009 (−$15.3K) and P006 (−$12.4K) accounted for severe profit declines due to unrecovered supplier cost spikes.


- Small Appliances (Favourable Mix Drivers):
SKUs P021 (+$39.0K), P023 (+$34.9K), and P022 (+$24.3K) generated mix effects of +$41.9k, +$38.5k, and +$31.9k respectively, absorbing cost increases and boosting total profitability.

- The P027 Paradox (Unprofitable Volume):
P027 generated +$94.1k in volume gains but suffered a -$63.4k mix drag and -$52.2k cost inflation, leaving net gross profit -$9.1k lower than in 2024. 
This highlights that top-line volume growth without margin controls erodes enterprise value.


### 4. Margin Relationship & ML Analysis
While the PVMC decomposition established the accounting drivers, a machine learning layer was developed to answer a specific operating question:
>**Do key operating variables (pricing, costs, discounts) exhibit nonlinear relationships or critical tipping points with margin deterioration?**

### Feature Engineering & Delta Transformations
Transaction-level metrics were calculated relative to each product's 2024 baseline to isolate operational movements:

```sql
SELECT
    -- operating movements relative to 2024 product baseline
    100 * (
        tm.unit_cost
        / NULLIF(b.base_unit_cost, 0)
        - 1
    ) AS unit_cost_change_pct,
    100 * (
        tm.unit_price
        / NULLIF(b.base_unit_price, 0)
        - 1
    ) AS list_price_change_pct,
    100 * (
        tm.discount_rate
        - b.base_discount_rate
    ) AS discount_change_pp,
    -- ML target
    tm.gross_margin_pct
        - b.base_margin_pct
        AS margin_delta_pp
FROM transaction_metrics AS tm
INNER JOIN baseline_2024 AS b
    ON tm.product_id = b.product_id
WHERE tm.year = 2025
```

### Order-Level Train/Test Split
To prevent data leakage across multi-line transactions belonging to the same purchase, the train/test split was strictly executed at the order level:

```python
train_orders, test_orders = train_test_split(
    ml_data['order_id'].unique(),
    test_size=0.20,
    random_state=42
)

train_mask = ml_data['order_id'].isin(train_orders)
test_mask = ml_data['order_id'].isin(test_orders)

x_train = x.loc[train_mask]
x_test = x.loc[test_mask]

y_train = y.loc[train_mask]
y_test = y.loc[test_mask]
```

### Model Evaluation & Tuning
Linear Regression was used as a linear benchmark, while XGBoost was evaluated and tuned to capture potential nonlinear relationships.

| Model | Mean CV R² | Mean CV RMSE |
|---|---:|---:|
| Linear Regression | 0.9890 | 0.2948 |
| XGBoost | 0.9959 | 0.1790 |

Final tuned XGBoost performance on the held-out test: R² = 0.9986, RMSE = 0.1048

>**Interpretation note:** the near-perfect predictive performance is expected given the strong mathematical relationship between gross margin and its underlying price, discount and unit cost components.

### 5. SHAP Relational & Dependence Analysis
SHAP tree explainers were utilized to inspect feature rankings and evaluate relationship shapes:

```python
# Initialize tree explainer on tuned model
explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer(x_test_transformed)

# Inspect feature rankings
shap.summary_plot(
    shap_values.values, 
    x_test_transformed, 
    feature_names=feature_names
)

# Evaluate feature impact and probe for nonlinear thresholds
shap.dependence_plot(
    'discount_change_pp', 
    shap_values.values, 
    x_test_transformed, 
    feature_names=feature_names
)
```
![SHAP Summary Plot](assets/2_SHAP_summary_plot.png)

<p align="center">
  <img src="assets/3_unit_cost_change_pct.png" width="32%">
  <img src="assets/4_discount_change_pp.png" width="32%">
  <img src="assets/5_list_price_change_pct.png" width="32%">
</p>

### **Key Findings**

- Feature Importance: Feature impact was heavily concentrated in unit_cost_change_pct, discount_change_pp, and list_price_change_pct. Category and store-level effects were negligible

- Relationship Dynamics: SHAP dependence plots confirmed that relationships were predominantly monotonic and near-linear:  
     - Increasing unit costs drove proportional, continuous margin degradation.  
     - Expanding discounts caused steady margin decline without abrupt threshold "cliffs"
     - Increasing list prices were associated with improved margin outcomes.

- Core Takeaway: The evidence did not support claiming a universal nonlinear threshold across operating variables. Commercial governance should therefore rely on dynamic, SKU-level margin floors rather than arbitrary universal rules.

## Recommendations

1. Targeted Cost Recovery in Electronics: Renegotiate procurement terms, source alternative suppliers, or selectively pass cost increases through to customers for high-drag Electronics SKUs (P011, P004, P001, P009, P006).

2. Protect and Scale Small Appliances: Capitalize on strong demand and favorable mix shifts in SKUs P021, P023, and P022 by ensuring inventory availability and prioritizing commercial focus.

3. Prioritize Profitable Volume Growth: Protect top-line momentum while aligning sales growth with net profitability. Restructure commercial and supply terms for SKUs like P027 so that increasing volume builds, rather than erodes gross profit.

4. Implement SKU-Level Margin Floors: Replace business-wide discount thresholds with dynamic discount governance that preserves minimum gross margin requirements based on each product's unit cost.


## Strategic Takeaway

The business does not have a top-line growth problem, it has a growth-quality problem. 

Higher sales volume generated approximately +$937K in incremental gross profit, but unit cost inflation and adverse product mix shifts absorbed nearly all of that gain.  

Commercial strategy should pair top-line expansion with strict unit-economic controls, recovering supplier costs where deterioration is concentrated, scaling high-margin SKUs, and enforcing SKU-level margin controls.  

## Tools & Technologies

- **SQL (DuckDB):** Data transformation, financial metric construction, annual performance analysis, PVMC decomposition, and SKU-level contribution analysis
- **Python:** Data preparation, statistical analysis, machine learning, and visualization
- **Pandas & NumPy:** Data manipulation and feature preparation
- **Scikit-learn:** Preprocessing, model pipelines, train/test splitting, cross-validation, and hyperparameter tuning
- **XGBoost:** Nonlinear regression modelling
- **SHAP:** Model interpretation and nonlinear relationship analysis
- **Power BI:** Executive KPI reporting and PVMC visualization
