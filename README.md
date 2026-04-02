# 🚗 Vehicle Sales Data Analysis

<div align="center">

![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-2.x-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Power BI](https://img.shields.io/badge/Power%20BI-Dashboard-F2C811?style=for-the-badge&logo=powerbi&logoColor=black)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-22C55E?style=for-the-badge)

**End-to-end data analysis project — from raw data to interactive dashboard**

[📓 View Notebook](#-notebook-structure) · [📊 Dashboard Preview](#-power-bi-dashboard) · [📋 Key Findings](#-key-findings) · [🚀 Getting Started](#-getting-started)

</div>

---

## 📌 Project Overview

A complete end-to-end data analysis of **534,177 real-world vehicle auction records** sourced from Kaggle. This project demonstrates the full analyst workflow:

```
Raw Data (558K rows)  →  Python Cleaning & EDA  →  DAX Measures  →  Power BI Dashboard
```

The analysis answers **9 structured business questions** across pricing, depreciation, regional performance, and model-level market dynamics — producing insights directly applicable to automotive auction strategy.

---

## 📁 Repository Structure

```
vehicle-sales-analysis/
│
├── 📓 Vehicle_Sales_EDA.ipynb          # Main analysis notebook (72 cells)
├── 📊 Dashboard_2.pbix                 # Power BI dashboard (3 pages, 46 visuals)
├── 📋 Vehicle_Sales_Project_Report.docx # Full project report
├── 📄 README.md                        # This file
│
├── data/
│   └── car_prices.csv                  # Source dataset (download from Kaggle)
│
└── exports/
    ├── vehicle_sales_clean.csv         # Cleaned dataset (Python output)
    ├── monthly_summary.csv             # Monthly aggregation
    ├── make_summary.csv                # Make-level KPIs
    └── state_summary.csv              # State-level revenue
```

> **Dataset:** Download `car_prices.csv` from [Kaggle — Vehicle Sales Data](https://www.kaggle.com/datasets/syedanwarafridi/vehicle-sales-data) and place it in the `data/` folder before running the notebook.

---

## 📊 Dataset

| Attribute | Value |
|-----------|-------|
| Source | Kaggle — syedanwarafridi/vehicle-sales-data |
| Raw Records | 558,837 |
| Cleaned Records | **534,177** (95.6% retention) |
| Features | 16 original + 6 engineered |
| Sale Period | Oct 2014 – Dec 2015 |
| Geography | 38 US states + Canadian provinces |
| Price Range | $1 – $35,100 |
| Makes | 40+ (Ford, Toyota, BMW, Mercedes, Nissan...) |

### Columns

| Column | Type | Description |
|--------|------|-------------|
| `year` | int | Vehicle model year |
| `make` / `model` | string | Manufacturer and model name |
| `body` | string | Body type (SUV, Sedan, Truck...) |
| `transmission` | string | Auto / Manual |
| `state` | string | US state abbreviation |
| `condition` | float | Condition score (1–49 scale) |
| `odometer` | float | Miles at time of sale |
| `mmr` | float | Manheim Market Report benchmark price ($) |
| `sellingprice` | float | Actual auction hammer price ($) |
| `saledate` | datetime | Full UTC datetime of sale |

---

## 🔑 Key Findings

### 1. 💰 Pricing Analysis

> **Q01 — Price Distribution**
> The selling price is moderately right-skewed (skewness = 0.526) with a **median of $11,900** and mean of $12,643. 82% of sales fall under $20,000. The KDE curve reveals a bimodal pattern — economy used vehicles (~$9K peak) and certified pre-owned (~$17K peak) represent two distinct buyer segments.

> **Q02 — Price vs MMR by Make**
> Most makes sell **below** MMR benchmark. Luxury brands lose most: Bentley (−$12,900), Plymouth (−$1,578), Porsche (−$1,175). Trucks beat market: Mazda tk (+$1,126), GMC truck (+$433), Buick (+$323).

> **Q03 — Above vs Below MMR**

| Metric | Value |
|--------|-------|
| Above MMR | 53.3% (avg premium **+$948**) |
| Below MMR | 46.7% (avg discount **−$1,140**) |
| Net per vehicle | ≈ −$27 |
| Asymmetry | Losses outweigh gains by **20%** |

---

### 2. 🚗 Vehicle Characteristics

> **Q05 — Mileage Depreciation Curve**

| Mileage | Avg Price | Drop |
|---------|-----------|------|
| 0 – 20K | $18,607 | Baseline |
| 20 – 50K | $16,844 | −9.5% |
| **50 – 100K** | **$11,561** | **−31.4% ⚠️ cliff** |
| 100 – 150K | $5,960 | −48.5% |
| 150 – 200K | $3,494 | −41.4% |
| 200K+ | $2,529 | −27.6% |

**Total depreciation 0–200K+: −86.4%** · Correlation r = −0.649

> **Q08 — Model Year vs Resale Value**
> Resale grows exponentially from $834 (1993) to $20,290 (2015) — a **+2,333% increase**. The 2012 model year is the sweet spot: highest volume (97,375 units) at a strong avg price ($15,709).

---

### 3. 🗺️ Regional Performance

| Rank | State | Revenue | Units | Avg Price |
|------|-------|---------|-------|-----------|
| 1 | FL | $1.02B | 79,404 | $12,861 |
| 2 | CA | $875M | 68,192 | $12,833 |
| 3 | PA | $754M | 50,933 | **$14,800** |
| 4 | TX | $541M | 43,961 | $12,312 |
| 8 | TN | $318M | 19,868 | **$16,026** |

> FL leads by volume. PA and TN are the only states ranking top 4 in **both** revenue and avg price.

---

### 4. 🔗 Correlation Analysis

| Feature | Pearson r | Spearman r | r² | Verdict |
|---------|-----------|------------|-----|---------|
| `mmr` | **+0.978** | +0.978 | 95.6% | ⚠️ Data leakage |
| `vehicle_age` | −0.655 | −0.675 | 42.9% | Best real predictor |
| `odometer` | −0.642 | −0.701 | 41.2% | Non-linear relationship |
| `condition` | +0.353 | +0.473 | 12.5% | Weak — needs encoding |

> **Note:** MMR is excluded from ML models as it constitutes data leakage (it IS the market price).

---

### 5. 🏆 Top 10 Models

| Rank | Model | Volume | Avg Price | vs MMR |
|------|-------|--------|-----------|--------|
| 1 | Nissan Altima | 19,188 | $11,425 | −$72 |
| 2 | Ford F-150 | 13,536 | **$17,905** | −$257 |
| 3 | Ford Fusion | 12,793 | $12,356 | −$212 |
| 4 | Toyota Camry | 12,418 | $11,194 | −$144 |
| 5 | Ford Escape | 11,707 | $13,992 | −$196 |

> All 10 best-sellers sell **below MMR**. Ford dominates with 4 models (48,319 combined units). Top 10 = only **19.7% of total revenue** — highly fragmented market.

---

## 📓 Notebook Structure

```
Vehicle_Sales_EDA.ipynb  (72 cells)
│
├── 📦 Load Libraries
├── 🔍 Load & Inspect          → shape, dtypes, duplicates, isnull()
├── 🧹 Handle Nulls            → mean/median/mode strategy per column
├── 🎯 Outlier Detection       → IQR + Z-score comparison
├── ⚙️  Feature Engineering    → vehicle_age, mileage_bucket, price_vs_market
├── 📈 Univariate Analysis     → distributions, value counts, describe()
├── 🔗 Bivariate Analysis      → scatter plots, correlation heatmap
│
├── 💰 Pricing Analysis
│   ├── Q01 — Selling Price Distribution
│   ├── Q02 — Price vs MMR by Make
│   └── Q03 — % Above vs Below MMR
│
├── 🚗 Vehicle Characteristics
│   ├── Q05 — Mileage Depreciation Curve
│   └── Q08 — Model Year vs Resale Value
│
├── 📅 Sales Trends
│   └── Q09 — Monthly & Quarterly Patterns
│
├── 🗺️  Regional Analysis
│   └── Q13 — Revenue & Avg Price by State
│
├── 🔗 Correlation & Multivariate
│   └── Q17 — Feature Correlation with Price
│
├── 🏆 Make & Model Deep Dive
│   └── Q21 — Top 10 Models by Volume
│
└── 📋 Final Report            → 10-section written summary
```

---

## 📊 Power BI Dashboard

Three-page interactive dashboard with dark executive theme, semantic colour coding, and 4 cross-page synced slicers.

### Page 1 — Overview
| Visual | Measure(s) |
|--------|-----------|
| 4 KPI Cards | Units Sold · Total Revenue · Avg Price · Avg vs MMR |
| Monthly Revenue Trend | Total Revenue + Units Sold (dual axis) |
| Top 5 Body Type Revenue | Donut chart |
| Quarterly Revenue | Combo chart |
| Top 10 Makes by Revenue | Bar (green→red gradient) |
| Revenue per Age | Colour-graded bar |

### Page 2 — Pricing & Depreciation
| Visual | Measure(s) |
|--------|-----------|
| 4 KPI Cards | Avg Mileage · Avg Price Diff % · Total Price Diff · % Above MMR |
| Top 10 Avg vs MMR by Make | Bar (all green = above market) |
| Odometer vs Price | Scatter with mileage bucket colours |
| Avg Price by Mileage Bucket | Green→red gradient + fleet avg line |
| Avg Price by Condition | Poor=red → Excellent=green |
| Price Efficiency by Make | Gradient bar with 1.0 reference line |

### Page 3 — Regional Analysis
| Visual | Measure(s) |
|--------|-----------|
| 2 KPI Cards | Total Revenue · Avg Selling Price |
| Top 15 States by Revenue | Green/amber/red gradient |
| Top 15 States by Avg Price | With fleet avg reference line |
| Revenue by State | Filled map |
| State Performance | Pivot table with conditional formatting |

### Slicers (synced across all pages)
`Sale Year` · `Make` · `Body Type` · `State`

---

## ⚙️ DAX Measures

<details>
<summary>Click to expand — 18 custom measures</summary>

```dax
-- Core KPIs
Total Revenue         = SUM(vehicle_sales_clean[sellingprice])
Avg Selling Price     = AVERAGE(vehicle_sales_clean[sellingprice])
Units Sold            = COUNT(vehicle_sales_clean[vin])
Avg MMR               = AVERAGE(vehicle_sales_clean[mmr])

-- MMR Deviation
Avg Price vs MMR      = [Avg Selling Price] - [Avg MMR]
Avg Price vs MMR %    = DIVIDE([Avg Selling Price] - [Avg MMR], [Avg MMR])
Price to MMR Ratio    = DIVIDE([Avg Selling Price], [Avg MMR])
Pricing Status        = IF([Avg Selling Price] > [Avg MMR], "Above Market", "Below Market")
% Above MMR           = DIVIDE(CALCULATE(COUNT(vehicle_sales_clean[vin]),
                          vehicle_sales_clean[above_mmr] = TRUE()), COUNT(...))
Total Price Difference = SUM(vehicle_sales_clean[price_vs_market])

-- Vehicle Attributes
Avg Vehicle Age       = AVERAGE(vehicle_sales_clean[vehicle_age])
Avg Mileage           = AVERAGE(vehicle_sales_clean[odometer])
Avg Price by Condition = AVERAGEX(VALUES(vehicle_sales_clean[cond_bucket]),
                          CALCULATE(AVERAGE(vehicle_sales_clean[sellingprice])))

-- Advanced / Recruiter-Level ⭐
Price Efficiency      = DIVIDE([Avg Selling Price], [Avg MMR])
Revenue per Age       = DIVIDE([Total Revenue], [Avg Vehicle Age])
Sales Contribution %  = DIVIDE([Total Revenue], CALCULATE([Total Revenue], ALL(...)))
Sales Std Dev         = STDEVX.P(vehicle_sales_clean, vehicle_sales_clean[sellingprice])
```

</details>

---

## 🚀 Getting Started

### Prerequisites
```bash
pip install pandas numpy matplotlib seaborn scipy jupyter
```

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/vehicle-sales-analysis.git
cd vehicle-sales-analysis
```

### 2. Download the dataset
Download `car_prices.csv` from [Kaggle](https://www.kaggle.com/datasets/syedanwarafridi/vehicle-sales-data) and place it in the `data/` folder.

### 3. Run the notebook
```bash
jupyter notebook Vehicle_Sales_EDA.ipynb
```

### 4. Export for Power BI
The notebook's final cells export cleaned CSVs for Power BI:
```python
df.to_csv('exports/vehicle_sales_clean.csv', index=False)
```

### 5. Open the dashboard
Open `Dashboard_2.pbix` in [Power BI Desktop](https://powerbi.microsoft.com/desktop/) (free) and refresh the data source to point to your exported CSV.

---

## 🧹 Data Cleaning Summary

| Column | Nulls | Strategy | Reason |
|--------|-------|----------|--------|
| `transmission` | 64,272 | Fill `"Unknown"` | 37% of data — dropping would gut the dataset |
| `body` / `make` / `model` | 10K–13K | Fill `"Unknown"` | Rows still have valid price/state data |
| `condition` | 11,506 | Fill **median** | Numeric, skewed — median is outlier-resistant |
| `odometer` | 83 | Fill **median by make** | Grouped median more realistic than global |
| `sellingprice` / `mmr` | 12 / 14 | **Drop rows** | Primary metric — cannot impute |
| `saledate` | 12 | **Drop rows** | No date = cannot place in time series |

**Outlier removal (IQR method):**
- `sellingprice`: 2,747 removed (0.5%)
- `odometer`: 9,502 removed (1.8%)
- Final dataset: **534,177 records**

---

## ⚠️ Known Limitations

| Limitation | Impact |
|------------|--------|
| Sale dates: Oct 2014 – Dec 2015 only | YoY volume comparison unreliable — use avg price growth (+20.4%) only |
| 9,797 "Unknown" make/model records | Filter before model-level analysis |
| Canadian provinces (ON, QC) present | Appear in state analysis; map requires province filter for correct US zoom |
| $1 minimum selling price | Data entry error — filter `sellingprice > 100` for regression |
| MMR r = 0.978 | Data leakage — exclude from ML feature set |

---

## 🔮 Potential Next Steps

- [ ] **Price Prediction Model** — XGBoost/Random Forest on vehicle_age, odometer, condition, make, body (exclude MMR)
- [ ] **Segmented Analysis** — Luxury vs Mainstream vs Truck tier depreciation curves
- [ ] **Canadian Province Deep Dive** — ON leads avg price at $16,663 vs $12,643 fleet avg
- [ ] **Statistical Significance Testing** — t-tests/ANOVA on state and body type price differences
- [ ] **Power BI Service** — Publish dashboard to web for live interactive sharing

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| **Python 3.x** | Primary analysis language |
| **pandas** | Data loading, cleaning, aggregation, feature engineering |
| **numpy** | Numerical operations |
| **matplotlib** | Chart construction, multi-panel figures |
| **seaborn** | Heatmaps, pairplots, distribution plots |
| **scipy.stats** | Linear regression (r, p, slope), Z-score outlier detection |
| **Power BI Desktop** | Interactive 3-page dashboard |
| **DAX** | 18 custom measures |
| **Jupyter Notebook** | 72-cell analysis with written conclusions per question |

---

## 📂 Project Deliverables

| File | Description |
|------|-------------|
| `Vehicle_Sales_EDA.ipynb` | Full EDA notebook — cleaning, analysis, 9 questions, written conclusions |
| `Dashboard_2.pbix` | Power BI dashboard — 3 pages, 46 visuals, 18 DAX measures |
| `Vehicle_Sales_Project_Report.docx` | 8-section professional report with all findings and tables |
| `README.md` | This file |

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

- Dataset: [Vehicle Sales Data](https://www.kaggle.com/datasets/syedanwarafridi/vehicle-sales-data) by Syed Anwar on Kaggle
- Analysis methodology inspired by real-world automotive auction market research

---

<div align="center">

**⭐ If this project helped you, please give it a star!**

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](www.linkedin.com/in/soumimukherjeeofficial)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/SoumiMukherjee22)

</div>
