# Power BI report

## What's in the folder
| Path | What it is |
|---|---|
| `powerbi/Ecommerce Analytics.pbip` | Power BI Project file. Open this in Power BI Desktop |
| `Ecommerce Analytics.SemanticModel/` | Data model: 8 tables, 8 relationships, 31 DAX measures |
| `Ecommerce Analytics.Report/` | The 5 report pages |
| `data/*.parquet` | Star-schema data, written by `python run_pipeline.py` (or `python -m src.export_powerbi`) |
| `theme/EcommerceTheme.json` | Custom colour theme |
| `screenshots/` | PNG of each page and a PDF of the full report |

## Open and refresh
1. Install Power BI Desktop (2024 or newer).
2. Run `python run_pipeline.py` so `powerbi/data/` has fresh Parquet files.
3. Double-click `powerbi/Ecommerce Analytics.pbip`.
4. **Set the data folder.** The `DataFolder` parameter holds an absolute path
   (`E:\Claude\project1_ecommerce_analytics\powerbi\data\`). On another machine, go to
   *Transform data → Manage parameters*, set `DataFolder` to your own `powerbi\data\` path (keep the
   trailing `\`), then click *Close & Apply*.
5. Click **Home → Refresh**. The first refresh takes a minute or two.

## Save or share
- **One file:** *File → Save as → Power BI file (.pbix)*.
- **Online:** *Home → Publish* (needs a work or school account).
- **Portfolio:** use the images and PDF in `screenshots/`.

## Data model
`fact_order_lines` is the main fact table. It joins `dim_date`, `dim_customer`, `dim_product` and
`dim_geography`. `fact_web_funnel` and `fact_cohort` share the date and geography tables. `Funnel Stage` sets
the order of the funnel steps.

Key measures (31 in total):
- **Sales:** Revenue, Gross Profit, Gross Margin %, Orders, AOV, Discount %
- **Time:** Revenue PY, Revenue YoY %, Revenue 3M Avg, Revenue YTD
- **Customers & segments:** Active Customers, Repeat Customer Rate, Customer Share, Predicted CLV 12m
- **Cohorts:** Cohort Size, Retention %
- **Funnel & operations:** Web Sessions, Conversion Rate, Bounce Rate, Return Rate %, Cancellation Rate %
