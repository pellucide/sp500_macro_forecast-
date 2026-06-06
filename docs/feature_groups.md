# Feature Groups — SSRF Model Pipeline

This document describes every feature that enters the model pipeline, how it is
organized into groups for supervised screening, and where in the code each stage
is defined.

---

## Overview

The **raw feature set** is a DataFrame with **151 columns** (`data/fred_cache/all_fred_data_enhanced.csv`):
- **126 raw FRED-MD indicators** fetched from the St. Louis Fed API
- **25 derived features** computed from the raw indicators

Of the 134 official FRED-MD series, **7 are discontinued** and silently skipped:
`HWI`, `HWIURATIO`, `CLAIMS`, `AMDMNO`, `CONSPI`, `COMPAPFF`, `MOVE`.

Three **alternative features** are merged in separately during `run_all_models_oos.py`:
`CAPE`, `PUT_CALL_RATIO`, `MARGIN_DEBT`.

### Code References

| Stage | File | Function / Line |
|-------|------|----------------|
| FRED data fetch | `fetch_fred_cache.py:221` | `fetch_fred_with_cache()` |
| Derived features | `fetch_fred_cache.py:319` | `compute_derived_features()` |
| Alternative features | `run_oos_real_data.py:206` | `load_alternative_features()` |
| Feature grouping | `run_oos_real_data.py:337` | `create_groups_from_data()` |
| Group-wise screening | `src/ssrf_model.py:143` | `GroupwiseScreen.fit_transform()` |
| Final model input | `src/ssrf_model.py:853` | `SSRFModel.fit()` |

---

## Group Definitions

All 151 raw FRED columns are distributed across **9 groups**. Within each group,
the screening stage (`GroupwiseScreen`) computes univariate t-statistics against
the target return and keeps features with `|t-stat| > 0.75`. If none pass, the
single best feature in the group is kept by default.

The groups are defined in `run_oos_real_data.py`, function
`create_groups_from_data()`, starting at line 337.

---

### OUTPUT & INCOME (18 features)

Real output, personal income, consumption, and industrial production.

```
Column              FRED ID         Description
────────────────────────────────────────────────────────────
RPI                 RPI             Real Personal Income
W875RX1             W875RX1         Real Personal Income ex Transfer Receipts
DPCERA3M086SBEA     DPCERA3M086SBEA Real Personal Consumption Expenditures
CMRMTSPL            CMRMTSPL        Real Manufacturing & Trade Sales
RETAIL              RETAIL          Retail & Food Services Sales
INDPRO              INDPRO          Industrial Production Index
IPFPNSS             IPFPNSS         IP: Final Products & Nonindustrial Supplies
IPFINAL             IPFINAL         IP: Final Products (Market Group)
IPCONGD             IPCONGD         IP: Consumer Goods
IPDCONGD            IPDCONGD        IP: Durable Consumer Goods
IPNCONGD            IPNCONGD        IP: Nondurable Consumer Goods
IPBUSEQ             IPBUSEQ         IP: Business Equipment
IPMAT               IPMAT           IP: Materials
IPDMAT              IPDMAT          IP: Durable Materials
IPNMAT              IPNMAT          IP: Nondurable Materials
IPMANSICS           IPMANSICS       IP: Manufacturing (SIC)
IPB51222S           IPB51222S       IP: Residential Utilities
IPFUELS             IPFUELS         IP: Fuels
```

Note: `CMRMTSPL` and `RETAIL` appear without the `x` suffix in the cached CSV
(the suffix is stripped during API fetch, `fetch_fred_cache.py:54-55`).

---

### LABOR (32 features)

Employment, unemployment, hours worked, earnings, and capacity utilization.

```
Column              FRED ID         Description
────────────────────────────────────────────────────────────
CUMFNS              CUMFNS          Capacity Utilization: Manufacturing
CLF16OV             CLF16OV         Civilian Labor Force
CE16OV              CE16OV          Civilian Employment
UNRATE              UNRATE          Unemployment Rate
UEMPMEAN            UEMPMEAN        Avg. Weeks Unemployed
UEMPLT5             UEMPLT5         <5 Weeks Unemployed
UEMP5TO14           UEMP5TO14       5-14 Weeks Unemployed
UEMP15OV            UEMP15OV        15+ Weeks Unemployed
UEMP15T26           UEMP15T26       15-26 Weeks Unemployed
UEMP27OV            UEMP27OV        27+ Weeks Unemployed
PAYEMS              PAYEMS          Total Nonfarm Employment
USGOOD              USGOOD          Goods-Producing Employment
CES1021000001       CES1021000001   Mining & Logging Employment
USCONS              USCONS          Construction Employment
MANEMP              MANEMP          Manufacturing Employment
DMANEMP             DMANEMP         Durable Goods Employment
NDMANEMP            NDMANEMP        Nondurable Goods Employment
SRVPRD              SRVPRD          Service-Providing Employment
USTPU               USTPU           Trade, Transport & Utilities Employment
USWTRADE            USWTRADE        Wholesale Trade Employment
USTRADE             USTRADE         Retail Trade Employment
USFIRE              USFIRE          Financial Activities Employment
USGOVT              USGOVT          Government Employment
CES0600000007       CES0600000007   Goods-Producing Employment (alt.)
AWOTMAN             AWOTMAN         Avg Weekly Overtime Hours: Manufacturing
AWHMAN              AWHMAN          Avg Weekly Hours: Manufacturing
CES0600000008       CES0600000008   Avg Hourly Earnings: Goods-Producing
CES2000000008       CES2000000008   Avg Hourly Earnings: Construction
CES3000000008       CES3000000008   Avg Hourly Earnings: Manufacturing
EMRATIO             EMRATIO         Employment-Population Ratio (extra)
IC4WSA              IC4WSA          Capacity Utilization: Total Index (extra)
UNRATE_CHANGE_12M   —               [derived] UNRATE 12-month change
```

`EMRATIO` and `IC4WSA` come from the "extra" series in
`fetch_fred_cache.py:204-205` — they are not in the official FRED-MD list but
are fetched anyway.

---

### HOUSING (10 features)

Housing starts and building permits, national and by region.

```
Column              FRED ID         Description
────────────────────────────────────────────────────────────
HOUST               HOUST           Housing Starts: Total
HOUSTNE             HOUSTNE         Housing Starts: Northeast
HOUSTMW             HOUSTMW         Housing Starts: Midwest
HOUSTS              HOUSTS          Housing Starts: South
HOUSTW              HOUSTW          Housing Starts: West
PERMIT              PERMIT          Building Permits: Total
PERMITNE            PERMITNE        Building Permits: Northeast
PERMITMW            PERMITMW        Building Permits: Midwest
PERMITS             PERMITS         Building Permits: South
PERMITW             PERMITW         Building Permits: West
```

---

### ORDERS & INVENTORIES (5 features)

New orders, unfilled orders, inventories, and the inventory/sales ratio.

```
Column              FRED ID         Description
────────────────────────────────────────────────────────────
ACOGNO              ACOGNO          New Orders for Consumer Goods
ANDENO              ANDENO          New Orders for Nondefense Capital Goods
AMDMUO              AMDMUO          Unfilled Orders for Durable Goods
BUSINV              BUSINV          Total Business Inventories
ISRATIO             ISRATIO         Inventory / Sales Ratio
```

Note: `ANDENO`, `AMDMUO`, `BUSINV`, `ISRATIO` appear without `x` suffix in the
CSV (suffix stripped during fetch, `fetch_fred_cache.py:115-119`).

---

### INFLATION (20 features)

Producer prices, consumer prices, PCE prices, and oil.

```
Column              FRED ID         Description
────────────────────────────────────────────────────────────
WPSFD49207          WPSFD49207      PPI: Finished Goods
WPSFD49502          WPSFD49502      PPI: Finished Consumer Goods
WPSID61             WPSID61         PPI: Intermediate Materials
WPSID62             WPSID62         PPI: Crude Materials
OILPRICE            OILPRICE        Crude Oil Price
PPICMM              PPICMM          PPI: Metals & Metal Products
CPIAUCSL            CPIAUCSL        CPI: All Items
CPIAPPSL            CPIAPPSL        CPI: Apparel
CPITRNSL            CPITRNSL        CPI: Transportation
CPIMEDSL            CPIMEDSL        CPI: Medical Care
CUSR0000SAC         CUSR0000SAC     CPI: Commodities
CUSR0000SAD         CUSR0000SAD     CPI: Durables
CUSR0000SAS         CUSR0000SAS     CPI: Services
CPIULFSL            CPIULFSL        CPI: All Items Less Food
CUSR0000SA0L2       CUSR0000SA0L2   CPI: All Items Less Shelter
CUSR0000SA0L5       CUSR0000SA0L5   CPI: All Items Less Medical Care
PCEPI               PCEPI           PCE Price Index
DDURRG3M086SBEA     DDURRG3M086SBEA PCE: Durable Goods
DNDGRG3M086SBEA     DNDGRG3M086SBEA PCE: Nondurable Goods
DSERRG3M086SBEA     DSERRG3M086SBEA PCE: Services
```

Note: `OILPRICE` appears without `x` suffix (`fetch_fred_cache.py:164`).

---

### INTEREST RATES (27 features)

Policy rates, treasury yields, corporate yields, spreads, and yield curve
derived features.

```
Column              FRED ID         Description
────────────────────────────────────────────────────────────
FEDFUNDS            FEDFUNDS        Effective Federal Funds Rate
CP3M                CP3M            3-Month AA Commercial Paper
TB3MS               TB3MS           3-Month Treasury Bill
TB6MS               TB6MS           6-Month Treasury Bill
GS1                 GS1             1-Year Treasury
GS5                 GS5             5-Year Treasury
GS10                GS10            10-Year Treasury
GS2                 GS2             2-Year Treasury (extra)
GS20                GS20            20-Year Treasury (extra)
GS30                GS30            30-Year Treasury (extra)
AAA                 AAA             AAA Corporate Bond Yield
BAA                 BAA             BAA Corporate Bond Yield
TB3SMFFM            TB3SMFFM        3M T-Bill − Fed Funds
TB6SMFFM            TB6SMFFM        6M T-Bill − Fed Funds
T1YFFM              T1YFFM          1Y T-Bill − Fed Funds
T5YFFM              T5YFFM          5Y T-Bill − Fed Funds
T10YFFM             T10YFFM         10Y T-Bill − Fed Funds
AAAFFM              AAAFFM          AAA − Fed Funds
BAAFFM              BAAFFM          BAA − Fed Funds
T10Y2YM             T10Y2YM         10Y−2Y Treasury Spread (extra)
TEDRATE             TEDRATE         TED Spread (extra)
YIELD_SLOPE_10Y3M   —               [derived] GS10 − TB3MS
YIELD_SLOPE_10Y2Y   —               [derived] GS10 − GS2
YIELD_SLOPE_2Y3M    —               [derived] GS2 − TB3MS
CREDIT_SPREAD_BAA   —               [derived] BAA − GS10
CREDIT_SPREAD_QUALITY —             [derived] BAA − AAA
REAL_10Y            —               [derived] GS10 − CPI 12m change
```

`GS2`, `GS20`, `GS30`, `T10Y2YM`, `TEDRATE` are extra series beyond official
FRED-MD (`fetch_fred_cache.py:197-201`).

---

### MONEY & CREDIT (27 features)

Money supply aggregates, monetary base, reserves, loans, and money supply
growth/momentum derived features.

```
Column              FRED ID         Description
────────────────────────────────────────────────────────────
M1SL                M1SL            M1 Money Supply
M2SL                M2SL            M2 Money Supply
M2REAL              M2REAL          Real M2 Money Supply
M3SL                M3SL            M3 Money Supply (extra)
BOGMBASE            BOGMBASE        Monetary Base
TOTRESNS            TOTRESNS        Total Reserves of Depository Institutions
NONBORRES           NONBORRES       Nonborrowed Reserves
BUSLOANS            BUSLOANS        Commercial & Industrial Loans
REALLN              REALLN          Real Estate Loans at Banks
NONREVSL            NONREVSL        Consumer Credit Outstanding
DTCOLNVHFNM         DTCOLNVHFNM     Consumer Motor Vehicle Loans
DTCTHFNM            DTCTHFNM        Total Consumer Loans and Leases
M1_GROWTH_12M       —               [derived] M1 12-month % change
M1_GROWTH_6M        —               [derived] M1 6-month % change
M1_GROWTH_3M        —               [derived] M1 3-month % change
M1_ACCEL            —               [derived] M1_GROWTH_3M − M1_GROWTH_12M
M2_GROWTH_12M       —               [derived] M2 12-month % change
M2_GROWTH_6M        —               [derived] M2 6-month % change
M2_GROWTH_3M        —               [derived] M2 3-month % change
M2_ACCEL            —               [derived] M2_GROWTH_3M − M2_GROWTH_12M
M3_GROWTH_12M       —               [derived] M3 12-month % change
M3_GROWTH_6M        —               [derived] M3 6-month % change
M3_GROWTH_3M        —               [derived] M3 3-month % change
M3_ACCEL            —               [derived] M3_GROWTH_3M − M3_GROWTH_12M
M1_M2_RATIO         —               [derived] M1 / M2
M2_M3_RATIO         —               [derived] M2 / M3
M1_VS_M3_GROWTH     —               [derived] M1_GROWTH_12M − M3_GROWTH_12M
```

`M3SL` is an extra series beyond official FRED-MD (`fetch_fred_cache.py:202`).

---

### EXCHANGE RATES (5 features)

Trade-weighted USD and bilateral rates.

```
Column              FRED ID         Description
────────────────────────────────────────────────────────────
TWEXAFEGSMTH        TWEXAFEGSMTH    Trade Weighted U.S. Dollar Index
EXSZUS              EXSZUS          Switzerland / U.S.
EXJPUS              EXJPUS          Japan / U.S.
EXUSUK              EXUSUK          U.K. / U.S.
EXCAUS              EXCAUS          Canada / U.S.
```

All appear without `x` suffix in the CSV (`fetch_fred_cache.py:153-157`).

---

### SENTIMENT (7 features)

Stock market, consumer sentiment, volatility, and sentiment-derived features.

```
Column              FRED ID         Description
────────────────────────────────────────────────────────────
SP500               SP500           S&P 500 Index
UMCSENT             UMCSENT         Consumer Sentiment Index
VIXCLS              VIXCLS          CBOE Volatility Index (VIX)
INVEST              INVEST          Investment in Securities (FRB)
SENTIMENT_REGIME    —               [derived] UMCSENT > 85 (binary)
VIX_REGIME_HIGH     —               [derived] VIX > 25 (binary)
VIX_REGIME_LOW      —               [derived] VIX < 15 (binary)
```

`SP500` is stored in the CSV under its FRED API series ID
(`fetch_fred_cache.py:182`). In the official FRED-MD the column is named
`S&P 500`.

`UMCSENT` and `VIXCLS` appear without `x` suffix (`fetch_fred_cache.py:187-188`).

---

## Alternative Features (Exuberance Group)

Three additional features from non-FRED sources are merged in
`run_all_models_oos.py:91-112` via `load_alternative_features()`:

| Feature | Source | File | Line |
|---------|--------|------|------|
| `CAPE` | Shiller (Yale) | `run_oos_real_data.py` | 45 |
| `PUT_CALL_RATIO` | CBOE | `run_oos_real_data.py` | 76 |
| `MARGIN_DEBT` | FINRA via FRED | `run_oos_real_data.py` | 130 |

These form the `exuberance` group (`run_oos_real_data.py:431`).

---

## Pipeline Flow

```
┌──────────────────────────────┐
│ all_fred_data_enhanced.csv   │  151 columns (126 raw + 25 derived)
│ (data/fred_cache/)           │
└──────────┬───────────────────┘
           │ load_fred_data()
           ▼
┌──────────────────────────────┐
│ Alternative features joined  │  + CAPE, PUT_CALL_RATIO, MARGIN_DEBT
│ (run_all_models_oos.py:91)   │
└──────────┬───────────────────┘
           │ create_groups_from_data()
           ▼
┌──────────────────────────────┐
│ 9 groups × 151+ features     │  GroupwiseScreen
│ (run_oos_real_data.py:337)   │  |t-stat| > 0.75 per group
└──────────┬───────────────────┘
           │ fit_transform(X, y, groups)
           ▼
┌──────────────────────────────┐
│ ~20-40 surviving features    │  PredictiveScaler + SupervisedFactorExtractor
│ (src/ssrf_model.py:897-912)  │  PCA → 10 factors
└──────────┬───────────────────┘
           │
           ▼
┌──────────────────────────────┐
│ factor_0..9 + interaction    │  Regime proxy interaction terms
│ terms + regime features      │  (src/ssrf_model.py:914-968)
└──────────┬───────────────────┘
           │
           ▼
┌──────────────────────────────┐
│ ~25-30 columns               │  Final ElasticNet fit
│                              │  X_final → target return
└──────────────────────────────┘
```

Original raw column names do not survive to the final model — they are
transformed into PCA factors, interaction terms, and regime features before the
ElasticNet estimator (`src/ssrf_model.py:975-984`).

---

## Discontinued / Unavailable Series

**7 series that failed to fetch** (no longer available from FRED API):

| FRED-MD Name | API ID | Reason |
|-------------|--------|--------|
| HWI | HWI | Help-Wanted Index (discontinued) |
| HWIURATIO | HWIURATIO | Help-Wanted/Unemployment Ratio |
| CLAIMSx | CLAIMS | Initial Jobless Claims (discontinued) |
| AMDMNOx | AMDMNO | New Orders for Durable Goods (discontinued) |
| CONSPI | CONSPI | Consumer Price Index (concluded) |
| COMPAPFFx | COMPAPFF | CP − Fed Funds Spread (discontinued) |
| MOVE | MOVE | MOVE Bond Volatility Index (discontinued) |

**2 series excluded from fetch** (not directly available as FRED API series):
`S&P div yield`, `S&P PE ratio`.

---

## With `exuberance` features, approximately 154 features enter the DataFrame

passed to `SSRFModel.fit(X, y)`.
