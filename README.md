# MS-Azure-VM-WL-CP-Forecast-Characterization

RQ0: what SOTA for this usecase?

📂literature Review list (Stete-Of-The-Art) notebook

📁Public dataset (V1, V2) + single VM
- 📁 EDA notebook
- 📁 Extracting long-term VM candidates using Polars notebook

📂 data sanitation Experiments
- 📂 imputations Ex
- Savitzky–golay Filter:
- paper:   

RQ1: pipeline for applying periodic patterns using period_detection algorithm

- journal paper:
- Pythonic library:

📁 notebooks foe RQ1
- 📁 Ex over single periodic VM
- 📁 Ex over VM candidates
- 📁 periodicality results aggregation notebook

RQ2: pipeline for applying ML-based Forecasters along Backtesting for Prediction Interval (PI) and focusing on Upper Bound (UP)

📁 notebooks for RQ2
- 📄experiment setups and dependencies 
- 📁BT Animation
- 📁 train BT-based over VM candidates
- 📁Ranking 
- Ranking with CPU resources 
- Ranking with GPU resources incl. RF with 10 trees
- 📁 final results with script
