# MS-Azure-VM-WL-CP-Forecast-Characterization

RQ0: what SOTA for this usecase?

📂[literature Review list (Stete-Of-The-Art) notebook](https://github.com/clevilll/MS-Azure-VM-WL-CP-Forecast-Characterization/tree/main/RQ0/literature%20Review%20list%20(Stete-Of-The-Art)%20notebook)


Used Azure cloud data:
| #Public datasets           | Total #VMs       | #VM candidates | %VM candidates |
|----------------------------|-----------------|----------------|---------------|
| **#1 [AzurePublicDatasetV1](https://github.com/Azure/AzurePublicDataset/blob/master/AzurePublicDatasetV1.md)** | 2,013,767 (~2M) | 103,230 (~103k) | 5.12%         |
| **#2 [AzurePublicDatasetV2](https://github.com/Azure/AzurePublicDataset/blob/master/AzurePublicDatasetV2.md)** | 2,695,548 (~2.6M) | 179,393 (~180k) | 6.65%         |

📁Public dataset (V1, V2) + single VM
- 📁 Read & Backup data
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



Contact: mehryar.majd@gmail.com


