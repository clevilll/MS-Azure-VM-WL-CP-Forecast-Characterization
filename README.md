# MS-Azure-VM-WL-CP-Forecast-Characterization
![img](https://i.imgur.com/1KsteJM.png)

RQ0: What SOTA for this use case?

📂[literature Review list (Stete-Of-The-Art) notebook](https://github.com/clevilll/MS-Azure-VM-WL-CP-Forecast-Characterization/tree/main/RQ0/literature%20Review%20list%20(Stete-Of-The-Art)%20notebook)

---

### **Dataset Characteristics**
- 📁[Public dataset (V1, V2) + single VM](https://github.com/clevilll/MS-Azure-VM-WL-CP-Forecast-Characterization/tree/main/RQ0/Public%20dataset%20(V1%2C%20V2)%20%2B%20single%20VM)

<center>

 public dataset | Year of Azure VM workload collection | Duration | Total #VMs | Time-resolution/epoch | Dataset size
---|:---:|:-----:|:---:|:---:|:---:
#1 [MS Azure cloud VM CPU Usage](https://github.com/amcs1729/Predicting-cloud-CPU-usage-on-Azure-data/blob/master/README.md)  | [2017](https://github.com/amcs1729/Predicting-cloud-CPU-usage-on-Azure-data?tab=readme-ov-file) | 30 consecutive days | 1 | 5-minute VM CPU utilization readings |  633KB (csv) [1 file]
#2 [AzurePublicDatasetV1](https://github.com/Azure/AzurePublicDataset/blob/master/AzurePublicDatasetV1.md) | [2017](https://github.com/Azure/AzurePublicDataset/tree/master) | maximum 30 consecutive days | 2,013,767 (~2M) | 5-minute VM CPU utilization readings (encrypted) |  117GB (78.5GB compressed) [128 files]
#3 [AzurePublicDatasetV2](https://github.com/Azure/AzurePublicDataset/blob/master/AzurePublicDatasetV2.md)  | [2019](https://github.com/Azure/AzurePublicDataset/tree/master) | maximum 30 consecutive days | 2,695,548 (~2.6M) | 5-minute VM CPU utilization readings (encrypted) |  235GB (156GB compressed) [198 files]

</center>

### **Used Azure cloud data:**

<center>
  
| #Public datasets           | Total #VMs       | #VM candidates | %VM candidates |
|----------------------------|-----------------|----------------|---------------|
| **#1 [AzurePublicDatasetV1](https://github.com/Azure/AzurePublicDataset/blob/master/AzurePublicDatasetV1.md)** | 2,013,767 (~2M) | 103,230 (~103k) | 5.12%         |
| **#2 [AzurePublicDatasetV2](https://github.com/Azure/AzurePublicDataset/blob/master/AzurePublicDatasetV2.md)** | 2,695,548 (~2.6M) | 179,393 (~180k) | 6.65%         |

</center>


---
- 📁 Visualizing Confidence and Prediction Intervals
  - Pythonic notebook: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Jpp2iaHROWzj5us5-GQvJTXy6HMgqR3c?usp=sharing)
  - R-based notebook: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/11hkJFPMpikYjw8Er_CA1FWLBxzw8Vd4I?usp=sharing)
- 📁 [Read & Backup data](https://github.com/clevilll/MS-Azure-VM-WL-CP-Forecast-Characterization/tree/main/RQ0/Public%20dataset%20(V1%2C%20V2)%20%2B%20single%20VM/%20Read%20%26%20Backup%20data)
- 📁 [EDA notebook](https://github.com/clevilll/MS-Azure-VM-WL-CP-Forecast-Characterization/tree/main/RQ0/Public%20dataset%20(V1%2C%20V2)%20%2B%20single%20VM/EDA%20notebook)
- 📁 [Extracting long-term VM candidates using Polars notebook](https://github.com/clevilll/MS-Azure-VM-WL-CP-Forecast-Characterization/tree/main/RQ0/Public%20dataset%20(V1%2C%20V2)%20%2B%20single%20VM/Extracting%20long-term%20VM%20candidates%20using%20Polars%20notebook)
---
### **Data Sanitation**

![img](https://i.imgur.com/7SoMy0a.png)

📂 [data sanitation Experiments](https://github.com/clevilll/MS-Azure-VM-WL-CP-Forecast-Characterization/tree/main/RQ0/data%20sanitation%20Experiments)
- 📂 [imputations Ex](https://github.com/clevilll/MS-Azure-VM-WL-CP-Forecast-Characterization/tree/main/RQ0/data%20sanitation%20Experiments/%20imputations%20Ex) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1-I1pi0inLlN74M-a9DGGsESsxYvm2IvS?usp=sharing)
- Savitzky–Golay Filter:
  - paper: [ARIMA-Based and Multiapplication Workload Prediction With Wavelet Decomposition and Savitzky–Golay Filter in Clouds](https://ieeexplore.ieee.org/document/10387464)  
---
RQ1: pipeline for applying periodic patterns using `period_detection` framework

- journal paper: [“On a method for detecting periods and repeating patterns in time series data with autocorrelation and function approximation”](https://www.sciencedirect.com/science/article/pii/S0031320323000560?via%3Dihub)
- Pythonic library: see [PyPi](https://pypi.org/project/period-detection/1.0.0/) using: `pip install Pyriod==0.2.6` 
- Github Repository: https://github.com/LauritzR/period-detection

📁 notebooks for RQ1
- 📁 Ex over single periodic VM
- 📁 Ex over VM candidates
- 📁 periodicality results aggregation notebook

RQ2: pipeline for applying ML-based Forecasters along Backtesting for Prediction Interval (PI) and focusing on Upper Bound (UP)
![img](https://i.imgur.com/hDPYhm0.png)
📁 notebooks for RQ2
- 📄experiment setups and dependencies 
- 📁BT Animation
- 📁 train BT-based over VM candidates
- 📁Ranking 
- Ranking with CPU resources 
- Ranking with GPU resources incl. RF with 10 trees
- 📁 final results with script

---
 RQ3: Characterizing VM Compute and Memory Utilization via Periodic Signal Analysis [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]([https://colab.research.google.com/drive/1-I1pi0inLlN74M-a9DGGsESsxYvm2IvS?usp=sharing)


Contact: mehryar.majd@gmail.com


