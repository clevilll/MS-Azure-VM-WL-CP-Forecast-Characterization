System and GPU Specifications

```markdown

GPU Status

Below is the current status of the system's GPU:
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 550.54.15              Driver Version: 550.54.15      CUDA Version: 12.4     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  Tesla T4                       Off |   00000000:00:04.0 Off |                    0 |
| N/A   60C    P8             11W /   70W |       0MiB /  15360MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
                                                                                         
+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI        PID   Type   Process name                              GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|  No running processes found                                                             |
+-----------------------------------------------------------------------------------------+

```

---

### Model Descriptions

1. **Linear Regression**:
   - **Purpose**: A simple linear model for regression tasks.
   - **Key Parameters**:
     - `n_jobs=-1`: Uses all available CPU cores for parallel computation.

2. **Random Forest Regressor**:
   - **Purpose**: An ensemble model that uses multiple decision trees for regression.
   - **Key Parameters**:
     - `n_estimators=1`: Number of trees in the forest (set to 1 for simplicity).
     - `random_state=123`: Ensures reproducible results.
     - `verbose=0`: No output during training.

3. **XGBoost Regressor**:
   - **Purpose**: A powerful gradient boosting algorithm for regression.
   - **Key Parameters**:
     - `random_state=42`: Ensures reproducible results.
     - `n_jobs=-1`: Uses all available CPU cores for parallel computation.

4. **MLP Regressor (Multi-layer Perceptron)**:
   - **Purpose**: A neural network model for regression.
   - **Key Parameters**:
     - `hidden_layer_sizes=(50,)`: One hidden layer with 50 neurons.
     - `solver='lbfgs'`: Optimizer for weight optimization.
     - `max_iter=1500`: Maximum number of iterations for training.
     - `random_state=42`: Ensures reproducible results.
     - `verbose=False`: No output during training.

5. **Support Vector Regressor (SVR)**:
   - **Purpose**: A regression model based on Support Vector Machines (SVM).
   - **Key Parameters**: Uses default settings.

6. **CatBoost Regressor**:
   - **Purpose**: A gradient boosting algorithm optimized for categorical data.
   - **Key Parameters**:
     - `random_state=123`: Ensures reproducible results.
     - `verbose=False`: No output during training.

7. **LightGBM Regressor**:
   - **Purpose**: A fast and efficient gradient boosting framework.
   - **Key Parameters**:
     - `random_state=123`: Ensures reproducible results.
     - `verbose=0`: No output during training.

8. **Linear Forest Regressor**:
   - **Purpose**: An ensemble model that combines Ridge regression with a forest structure.
   - **Key Parameters**:
     - `base_estimator=Ridge(random_state=42)`: Uses Ridge regression as the base estimator.
     - `n_estimators=10`: Number of estimators in the forest.
     - `n_jobs=-1`: Uses all available CPU cores for parallel computation.
     - `max_features='sqrt'`: Number of features to consider for splitting.

---

