# 🌬️ Machine Learning on Wind Turbine Data

This project applies **machine learning techniques** using **FastAI** to analyze and predict wind turbine performance metrics such as **power output**, **nacelle position**, and **rotor speed** based on sensor data.

---

## 📘 Project Overview

The project uses real-world wind turbine sensor data to build a **regression model** that predicts operational parameters from continuous numerical features such as wind speed, wind direction, and blade angles.

**Key Goals:**
- Understand correlations between wind speed and generated power.
- Predict nacelle position and power output using tabular data.
- Build and train a neural network using **FastAI’s Tabular Learner**.

---

## 🧠 Machine Learning Workflow

### 1. Data Preparation
- Imported dataset: `Wind_Turbine_Data_ML.csv`
- Cleaned and inspected using `pandas`.
- Normalized and handled missing values using FastAI’s preprocessing tools:
  ```python
  procs = [Categorify, FillMissing, Normalize]
  ```

**Columns:**
```
['WT1 - Wind speed (m/s)',
 'WT1 - Wind speed, Standard deviation (m/s)',
 'WT1 - Wind speed, Minimum (m/s)',
 'WT1 - Wind speed, Maximum (m/s)',
 'WT1 - Wind direction (°)',
 'WT1 - Nacelle position (°)',
 'WT1 - Power (kW)',
 'WT1 - Rotor speed (RPM)',
 'WT1 - Generator RPM (RPM)',
 'WT1 - Blade angle (pitch position) (°)',
 'WT1 - Blade angle (pitch position) A (°)',
 'WT1 - Blade angle (pitch position) B (°)',
 'WT1 - Blade angle (pitch position) C (°)']
```

### 2. Data Loading
Data is loaded into FastAI’s `TabularDataLoaders`:
```python
dls = TabularDataLoaders.from_csv(
    'Wind_Turbine_Data_ML.csv',
    y_names=['WT1 - Power (kW)'],
    cont_names=['WT1 - Wind speed (m/s)', 'WT1 - Wind direction (°)', 'WT1 - Rotor speed (RPM)'],
    procs=[Categorify, FillMissing, Normalize],
    bs=64
)
```

### 3. Model Training
A **tabular learner** is created and trained:
```python
learn = tabular_learner(dls, metrics=rmse)
learn.fit_one_cycle(50, lr_max=0.5)
```

The training showed a consistent **decrease in loss** over multiple cycles, demonstrating effective learning on the dataset.

---

## 📊 Results

| Metric | Description | Result |
|--------|--------------|--------|
| `train_loss` | Training loss during epochs | ↓ over time |
| `valid_loss` | Validation loss during epochs | ↓ significantly |
| Model | FastAI Tabular Neural Net | ✅ Trained successfully |

---

## ⚙️ Requirements

```bash
pip install fastai pandas matplotlib
```

Or in Google Colab:
```python
from fastai.tabular.all import *
```

---

## 📁 File Structure

```
├── Wind_Turbine_Data_ML.csv
├── Maskininlärning_slutprojekt.ipynb
├── README.md
```

---

## 🚀 Future Work
- Optimize hyperparameters for better prediction accuracy.
- Introduce feature importance analysis.
- Add visualization of model predictions.
- Deploy trained model using FastAPI or Streamlit.

---

## 📜 License
This project is for educational purposes and is licensed under the **MIT License**.

---

## 👤 Author
**Machine Learning Wind Turbine Data Project**
Developed as part of a **Machine Learning final project** using Google Colab and FastAI.
