## 🏆 MLOps Project: Bank Marketing Prediction API

### 📌 Problem Statement

Financial institutions often run marketing campaigns to promote term deposit products. The goal of this project is to **predict whether a customer will subscribe** to a term deposit using historical campaign data.

This project demonstrates an end-to-end MLOps workflow by building a machine learning pipeline, deploying it via **FastAPI**, **containerizing with Docker**, and provisioning infrastructure using **Terraform** and **Ansible** on **AWS EC2**.

---

### 📊 Dataset

- **Source:** UCI Machine Learning Repository – Bank Marketing Dataset  
- **File:** `bank-additional-full.csv`  
- **Samples:** 41,188  
- **Features:** 21  
- **Target:** `y` – whether the client subscribed to a term deposit (`yes` / `no`)

---

### ⚙️ Model Pipeline

#### 🔧 Preprocessing
- Label encoding for categorical variables

#### 🧠 Feature Engineering
- `contacted_before`: derived from `pdays`
- `age_group`: bucketized from `age`
- Removed multicollinearity (via correlation matrix + VIF)
- Feature selection prior to scaling
- Scaling with `RobustScaler`

#### 🧮 Modeling
- **VotingClassifier** ensemble with:
  - `XGBoostClassifier` (`use_label_encoder=False`, `eval_metric='logloss'`)
  - `CatBoostClassifier` (silent mode)
  - `RandomForestClassifier` (`n_estimators=100`)
- Evaluated using Stratified 5-Fold Cross-Validation
- Best model saved as `voting_model.pkl`

#### 🔢 Features Used

```python
['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month',
 'day_of_week', 'campaign', 'pdays', 'previous', 'poutcome', 'cons.conf.idx',
 'nr.employed', 'age_group']

---

### 🧪 Running Locally

  1. Build Docker Image
    docker build -t mlops-fastapi -f docker/Dockerfile .

  2. Run the API Container
    docker run -p 8000:8000 mlops-fastapi

  3. Test Prediction

    curl -X POST http://localhost:8000/predict \
    -H "Content-Type: application/json" \
    -d @api/test_payload.txt

  Or open http://localhost:8000/docs for Swagger UI.

---

### ☁️ Deploy to AWS EC2

1. Provision EC2 Instance with Terraform
    cd infra/terraform
    terraform init
    terraform apply

2. Deploy App with Ansible
    cd ../ansible
    ansible-playbook -i inventory playbook.yml

---

📨 Example Request Payload
{
  "job": "technician",
  "marital": "single",
  "education": "university.degree",
  "default": "no",
  "housing": "yes",
  "loan": "no",
  "contact": "cellular",
  "month": "may",
  "day_of_week": "mon",
  "campaign": 1,
  "pdays": -1,
  "previous": 0,
  "poutcome": "nonexistent",
  "cons_conf_idx": -36.4,
  "nr_employed": 5191.0,
  "age_group": "25-35"
}

---

### 🔍 Drift Detection (Future Scope)

Although drift detection is not implemented in this version, the project is structured for easy integration of drift detection techniques, such as:

Statistical monitoring of feature distribution (e.g., with EvidentlyAI)

Performance monitoring via accuracy/AUC tracking

Triggering retraining when significant drift is detected

#### Directory Structure

MLOPS_PROJECT/
├── api/              # FastAPI implementation
│   ├── api_main.py
│   ├── predict.py
│   └── test_payload.txt
├── docker/           # Dockerfile
│   └── Dockerfile
├── infra/            # Infrastructure as code
│   ├── terraform/
│   └── ansible/
├── ml_model/         # ML logic and training pipeline
│   ├── train_model.py
│   ├── preprocess.py
│   ├── main.py
│   ├── EDA.py
│   ├── utils.py
│   ├── bank-additional-full.csv
│   └── saved_models/
├── requirements.txt  # Dependencies


#### Maintainer

Project by: Berna YILMAZ

Contact: berna14y@gmail.com

### ✅ Summary

This project demonstrates a complete MLOps lifecycle: from preprocessing and modeling to containerized deployment on AWS infrastructure. It provides a fast, stateless prediction service and is extensible for further automation, CI/CD, and monitoring.
