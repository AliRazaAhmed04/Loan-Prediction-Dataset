# Credit Risk Prediction

##  Objective

Predict whether a loan application will be approved (`Loan_Status`) based on applicant and loan-related features using supervised learning methods — specifically **Logistic Regression** and **Decision Tree**.

##  Dataset Information

- **Source**: Kaggle Loan Prediction Dataset
- **File used**: `LOAN PREDICTION DATASET.csv`
- **Total Rows**: ~614 entries  
- **Target Variable**: `Loan_Status` (Y/N)

## Key Features:
- Gender, Married, Dependents  
- Education, Self_Employed  
- ApplicantIncome, CoapplicantIncome  
- LoanAmount, Loan_Amount_Term  
- Credit_History, Property_Area

---

##  Approach

## **Data Cleaning**
- Handled missing values using mode (for categorical) and median/mode (for numerical).
- Ensured no null values remain before modeling.

## **Feature Engineering**
- Created `TotalIncome = ApplicantIncome + CoapplicantIncome`
- Applied log transformation to `LoanAmount` and `TotalIncome` to reduce skew:
  - `LoanAmountLog = log1p(LoanAmount)`
  - `TotalIncomeLog = log1p(TotalIncome)`

## **Encoding**
- Used `LabelEncoder` for categorical columns:
  - Gender, Married, Education, Property_Area, etc.

## **Modeling**
- Split data using `train_test_split` (80/20)
- Trained **Logistic Regression** (primary model, achieved better accuracy) 
- Applied `StandardScaler` to scale numeric features (important for Logistic Regression)

## **Evaluation**
- Accuracy
- Confusion Matrix
- Classification Report (precision, recall, F1-score)


##  Results

Accuracy: **78.86**
Confusion Matrix:
 [[18 25]
 [ 1 79]]
Classification Report:
               precision    recall  f1-score   support

           0       0.95      0.42      0.58        43
           1       0.76      0.99      0.86        80

    accuracy                           0.79       123
   macro avg       0.85      0.70      0.72       123
weighted avg       0.83      0.79      0.76       123


##  Visualizations
- Histogram of LoanAmount and TotalIncome (original and log-transformed)
- Boxplots of Applicant Income and Loan Amount by Education

## Tools & Libraries Used

- Python (pandas, numpy)
- Visualization: seaborn, matplotlib
- ML: scikit-learn (LogisticRegression)

