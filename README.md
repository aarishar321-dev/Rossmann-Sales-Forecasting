**# Rossmann Store Sales Forecasting**

\> An end-to-end time-series machine learning project that started as a forecasting exercise and turned into a real lesson in model selection, time-aware validation, feature engineering, and deployment engineering.

**## Project Overview**

This project uses the **\*\*Rossmann Store Sales\*\*** dataset to build a machine learning system that predicts daily sales for individual stores.

The goal was not simply to train a model and report a metric. I wanted to understand the complete workflow:

**\*\*data → features → forecasting strategy → model comparison → evaluation → interpretation → future forecasting → deployment\*\***

The final application is a Streamlit interface where a user can enter store and calendar information and receive an estimated daily sales value. The deployed application uses the lightweight XGBoost model and its matching preprocessing pipeline.

The final **deployment model is a forecast-ready XGBoost Regressor** with a holdout MAE of approximately **\*\*1603.74\*\***. It was selected specifically for deployment because the higher-performing Random Forest was far too large and memory-intensive for the target hosting environment.

**---**

**## Why This Project Matters**

At first glance, sales forecasting looks like a normal regression problem.

It is not quite that simple.

The biggest constraint is that when predicting the future, the future Sales value is unknown. That means features such as future Sales lags and rolling Sales statistics have to be treated carefully.

This project therefore separates:

\- features that are useful when historical Sales are available
\- features that are genuinely available at prediction time

That distinction became one of the most important lessons from the project.

**---**

**## What I Built**

The project contains:

\- Exploratory Data Analysis
\- Date and calendar feature engineering
\- Historical Sales lag and rolling features for model experimentation
\- Chronological train/validation/test strategy
\- Baseline comparison
\- Linear Regression
\- Random Forest Regression
\- XGBoost Regression
\- MAE, RMSE, R² and MAPE evaluation
\- Feature importance analysis
\- Prediction error analysis
\- Hyperparameter tuning experiments
\- Forecast-ready feature engineering
\- Official Rossmann test-set forecasting
\- Prediction distribution analysis
\- Saved model and preprocessing pipeline
\- Streamlit prediction application
\- Deployment and model-size investigation

**---**

**# Project Workflow**

\`\`\`text
Raw Rossmann Data
        |
        v
Data Exploration
        |
        v
Data Cleaning
        |
        v
Date / Calendar Features
        |
        v
Historical Feature Engineering
        |
        v
Time-Based Train/Test Split
        |
        +----------------------+
        |                      |
        v                      v
   Baseline Model        ML Model Comparison
                               |
                    +----------+----------+
                    |          |          |
                    v          v          v
              Linear RF     XGBoost
                    |
                    v
             Model Evaluation
                    |
                    v
           Feature Importance
                    |
                    v
             Error Analysis
                    |
                    v
          Forecast-Safe Features
                    |
                    v
        Forecast-Ready Random Forest
                    |
                    v
        Official Future Predictions
                    |
                    v
              Streamlit App
\`\`\`

**---**

**# Feature Engineering**

Calendar features were created from the \`Date\` column:

\- \`Year\`
\- \`Month\`
\- \`Day\`
\- \`Week\`
\- \`Quarter\`
\- \`DayOfWeek\`
\- \`IsWeekend\`

Additional historical Sales features were explored during model development:

\- \`Sales\_lag\_1\`
\- \`Sales\_lag\_7\`
\- \`Sales\_Rolling\_7\`
\- \`Sales\_Rolling\_30\`

These historical Sales features were useful for understanding how much recent Sales history could improve predictions.

However, they cannot be directly used when predicting genuinely unseen future dates unless the required previous Sales values are already available.

That led to the creation of a separate **\*\*forecast-ready feature set\*\***.

**## Forecast-ready features**

The final deployment-safe model uses only information that is available for the future prediction period:

\`\`\`text
Store
DayOfWeek
Open
Promo
StateHoliday
SchoolHoliday
Year
Month
Day
Week
Quarter
IsWeekend
\`\`\`

This was an important design decision because the final model should not depend on future Sales values.

**---**

**# Model Comparison**

Several models were evaluated during the project.

\| Model | Holdout MAE | Decision |
\|---|---:|---|
\| Baseline | 2846.78 | Reference |
\| Linear Regression | 996.20 | Not selected |
\| Original forecast-ready Random Forest | **\*\*916.97\*\*** | Best accuracy, but too large for deployment | |
\| Original RF with historical Sales features | 538.72 | Not directly future-safe |
\| XGBoost with historical Sales features | 548.38 | Not directly future-safe |
\| Tuned forecast-ready Random Forest | 1352.59 | Not selected |
\| Small deployment Random Forest | 1549.94 | Not selected |
\| Deployment XGBoost | **\*\*1603.74\*\*** | **\*\*Selected for deployment\*\*** |
\| HistGradientBoosting | 1641.74 | Not selected |

**### Final model decision**

The model with the lowest raw holdout MAE was not automatically selected.

The Random Forest using historical Sales features achieved a much lower MAE, but those Sales-history features are not available in the same way for the official future prediction period.

Therefore, the **\*\*forecast-ready Random Forest with MAE ≈ 916.97\*\*** was selected as the final forecasting model.

This was a deliberate trade-off between predictive performance and real forecasting usability.

**---**

**# Evaluation**

The main evaluation metrics used were:

**### MAE — Mean Absolute Error**

Measures the average absolute difference between actual and predicted Sales.

It was the main metric used for model comparison because it is easy to interpret in the original Sales scale.

**### RMSE — Root Mean Squared Error**

Penalizes larger errors more strongly than MAE.

**### R² — Coefficient of Determination**

Measures how much of the variance in the target is explained by the model.

**### MAPE — Mean Absolute Percentage Error**

Used as an additional relative-error measure, with zero-target observations handled separately.

**---**

**# Feature Importance**

Random Forest feature importance showed that the most influential features included:

1\. \`Open\`
2\. \`Sales\_Rolling\_30\`
3\. \`Promo\`
4\. \`Sales\_lag\_1\`
5\. \`Sales\_lag\_7\`
6\. \`Day\`
7\. \`Week\`
8\. \`Sales\_Rolling\_7\`
9\. \`DayOfWeek\`
10\. \`Month\`

The result was useful because it showed how strongly store operating status and recent Sales history can influence forecasting when those historical features are available.

For the final forecast-ready model, the feature set was restricted to variables that can actually be known for future dates.

**---**

**# Error Analysis**

Error analysis was used to inspect individual predictions instead of relying only on a single overall metric.

The analysis compared:

\- Actual Sales
\- Predicted Sales
\- Error
\- Absolute Error

This made it possible to see where the model was underpredicting or overpredicting and helped move the project from simply reporting a score toward understanding model behaviour.

**---**

**# Hyperparameter Tuning: What I Learned**

Hyperparameter tuning did not automatically make the model better.

An initial \`RandomizedSearchCV\` experiment produced a cross-validation MAE of approximately **\*\*1391.84\*\***, which was substantially worse than the **\*\*916.97\*\*** holdout MAE of the selected forecast-ready Random Forest.

More importantly, ordinary cross-validation is not the right validation strategy for a time-dependent forecasting problem because the chronological order of observations matters.

A later observation should not be allowed to influence the validation of an earlier period.

I therefore investigated chronological tuning using a separate tuning-training and later validation period.

The tuned configuration still did not improve performance on the untouched holdout, producing a final MAE of approximately **\*\*1352.59\*\***.

The lesson was simple:

\> **\*\*Hyperparameter tuning is not automatically an improvement. The validation strategy has to match the real problem.\*\***

**---**

**# The Deployment Challenge**

This was probably the most unexpected part of the project.

After selecting the forecast-ready Random Forest, I saved the trained model using Joblib.

The result was approximately:

\`\`\`text
Original model size: 6.42 GB
\`\`\`

That is far too large for a normal GitHub-based deployment workflow.

So instead of immediately accepting the problem, I treated it as another engineering experiment.

**### Experiment 1 — Smaller Random Forest**

A smaller Random Forest was trained using:

\`\`\`text
n\_estimators = 50
max\_depth = 20
\`\`\`

Result:

\`\`\`text
MAE ≈ 1549.94
\`\`\`

The model was smaller, but the accuracy loss was too large.

**### Experiment 2 — Deployment XGBoost**

A future-safe XGBoost model was tested.

Result:

\`\`\`text
MAE ≈ 1603.74
\`\`\`

Again, it was not competitive with the selected Random Forest.

**### Experiment 3 — HistGradientBoosting**

A compact histogram-based gradient boosting model was also tested.

Result:

\`\`\`text
MAE ≈ 1641.74
\`\`\`

It was also rejected.

**### Experiment 4 — Model Compression**

Instead of changing the model, I tried compressing the original Random Forest with Joblib.

The result was:

\`\`\`text
6.42 GB  →  1.17 GB
\`\`\`

That was a major reduction while preserving the same trained estimator.

However, attempting to reload the compressed model in the notebook caused the kernel to restart, showing that the compressed disk size did not eliminate the memory pressure involved in deserializing the model.

At this point, the lightweight XGBoost model became the more practical deployment choice.

This became another important lesson:

\> **\*\*A smaller file on disk does not necessarily mean a lightweight model at runtime.\*\***

**---**

**# Deployment Environment Challenge**

When the model was moved from the notebook environment to a local Streamlit environment, another issue appeared.

The saved preprocessing pipeline had been created with:

\`\`\`text
scikit-learn 1.6.1
\`\`\`

while the local environment initially used:

\`\`\`text
scikit-learn 1.7.2
\`\`\`

This produced an \`\_RemainderColsList\` unpickling error.

The issue was fixed by using the same scikit-learn version as the training environment:

\`\`\`text
scikit-learn==1.6.1
\`\`\`

The preprocessor and compressed Random Forest then loaded successfully.

This reinforced another practical ML lesson:

\> **\*\*Reproducible environments matter. A saved model is tied to the software environment that created it.\*\***

**---**

**# Streamlit Application**

The final model was integrated into a Streamlit application.

The application allows a user to enter:

\- Store ID
\- Forecast Date
\- Day of Week
\- Store status
\- Promotion status
\- State Holiday
\- School Holiday

The app then generates the same calendar features used during training, applies the saved preprocessing pipeline, and passes the processed data to the saved XGBoost model.

The application converts the user inputs into **12 forecast-safe input features**. The saved preprocessing pipeline expands the categorical `StateHoliday` feature, producing **15 processed features** for the XGBoost model.

The application then returns an estimated daily Sales value.

The application was successfully tested locally.

**---**

**# Project Structure**

\`\`\`text
Rossmann\_Sales\_Forecasting/
│
├── data/
│   ├── train.csv
│   └── test.csv
│
├── models/
│   ├── forecast\_preprocessor.pkl
│   └── forecast\_rf\_model\_compressed.pkl
│
├── notebook/
│   └── sales-forecasting.ipynb
│
├── app.py
├── requirements.txt
├── README.md
└── LICENSE
\`\`\`

**---**

**# Requirements**

The deployment environment uses pinned versions to keep the saved preprocessing pipeline compatible:

\`\`\`text
streamlit==1.61.1
pandas==2.3.3
numpy==2.2.6
joblib==1.5.3
scikit-learn==1.6.1
\`\`\`

XGBoost is required by the final Streamlit application because the deployed model is an XGBoost Regressor.

**---**

**# Running the Project Locally**

**## 1. Clone the repository**

\`\`\`bash
git clone \<YOUR-GITHUB-REPOSITORY-URL>
cd Rossmann\_Sales\_Forecasting
\`\`\`

**## 2. Create a virtual environment**

\`\`\`bash
python -m venv venv
\`\`\`

Activate it on Windows:

\`\`\`bash
venv\Scriptsctivate
\`\`\`

**## 3. Install dependencies**

\`\`\`bash
pip install -r requirements.txt
\`\`\`

**## 4. Run the Streamlit application**

\`\`\`bash
streamlit run app.py
\`\`\`

Or:

\`\`\`bash
python -m streamlit run app.py
\`\`\`

**---**

**# A Note About the Model File**

The final deployment XGBoost model is approximately **\*\*1.43 MB\*\***.

This is dramatically smaller than the original Random Forest artifact, which was approximately **\*\*6.42 GB\*\*** before compression and approximately **\*\*1.17 GB\*\*** after Joblib compression.

The small XGBoost artifact makes it practical to package with the application and load during Streamlit startup.

This model-size difference was a major factor in the final deployment decision.

**---**

**# My Learning Journey**

I started this project with the goal of building a practical sales forecasting model.

The first stage was mostly about understanding the dataset and getting a model to produce predictions.

Then the project became more challenging.

I had to understand why time-based splitting was necessary, how lag and rolling features work, why future Sales features can become unavailable, and why a model that looks excellent during experimentation may not be usable for genuine future forecasting.

After that came model comparison.

Linear Regression gave me a useful baseline. Random Forest improved the results. XGBoost was also tested. Then came feature importance and error analysis, which made the model easier to understand instead of treating it as a black box.

The tuning stage taught me another lesson: a more complicated search does not guarantee a better result.

The biggest surprise came after the modelling work was finished.

The final model was about 6.42 GB.

At that point, the problem was no longer purely machine learning. It became an engineering problem.

I tried smaller models. They were easier to deploy but lost too much predictive performance. I tried compression. The file dropped to about 1.17 GB, but loading it still created substantial memory pressure.

Then I moved the project into a Streamlit environment and ran into a scikit-learn version mismatch. After identifying that the preprocessing pipeline had been created with scikit-learn 1.6.1, I aligned the local environment with that version and successfully loaded both the preprocessor and model.

Finally, the model was running inside a real Streamlit interface and producing predictions.

That was probably the most valuable part of the project for me.

The project did not go perfectly from the first cell to the final application.

It went through:

**\*\*experiment → failure → investigation → correction → validation → deployment\*\***

And that is exactly what I wanted from a hands-on machine learning project.

**---**

**# What I Learned**

This project taught me much more than how to call \`RandomForestRegressor\`.

**### 1. Forecasting is different from ordinary regression**

A model can achieve an impressive score by using information that will not actually exist when making a future prediction.

That distinction matters.

**### 2. Validation strategy matters**

Random cross-validation can give misleading results when time order matters.

For forecasting, the validation setup should resemble the real future prediction scenario.

**### 3. The best metric does not always mean the best model**

The model with MAE 538.72 was not selected for the final forecasting pipeline because it depended on historical Sales features that were not safely available for the official future test period.

**### 4. Hyperparameter tuning can make things worse**

The tuned Random Forest did not beat the original forecast-ready model.

More tuning does not automatically mean a better model.

**### 5. Deployment is part of machine learning**

The 6.42 GB model forced me to think about:

\- serialization
\- compression
\- memory
\- dependency versions
\- model portability
\- application startup time

Those problems do not appear when you only work inside a notebook.

**### 6. Failed experiments are useful**

The smaller Random Forest, XGBoost, HistGradientBoosting, tuning attempts, compression experiment, and dependency mismatch were not simply wasted attempts.

They helped answer a practical question:

\> **\*\*What actually works when the model has to leave the notebook and become an application?\*\***

**---**

**# Final Takeaway**

The final result is not just a notebook containing a trained Random Forest.

It is an end-to-end forecasting project that demonstrates:

\- data preparation
\- feature engineering
\- time-aware validation
\- model comparison
\- model evaluation
\- interpretability
\- error analysis
\- forecasting constraints
\- deployment engineering
\- environment reproducibility
\- Streamlit application development

The final deployment XGBoost model achieved approximately **\*\*1603.74 MAE\*\*** on the chronological holdout and was successfully integrated into the Streamlit application.

Although this MAE is higher than the **\*\*916.97\*\*** achieved by the forecast-ready Random Forest, the Random Forest was not practical for the target deployment environment because of its extremely large model footprint and runtime memory requirements.

The XGBoost model was approximately **\*\*1.43 MB\*\***, making it dramatically more practical to package, load, and serve.

This is an important real-world ML lesson: **the best model on a validation metric is not necessarily the best model for production deployment.**

The biggest lesson was that building the model is only one part of the job.

Getting that model to behave correctly outside the notebook is another.

**---**

**## Future Improvements**

Possible next steps include:

\- Reduce the model footprint further without a major accuracy loss
\- Use dedicated model storage for large artifacts
\- Optimize inference memory usage
\- Add batch prediction support
\- Add prediction confidence or uncertainty estimates
\- Add historical sales visualization
\- Add store-level analytics
\- Add automated retraining
\- Add monitoring for model drift
\- Deploy using a production-oriented model-serving architecture

**---**

**## Author**

Built as a hands-on machine learning and deployment learning project.

If you found the project useful or have suggestions for improving the forecasting pipeline, feel free to open an issue or start a discussion.