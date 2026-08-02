## Stage 2B -- Classification Pipeline Explanation

## Objective

The objective of Stage 2B was to develop and evaluate supervised machinelearning classification models capable of predicting the status of cloudcomputing tasks using cloud telemetry and engineered performancefeatures.

The target variable selected for prediction wastask_status_encoded, which represents the encoded form of thecloud task status.

Original Status     Encoded Value

completed                       0running                         1waiting                         2

## Dataset Preparation

The cleaned dataset generated during Stage 1 was used as the input tothe classification pipeline. The dataset contained approximately 2million records and 26 variables.

Only predictive features were selected for training. The final featureset consisted of:

cpu_usage

memory_usage

network_traffic

power_consumption

num_executed_instructions

execution_time

energy_efficiency

cpu_memory_ratio

power_per_instruction

throughput

system_load_index

cpu_power_interaction

task_type_encoded

task_priority_encoded

cpu_load_level_encoded

exec_time_bucket_encoded

Identifier columns (vm_id, timestamp) and duplicate categoricalstring columns were excluded because they either contained no predictiveinformation or duplicated the encoded variables.

## Data Splitting

The dataset was divided into training, validation and testing subsets.This ensured that the models were trained on one portion of the datawhile their performance was evaluated on previously unseen data.

## Feature Scaling

A StandardScaler was applied to standardize the numerical features.

To prevent data leakage, the scaler was fitted only on the training setand then applied to the validation and test sets.

## Classification Models

The pipeline was designed to compare several supervised learningalgorithms, including:

Logistic Regression

Linear Discriminant Analysis (LDA)

Quadratic Discriminant Analysis (QDA)

Gaussian Naïve Bayes
Decision Tree
Random Forest
Support Vector Machines (Linear and RBF)

These algorithms represent different machine learning approaches,allowing comparison between linear, probabilistic, tree-based, ensemble,instance-based and margin-based classifiers.

## Model Evaluation

Each classifier was evaluated using:

Accuracy

Precision

Recall

F1-score

ROC-AUC

5-fold Cross-Validation

These metrics provided a comprehensive assessment of model performancebeyond simple accuracy.

##  Analysis

Training accuracy and testing accuracy were compared for every model.

The difference between these values (the overfitting gap) was used todetermine whether a model generalized well or memorized the trainingdata.

## Generative vs Discriminative Analysis

Learning curves were generated to compare generative models (NaïveBayes, LDA and QDA) with discriminative models (Logistic Regression, SVMand Random Forest).

The curves illustrate how validation performance changes as the amountof training data increases and help identify underfitting or overfittingbehaviour.

## Confusion Matrix Analysis

Confusion matrices were generated for the highest-performing models.

These visualizations showed:

Correct classifications

Misclassifications

Class-wise prediction accuracy

This helped identify which cloud task states were most difficult for themodels to distinguish.

## Overall Outcome

The Stage 2B classification pipeline successfully:

Loaded the cleaned cloud telemetry dataset.

Selected appropriate predictive features.

Split the data into training, validation and testing sets.

Standardized numerical variables while preventing data leakage.

Trained multiple supervised classification algorithms.

Evaluated each model using multiple performance metrics.

Performed 5-fold cross-validation.

Assessed potential overfitting.

Generated learning curves for generative and discriminative models.

Produced confusion matrices for the best-performing classifiers.

Overall, the pipeline provides a comprehensive framework for comparingsupervised learning algorithms in predicting cloud task status andsupports the Intelligent Cloud Resource Optimization System byidentifying models capable of assisting resource management decisions.