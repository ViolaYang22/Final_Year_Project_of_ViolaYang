# My Final Year Project: Medicare fraud detection by using resampling and machine learning methods

## Introduction
This project delves into the intricate realm of Medicare fraud detection, leveraging resampling methods and machine learning algorithms to address the inherent class imbalance problem in medical insurance datasets. Seven distinct resampling methods combined with four machine learning models were rigorously evaluated to determine their efficacy in identifying fraudulent healthcare claims. Additionally, through a meticulous analysis of performance metrics, including F1-score and AUC, we offer a comprehensive view of the strengths and weaknesses of different approaches. By conducting extensive experiments and utilizing a diverse array of CMS datasets, we provide insights into the optimal combinations of resampling techniques and machine learning algorithms. The results reveal that the XGBoost algorithm under BorderlineSMOTE resampling method demonstrates remarkable predictive power in identifying fraudulent Medicare claims, with AUC of 0.95.


## Code File Description

### `CMSdata_processing.ipynb`

- In this 'ipynb' file, we eliminate all 'NaN' entries from the datasets and replace them with the median value.
- Secondly, we calculate the number of 'NaN' values in each dataset to ensure all the inputs are numerical. Besides, commas in the value section are replaced with spaces.
- Thirdly, all 'string' values are converted to 'double' for descriptive analysis and model training.
- The datasets for the years 2019 and 2020 are merged within each dataset.
- A descriptive statistics table is generated and saved in the 'description_statistic_of_datasets' folder.

### `LEIE_data_processing.ipynb`

- In this 'ipynb' file, we import the LEIE dataset because it contains detailed information about the excluded parties, including their National Provider Identifier (NPI), names, addresses, and the reasons for their exclusion. if a healthcare provider's information matches an entry in the LEIE, it suggests that the provider is excluded from participating in Medicare and is potentially engaging in fraudulent activities.
- And then we extract data on frauds committed in the year of 2019, 2020 and 2021. 

### `frauds_label.ipynb`

- In this 'ipynb' file, we utilized the LEIE dataset to establish fraud labels, categorizing physicians as either having committed fraud or not. we labeled all three datasets (Part B, Part D and DMEPOS) by appending a new 'EXCLUSION' column at the end of each dataset. This 'EXCLUSION' column was assigned a value of 1 if fraud was detected based on NPI matching and 0 if no fraud was identified.
- And we calculate the number of fraud labels, Exclusion = 1, in Part B datasets of 2019, 2020 and 2021 after labeling.
- All datasets are encoded using One-hot Encoder.

### `dataset_combination.ipynb`

- In this 'ipynb' file, the feature 'NPI' is removed from the datasets: dataset_2019 and dataset_2020 as it has no impact on model training.
- Then we combine these two datasets and add zeros to the combined dataset. 


### `resampling&model_training.ipynb`

- In this 'ipynb' file, we firstly filter the training set(combined_dataset_1920) and the testing set (dataset_2021) by 
converting the 'Rfrg_Prvdr_State_FIPS' column to string type, creating a boolean mask to identify rows with mixed letters and numbers and inverting the mask to select rows without mixed letters and numbers.
- Then we apply seven distinct resampling methods to the filtered training set and save the new training set after resampling in the 'training_set_after_resampling' folder.
- Finally, we train four machine learning models including Logistic Regression, Random Forest, Naive Bayes and XGBoost by using the resampled training datasets, and the results are saved in the folder called 'model_training'.


### `model_evaluation.ipynb`

- In this 'ipynb' file, we use two evaluation metrics (ROC-AUC and F1 score) to evaluate the performance of four machine learning models under seven different resampling methods. It will generate related ROC curves with corresponding AUC values and F1-scores. And we save the figures of our outcomes in the folder 'model_evaluation_outcomes'.








