# Salary Prediction Summary

Salary Prediction Using Machine Learning (Linear regression, Random Forest, and Gradient Boost Algorithms)
<!-- <table>
  <tr>
    <th>Files</th>
    <th>Notes</th>
  </tr>
  <tr>
    <td>/module/helpers.py</td>
    <td>It contains the tools built to facilitate EDA and preprocessing</td>
  </tr>
  <tr>
    <td>salary_EDA.ipynb</td>
    <td>Priliminary steps and EDA (Exploratory Data Analysis)</td>
  </tr>
    <tr>
    <td>salary_model.ipynb</td>
    <td>Machine Learning model building and Hyper Parameter Tuning</td>
  </tr>
</table>
<br> -->

# 1. Problem Definition
Salary of a employee in general is related to many factors such as yearsExperience, job title and degree.
In this project, we use multiple factors to predict salary. The factors include:
  * CompanyID
  * jobType
  * degree
  * major
  * industry
  * yearsExperience
  * milesFromMetropolis

We use a metric MSE (Mean Squared Error) to assess the prediction accuracy.
The lower MSE, the better the prediction.

# 2. DISCOVER
## 2.1 Dataset Description
* There are 1 million records in the training set
* Numeric variables:
      - yearsExperience
      - milesFromMetropolis
      - salary
* Categorical variables:
      - jobId
      - companyId
      - jobType
      - degree
      - major
      - industry
* jobId is all unique, not included as a feature
* companyId has 63 unique values, can not easily visualize
* The rest of the categorical variables - jobType, degree, major and industry have a small amount of unique values and can visualize
<img width="433" alt="img1" src="https://user-images.githubusercontent.com/90524579/220082382-c639dde6-3893-4cf4-aa22-db87222b2248.png">
<img width="885" alt="img2" src="https://user-images.githubusercontent.com/90524579/220082355-16ee992d-f753-4676-b8cd-68afd2ae6aa6.png">

## 2.2 Checking salary = 0 Situation
* The 0 salary has valid fields in other columns, so it does not look like it is an unpaid position
* There are only a small amount of them (n = 5)
* Remove salary = 0 rows
<img width="701" alt="img3" src="https://user-images.githubusercontent.com/90524579/220082334-4b363405-8e72-4755-a8b1-89fe12e99be1.png">
<img width="442" alt="img4" src="https://user-images.githubusercontent.com/90524579/220082308-b5f1e866-58c0-4030-b101-e1c29c0cdf8b.png">

## 2.3 Plotting Salary
* There are some outliers above the upper bound
* Salary has a nice normal distribution
<img width="827" alt="img5" src="https://user-images.githubusercontent.com/90524579/220082280-d9fee93c-d9fa-4630-9475-cefdeda46228.png">

## 2.4 study salary outliers
* Salaries that are above the upper bounds have the job titles as 'CEO', 'CFO', 'CTO', 'VICE_PRESIDENT', 'SENIOR', 'MANAGER'. But it also has the job title 'JUNIOR'.
* Further examined the rows with salary above the upper bound and job title being 'JUNIOR', the degree field are all advanced degree, and the industry are all oil/web/finance.
* They make sense so decided to keep the outliers.
<img width="249" alt="img12" src="https://user-images.githubusercontent.com/90524579/220082246-c210f261-10bc-492a-906e-13eb76df2f71.png">
<img width="861" alt="img13" src="https://user-images.githubusercontent.com/90524579/220082215-fceb7eb9-7e60-42ab-8d19-edcf6c12fb72.png">

## 2.5 Plotting each feature in relation to Salary
### jobType
* There were fairly equal amount of job types in the training set
* Salary goes up in the order of 'JANITOR', 'JUNIOR', 'SENIOR', 'MANAGER', 'VICE_PRESIDENT', 'CFO', 'CTO', 'CEO'
<img width="851" alt="img6" src="https://user-images.githubusercontent.com/90524579/220082117-f7285e0e-f6aa-4e1e-99fb-07db713f1df9.png">

### degree
* There were more high school and none degrees than other categories
* The salaries in high school and none degrees are lower than other categories
<img width="862" alt="img7" src="https://user-images.githubusercontent.com/90524579/220082092-c8c2f0ee-2487-4eef-b80e-510dbb34c753.png">

### major
* More than 50% of the case have none major.
* Cases that have a major are pretty evenly distributed across different majors.
* None major has a lower salary than any major
<img width="871" alt="img8" src="https://user-images.githubusercontent.com/90524579/220082060-d1f36967-8beb-4710-ae8a-04df8b7c6bf5.png">

### industry
* There were fairly equal amount of industry types in the training set
* salaries are the highest in 'FINANCE', 'OIL'
* salaries are the loest in 'EDUCATION', 'SERVICE'
<img width="864" alt="img9" src="https://user-images.githubusercontent.com/90524579/220081967-7a80abc9-00fb-4e05-bf89-48500a411e8e.png">

### yearsExperience
* Years of experience is fairly evenly distributed across the range of 0 to 24 years.
* There is a positive linear correlation between salary and years of experience
<img width="898" alt="img10" src="https://user-images.githubusercontent.com/90524579/220081931-abccfc55-5573-418a-83fc-d0147f9d820d.png">


### milesFromMetropolis
* Miles from metropolis is fairly evenly distributed across the range of 0 to 100 miles.
* There is a negative linear correlation between salary and miles from metropolis
<img width="870" alt="img11" src="https://user-images.githubusercontent.com/90524579/220082010-21b09cca-ecbf-4fcd-9aea-be9575308bb0.png">

## 2.6 Encode categorical variables and plot feature correlation with salary
* Encode each categorical variables with the mean of the salary of that category
* Salary is positively related with encoded jobType, degree, major, industry and yearsExperience
* Salary is negatively related with milesFromMetropolis
<img width="702" alt="img14" src="https://user-images.githubusercontent.com/90524579/220081790-3724622d-2ece-44b7-95c4-91545936bf21.png">

# 3. DEVELOP
## 3.1 Preprocessing using scripts in modules/helpers.py
* Combine the training features with salary
* Drop rows with duplicated jobId
* For both the training and test sets, convert the following variables to category 'companyId', 'jobType', 'degree', 'major', 'industry'
* Remove the rows in training when salary = 0
* Transform categorical variables to be the mean salary of each category and use them as part of the feature group: feature_transform
* Encode categorical variables and use them as part of the feature group: feature_encode

## 3.2 Test baseline models
* Establish the metrics, will use Mean Square Error (MSE) as the metrics to determine prediction accuracy
* Test the baseline prediction, tried using the following transformed categorical columns as the prediction: industry, major, degree, jobType. JobType generated the smallest MSE = 964.1529
<table>
  <tr>
    <th>baseline models</th>
    <th>MSE</th>
  </tr>
  <tr>
    <td>mean salary for each industry</td>
    <td>1367.5539</td>
  </tr>
  <tr>
    <td>mean salary for each major</td>
    <td>1284.3599</td>
  </tr>
  <tr>
    <td>mean salary for each degree</td>
    <td>1257.9450</td>
  </tr>
    <tr>
    <td>mean salary for each jobType</td>
    <td>964.1529</td>
  </tr>
</table>

## 3.3 Testing machine learning models
* Several Machine Learning Models are tried with different feature combinations
* note: transformed means categorical features transformed to be the mean salary of each category.
* note: encoded means categorical features were coded with arbitary numeric numbers.
* note: also included an option of whether to use scaler (normalize all the numeric variables).
<img width="915" alt="img18" src="https://user-images.githubusercontent.com/90524579/220081734-8e2ce5f0-6a01-49b9-8401-f64eed0ba5e3.png"> 

Observations:
* Different models generated similar MSE, except linear regression did really poorly with encoded features.
* The model with the lowest MSE is GradientBoosting with transformed features and scaled.
* Whether to scale the numeric variables makes very little or no difference in the MSE.

<table>
  <tr>
    <th>Features</th>
    <th>ML models</th>
    <th>MSE</th>
  </tr>
  <tr>
    <th>Numeric + transformed categorical not scaled</th>
    <td>Linear Regression</td>
    <td>399.7641</td>
  </tr>
  <tr>
    <th></th>
    <td>Random Forest</td>
    <td>365.8091</td>
  </tr>
  <tr>
    <th></th>
    <td>GradientBoosting</td>
    <td>364.1131</td>
  </tr>
  <tr>
    <th>Numeric + transformed categorical scaled</th>
    <td>Linear Regression</td>
    <td>399.7641</td>
  </tr>
  <tr>
    <th></th>
    <td>Random Forest</td>
    <td>365.7466</td>
  </tr>
  <tr>
    <th></th>
    <td>GradientBoosting</td>
    <td>364.1131</td>
  </tr>
  <tr>
    <th>Numeric + encoded categorical not scaled</th>
    <td>Linear Regression</td>
    <td>925.0988</td>
  </tr>
  <tr>
    <th></th>
    <td>Random Forest</td>
    <td>372.4568</td>
  </tr>
  <tr>
    <th></th>
    <td>GradientBoosting</td>
    <td>379.0314</td>
  </tr>
  <tr>
    <th>Numeric + encoded categorical scaled</th>
    <td>Linear Regression</td>
    <td>925.0988</td>
  </tr>
  <tr>
    <th></th>
    <td>Random Forest</td>
    <td>372.5084</td>
  </tr>
  <tr>
    <th></th>
    <td>GradientBoosting</td>
    <td>379.0315</td>
  </tr>
</table>

## 3.4 Choose the best model and buid pipeline
* GradientBoosting with transformed and scaled features provided the best results (lowest MSE)
<img width="901" alt="img15" src="https://user-images.githubusercontent.com/90524579/220081683-d833242c-c632-4e44-8fac-1f3255e36d86.png">


## 3.5 Train the model on the whole dataset

# 4. DEPLOY
## 4.1 Appling the model on the test data
<img width="908" alt="img16" src="https://user-images.githubusercontent.com/90524579/220081633-01cd7c58-18d9-47cf-ba81-afb956630994.png">

## 4.2 Feature importance
* jobType_transformed has the greatest contribution. It is consistent with our baseline analysis where using the mean average for each jobType gave us the best baseline results (compared to using degree, major and industry)
* Other important features are yearsExperience, milesFromMetropolis
<img width="903" alt="img17" src="https://user-images.githubusercontent.com/90524579/220081575-ad048856-19f5-4c02-898d-67a7aebd62fd.png">
<!-- 
## 4.3 Improve the model
* The model can be improved by removing some colinearity in the data for linear regression -->
