## ML_predict

### Introduction
- The goal of this competition is to use various factors to predict obesity risk in individuals, which is related to cardiovascular disease.

### Project Period
- 4 days (February 26th, 2024 ~ February 29th, 2024)

### Dataset Description
- train.csv
- test.csv

| Column    | Full Form |     Description |
|-----------|-----------|--------------------------|
| 'id'     | id     | Unique for each person(row)    |
| 'Gender' | Gender  |person's Gender   |
| 'Age'	| Age	| Dtype is float. Age is between 14 years to 61 years |
|'Height'	| Height	 | Height is in meter it's between 1.45m to 1.98m |
|'Weight' |	Weight	| Weight is between 39 to 165. I think it's in KG.|
|'family_history_with_overweight'	|family history with overweight	 | yes or no question |
|'FAVC'	| Frequent consumption of high calorie food	| it's yes or no question. i think question they asked is do you consume high calorie food |
|'FCVC' | Frequency of consumption of vegetables	| Similar to FAVC. this is also yes or no question|
|'NCP'	| Number of main meals	| dtype is float, NCP is between 1 & 4. I think it should be 1,2,3,4 but our data is synthetic so it's taking float values |
|'CAEC'	| Consumption of food between meals	| takes 4 values Sometimes, Frequently, no, & Always |
|'SMOKE' |	Smoke	| yes or no question. i think the question is "Do you smoke?" |
|'CH2O'|Consumption of water daily	|CH2O takes values between 1 & 3. again it's given as float may be because of synthetic data. it's values should be 1,2 or 3|
|'SCC'	|Calories consumption monitoring|	yes or no question|
|'FAF'	|Physical activity frequency	|FAF is between 0 to 3, 0 means no physical activity and 3 means high workout. and again, in our data it's given as float|
|'TUE'	|Time using technology devices	|TUE is between 0 to 2. I think question will be "How long you have been using technology devices to track your health." in our data it's given as float |
|'CALC'	| Consumption of alcohol	|Takes 4 values: Sometimes, no, Frequently, & Always |
|'MTRANS'|	Transportation used	| MTRANS takes 5 values Public_Transportation, Automobile, Walking, Motorbike, & Bike|
|'NObeyesdad'	|TARGET	|This is our target, takes 7 values, and in this comp. we have to give the class name (Not the Probability, which is the case in most comp.)|

### Libraries
#### Data preprocessing      
- **pandas==2.2.0**
- **numpy==1.26.3**
- **sklearn==1.2.2**
- 
#### Visualization
- **matplotlib==3.7.4**
- **seaborn==0.12.2**
- 
#### Hyperparameter Tuning
- **optuna==3.5.0**

#### Modeling
- **lightgbm==4.2.0**

### Project Link
-[Kaggle_Multi-Class Prediction of Obesity Risk](https://www.kaggle.com/competitions/playground-series-s4e2/overview)
-[Google Slide presentation materials]
