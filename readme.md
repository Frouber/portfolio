#Overview of scripts:  
ETFModels.py is a wrapper for TSCV that takes a time series of signals and calculates positions and strategy performance. TSCV takes data set, output from ETFModels.py, and the number of k-folds to output various training and test set metrics. These are dependent on the cross validation function chosen. Test_framework.py and test_linearModel.py, are templates for executing the backtest cross validation, based on the instructions in the Readme file.      

#Instructions: 
##Steps with sample code under each.    
1. Import ETFModels and TCSV classes   
from ETFModels import ETFModels   
from TSCV import TSCV

2. Create/import data, should be DataFrame of two columns, signals and returns  
example_data = pd.DataFrame(np.random.rand(1000,2),columns=['signals','returns'])

3. Instantiate ETFModel object with appropriate model  
btModel=ETFModels('backtestEval')

4. Define k-folds  
kfold=5

5. Instantiate TSCV object with three arguments: dataFrame, ETFModel object, k-folds     
tscv=TSCV(example_data,btModel.estimate,kfold)

6. Select a tcsv function w/appropriate arguments:      
-onePass(params)  
-aux_fun(params)  
-findHP()  
-optimal_params()     

7.  Define parameters:   
-For onepass and aux_fun:    
Define parameters to pass to tscv in dictionary, specifically:  tcost, tcost_aversion,'target_avg':5,'n_max':10  
params={'tcost':0.0001,'tcost_aversion':0,'target_avg':5,'n_max':10}.  
-For findHP() and optimal_params():  
Define parameters in dictionaries for these functions.  

8. Instantiate and run estimation  
metricArr=tscv.onePass(params)

9. Print output  
print metricArr  
  
#Expected Outputs:    
-onepass: array with test/train data    
-aux_fun: cross validation metric    
-findHP: ridge-alpha, test_err, train_err  
-optimal_params: alpha value  

#Sample script (for reference):  
import numpy as np  
import pandas as pd  

from TSCV import TSCV  
from ETFModels import ETFModels  

example_data = pd.DataFrame(np.random.rand(1000,2),columns=['signals','returns'])  
btModel=ETFModels('backtestEval')  
kfold=3  
tscv=TSCV(example_data,btModel.estimate,kfold)  
params={'tcost':0.0001,'tcost_aversion':0,'target_avg':5,'n_max':10}  
metricArr=tscv.onePass(params)  
print metricArr  
