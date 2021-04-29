# libraries
import numpy as np;
import pandas as pd;
import matplotlib
import matplotlib.pyplot as plt;
import seaborn as sns;
from scipy import stats;
from scipy.stats import zscore;
from sklearn.model_selection import train_test_split
from sklearn import metrics

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import BaggingRegressor
from xgboost.sklearn import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor


import warnings
warnings.filterwarnings("ignore")

# pip install pyforest
# pip install lazypredict

# import pyforest
# import lazypredict
# from lazypredict.Supervised import LazyRegressor 

import streamlit as st 
from PIL import Image
# matplotlib.use('Agg')

# Set title
st.title("Machine Learning Project with Streamlit Deployment")
st.subheader("Predicting Compressive Concrete Strength")
# image = Image.open('tdslogo.png')
# st.image(image, use_column_width=True)
# Main activities
def main():
	activities=['EDA', "Feature Engineering", 'Model Building & Performance','Summary']
	option=st.sidebar.selectbox('Selection Option', activities)

	# # EDA		
	if option == "EDA":
		st.subheader('Exploratory Data Analysis')
		df=pd.read_csv('data/compresive_strength_concrete.csv') # upload the data chossen above
		st.success("Data Sucessfully loaded")
		st.dataframe(df.head())

		st.markdown("Notice that the column names are very long. We will rename these to a shorter version before continuing with the Exploratory Data Analysis")
		df = df.rename(columns={'Cement (component 1)(kg in a m^3 mixture)':'cement',
       'Blast Furnace Slag (component 2)(kg in a m^3 mixture)':'slag',
       'Fly Ash (component 3)(kg in a m^3 mixture)':'ash',
       'Water  (component 4)(kg in a m^3 mixture)':'water',
       'Superplasticizer (component 5)(kg in a m^3 mixture)':'superplasticizer',
       'Coarse Aggregate  (component 6)(kg in a m^3 mixture)':'coarse_agg',
       'Fine Aggregate (component 7)(kg in a m^3 mixture)':'fine_agg',
       'Age (day)':'age',
       'Concrete compressive strength(MPa, megapascals) ':'strength'})
		st.write(df.head(20))

		# display dataframe
		if st.checkbox("Display Shape"):
				st.write(df.shape)
		if st.checkbox('Display columns'):
				st.write(df.columns)
		if st.checkbox("Display Summary"):
				st.write(df.describe().T)
		if st.checkbox("Display Variable Types"):
				st.write(df.dtypes)
		if st.checkbox('Display Null Values'):
				st.write(df.isnull().sum())
		
		
		if st.checkbox('Display IQR and Upper/Lower Limits'):	
			st.markdown('IQR= Q3-Q1: interquartile range is a measure of where the bulk of the values lieand outliers are anything above or below the following limits will be outliers.')
			st.markdown('Lower limit = Q1 - 1.5*IQR ')
			st.markdown('Upper limit = Q3 + 1.5*IQR ')
			st.markdown('IQR = IQR= Q3-Q1 ')
			for var in df.columns:

				lower = np.round(df[var].quantile(q=0.25)-1.5*(df[var].quantile(q=0.75)-df[var].quantile(q=0.25)),3)
				upper = np.round(df[var].quantile(q=0.75)+1.5*(df[var].quantile(q=0.75)-df[var].quantile(q=0.25)),3)	
				st.subheader(var)
				st.write('Interquartile range (IQR) for',var, np.round(stats.iqr(df[var]), 3))
				st.write('Upper Limit = ',upper)
				st.write('Lower Limit = ',lower)
				st.write('No. of outliers in upper',var,'=',df[df[var]>upper][var].count())
				st.write('No. of outliers in lower',var,'=',df[df[var]<lower][var].count())
				st.write('% of outliers in upper',var,'=',np.round(df[df[var]>upper][var].count()*100/len(df),2),'%')
				st.write('% of outliers in lower',var,'=',np.round(df[df[var]<lower][var].count()*100/len(df),2),'%')
				st.text("----"*100)

		if st.checkbox('Display boxplot of outliers'):
			fig = plt.figure()
			st.write(df.boxplot(figsize=(15,10)))
			plt.title("Boxplot of Outliers")
			st.pyplot(fig)

		# Variable Distribution
		if st.checkbox('Display Distributions'):
			fig, ax2 = plt.subplots(3,3, figsize=(16,16))
			sns.distplot(df['cement'], ax=ax2[0][0])
			sns.distplot(df['slag'], ax=ax2[0][1])
			sns.distplot(df['ash'], ax=ax2[0][2])
			sns.distplot(df['water'], ax=ax2[1][0])
			sns.distplot(df['superplasticizer'], ax=ax2[1][1])
			sns.distplot(df['coarse_agg'], ax=ax2[1][2])
			sns.distplot(df['fine_agg'], ax=ax2[2][0])
			sns.distplot(df['age'], ax=ax2[2][1])
			sns.distplot(df['strength'], ax=ax2[2][2])
			st.pyplot(fig)
			st.markdown('Note, some distributions above appear normal and some are right skewed. Some look like normal but have more than one gaussian')
			st.markdown('For linear models to perform best (we assume that variables are normally distributed), we need to account for non-Gaussian distributions. We can transform our variables above (see if we can change the distribution from skewed to Gaussian) during feature engineering. One way of transforming variables is by applying the logarithm.')

		# Pairplot
		if st.checkbox('Display a Pairplot'):
			fig = sns.pairplot(df, diag_kind='kde')
			st.pyplot(fig)
			st.markdown('From the pairplot there appears to be no multicollinearity.')
			st.markdown('Multicollinearity exists whenever an independent variable is highly correlated with one or more of the other independent variables in a multiple regression equation. Multicollinearity is a problem because it undermines the statistical significance of an independent variable.')
		# Correlation	
		if st.checkbox('Display Heatmap (Correlation)'):
				fig = plt.figure(figsize=(15,8))
				plt.title('Correlation between different attributes')
				st.write(sns.heatmap(df.corr(), vmax=1, square=True, annot=True, cmap='viridis'))
				st.pyplot(fig)

				fig = plt.figure(figsize=(12,8))
				plt.title('Attribute Scale of Correlation')
				st.write(df.corr()['strength'][:-1].sort_values().plot(kind='bar'))
				st.pyplot(fig)


	# Feature Engineering		
	elif option == "Feature Engineering":
		st.subheader('Feature Engineering')
		df=pd.read_csv('data/compresive_strength_concrete.csv') # upload the data chossen above
		st.success("Data Sucessfully loaded")
		df = df.rename(columns={'Cement (component 1)(kg in a m^3 mixture)':'cement',
       'Blast Furnace Slag (component 2)(kg in a m^3 mixture)':'slag',
       'Fly Ash (component 3)(kg in a m^3 mixture)':'ash',
       'Water  (component 4)(kg in a m^3 mixture)':'water',
       'Superplasticizer (component 5)(kg in a m^3 mixture)':'superplasticizer',
       'Coarse Aggregate  (component 6)(kg in a m^3 mixture)':'coarse_agg',
       'Fine Aggregate (component 7)(kg in a m^3 mixture)':'fine_agg',
       'Age (day)':'age',
       'Concrete compressive strength(MPa, megapascals) ':'strength'})
		st.write(df.head())

		# Replace Outliers
		st.markdown(" ### Replace outliers with the Median")
		st.markdown('If a point is below lower OR higher than upper then replace with the median of that column')

		if st.checkbox('Display Boxplot of Outliers'):
			fig = plt.figure()
			st.write(df.boxplot(figsize=(15,10)))
			plt.title("Boxplot of Outliers (Replaced with Median)")
			st.pyplot(fig)

		for cols in df.columns[:-1]:
		   	q1 = df[cols].quantile(0.25)
		   	q3 = df[cols].quantile(0.75)
		   	iqr = q3-q1 
		   	lower = q1-1.5*iqr # calculate the lower quartile for cement
		   	upper = q3+1.5*iqr # calculate the upper quartile for cement
		   	df.loc[(df[cols] < lower) | (df[cols] > upper), cols] = df[cols].median() 
		
		if st.checkbox('Display Boxplot of Outliers - replaced with Median'):
			fig = plt.figure()
			st.write(df.boxplot(figsize=(15,10)))
			plt.title("Boxplot of Outliers (Replaced with Median)")
			st.pyplot(fig)

		if st.checkbox('Display Distributions'):
			fig, ax2 = plt.subplots(3,3, figsize=(16,16))
			sns.distplot(df['cement'], ax=ax2[0][0])
			sns.distplot(df['slag'], ax=ax2[0][1])
			sns.distplot(df['ash'], ax=ax2[0][2])
			sns.distplot(df['water'], ax=ax2[1][0])
			sns.distplot(df['superplasticizer'], ax=ax2[1][1])
			sns.distplot(df['coarse_agg'], ax=ax2[1][2])
			sns.distplot(df['fine_agg'], ax=ax2[2][0])
			sns.distplot(df['age'], ax=ax2[2][1])
			sns.distplot(df['strength'], ax=ax2[2][2])
			st.pyplot(fig)
			st.write('Note, some distributions appear normal and some are right skewed. Some look normal but have more than one gaussian')
			st.info('For linear models to perform best (we assume that variables are normally distributed), we need to account for non-Gaussian distributions. We can transform our variables above (see if we can change the distribution from skewed to Gaussian) during feature engineering. One way of transforming variables is by applying the logarithm.')

		if st.checkbox('Apply and Display the Transform'):
			for var in df.columns:
				# logarithm does not take zero or negative values - skip these variables
				if 0 in df[var].unique():
					pass
				else:
					df[var] = np.log(df[var])
			st.success("Logarithm Sucessfully Completed")
			fig, ax2 = plt.subplots(3,3, figsize=(16,16))
			sns.distplot(df['cement'], ax=ax2[0][0])
			sns.distplot(df['slag'], ax=ax2[0][1])
			sns.distplot(df['ash'], ax=ax2[0][2])
			sns.distplot(df['water'], ax=ax2[1][0])
			sns.distplot(df['superplasticizer'], ax=ax2[1][1])
			sns.distplot(df['coarse_agg'], ax=ax2[1][2])
			sns.distplot(df['fine_agg'], ax=ax2[2][0])
			sns.distplot(df['age'], ax=ax2[2][1])
			sns.distplot(df['strength'], ax=ax2[2][2])
			st.pyplot(fig)
			st.info("The results above indicate that applying the logarithm does little on improving non-gaussian distributions. Also, some of the variables have zero values so taking the log was not possible")

		if st.checkbox('Scale the Data'):
			X = df.drop('strength', axis=1)
			y = df['strength']
			Xscaled = X.apply(zscore)
			st.success('Data successfully scaled using zscore')
			st.dataframe(Xscaled.head())


	# Model Building		
	elif option == "Model Building & Performance":
		st.subheader('Model Building & Performance')
		df=pd.read_csv('data/compresive_strength_concrete.csv') # upload the data chossen above
		st.success("Data Sucessfully loaded")
		df = df.rename(columns={'Cement (component 1)(kg in a m^3 mixture)':'cement',
       'Blast Furnace Slag (component 2)(kg in a m^3 mixture)':'slag',
       'Fly Ash (component 3)(kg in a m^3 mixture)':'ash',
       'Water  (component 4)(kg in a m^3 mixture)':'water',
       'Superplasticizer (component 5)(kg in a m^3 mixture)':'superplasticizer',
       'Coarse Aggregate  (component 6)(kg in a m^3 mixture)':'coarse_agg',
       'Fine Aggregate (component 7)(kg in a m^3 mixture)':'fine_agg',
       'Age (day)':'age',
       'Concrete compressive strength(MPa, megapascals) ':'strength'})
		st.write(df.head())

		# Replace outliers
		for cols in df.columns[:-1]:
		   	q1 = df[cols].quantile(0.25)
		   	q3 = df[cols].quantile(0.75)
		   	iqr = q3-q1 
		   	lower = q1-1.5*iqr # calculate the lower quartile for cement
		   	upper = q3+1.5*iqr # calculate the upper quartile for cement
		   	df.loc[(df[cols] < lower) | (df[cols] > upper), cols] = df[cols].median() 
		st.success('Outliers successfully replaced with the median')   	

		# Scale the data
		X = df.drop('strength', axis=1)
		y = df['strength']
		Xscaled = X.apply(zscore)
		st.success('Data successfully scaled using zscore')

		seed=st.sidebar.slider('Seed',1,200)

		# choose a classifier from the sidebar
		regressor_name=st.sidebar.selectbox('Select a classifier:',('Random Forest','Gradient Boosting','Bagging','XGB','KNN','Linear R'))

		# define the parameters
		def add_parameter(name_of_clf):
			param=dict() 
			if name_of_clf=='Random Forest':
				n_est=st.sidebar.slider('n_estimators', 50, 350, (100))
				param['n_estimators']=n_est 
			elif name_of_clf=='XGB':
				n_est=st.sidebar.slider('n_estimators', 50, 350, (100))
				param['n_estimators']=n_est 
			else:
				name_of_clf=='KNN'
				K=st.sidebar.slider('K', 1,20, (5))
				param['K']=K 
			return param
		# call the parameter function
		param=add_parameter(regressor_name) # classifier chosen from the sidebar

			# Initalise and define classifier function
		def get_regressor(name_of_clf, param):
			clf=None
			if name_of_clf=='Gradient Boosting':
			   	clf=GradientBoostingRegressor()
			elif name_of_clf=='KNN':
				clf=KNeighborsRegressor(n_neighbors=param['K'])
			elif name_of_clf=='Linear R':
				clf=LinearRegression()
			elif name_of_clf=='Bagging':
				clf=BaggingRegressor()
			elif name_of_clf=='Random Forest':
				clf=RandomForestRegressor(n_estimators=param['n_estimators'])
			elif name_of_clf == 'XGB':
				clf=XGBRegressor(n_estimators=param['n_estimators'])
			else:
				st.warning('Select a Algorithim')

			return clf

		clf=get_regressor(regressor_name, param)

		X_train, X_test, y_train, y_test=train_test_split(Xscaled,y,
				test_size=0.2, random_state=seed)
		# fit the data to the regressor
		clf.fit(X_train, y_train)

		# make predictions
		y_pred=clf.predict(X_test)
		st.write("Predictions:",y_pred)

		# Accuracy
		model_score=clf.score(X_train,y_train)
		test_score=clf.score(X_test, y_test)

		st.write('Name of Regressor: ',regressor_name)
		st.write('MAE:', metrics.mean_absolute_error(y_test, y_pred))
		st.write('MSE:', metrics.mean_squared_error(y_test, y_pred))
		st.write('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
		st.write('Training Data Performance', model_score)
		st.write('R Squared Statistic: ', metrics.r2_score(y_test, y_pred))

		# Scatter Plot of predictions
		if st.checkbox('Display Scatter plot of predictions'):
			fig = plt.figure()
			st.write(plt.scatter(y_test,y_pred))
			plt.title("Scatter plot of Predictions")
			st.pyplot(fig)

		# Distribution of Residuals
		if st.checkbox('Display distribution of Residuals'):
			fig = plt.figure()
			st.write(sns.distplot((y_test-y_pred), bins=50))
			st.pyplot(fig)
			st.info('The residuals should be normally distributed otherwise the model may not be the best choice for the dataset')

	# Summary		
	elif option == "Summary":
		st.subheader('Summary')
		st.text("-"*100)
		st.markdown('Judging from the various model performaces it looks like the XGB model is the clear winner. However there are quite a few options and tweaks we could make to these models that could improve model performance.')
		st.markdown('If we compare the training score and the test scores we can see there is a big difference between some of the models. This indicates that models are overfitting (on the training data).')
		st.markdown('For most of the models MSE is pretty large so we need to minimise these errors by tunning')
		st.markdown('The following recommendations can make a huge difference to model performance:')

		"""
		* This project includes just a few hyperparameter tuning options, such as number of estimators for Random Forest and number of nearest neighbors for KNN. There are many more hyperparameter tuning options which may have a significant impact on the performance of the model.
		* Remove features that are not important and do not impact the performance of the model
		* Splitting the data before processing to prevent data leakage. There's arguments for and against and here's one: https://machinelearningmastery.com/data-preparation-without-data-leakage/
		* Get more data
		"""
		st.text("-"*100)
		st.subheader('The Streamlit Application')
		st.markdown('Streamllit is an open source application framework ideal for python data visualisation, versatile model building and testing support, and machine learning applications')
		st.markdown('No knowledge of HTML is required; widgets are treated as variables so there is no callbacks required; a small amount of code creates a lot of output; and data caching allows for faster computations.')
		st.text("-"*100)
		st.subheader('Notes')
		st.markdown(' ### Regression Evaluation Metrics')
		st.info('**Mean Absolute Error** (MAE) is the mean of the absolute value of the errors')
		st.info('**Mean Squared Error** (MSE) is the mean of the squared errors')
		st.info('**Root Mean Squared Error** (RMSE) is the square root of the mean of the squared errors')	
		st.info('R Squared Statistic is the Proportion of Variance Explained. Close to 1 indicates large proportion of variability has been explained by the model, i.e. proportion of variabilty in Y that can be explained using X.')

if __name__ == '__main__':
	main()