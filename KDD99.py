from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression

import pandas as pd
pd.set_option('mode.chained_assignment', None)

from IPython.display import display
import tensorflow as tf
from tensorflow.keras.utils import get_file

import matplotlib.pyplot as pyplot
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import preprocessing
import joblib
import numpy as np

import io
import requests
import os
from sklearn import metrics
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.callbacks import EarlyStopping

import time


def download_data():
	try:
		path = get_file('kddcup.data.gz', origin=
		'http://kdd.ics.uci.edu/databases/kddcup99/kddcup.data.gz')
	except:
		print('Error downloading')
		raise
    
	return path 


#adding labels to columns and defining attack types
def preprocessing_data():
	cols = """
		duration,
		protocol_type,
		service,
		flag,
		src_bytes,
		dst_bytes,
		land,
		wrong_fragment,
		urgent,
		hot,
		num_failed_logins,
		logged_in,
		num_compromised,
		root_shell,
		su_attempted,
		num_root,
		num_file_creations,
		num_shells,
		num_access_files,
		num_outbound_cmds,
		is_host_login,
		is_guest_login,
		count,
		srv_count,
		serror_rate,
		srv_serror_rate,
		rerror_rate,
		srv_rerror_rate,
		same_srv_rate,
		diff_srv_rate,
		srv_diff_host_rate,
		dst_host_count,
		dst_host_srv_count,
		dst_host_same_srv_rate,
		dst_host_diff_srv_rate,
		dst_host_same_src_port_rate,
		dst_host_srv_diff_host_rate,
		dst_host_serror_rate,
		dst_host_srv_serror_rate,
		dst_host_rerror_rate,
		dst_host_srv_rerror_rate"""
	
	cols = [c.strip() for c in cols.split(",") if c.strip()]
	cols.append('target')

	attacks_type = {
	'normal': 'normal',
	'back': 'dos',
	'buffer_overflow': 'u2r',
	'ftp_write': 'r2l',
	'guess_passwd': 'r2l',
	'imap': 'r2l',
	'ipsweep': 'probe',
	'land': 'dos',
	'loadmodule': 'u2r',
	'multihop': 'r2l',
	'neptune': 'dos',
	'nmap': 'probe',
	'perl': 'u2r',
	'phf': 'r2l',
	'pod': 'dos',
	'portsweep': 'probe',
	'rootkit': 'u2r',
	'satan': 'probe',
	'smurf': 'dos',
	'spy': 'r2l',
	'teardrop': 'dos',
	'warezclient': 'r2l',
	'warezmaster': 'r2l',
		}
		
	df = pd.read_csv(download_data(), names=cols)
	
	df['Attack'] = df.target.apply(lambda r: attacks_type[r[:-1]])
	
	return df

def plotCorrelationMatrix(df, graphWidth, dataframeName):
    filename = dataframeName#df.dataframeName
    df = df.dropna('columns') 
    df = df[[col for col in df if df[col].nunique() > 1]] #Keep columns with more than one unique value
    if df.shape[1] < 2:
        print(f'No correlation plots shown: The number of non-NaN or constant columns ({df.shape[1]}) is less than 2')
        return
    corr = df.corr()
    pyplot.figure(num=None, figsize=(graphWidth, graphWidth), dpi=80, facecolor='w', edgecolor='k')
    corrMat = pyplot.matshow(corr, fignum = 1)
    pyplot.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    pyplot.yticks(range(len(corr.columns)), corr.columns)
    pyplot.gca().xaxis.tick_bottom()
    pyplot.colorbar(corrMat)
    pyplot.title(f'Correlation Matrix for {filename}', fontsize=30)
    pyplot.show()
   
   
def expand_categories(values):
    result = []
    s = values.value_counts()
    t = float(len(values))
    for v in s.index:
        result.append("{}:{}%".format(v,round(100*(s[v]/t),2)))
    return "[{}]".format(", ".join(result))
	

def analyze():
	#Pie chart to show dependency of frequencies 
	df = preprocessing_data()
	print("The data structure is (lines, columns):",df.shape)
	type_frequencies = df['target'].value_counts()
	normal_frequency = type_frequencies['normal.']
	intrusion_frequency = sum([count for outcome_type, count in type_frequencies.iteritems() if outcome_type != 'normal.'])

	figure = pyplot.figure()
	pyplot.pie(
		[normal_frequency, intrusion_frequency],
		labels=["Normal", "Intrusions"],
		explode=[0, .25],
		autopct='%1.1f%%',
		shadow=True,
	)
	pyplot.show()
	
	#********************************************
	
	#Print types of attacks and the % they appear
	cols = df.columns.values
	total = float(len(df))
	
	
	for col in cols:
		if col==cols[-1]:
			uniques = df[col].unique()
			unique_count = len(uniques)
			if unique_count>100:
				print("** {}:{} ({}%)".format(col, unique_count, int(((unique_count)/total)*100)))
			else:
				print("** {}:{} )".format(col, expand_categories(df[col])))
				expand_categories(df[col])

#################################################################################
#																				#
#								Random Forest									#
#																				#
#################################################################################

from sklearn.ensemble import RandomForestClassifier

#*******************************************************

def correlated_features():
	correlated_features = {
    'is_hot_login' : 'is_host_login',
	'urg' : 'urgent',
	'protocol' : 'protocol_type',
	'count_sec' : 'count',
	'srv_count_sec' : 'srv_count',
	'serror_rate_sec' : 'serror_rate',
	'srv_serror_rate_sec' : 'srv_serror_rate',
	'rerror_rate_sec' : 'rerror_rate',
	'srv_error_rate_sec' : 'srv_rerror_rate',
	'same_srv_rate_sec' : 'same_srv_rate',
	'diff_srv_rate_sec' : 'diff_srv_rate',
	'srv_diff_host_rate_sec' : 'srv_diff_host_rate',
	'count_100' : 'dst_host_count',
	'srv_count_100' : 'dst_host_srv_count',
	'same_srv_rate_100' : 'dst_host_same_srv_rate',
	'diff_srv_rate_100' : 'dst_host_diff_srv_rate',
	'same_src_port_rate_100' : 'dst_host_same_src_port_rate',
	'srv_diff_host_rate_100' : 'dst_host_srv_diff_host_rate',
	'serror_rate_100' : 'dst_host_serror_rate',
	'srv_serror_rate_100' : 'dst_host_srv_serror_rate',
	'rerror_rate_100' : 'dst_host_rerror_rate',
	'srv_rerror_rate_100' : 'dst_host_srv_rerror_rate',
	}
	return correlated_features
	
#Removing correlated features
def standardize_columns(df, cols_map=correlated_features()):
	# Delete the 'service' column; if there is a TCPDUMP column, rename it
	if 'service' in df.columns:
		df = df.drop(['service'], axis = 1)
	df.rename(columns = cols_map)

	return df


#******************************************

#Dividing data in training set and test set
def divide_data():
	df = preprocessing_data()
	df = standardize_columns(df, cols_map=correlated_features())
	plotCorrelationMatrix(df, graphWidth=20, dataframeName="Packets") #Correlation Matrix
	df = df.drop(['target',], axis=1)
	
	
	#The previous 41 items are used as input X
	#the Attack column is used as the detection label y
	y = df.Attack
	X = df.drop(['Attack',], axis=1)
	
	#Random generated training and test data
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

	le_X_cols = {}
	le_y = preprocessing.LabelEncoder()

	for c in X_train.columns:
		if str(X_train[c].dtype) == 'object': 
			le_X = preprocessing.LabelEncoder()
			X_train[c] = le_X.fit_transform(X_train[c])
			X_test[c] = le_X.transform(X_test[c])
			le_X_cols[c] = le_X

	y_train = le_y.fit_transform(y_train.values)
	y_test = le_y.transform(y_test.values)

	class_names, class_index = le_y.classes_, np.unique(y_train)
	
	#Feature scaling
	scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
	X_train[['dst_bytes','src_bytes']] = scaler.fit_transform(X_train[['dst_bytes','src_bytes']])
	X_test[['dst_bytes','src_bytes']] = scaler.transform(X_test[['dst_bytes','src_bytes']])
	

	return X, y, X_train, X_test, y_train, y_test, class_names, class_index


def random_forest():
	
	global rf_score, rf_time
	X, y, X_train, X_test, y_train, y_test, class_names, class_index = divide_data()		#get train and test data
	
	clf = RandomForestClassifier(n_estimators=30)
	clf = clf.fit(X_train, y_train)
	fti = clf.feature_importances_
	model = SelectFromModel(clf, prefit=True, threshold= 0.005)
	X_train_new = model.transform(X_train)
	X_test_new = model.transform(X_test)
	selected_features = X_train.columns[model.get_support()]
	
	#Parameter adjustment

	parameters = {
		#'n_estimators'      : [20,40,128,130],
		#'max_depth'         : [None, 14, 15, 17],
		'criterion' :['gini','entropy'],
		'random_state'      : [42],
		#'max_features': ['auto']
	}
	clf = GridSearchCV(RandomForestClassifier(), parameters, cv=2, n_jobs=-1, verbose=3)
	
	
	clf.fit(X_train_new, y_train)
	
	print("Training Accuracy:",clf.best_score_)
	print("Test Accuracy:",clf.score(X_test_new,y_test))
	
	rf_score = clf.score(X_test_new,y_test)

	#Predict test data set
	
	start_time = time.time()
	y_pred = clf.predict(X_test_new)
	end_time = time.time()
	rf_time = end_time-start_time
	
	reversefactor = dict(zip(class_index,class_names))
	y_test_rev = np.vectorize(reversefactor.get)(y_test)
	y_pred_rev = np.vectorize(reversefactor.get)(y_pred)
	#Generate Confusion matrix
	print(pd.crosstab(y_test_rev, y_pred_rev, rownames=['Actual packets attacks'], colnames=['Predicted packets attcks']))


#################################################################################
#																				#
#									NN											#
#																				#
#################################################################################

def encode_numeric_zscore(df, name, mean=None, sd=None):
	if mean is None:
		mean = df[name].mean()

	if sd is None:
		sd = df[name].std()

	df[name] = (df[name] - mean) / sd

def encode_text_dummy(df, name):
	dummies = pd.get_dummies(df[name])
	for x in dummies.columns:
		dummy_name = f"{name}-{x}"
		df[dummy_name] = dummies[x]
	df.drop(name, axis=1, inplace=True)
		
def NN_data():
	df = preprocessing_data()
	df = df.iloc[:, :-1] #removes last column

	df.dropna(inplace=True,axis=1)
	
	encode_numeric_zscore(df, 'duration')
	encode_text_dummy(df, 'protocol_type')
	encode_text_dummy(df, 'service')
	encode_text_dummy(df, 'flag')
	encode_numeric_zscore(df, 'src_bytes')
	encode_numeric_zscore(df, 'dst_bytes')
	encode_text_dummy(df, 'land')
	encode_numeric_zscore(df, 'wrong_fragment')
	encode_numeric_zscore(df, 'urgent')
	encode_numeric_zscore(df, 'hot')
	encode_numeric_zscore(df, 'num_failed_logins')
	encode_text_dummy(df, 'logged_in')
	encode_numeric_zscore(df, 'num_compromised')
	encode_numeric_zscore(df, 'root_shell')
	encode_numeric_zscore(df, 'su_attempted')
	encode_numeric_zscore(df, 'num_root')
	encode_numeric_zscore(df, 'num_file_creations')
	encode_numeric_zscore(df, 'num_shells')
	encode_numeric_zscore(df, 'num_access_files')
	encode_numeric_zscore(df, 'num_outbound_cmds')
	encode_text_dummy(df, 'is_host_login')
	encode_text_dummy(df, 'is_guest_login')
	encode_numeric_zscore(df, 'count')
	encode_numeric_zscore(df, 'srv_count')
	encode_numeric_zscore(df, 'serror_rate')
	encode_numeric_zscore(df, 'srv_serror_rate')
	encode_numeric_zscore(df, 'rerror_rate')
	encode_numeric_zscore(df, 'srv_rerror_rate')
	encode_numeric_zscore(df, 'same_srv_rate')
	encode_numeric_zscore(df, 'diff_srv_rate')
	encode_numeric_zscore(df, 'srv_diff_host_rate')
	encode_numeric_zscore(df, 'dst_host_count')
	encode_numeric_zscore(df, 'dst_host_srv_count')
	encode_numeric_zscore(df, 'dst_host_same_srv_rate')
	encode_numeric_zscore(df, 'dst_host_diff_srv_rate')
	encode_numeric_zscore(df, 'dst_host_same_src_port_rate')
	encode_numeric_zscore(df, 'dst_host_srv_diff_host_rate')
	encode_numeric_zscore(df, 'dst_host_serror_rate')
	encode_numeric_zscore(df, 'dst_host_srv_serror_rate')
	encode_numeric_zscore(df, 'dst_host_rerror_rate')
	encode_numeric_zscore(df, 'dst_host_srv_rerror_rate')
		
	df.dropna(inplace=True,axis=1)
		
	x_columns = df.columns.drop('target')
	x = df[x_columns].values
	dummies = pd.get_dummies(df['target']) # Classification
	outcomes = dummies.columns
	num_classes = len(outcomes)
	y = dummies.values
		
	return x, y
	
def NN():
	global nn_score, nn_time
	
	x, y = NN_data()
	# Create a test/train split.  25% test
	# Split into train/test
	x_train, x_test, y_train, y_test = train_test_split(
			x, y, test_size=0.25, random_state=42)
	
	# Create neural net
	model = Sequential()
	model.add(Dense(10, input_dim=x.shape[1], activation='relu'))
	model.add(Dense(50, input_dim=x.shape[1], activation='relu'))
	model.add(Dense(10, input_dim=x.shape[1], activation='relu'))
	model.add(Dense(1, kernel_initializer='normal'))
	model.add(Dense(y.shape[1], activation='softmax'))
	model.compile(loss='categorical_crossentropy', optimizer='adam')
	monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, 
							patience=5, verbose=1, mode='auto',
							   restore_best_weights=True)
	model.fit(x_train,y_train,validation_data=(x_test,y_test),
			  callbacks=[monitor],verbose=2,epochs=1000)
	
	


	# Measure accuracy
	start_time = time.time()
	pred = model.predict(x_test)
	pred = np.argmax(pred,axis=1)
	y_eval = np.argmax(y_test,axis=1)
	end_time = time.time()
	nn_time = end_time-start_time
	nn_score = metrics.accuracy_score(y_eval, pred)
	print("Accuracy: {}".format(nn_score))

#***********************************************************************************

def compare():
	random_forest()
	NN()
	
	labels = ['NN', 'RF']
	values1 = [nn_score, rf_score]
	x = np.arange(len(labels))  # the label locations
	fig, (ax1, ax2) = pyplot.subplots(1, 2)
	rects1 = ax1.bar(x, values1, width=0.3)
	ax1.set_ylabel('Accuracy')
	ax1.set_title('Testing Accuracy')
	ax1.set_xticks(x)
	ax1.set_xticklabels(labels)

		
	values2 = [nn_time, rf_time]
	rects2 = ax2.bar(x, values2, width=0.3)
	ax2.set_ylabel('Time')
	ax2.set_title('Testing Time')
	ax2.set_xticks(x)
	ax2.set_xticklabels(labels)

	ax1.plot()
	ax2.plot()
	fig.tight_layout()
	pyplot.show()
	

analyze()
compare()
