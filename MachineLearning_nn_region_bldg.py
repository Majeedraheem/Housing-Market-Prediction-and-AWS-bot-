#!/usr/bin/env python
# coding: utf-8

# In[1]:


# importing libraries
import numpy as np
import pandas as pd
from pathlib import Path
import hvplot.pandas
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import time
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.metrics import MeanSquaredError
from pandas.tseries.offsets import DateOffset
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split


# In[2]:


all_values = pd.read_csv(Path("Resources/all_values_superset.csv"))
all_values['date'] = pd.to_datetime(all_values['date'], format='%Y-%m')
all_values.set_index('date', inplace=True)
# all_values.drop(columns=['All-items 8', 'All-items excluding food', 'All-items excluding food and energy', 'Clothing and footwear'], inplace=True)
all_values.drop(columns=['All-items 8', 'All-items excluding food', 'All-items excluding food and energy'], inplace=True)
all_values.dtypes


# In[3]:


scaler_all = StandardScaler()
all_scaler = scaler_all.fit(all_values)
all_values_scaled = all_scaler.transform(all_values)

all_values_scaled_df = pd.DataFrame(all_values_scaled, columns=all_values.columns, index=all_values.index)
all_values_scaled_df.head()


# In[4]:


targets = [col for col in all_values_scaled_df.columns.tolist() if col.find('Benchmark') > 0]


# In[5]:


y = all_values_scaled_df[targets]


# In[6]:


X = all_values_scaled_df.drop(columns = targets)


# In[7]:


y.shape


# In[8]:


y.head()


# In[9]:


X.head()


# In[10]:


all_values_scaled_df.hvplot(
    height=800,
    width=1600
)


# In[11]:


X_train, X_test, y_train, y_test = train_test_split(X, y)
X_train
X_test.sort_index(ascending=True, inplace=True)
y_train
y_test.sort_index(ascending=True, inplace=True)
y_test_size = y_test.shape[0]
y_test_size


# In[12]:


# scaler = StandardScaler()

# X_scaler = scaler.fit(X_train)
# X_train_scaled = X_scaler.transform(X_train)
# X_test_scaled = X_scaler.transform(X_test)


# In[13]:

prev_test_r2 = 0
# Neural network parameters
n_input_feats = len(X.columns)

n_outputs = 161

nodes = [16, 32, 64, 128, 256, 512, 1024]

for node1 in nodes:
    for node2 in nodes:
        for node3 in nodes:
            hidden_nodes_layer1 = node1
            hidden_nodes_layer2 = node2
            hidden_nodes_layer3 = node3

           # In[14]:


            # define model
            nn = Sequential()


            # In[15]:


            # Add the first hidden layer
            nn.add(Dense(
                units=hidden_nodes_layer1,
                input_dim = n_input_feats,
                activation='relu',
                name='hidden1'
            ))


            # In[16]:


            # Add the second hidden layer
            nn.add(Dense(
                units=hidden_nodes_layer2,
                activation='relu',
                name='hidden2'
            ))


            # In[17]:


            # Add the third hidden layer
            nn.add(Dense(
                units=hidden_nodes_layer3,
                activation='relu',
                name='hidden3'
            ))


            # In[18]:


            # Add output
            nn.add(Dense(
                n_outputs,
                activation='linear',
                name='output'
            ))


            # In[19]:


            # Display the Sequential model summary
            nn.summary()


            # In[20]:


            # Compile the Sequential model
            nn.compile(loss='mean_squared_error', optimizer='adam', metrics=[MeanSquaredError()])


            # In[21]:


            # Fit the model using 50 epochs and the training data
            model = nn.fit(
                X_train,
                y_train,
                # batch_size=11,
                validation_split=0.2,
                epochs=100
            )
            predictions = nn.predict(X_test)
            # R2 score from test
            test_r2 = r2_score(y_test.values, predictions)
            if test_r2 > prev_test_r2:
                nn.save(Path(f"./Models/trial3/nn_in{n_input_feats}_relu{hidden_nodes_layer1}_relu{hidden_nodes_layer2}_relu{hidden_nodes_layer3}_out{n_outputs}_r2score{test_r2}.h5"))
                prev_test_r2 = test_r2
            
                print(f"""
                Best Neural Network Parameters:
                -------------------------------
                Input Features: {n_input_feats}
                Layers:         3
                Layer1 Nodes:   {hidden_nodes_layer1}
                Layer2 Nodes:   {hidden_nodes_layer2}
                Layer3 Nodes:   {hidden_nodes_layer3}
                Output Nodes:   {n_outputs}
                R2 Score:       {test_r2}
                """)

# # In[22]:


# # R2 score for training
# train_pred = nn.predict(X_train)
# training_r2 = r2_score(y_train.values, train_pred)
# # model_loss, model_accuracy = nn.evaluate(X_test, y_test)
# training_r2


# # In[23]:


# # Create a DataFrame with the history dictionary
# model_df = pd.DataFrame(model.history, index=range(1, len(model.history["loss"]) + 1))


# # In[24]:


# # Plot the loss
# model_df.hvplot(
#     y="val_loss",
    
# )


# # In[25]:


# # Plot the accuracy
# model_df.hvplot(y="mean_squared_error")


# # In[26]:


# model_df[['val_loss', 'loss']].hvplot(

# )


# # In[27]:


# predictions = nn.predict(X_test)
# # predictions = nn.predict(X_test.sort_index(ascending=True))


# # In[28]:


# # R2 score from test
# test_r2 = r2_score(y_test.values, predictions)
# # test_r2 = r2_score(y_test.sort_index(ascending=True).values, predictions)
# test_r2


# # In[29]:


# all_values_pred = nn.predict(X)
# all_values_pred.shape


# # In[30]:


# all_feats_pred = nn.predict(X)
# all_feats_pred_df = pd.DataFrame(all_feats_pred, columns=y.columns, index=X.index)
# all_preds_feats_scaled_df = pd.concat([all_feats_pred_df, X], axis=1)
# all_preds_feats_descaled = all_scaler.inverse_transform(all_preds_feats_scaled_df)
# all_preds_feats_descaled_df = pd.DataFrame(all_preds_feats_descaled, columns=all_preds_feats_scaled_df.columns, index=all_preds_feats_scaled_df.index)
# all_preds_descaled_df = all_preds_feats_descaled_df[targets]
# all_preds_descaled_df.columns = ['PRED_'+col for col in all_preds_descaled_df.columns]
# all_preds_descaled_df
# # all_preds_feats_descaled_df
# # all_pred_real = all_values_scaled_df.copy()
# # all_pred_real['Composite'] = all_values_pred
# # all_values_descaled = all_scaler.inverse_transform(all_pred_real)
# # all_values_descaled_df = pd.DataFrame(all_values_descaled, columns=all_values_scaled_df.columns, index=all_values_scaled_df.index)
# # all_values_descaled_df
# # all_values_descaled_df['Predicted'] = all_values_descaled_df['Composite']
# # all_values_descaled_df['Composite'] = all_values['Composite']
# # all_values_descaled_df


# # In[31]:


# bancroft_pred_compare_df = pd.concat([all_preds_descaled_df.loc[:,'PRED_Composite_Benchmark_SA_BANCROFT_AND_AREA'], all_values.loc[:,'Composite_Benchmark_SA_BANCROFT_AND_AREA']], axis=1)
# bancroft_pred_compare_df.hvplot(
#     height=800,
#     width=1600
# )


# # In[37]:


# oakville_pred_compare_df = pd.concat([all_preds_descaled_df.loc[:,'PRED_Two_Storey_Benchmark_SA_OAKVILLE_MILTON'], all_values.loc[:,'Two_Storey_Benchmark_SA_OAKVILLE_MILTON']], axis=1)
# oakville_pred_compare_df.hvplot(
#     height=800,
#     width=1600
# )


# # In[32]:


# all_targets_pred_compare_df = pd.concat([all_preds_descaled_df, all_values[targets]], axis=1)
# all_targets_pred_compare_df.hvplot(
#     height=800,
#     width=1600
# )


# # In[38]:


# nn.save(Path('./Models/nn_relu128_relu256_relu256_outRelu_region_bldg.h5'))


# # In[ ]:




