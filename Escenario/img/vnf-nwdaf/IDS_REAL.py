#!/usr/bin/env python
# coding: utf-8

# ## **Read in Raw KDD-99 Dataset Unlabeled**

# In[64]:
from silence_tensorflow import silence_tensorflow
silence_tensorflow()
import time
import tensorflow as tf
from keras import backend as K

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# In[65]:
import pandas as pd

data = ['OBU_data2.csv', 'OBU_data4.csv', 'OBU_data4.csv', 'OBU_data1.csv', 'OBU_data3.csv', 'OBU_data4.csv']
OBU = 0

for path in data:
    print('\n')
    print("--Waiting for incoming packets")
    time.sleep(3)
    OBU += 1
    
    if (OBU % 2) != 0:
        print("--Receiving packets from the 1st OBU in the 5G Vehicular Network...")
    else:
        print("--Receiving packets from the 2nd OBU in the 5G Vehicular Network...")
    
    # This file is a CSV, just no CSV extension or headers
    # Download from: http://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html
    df_nolabel = pd.read_csv(path)

    print("Read 20 packets")

    print("Cleaning and preprocessing the data")
    # df = df.sample(frac=0.1, replace=False) # Uncomment this line to sample only 10% of the dataset
    df_nolabel.dropna(inplace=True,axis=1) # For now, just drop NA's (rows with missing values)

    # The CSV file has no column heads, so add them

    df_nolabel.columns = [
        'duration',
        'protocol_type',
        'service',
        'flag',
        'src_bytes',
        'dst_bytes',
        'land',
        'wrong_fragment',
        'urgent',
        'hot',
        'num_failed_logins',
        'logged_in',
        'num_compromised',
        'root_shell',
        'su_attempted',
        'num_root',
        'num_file_creations',
        'num_shells',
        'num_access_files',
        'num_outbound_cmds',
        'is_host_login',
        'is_guest_login',
        'count',
        'srv_count',
        'serror_rate',
        'srv_serror_rate',
        'rerror_rate',
        'srv_rerror_rate',
        'same_srv_rate',
        'diff_srv_rate',
        'srv_diff_host_rate',
        'dst_host_count',
        'dst_host_srv_count',
        'dst_host_same_srv_rate',
        'dst_host_diff_srv_rate',
        'dst_host_same_src_port_rate',
        'dst_host_srv_diff_host_rate',
        'dst_host_serror_rate',
        'dst_host_srv_serror_rate',
        'dst_host_rerror_rate',
        'dst_host_srv_rerror_rate'
    ]

    #Adding dummy columns
    dummy_columns = [
        'protocol_type-icmp',
        'protocol_type-tcp',
        'protocol_type-udp', 
        'service-IRC',
        'service-X11',
        'service-Z39_50',
        'service-auth',
        'service-bgp',
        'service-courier',
        'service-csnet_ns',
        'service-ctf',
        'service-daytime',
        'service-discard',
        'service-domain',
        'service-domain_u',
        'service-echo',
        'service-eco_i',
        'service-ecr_i',
        'service-efs',
        'service-exec',
        'service-finger',
        'service-ftp',
        'service-ftp_data',
        'service-gopher',
        'service-hostnames',
        'service-http',
        'service-http_443',
        'service-imap4',
        'service-iso_tsap',
        'service-klogin',
        'service-kshell',
        'service-ldap',
        'service-link',
        'service-login',
        'service-mtp',
        'service-name',
        'service-netbios_dgm',
        'service-netbios_ns',
        'service-netbios_ssn',
        'service-netstat',
        'service-nnsp',
        'service-nntp',
        'service-ntp_u',
        'service-other',
        'service-pm_dump',
        'service-pop_2',
        'service-pop_3',
        'service-printer',
        'service-private',
        'service-red_i',
        'service-remote_job',
        'service-rje',
        'service-shell',
        'service-smtp',
        'service-sql_net',
        'service-ssh',
        'service-sunrpc',
        'service-supdup',
        'service-systat',
        'service-telnet',
        'service-tftp_u',
        'service-tim_i',
        'service-time',
        'service-urh_i',
        'service-urp_i',
        'service-uucp',
        'service-uucp_path',
        'service-vmnet',
        'service-whois',
        'flag-OTH',
        'flag-REJ',
        'flag-RSTO',
        'flag-RSTOS0',
        'flag-RSTR',
        'flag-S0',
        'flag-S1',
        'flag-S2',
        'flag-S3',
        'flag-SF',
        'flag-SH',
        'land-0',
        'land-1',
        'logged_in-0',
        'logged_in-1',
        'is_host_login-0',
        'is_guest_login-0',
        'is_guest_login-1'
    ]

    for dummy_col in dummy_columns:
        df_nolabel[dummy_col] = 0

    # display 5 rows
    #print(df_nolabel[0:5])


    # ## **Encode the feature vector**

    # In[66]:


    # Analyze KDD-99

    import os
    import numpy as np
    from sklearn import metrics
    from scipy.stats import zscore

    # Encode a numeric column as zscores
    def encode_numeric_zscore(df, name, mean=None, sd=None):
        if mean is None:
            mean = df[name].mean()

        if sd is None:
            sd = df[name].std()

        df[name] = (df[name] - mean) / sd

    # Encode text values to dummy variables(i.e. [1,0,0],[0,1,0],[0,0,1] for red,green,blue)
    def encode_text_dummy(df, name):
        dummies = pd.get_dummies(df[name])
        for x in dummies.columns:
            dummy_name = "%s-%s" % (name,x)
            df[dummy_name] = dummies[x]
        df.drop(name, axis=1, inplace=True)


    # In[67]:

    # Now encode the feature vector

    encode_numeric_zscore(df_nolabel, 'duration')
    encode_text_dummy(df_nolabel, 'protocol_type')
    encode_text_dummy(df_nolabel, 'service')
    encode_text_dummy(df_nolabel, 'flag')
    encode_numeric_zscore(df_nolabel, 'src_bytes')
    encode_numeric_zscore(df_nolabel, 'dst_bytes')
    encode_text_dummy(df_nolabel, 'land')
    encode_numeric_zscore(df_nolabel, 'wrong_fragment')
    encode_numeric_zscore(df_nolabel, 'urgent')
    encode_numeric_zscore(df_nolabel, 'hot')
    encode_numeric_zscore(df_nolabel, 'num_failed_logins')
    encode_text_dummy(df_nolabel, 'logged_in')
    encode_numeric_zscore(df_nolabel, 'num_compromised')
    encode_numeric_zscore(df_nolabel, 'root_shell')
    encode_numeric_zscore(df_nolabel, 'su_attempted')
    encode_numeric_zscore(df_nolabel, 'num_root')
    encode_numeric_zscore(df_nolabel, 'num_file_creations')
    encode_numeric_zscore(df_nolabel, 'num_shells')
    encode_numeric_zscore(df_nolabel, 'num_access_files')
    encode_numeric_zscore(df_nolabel, 'num_outbound_cmds')
    encode_text_dummy(df_nolabel, 'is_host_login')
    encode_text_dummy(df_nolabel, 'is_guest_login')
    encode_numeric_zscore(df_nolabel, 'count')
    encode_numeric_zscore(df_nolabel, 'srv_count')
    encode_numeric_zscore(df_nolabel, 'serror_rate')
    encode_numeric_zscore(df_nolabel, 'srv_serror_rate')
    encode_numeric_zscore(df_nolabel, 'rerror_rate')
    encode_numeric_zscore(df_nolabel, 'srv_rerror_rate')
    encode_numeric_zscore(df_nolabel, 'same_srv_rate')
    encode_numeric_zscore(df_nolabel, 'diff_srv_rate')
    encode_numeric_zscore(df_nolabel, 'srv_diff_host_rate')
    encode_numeric_zscore(df_nolabel, 'dst_host_count')
    encode_numeric_zscore(df_nolabel, 'dst_host_srv_count')
    encode_numeric_zscore(df_nolabel, 'dst_host_same_srv_rate')
    encode_numeric_zscore(df_nolabel, 'dst_host_diff_srv_rate')
    encode_numeric_zscore(df_nolabel, 'dst_host_same_src_port_rate')
    encode_numeric_zscore(df_nolabel, 'dst_host_srv_diff_host_rate')
    encode_numeric_zscore(df_nolabel, 'dst_host_serror_rate')
    encode_numeric_zscore(df_nolabel, 'dst_host_srv_serror_rate')
    encode_numeric_zscore(df_nolabel, 'dst_host_rerror_rate')
    encode_numeric_zscore(df_nolabel, 'dst_host_srv_rerror_rate')

    # display 5 rows
    #df_nolabel.dropna(inplace=True,axis=1)
    df_nolabel = df_nolabel.fillna(0)
    df_nolabel.drop('num_outbound_cmds', inplace=True, axis=1)
    '''
    for col in df_nolabel.columns:
        print(col)
    '''

    # In[68]:

    # Me quedo con el 90% del dataframe de forma aleatoria así en todo momento son paquetes distintos en cada ejecución
    df_nolabel = df_nolabel.sample(frac = 10/11)
    #print(df_nolabel[0:5])
    #print(df_nolabel.shape)


    # In[69]:

    # Loading the model with the custom metrics

    def precision_m(y_true, y_pred):
            true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
            predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
            precision = true_positives / (predicted_positives + K.epsilon())
            return precision

    def recall_m(y_true, y_pred):
            true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
            possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
            recall = true_positives / (possible_positives + K.epsilon())
            return recall

    def f1_m(y_true, y_pred):
            precision = precision_m(y_true, y_pred)
            recall = recall_m(y_true, y_pred)
            return 2*((precision*recall)/(precision+recall+K.epsilon()))

    model = tf.keras.models.load_model("model.h5", compile = False)

    model.compile(optimizer='adam',loss='categorical_crossentropy', metrics=['accuracy', precision_m, recall_m, f1_m])

    print("Model loaded and compiled")


    # In[70]:

    print("Making the prediction...")
    y_p = model.predict(df_nolabel)

    # Grouping in the 5 main categories in the KDD-99 dataset
    # 0 Normal (no attack)
    y_p0 = np.delete(y_p, [0,1,2,3,4,5,6,7,8,9,10,12,13,14,15,16,17,18,19,20,21,22], axis=1)
    y_p0s = np.sum(y_p0, axis=1)
    y_p0r= y_p0s.reshape(y_p0s.shape[0], 1)

    # 1 Probe
    y_p1 = np.delete(y_p, [0,1,2,3,4,6,7,8,9,11,12,13,14,16,18,19,20,21,22], axis=1)
    y_p1s = np.sum(y_p1, axis=1)
    y_p1r= y_p1s.reshape(y_p1s.shape[0], 1)

    # 2 DoS
    y_p2 = np.delete(y_p, [1,2,3,4,5,7,8,10,11,12,13,15,16,17,19,21,22], axis=1)
    y_p2s = np.sum(y_p2, axis=1)
    y_p2r= y_p2s.reshape(y_p2s.shape[0], 1)

    # 3 U2R
    y_p3 = np.delete(y_p, [0,2,3,4,5,6,8,9,10,11,13,14,15,17,18,19,20,21,22], axis=1)
    y_p3s = np.sum(y_p3, axis=1)
    y_p3r= y_p2s.reshape(y_p3s.shape[0], 1)

    # 4 R2L
    y_p4 = np.delete(y_p, [0,1,5,6,7,9,10,11,12,14,15,16,17,18,20], axis=1)
    y_p4s = np.sum(y_p4, axis=1)
    y_p4r= y_p4s.reshape(y_p4s.shape[0], 1)

    y_predT = np.concatenate((y_p0r, y_p1r, y_p2r, y_p3r, y_p4r), axis=1)
    y_pred2 = np.argmax(y_predT,axis=1) #Return a 1D binary vector indicating the ID of the max number in the row


    # In[71]:

    from collections import Counter

    resultados = Counter(y_pred2)

    ataques = resultados[1] + resultados[2] + resultados[3] + resultados[4]

    if ataques < 1:
        print("  There is no anomalies in the packets received")
    else:
        print("  The IDS has detected " + str(resultados[0]) + " Normal traffic packets and " + str(ataques) + " malicious packets (anomalies)")

        print("  The following types of attacks have been detected: ")
        print("  " + str(resultados[1]) + " Probe attacks")
        print("  " + str(resultados[2]) + " DoS attacks")
        print("  " + str(resultados[3]) + " U2R attacks")
        print("  " + str(resultados[4]) + " R2L attacks")
    
    if OBU == 6:
        print('\n')
        print("--Stopping the IDS...")