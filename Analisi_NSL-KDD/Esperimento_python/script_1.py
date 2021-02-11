from scipy.io import arff
from sklearn.svm import OneClassSVM
from sklearn.metrics import accuracy_score
import sys
import pandas as pd 
import numpy as np
import modules as md
import datetime



train_path = str(sys.argv[1])
test_path = str(sys.argv[2])

print('Recupero dati in corso...')
train_dataset = arff.loadarff(train_path)
train_data = np.array(train_dataset[0]) #training set 
tot_ist_train_data = train_data.size

test_dataset = arff.loadarff(test_path)
test_data = np.array(test_dataset[0]) #test set
tot_ist_test_data = test_data.size


# è necessario convertire i tipi string
# in numeric per l' addestramento
print('Conversione dei dati in corso...')
train_data_list = train_data.tolist()
test_data_list = test_data.tolist()
md.convertType(train_data_list)
md.convertType(test_data_list)


print('Inizio addestramento....')
clf = OneClassSVM(gamma='auto').fit(train_data_list)


print('Classificazione delle istanze di test....')
results = clf.predict(test_data_list)

#accuratezza
acc = []
for k in test_data_list:
    if not isinstance(k[8],float):
        acc.append(k[8]) #recupero tipo della classe 1 o -1
    else:
        acc.append(k[10])

score = accuracy_score(acc,results)

print('Elaborazione dei dati in corso...')
numero_anomalie_classificate_correttamente = 0
numero_normali_classificate_correttamente = 0
numero_anomalie_classificate_erroneamente = 0
numero_normali_classificate_erroneamente = 0
numero_istanze_normali = 0
numero_istanze_anomalie = 0
for k,res in zip(test_data_list,results):
    if not isinstance(k[8],float):
        exp = k[8] #recupero tipo della classe 1 o -1
    else:
        exp = k[10]
    
    if exp == -1 and exp == res:
        numero_istanze_anomalie += 1
        numero_anomalie_classificate_correttamente += 1
    elif exp == -1 and exp != res:
        numero_istanze_anomalie += 1
        numero_anomalie_classificate_erroneamente += 1
    elif exp == 1 and exp == res:
        numero_istanze_normali += 1
        numero_normali_classificate_correttamente += 1
    elif exp == 1 and exp != res:
        numero_istanze_normali += 1
        numero_normali_classificate_erroneamente += 1

assert numero_istanze_normali + numero_istanze_anomalie == tot_ist_test_data, 'Numero delle istanze incongruente!!'
assert numero_anomalie_classificate_correttamente + numero_anomalie_classificate_erroneamente == numero_istanze_anomalie, 'Numero delle anomalie incongruente!!!'
assert numero_normali_classificate_correttamente + numero_normali_classificate_erroneamente == numero_istanze_normali, 'Numero delle istanze normali incongruente!!!'


print('Creazione file di output...')
name_file = str(int(datetime.datetime.now().timestamp())) + '_results'
f = open(name_file,'w')

print('\n****** Risultati ******',file=f)
print('Istanze totali nel training set: ',tot_ist_train_data,file=f)
print('Istanze totali nel test: ',tot_ist_test_data,file=f)
print('Istanze normali nel test: ',numero_istanze_normali,file=f)
print('Istanze anomalie nel test: ',numero_istanze_anomalie,file=f)
print('percentuale di anomalie classificate correttamente: ', format((numero_anomalie_classificate_correttamente/numero_istanze_anomalie)*100,'.2f'),file=f)
print('percentuale di anomalie classificate come normali (FN): ',format((numero_anomalie_classificate_erroneamente/numero_istanze_anomalie)*100,'.2f'),file=f)
print('percentuale di istanze normali classificate come anormali (FP): ', format((numero_normali_classificate_erroneamente/numero_istanze_normali)*100,'.2f'),file=f)
print('percentuale accuratezza generale dell\'algoritmo: ', format(score*100,'.2f'),file=f)

f.close()

print(f'File di output {name_file} creato con successo!!')