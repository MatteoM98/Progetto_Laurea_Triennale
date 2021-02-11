from scipy.io import arff
from sklearn.covariance import EllipticEnvelope
from sklearn.metrics import accuracy_score
import sys
import time
import pandas as pd 
import numpy as np
import modules as md
import datetime



train_path = str(sys.argv[1])
test_path = str(sys.argv[2])
N = int(sys.argv[3])

print('Recupero dati in corso...')
t0 = time.time()
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

#variabili per calcolo della media
lista_percentuale_anomalie_cls_correttamente = []
lista_percentuale_anomalie_cls_erroneamente = []
lista_percentuale_normali_cls_erroneamente = []
lista_percentuale_accuratezza = []

i=1
while i<=N:
    print('Inizio addestramento.... ',i)
    outliers_fraction = 0.15 #soglia
    algorithm=EllipticEnvelope(contamination=outliers_fraction)
    clf = algorithm.fit(train_data_list)

    print('Classificazione delle istanze di test....  ',i)
    results = clf.predict(test_data_list)

    #calcolo accuratezza
    acc = []
    for k in test_data_list:
        if not isinstance(k[8],float):
            acc.append(k[8]) #recupero tipo della classe 1 o -1
        else:
            acc.append(k[10])
    score = accuracy_score(acc,results)


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

    percentuale_anomalie_cls_correttamente = (numero_anomalie_classificate_correttamente/numero_istanze_anomalie)*100
    percentuale_anomalie_cls_erroneamente = (numero_anomalie_classificate_erroneamente/numero_istanze_anomalie)*100
    percentuale_normali_cls_erroneamente = (numero_normali_classificate_erroneamente/numero_istanze_normali)*100


    lista_percentuale_anomalie_cls_correttamente.append(percentuale_anomalie_cls_correttamente)
    lista_percentuale_anomalie_cls_erroneamente.append(percentuale_anomalie_cls_erroneamente)
    lista_percentuale_normali_cls_erroneamente.append(percentuale_normali_cls_erroneamente)
    lista_percentuale_accuratezza.append(score)

    i+=1


print('Creazione file di output...')
name_file = str(int(datetime.datetime.now().timestamp())) + '_results'
f = open(name_file,'w')

print(f'****** Risultati dopo {N} iterazioni ******',file=f)
print('Istanze totali nel training set: ',tot_ist_train_data,file=f)
print('Istanze totali nel test: ',tot_ist_test_data,file=f)
print('Istanze normali nel test: ',numero_istanze_normali,file=f)
print('Istanze anomalie nel test: ',numero_istanze_anomalie,file=f)
print('percentuali di anomalie classificate correttamente: ', lista_percentuale_anomalie_cls_correttamente,file=f)
print('media percentuale di anomalie classificate correttamente: ', format(sum(lista_percentuale_anomalie_cls_correttamente)/len(lista_percentuale_anomalie_cls_correttamente),'.2f'),file=f)
print('percentuali di anomalie classificate come normali (FN): ', lista_percentuale_anomalie_cls_erroneamente,file=f)
print('media percentuale di anomalie classificate come normali (FN): ',format(sum(lista_percentuale_anomalie_cls_erroneamente)/len(lista_percentuale_anomalie_cls_erroneamente),'.2f'),file=f)
print('percentuali di istanze normali classificate come anormali (FP): ', lista_percentuale_normali_cls_erroneamente,file=f)
print('media percentuale di istanze normali classificate come anormali (FP): ',format(sum(lista_percentuale_normali_cls_erroneamente)/len(lista_percentuale_normali_cls_erroneamente),'.2f'),file=f)
print('percentuali accuratezza media dell\'algoritmo: ', lista_percentuale_accuratezza,file=f)
print('media percentuale accuratezza dell\'algoritmo: ', format(sum(lista_percentuale_accuratezza)/len(lista_percentuale_accuratezza)*100,'.2f'),file=f)

f.close()

print(f'File di output {name_file} creato con successo!!')
t1 = time.time()
print('Tempo totale impiegato: ', format(t1-t0,".2f"), "sec")
