import sys

mapping_dict = {
    'OTH': 0,
    'REJ': 1,
    'RSTO': 2,
    'RSTOS0': 3,
    'RSTR': 4,
    'S0': 5,
    'S1': 6,
    'S2': 7,
    'S3': 8,
    'SF': 9,
    'SH': 10,
    '1':1,
    '0':0,
    'normal':1,
    '?':-1
}

def replaceValue(strl):
    try:
        return mapping_dict[strl]
    except:
        print('Errore. chiave ' + str(strl) +  ' non presente')
        return

def convertType(l):
    i=0
    for elem in l:
        elem_list = list(elem)
        elem_list[0] = replaceValue(elem_list[0].decode('ascii'))
        elem_list[3] = replaceValue(elem_list[3].decode('ascii'))
        #per distinguere i due diversi dataset degli esperimenti
        if not isinstance(elem_list[8],float):
            elem_list[8] = replaceValue(elem_list[8].decode('ascii'))
        else:
           elem_list[10] = replaceValue(elem_list[10].decode('ascii'))
        l[i] = elem_list
        i+=1