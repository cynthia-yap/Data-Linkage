import csv
import numpy as np
import pandas as pd
def blocking(price):
    label='o'
    if price==0:
        label='ok'
    if 0<price<=20:
        label='ok'
    if 20<=price<=30:
        label='lplpl'
    if 30<=price<50:
        label='odk'
    
    if 50<=price<60:
        label='li'
    if 60<=price<80:
        label='oik'
    
    if 80<=price<100:
        label='lidd'
    if 100<=price<300:
        label='ofek'
    
    if 300<=price<600:
        label='lide'
    if 400<=price<900:
        label='okww'
    
    if 900<=price<1100:
        label='liee'
    if 1200<=price<=3300:
        label='fre'
    if 33000<=price<=7000:
        label='lolff'
    if price>60000:
        label='ok'
    if price>3300:
        label='asaln'
    return label
        
        
    
        
labels=[]
link=[]
start=0
for k in list(open("google.csv")):
    idgoogle=(k.strip('\n').split(',')[0])
    price=(k.strip('\n').split(',')[4])

    if start==0:       
            start+=1
    else:
        if 'gbp' in price:
            price=price.replace('gbp','')
        price=float(price)
        
        link.append(idgoogle)
        labels.append(blocking(price))
            
            
data={'block_key':labels,'product_id': link}

articles1= pd.DataFrame.from_dict(data)
articles1.to_csv('google_blocks.csv', index = False)  
#print(labels)    

labels=[]
link=[]
start=0
for k in list(open("amazon.csv")):
    idamazon=(k.strip('\n').split(',')[0])
    price=(k.strip('\n').split(',')[4])

    if start==0:       
            start+=1
    else:
        if 'gbp' in price:
            price=price.replace('gbp','')
        if 'summit software'  not in price:
            
            price=float(price)
          
            link.append(idamazon)
            labels.append(blocking(price))
        else:
            link.append(idamazon)
            labels.append('jiji')


data2={'block_key':labels, 'product_id': link}
articles2= pd.DataFrame.from_dict(data2)
articles2.to_csv('amazon_blocks.csv', index = False)  
