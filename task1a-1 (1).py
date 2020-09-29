import numpy as np
import textdistance
import pandas as pd
import nltk
from nltk.stem.porter import *
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
nltk.download('wordnet')
# if running the first time with errors:
nltk.download('punkt')
nltk.download('stopwords')
porterStemmer = PorterStemmer()
# run the line to download it the first time:
stopWords = set(stopwords.words('english'))
levenshtein={}
levenshtein_normalize=[]
probably_no={}
baseid={}
considered={}
le={}
prob_no={}
length={}
googlelastid=[]
def stemsentence(sentence):
    wordList= nltk.word_tokenize(sentence)
    filteredList = [w for w in wordList if not w in stopWords]
    stem_sentence=[]
    for word in wordList:
        stem_sentence.append(porterStemmer.stem(word))
        stem_sentence.append(" ")
    return "".join(stem_sentence)

def rp(str1):
    bad_chars = [';', ':', '-','!', "*",")","("] 
    for i in bad_chars : 
        str1 = str1.replace(i, ' ') 
    a=stemsentence(str1).split()
    return a

def remove_punct(str1, str2):
    bad_chars = [';', ':', '-','!', "*",")","("] 
    for i in bad_chars : 
        str1 = str1.replace(i, ' ') 
        str2 = str2.replace(i, ' ') 
    a=stemsentence(str1).split()
    b=stemsentence(str2).split()
    intersection=set(a).intersection(set(b))
    return intersection

def findsim(google,amazon):
    leven=0
    ok=0
    google=rp(google)
    google=' '.join(word for word in google)
    amazon=rp(amazon)
    amazon=' '.join(word for word in amazon)
    
        
    same=list(set(google.split()).intersection(set(amazon.split())))  
    similar=len(same)
    nonsame=[n  for n in google.split() if n not in same]
    
    if nonsame==0:
        return similarity
    else:
        for word in nonsame:
            for A in (amazon.split()):               
                if textdistance.levenshtein.normalized_similarity(word,A)>0.8   and (not A[0].isdigit() or len(A)>2):
                    same.append(word)                  
                    levenshtein_normalize.append(word)
                    similar+=1
                    leven=1  
                    break     
    if leven==1 and len(same)==1:
        levenshtein[google]=leven
    else:
        levenshtein[google]=0
    
    if similar-len(levenshtein_normalize)<3:
        count_digit=0
        
        for word in levenshtein_normalize:
            if word.isdigit():
                count_digit+=1           
            if len(levenshtein_normalize)-count_digit<=2:
                probably_no[google]=1
            else:
                probably_no[google]=0
    probably_no[google]=0
    return same

df2=pd.read_csv("google_small.csv")

def diction(amazon, description_of_amazon ):
    
    start=0
    if description_of_amazon!="":
        intersection_of_amazon=remove_punct(amazon,description_of_amazon)
    
    for google_name in df2['name']:
        a=findsim( google_name, amazon)
        distance=len(findsim( google_name, amazon))/len(rp(amazon))
        googleid=df2['idGoogleBase'][start]   
        baseid[googleid]=str(distance)
        length[googleid]=len((' '.join(word for word in rp(google_name))).split())
        same_only_leven=(' '.join(word for word in rp(google_name)))     
        if same_only_leven in levenshtein and levenshtein[same_only_leven]==1:      
            le[googleid]=1
        else:
            le[googleid]=0
       
        if same_only_leven in probably_no and probably_no[same_only_leven]==1:
            
            prob_no[googleid]=1
        else:
            prob_no[googleid]=0
      
        if len(a)==1 :            
            for char in set(a):
                counting=(len(char))
            if len(char)<=2:
                
                considered[googleid]='yes'
            
            else:
                considered[googleid]='no'
        else:
            considered[googleid]='no'
        start+=1

    return baseid

df1 = pd.read_csv("amazon_small.csv")
title_amazon,description_amazon=[],[]
now=0
id_amazon=[]
googleidlast=[]
task1adict={}
for i in list(open("amazon_small.csv")):
    idamazon=(i.strip('\n').split(',')[0])
    title=(i.strip('\n').split(',')[1])        
    if (now==0):
        now+=1
    else:
       
        id_amazon.append(idamazon)
        description=(i.strip('\n').split(',')[2])        
        a=diction(title,description)

        sort=sorted(baseid.items(), key=lambda kv: (kv[1], kv[0]))
        last_percentage=sort[-1][-1]
        last_googleid=sort[-1][0]
        found=0
      
        count=0
        same_value=[a for a in sort if a[-1]==last_percentage]
        no_leven_id=[]
        newdict={}
        if len(same_value)==1:           
            find=1            
            googlelastid.append(same_value[0][0])
           
        else :
            leven_value=[a for a,b in le.items() if b==1]
            no_leven_id=[value[0] for value in same_value if value[0] not in leven_value]            
            for value in same_value:
            
                if considered[value[0]]=='no':

                    if len(no_leven_id)!=0:

                        no_leven=1
                        
                        for web in no_leven_id:
                            newdict[web]=length[web]
                        minimum_title=[k for k, v in sorted(newdict.items(), key=lambda item: item[1])]
                        find=1
                        googlelastid.append(minimum_title[0])
                        
                        break;
                    else:
                  
                        no_leven=0
                        googlelastid.append(value[0])     
                        find=1
                        break;
                else :
                    count+=1
                    find=0
        if find==0:          
            googlelastid.append(same_value[0][0])
            
data={'idAmazon':id_amazon,'idGoogleBase': googlelastid}
articles1= pd.DataFrame.from_dict(data)
articles1.to_csv('task1a.csv', index = False)    



    
