import nltk
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
import os, re
from nltk.stem.porter import *
import gensim
from gensim.models import Word2Vec
import pandas as pd
import numpy as np
import xml.etree.ElementTree as xmlTree



class MySentences(object):
    def __init__(self, fname):
        self.fname = fname
    def __iter__(self):
        for line in open(self.fname):
            yield line.split()
            
            

import spellChecker 
STEMMER = PorterStemmer()
STOPLIST = stopwords.words('english')
MODEL = None

def synonyms(word):
    syns = set()
    for synset in wn.synsets(word):
        for lemma in synset.lemmas():
            syns.add(str(lemma.name()))
            #print syns
    return syns 

def stem(wordSet):
    stemmedSet = set()
    for word in wordSet:
        stemmedSet.add(STEMMER.stem(word))
    return stemmedSet


def quickScore(response, target, stemming = 1):
    ##preprocess target
    keywords = re.findall('[a-z_]+', target.lower())
    keywords = [i for i in  keywords if i not in STOPLIST]
    if len(keywords) == 0:
        return 0;
    ##preprocess response
    exactResponse = re.findall('[a-z_]+', response.lower())
    exactResponse = [i for i in  exactResponse if i not in STOPLIST]
    exactResponse = set(exactResponse)
    correctedResponse = set()
    for i in  exactResponse:
        candidates = spellChecker.spellCorrect(i)
        for word in candidates:
            #print ("word "+ word + " is added as a correct candidate for " + i)
            correctedResponse.add(word)    
    exactMatch, corMatch, synMatch = 0,0,0 
    if(stemming == 0):
        for word in keywords:
            syns = synonyms(word)
            if word in exactResponse:
                exactMatch = exactMatch + 1
            elif( word in correctedResponse):
                corMatch = corMatch + 1
            elif (len( correctedResponse   & syns ) >= 1 ):
                    synMatch = synMatch + 1
    else:
        exactResponse = stem(exactResponse)
        correctedResponse = stem(correctedResponse)
        for word in keywords:
            syns = synonyms(word)
            syns = stem(syns)
            word = STEMMER.stem(word)
            if word in exactResponse:
                exactMatch = exactMatch + 1
            elif( word in correctedResponse):
                corMatch = corMatch + 1
            elif (len( correctedResponse   & syns ) >= 1 ):
                    synMatch = synMatch + 1
    print("\nExact matches: " + str(exactMatch)+
        "\nCorrected mathces: " + str(corMatch)+
        "\nSynonymy matches: " + str(synMatch))
    return (exactMatch + corMatch + synMatch) * 1.0 / len(keywords)
    
        

def w2vScore(response, target, model, bigramTransformer = None ):
    #print("Response:" + response)
    #print("Target:" + target)
    keywords = re.findall('[a-z_]+', target.lower())
    responses = re.findall('[a-z_]+', response.lower())
    if( bigramTransformer != None ):
        keywords = bigramTransformer[keywords]
        responses = bigramTransformer[responses]
    keywords = [i for i in  keywords if i not in STOPLIST]
    responses = [i for i in  responses if i not in STOPLIST]
    if len(keywords) == 0 :
        return 0;
    keywordsPrepared = []
    responsesPrepared = []
    for i in keywords:
        if i in model.vocab:
            keywordsPrepared.append(i)
    for i in responses:
        if i in model.vocab:
            responsesPrepared.append(i)
        else:
            for candidate in spellChecker.spellCorrect(i):
                if candidate in model.vocab:
                    responsesPrepared.append(candidate)
    print(responsesPrepared)
    print(keywordsPrepared)
    if len(keywordsPrepared) == 0 or len(responsesPrepared) == 0 :
        return 0;
    result = model.n_similarity(responsesPrepared, keywordsPrepared)
    #print(result)
    return result

def processExcel(inFileName, outFileName):
    df = pd.ExcelFile(fileName)
    df = df.parse()
    print "Start: processExcel"
    print df.head()
    newColumns = list(df.columns) + list(["MYQS", "MYW2V"])
    newDf = pd.DataFrame(columns = newColumns ) 
    #print newDf.columns
    for index, row in df.iterrows():
        #print str(row['Target']) 
        #print str(row['Response'])
        row['MYQS'] = quickScore(str(row['Target']) , str(row['Response']))
        row['MYW2V'] = w2vScore(str(row['Target']) , str(row['Response']), MODEL, None)
        newDf.loc[index] = row 
    print newDf.head()
    print "End: processExcel"
    newDf.to_csv(outFileName)
   
def processXML(inFileName, outFileName, mode = 'w'):
    print "Start: processXML"
    question = xmlTree.parse(inFileName).getroot()
    QuestionId = question.get("id")
    QuestionType = question.get("qtype")
    QuestionText = question.find("questionText").text
    columns = list(["QuestionId", "QuestionType", "QuestionText","ResponseId", "ResponseText", "W2V", "QS", "QW2V", "QQS", "Label" ])
    df = pd.DataFrame(columns = columns ) 
    refAnswers = question.find("referenceAnswers")
    #for ra in refAnswers.iter("referenceAnswer"):
    #    print ra.get('category'), ra.text
    stuAnswers = question.find("studentAnswers")
    counter = 0
    for sa in stuAnswers.iter("studentAnswer"):
        ResponseText = sa.text
        ResponseId = sa.get("id")
        Label = sa.get("accuracy")
        #print sa.get('accuracy'), sa.text
        QQS = quickScore(ResponseText, QuestionText) 
        QW2V = w2vScore(ResponseText , QuestionText, MODEL, None)
        W2V = -1
        QS = -1
        for ra in refAnswers.iter("referenceAnswer"):
            currentW2V = w2vScore(ResponseText , ra.text, MODEL, None)
            if(currentW2V > W2V):
                W2V = currentW2V
            currentQS = quickScore(ResponseText , ra.text)
            if(currentQS > QS):
                QS = currentQS
            print ra.get('category'), ra.text
        df.loc[counter] = [QuestionId, QuestionType, QuestionText, ResponseId, ResponseText, W2V, QS, QW2V, QQS, Label]
        counter += 1
    #print(df)
    if(mode == 'a'):
        df.to_csv(outFileName, mode='a', header=False)
    else:
        df.to_csv(outFileName)
    print "End: processXML"
    return
    
if __name__ == '__main__':
    ## preparation ##
    MODEL= Word2Vec.load_word2vec_format("/Users/fa/workspace/repos/_codes/trunk/vectors-phrase.bin", binary=True)  # C text format
    MODEL.init_sims(replace=True)
    
    ## test ##
    #bigramTransformer = gensim.models.Phrases(sentences)
    #sent = [u'the', u'mayor', u'of', u'new', u'york', u'was', u'there']
    #print(bigramTransformer[sent])
    target = "because they can eat or use them for camouflage" # or food and shelter
    responses = ["because some animals eat algae and green plant, some fish use green plant for camouflage.",
                "they can blend into them",
                "they look like food to eat",
                "they feed from them",
                "there are good stuff in it for them",
                "It's bright"]
    for response in responses:
        s1 = quickScore( response, target)
        #print "WN : " + str(s1)
        s2 = w2vScore(response, target, MODEL, None)
        #print "W2V : " + str(s2)
        print "Comparing " + response + "\t" + str(s1) + "\t" + str(s2)
        
    #processXLS('/Users/fa/workspace/repos/_codes/quickScore/studentData.xlsx','/Users/fa/workspace/repos/_codes/quickScore/studentData_MYQS_MYW2V.csv')
'''    
    mode = 'w'
    for i in os.listdir("/Users/fa/workspace/temp/data/SemEval2013_Task7/semeval2013-Task7-5way/beetle/train/Core/"):
        if i.endswith(".xml"): 
            print i
            processXML("/Users/fa/workspace/temp/data/SemEval2013_Task7/semeval2013-Task7-5way/beetle/train/Core/" + i, 
                 '/Users/fa/workspace/repos/_codes/quickScore/beetle/beetleTrain.csv', mode)
            mode = 'a'
           
    mode = 'w'
    for i in os.listdir("/Users/fa/workspace/temp/data/SemEval2013_Task7/semeval2013-Task7-5way/beetle/test-unseen-answers/Core/"):
        if i.endswith(".xml"): 
            print i
            processXML("/Users/fa/workspace/temp/data/SemEval2013_Task7/semeval2013-Task7-5way/beetle/test-unseen-answers/Core/" + i, 
                 '/Users/fa/workspace/repos/_codes/quickScore/beetle/beetleTest-unseen-answers.csv', mode)
            mode = 'a'
    mode = 'w'
    for i in os.listdir("/Users/fa/workspace/temp/data/SemEval2013_Task7/semeval2013-Task7-5way/beetle/test-unseen-questions/Core/"):
        if i.endswith(".xml"): 
            print i
            processXML("/Users/fa/workspace/temp/data/SemEval2013_Task7/semeval2013-Task7-5way/beetle/test-unseen-questions/Core/" + i, 
                 '/Users/fa/workspace/repos/_codes/quickScore/beetle/beetleTest-unseen-questions.csv', mode)
            mode = 'a'
            
 '''   
   # processXML('test.xml', 'test.out')
    

    
    