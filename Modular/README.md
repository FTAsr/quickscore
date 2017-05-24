# Instructions to use the Modular library for short answer scoring

This library makes it possible for several programmers to work on the same code. The interface of the short answer scoring systems is user friendly and a quick start to train and test a model would be to look at the main function implemented in file supervisedEval/main.py. However, to get a better understanding of the implemented models please study the following information.

## Models to choose from
- bow or the bag-of-words model is implemented based on distributional semantics. The current implementation can read word representations from any pre-trained vector space (such as word2vec or GloVe) and compute a sentence vector through averaging over all present words within a sentence. In order to score the similarity between two sentences, in unsupervised manner, one could simply compute the cosine similarity between the two bow vectors (see class bow:sentence_similarity() in models/models.py). However, in supervised framework, a machine-learning component (classifier) can be trained on features obtained from the two sentence vectors. This will allow the classifier to find the important hidden components of the sentences' that have some contribution in measuring the sentence similarity. This has been implemented in supervisedEval/main.py.
- quickScore is the old model and our baseline method for scoring the similarity between two sentence. It basically finds the number of exact matches, synonyms (based on WordNet) and dictation-corrected words between the two sentences. Again, quickScore features can either be used in an unsupervised manner to evaluate the similarity of the two sentences (see class quickScore:sentence_similarity() in models/models.py) or be passed to a classifier for supervised learning and scoring.
- feature-based model is based on a general feature-rich approach to many NLP tasks. In this model, a lot of different characteristics of the two sentences, such as their length and lexical density, as well as pairwise features, such as the number of common n-grams between the two sentences are computed. Given the complexity of the relation between these features and the degree of similarity between two sentences, this relation needs to be discovered by a machine-learning component. Therefore, the feature-based model is only applicable in a supervised evaluation framework, where a classifier is trained on some labeled data. Note that you can turn off/on desired features in the feature-based model by commenting/uncommenting the relevant lines of code.

## Supervised evaluation
In supervised evaluation, we need labeled data (sentence pairs and their similarity scores) in order to train a classifier and then we can use this classifier for unlabeled data (new sentence pairs that we would like to score based on the "learned" patterns). The supervised evaluation has the advantage of fitting to our desired type of data (for example student answers might be scored in a less strict manner when kid's are the target than when adult answers are being evaluated). At the same time, this means we would need training data of the same type. The more similar training and test data, the better result we get from the automated scoring system.

In order to use the implemented supervised evaluation framework, you need to learn about 4 main functions:

1) Setting up the scorer model(s): this is done by making a list of models (which we call ensemble) and passing each model the initialization parameters/input. For bow model, the only required input is a vector space file in binary format. The quickScore and featureBased models do not have any input parameter:
    
    ensemble = list()
    
    
    ensemble.append(models.bow("../pretrained/word2vec/small.bin"))
    
    
    ensemble.append(models.featureBased())
    
 2) Training the scorer on some labeled data: for this you need to have training, development and test data in separate files, or all in one file ready for random split. See the following example for reading SICK dataset composed of three separate files:
 
   
    trainSet, devSet, testSet = load_data_SICK('../data/SICK/')
    
    
    classifier = train(ensemble, trainSet, devSet)
    
 3) Test the scorer on same type or different type of data: once you have a trained classifier you could use it on the test portion of your data, or data of different type. Examples of using a trained classifier on SICK data to evaluate similarity scores for college students' and school kids' data are below:
 
    
    test(ensemble, classifier, testSet).to_csv('../data/local/SICK-trained_SICK-test.csv')
    
    
    x, y, testSet = load_data('../data/local/CollegeOldData_HighAgreementPartialScoring.txt')
    test(ensemble, newClassifier,testSet).to_csv('../data/local/SICK-trained_College-test.csv')
    
    x, y, testSet = load_data('../data/local/IES-2Exp1A_AVG.txt')
    test(ensemble, newClassifier,testSet).to_csv('../data/local/SICK-trained_Exp1A-test.csv')
    
    x, y, testSet = load_data('../data/local/IES-2Exp2A_AVG.txt')
    test(ensemble, newClassifier,testSet).to_csv('../data/local/SICK-trained_Exp2A-test.csv')
    
 4) Saving the classifier for future use: since training the classifier takes some time, you should know how to save and load it for future use (this is very important in online application of the system). Always do a checksum after loading a classifier to make sure it has been read correctly from the disk:
 
    
    fileName = '../pretrained/SICK-Classifier.h5'
    
    
    classifier.save(fileName)
    newClassifier = load_model(fileName)
    
    
    test(ensemble, newClassifier, testSet)
    
## Data format
Data files need be tab-delimited text files (.txt); sample data can be found in data/SICK directory. The first line of an input data file consists of the column names (goldAnswer, teacherAnswer, similarityScore) and other lines each demonstrates one data point. If you want to read train, development and test data from separate files (SICK format), you can put them in one category and call the function load_data_SICK(). Otherwise, train/dev/test data can be randomly chosen from a single input file by load_data() with a pre-specified split function.
 

    
    
    

