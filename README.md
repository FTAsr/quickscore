# quickscore

QuickScore is a simple method of short answer scoring using keyword matching and WordNet as a semantic resource to detect synonyms. The input to this method are the student's answer and the gold answer (teacher's answer) and the output is a score between 0 and 1.

-----

A distributional semantic scorer (using word2vec embeddings) is included in the code. In order to use this scorer you need to download some pre-trained word2vec model from https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit and load the model (which takes a few minutes in python code).
