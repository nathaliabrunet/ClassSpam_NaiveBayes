######## Classificador de spam with naive_bayes#####

import pandas as pd
import re
import string
import nltk

from nltk import FreqDist

import matplotlib.pyplot as plt 

from sklearn.feature_extraction.text import CountVectorizer#, TfidfVectorizer, TfidfTransformer

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

import pickle

#Carregando base
'''
CSV obtido no curso Machine Learning para cientista de dados da Data Science Academy (DSA)
'''
df = pd.read_csv('sms_spam.csv')

#FUNÇÕES
def remove_stopwords(text):
    '''
    Funcao para remover stopwords em inglês
    input:
        text: string
    output:
        string
    '''
    from nltk.corpus import stopwords
    stopwords = stopwords.words('english') #define aqui que é 'english'
    stopwords.extend(stopwords)
    #stopwords.extend(stop_pt)
    
    return ' '.join([word for word in str(text).split() if word not in stopwords])

def trata_texto(text, transform_numbers=True):
	'''
	Funcao para tratamento textual.
	    text: string
	    output: string
	'''
	text = re.sub('[%s]' % re.escape(string.punctuation), ' ', text)
	text = re.sub('[^a-zA-Z]', ' ', text)
	text = text.lower()
	text = text.replace('\xa0', ' ')

	re_transform_numbers = re.compile(r'\d', re.UNICODE)
	if transform_numbers:
	    text = re_transform_numbers.sub('0', text)

	return text.strip()


def Stemming(text):
    '''
    Função para aplicação do processo de stemização.
    É o processo de reduzir palavras flexionadas (ou às vezes derivadas) ao seu tronco (stem), base ou raiz, 
    geralmente uma forma da palavra escrita.
    Exemplo em inglês:
        studies
            sufixo: es
            stem: studi
    '''
    stemmer = nltk.stem.RSLPStemmer()
    palavras = []
    for w in text.split():
        palavras.append(stemmer.stem(w))
    return (" ".join(palavras))



#Aplicando funções
text_tratado = df['text'].apply(trata_texto).apply(remove_stopwords).apply(Stemming)

#Vetorizando - Bag of Words
	#tokenizing - quebrar uma frase em palavras
	#texto vira parágrafos >> parágrafos vira frase >> frase vira palavra
#CountVectorizer vai gerar um vetor com essas contagens

count_vector = CountVectorizer(max_features=2500) #cria o objeto
X = count_vector.fit_transform(text_tratado).toarray() #passa o texto "tratado, póps aplicação das funções
X.shape

y  = df['type']


#Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 42)

#É bom verificar a proporção
#print('treino:\n', y_train.value_counts(normalize=True))
#print('\nteste:\n', y_test.value_counts(normalize=True))

#Instanciando
spam_model = MultinomialNB().fit(X_train, y_train) #alpha=0 --laplace

#predict
y_pred = spam_model.predict(X_test)

#Performance
print('classification_report:\n', metrics.classification_report(y_test, y_pred,target_names=df.type.unique()))

print('confusion_matrix:\n', metrics.confusion_matrix(y_test, y_pred))

#Guardando modelo treinado
pickle.dump(spam_model, open('spam_model.pkl', 'wb'))

#para puxar modelo do diretório
# Carregar modelo
#with open('spam_model.pkl', 'rb') as x:
 #   model = pickle.load(x)

#model = pickle.load(open('spam_model.pkl', 'rb'))