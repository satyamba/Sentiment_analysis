import nltk.data
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt


# we initialize VADER so we can use it within our Python script
sia = SentimentIntensityAnalyzer() 

# We also initialize our 'english.pickle' function 

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')


# The tokenize method breaks up the paragraph into a list of strings.
file = open('testdata.txt', 'r')
text = file.read()
sentences = tokenizer.tokenize(text)

# calculating and printing polarity scores for each one.
neutral=[]
negetive=[]
positive=[]
compound=[]
for sentence in sentences:
        print(sentence)
        value = sia.polarity_scores(sentence)
        for key in (value):
                print('{0}: {1}, '.format(key, value[key]))
                for i in range(1):
                    if key=='neu':
                        neutral.append(value[key])
                    elif key=='neg':
                        negetive.append(value[key])
                    elif key=='pos':
                        positive.append(value[key])
                    else: compound.append(value[key])
        print()
        
value1 = sia.polarity_scores(text)  


a=[]
print('total score\n')
for key in (value1):
    
    print('{0}: {1}, '.format(key, value1[key]), end='')
    
    for i in range(1):
        if key=='neu':
            a.append(value1[key])
        
              
print(a)
print(neutral)
print()
print(negetive)
print()
print(positive)
print()
print(compound)
l=len(compound)
y=[]
for i in range(l):
    y.append(i/l)
    plt.figure(figsize=(5,10))
plt.scatter(y,negetive)
plt.scatter(y,positive)
plt.scatter(y,neutral)
plt.plot(y,compound)
plt.legend(['neg','pos','neu','compound'])
plt.xlabel('No. of sentences')
plt.ylabel('sentiment score')
plt.show()
plt.hist(compound)
plt.show()
