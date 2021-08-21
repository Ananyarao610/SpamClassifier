#############################
### SPAM - Spam emails    ###
### HAM - Not spam emails ###
#############################

#Program logic:
# We have extracted 310 emails from our gmail account and stored it in .csv format
# Classification as spam or not spam is based on the 'Subject' of these mails
# The most frequently repeated words in both spam and ham mails are calculated and our model is trained on this data
# During testing, the model compares words in the input with the words it is trained with and classifies as Spam or Ham

import matplotlib.pyplot as plt
import pandas as pd

email_spam = pd.read_csv('SpamOrHam.csv')
email_spam.head()

email_spam['Label'].value_counts(normalize=True)

# Randomize the dataset to split into test and train data
data_randomized = email_spam.sample(frac=1, random_state=1)

# Calculate index for split
training_test_index = round(len(data_randomized) * 0.8)

# Split into training and test sets
training_set = data_randomized[:training_test_index].reset_index(drop=True)
test_set = data_randomized[training_test_index:].reset_index(drop=True)

#print(training_set.shape)
#print(test_set.shape)

training_set['Label'].value_counts(normalize=True)
test_set['Label'].value_counts(normalize=True)

# Before cleaning
training_set.head(3)

# After cleaning. Cleaning
training_set['Subject'] = training_set['Subject'].str.replace(
   '\W', ' ') # Removes punctuation
training_set['Subject'] = training_set['Subject'].str.lower()
training_set.head(3) #coverts all text into lowercase

training_set['Subject'] = training_set['Subject'].str.split() #removes whitespace

vocabulary = []
for sms in training_set['Subject']:
   for word in sms:
      vocabulary.append(word)

vocabulary = list(set(vocabulary))

# Finding out the number of times each word repeats
word_counts_per_mail= {'secret': [2,1,1],
                       'prize': [2,0,1],
                       'claim': [1,0,1],
                       'now': [1,0,1],
                       'coming': [0,1,0],
                       'to': [0,1,0],
                       'my': [0,1,0],
                       'party': [0,1,0],
                       'winner': [0,0,1]
                      }

word_counts = pd.DataFrame(word_counts_per_mail)
word_counts.head()

word_counts_per_mail= {unique_word: [0] * len(training_set['Subject']) for unique_word in vocabulary}

for index, mail in enumerate(training_set['Subject']):
   for word in mail:
      word_counts_per_mail[word][index] += 1

word_counts = pd.DataFrame(word_counts_per_mail)
word_counts.head()

train_clean = pd.concat([training_set, word_counts], axis=1)
train_clean.head()


# Isolating spam and ham messages first
spam_messages = train_clean[train_clean['Label'] == 'spam']
ham_messages = train_clean[train_clean['Label'] == 'ham']

# P(Spam) and P(Ham)
p_spam = len(spam_messages) / len(train_clean)
p_ham = len(ham_messages) / len(train_clean)

# N_Spam
spam_n = spam_messages['Subject'].apply(len)
n_spam = spam_n.sum()

# N_Ham
ham_n = ham_messages['Subject'].apply(len)
n_ham = ham_n.sum()

# N_Vocabulary
n_vocabulary = len(vocabulary)

# Laplace smoothing
alpha = 1


# Initiate parameters
parameters_spam = {unique_word:0 for unique_word in vocabulary}
parameters_ham = {unique_word:0 for unique_word in vocabulary}

# Calculate parameters
for word in vocabulary:
   n_word_given_spam = spam_messages[word].sum() # spam_messages already defined
   p_word_given_spam = (n_word_given_spam + alpha) / (n_spam + alpha*n_vocabulary)
   parameters_spam[word] = p_word_given_spam

   n_word_given_ham = ham_messages[word].sum() # ham_messages already defined
   p_word_given_ham = (n_word_given_ham + alpha) / (n_ham + alpha*n_vocabulary)
   parameters_ham[word] = p_word_given_ham

import re

def classify(message):
   '''
   message: a string
   '''

   message = re.sub('\W', ' ', message)
   message = message.lower().split()

   p_spam_given_message = p_spam
   p_ham_given_message = p_ham

   for word in message:
      if word in parameters_spam:
         p_spam_given_message *= parameters_spam[word]

      if word in parameters_ham: 
         p_ham_given_message *= parameters_ham[word]

   print('P(Spam|message):', p_spam_given_message)
   print('P(Ham|message):', p_ham_given_message)

   if p_ham_given_message > p_spam_given_message:
      print('Label: Ham')
   elif p_ham_given_message < p_spam_given_message:
      print('Label: Spam')
   else:
      print('Equal proabilities!')


# To determine spam or ham(not spam). 
#We check the words and compare the probability of it being spam or ham
def classify_test_set(message):
#message is a string

   message = re.sub('\W', ' ', message)
   message = message.lower().split()

   p_spam_given_message = p_spam
   p_ham_given_message = p_ham

   for word in message:
      if word in parameters_spam:
         p_spam_given_message *= parameters_spam[word]

      if word in parameters_ham:
         p_ham_given_message *= parameters_ham[word]

   if p_ham_given_message > p_spam_given_message:
      return 'ham'
   elif p_spam_given_message > p_ham_given_message:
      return 'spam'
   else:
      return 'needs human classification'


test_set['predicted'] = test_set['Subject'].apply(classify_test_set)
test_set.head()

# Testing the model

print("Results from the testing data given below:")
correct = 0
total = test_set.shape[0]

for row in test_set.iterrows():
   row = row[1]
   if row['Label'] == row['predicted']:
      correct += 1

print('Correct:', correct)
print('Incorrect:', total - correct)
print('Accuracy:', correct/total)

print()
print("Testing for input: How is the weather, Ananya?")
classify('How is the weather, Ananya?')
print()
print("Testing for input: Congrats for winning the new iPhone!")
classify('Congrats for winning the new iPhone!')
print()

print("Let's check the model's accuracy with your own inputs!")

# Function to test the model with user's input
def predict():
   email = input("Enter your message: ")
   classify(email)

predict()

# Graph of the results
def graph():
   predicted = []
   labels = []
   predict = []
   predicted = test_set['predicted']
   labels = test_set['Label']
   for i in range(len(labels)):
      if labels[i] == predicted[i]:
         predict.append(1)
      else:
         predict.append(0)

   plt.plot(range(0,len(predict)),predict)
   plt.xlabel('Emails')
   plt.ylabel('Prediction')
   plt.title('1: Correct Prediction; 0: Wrong Prediction')
   plt.show()

graph()





