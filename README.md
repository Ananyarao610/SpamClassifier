# SpamClassifier
Spam classifier for internship selection at AI/ML for cyber-security, PESU Research centre.
Allowed python libraries - only pandas, numpy and matplotlib

## Algorithm
It is a probabilistic model which predicts whether a mail is Spam or Ham(not spam). It uses the Naive Bayes theorem.
Classification as spam or not spam is based on the 'Subject' of these mails
The most frequently repeated words in both spam and ham mails are calculated and our model is trained on this data
During testing, the model compares words in the input with the words it is trained with and classifies as Spam or Ham

## Dataset
We have used a chrome extension called Cloud HQ to download our emails in the form of a .csv file. We have extracted 310 emails from our gmail account
This data has been filtered and then used to train our model. 

## Result
The model classifies the emails coorectly with an accuracy of 83%.




