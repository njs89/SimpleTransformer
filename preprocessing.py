import re
import pandas as pd
from difflib import get_close_matches
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import emoji
import numpy as np


gibert = "gibert/"
olid = "OLIDv1.0/"

training_data = "OLIDv1.0/olid-training-v1.0.tsv"
test_data_a = "OLIDv1.0/testset-levela.tsv"
#test_data_b = "OLIDv1.0/testset-levelb.tsv"
#test_data_c = "OLIDv1.0/testset-levelc.tsv"
label_a = "OLIDv1.0/labels-levela.csv"
#label_b = "OLIDv1.0/labels-levelb.csv"
#label_c = "OLIDv1.0/labels-levelc.csv"

gibert_training = "gibert/trainData.csv"

def open_olid_training(filename):
    """
    extracts the tweets and labeld from OLID-A
    """
    corpus = pd.read_csv(filename, delimiter="\t", names=["id", "tweet", "subtask_a", "subtask_b", "subtask_c"])
    tweet_ids = corpus["id"]
    tweets = corpus["tweet"]
    label_a = corpus["subtask_a"]  # OFF or NOT offensive
    return tweet_ids, tweets, label_a

def open_gibert(filename):
    """
    extracts the sentences and labels from Gibert
    """
    corpus = pd.read_csv(filename, delimiter="\t", names=["Id", "Text", "Label"])
    sentence_ids = corpus["Id"]
    sentences = corpus["Text"]
    label = corpus["Label"]  # OFF or NOT offensive
    return sentence_ids, sentences, label 

def hashtag_splitter(word):
    """
    splits a hashtag into space separated words
    """
    word = word.replace("#", "")
    if word.isupper():
        return [word]
    elif word.islower():
        return [word]
    else:
        newword = word[0].upper()+word[1:]
        unhashtagged_words = re.findall('[A-Z][^A-Z]*', word)
        return unhashtagged_words

def word_numeral_splitter(word):
    """
    separates words from numbers: Trump2020 > Trump 2020
    """    
    wordlist = list()
    split_tuple = re.findall(r"([a-z]+)([0-9]+)|([0-9]+)([a-z]+)", word, re.I)
    if split_tuple:
        split_list = list(split_tuple[0])
        divided_items = list()
        for item in split_list:
            if item != "":
                divided_items.append(item)
        return divided_items
    else:
        return [word]


def delete_user_repetitions(sentence):
    """
    OLID consists of anonymised @USER tokens.
    they dont carry any meaning, so they will be deleted.
    """
    filtered_sentence = sentence.copy()
    last_word = ""
    next_word = ""
    counter = 0
    for word in sentence:
        if counter%2 == 0:
            last_word = word
        else:
            next_word = word
        if last_word == next_word and last_word == "@USER":
            filtered_sentence.pop(counter)
            counter-=1
        counter+=1
    return filtered_sentence
    
def correct_obscenities(word):
    """
    takes all words with a * and compares to list of obscenities
    returns normalised obscenity
    """
    x = re.search(r'\w+\*+\w+',word) #if "*" in token
    if x is None:
        return word
    else:
        with open("badwords.txt", "r", encoding = "utf-8") as textfile:
            curses = (textfile.read())
            curselist = curses.split("\n")
            closest_curses = get_close_matches(word, curselist)
            if len(closest_curses) == 0:
                return word
            else:
                top_curse = closest_curses[0]
                return top_curse

def emoji_to_phrase(sentence):
    """
    changes emoji into phrase: :( > sad face etc.
    """
    cleaned_emoji_description = list()
    for word in sentence:
        emoji_description = emoji.demojize(word)
        cleaned_emoji_description.append(emoji_description.replace(":"," ").replace("_"," "))
    return cleaned_emoji_description

def delete_stopwords(tweet):
    """
    deleting stopwords. is it necessary for BERT?
    not used
    """
    data = tweet
    stopWords = set(stopwords.words('english'))
    words = word_tokenize(data)
    wordsFiltered = []
    
    for w in words:
        if w not in stopWords:
            wordsFiltered.append(w)
    return(wordsFiltered)
 
    
def delete_special_characters(word):
    symbols = "\\',!.\"#$%&()*+-/:;<=>?@[\]^_`{|}~\n" 
    for i in symbols:
        word = np.char.replace(word, i, ' ')
    if word != None:
        return word
    
def replace_apostrophe(phrase):
    # specific    
    phrase = re.sub(r"won('|’|`)t", "will not", phrase)
    phrase = re.sub(r"can('|’|`)t", "can not", phrase)
    # general
    phrase = re.sub(r"n('|’|`)t", " not", phrase)
    phrase = re.sub(r"('|’|`)re", " are", phrase)
    phrase = re.sub(r"('|’|`)s", " is", phrase)
    phrase = re.sub(r"('|’|`)d", " would", phrase)
    phrase = re.sub(r"('|’|`)ll", " will", phrase)
    phrase = re.sub(r"('|’|`)t", " not", phrase)
    phrase = re.sub(r"('|’|`)ve", " have", phrase)
    phrase = re.sub(r"('|’|`)m", " am", phrase)

    return phrase

      
def start_preprocessor(tweets):
    """
    returns preprossed tweets
    """
    preprocessed_tweets = list()
    counter = 0
    special_processing = list()
    for tweet in tweets:
        if counter == 14000:
            break
        clean_tweet = list()
        space_split_tweets = tweet.split(" ") #splits tweets into words
        #print(space_split_tweets)
                
        hashtag_splitted_tweets = list()
        for word in space_split_tweets:
            word = correct_obscenities(word) #exchanges words with * to cursewords
            if "#" in word and len(word)>1: #spacesplits hashtahgs
                split_words = hashtag_splitter(word)
                hashtag_splitted_tweets+=(split_words)
            else:
                hashtag_splitted_tweets.append(word)
        
        word_number_splitted_tweets = list()        
        
        for word in hashtag_splitted_tweets:
            if word == "URL": 
                word = "html"
            word_number_splitted_tweets += word_numeral_splitter(word)

        user_repetions_deleted = delete_user_repetitions(word_number_splitted_tweets)
        emojis_into_phrases = emoji_to_phrase(user_repetions_deleted)
        lowercased_tweets = np.char.lower(emojis_into_phrases) #not necessary
        #stopwords_deleted = delete_stopwords(string_tweet) #for bert not necessary
        string_tweet = ' '.join(word for word in lowercased_tweets)
        clean_tweet = replace_apostrophe(string_tweet)
        tweet_without_special_character = list()
        tokens = word_tokenize(clean_tweet)
        for word in tokens:
            clean_word = delete_special_characters(word)
            if clean_word:
                tweet_without_special_character.append(str(clean_word))
        string_tweet = ' '.join(word for word in tweet_without_special_character) #for naive bayes
        preprocessed_tweets.append(string_tweet)
        counter+=1    
    
    #print(preprocessed_tweets[:10])
    return preprocessed_tweets
        

tweet_ids, tweets, tweet_labels = open_olid_training(training_data)
tweet_ids = tweet_ids[1:]
tweet_labels = tweet_labels[1:]
preprocessed_tweets = start_preprocessor(tweets[1:])
"""
sentence_ids, sentences, sentence_label = open_gibert(gibert_training)
sentence_ids = sentence_ids[1:]
sentence_label = sentence_label[1:]
#for sent in sentences:
 #   print(sent)
#print(sentences)
preprocessed_sentences = start_preprocessor(sentences[1:])
print(preprocessed_sentences)   
"""