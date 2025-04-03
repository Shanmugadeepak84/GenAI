# Assignment 2: Word Frequency Counter
Sentence=input("Enter a sentence: ")
print("The sentence is: ", Sentence)
WordList=Sentence.split()
print("The words in the sentence are: ", WordList)
WordCount={}
sortedWordList=sorted(WordList, key=str.lower)
print("The sorted list of words is: ", sortedWordList)
for word in sortedWordList:
    if word in WordCount:
        WordCount[word] += 1    
    else:
        WordCount[word] = 1

print("The word frequency count is: ", WordCount)