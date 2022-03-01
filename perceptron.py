# Name: Tra Bui
# AndrewID: tpbui
# 15-282: AI in Medicine
# Homework 2
# Due date: February 15th, 2022

# This program needs numpy library to be installed!
import numpy as np
import math

############### Helper functions ###############

# Compare value and threshold
# Return the according values
def compare(num1, num2):
    if abs(num1 - num2) < 10**-9: # compare two floats
        return "wrong value" # if num == threshold --> wrong value
    elif num1 > num2:
        return 1
    elif num1 < num2:
        return -1

# Determine whether y and score have the same sign
def hasSameSign(score, num):
    if type(score) != int or type(num) != int:
        return False
    if score * num > 0:
        return True
    return False

# Mark the position of score slot in the flattened array of the given matrix
# For the use of score calculation
def markingScoreSlot(m):
    flattened = m.flatten()
    ret = []
    
    # mark the position of index that has true value
    for i in range(len(flattened)):
        if flattened[i] == 1:
            ret.append(i)
    return ret

# Calculate the score with respect to the weight matrix and the value matrix
# Based on the marked slots
def scoreX(m1, m2):
    scoreTable = np.dot(m1,m2)
    scoreList = scoreTable.flatten()
    score = 0
    for index in markingScoreSlot(m2):
        score += scoreList[index]
    return score
   
############### Main function ###############    
'''The function serves as a perceptron classifier using Python to classify
whether a given Pribnow sequence is a true-site or a non-site.'''

'''Given the initial values, we compare the predicted classification to the
given input(supervised learning), and change the weightMatrix within some
numbers of iteration.'''

def perceptron(data, learningRate, weightMatrix, threshold, iterationNum):
    for i in range(iterationNum):
        score = 0
        for i in range(len(data)):
            x = data[i][0] #extract value matrix
            y = data[i][1] #extract true classification
            score = scoreX(weightMatrix, x) 

            #check whether y and score has the same sign  
            signOfScore = compare(score, threshold)        
            check = hasSameSign(signOfScore, y)
            
            # if there is misclassification
            # change the weight matrix
            if (check == False): 
                if y == 1:
                    myMatrix = np.ones((6,4))
                else:
                    myMatrix = np.ones((6,4)) * (-1)
                
                #change the weightMatrix by delta: a*x* y
                delta = learningRate * np.dot(x, myMatrix)
                weightMatrix = weightMatrix + delta
            print(weightMatrix)
            print("\n")
                
    return weightMatrix
 
############### Input Data ###############

# All Pribnow sequences
s1 = np.array([[0,1,0,0,1,0], [1,0,0,0,0,1], [0,0,0,1,0,0], [0,0,1,0,0,0]])
s2 = np.array([[0,1,0,1,1,0], [1,0,1,0,0,1], [0,0,0,0,0,0], [0,0,0,0,0,0]])
s3 = np.array([[0,1,0,1,0,0], [0,0,1,0,0,1], [1,0,0,0,1,0], [0,0,0,0,0,0]])
s4 = np.array([[1,1,0,0,0,0], [0,0,1,1,0,0], [0,0,0,0,0,0], [0,0,0,0,1,1]])
s5 = np.array([[0,0,0,0,1,1], [0,0,1,1,0,0], [1,1,0,0,0,0], [0,0,0,0,0,0]])


dataList = list()
dataList.append((s1, 1))
dataList.append((s2, 1))
dataList.append((s3, 1))
dataList.append((s4, -1))
dataList.append((s5, -1))

# random initial weight matrix
initWeightMatrix1 = np.random.uniform(-1,1, size=(4,4)).round(decimals = 2)

# run the program for 10 rounds
result = perceptron(dataList, 0.5, initWeightMatrix1, 0, 10)

print(result)


# Task 3: Part d), calculate the score of TATGTT for final weight matrix
# One possible return
# a = np.array([[-0.07,-0.76,-1.43,0.23],
#                  [-0.2,-1.27,-0.35,-1.26],
#                  [-1.2 ,-1.96, -1.17, -1.94],
#                  [-0.38, -1.62 ,-0.25, -1.22]])
# 
# b = np.array([[0,1,0,0,0,0], [1,0,1,0,1,1], [0,0,0,1,0,0], [0,0,0,0,0,0]])
# 
# print(scoreX(a,b))
# print("Predicted class is non-true site")
            
        
    
    