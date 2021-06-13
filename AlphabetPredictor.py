#------------------------------------------------------------------AlphabetImagePredictor.py------------------------------------------------------------------#
import sys
import time as tm
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score as a_s
from sklearn.metrics import confusion_matrix as conf_mat
from sklearn.model_selection import train_test_split as tts
from sklearn.linear_model import LogisticRegression as LogReg

'''
Importing modules:
-sys
-time (tm)
-numpy (np)
-pandas (pd)
-seaborn (sns)
-matplotlib.pyplot (plt) 
-accuracy_score (a_s) :-sklearn.metrics
-confusion_matrix (conf_mat) :-sklearn.metrics
-train_test_split (tts) :-sklearn.model_selection
-LogisticRegression (LogReg) :-sklearn.linear_model
'''

#Defining a function to print the ending message
def PrintEndingMessage(err_arg):
  print("Request Terminated.")
  print("Invalid Input.")
  print(err_arg)

  #Printing the ending message
  print("Thank You for using AlphabetImagePredictor.py")

  #Terminating the runtime
  sys.exit("Error: Invalid Input.")
  


#Defining a function to chooose the method of class selection for the user
def ChooseMethodOfSelection(user_choice_arg):
  final_list_param=None

  #Assessing the user's choice on the medium of class selection -i
  #Case-1 (i)
  if(user_choice_arg=="Choose particular classes"):
    class_input_param=int(input("Please enter the number of classes desried to predict and view:"))

    #Verifying if the value provided by the user is greater than 2 and lesser than 26 -ii
    #Case-1 (ii)
    if(class_input_param>2 and class_input_param<=26):
      final_list_param=[]

      alphabet_count_param=0

      for loop in range(class_input_param):
        alphabet_count_param+=1
        alphabet_list_param=["Unusable_Element","A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"]
        alphabet_count_loop_param=0

        for alphabet_param in alphabet_list_param[1:]:
          alphabet_count_loop_param+=1
          print("{}:{}".format(alphabet_count_loop_param,alphabet_param))
        
        user_input_param=int(input("Enter the index of class number {}:".format(alphabet_count_param)))
        user_choice_param=alphabet_list_param[user_input_param] 

        final_list_param.append(user_choice_param)

      return final_list_param

    #Case-2 (ii) 
    else: 
      PrintEndingMessage("The number should be within the range of 2 and 27, range boundaries excluded.")    

        

    
  #Case-2 (i)
  elif(user_choice_arg=="Choose all classes"):
     alphabet_list_param=["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"]
     final_list_param=alphabet_list_param

     return final_list_param

  #Case-3 (i)
  elif(user_choice_arg=="Choose random classes"):
    final_list_param=[]

    alphabet_list_param=["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"]

    number_input_param=int(input("Enter number of random values:"))

    #Verifying if the value provided by the user is greater than 2 and lesser than 26 -iii
    #Case-1 (iii)
    if(number_input_param>2 and number_input_param<=26):
      repetitive_input_param=input("Should the classes should be completely independent of each other?")

      #Asking the user whrther they want to have repeated values in the random slection or not -iv
      #Case-1 (iv)
      if(repetitive_input_param=="Yes" or repetitive_input_param=="yes" or repetitive_input_param=="YES" or repetitive_input_param=="yEs" or repetitive_input_param=="yeS" or repetitive_input_param=="Y"):
        final_list_param.append(np.random.choice(alphabet_list_param,number_input_param,replace=True))

        return final_list_param[0]

      #Case-2 (iv)
      else:
        final_list_param.append(np.random.choice(alphabet_list_param,number_input_param,replace=False))

        return final_list_param[0]

    #Case-2 (iii)
    else:
      PrintEndingMessage("The number should be within the range of 2 and 27, range boundaries excluded.")    

      
      



print("Welcome to AlphabetImagePredictor.py. We provide prediction services for images of alphabets")
tm.sleep(0.7)

print("Loading Data...")
tm.sleep(2.4)  
        
selection_method_list=["Unusble_Element","Choose particular classes","Choose all classes","Choose random classes"]
selection_count=0

for selection_method in selection_method_list[1:]:
  selection_count+=1
  print("{}:{}".format(selection_count,selection_method))

user_input=int(input("Enter the index:"))
user_choice=selection_method_list[user_input]

if(user_input<=3 and user_input>=1):

  class_list=ChooseMethodOfSelection(user_choice)

  class_list_length=len(class_list)
else:
  PrintEndingMessage("The number should be within the ranges of 1 and 3, range boundaries included")  

#Reading data from the file
df=pd.read_csv("data.csv")

#Loading the images from the file
img=np.load("image.npz")["arr_0"]

print("Rendering Images...")
tm.sleep(2.3)
plt.figure(figsize=((class_list_length*2),(11)))

index_count=0

for class_b in class_list:
  indexes=np.flatnonzero(df==class_b)
  indexes=np.random.choice(indexes,5,replace=False)

  value=0

  for index in indexes:
    plot_index=value*class_list_length+index_count+1

    p=plt.subplot(5,class_list_length,plot_index)
    p=sns.heatmap(np.reshape(img[index],(22,30)),cmap=plt.cm.gray,xticklabels=False,yticklabels=False,cbar=False)
    p=plt.axis("off")
    
    value+=1

  index_count+=1  

plt.show()

img_train,img_test,df_train,df_test=tts(img,df,test_size=0.67,random_state=9)

img_train=img_train/255
img_test=img_test/255

print("Predicting Data...")
tm.sleep(1.2)

LR=LogReg(solver="saga",multi_class="multinomial")
LR.fit(img_train,df_train)
prediction=LR.predict(img_test)

veracity=a_s(df_test,prediction)
print("The veracity of the data is {}%".format(round(veracity*100,2)))

print("Rendering Confusion Matrix...")
tm.sleep(2.3)

c_m=conf_mat(df_test,prediction)

plt.figure(figsize=(20,20))

s_b=plt.subplot()

sns.heatmap(c_m,annot=True,cbar=False,ax=s_b,fmt="d")

s_b.set_xlabel("Actual")
s_b.set_ylabel("Predictied")

s_b.set_title("Values")

plt.show()

#Printing the ending message
print("Thank You for using AlphabetImagePredictor.py")
#------------------------------------------------------------------AlphabetImagePredictor.py------------------------------------------------------------------#
