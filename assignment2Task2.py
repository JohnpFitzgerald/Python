# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 14:16:53 2020


@author: John fitzgerald

Assignment 2 - Task 2

Task 2
Write a Python function that accepts two parameters. The first parameter should be a list of
items and the second parameter should be an integer value representing the number of indexes.
Use a random generator to generate indexes in the range of 2 to 5 based on the second
parameter. Use the generated indexes to randomly remove from the list the items that
corresponds to those indexes. The function should return the remaining items as a tuple.

Interactively create a list of words or numbers that has at least 12 items. Then request a user to
enter the number of items to be deleted. This number should fall between 2 and 6. Invoke your
function with the appropriate parameters. Output per line, first the returned tuple and then the
original lists.

Prompt the user to create a list with 12 elements
ensure 12 only are entered
Prompt user for nunumber of elements they want to remove
pass both list and value to a function which will perform the random removal
run the random check until the list of elements for deletion is the same size as
   the value keyed in by the user for deletion
   

"""
import random #module for impmenting random alogorithm
import sys # used to exit the program
import time #used for holding up the program

def getRandom(myList,idx):
    #create a list of element numbers to be removed
    deleteElement = []
    #Keep getting a random number until
    #the list is equal to the size of the number to be deleted
    while len(deleteElement) < idx: 
      #get a random number in the range 2-6 and step 1 digit each time 
      num  = random.randrange(2, 6, 1)
      #append the random number to the list
      deleteElement.append(num)
      #remove duplicate selections from the list and continue
      deleteElement = list(dict.fromkeys(deleteElement))      
    # for demo purposes - print the indexes for deletion
    print(deleteElement)
    # print the list before deletion
    print(myList)
    #iterate trough the delete delete the elements
    for x in deleteElement:
        myList.pop(x)
    #print the list after the deletion    
    print(myList)    
    

def newList(myList,idx):
    print("************Task 2 : Create a list with 12 items **********************")
    print("************and delete random elements based on number input by user***")
    #ask user do they want to create a list and declare list name
    nList = input("Create a list? [Y/N] ")
    #if they want to create a list
    if nList == "Y" or nList == "y":
       aNother = 0
       #as them 12 tomes to enter a value
       while (aNother < 12):
         addItem = input("Enter a value : ")
         #if nothing is entered ak them to enter something
         if not addItem:
            print(" Please enter a valid value...")
         else:
         #if something is entered add it to list and increment counter to 12    
            myList.append(str(addItem))    
            aNother = aNother+1
       #set a boolean loop
       ok = False
       while not ok:
         #ask the user to input a number between 2 aand 6
         idx = int(input("Enter number of elements to be delete (2-6): "))
         #if input is between 2 and 6 run the random function passing the 
         #list and the number of elements for deletion 
         if (idx >= 2) and (idx <= 6):
            getRandom(myList,idx)
            ok = True
         else:
            print(" Please enter a number between 2 and 6...")
            time.sleep(3)
    else:
      print("Goodbye, and thanks for paricipating")       
      time.sleep(3)
      sys.exit()


#declare a global list for use in the program        
myList = []
#declare a global value for the number of items to be deleted
idx = ""
#run function to propmt user for inputs
newList(myList,idx)
       
