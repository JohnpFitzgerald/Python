# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 23:18:38 2020

@author: John Fitzgerald r00156081
"""
#Task 1 Write 3 functions to perform tests on a list of string items

import sys #use the sys.exit command to quit/logout of the application
import time # use time command to pause the program for displays


def divisorOf3(theList):
    for x in theList:    
        if len(x)/3 == 1:  
           print('Output from Function 1: '+x)
 

def lastCharOf4(theList):
   # for x in theList:
         
      #  if len(x)>4: 
         print('Output from Function 2: '+max(theList , key = len))
            
def replicateBy3(theList):
    theList = theList + theList + theList
    for x in theList:
        if len(x)%2 != 0:
            print(x)

#def createList():
    

def menu():
    print("           ************Task 1 r00156081**************")
    #time.sleep(1)
    print()

    choice = input("""
                      A or a: Creates a new list
                      F or f: Outputs first element divisible by 3
                      L or l: Outputs last element with more than 4 chars
                      any Number: Replicates list by 3 and Output odd values
                      E or e: Exits from this task

                      Please enter your choice: """)
    if choice == "A" or choice == "a":
       # createList()
       replicateBy3(myList)  
    elif choice == "F" or choice =="f":
       divisorOf3(myList)
    elif choice == "L" or choice =="l":
         lastCharOf4(myList)
    elif choice=="E" or choice=="e":
        print("Goodbye and thank you for participating in this task")
        time.sleep(3)
        sys.exit
    elif int(choice):
        replicateBy3(myList)         
    else:
        print("You can only choose from the options provided")
        print("Please try again")
        menu()
       
# Create a list if string values here
myList = ['zero','oneone', 'two', 'three']
# Function 1 - returns first item in list divisible by 3
#divisorOf3(myList)
#Function 2 - returns the last item in list with more than 4 chars
#lastCharOf4(myList)
#Function 3 - replicates the lilst and outputs per line odd num chars
#replicateBy3(myList)
menu()
