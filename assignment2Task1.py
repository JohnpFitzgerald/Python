"""
Created on Mon Apr 13 16:09:30 2020

@author: Jfitz

Assignment 2 - Task 1

Task 1
Write three Python functions that accept a list of string items as parameter. The first function
should return the first item in the list whose length is divisible by 3. The second function should
return the last item in the list with more than 4 characters. The third function should replicate
the list by 3 and output one entry per line all the items with odd number of characters.
In the main program, interactively create a list of 5 string items (words, sentences etc) of
varying length. 

Present the user with a set of menu to select from. If the user entersthe character
‘F’ or ‘f’, the program should output the first item whose length is divisible by 3. When the
user enters the character ‘L’ or ‘l’, the program should output the last item with more than 4
characters. When the user enters any number, the program should perform the replication and
output the items with odd number of characters. The program should go back to the menu
options for new selection. When the user enters ‘E’ or ‘e’, the program should terminate with
a goodbye message


My comment:
This program applies the 3 functions to any list of 5 strings
entered by the user.

on launch an we check for existing list - if empty, the
user is prompted to create a 5 string list.
once created a menu of the 3 functions is displayed and the
user can select which function to run or exit.

On exit the user is returned to a new menu - which allows them
to display their existing list, use the list (ie display and the 
functions menu again), or they can create a new list of 5 strings 
to run the functions..or they can exit from the system.

The first function menu() might now be redundant as I had the list
as a global variable intially and had problems with it. 
To fix this I decalred it instide the menu() function and passed 
the list to the functions in the program that use it thereafter

"""

import os #operating system Windows to use 'cls' command
import sys #use the sys.exit command to quit/logout of the application
import time # use time command to pause the program for displays


#function to write the first divisor 
#by 3 found in the list
def divisorOf3(theList):
    for x in theList:    
        if len(x)%3 == 0:  
           print('Output from Function 1: '+x)
           returnToMenu(theList)
  

#function to write the last string
#greater than 4 characters           
def lastCharOf4(theList):
    for x in theList:         
        if len(x)>4:
           subList = []
           subList.append(x)
    print('Output from Function 2: '+subList[-1])
  

#function to replicate list*3 and write out all odd
#length strings in the list            
def replicateBy3(theList):
    rList = theList + theList + theList
    for x in rList:
        if len(x)%2 != 0:
            print(x)

#function that will ask the user to input 5 string values
# it then appends value to the list            
def createList(theList):    
    for i in range(5):
       i2 = i+1 
       print("String: "+str(i2)) 
       createaList = input("Please enter text: ")
       theList.append(str(createaList))
       
#after completion of each function, return 
#to the menu     
def returnToMenu(theList):
    print("Returning to functions menu")
    time.sleep(5)
    menu2(theList)   

#Menu allowing the user to select which function
#to perform on the list    
def menu2(theList):
    print("           ************Task 1 Part 3: use the functions**************")
    print()
    choice = input("""
                      F or f: Outputs first element divisible by 3
                      L or l: Outputs last element with more than 4 chars
                      any Number: Replicates list by 3 and Output odd values
                      E or e: Exits back to main menu

                      Please enter your choice: """)
  
    if choice == "F" or choice =="f":
       divisorOf3(theList)
       returnToMenu(theList)       
    elif choice == "L" or choice =="l":
         lastCharOf4(theList)
         returnToMenu(theList)       
    elif choice=="E" or choice=="e":
          existingList(theList)
    elif int(choice):
        replicateBy3(theList)
        returnToMenu(theList)         
    else:
        print("You can only choose from the options provided")
        print("Please try again")
        returnToMenu(theList)
       
#If the list already exists, after each function is completed
# the user is given the choice to use the existing list, create 
# a new one or exit from the program        
def existingList(theList):
    #os.system('cls')
    print("           ************Task 1 Part 2: List options**************")
    #time.sleep(1)
    print()
    opt = input("""
                      1. Display current list
                      2. Use existing list
                      3. Create a new list
                      4. Exit system

                      Please enter your choice: """)
    if opt == '1':
       #displays the current list and returns to menu 
       print(theList)
       print("Returning to main menu")
       time.sleep(5)
       existingList(theList)
    elif opt == '2':
         # run functions menu using existing list 
         menu2(theList)
    elif opt  == '3':
         #initialize original list 
         theList = [] 
         #query user to input new list
         createList(theList)
         #pass new list to menu function
         menu2(theList)            
    elif opt == '4':     
         print("Goodbye and thanks for participating ")
         time.sleep(2)
         #exit from the procedure
         sys.exit()
    else:
         # if i did not get expected entry
         print("Select only 1, 2 or 3")     
         time.sleep(2)
         existingList(theList)

#First time in (ie if the list is empty) we ask the
# user to create  a new list         
def newList(theList):
    os.system('cls')
    print("           ************Task 1 Part 1: Create a new list**************")
    #time.sleep(1)
    print()    
    nList = input("""Create a new list with 5 string values? [Y/N] """)
    if nList == "Y" or nList == "y":
       createList(theList)
       menu2(theList)
    elif nList == "N" or nList =="n":
        print("Goodbye and thanks for participating ")
        time.sleep(2)
        sys.exit
    else: 
        print("Please enter Y or N only")     
        time.sleep(2)
        menu()    
#this function is used to check 
#if there are values in the list 
# it manages which menu will be used        
#in an earlier version it controlled which menu
def menu():
    theList = []
    if not theList:
       newList(theList)
    else:
       existingList(theList) 
  
menu()
