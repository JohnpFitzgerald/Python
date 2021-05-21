# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 15:58:03 2020

@author: Jfitz

Assignment 2 - Task 3

Task 3
Write a function named reducer() that accepts an integer parameter named number. If number
is even then reducer() should divide number by 2 and return this value. If number is odd, then
reducer should return 3 times number + 1.
Then write a program that requires a user to enter an integer number and that keeps calling
reducer() on that number until the function returns the value 1. (Amazingly, this sequence
works for any integer value. Sooner or later you will arrive at value 1). Example output
sequence for entering the number 3 is:
10
5
16
8
4
2
1 
"""
import sys # used to exit the program
import time #used for holding up the program

def reducer(num):
    #we only want to stop when number is 1
    while num != 1:
      #divide number by 2 and check for no remainder  
      if num%2 == 0:
         #no remainder divide number by 2 and create a new number 
         num = num/2
         #print new number
         print(int(num))
      else:
         #if there is a remainder multiply by 3 and add 1 
         num = (num*3)+1
         #print the result and start again
         print(int(num))
         


def getNumber():
    print("************Task 3 : The Reducer **********************")
    print("")
    #ask user do they want to create a list and declare list name
    nList = input("Play the game? [Y/N] ")
    #if they want to create a list
    if nList == "Y" or nList == "y":
       #ask user to enter a number
       #if they do not enter a number program will fail with a message
       try:
         num = int(input("Enter a number: "))
       except:
         print('not a valid entry - numbers only')  
       #call the function to perform the task passing the number
       reducer(num)  
    else:
      #if user doesnt want to play they see this message
      print("Goodbye, and thanks for paricipating")       
      time.sleep(3)
      sys.exit()

getNumber()

