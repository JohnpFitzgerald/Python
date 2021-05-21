# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 17:27:32 2020

@author: Jfitz
"""


# Easter Project Modular Programming
# John Fitzgerald
# managing the contents of a file in Lists
# Using git as my VCS
#
#I want to clear the screen every time I use the menu 
#so I am importing the operating system details to run the 
#correct command for clearing the screem 'clear' in ubuntu & 'cls' in windows 
import os
import random
import time
from datetime import date
#clear screen
os.system('cls' if os.name == 'nt' else 'clear')
#Page headings for main menu and each option
w = "Welcome to the Banking Menu"
r = "Close and delete an account"
o = "Open a new account"
d = "Withdraw money from account"
m = "Make a deposit to an account"
a = "Accounts report"
#initialize the option entry of 1 - 6
opt = 0
#read the data file and put the details of each record into a list ** return lists for processing 
def get_data():
  infile = open("bank.txt", "r")
  acnum = []
  acname = []
  acamount = []
  while True:
    line = infile.readline().rstrip()
    if line == "":
       break
    line = line.split('~')
    acnum.append(int(line[0]))
    acname.append(line[1])
    acamount.append(float(line[2]))
  infile.close
  return acnum, acname, acamount


#function for main menu screen
def menu():
  print(w.center(50, "="))
  print("")
  print("")
  print("               1. Open an account")
  print("               2. Close an account")
  print("               3. Withdraw money")
  print("               4. Deposit Money")
  print("               5. Display and then generate a report for Managment")
  print("               6. Quit")
  print("")
  print("")

#the following function displays the report on screen 
#it displays the account with the largest deposit in red (NB on windows only) 
#it also writes the details to a file - and adds *** to the account with the 
#largest deposit
def create_report(acnum, acname, acamount):
  AccountDetails = open("AccountDetails.txt", "w")
  os.system('cls' if os.name == 'nt' else 'clear')
  print(a.center(80, "="))
  print("")
  print("           Date: "+str(date.today()))
  print("")
  print(80* "=")
  print("           Number               Name                     Balance            ")
  print(80* "=")  
  AccountDetails.write(a.center(80, "=")+"\n")
  AccountDetails.write(" ")
  AccountDetails.write("           Date: "+str(date.today())+"\n")
  AccountDetails.write(80* "="+"\n")
  AccountDetails.write("           Number               Name                    Balance            \n")
  AccountDetails.write(80* "="+"\n")
#Hi Cliona, I have a the text coloured red on the report on screen for the account with the largest amount
# on deposit. It works on my desktop in windows - It does not work in UNIX and it does not work on 
# the lab machines I have used. I hope it works on your machine,  thanks...John
  class color:
    RED = '\033[91m'
    END = '\033[0m'
  for i in range(len(acnum)):
    if acamount[i] == (max(acamount)):
       print(color.RED+"           "+str(acnum[i])+"              "+acname[i].ljust(25)+" € "+str(acamount[i])+ color.END)
       print()
       AccountDetails.write("***        "+str(acnum[i])+"              "+acname[i].ljust(25)+" € "+str(acamount[i])+" ***"+"\n")
       AccountDetails.write(" "+"\n")
    else:
       print("           "+str(acnum[i])+"              "+acname[i].ljust(25)+" € "+str(acamount[i]))
       print()
       AccountDetails.write("           "+str(acnum[i])+"              "+acname[i].ljust(25)+" € "+str(acamount[i])+"\n")
       AccountDetails.write(" "+"\n")
  totamt = sum(acamount)
  totDeposits = float(("%0.2f"%totamt))
  totac = len(acnum)
  print(80* "=")
  print("       Total number of accounts: "+str(totac)+"    Total amount on deposit: €"+str(totDeposits))
  print(80* "=")  
  print("")
  print("")
  print("Please wait...generating file AccountDetails.txt ")  
  AccountDetails.write(80* "="+"\n")
  AccountDetails.write("       Total number of accounts: "+str(totac)+"    Total amount on deposit: €"+str(totDeposits)+"\n")
  AccountDetails.write(80* "="+"\n")
  AccountDetails.close
  time.sleep(6)


#if the option chosen needs an Account number to find details 
# call this function
def get_ac():
  ok = False
  while not ok:
      try:
          print ("")
          print ("")
          anumber = int(input('          Enter the account number: '))
          if anumber > 99999:
             ok = True
          else:
             print("Account number must be a positive integer 6 digits in length")
      except:
        print("Account number must be numeric...")
  return anumber


# function here to display a message if valid account number is not found
def noaccount(anumber):
  print("")
  print("")
  print("Account: "+str(anumber)+ " does not exist...Please wait... ")
  time.sleep(5)

def write_updates(acnum, acname, acamount):
#  if amendment_made_flag > 0:
  outfile = open("bank.txt", "w")
  for i in range(len(acnum)):
      outfile.write(str(acnum[i])+"~"+str(acname[i])+"~"+str(acamount[i])+"\n")
  outfile.close

def account(opt, acnum, acname, acamount):
  os.system('cls' if os.name == 'nt' else 'clear') 
#initialize a flag to check for any updated to any of the back details
  amendment_made_flag = 0
  if opt ==  1:
     print(o.center(50, "="))
     print("")
     print("")
     acname.append(str(input('       Enter the account holders name: ')))
     print("")
     print("")
     acamount.append(float(input('       Enter the opening deposit amount: ')))
     newnum = (int(random.SystemRandom().randint(100000,999999)))
     acnum.append(int(random.SystemRandom().randint(100000,999999)))
# ALTERNATIVE
#if newnum in acnum:
# position_of_account = acnum.index(newnum)
# else:
#   print("MESSAGE")     
     print("")
     print("")
     print("New account created for: "+acname[-1]+ " Initial deposit: "+str(acamount[-1])+ " Account Number generated: "+str(acnum[-1]))
     amendment_made_flag = amendment_made_flag+1
     time.sleep(5)
  if opt ==  2:
     print(r.center(50, "="))
     anumber = get_ac()
     found_flag = 0
     i = 0
     while i < (len(acnum)): 
       if anumber == acnum[i]:
          found_flag = 1
          print("")
          print("")
          print("          Name: "+acname[i]+ "       Amount: "+str(acamount[i]))
          print("")
          print("")
          confirm = str(input('          Close this account, and delete all details? [y/n]: '))
          if confirm == 'y':
             acnum.remove(acnum[i])
             acamount.remove(acamount[i])
             acname.remove(acname[i])
             print("please wait...")
             amendment_made_flag = amendment_made_flag+1
             time.sleep(5)
          else:
             print("returning to main menu....")
             time.sleep(5)
       i = i+1
     if found_flag == 0:
        noaccount(anumber)
  if opt == 3:
     print(d.center(50, "=")) 
     anumber = get_ac()
     found_flag = 0
     i = 0
     while i < (len(acnum)): 
       if anumber == acnum[i]:
          found_flag = 1
          print("")
          print("")
          print("          Name: "+acname[i]+ "       Amount: "+str(acamount[i]))
          print("")
          print("")
          confirm = str(input('          Is this the correct account? [y/n]: '))
          if confirm == 'y':
             print("")
             print("")
             withdraw_amt = float(input('          How much do you want to withdraw? : '))
             if withdraw_amt < acamount[i]:
                acamount[i] = acamount[i] - withdraw_amt
             else:
                print("Maximum withdrawl limit reached. Try a smaller amount. Please wait...")
             print("")
             print("")
             print("Balance: "+str(acamount[i]))
             time.sleep(5)
          else:
             print("returning to main menu....")
             time.sleep(5)
       i = i+1
     if found_flag == 0:
        noaccount(anumber)
  if opt == 4:
     print(m.center(50, "="))
     anumber = get_ac()
     found_flag = 0
     i = 0
     while i < (len(acnum)): 
       if anumber == acnum[i]:
          found_flag = 1
          print("")
          print("")
          print("          Name: "+acname[i]+ "       Amount: "+str(acamount[i]))
          print("")
          print("")
          confirm = str(input('          Is this the correct account? [y/n]: '))
          if confirm == 'y':
             print("")
             print("")
             deposit_amt = float(input('          How much do you want to deposit? : '))
             acamount[i] = deposit_amt + acamount[i]
             print("")
             print("")
             print("Balance: "+str(acamount[i]))
             amendment_made_flag = amendment_made_flag+1
             time.sleep(5)
          else:
             print("returning to main menu....")
             time.sleep(5)
       i = i+1
     if found_flag == 0:
        noaccount(anumber)
  if opt == 5:
     create_report(acnum, acname, acamount)

def main(opt):
  acnum, acname, acamount = get_data()
  ok = False
  while not ok:
      os.system('cls' if os.name == 'nt' else 'clear')
      menu()
      opt = int(input('               Select an option from 1 to 6: '))
      if opt < 6:
         account(opt, acnum, acname, acamount)
      elif opt == 6:
           write_updates(acnum, acname, acamount)
           ok = True
      else:
           print(" Please enter a number between 1 and 6...")
           time.sleep(3)

main(opt)
