# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 16:23:24 2020

@author: Jfitz

Assignment 2 - Task 4

Task 4
Write a Python script that operates in two phases. The first phase should automate the creating
of a folder called “task4”. If this folder exist, delete it and recreate it. Inside this folder, create
two subfolders named “backup” and “working”. Inside the “working” folder create three other
subfolders named “pics”, “docs” and “movie”. Inside the “docs” folder create five files
(CORONAVIRUS.txt, DANGEROUS.txt, KEEPSAFE.txt, STAYHOME.txt, HYGIENE.txt)
with varying content of your choice and two subfolders (school and party).

In the second phase, the script should rename all the files in the “docs” folder to lowercase.
The extension “.txt” should be renamed to uppercase. Note that the subfolders in that directory
should remain unchanged. When the renaming is complete, the program should use the Python
zipfile module to archive the “docs” folder and make five backup archives of it in the top-level
“backup” folder. Output the content of the backup folder and one of the zip archives for
verification purpose.
"""

#NOTES:
#My source code for the 4 tasks is located in folder 
# c:\systemscripting\john-fitzgerald
# i have added folder c:\systemscripting\john-fitzgerald\assignment2
# to create the task4 folder and manage the copy and deleting of files 
# as I dont want this program to delete my source code

import os #to manage the system
#import shutil # to delete subfolders
from zipfile import ZipFile #used for zipping 'docs' folder

def check4Folders(): 
  #establish folder name *** YOU MAY HAVE TO EDIT THIS TO RUN *****
  #folder = 'C:/systemscripting/john-fitzgerald/'
  os.chdir("C:/systemscripting/john-fitzgerald/assignment2/")
  for root, dirs, files in os.walk(".", topdown = False):
   for file in files:
      print(os.path.join(root, file))
      os.remove(os.path.join(root, file))


def archiveFiles():
  print(os.listdir())      
    # printing the list of all files to be zipped 
  print('files to be zipped:') 
  bkupdir = 'C:/systemscripting/john-fitzgerald/assignment2/task4/backup/'
  for file_name in os.listdir(): 
        print(file_name) 
  for i in range(5):
    # writing files to a zipfile 
    with ZipFile(bkupdir+'ArchiveBkup_'+str(i)+'.zip','w') as zip: 
        # writing each file one by one 
        for file in os.listdir('.'): 
            filename, file_extension = os.path.splitext(file)
            if file_extension == '.TXT':
               zip.write(file)   
  print('All 5 files zipped!')
  print(os.listdir(bkupdir)) 
  contentList = ZipFile(bkupdir+'ArchiveBkup_4.zip', 'r')
  print(contentList.namelist())
  
def createFoldersAndFiles():
  #make the directories first
  os.makedirs('task4/backup')
  os.makedirs('task4/working/pics')
  os.makedirs('task4/working/movies')
  os.makedirs('task4/working/docs')
  os.makedirs('task4/working/docs/school')
  os.makedirs('task4/working/docs/party')
  # then make the files - open, populate and close
  file1 = open("task4/working/docs/CoronaVIrus.txt","w+")
  file2 = open("task4/working/docs/DangerOUS.txt","w+")
  file3 = open("task4/working/docs/KEEpsafe.txt","w+")
  file4 = open("task4/working/docs/StAYHome.txt","w+") 
  file5 = open("task4/working/docs/Hygiene.txt","w+")  
  for i in range(10):
      file1.write("This is line %d\r\n " % (i+1))    
      file2.write("This is line %d\r\n " % (i+1)) 
      file3.write("This is line %d\r\n " % (i+1)) 
      file4.write("This is line %d\r\n " % (i+1)) 
      file5.write("This is line %d\r\n " % (i+1))
  file1.close()
  file2.close()
  file3.close()
  file4.close()
  file5.close()
  #changing directory to show our files before case changes
  os.chdir('C:/systemscripting/john-fitzgerald/assignment2/task4/working/docs/')
  print(os.listdir())

def renameFiles(): 
  for afile in os.listdir('.'):
      #split filename and extension here
      filename, file_extension = os.path.splitext(afile)
      if file_extension == '.txt':
         #files with extention .txt will be changed as required
         os.rename(afile, filename.lower() + '.TXT')

 
check4Folders()
createFoldersAndFiles()
renameFiles()
archiveFiles()
        
  
