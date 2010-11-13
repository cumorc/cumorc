import sys
import os
from random import *


dirname = sys.argv[1]
d = [os.path.join(dirname, f) for f in os.listdir(dirname) if os.path.isdir(os.path.join(dirname, f))] 

def nfiles(di):
  files = [os.path.join(di, f) for f in os.listdir(di) if os.path.isfile(os.path.join(di, f)) and f.find(".tga")!=-1]
  return len(files)

d.sort(key=nfiles)

i=0
#categs
for di in d:
  files = [os.path.join(di, f) for f in os.listdir(di) if os.path.isfile(os.path.join(di, f)) and f.find(".tga")!=-1]
  if len(files)>0:
    i+=1
  j=0
  files.sort()
  shuffle(files)
#  files = files[0:30]+files[0:30]
  np = int(sys.argv[2]) #number of pics
  #pics
  for f in files:
    j+=1
    if (0<j<=np) or (np<0 and -np<j):
      print f,i

