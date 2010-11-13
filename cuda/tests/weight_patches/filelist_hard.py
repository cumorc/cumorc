import sys
import os
from random import *


dirname = sys.argv[1]
d = [os.path.join(dirname, f) for f in os.listdir(dirname) if os.path.isdir(os.path.join(dirname, f))] 

def nfiles(di):
  files = [os.path.join(di, f) for f in os.listdir(di) if os.path.isfile(os.path.join(di, f)) and f.find(".tga")!=-1]
  return len(files)

d.sort(key=nfiles)

#seed(3123)
seed(int(sys.argv[3]))

i=0
#categs
k=0
for di in d:
  files = [os.path.join(di, f) for f in os.listdir(di) if os.path.isfile(os.path.join(di, f)) and f.find(".tga")!=-1]
  if len(files)==0:
    continue

  k+=1
  #if (k%2)==1:
  #  continue

  i+=1
  j=0
  files.sort()
  shuffle(files)
  np = int(sys.argv[2]) #number of pics
  files = files[0:(100+abs(np))]
  #pics
  for f in files:
    j+=1
    if (0<j<=np) or (np<0 and -np<j):
      print f,i

