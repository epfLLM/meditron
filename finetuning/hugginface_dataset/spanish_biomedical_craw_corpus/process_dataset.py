from datasets import load_dataset
import os
import re

from pathlib import Path




path = Path(__file__).parent.absolute()

with open( str(path) + os.sep + 'example.txt',encoding='utf8') as file:
  """
     # Build a dictionary with ICD-O-3 associated with 
     # healtcare problems
  """
  linesInFile = file.readlines()
 
  for index, iLine in enumerate(linesInFile): 
    print(linesInFile[index]) if len(linesInFile[index]) > 1 else  print('**************') if linesInFile[index] == '\n' else print ('******* ERROR ********')
 

    # if re.match('^Las dilataciones bronquiales',iLine):
    #   break
   

    # code = listOfData[0]
    # description = reduce(lambda a, b: a + " "+ b, listOfData[1:2], "")
    # royalListOfCode[code.strip()] = description.strip()