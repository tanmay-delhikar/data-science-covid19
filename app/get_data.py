
import subprocess
import os
from os import path

# os.chdir("../../")

# print(os.getcwd())

def get_johns_hopkins():
    ''' Get data by a git pull request, the source code has to be pulled first
        Result is stored in the predifined csv structure

    
    '''

    if(os.path.exists('../data/raw/COVID-19/')):
        print('Johns hopkins git exists, pulling new data...')
        git_pull = subprocess.Popen( "git pull" ,
                            cwd = os.path.dirname( '../data/raw/COVID-19/' ),
                            shell = True,
                            stdout = subprocess.PIPE,
                            stderr = subprocess.PIPE )
        (out, error) = git_pull.communicate()

    else:
        print('Cloning Johns hopkins data...')
        git_pull = subprocess.Popen( "git clone https://github.com/CSSEGISandData/COVID-19.git" ,
                            cwd = os.path.dirname( '../data/raw/' ),
                            shell = True,
                            stdout = subprocess.PIPE,
                            stderr = subprocess.PIPE )
        (out, error) = git_pull.communicate()       

    print("Error : " + str(error))
    print("out : " + str(out))
