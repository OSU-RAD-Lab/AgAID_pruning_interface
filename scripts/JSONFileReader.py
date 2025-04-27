#!/usr/bin/env python3

import numpy as np
import json as json
import os
import sys


###################################
# NAME: JSONFileReader.py
# DATE: September 18, 2024
# DESCRIPTION: This file is meant to read json files for use in the interface.
#              These json files can either be for workflow/interface layout or
#              object file related
###################################


class JSONFile:

    def __init__(self): # fname=None, ftype="o"
        self.dir = "../json/"
        self.data = {} 

        # # Two types of files: Workflow based or object based
        # if ftype is None:
        #     self.ftype = "w" # workflow
        # else:
        #     self.ftype = ftype # object
        
        # self.read_file()
        

    # READS IN THE FILE FOR THE OBJECT
    def read_file(self, fname=None):
        if fname is not None:
            self.pname = self.dir + str(fname)
            try:
                print(f"Opening the file {fname}")
                with open(self.pname, 'r') as fp:
                    self.data = json.load(fp)
            except FileNotFoundError:
                print(f"Could not find the file at {self.pname}")
        return self.data

    
    def write_file(self, userData, pid, treeName=None, treeNum=None):

        # Need to write to the user's folder
        # fname = "PID_" + str(pid) + "_data.json"

        pid_directory = f"../user_data/PID_{pid}"

        # Create a directory for the PID if it doesn't exist
        # It should already exist, but this is double checking
        if not os.path.exists(pid_directory):
            os.makedirs(pid_directory)


        if treeName is not None and treeNum is not None:
            fname = pid_directory + f"/Tree_{treeNum}_" + str(treeName) + ".json"
        else:
            fname = pid_directory + f"/Evalatuations.json"

        
        with open(fname, "w") as file:
            data = json.dumps(userData, indent=4)
            file.write(data)
            
                

if __name__ == "__main__":
    jsonData = JSONFile("objFileDescription.json", "o").data
    print(jsonData["Tree Files"]["exemplarTree.obj"])
    print(jsonData["Tree Files"]["exemplarTree.obj"]["Features"])