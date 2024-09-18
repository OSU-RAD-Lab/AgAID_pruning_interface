#!/usr/bin/env python3

import numpy as np
import json as json


###################################
# NAME: JSONFileReader.py
# DATE: September 18, 2024
# DESCRIPTION: This file is meant to read json files for use in the interface.
#              These json files can either be for workflow/interface layout or
#              object file related
###################################


class JSONFile:

    def __init__(self, fname, ftype="o"):
        self.dir = "../json/"
        self.fname = fname
        self.pname = self.dir + str(fname)
        self.data = {} 

        # Two types of files: Workflow based or object based
        if ftype is None:
            self.ftype = "w" # workflow
        else:
            self.ftype = ftype # object
        
        self.read_file()
        

    # READS IN THE FILE FOR THE OBJECT
    def read_file(self):
        try:
            print(f"Opening the file {self.fname}")
            with open(self.pname, 'r') as fp:
                self.data = json.load(fp)
        except FileNotFoundError:
            print(f"Could not find the file at {self.pname}")
                

if __name__ == "__main__":
    jsonData = JSONFile("objFileDescription.json", "o").data
    print(jsonData["Tree Files"]["exemplarTree.obj"])
    print(jsonData["Tree Files"]["exemplarTree.obj"]["Features"])