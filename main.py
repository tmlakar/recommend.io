import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import warnings
from data_processing import *
from collaborative_filtering import *



print("........................................................")
print(" DATA IS BEING PROCESSED ")
print("........................................................")
# dataProcess()
print(" DATA IS PROCESSED ")

print("........................................................")
print(" COLLABORATIVE SYSTEM IS STARTING ")
print("........................................................")

collaborativeFiltering()

print(" DONE ")

print("........................................................")