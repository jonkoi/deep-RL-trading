import random, os, datetime, pickle, json, keras, sys
import pandas as pd
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import numpy as np
import time

OUTPUT_FLD = os.path.join('..','results')
PRICE_FLD = '/Users/xianggao/Dropbox/distributed/code_db/price coinbase/vm-w7r-db'

def makedirs(fld):
	if not os.path.exists(fld):
		os.makedirs(fld)
