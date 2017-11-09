from ._isaac import *

try:
	from glob import glob
	from os.path import join, dirname, realpath 
	tensorflow = glob(join(dirname(realpath(__file__)), '_tensorflow*.so'))[0]
except IndexError:
	pass
