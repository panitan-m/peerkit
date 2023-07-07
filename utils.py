import os
import sys
import errno
import logging
import random
import torch
import numpy as np
import re

def mkdir_p(path):
	if path == '':
		return
	try:
		os.makedirs(path)
	except OSError as exc: # Python >2.5
		if exc.errno == errno.EEXIST and os.path.isdir(path):
			pass
		else: raise
  
def set_logger(out_dir=None):
	console_format = BColors.OKBLUE + '[%(levelname)s]' + BColors.ENDC + ' (%(name)s) %(message)s'
	#datefmt='%Y-%m-%d %Hh-%Mm-%Ss'
	logger = logging.getLogger()
	logger.handlers.clear()
	logger.setLevel(logging.INFO)
	console = logging.StreamHandler()
	console.setLevel(logging.INFO)
	console.setFormatter(logging.Formatter(console_format))
	logger.addHandler(console)
	if out_dir:
		file_format = '[%(levelname)s] (%(name)s) %(message)s'
		log_file = logging.FileHandler(out_dir + '/log.txt', mode='w')
		log_file.setLevel(logging.INFO)
		log_file.setFormatter(logging.Formatter(file_format))
		logger.addHandler(log_file)
  
def print_args(args, path=None):
	if path:
		output_file = open(path, 'w')
	logger = logging.getLogger(__name__)
	logger.info("Arguments:")
	args.command = ' '.join(sys.argv)
	items = vars(args)
	for key in sorted(items.keys(), key=lambda s: s.lower()):
		value = items[key]
		if not value:
			value = "None"
		logger.info("  " + key + ": " + str(items[key]))
		if path is not None:
			output_file.write("  " + key + ": " + str(items[key]) + "\n")
	if path:
		output_file.close()
	del args.command
 
def setup(args):
	mkdir_p(args.out_dir)
	set_logger(args.out_dir)
	print_args(args)
	if args.seed is not None:
		set_seed(args.seed)
  
def set_seed(seed):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False
  
class BColors:
	HEADER = '\033[95m'
	OKBLUE = '\033[94m'
	OKGREEN = '\033[92m'
	WARNING = '\033[93m'
	FAIL = '\033[91m'
	ENDC = '\033[0m'
	BOLD = '\033[1m'
	UNDERLINE = '\033[4m'
	WHITE = '\033[37m'
	YELLOW = '\033[33m'
	GREEN = '\033[32m'
	BLUE = '\033[34m'
	CYAN = '\033[36m'
	RED = '\033[31m'
	MAGENTA = '\033[35m'
	BLACK = '\033[30m'
	BHEADER = BOLD + '\033[95m'
	BOKBLUE = BOLD + '\033[94m'
	BOKGREEN = BOLD + '\033[92m'
	BWARNING = BOLD + '\033[93m'
	BFAIL = BOLD + '\033[91m'
	BUNDERLINE = BOLD + '\033[4m'
	BWHITE = BOLD + '\033[37m'
	BYELLOW = BOLD + '\033[33m'
	BGREEN = BOLD + '\033[32m'
	BBLUE = BOLD + '\033[34m'
	BCYAN = BOLD + '\033[36m'
	BRED = BOLD + '\033[31m'
	BMAGENTA = BOLD + '\033[35m'
	BBLACK = BOLD + '\033[30m'
	
	@staticmethod
	def cleared(s):
		return re.sub("\033\[[0-9][0-9]?m", "", s)