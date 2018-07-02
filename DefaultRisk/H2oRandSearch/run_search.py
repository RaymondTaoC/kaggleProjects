import sys
import argparse
sys.path.append('../../..')
from kaggleProjects.DefaultRisk.H2oRandSearch.search_methods import run

parser = argparse.ArgumentParser()
parser.add_argument('-c', help='configuration file (python script in Search_Configurations)')
args = parser.parse_args()
run('kaggleProjects.DefaultRisk.H2oRandSearch.Search_Configurations.' + args.c)
