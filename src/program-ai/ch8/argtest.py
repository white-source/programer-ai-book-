import argparse


parser = argparse.ArgumentParser()
parser.add_argument('-c', '--conf', help='path to configuration file')   
print('start')
args = parser.parse_args()
print(args.conf)

