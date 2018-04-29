#translate all five checkpoints in $out/eval/

import sys
from subprocess import call

out_path = sys.argv[1]
print ("the output path is:", out_path)

with open(out_path+'/eval/checkpoint', 'r') as f:
	lines = f.readlines()

path0, model0 = lines[0].split()
for i in range(1,6):
	path, model = lines[i].split()
	lines[0] = path0+' '+model+'\n'
	string = ''.join(lines)

	with open(out_path+'/eval/checkpoint', 'w') as f:
		f.write(string)

	# call('./translate.sh')
	print ("Finish %d checkpoints!" %i)