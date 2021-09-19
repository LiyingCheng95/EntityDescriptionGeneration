filein=open('src-test.txt','r').readlines()
fileout=open('test_table.txt','w')
for i in filein:
	i=i.replace(' | ','|||').replace(' < TSP > ','\t').replace(' ','_')
	fileout.write(i)

fileout.close()