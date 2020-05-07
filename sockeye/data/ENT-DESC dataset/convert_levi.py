# separate train/dev/test files
def sep_train_dev_test():
	filei=open('levi.txt','r').readlines()
	fileo=open('output_new_ori.txt','r').readlines()
	amrgrh_train=open('train.amrgrh','w')
	amrgrh_dev=open('dev.amrgrh','w')
	amrgrh_test=open('test.amrgrh','w')
	amr_train=open('train.amr','w')
	amr_dev=open('dev.amr','w')
	amr_test=open('test.amr','w')
	grh_train=open('train.grh','w')
	grh_dev=open('dev.grh','w')
	grh_test=open('test.grh','w')
	surf_train=open('train_surface.pp.txt','w')
	surf_dev=open('dev_surface.pp.txt','w')
	surf_test=open('test_surface.pp.txt','w')
	for i in range(len(filei)):
		loc=filei[i].find('gnode	(')
		amr=filei[i][:loc+5]
		grh=filei[i][loc+6:]
		if i%10==5:
			amrgrh_test.write(filei[i])
			amr_test.write(amr+'\n')
			grh_test.write(grh)
			surf_test.write(fileo[i])
		elif i%10==8:
			amrgrh_dev.write(filei[i])
			amr_dev.write(amr+'\n')
			grh_dev.write(grh)
			surf_dev.write(fileo[i])
		else:
			amrgrh_train.write(filei[i])
			amr_train.write(amr+'\n')
			grh_train.write(grh)
			surf_train.write(fileo[i])

# change the separator between amr and grh from space to tab
def use_tab():
	filei=open('test.amrgrh','r').readlines()
	fileo=open('test.amrgrh','w')
	for i in range(len(filei)):
		loc=filei[i].find('gnode (')
		str1=filei[i][:loc+5].replace('\t',' ')
		str2=filei[i][loc+6:].replace('\t',' ')
		fileo.write(str1+'\t'+str2)
		# fileo.write(filei[i][:loc+5]+'\t'+filei[i][loc+6:])
def use_tab1():
	filei=open('test.amrgrh','r').readlines()
	fileo=open('test.amrgrh','w')
	for i in range(len(filei)):
		loc=filei[i].find('gnode')
		str1=filei[i][:loc+1].replace('\t',' ')
		# str2=filei[i][loc+6:].replace('\t',' ')
		fileo.write(str1+filei[i][loc+1:])


def levi(levi_file,graph):
	
	ori_nodes=set()
	edges=set()
	graph=[x for x in graph if x]
	for line in graph:
		line=line.split(' | ')
		ori_nodes.add(line[0].replace(' ','_'))
		ori_nodes.add(line[2].replace(' ','_'))
		edges.add(line[1].replace(' ','_'))
	new_nodes=ori_nodes.union(edges)
	new_nodes=list(new_nodes)
	new_nodes.append('gnode')
	# gnode='gnode'
	for node in new_nodes:
		levi_file.write(node + '\t')
	# levi_file.write(gnode + '\n')
	for i in range(len(new_nodes)):
		levi_file.write('(' + str(i) + ',' + str(i) + ',s) ')
		if i<len(new_nodes)-1:
			levi_file.write('(' + str(len(new_nodes)-1) + ',' + str(i) + ',g) ')
	for line1 in graph:
		line=line1.split(' | ')
		node1=new_nodes.index(line[0].replace(' ','_'))
		edge=new_nodes.index(line[1].replace(' ','_'))
		node2=new_nodes.index(line[2].replace(' ','_'))
		levi_file.write('(' + str(node1) + ',' + str(edge) + ',d) ')
		levi_file.write('(' + str(edge) + ',' + str(node2) + ',d) ')
		levi_file.write('(' + str(node2) + ',' + str(edge) + ',r) ')
		levi_file.write('(' + str(edge) + ',' + str(node1) + ',r) ')
		levi_file.write('(' + str(node1) + ',' + str(node2) + ',d2) ')
		levi_file.write('(' + str(node2) + ',' + str(node1) + ',r2) ')
		# for line2 in graph:
		# 	if line1==line2:
		# 		continue
		# 	line2=line2.split(' | ')
		# 	node21=new_nodes.index(line2[0].replace(' ','_'))
		# 	edge2=new_nodes.index(line2[1].replace(' ','_'))
		# 	node22=new_nodes.index(line2[2].replace(' ','_'))
		# 	#pointing outwards
		# 	if node1==node21:
				

	levi_file.write('\n')

# filei=open('input_new_ori.txt','r').readlines()
# levi_file=open('levi.txt','w')
# for i in filei:
# 	i=i.replace('\n','').split(' < TSP > ')
# 	levi(levi_file,i)

# sep_train_dev_test()

use_tab1()


