def process(k):
	filei=open('multiGraph_'+k+'.txt','r').readlines()
	amrgrh_file=open(k+'.amrgrh','w')
	amr_file=open(k+'.amr','w')
	grh_file=open(k+'.grh','w')
	print(len(filei))
	for i in range(len(filei)):
		loc=filei[i].find('gnode	(')
		if loc==0:
			print(i)
		amr=filei[i][:loc+5]
		grh=filei[i][loc+6:]
		amrgrh_file.write(filei[i])
		amr_file.write(amr+'\n')
		grh_file.write(grh)

	amrgrh_file.close()
	grh_file.close()
	amr_file.close()

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


def multiGraph(multiGraph_file,graph):
	
	ori_nodes=set()
	edges=set()
	edges_inv=set()
	graph=[x for x in graph if x]
	for line in graph:
		line=line.split(' | ')
		ori_nodes.add(line[0].replace(' ','_'))
		ori_nodes.add(line[2].replace(' ','_'))
		edges.add(line[1].replace(' ','_'))
		# edges_inv.add(line[1].replace(' ','_')+'_inv')
	new_nodes=ori_nodes.union(edges)
	# new_nodes=ori_nodes.union(edges).union(edges_inv)
	new_nodes=list(new_nodes)
	# new_nodes.append('enode')
	# new_nodes.append('rnode')
	new_nodes.append('gnode')
	# gnode='gnode'
	for node in new_nodes[:-1]:
		multiGraph_file.write(node + ' ')
	multiGraph_file.write('gnode\t')
	for i in range(len(new_nodes)):
		multiGraph_file.write('(' + str(i) + ',' + str(i) + ',s) ')
		# if new_nodes[i] in ori_nodes:
		# 	multiGraph_file.write('(' + str(i) + ',' + str(len(new_nodes)-3) + ',n) ')
		# elif new_nodes[i] in edges:
		# 	multiGraph_file.write('(' + str(i) + ',' + str(len(new_nodes)-2) + ',e) ')
		# elif new_nodes[i] in edges_inv:
		# 	multiGraph_file.write('(' + str(i) + ',' + str(len(new_nodes)-2) + ',e) ')
		if i<len(new_nodes)-1:
			multiGraph_file.write('(' + str(len(new_nodes)-1) + ',' + str(i) + ',g) ')
	for line1 in graph:
		line=line1.split(' | ')
		node1=new_nodes.index(line[0].replace(' ','_'))
		edge=new_nodes.index(line[1].replace(' ','_'))
		# edge_inv=new_nodes.index(line[1].replace(' ','_')+'_inv')
		node2=new_nodes.index(line[2].replace(' ','_'))
		multiGraph_file.write('(' + str(node1) + ',' + str(edge) + ',d) ')
		multiGraph_file.write('(' + str(edge) + ',' + str(node2) + ',d) ')
		# multiGraph_file.write('(' + str(node2) + ',' + str(edge_inv) + ',r) ')
		# multiGraph_file.write('(' + str(edge_inv) + ',' + str(node1) + ',r) ')
		multiGraph_file.write('(' + str(node2) + ',' + str(edge) + ',r) ')
		multiGraph_file.write('(' + str(edge) + ',' + str(node1) + ',r) ')
		multiGraph_file.write('(' + str(node1) + ',' + str(node2) + ',d2) ')
		multiGraph_file.write('(' + str(node2) + ',' + str(node1) + ',r2) ')
		# for line2 in graph:
		# 	if line1==line2:
		# 		continue
		# 	line2=line2.split(' | ')
		# 	node21=new_nodes.index(line2[0].replace(' ','_'))
		# 	edge2=new_nodes.index(line2[1].replace(' ','_'))
		# 	node22=new_nodes.index(line2[2].replace(' ','_'))
		# 	#pointing outwards
		# 	if node1==node21:
				

	multiGraph_file.write('\n')

if __name__ == '__main__':
	filei=open('test-triples.txt','r').readlines()
	multiGraph_file=open('multiGraph_test.txt','w')
	for i in filei:
		i=i.replace('\n','').split(' < TSP > ')
		multiGraph(multiGraph_file,i)
	multiGraph_file.close()


	filei=open('dev-triples.txt','r').readlines()
	multiGraph_file=open('multiGraph_dev.txt','w')
	for i in filei:
		i=i.replace('\n','').split(' < TSP > ')
		multiGraph(multiGraph_file,i)
	multiGraph_file.close()


	filei=open('train-triples.txt','r').readlines()
	multiGraph_file=open('multiGraph_train.txt','w')
	for i in filei:
		i=i.replace('\n','').split(' < TSP > ')
		multiGraph(multiGraph_file,i)
	multiGraph_file.close()

	process('test')
	process('dev')
	process('train')

	# use_tab1()


