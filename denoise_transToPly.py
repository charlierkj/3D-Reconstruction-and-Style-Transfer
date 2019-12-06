import json
import yaml
import numpy as np
'''
# read .json
with open("exam1.json", 'r') as f:
	temp = json.loads(f.read())
	point=temp['Points']
	color=temp['Colors']
'''

'''
# read .yml
with open("cat-structure.yml", 'r') as f:
	temp = yaml.load(f.read())
	point=temp['Points']
	color=temp['Colors']
#print(point)
'''

# read .obj
point = []
color = []

f = open('cat-1_10.obj', 'r')
line = f.readline()
while line:
    num = list(map(str,line.split()))
    if len(num)>0:
        if num[0]=='v':
            point.append([float(x) for x in num[1:4]])
            color.append([float(x) for x in num[4:7]])
    line = f.readline()
f.close()

##### denoise #####

point = np.array(point,dtype=float)
color = np.array(color,dtype=float)
point = np.hstack((point,color))

lenth=len(point)
point_sort=point[np.lexsort(point[:,::-1].T)]
#print(point_sort)
final=[]
proc=0

for i in range(lenth):
    k=0.0
    num_min=0
    num_max=0
    num=1
    flag1 = True
    flag2 = True
    
    while flag1 or flag2:
        if i+num<lenth and point_sort[i+num,0]-point_sort[i,0]<1.5:
            num_max=num
        else:
            flag1 = False
        if i-num>=0 and point_sort[i,0]-point_sort[i-num,0]<1.5:
            num_min=num
        else:
            flag2 = False
        num+=1
        
    for j in range(1,num_min):
        if np.sqrt(np.sum(np.square(point_sort[i] - point_sort[i-j])))<1.5 and point_sort[i,0]!=point_sort[i-j,0]:
            k=k+1/np.sum(np.square(point_sort[i] - point_sort[i-j]))
    for j in range(1,num_max):
        if np.sqrt(np.sum(np.square(point_sort[i] - point_sort[i+j])))<1.5 and point_sort[i,0]!=point_sort[i+j,0]:
            k=k+1/np.sum(np.square(point_sort[i] - point_sort[i+j]))
    
    print(k)
    if k >= 1000 and k <= 3000:
        final.append(point_sort[i])
        proc+=1
        print("!!!!!!!!!!!"+str(proc)+'!!!!!!!!!!!')
    print(i)

final = np.array(final)


##### rewrite into file #####

lenth=len(final)
print(lenth)
target = open('exam1.ply', 'w')
target.write("ply\n")
target.write("format ascii 1.0\n")
target.write("element vertex "+str(lenth)+'\n')
target.write("property float x\nproperty float y\nproperty float z\nproperty float nx\nproperty float ny\nproperty float nz\nproperty uchar diffuse_red\nproperty uchar diffuse_green\nproperty uchar diffuse_blue\nproperty uchar class\nproperty uchar detection\nend_header\n")
for i in range(lenth):
	#target.write(str(final[i][0])+' '+str(final[i][1])+' '+str(final[i][2])+' 0 0 0 '+str(int(final[i][3]))+' '+str(int(final[i][4]))+' '+str(int(final[i][5]))+' 0 0'+'\n')
	target.write(str(final[i][0])+' '+str(final[i][1])+' '+str(final[i][2])+' 0 0 0 0 0 0 '+'0 0'+'\n')

target.close()

target_2 = open('exam1.obj', 'w')

for i in range(lenth):
	target_2.write('v '+str(final[i][0])+' '+str(final[i][1])+' '+str(final[i][2])+'\n')

target_2.close()