import json
import os


with open("./mpii/annot_mpii.json") as anno_file:
    alphajson = json.load(anno_file)

with open("./mpii/mpii_annotations.json") as anno_file:
    uniposejson = json.load(anno_file)

'''
get bbox from alphapose annotations
'''
pid = 0
bboxlist = []
dic = {'key':'value'}
for iter in alphajson['annotations']:
    iid =str( iter['image_id'])
    if pid != iid:
        bboxlist = [] 
        bboxlist.append(iter['bbox'])
        pid = iid
    
    else:
        bboxlist.append(iter['bbox'])
    
    index = '0' + iid
    dic[index] = bboxlist
# print(dic)

'''get iid from unipose annotations
'''

# print(uniposejson[0])


for iter in uniposejson:
    
    tid = iter['img_paths'].split('.')[0]
    
    iter['bbox'] = dic.get(tid)
    
print(uniposejson)


with open('./mpii_new.json', 'w') as outfile:
    json.dump(uniposejson, outfile)