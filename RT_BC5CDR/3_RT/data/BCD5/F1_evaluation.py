flag=0
# from seqeval.metrics import f1_score, precision_score, recall_score
from sklearn.metrics import f1_score
F=0
import json
all_prediction=[]

label_map = {'Chemical', 'Disease'}



with open("our_method_shot5_output_just_right.json","r",encoding="utf-8") as fr:
    for line in fr.readlines():
        flag = flag + 1
        line=json.loads(line.strip())
        sentence=line["sentence"]
        pre_labels=line["pre_labels"].split(".")[1:]
        gold_labels = line["gold_labels"]

        # print(sentence)
        pre_dic={}
        for pre in pre_labels:
            pre=pre.strip().split("|")
            if len(pre)==3:
                if pre[1]=="True":
                    for g in label_map:
                        if g in pre[2]:
                            pre_dic[pre[0]]=g
        for key, value in pre_dic.items():
            end_=[]
            key_=key.split(" ")
            for k in key_:
                end_.append(k+"|"+value)
            end_=" ".join(end_)
            sentence=sentence.replace(key,end_)

        pr_labels=[]
        for word in sentence.split(" "):
            word=word.split("|")
            if len(word)==1:
                pr_labels.append("O")
            else:
                pr_labels.append(word[1])

        f1 = f1_score(gold_labels.split(" "), pr_labels, average='micro')

        F = F + f1
#
print("the final value is", F / flag)

"""
BCD  shot1---0.9350361398760133
"""

