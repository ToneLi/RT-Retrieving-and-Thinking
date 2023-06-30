flag=0
# from seqeval.metrics import f1_score, precision_score, recall_score
from sklearn.metrics import f1_score
F=0
import re
import json
all_prediction=[]

label_map = {'SpecificDisease', 'DiseaseClass',   'Modifier','CompositeMention'}



with open("our_method_NCBI_shot5.json", "r", encoding="utf-8") as fr:   # GPTNER_gpt4_output_shot1_NCBI_self_verified.json    GPTNER_gpt4_output_shot1_NCBI.json
    for line in fr.readlines():
        flag = flag + 1
        line=json.loads(line.strip())

        sentence=line["sentence"]
        pre_labels=line["pre_labels"]

        gold_labels = line["gold_labels"]

        # SpecificDisease
        if '</S>' in pre_labels:
            s = r'</S>(.*?)<S>'
            s_entity = re.findall(s, pre_labels)
            # print(s_entity)
            for en in s_entity:
                en=en.split("|")
                if len(en)==3:
                    if en[1]=="True":
                        end_=[]
                        en_=en[0].split(" ")
                        for e in en_:
                            end_.append(e+"|"+"SpecificDisease")
                        end_=" ".join(end_)
                        # print(end_)

                        sentence=sentence.replace(en[0], end_)
        # print(sentence)

        if '</D>' in pre_labels:
        # DiseaseClass
            d = r'</D>(.*?)<D>'
            d_entity = re.findall(d, pre_labels)

            for dn in d_entity:
                end_d = []
                dn = dn.split("|")
                if len(dn)==3:
                    if dn[1]=="True":
                        dn_ = dn[0].split(" ")
                        for d in dn_:
                            end_d.append(d + "|" + "DiseaseClass")

                        end_d = " ".join(end_d)
                        # print(end_d)
                        sentence=sentence.replace(dn[0], end_d)

        if '</M>' in pre_labels:
        # Modifier
            m = r'</M>(.*?)<M>'
            m_entity = re.findall(m, pre_labels)

            for mn in m_entity:
                end_m = []
                mn = mn.split("|")
                if len(mn) == 3:
                    if mn[1] == "True":
                        mn_ = mn[0].split(" ")
                        for m in mn_:
                            end_m.append(m + "|" + "Modifier")
                        end_m = " ".join(end_m)
                        # print(end_m)
                        sentence=sentence.replace(mn[0], end_m)

        if '</C>' in pre_labels:
        # CompositeMention
            c = r'</C>(.*?)<C>'
            c_entity = re.findall(c, pre_labels)

            for cn in c_entity:
                cn = cn.split("|")
                if len(cn) == 3:
                    if cn[1] == "True":
                        end_c = []
                        cn_ = cn[0].split(" ")
                        for c in cn_:
                            end_c.append(c + "|" + "CompositeMention")
                        end_c = " ".join(end_c)
                        # print(end_c)
                        sentence=sentence.replace(cn[0], end_c)

        pr_labels=[]
        for word in sentence.split(" "):
            word=word.split("|")
            if len(word)==1:
                pr_labels.append("O")
            else:
                pr_labels.append(word[1])

        f1 = f1_score(gold_labels.split(" "), pr_labels, average='micro')
        print(f1)

        F = F + f1
#
print("the final value is", F / flag)

# """
# 1 shot   --   the final value is 0.9155684390936619
# """
