import json

ass=[]
with open("GPTNER_gpt4_output_shot1_BCD5_self_verified.json","r") as fr1:
    for line in fr1.readlines():
        s1=json.loads(line.strip())["sentence"]
        ass .append(s1)

# print(len(ass))
with open("sampled_100_test_BCDR.json","r") as fr:
    for line in fr.readlines():
        s=" ".join(json.loads(line.strip())["tokens"])

        if s not in ass:
            print(s)

# print(len(ass))
