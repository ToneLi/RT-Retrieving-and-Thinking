from simcse import SimCSE
import json
import numpy as np
import os
import faiss
import random
import csv


def read_feature(dir_, prefix):
    info_file = json.load(open(os.path.join(dir_, f"{prefix}.start_word_feature_info.json")))
    features = np.memmap(os.path.join(dir_, f"{prefix}.start_word_feature.npy"),
                         dtype=np.float32,
                         mode="r",
                         shape=(info_file["entity_num"], info_file["hidden_size"]))
    index_file = []
    file = open(os.path.join(dir_, f"{prefix}.start_word_feature_index.json"), "r")
    for line in file:
        index_file.append(int(line.strip()))
    file.close()
    return info_file, features, index_file


def read_mrc_data(dir_):
    # file_name = os.path.join(dir_, f"conll.mrc-ner.{prefix}")
    label_sentence = []
    with open(dir_, "r", encoding="utf-8") as fr:
        for line in fr.readlines():
            data_ = json.loads(line)
            sentence = " ".join(data_["tokens"])
            for tuple_ in data_["entity"]:
                label_sentence.append([tuple_["type"], tuple_["text"], sentence])
            # print(data_["tokens"])
            # print(data_["entity"])
            # print(data_.keys())

            # break
    # print(label_sentence)
    return label_sentence
    # return json.loads(open(dir_, encoding="utf-8"))


def read_idx(dir_, prefix="test"):
    print("reading ...")
    file_name = os.path.join(dir_, f"{prefix}.knn.jsonl")
    example_idx = []
    file = open(file_name, "r")
    for line in file:
        example_idx.append(json.loads(line.strip()))
    file.close()
    return example_idx


def compute_mrc_knn(test_info, test_features, train_info, train_features, train_index, knn_num=32):
    quantizer = faiss.IndexFlatIP(train_info["hidden_size"])
    index = quantizer
    index.add(train_features.astype(np.float32))
    # 10 is a default setting in simcse
    index.nprobe = min(10, train_info["entity_num"])
    index = faiss.index_gpu_to_cpu(index)

    top_value, top_index = index.search(test_features.astype(np.float32), knn_num)

    sum_ = 0
    vis_index = {}
    for idx_, value in enumerate(train_index):
        if value == 0:
            continue
        for i in range(sum_, value + sum_):
            vis_index[i] = idx_
        sum_ += value

    example_idx = [[vis_index[int(i)] for i in top_index[idx_]] for idx_ in range(test_info["entity_num"])]
    example_value = [[float(value) for value in top_value[idx_]] for idx_ in range(test_info["entity_num"])]

    return example_idx, example_value


def compute_simcse_knn(test_mrc_data, train_mrc_data, knn_num, test_index=None):
    sim_model = SimCSE("princeton-nlp/sup-simcse-roberta-large")  # princeton-nlp/sup-simcse-roberta-base

    train_sentence = {}
    train_sentence_index = {}
    for idx_, item in enumerate(train_mrc_data):
        label = item[0]
        label_text = item[1]
        context = item[2]
        # if len(item["start_position"]) == 0:
        #     if label not in train_sentence:
        #         train_sentence[label] = []
        #         train_sentence_index[label] = []
        #     train_sentence[label].append(context)
        #     train_sentence_index[label].append(idx_)
        if label not in train_sentence:
            train_sentence[label] = []
            train_sentence_index[label] = []
        train_sentence[label].append(context)
        train_sentence_index[label].append(idx_)

    train_index = {}
    for key, _ in train_sentence.items():
        # print("---train_sentence[key]",len(train_sentence[key]))
        embeddings = sim_model.encode(train_sentence[key], batch_size=128, normalize_to_unit=True, return_numpy=True)
        quantizer = faiss.IndexFlatIP(embeddings.shape[1])
        index = quantizer
        index.add(embeddings.astype(np.float32))
        # 10 is a default setting in simcse
        index.nprobe = min(10, len(train_sentence[key]))
        index = faiss.index_gpu_to_cpu(index)

        train_index[key] = index

    example_idxs = []
    example_values = []

    if test_index is None:
        for key, value in test_mrc_data.items():
            context = key
            if "|" in value:
                labels = value.split("|")
                # if len(labels)!=0:
                #     print(labels,"-----------")
                # print("------label",label)

                embedding = sim_model.encode([context], batch_size=128, normalize_to_unit=True, keepdim=True,
                                             return_numpy=True)
                example_idx = []
                example_value = []
                for label in labels:
                    top_value, top_index = train_index[label].search(embedding.astype(np.float32), knn_num)
                    example_idx.append([train_sentence_index[label][int(i)] for i in top_index[0]][0])
                    example_value.append([float(value) for value in top_value[0]][0])
                example_idxs.append(example_idx)
                example_values.append(example_value)
            else:
                example_idxs.append([])
                example_values.append([])

        return example_idxs, example_values


def combine_full_knn(test_index, mrc_knn_index, simcse_knn_index):
    results = []
    mrc_idx = 0
    simcse_idx = 0
    for idx_, num in enumerate(test_index):
        if num == 0:
            results.append(simcse_knn_index[simcse_idx])
            simcse_idx += 1
        else:
            knn_num = len(mrc_knn_index[mrc_idx])
            span_ = int(knn_num // num)
            if span_ * num != knn_num:
                span_ += 1
            sub_results = []
            for sub_idx in range(mrc_idx, mrc_idx + num):
                sub_results = sub_results + mrc_knn_index[sub_idx][:span_]
            sub_results = sub_results[:knn_num]
            results.append(sub_results)
            mrc_idx += num

    return results


def random_knn(test_mrc_data, train_mrc_data, knn_num):
    train_sentence = {}
    train_sentence_index = {}
    for idx_, item in enumerate(train_mrc_data):
        label = item["entity_label"]
        context = item["context"]

        if label not in train_sentence:
            train_sentence[label] = []
            train_sentence_index[label] = []
        train_sentence[label].append(context)
        train_sentence_index[label].append(idx_)

    example_idx = []

    for idx_ in range(len(test_mrc_data)):
        context = test_mrc_data[idx_]["context"]
        label = test_mrc_data[idx_]["entity_label"]

        random.shuffle(train_sentence_index[label])

        example_idx.append(train_sentence_index[label][:knn_num])

    return example_idx, None


def write_file(dir_, data):
    file = open(dir_, "w")
    for item in data:
        file.write(json.dumps(item, ensure_ascii=False) + '\n')
    file.close()


def read_ner_pre_train_data(test_dir, ner_model_pre_dir_):
    # file_name = os.path.join(dir_, f"conll.mrc-ner.{prefix}")
    # groud_labels=["SpecificDisease","CompositeMention","Modifier","DiseaseClass"]
    # groud_labels = ["SpecificDisease", "CompositeMention", "Modifier", "DiseaseClass"]
    groud_labels = ["Chemical", 'Disease']  # {'Chemical', 'Disease'}
    csv_reader = csv.reader(open(ner_model_pre_dir_, "r", encoding="utf-8"))
    Dic_ = {}
    for line in csv_reader:
        sentence = line[0]

        predicted_labels = line[1]

        L = ""
        for labels in groud_labels:
            if labels in predicted_labels:
                L = L + labels + "|"
        Dic_[sentence] = L.strip("|")

        # label_sentence.append([sentence,L.strip("|")])
    # print(Dic_)
    new_dic = {}
    with open(test_dir, "r", encoding="utf-8") as fr:
        for line in fr.readlines():
            line = json.loads(line)
            sen_ = " ".join(line["tokens"])
            if sen_ in Dic_:
                new_dic[sen_] = Dic_[sen_]
            else:
                new_dic[sen_] = "None"
    # print(new_dic)
    return new_dic

    # return label_sentence


if __name__ == '__main__':
    # test_mrc_data = read_mrc_data(dir_="data", prefix="test.100")
    # print(test_mrc_data)
    train_mrc_data = read_mrc_data(dir_="data/BCD5/dev_CDR.json")
    test_mrc_data = read_ner_pre_train_data("data/BCD5/sampled_100_test_BCDR.json",
                                            "data/BCD5/CDR_5shot_prompt2_gpt4_output1.csv")
    # print(train_mrc_data)
    index_, value_ = compute_simcse_knn(test_mrc_data=test_mrc_data, train_mrc_data=train_mrc_data, knn_num=5)
    # # print(index_)
    # write_file(dir_="data/BCD5/5shot_test.jsonl", data=index_)