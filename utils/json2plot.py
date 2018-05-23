#-*- coding: utf-8 -*-
import os
import json
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False #用来正常显示负号
LIST = [u"background", u"鼓掌", u"贾珈", u"老大", u"立思辰", u"笑声", u"喻纯", u"许斌"]
JSON_FILE = '..\\result\\test_json'
for fid in os.listdir(JSON_FILE):
    with open(JSON_FILE+"/"+fid, 'r', encoding='utf-8') as json_file:
        data = json.load(json_file)
        for result in data["results"]:
            delta = int(fid[24:26])*60*60+int(fid[27:29])*60
            delta_delta = (int(fid[22])-1)*(1*60*60+35*60+3)
            # plt.figure(int(fid[22]))
            plt.plot([delta_delta + delta+result["start"], delta_delta + delta+result["end"]],
                     [LIST.index(result["speaker_id"]), LIST.index(result["speaker_id"])],
                     "r")
scale_ls = range(len(LIST))
# plt.figure(1)
plt.yticks(scale_ls, [l.encode("utf-8").decode("utf-8") for l in LIST])
# plt.figure(2)
# plt.yticks(scale_ls, [l.encode("utf-8").decode("utf-8") for l in LIST])
plt.show()


