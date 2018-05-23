"""
supervised: process： enroll --> predictv
"""
import os
import json
import itertools
import glob
from utils import read_wav, write_wav, mt_feature_extraction, vad
from model.supervised_model import ModelInterface


def task_enroll(in_dirs, output_model, mt_size=2.0, mt_step=0.2, st_win=0.05):
    """
    输入音频，生成判别模型
    :param in_dirs: 文件夹list，文件夹名为ID
    :param output_model:  输出模型
    :param mt_size: mid-term窗口大小
    :param mt_step: mid-term步长
    :param st_win: short-term窗口大小
    """
    trained_model = ModelInterface()
    in_dirs = [os.path.expanduser(k) for k in in_dirs.strip().split()]
    dirs = itertools.chain(*(glob.glob(d) for d in in_dirs))
    dirs = [d for d in dirs if os.path.isdir(d)]

    if not dirs:
        print("No valid directory found!")

    for directory in dirs:
        label = os.path.basename(directory.rstrip('/'))
        wavs = glob.glob(directory + '/*.wav')

        if not wavs:
            print("No wav file found in", directory)
            continue
        for wav in wavs:
            fs, signal = read_wav(wav)
            print("wav %s has been enrolled", wav)
            # [_, st_features] = mt_feature_extraction(signal, fs, mt_size * fs, mt_step * fs,
            #                                          round(fs * st_win))
            # # VAD：
            #
            # segments = vad(st_features, st_step, smooth_window=0.5, weight=0)
            # for seg in segments:
            #     trained_model.enroll(label, fs, signal[int(seg[0]*fs):int(seg[1]*fs)])
            trained_model.enroll(label, fs, signal)

        trained_model.train()
        trained_model.dump(output_model)


def task_predict(in_files, input_model, mt_size=2.0, mt_step=0.2, st_win=0.05):
    """
    输入音频和判别模型，显示那段时间是谁在说话
    process: Read Wav --> Extract Features --> Remove Outliers --> VAD
     --> Model Interface --> print who
    :param in_files: 文件夹list
    :param input_model: 模型
    :param mt_size: mid-term窗口大小
    :param mt_step: mid-term步长
    :param st_win: short-term窗口大小
    """
    trained_model = ModelInterface.load(input_model)
    for fid in in_files:
        seg_ditail(fid, trained_model, mt_size, mt_step, st_win)


def seg_ditail(fid, trained_model, mt_size, mt_step, st_win):
    """segment ditail"""
    st_step = st_win
    results = {} ###
    fs, signal = read_wav(fid)

    [_, st_features] = mt_feature_extraction(signal, fs, mt_size * fs, mt_step * fs,
                                             round(fs * st_win))

    # VAD：
    segments = vad(st_features, st_step, smooth_window=0.5, weight=0)
    i = 0
    delta_t = 0.4
    for seg in segments:
        if seg[1] - seg[0] > 2*delta_t:
            start_seg = seg[0]
            end_seg = seg[0] + delta_t
            while start_seg < end_seg:
                label = trained_model.predict(fs, signal[int(start_seg * fs):int(end_seg * fs)])
                print(fid, '--', [start_seg, end_seg], '->', label)
                # # ***********
                # write_wav(os.path.join(os.path.pardir, "result", "result_wav",
                #                        os.path.basename(fid)[:-3] + "-" + str(start_seg) + "-" +
                #                        str(end_seg) + "-" + label + ".wav"),
                #           fs, signal[int(start_seg * fs):int(end_seg * fs)])
                results[i] = {"label": label, "start": start_seg, "end": end_seg}
                i = i + 1
                start_seg = end_seg
                end_seg = start_seg + delta_t if start_seg + 2*delta_t < seg[1] else seg[1]
        else:
            label = trained_model.predict(fs, signal[int(seg[0] * fs):int(seg[1] * fs)])
            print(fid, '--', seg, '->', label)
            results[i] = {"label": label, "start": seg[0], "end": seg[1]}
            i = i + 1
            # # ***********
            # write_wav(os.path.join(os.path.pardir, "result", "result_wav",
            #                        os.path.basename(fid)[:-3] + "-" + str(last) + "-" +
            #                        str(seg[0]) + "-静音.wav"),
            #           fs, signal[int(last * fs):int(seg[0] * fs)])
            # write_wav(os.path.join(os.path.pardir, "result", "result_wav",
            #                        os.path.basename(fid)[:-3] + "-" + str(seg[0]) + "-" +
            #                        str(seg[1]) + "-" + label + ".wav"),
            #           fs, signal[int(seg[0] * fs):int(seg[1] * fs)])
            # last = seg[1]

    data = {"video_info": {}, "results": []}
    min_duration = 0.5
    start_seg = results[0]["start"]
    end_seg = results[0]["end"]
    label = results[0]["label"]
    # k = 0 ###
    # test = {}###
    # last = 0 ###
    for j in range(1, i-1):
        if results[j]["start"] - end_seg < min_duration \
                and results[j]["label"] == label:
            end_seg = results[j]["end"]
        else:
            if end_seg - start_seg >= 2*min_duration:
                data["results"].append({"start": start_seg, "end": end_seg, "speaker_id": label})
                # write_wav(os.path.join(os.path.pardir, "result", "result_wav",
                #                        str(int(k/10))+str(k%10)+os.path.basename(fid)[:-3] + "-" + str(last) + "-" +
                #                        str(start_seg) + "-静音.wav"),
                #           fs, signal[int(last * fs):int(start_seg *
                # write_wav(os.path.join(os.path.pardir, "result", "result_wav",
                #                        str(int(k / 10)) + str(k % 10) +os.path.basename(fid)[:-3] +
                #                        "-" + str(start_seg) + "-" +
                #                        str(end_seg) + "-" + label + ".wav"),
                #           fs, signal[int(start_seg * fs):int(end_seg * fs)])

            # if start_seg - last > 0.5:
            #     test[k] = {"label": "无人声", "start": last, "end": start_seg}  ###
            #     k = k + 1
            # if end_seg - start_seg > 0.5:
            #     test[k] = {"label": label, "start": start_seg, "end": end_seg}  ###
            #     k = k + 1
            #     last = end_seg
            start_seg = results[j]["start"]
            end_seg = results[j]["end"]
            label = results[j]["label"]

    # test[k] = {"start": start_seg, "end": end_seg, "label": label}
    # with open("D:\\pro_file\\untitled\\Amber_SpeechSeparation\\test\\result_wav\\example.json",
    #           'w', encoding='utf-8') as fid_exam:
    #     json.dump(test, fid_exam, ensure_ascii=False)
    data["results"].append({"start": start_seg, "end": end_seg, "speaker_id": label})
    write_wav(os.path.join(os.path.pardir, "result", "result_wav",
                           os.path.basename(fid)[:-4] + "-" + str(start_seg) + "-" +
                           str(end_seg) + "-" + label + ".wav"),
              fs, signal[int(start_seg * fs):int(end_seg * fs)])

    with open(os.path.join(os.path.pardir, "result", "test_json",
                           os.path.basename(fid)[:-3] + "json"),
              'w', encoding='utf-8') as json_file:
        print("..\\result\\test_json\\" + os.path.basename(fid)[:-3] + "json->Generated")
        json.dump(data, json_file, ensure_ascii=False)
