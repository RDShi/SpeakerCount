import json
import wave
import pyaudio
import matplotlib.pyplot as plt
SPEED = 2


def main(file_dir):

    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    chunk = 1024
    fid = wave.open(file_dir + "\\test03.wav", 'rb')
    with open(file_dir + "\\example.json", "r", encoding="utf-8") as json_file:
        test = json.load(json_file)
    pms = fid.getparams()
    nchannels, sampwidth, fs, nframes = pms[:4]
    framerate = int(SPEED * fs)
    p = pyaudio.PyAudio()
    stream = p.open(format=p.get_format_from_width(sampwidth),
                    channels=nchannels,
                    rate=framerate,
                    output=True)
    data = fid.readframes(chunk)
    i = 0
    t = 0
    list_name = ["东东", "北北", "冲冲", "民民", "栋栋", "飞飞", "林林", "凯凯"]
    while data != b"":
        # print(test[str(i)]["start"])
        # print(t)
        if i < 39 and test[str(i)]["start"] < t:
            print(test[str(i)]["label"])
            if test[str(i)]["label"] != "无人声":
                start_pos = int(test[str(i)]["start"])
                end_pos = int(test[str(i)]["end"])
                num = list_name.index(test[str(i)]["label"])
                plt.plot([start_pos, end_pos], [num, num], "r")

            i = i + 1
        t = t + chunk / fs
        stream.write(data)
        data = fid.readframes(chunk)
    stream.stop_stream()
    stream.close()

    scale_ls = range(len(list_name))
    plt.yticks(scale_ls, [l.encode("utf-8").decode("utf-8") for l in list_name])
    plt.show()


if __name__ == "__main__":
    FILE_DIR = '..\\test\\result_wav'
    main(FILE_DIR)