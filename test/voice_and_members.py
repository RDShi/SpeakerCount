import os
import winsound
import threading

import matplotlib.pyplot as plt
import time


def plot_line(start_pos, end_pos, member):
    delta_t = 0.1
    start_pos = float(start_pos)
    end_pos = float(end_pos)
    member = int(member)
    try:
        for t in range(int((end_pos-start_pos)/delta_t)):
            obsX = start_pos + delta_t*t
            obsY = member
            plt.plot([start_pos, obsX], [obsY, obsY], "r")
            plt.show()
            time.sleep(delta_t)
            # ax.scatter(obsX, obsY, c='b', marker='.')  # 散点图
            # plt.pause(delta_t)
    except Exception as err:
        print(err)


def main(file_dir):
    list = ["东东", "北北", "冲冲", "民民", "栋栋", "飞飞", "林林", "凯凯"]
    labels = os.listdir(file_dir)
    for label in labels[:2]:
        thread_pool = []
        t = threading.Thread(target=winsound.PlaySound,
                             args=(file_dir + "\\" + label, winsound.SND_NODEFAULT))
        thread_pool.append(t)
        # tmp = label.split("-")
        # start_pos = tmp[1]
        # end_pos = tmp[2]
        # t = threading.Thread(target=plot_line,
        #                      args=(start_pos, end_pos,
        #                            str(list.index(label[-6:-4]))))
        # thread_pool.append(t)
        thread_pool.append(threading.Thread(target=print,
                             args=(str(list.index(label[-6:-4])))))
        for t in thread_pool:
            t.start()
        for t in thread_pool:
            t.join()




if __name__ == "__main__":
    FILE_DIR = '..\\test\\result_wav'

    main(FILE_DIR)
