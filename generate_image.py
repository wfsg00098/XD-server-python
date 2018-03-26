import pickle

import numpy as np
from PIL import Image


def process_images(img):
    # img = img.resize((500, 500), Image.ANTIALIAS)
    r, g, b = img.split()
    r_arr = np.array(r)
    g_arr = np.array(g)
    b_arr = np.array(b)

    r_arr = r_arr.astype(np.float64)
    g_arr = g_arr.astype(np.float64)
    b_arr = b_arr.astype(np.float64)

    r_arr -= np.mean(r_arr, axis=0)
    g_arr -= np.mean(g_arr, axis=0)
    b_arr -= np.mean(b_arr, axis=0)

    r1 = Image.fromarray(r_arr).convert("L")
    g1 = Image.fromarray(g_arr).convert("L")
    b1 = Image.fromarray(b_arr).convert("L")
    img1 = Image.merge("RGB", (r1, g1, b1))
    img1 = img1.resize((100, 100), Image.ANTIALIAS)
    img1 = img1.convert("L")
    return img1


data = pickle.load(open("data\\teX.pkl", 'rb'))


for i in range(11):
    print(str(i))
    img = Image.open("data\\test\\" + str(i) + ".jpg")
    img = process_images(img)
    temp = []
    temp.append(np.array(img))
    temp = np.asarray(temp, dtype='float64')
    temp /= 256
    data = np.concatenate((data, temp), axis=0)





print(data.shape)

pickle.dump(data, open("data\\teX.pkl", 'wb'))
