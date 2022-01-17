from tonemapping import ACESToneMapping, NewToneMapping, ToneMapping, gamma, ReinhardToneMapping, bloom
import time
import cv2
import os
base_dir = os.path.dirname(__file__)

times = 100
result = []

def TestDelay(image):
    t = time.time()
    for _ in range(times):
        t1 = NewToneMapping(image)
    result.append(('Ours',int((time.time()-t)/times*1000)))
    t = time.time()
    for _ in range(times):
        t1 = ToneMapping(image)
    result.append(('Tone Mapping',int((time.time()-t)/times*1000)))
    t = time.time()
    for _ in range(times):
        t1 = ReinhardToneMapping(image,.3)
    result.append(('Reinhard',int((time.time()-t)/times*1000)))
    t = time.time()
    for _ in range(times):
        t1 = ACESToneMapping(image)
    result.append(('ACES',int((time.time()-t)/times*1000)))
    t = time.time()
    for _ in range(times):
        t1 = bloom(image)
    result.append(('ACES+Bloom',int((time.time()-t)/times*1000)))
    t = time.time()
    for _ in range(times):
        t1 = gamma(image,.3)
    result.append(('Gamma',int((time.time()-t)/times*1000)))


if __name__ == '__main__':
    print(os.path.join(base_dir,'test1.jpg'))
    image = cv2.imread(os.path.join(base_dir,'test1.jpg'))
    TestDelay(image)
    with open('result.txt', 'w',encoding='utf-8') as f:
        f.writelines(['%s: %sms\n' % item for item in result])