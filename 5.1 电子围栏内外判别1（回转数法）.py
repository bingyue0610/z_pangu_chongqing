import json
import math
import copy
lnglatlist = []
data = '[{"name":"武汉市三环","points":[{"lng":114.193437,"lat":30.513069},{"lng":114.183376,"lat":30.509211},{"lng":114.188191,"lat":30.505291},{"lng":114.187975,"lat":30.504731},{"lng":114.201773,"lat":30.492782},{"lng":114.213559,"lat":30.48855},{"lng":114.239143,"lat":30.484006},{"lng":114.248341,"lat":30.470062},{"lng":114.267888,"lat":30.470062},{"lng":114.286286,"lat":30.46309},{"lng":114.294335,"lat":30.459105},{"lng":114.298934,"lat":30.459105},{"lng":114.305833,"lat":30.459105},{"lng":114.341478,"lat":30.453128},{"lng":114.422613,"lat":30.462591},{"lng":114.424337,"lat":30.453688},{"lng":114.444316,"lat":30.456303},{"lng":114.466809,"lat":30.466078},{"lng":114.473708,"lat":30.549713},{"lng":114.443813,"lat":30.624326},{"lng":114.407593,"lat":30.683478},{"lng":114.388621,"lat":30.703352},{"lng":114.3616,"lat":30.704843},{"lng":114.311582,"lat":30.678466999999998},{"lng":114.241442,"lat":30.64123},{"lng":114.201773,"lat":30.63079},{"lng":114.182226,"lat":30.63427},{"lng":114.165553,"lat":30.626812},{"lng":114.162679,"lat":30.6109},{"lng":114.170153,"lat":30.59598},{"lng":114.167853,"lat":30.552201},{"lng":114.179351,"lat":30.529309}],"type":0}]'
data = json.loads(data)

if 'points' in data[0]:
    for point in data[0]['points']:
        print(str(point['lng'])+" "+str(point['lat']))
        lnglat = [float(str(point['lng'])), float(str(point['lat']))]
        lnglatlist.append(lnglat)

def windingNumber(point, polly):
    poly = copy.deepcopy(polly)
    poly.append(poly[0])
    px = point[0]
    py = point[1]
    sum = 0
    length = len(poly)-1

    for index in range(0, length):
        sx = poly[index][0]
        sy = poly[index][1]
        tx = poly[index+1][0]
        ty = poly[index+1][1]

        # 点与多边形顶点重合或在多边形的边上
        if (sx - px) * (px - tx) >= 0 and (sy - py) * (py - ty) >= 0 and (px - sx) * (ty - sy) == (py - sy) * (tx - sx):
            return "on"

        # 点与相邻顶点连线的夹角
        angle = math.atan2(sy - py, sx - px) - math.atan2(ty - py, tx - px)

        # 确保夹角不超出取值范围（-π 到 π）
        if angle >= math.pi:
            angle -= math.pi * 2
        elif angle <= -math.pi:
            angle += math.pi * 2
        sum += angle

        # 计算回转数并判断点和多边形的几何关系
    result = 'out' if int(sum / math.pi) == 0 else 'in'
    return result

point = [113.970082, 30.672545]
print(windingNumber(point, lnglatlist))