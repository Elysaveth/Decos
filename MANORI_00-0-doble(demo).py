import numpy as np
from eccodes import *
from os import listdir
from os.path import isfile, join

import sys

def get_data():

    datafiles = [f for f in listdir("./gribs") if isfile(join("./gribs", f))]
    for file in datafiles:
        if file.startswith("ECMWF_REE_d0007_01.") != True:
            pass
        grib = open("./gribs/" + file, 'rb')
        i = 0
        data_u = []
        data_v = []
        while 1:
            gid = codes_grib_new_from_file(grib)
            if not gid:
                break
            if i > 1:
                data_u = np.array(data_u).reshape(35,28,1)
                data_v = np.array(data_v).reshape(35,28,1)
                data = np.concatenate((data_u, data_v), axis=2)
                data = data.reshape(1,35,28,2)
                codes_release(gid)
                break

            iterid = codes_grib_iterator_new(gid, 0)
            while 1:
                result = codes_grib_iterator_next(iterid)
                if not result:
                    break

                [lat, lon, value] = result
                lat = round(lat, 2)
                lon = round(lon, 2)
                if lat <= 41.4 and lat >= 38.0 and lon >= -3.6 and lon <= -0.9:
                    if i % 2:
                        data_v.append(value)
                    else:
                        data_u.append(value)

            i += 1

            codes_grib_iterator_delete(iterid)
            codes_release(gid)
        grib.close()
        try:
            matrix = np.concatenate((matrix, data))
        except:
            matrix = data
    return (matrix)

if __name__ == "__main__":
    results = get_data()
    sys.exit(print(results, results.shape))
