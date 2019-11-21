## Decoders for ECMWF GRIB 2

–> MANORI_00-0-double:
   Function that returns matrix of shape (X, 35, 28, 2), with X as the number of grib's decodes (create a folder named "gribs" with just the grib's to decode in the location where python is being executed), 35 and 28 the size of the image (latitudes between 41.4° and 38.0° and longitudes between -3.6° and -0.9°), and the two components 100u and 100v
   
–> MANORI_00-0-double(demo): Script that uses MANORI_00-0-double function to print in shell the created matrix and its shape.
