
import numpy as np
import matplotlib.pyplot as plt
import cv2
from perlin_numpy import generate_fractal_noise_2d, generate_perlin_noise_2d
import pathlib

def make_fractal_image(res, octaves, persistence):
    np.random.seed(0)

    lacunarity = 2 # for simplicity, always use 2, otherwise noise is spread too far, and calculations in choosing target divisions and sizes break

    shape = lacunarity**(octaves-1)*res
    shape = (shape,shape)
    res = (res,res)

    noise = generate_fractal_noise_2d(shape, res, octaves=octaves, persistence=persistence, lacunarity=lacunarity)

    if noise.max() == noise.min():
        print('Error: noise max and noise min are equal')
        exit()

    # normalize for opencv viewing as 16-bit image
    # smaller than 16 bit would lose a lot of data, which we don't want to do before normalizing
    noise -= noise.min()
    noise *= ((-1+2.0**16)/noise.max())
    noise = noise.astype('uint16') # unit16 required for CLAHE

    # use a grid size that matches the largest grid of noise, to make sure we aren't normalizing it out. fewer grids would also be acceptable
    clahe = cv2.createCLAHE(clipLimit=0.0, tileGridSize=res)
    clahe_noise = clahe.apply(noise)

    return (noise, clahe_noise)



'''
find min and max grain size
min_grain_size = 1 pixel in real world dimensions against the target


for a target that is some distance wide, resolution should match max pixel width/target width

number of octaves should be how many divisions of the max pixel width by 2 becomes smaller than the min pixel width

'''

# find the minimum and maximum pixel sizes when projected onto the scoring target for each camera at max and min distance

#sizes array has a tuple to represent each camera
# (focal length in pixels, optimal pixel size in mm, (min distance, max distance))
# pixel focal lengths pulled from P matrix of extrinsics calibration, not using ideal pixel size and ideal focal length
sizes = []
sizes.append((870.0, 0.0055, (1.5,7.0))) # S21, 4.8mm
sizes.append((975.0, 0.0055, (1.5,7.0))) # S21, 5.0mm
sizes.append((1165.0, 0.0055, (2.0,8.5))) # S21, 6.0mm
sizes.append((1550.0, 0.0055, (3.0 ,10.0))) # S21, 8.0mm

sizes.append((625.0, 0.0055, (0.5,4.0))) # S7, 4.8mm, spider cameras are far more zoomed out than others
sizes.append((1225.0, 0.0055, (1.0,4.0))) # S7, 6.5mm

sizes.append((1200.0, 0.003, (2.0, 10.0))) #S27/S30, 3.9mm, fixed focal lengths
sizes.append((1300.0, 0.003, (2.0, 10.0))) #S27/S30, 3.9mm, fixed focal lengths, not sure if we might use this focal length

pixel_sizes = []
for size in sizes:
    px_size = size[1]/1000.0
    focal_m = size[0] * px_size
    for distance in size[2]:
        pixel_sizes.append(px_size * distance / focal_m)

min_size = min(pixel_sizes)
max_size = max(pixel_sizes)

print('found minimum pixel size of ', min_size*1000, 'mm, and max pixel size of ', max_size*1000, 'mm\n')



# target is assumed square. use largest dimension then crop later if non-square target is needed
target_size = 1.5 # meters

# pitch of the noise should match roughly the maximum pixel size, otherwise it is wasted
resolution = int((target_size / max_size) + 0.5)

# need to use enough octaves so the smallest noise is smaller than the minimum pixel size
octaves = 1
max_size_tmp = max_size
while max_size_tmp > min_size:
    octaves += 1
    max_size_tmp /= 2
min_size = max_size_tmp

print('low frequency noise will have min resolution of ', max_size*1000, 'mm')
print('high frequency noise will have min resolution of ', min_size*1000, 'mm\n')
print('resolution', resolution)
print('octaves', octaves, '\n')

clahes = []
noises = []
# test persistance from 0.0-1.0
# persistence of 1.0 means high frequency noise equally weighted to low frequency
# persistence of 0.0 means only the lowest frequency noise layer is kept
# 1.0 will mean high frequency dominates, and when cameras are at far distances, target will appear gray
# 0.0 will mean low frequency dominates, and when camera is close up, there will not be enough noise for good disparity
for persistence_int in range(11):
    persistence = persistence_int/10.0
    dir_name = 'persistence_'+str(persistence)
    pathlib.Path(dir_name).mkdir(exist_ok=True)

    noise, clahe_noise = make_fractal_image(resolution, octaves, persistence)

    print('saving persistence ', persistence)
    cv2.imwrite(dir_name + '/raw_noise.png', noise)
    cv2.imwrite(dir_name + '/equalized_noise.png', clahe_noise)
    # show what the furthest camera view will look like during scoring
    cv2.imwrite(dir_name + '/zoomed_out_cam_view.png', cv2.resize(clahe_noise, (resolution, resolution)))
    # show what the closest camera view will look like during scoring
    cv2.imwrite(dir_name + '/zoomed_in_cam_view.png', clahe_noise[0:resolution, 0:resolution])

print('each image pixel should be ', max_size*1000, 'mm when printed')
print('total image size should be ', target_size, 'x', target_size, ' meters when printed')



#
#cv2.waitKey(0)
