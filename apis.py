import pygame
import pygame.gfxdraw
from itertools import cycle
import itertools
try:
    import pyzed.camera as zcam
    import pyzed.defines as sl
    import pyzed.types as tp
    import pyzed.core as core
except ImportError:
    print("PyZed is not installed.")

try: 
    from sklearn.cluster import KMeans, DBSCAN, MeanShift, estimate_bandwidth
    from sklearn.metrics import pairwise_distances_argmin
    from sklearn.datasets import load_sample_image
    from sklearn.utils import shuffle
    from sklearn import metrics
    from sklearn.preprocessing import StandardScaler
except ImportError : 
    print('Sklearn not installed !')

import numpy as np
import time
import sys
import yaml

from datetime import datetime, timedelta
from functools import wraps


#cluster definition 
type_cluster = [('position', int, 3),
                ('height', int, 1),
                ('energy', int, 1),
                ('weight', float, 1),
                ('speed', int, 1),
                ('circle', int, 1),
                ('jump_up', bool, 1),
                ('jump_down', bool, 1),
                ('jump_max', int, 1),
                ('age', int, 1)]
type_ripple = [('position', int, 3),
                ('age', int, 1),
                ('energy', int, 1)]


def estimate_weight(image_array):
    weight = len(image_array)*config['weight']['g']
    #print(weight)
    #return max(min(weight,config['weight']['max']), config['weight']['min'])
    return 0.13#*config['weight']['g']


def estimate_height(image_array):
    return config['scene']['depth'] - np.percentile(image_array[:,2], 1)


def estimate_position(image_array):
    x, y = np.mean(image_array[:,:2], axis=0)
    z = np.percentile(image_array[:,2], 3)
    return np.array([x, y, z])


def get_height(clusters):
    return config['scene']['depth'] - clusters['position'][:,2]


def get_energy(clusters, k):
    height = get_height(clusters)[k]
    return max(config['min_energy'], min(config['max_energy'], (height+100)*.13))



def get_point_cloud(zed, point_cloud):
    if zed.grab(runtime_parameters) == tp.PyERROR_CODE.PySUCCESS:
        zed.retrieve_measure(zed_pc, sl.PyMEASURE.PyMEASURE_XYZRGBA)
        np.copyto(point_cloud, zed_pc.get_data())
        return True
    return False


def clustering(image_array):
    if image_array.size == 0:
        return [],[]

    try:
        db = DBSCAN(eps=config['cluster']['epsilon'], min_samples=config['cluster']['min_samples']).fit(image_array)
    except ValueError:
        print("NaN in clustering")
        return [], []

    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

    return labels, core_samples_mask


def scene2screen(coords, dtype=int):
    """ Convert a numpy array of coordinates from screen to screen size"""
    return np.array(size_screen/size_scene*(coords - origin_scene), dtype=dtype)




def find_clusters(image_array):
    """ Find clusters of object inside scene """
    labels, core_samples_mask = clustering(image_array)
    xyz = []
    unique_labels = set(labels)
    for k in unique_labels:
        class_member_mask = (labels == k)
        image = image_array[class_member_mask & core_samples_mask]
        if image.size:
            xyz.append(image)

    return xyz


def initialize_clusters(cluster):
    new_cluster = np.zeros(1, dtype=type_cluster)
    new_cluster['position'] = np.mean(cluster, axis=0)
    new_cluster['height'] = estimate_height(cluster)
    new_cluster['weight'] = estimate_weight(cluster)
    new_cluster['energy'] = get_energy(new_cluster, 0)
    new_cluster['circle'] = new_cluster['energy'] + config['radius_level']
  #  new_cluster['date'] = new_cluster['energy'] + config['radius_level']
    return new_cluster



def update_clusters(image_array, clusters):
    xyz = find_clusters(image_array)

    # matches centroids, delete unmatches clusters and initialize new clusters
    matches = np.zeros(len(clusters), dtype=bool)
    new_clusters = []
    for image_array in xyz:
        centroid = estimate_position(image_array)
        try:
            distances = np.linalg.norm(clusters['position'][:,:2]-centroid[:2], axis=1)
            nearest_cluster = distances.argmin()
            matches[nearest_cluster] = True
            min_dist = distances[nearest_cluster]
        except ValueError:
            # Case when there is no cluster
            min_dist = config['max_dist_cluster']+1
        if min_dist > config['max_dist_cluster']:
            new_clusters.append(initialize_clusters(image_array))
        else:
            clusters['position'][nearest_cluster] = centroid
            # clusters['weight'][nearest_cluster] = (clusters['weight'][nearest_cluster]*
            #                                     clusters['age'][nearest_cluster]+
            #                                     estimate_weight(image_array)) / \
            #                                     (clusters['age'][nearest_cluster]+1)
            # clusters['height'][nearest_cluster] = (clusters['height'][nearest_cluster]
            #                                     *clusters['age'][nearest_cluster]
            #                                     + estimate_height(image_array))/ \
            #                                     (clusters['age'][nearest_cluster]+1)
            clusters['energy'][nearest_cluster] = get_energy(clusters, nearest_cluster)
    clusters = np.concatenate((clusters[matches], *new_clusters), axis=0)
    clusters['age'] += 1
    return xyz, clusters


def update_ripples(clusters, ripples):
    """ detect a jump, activate or kill ripples """

    delta = get_height(clusters) - clusters['height']
    detect_jump = delta > config['thresh_jump']
    clusters['jump_max'][detect_jump] = np.maximum.reduce([clusters['jump_max'][detect_jump], delta[detect_jump]])

    # start jump
    clusters['jump_up'][detect_jump & ~clusters['jump_up']] = True
    if (detect_jump & ~clusters['jump_up']).any():
        print("Jumping")

    # finish jump
    detect_landing = delta < 1
    is_jump_finished = detect_landing & clusters['jump_up']
    Nover = len(clusters['jump_max'][is_jump_finished])
    new_ripples = np.zeros(Nover, dtype=type_ripple)
    new_ripples['position'][:,:2] = clusters['position'][is_jump_finished][:,:2]
    new_ripples['energy'] = clusters['jump_max'][is_jump_finished]
    clusters['jump_max'][is_jump_finished] = 0
    clusters['jump_up'][is_jump_finished] = False


    # destroy ripples
    subset_ripples = ripples[np.where(ripples['age'] < config['ripples']['max_age'])]
    ripples = np.concatenate((subset_ripples, new_ripples), axis=0)

    return ripples


def get_data(zed):
    """ Get point_cloud image """
    point_cloud = np.zeros((720,1280,4), dtype=float)
    while not get_point_cloud(zed, point_cloud):
        pass

    # filter data
    crop = config['scene']['crop']
    inc = config['scene']['decimation']
    ground_level = config['scene']['depth']
    #point_cloud = point_cloud[crop[0]:crop[1]:inc, crop[2]:crop[3]:inc, :3]
    point_cloud = point_cloud[::inc, ::inc, :3]
    filtered = ~np.isnan(point_cloud).any(axis=2)
    image_array = point_cloud[filtered, :]
    image_array = image_array[np.where((image_array[:,2] < ground_level) & \
                      (image_array[:,0] > crop[0]) & (image_array[:,0] <   \
                      crop[1]) & (image_array[:,1] > crop[2]) &            \
                      (image_array[:,1] < crop[3]))]
    #image_array = image_array[np.where(image_array[:,2] < ground_level)]
    return image_array


def draw_image_array(screen, image_array):
    """ Draw the point cloud inside an image array. This is for debug purpose """

    for point in image_array:
        x, y, z = scene2screen(point)
        if np.isnan((x,y,z)).any():
            continue
        color = [255-(config['scene']['depth']-z),]*3
        radius = 3
        pygame.gfxdraw.aacircle(screen, x, y, radius, COLORS['point_cloud'])


def draw_image_arrays(screen, *images_array):
    for image in images_array:
        draw_image_array(screen, image)


def update_data(clusters, ripples, zed):   # function returns a 2D data array
    ripples['age'] += config['ripples']['aging']
    image_array = get_data(zed)

    xyz, clusters = update_clusters(image_array, clusters)
    ripples = update_ripples(clusters, ripples)

    if config['display_cluster']:
        draw_image_arrays(screen, *xyz)

    return clusters, ripples


def draw(clusters, ripples, screen):

    for cluster in clusters:
        
        x, y, z = scene2screen(cluster['position'])
        color = [255*(cluster['energy']-config['min_energy'])/(config['max_energy']-config['min_energy']), 0, 255*(config['max_energy']-cluster['energy'])/(config['max_energy']-config['min_energy'])]
        c_level = [255, 25, 25]
        pygame.gfxdraw.aacircle(screen, x, y,  cluster['energy'], color)#COLORS['energy'])
        pygame.gfxdraw.aacircle(screen, x, y,  cluster['circle'], c_level)#COLORS['energy'])
        pygame.gfxdraw.filled_circle(screen, x, y,  cluster['circle'], c_level)
        pygame.gfxdraw.filled_circle(screen, x, y,  cluster['circle']-5, COLORS['background'])
        if config['cluster']['fill']:
            pygame.gfxdraw.filled_circle(screen, x, y,  cluster['energy'], color)#COLORS['energy'])

    for ripple in ripples:
        x, y, z = scene2screen(ripple['position'])
        for i in range(config['ripples']['number']):
            color = max(255-(i+config['ripples']['speed'])*ripple['age'], 50)
            #radius = ripple['energy'] + (i+config['ripples']['speed'])*ripple['age'] + i*config['ripples']['delay']
            radius =  (i+config['ripples']['speed'])*ripple['age'] + i*config['ripples']['delay']
            pygame.gfxdraw.aacircle(screen, x, y, radius, (color,)*3)

    pygame.display.flip()



if __name__ == '__main__':

    # Parameters
    ############

    with open("config.yaml", 'r') as stream:
        try:
            config = yaml.load(stream)
        except yaml.YAMLError as exc:
            print("Can't open config file")
            sys.exit(1)

    height = config["screen"]["height"]
    width = config["screen"]["width"]
    size_screen = np.array([width, height, config["scene"]["depth"]])
    size_scene = np.array([config["scene"]["width"], config["scene"]["height"], config["scene"]["depth"]])
    origin_scene = np.array(config["scene"]["origin"])
    COLORS = config['colors']


    # Initialize ZED
    ################

    zed = zcam.PyZEDCamera()
    init_params = zcam.PyInitParameters()
    init_params.depth_mode = sl.PyDEPTH_MODE.PyDEPTH_MODE_PERFORMANCE  # Use PERFORMANCE depth mode
    init_params.coordinate_units = sl.PyUNIT.PyUNIT_MILLIMETER  # Use milliliter units (for depth measurements)
    err = zed.open(init_params)
    if err != tp.PyERROR_CODE.PySUCCESS:
        print("Can't open the camera")
        exit(1)
    runtime_parameters = zcam.PyRuntimeParameters()
    runtime_parameters.sensing_mode = sl.PySENSING_MODE.PySENSING_MODE_STANDARD  # Use STANDARD sensing mode
    zed_pc = core.PyMat()

    clusters, ripples = np.zeros(0, dtype=type_cluster), np.zeros(0, dtype=type_ripple)


    # Launch animation
    ##################

    pygame.init()

    #screen = pygame.display.set_mode([width, height], pygame.FULLSCREEN)
    screen = pygame.display.set_mode([width, height], pygame.RESIZABLE)
    pygame.display.set_caption("Trampo Filipe & Jben")
    #pygame.mouse.set_visible(False)
    pygame.mouse.set_visible(True)
    done = False
    clock = pygame.time.Clock()

    while not done:
        # 30 FPS
        clock.tick(30)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

        screen.fill(COLORS['background'])
        clusters, ripples = update_data(clusters, ripples, zed)
        draw(clusters, ripples, screen)
        #print(" %d" % len(clusters))

    # Close
    #######
    if zed:
        zed.close()
