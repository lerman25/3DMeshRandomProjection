import numpy as np
from VirtualCamera import VirtualCamera
from FootballPitchMeshLoader import PitchMeshLoader
from MeshOpenGL import MeshOpenGL
from random import uniform
import os
import cv2
import json
import pickle
class Dataset:
    def __init__(self) -> None:
        pass
    @staticmethod 
    def generateDataSet(ply_file : str,image_count : tuple, base_vc : VirtualCamera,K_lim : tuple,base_radius: int,radius_lim:tuple, output_folder:str):
        os.makedirs(output_folder,exist_ok=True)
        random_count = image_count[0]
        mesh_loader = PitchMeshLoader(ply_file)
        image_counter = 0
        max_image = image_count[0]*image_count[1] - 1
        dataset = {}
        for i in range(random_count):
            vc = base_vc.copy()
            focal_rand = uniform(K_lim[0],K_lim[1])
            vc._K[0,0] += focal_rand
            vc._K[1,1] += focal_rand 
            meshGL = MeshOpenGL(mesh_loader,vc)
            radius = base_radius + uniform(radius_lim[0],radius_lim[1])
            images = meshGL.generate_random_images(image_count[1],radius=radius)
            for image,t,R in images:
                image_name = str(image_counter).zfill(len(str(max_image)))+'.png'
                image_file = os.path.join(output_folder,image_name)
                dataset[image_name] = {
                    'K' : vc._K,
                    'radius' : radius,
                    'R': R,
                    't' : t,
                }
                cv2.imwrite(image_file,image)
                image_counter+=1
        
        # with open(os.path.join(output_folder+'dataset.json'), 'w',encoding='utf-8') as f:
        #     json.dump(dataset, f)
        with open(os.path.join(output_folder,'dataset'),'wb') as f:
            pickle.dump(dataset,f)
        return dataset

if __name__ == "__main__":
    from VirtualCamera import create_camera_matrix
    PLY_FILE = 'c:/SoccerBaked.ply'
    mesh_loader = PitchMeshLoader(PLY_FILE)

    # Camera
    fx, fy = 572.4114, 573.5704   # Focal lengths
    cx, cy = 325.2611, 242.0489   # Central point
    image_size = (640, 480)       # Image size

    K = create_camera_matrix(fx, fy, cx, cy)
    vc = VirtualCamera(K,background_color= (0,0,0))
    ds = Dataset.generateDataSet(PLY_FILE,(1,10),vc,(-100,+100),30,(-10,+10),"c:/mesh_demo")
    print(ds)