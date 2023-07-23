import json
import os
import numpy as np
import cv2
import tqdm
from constant import JSON_DIR, MASK_DIR

jsons = os.listdir(JSON_DIR) 

for json_name in tqdm.tqdm(jsons):    
    
    json_path = os.path.join(JSON_DIR, json_name)
    json_file = open(json_path, "r") 
    json_dict = json.load(json_file) 
    
    mask = np.zeros((json_dict["size"]["height"], json_dict["size"]["width"]), dtype=np.uint8)
    
    mask_path = os.path.join(MASK_DIR, json_name[:-5])     
    
    for obj in json_dict["objects"]:
        if obj["classTitle"]=="Freespace":
            cv2.fillPoly(mask, np.array([obj["points"]["exterior"]], dtype=np.int32), color=100)
            if obj["points"]["interior"] != []:
                for interior in obj["points"]["interior"]:
                    cv2.fillPoly(mask, np.array([interior], dtype=np.int32), color=1)

    cv2.imwrite(mask_path, mask.astype(np.uint8))
