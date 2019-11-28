from __future__ import absolute_import
import pickle
import argparse
import numpy as np
import tensorflow as tf

import network
from actions import command2action, generate_bbox, crop_input

import runway
from runway.data_types import *


@runway.setup(options={"checkpoint" : file(extension=".pkl")})
def setup(opts):
    with open(opts["checkpoint"], 'rb') as f:
        var_dict = pickle.load(f)

    image_placeholder = tf.placeholder(dtype=tf.float32, shape=[None,227,227,3])
    global_feature_placeholder = network.vfn_rl(image_placeholder, var_dict)

    h_placeholder = tf.placeholder(dtype=tf.float32, shape=[None,1024])
    c_placeholder = tf.placeholder(dtype=tf.float32, shape=[None,1024])
    action, h, c = network.vfn_rl(image_placeholder, var_dict, global_feature=global_feature_placeholder,
                                                           h=h_placeholder, c=c_placeholder)
    sess = tf.Session()

    return {"sess" : sess,
            "action" : action,
            "h" : h,
            "c" : c,
            "img_ph" : image_placeholder,
            "global_f_ph" : global_feature_placeholder,
            "h_ph" : h_placeholder,
            "c_ph" : c_placeholder
                }

command_inputs = {"input_image" : image}
command_outputs = {"output_image" : image}

@runway.command("crop_image", inputs=command_inputs, outputs=command_outputs, description="Crop image automatically")
def crop_image(model, inputs):

    im = np.array(inputs["input_image"])
    ip_img = im.astype(np.float32) / 255
    batch_size = len(im)
    
    def auto_cropping(origin_image, sess, action, h, c):
        batch_size = len(origin_image)

        terminals = np.zeros(batch_size)
        ratios = np.repeat([[0, 0, 20, 20]], batch_size, axis=0)
        img = crop_input(origin_image, generate_bbox(origin_image, ratios))

        global_feature = sess.run(model["global_f_ph"], feed_dict={model["img_ph"]: img})
        
        h_np = np.zeros([batch_size, 1024])
        c_np = np.zeros([batch_size, 1024])

        while True:
            action_np, h_np, c_np = sess.run((action, h, c), feed_dict={model["img_ph"]: img,
                                                                    model["global_f_ph"]: global_feature,
                                                                    model["h_ph"]: h_np,
                                                                    model["c_ph"]: c_np})
            ratios, terminals = command2action(action_np, ratios, terminals)
            bbox = generate_bbox(origin_image, ratios)
            if np.sum(terminals) == batch_size:
                return bbox
            
            img = crop_input(origin_image, bbox)    

    xmin, ymin, xmax, ymax = auto_cropping([ip_img - 0.5], model["sess"], model["action"], model["h"], model["c"])[0]
    

    return {"output_image" : im[ymin:ymax, xmin:xmax]}

if __name__ == "__main__":
    runway.run(model_options={"checkpoint" : "vfn_rl.pkl"})




    
