from __future__ import\
    absolute_import, print_function, division, unicode_literals
from future.utils import raise_with_traceback, raise_from
# catch exception with: except Exception as e
from builtins import range, map, zip, filter
from io import open
import six
# End: Python 2/3 compatability header small


###############################################################################
###############################################################################
###############################################################################


import imp
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import os
import keras.applications.vgg16 as vgg16
from keras.preprocessing import image
import numpy as np
import glob
from keras import backend as bk
import innvestigate
import innvestigate.utils

from .utils import load_image

base_dir = os.path.dirname(__file__)
utils = imp.load_source("utils", os.path.join(base_dir, "utils.py"))

base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from mlflow import log_metric, log_param, log_artifact, start_run, set_tracking_uri, set_experiment, sklearn

class InnvestigateAnalyzer(object):
    # algms = ['input', 'random', 'gradient', 'gradient.baseline', 'input_t_gradient', 'deconvnet', 'guided_backprop', 'integrated_gradients', 'smoothgrad', 'lrp' ,'lrp.z', 'lrp.z_IB', 'lrp.epsilon', 'lrp.epsilon_IB', 'lrp.w_square', 'lrp.flat', 'lrp.alpha_2_beta_1', 'lrp.alpha_2_beta_1_IB', 'lrp.alpha_1_beta_0', 'lrp.alpha_1_beta_0_IB', 'lrp.z_plus', 'lrp.z_plus_fast', 'lrp.sequential_preset_a', 'lrp.sequential_preset_b', 'lrp.sequential_preset_a_flat', 'lrp.sequential_preset_b_flat','lrp.alpha_beta', 'deep_taylor', 'deep_lift.wrapper']

    # algms = ['lrp', 'lrp.alpha_beta', 'deep_taylor.bounded']   #Error Algorithms
    algms = ['input']
    def __init__(self):
        # self.algms = ['input', 'random', 'gradient', 'gradient.baseline', 'input_t_gradient', 'deconvnet', 'guided_backprop', 'integrated_gradients', 'smoothgrad', 'lrp' ,'lrp.z', 'lrp.z_IB', 'lrp.epsilon', 'lrp.epsilon_IB', 'lrp.w_square', 'lrp.flat', 'lrp.alpha_2_beta_1', 'lrp.alpha_2_beta_1_IB', 'lrp.alpha_1_beta_0', 'lrp.alpha_1_beta_0_IB', 'lrp.z_plus', 'lrp.z_plus_fast', 'lrp.sequential_preset_a', 'lrp.sequential_preset_b', 'lrp.sequential_preset_a_flat', 'lrp.sequential_preset_b_flat','lrp.alpha_beta', 'deep_taylor', 'deep_lift.wrapper']

        # self.algms = ['lrp', 'lrp.alpha_beta', 'deep_taylor.bounded']   #Error Algorithms
        self.status_details = [{
            "condition": "Confidence < 10",
            "status": "Very Poor",
        },{
            "condition": "25 > Confidence >= 10",
            "status": "Below Average",
        },{
            "condition": "50 > Confidence >= 25",
            "status": "Average",
        },{
            "condition": "75 > Confidence >= 50",
            "status": "Above Average",
        },{
            "condition": "90 > Confidence >= 75",
            "status": "Good",
        },{
            "condition": "98 > Confidence >= 90",
            "status": "Very Good",
        },{
            "condition": "Confidence >= 98",
            "status": "Excellent",
        }]
        self.model = vgg16.VGG16()
        self.preprocess = vgg16.preprocess_input
        self.decode_predictions = vgg16.decode_predictions

    def prediction(self, image_name = "zebra.jpg"):
        img_path = os.path.join(base_path, "media/img_source/{}".format(image_name))
        # img_path = "/home/aswin/Projects/ExternalProject/AI/innvestigate/examples/img_source/{}".format(image_name)
        img = image.load_img(img_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = self.preprocess(x)
        features = self.model.predict(x)
        print('Predicted:', self.decode_predictions(features, top=3)[0])

    def analyzer(self, algms = algms, image_name = "img_source/zebra.jpg", activation= "linear"):
        result_data= {}
        files_ip = glob.glob(os.path.join(base_path,'media/input/*'))
        for f in files_ip:
            os.remove(f)
        files_op = glob.glob(os.path.join(base_path,'media/output/*'))
        for f in files_op:
            os.remove(f)


        image_path = os.path.join(base_path, "media/{}".format(image_name))
        image = load_image(image_path, 224)
        # Code snippet.
        plt.imshow(image/255)
        plt.axis('off')
        input_img_url= "media/input/{}_input.png".format(image_name)
        plt.savefig(input_img_url)
        result_data['status_details']= self.status_details
        result_data['input_img_url']= input_img_url
        result_data['input_img_name']= input_img_url.split("/")[-1]
        result_data['trained_output']= []
        for algm in algms:
            try:
                trained_output_temp = {}
                trained_output_temp['algm'] = algm;
                self.apply_analyzer_algm(image, algm, image_name, trained_output_temp, activation)
                result_data['trained_output'].append(trained_output_temp)
            except Exception as ex:
                print("Failed Algorithm: %s" %(algm))
                print("Exception: %s" %(ex))
                continue;
        bk.clear_session()
        return result_data;


    def apply_analyzer_algm(self, image, algm, image_name, trained_output_temp, activation):
        print("Started Algorithm: %s" %(algm))
        # Get model
        model, preprocess = self.model, self.preprocess
        # Strip softmax layer
        # model = innvestigate.utils.model_wo_softmax(model)
        model = innvestigate.utils.model_activation_fn(model, activation[0])
        # sklearn.log_model(sk_model=model,
        #                   artifact_path="model after softmax",
        #                   registered_model_name="innvestigate-vgg16-model")
        kwargs = {}
        if algm == "lrp":
            kwargs["rule"] = "Z" # Ref https://innvestigate.readthedocs.io/en/latest/modules/analyzer.html
            analyzer = innvestigate.create_analyzer(algm, model, **kwargs)
        elif algm == "lrp.alpha_beta":
            analyzer = innvestigate.create_analyzer(algm, model, alpha=1, **kwargs)
        elif algm == "deep_taylor.bounded":
            analyzer = innvestigate.create_analyzer(algm, model, low=1, high=1, **kwargs)
        elif algm in ["pattern.net", "pattern.attribution"]:
            patterns = [x for x in model.get_weights()
                    if len(x.shape) > 1]
            analyzer = innvestigate.create_analyzer(algm, model, patterns= patterns, pattern_type = "relu", **kwargs)
        else:
            analyzer = innvestigate.create_analyzer(algm, model, **kwargs)
        # Add batch axis and preprocess
        x = preprocess(image[None])

        features_b = model.predict(x)
        before_analyzer_prediction= [{"class":v[0],"description":v[1], "confidence":v[2]} for v in (self.decode_predictions(features_b, top=3)[0])]


        # Apply analyzer w.r.t. maximum activated output-neuron
        a = analyzer.analyze(x)

        features_a = model.predict(a)
        after_analyzer_prediction = [{"class":v[0],"description":v[1], "confidence":v[2]} for v in (self.decode_predictions(features_a, top=3)[0])]
        self.set_color_and_status(before_analyzer_prediction, after_analyzer_prediction)

        # Aggregate along color channels and normalize to [-1, 1]
        a = a.sum(axis=np.argmax(np.asarray(a.shape) == 3))
        a /= np.max(np.abs(a))

        # Plot
        plt.imshow(a[0], cmap="seismic", clim=(-1, 1))
        plt.axis('off')
        output_path= "media/output/{}_{}_analysis.png".format(image_name, algm)
        plt.savefig(output_path)

        trained_output_temp["before_analyzer_prediction"]= before_analyzer_prediction
        trained_output_temp["after_analyzer_prediction"]= after_analyzer_prediction
        trained_output_temp["op_img_path"]= output_path
        trained_output_temp["op_img_name"]= output_path.split("/")[-1]

        log_artifact("media/output/{}_{}_analysis.png".format(image_name, algm))
        print("Completed Algorithm: %s" %(algm))
        trained_output_temp["status"]= "success"
        print("Prediction completed")

    def set_color_and_status(self, before_analyzer_prediction, after_analyzer_prediction):
        for b_analyzer_pred in before_analyzer_prediction:
            for a_analyzer_pred in after_analyzer_prediction:
                self.set_status(b_analyzer_pred, a_analyzer_pred)
                if b_analyzer_pred["description"] == a_analyzer_pred["description"]:
                    b_analyzer_pred["bg_color"] = "green"
                    a_analyzer_pred["bg_color"] = "green"
                # else:
                #     b_analyzer_pred["bg_color"] = "red"
                #     a_analyzer_pred["bg_color"] = "red"

    def set_status(self, b_analyzer_pred, a_analyzer_pred):
        b_analyzer_pred["status"] = self.analyzer_pred_status(b_analyzer_pred)
        a_analyzer_pred["status"] = self.analyzer_pred_status(a_analyzer_pred)

    def analyzer_pred_status(self, analyzer_pred):
        status = None
        if analyzer_pred["confidence"]<10:
            status = "Very Poor"
        elif  25> analyzer_pred["confidence"] >=10:
            status = "Below Average"
        elif  50> analyzer_pred["confidence"] >=25:
            status = "Average"
        elif  75> analyzer_pred["confidence"] >=50:
            status = "Above Average"
        elif  90> analyzer_pred["confidence"] >=75:
            status = "Good"
        elif  98> analyzer_pred["confidence"] >=90:
            status = "Very Good"
        elif analyzer_pred["confidence"] >=98:
            status = "Excellent"
        return status
