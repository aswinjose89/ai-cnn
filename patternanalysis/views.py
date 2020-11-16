from django.shortcuts import render
from django.views.generic import TemplateView
from django.conf import settings
from rest_framework.response import Response
from rest_framework.views import APIView
import logging
logger = logging.getLogger(__package__)
# Create your views here.
from mlflow import log_metric, log_param, log_artifact, set_tracking_uri, set_experiment, start_run, end_run
import mlflow.keras
from django.core.files.storage import FileSystemStorage
import glob
import os
from os import listdir
from os.path import isfile, join
import io
import urllib, base64
from keras.preprocessing import image
from keras.models import load_model
from matplotlib import pyplot as plt
import numpy as np
from keras import models
from aivalidation.views import BaseView
import tensorflow as tf
import sys
np.set_printoptions(threshold=sys.maxsize)

from .innvestigate_analyzer import InnvestigateAnalyzer

base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
class ImageAnalyzerView(BaseView):
    template_name = "innvestigator.html"
    activation_fn = ["relu", "softmax", "linear", "elu","softplus","softsign","hard_sigmoid"]
    algms = ['input', 'random', 'gradient', 'gradient.baseline', 'input_t_gradient', 'deconvnet', 'guided_backprop', 'integrated_gradients', 'smoothgrad', 'lrp' ,'lrp.z', 'lrp.z_IB', 'lrp.epsilon', 'lrp.epsilon_IB', 'lrp.w_square', 'lrp.flat', 'lrp.alpha_2_beta_1', 'lrp.alpha_2_beta_1_IB', 'lrp.alpha_1_beta_0', 'lrp.alpha_1_beta_0_IB', 'lrp.z_plus_fast', 'lrp.sequential_preset_a', 'lrp.alpha_beta', 'deep_taylor', 'deep_lift.wrapper', 'pattern.net', 'pattern.attribution']
    def get_context_data(self, *args, **kwargs):
        context = super(ImageAnalyzerView, self).get_context_data(*args, **kwargs)
        context["algorithms"] = self.algms
        context["activation_fn"] = self.activation_fn
        context["sample_img"] = self.get_file_path("sample_img")
        return context

    def post(self, request, *args, **kwargs):
        context = self.get_context_data(*args, **kwargs)
        post_data = request.POST
        if request.FILES.get('investigator_file') or post_data.get("sample_img"):
            try:
                myfile = request.FILES.get('investigator_file')
                innvestAlgm = post_data.getlist("innvestAlgm")
                activationName = post_data.getlist("activationName")
                sampleImg = post_data.get("sample_img")
                if myfile:
                    fs = FileSystemStorage()
                    filename = fs.save(myfile.name, myfile)
                    uploaded_file_url = fs.url(filename)
                    src_img_path= None
                else:
                    sampleImg = eval(post_data.get("sample_img"))
                    uploaded_file_url = sampleImg.get("path")
                    filename = sampleImg.get("file")
                    src_img_path= "sample_img/"
                remote_server_uri = settings.MLFLOW_URL # set to your server URI
                set_tracking_uri(remote_server_uri)
                set_experiment("/ai-back-propagation")
                with start_run():
                    mlflow.keras.autolog()
                    log_param("Selected Algorithms", innvestAlgm)
                    log_param("Activation Function Name", activationName)
                    log_param("image_name", filename)
                    innvestigateObj = InnvestigateAnalyzer()
                    result_data = innvestigateObj.analyzer(algms = innvestAlgm, image_name = filename, src_img_path= src_img_path, activation= activationName)
                    context["original_img_url"] = uploaded_file_url
                    context["input_img_url"] = glob.glob('media/input/*')
                    context["output_img_url"] = glob.glob('media/output/*')
                    context["result_data"] = result_data
                    end_run()
                    return render(request, "innvestigator.html", context)
            except Exception as ex:
                logger.error("ClassName: ImageAnalyzerView, Exception: %s" % ex)
        return render(request, "innvestigator.html", context)


class ModelNeuronView(BaseView):
    template_name = "neurons.html"
    fs = FileSystemStorage(location=settings.MEDIA_ROOT)
    def get_context_data(self, *args, **kwargs):
        context = super(ModelNeuronView, self).get_context_data(*args, **kwargs)
        context["trained_models"] = self.get_file_path("sample_models")
        context["sample_img"] = self.get_file_path("sample_img")
        return context

    def get_model_files(self):
        model_path = '{}/{}'.format(settings.MEDIA_ROOT, "sample_models")
        model_files = [f for f in listdir(model_path) if isfile(join(model_path, f))]
        return model_files

    def post(self, request, *args, **kwargs):
        context = self.get_context_data(*args, **kwargs)
        if request.method == 'POST':
            context["status"] = "success"
            post_data = request.POST
            input_img = request.FILES.get('input_img')
            model_file_list = request.FILES.getlist('model_file')
            trainedModelName = post_data.getlist("trainedModelName")
            model_weight_file = request.FILES.get('model_weight_file')
            trainedModelWeight = post_data.get("trainedModelWeight")
            sampleImg = post_data.get("sample_img")
            if (input_img and model_file_list) or (sampleImg and trainedModelName):
                try:
                    if model_file_list:
                        uploaded_model_file_url_list, uploaded_ip_img_file_url= self.file_storage(input_img, model_file_list)
                    elif trainedModelName:
                        sampleImg = eval(post_data.get("sample_img"))
                        uploaded_ip_img_file_url = '{}/{}/{}'.format(settings.MEDIA_ROOT, "sample_img", sampleImg.get('file'))
                        uploaded_model_file_url_list= []
                        for model in trainedModelName:
                            model_path = '{}/{}/{}'.format(settings.MEDIA_ROOT, "sample_models", eval(model).get('file'))
                            uploaded_model_file_url_list.append(model_path)

                    all_model_neurons= []
                    for uploaded_model_file_url in uploaded_model_file_url_list:
                        #Loading model
                        trained_model = load_model(uploaded_model_file_url)
                        img_path = uploaded_ip_img_file_url
                        img_width, img_height = trained_model.input_shape[1], trained_model.input_shape[2]
                        img = image.load_img(img_path, target_size=(img_width, img_height))
                        img_tensor = image.img_to_array(img)
                        img_tensor = np.expand_dims(img_tensor, axis=0)
                        img_tensor /= 255.

                        layer_outputs = [layer.output for layer in trained_model.layers]
                        activation_model = models.Model(inputs=trained_model.input, outputs=layer_outputs)
                        activations = activation_model.predict(img_tensor)
                        model_neurons= []
                        for i in range(len(activation_model.layers)):
                            layer= activation_model.layers[i]
                            layer_config= layer.get_config()
                            if "filters" in layer_config: #It help to extract only convolution layer
                                temp= {}
                                temp["filters"]= layer.filters
                                temp["name"]= layer.name
                                temp["shape"] = layer.shape
                                temp["layer_neurons"]= []
                                for val in range(layer.filters):
                                    plt.matshow(activations[i][0, :, :,val], cmap='viridis')
                                    plt.title('Neuron %s' %(val+1,))
                                    plt.axis('off')
                                    fig= plt.gcf()
                                    buf= io.BytesIO()
                                    fig.savefig(buf, format="jpeg")
                                    buf.seek(0)
                                    string= base64.b64encode(buf.read())
                                    uri= urllib.parse.quote(string)
                                    temp["layer_neurons"].append({"plot": uri, "weights": activations[i][0, :, :,val]})
                                model_neurons.append(temp)
                            elif layer.name.startswith("dense") or layer.name.startswith("flatten"):
                                temp = {}
                                temp["name"] = layer.name
                                temp["shape"] = layer.shape
                                temp["layer_neurons"] = []
                                plt.matshow(activations[i], cmap='viridis')
                                plt.title('Neuron %s' % (val + 1,))
                                plt.axis('off')
                                fig = plt.gcf()
                                buf = io.BytesIO()
                                fig.savefig(buf, format="jpeg")
                                buf.seek(0)
                                string = base64.b64encode(buf.read())
                                uri = urllib.parse.quote(string)
                                temp["layer_neurons"].append({"plot": uri, "weights": str(activations[i][0, :, :,val])})

                                model_neurons.append(temp)

                        model_file_name= uploaded_model_file_url.split("/")[-1]
                        model_plot_path= self.model_plot(trained_model, model_file_name.split(".")[0])
                        all_model_neurons.append({"model_neurons": model_neurons, "model_plot_path": model_plot_path, "model_file_name": model_file_name})
                        # plt.matshow(activations[12][0, :, :,1], cmap='viridis')
                    context["all_model_neurons"]= all_model_neurons
                except Exception as ex:
                    context["msg"] = "Uploaded model file is not supported. Make sure you are uploading Keras sequential model set.."
                    context["status"] = "error"
                    logger.error("ClassName: ModelNeuronView,  Exception: %s" % ex)
            else:
                context["msg"] = "Please upload image and model file or choose trained model and sample image"
                context["status"] = "error"
            # if model_weight_file or trainedModelWeight:
            #     if model_weight_file:
            #         uploaded_model_weight_file_url = self.save_model(model_weight_file)
            #     else:
            #         uploaded_model_weight_file_url = '{}/{}/{}'.format(settings.MEDIA_ROOT, "sample_models", eval(trainedModelWeight).get('file'))
            #     loaded_model = load_model(uploaded_model_weight_file_url)
            #     model_weights= loaded_model.get_weights()
            #     model_neurons= []
            #     for i in range(len(loaded_model.layers)):
            #         layer= loaded_model.layers[i]
            #         layer_config= layer.get_config()
            #         if "filters" in layer_config: #It help to extract only convolution layer
            #             temp= {}
            #             temp["filters"]= layer.filters
            #             temp["name"]= layer.name
            #             temp["layer_neurons"]= []
            #             for val in range(layer.filters):
            #                 plt.figure()
            #                 plt.matshow(activations[i][0, :, :,val], cmap='viridis')
            #                 plt.axis('off')
            #                 fig= plt.gcf()
            #                 buf= io.BytesIO()
            #                 fig.savefig(buf, format="jpeg")
            #                 buf.seek(0)
            #                 string= base64.b64encode(buf.read())
            #                 uri= urllib.parse.quote(string)
            #                 temp["layer_neurons"].append(uri)
            #             model_neurons.append(temp)
            # if False:
            #     from keras.utils.vis_utils import plot_model
            #     plot_model(trained_model, to_file='media/models/model_plot.png', show_shapes=True, show_layer_names=True)

        return render(request, "neurons.html", context)

    def model_plot(self, trained_model, model_file_name):
        from keras.utils.vis_utils import plot_model
        plot_path= 'media/models/plots/{}_plot.png'.format(model_file_name)
        plot_model(trained_model, to_file=plot_path, show_shapes=True, show_layer_names=True)
        return plot_path

    def img_storage(self, input_img):
        uploaded_ip_img_file_url = self.save_img(input_img) # saving input image
        return uploaded_ip_img_file_url

    def file_storage(self, input_img, model_file_list):
        uploaded_model_file_url_list= []
        for model_file in model_file_list:
            uploaded_model_file_url = self.save_model(model_file)
            uploaded_model_file_url_list.append(uploaded_model_file_url)

        # input_img_path = os.path.join(base_path, "media/input/{}".format(input_img.name))
        # ip_img_filename = self.fs.save(input_img_path, input_img)
        uploaded_ip_img_file_url = self.save_img(input_img) # saving input image
        return uploaded_model_file_url_list, uploaded_ip_img_file_url

    def save_model(self, model_file):
        model_file_path = os.path.join(base_path, "media/models/{}".format(model_file.name))
        model_filename = self.fs.save(model_file_path, model_file) #saving model
        uploaded_model_file_url = model_filename
        return uploaded_model_file_url

    def save_img(self, input_img):
        input_img_path = os.path.join(base_path, "media/input/{}".format(input_img.name))
        ip_img_filename = self.fs.save(input_img_path, input_img)
        uploaded_ip_img_file_url = ip_img_filename # saving input image
        return uploaded_ip_img_file_url


class ModelWeightView(BaseView):
    template_name = "modal_weight.html"
    fs = FileSystemStorage(location=settings.MEDIA_ROOT)
    def get_context_data(self, *args, **kwargs):
        context = super(ModelWeightView, self).get_context_data(*args, **kwargs)
        context["trained_models"] = self.get_file_path("sample_models")
        return context

    def get_model_files(self):
        model_path = '{}/{}'.format(settings.MEDIA_ROOT, "sample_models")
        model_files = [f for f in listdir(model_path) if isfile(join(model_path, f))]
        return model_files

    def post(self, request, *args, **kwargs):
        context = self.get_context_data(*args, **kwargs)
        if request.method == 'POST':
            post_data = request.POST
            model_weight_file = request.FILES.get('model_weight_file')
            trainedModelWeight = post_data.get("trainedModelWeight")
            if model_weight_file or trainedModelWeight:
                if model_weight_file:
                    uploaded_model_weight_file_url = self.save_model(model_weight_file)
                else:
                    uploaded_model_weight_file_url = '{}/{}/{}'.format(settings.MEDIA_ROOT, "sample_models", eval(trainedModelWeight).get('file'))
                loaded_model = load_model(uploaded_model_weight_file_url)

                layer_outputs = [layer.output for layer in loaded_model.layers]
                activation_model = models.Model(inputs=loaded_model.input, outputs=layer_outputs)
                activations = activation_model.predict(img_tensor)

                model_weights= loaded_model.get_weights()
                model_neurons= []
                for i in range(len(loaded_model.layers)):
                    layer= loaded_model.layers[i]
                    layer_config= layer.get_config()
                    if "filters" in layer_config: #It help to extract only convolution layer
                        temp= {}
                        temp["filters"]= layer.filters
                        temp["name"]= layer.name
                        temp["layer_neurons"]= []
                        for val in range(layer.filters):
                            plt.figure()
                            plt.matshow(activations[i][0, :, :,val], cmap='viridis')
                            plt.axis('off')
                            fig= plt.gcf()
                            buf= io.BytesIO()
                            fig.savefig(buf, format="jpeg")
                            buf.seek(0)
                            string= base64.b64encode(buf.read())
                            uri= urllib.parse.quote(string)
                            temp["layer_neurons"].append(uri)
                        model_neurons.append(temp)
            if False:
                from keras.utils.vis_utils import plot_model
                plot_model(trained_model, to_file='media/models/model_plot.png', show_shapes=True, show_layer_names=True)
        return render(request, "neurons.html", context)

    def model_plot(self, trained_model, model_file_name):
        from keras.utils.vis_utils import plot_model
        plot_path= 'media/models/plots/{}_plot.png'.format(model_file_name)
        plot_model(trained_model, to_file=plot_path, show_shapes=True, show_layer_names=True)
        return plot_path

    def img_storage(self, input_img):
        uploaded_ip_img_file_url = self.save_img(input_img) # saving input image
        return uploaded_ip_img_file_url

    def file_storage(self, input_img, model_file_list):
        uploaded_model_file_url_list= []
        for model_file in model_file_list:
            uploaded_model_file_url = self.save_model(model_file)
            uploaded_model_file_url_list.append(uploaded_model_file_url)

        # input_img_path = os.path.join(base_path, "media/input/{}".format(input_img.name))
        # ip_img_filename = self.fs.save(input_img_path, input_img)
        uploaded_ip_img_file_url = self.save_img(input_img) # saving input image
        return uploaded_model_file_url_list, uploaded_ip_img_file_url

    def save_model(self, model_file):
        model_file_path = os.path.join(base_path, "media/models/{}".format(model_file.name))
        model_filename = self.fs.save(model_file_path, model_file) #saving model
        uploaded_model_file_url = model_filename
        return uploaded_model_file_url

    def save_img(self, input_img):
        input_img_path = os.path.join(base_path, "media/input/{}".format(input_img.name))
        ip_img_filename = self.fs.save(input_img_path, input_img)
        uploaded_ip_img_file_url = ip_img_filename # saving input image
        return uploaded_ip_img_file_url


class ArticleView(APIView):
    def get(self, request):
        return Response({"articles": "Hello"})
