from django.shortcuts import render
from django.conf import settings
import logging
logger = logging.getLogger(__package__)

from django.core.files.storage import FileSystemStorage
import os
from os import listdir
from os.path import isfile, join
import plotly.express as px
import plotly.offline as opy
import numpy as np
import pandas as pd
from aivalidation.views import BaseView
import sys
np.set_printoptions(threshold=sys.maxsize)

base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Create your views here.
class DataVisualizerView(BaseView):
    template_name = "visualizer.html"
    fs = FileSystemStorage(location=settings.MEDIA_ROOT)
    chart_config = {"pie": {"disp_cnt": 10}, "bar": {"disp_cnt": 10}}
    def get_context_data(self, *args, **kwargs):
        context = super(DataVisualizerView, self).get_context_data(*args, **kwargs)
        context['plot_types'] = [{"value": "piechart", "label": "Pie Chart"}, {"value": "barchart", "label": "Bar Chart"}]
        context['is_uploaded'] = False
        context['ml_scatter_plot'] = self.ml_scatter_plot()
        return context

    def get_specific_file(self):
        path = '{}/{}'.format(settings.MEDIA_ROOT, "sample_models")
        files = [f for f in listdir(path) if isfile(join(path, f))]
        return files

    def post(self, request, *args, **kwargs):
        context = self.get_context_data(*args, **kwargs)
        if request.method == 'POST':
            context["status"] = "success"
            post_data = request.POST
            data_file = request.FILES.get('data_file')
            column_name = post_data.getlist('columns')
            plot_types = post_data.getlist('plot_types')
            hidden_file_path = post_data.get('hidden_file_path')
            if data_file or hidden_file_path:
                context['is_uploaded'] = True
                if data_file:
                    context["hidden_file_path"] = self.file_storage(data_file)
                elif hidden_file_path:
                    data_file = hidden_file_path
                    context["hidden_file_path"] = hidden_file_path
                df = pd.read_json(data_file)
                context["columns"] = list(df.columns)
                if column_name:
                    context['all_plots'] = []
                    for col in column_name:
                        context['all_plots'].append({"col_name": col, "plots": self.get_plot(df, col, plot_types)})
            else:
                context["msg"] = "Please upload image and model file or choose trained model and sample image"
                context["status"] = "error"

            context["data_file"] = data_file
        return render(request, "visualizer.html", context)

    def get_plot(self, src_df, col, plot_types):
        # df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/finance-charts-apple.csv')
        # date= self.get_column_data(dataset, 'date')
        # open = self.get_column_data(dataset, 'open')
        # high = self.get_column_data(dataset, 'high')
        # low = self.get_column_data(dataset, 'low')
        # close = self.get_column_data(dataset, 'close')
        # # fig = go.Figure(data=[go.Candlestick(x=df['Date'],
        # #                                      open=df['AAPL.Open'],
        # #                                      high=df['AAPL.High'],
        # #                                      low=df['AAPL.Low'],
        # #                                      close=df['AAPL.Close'])])
        # fig = go.Figure(data=[go.Candlestick(x=date,
        #                                      open=open,
        #                                      high=high,
        #                                      low=low,
        #                                      close=close)])
        # div = opy.plot(fig, auto_open=False, output_type='div')

        plot_list = []
        for type in plot_types:
            if type == 'piechart':
                df = src_df.copy()
                df = df[col].value_counts().to_frame().reset_index()
                df.columns = [col, 'values']
                df.replace('', np.nan, inplace=True)
                df.dropna(inplace=True)
                df = df[['name','values']][df['values'] >= 3]
                total_cnt = len(df)
                # df = df.loc[df['values'] < 3, col]
                # df.loc[df['values'] < 5, col] = 'Others'
                df=df.head(self.chart_config['pie']['disp_cnt']) #It will display top 5 data
                # df = px.data.gapminder().query("year == 2007").query("continent == 'Europe'")
                # df.loc[df['pop'] < 2.e6, 'country'] = 'Other countries'  # Represent only large countries
                title = 'Pie Chart (Displaying Only Top {} / {} Data)'.format(self.chart_config['pie']['disp_cnt'], str(total_cnt))
                msg = "Displaying Top {} / {}".format(self.chart_config['pie']['disp_cnt'], str(total_cnt))
                fig = px.pie(df, values='values', names=col, title=title, width=800, height=400)
                div = opy.plot(fig, auto_open=False, output_type='div')
                plot_list.append({"msg": msg, "div": div, "total_cnt": str(total_cnt)})
            elif type == 'barchart':
                df = src_df.copy()
                df = df[col].value_counts().to_frame().reset_index()
                df.columns = [col, 'values']
                df.replace('', np.nan, inplace=True)
                df.dropna(inplace=True)
                total_cnt = len(df)
                df = df.head(self.chart_config['bar']['disp_cnt'])  # It will display top 5 data
                # df.loc[df['values'] < 5, col] = 'Others'
                # df = px.data.gapminder().query("year == 2007").query("continent == 'Europe'")
                # df.loc[df['pop'] < 2.e6, 'country'] = 'Other countries'  # Represent only large countries
                title = 'Bar Chart (Displaying Only Top {} / {} Data)'.format(self.chart_config['pie']['disp_cnt'],
                                                                              str(total_cnt))
                msg = "Displaying Top {} / {}".format(self.chart_config['pie']['disp_cnt'], str(total_cnt))
                fig = px.bar(df, x=col, y='values' ,title=title, width=800, height=400)
                div = opy.plot(fig, auto_open=False, output_type='div')
                plot_list.append({"msg": msg, "div": div, "total_cnt": str(total_cnt)})
        # fig.show()
        return plot_list

    def model_plot(self, trained_model, model_file_name):
        from keras.utils.vis_utils import plot_model
        plot_path= 'media/models/plots/{}_plot.png'.format(model_file_name)
        plot_model(trained_model, to_file=plot_path, show_shapes=True, show_layer_names=True)
        return plot_path

    def file_storage(self, file):
        uploaded_file_url = self.save_file(file) # saving input image
        return uploaded_file_url

    # def file_storage(self, input_img, model_file_list):
    #     uploaded_model_file_url_list= []
    #     for model_file in model_file_list:
    #         uploaded_model_file_url = self.save_model(model_file)
    #         uploaded_model_file_url_list.append(uploaded_model_file_url)
    #
    #     # input_img_path = os.path.join(base_path, "media/input/{}".format(input_img.name))
    #     # ip_img_filename = self.fs.save(input_img_path, input_img)
    #     uploaded_ip_img_file_url = self.save_img(input_img) # saving input image
    #     return uploaded_model_file_url_list, uploaded_ip_img_file_url
    #
    # def save_model(self, model_file):
    #     model_file_path = os.path.join(base_path, "media/models/{}".format(model_file.name))
    #     model_filename = self.fs.save(model_file_path, model_file) #saving model
    #     uploaded_model_file_url = model_filename
    #     return uploaded_model_file_url

    def save_file(self, file):
        file_path = os.path.join(base_path, "media/visualizer/{}".format(file.name))
        uploaded_file_url = self.fs.save(file_path, file) # saving input image
        return uploaded_file_url

    def ml_scatter_plot(self):
        import dash
        import dash_core_components as dcc
        import dash_html_components as html
        from dash.dependencies import Input, Output
        import numpy as np
        import plotly.graph_objects as go
        import plotly.express as px
        from sklearn.model_selection import train_test_split
        from sklearn import linear_model, tree, neighbors

        file_path = os.path.join(base_path, "media/visualizer/npz/{}".format("aisdk_20181101_first35kLines_train_test.npz"))
        data = np.load(file_path)

        X_train = data['x_train']
        y_train = data['y_train']

        X_test = data['x_test']
        y_test = data['y_test']

        # df = px.data.tips()
        # X = df.total_bill.values[:, None]
        # X_train, X_test, y_train, y_test = train_test_split(
        #     X, df.tip, random_state=42)
        #
        # models = {'Regression': linear_model.LinearRegression,
        #           'Decision Tree': tree.DecisionTreeRegressor,
        #           'k-NN': neighbors.KNeighborsRegressor}
        #
        # model = models['Regression']()
        # model.fit(X_train, y_train)
        #
        # x_range = np.linspace(X.min(), X.max(), 100)
        # y_range = model.predict(x_range.reshape(-1, 1))
        # lat_x_train_shape = X_train[:,:,[1]].shape
        # lat_y_train_shape = y_train[:,[0]].shape
        #
        # lat_x_test_shape = X_test[:, :, [1]].shape
        # lat_y_test_shape = y_test[:, [0]].shape
        #
        # lat_X_train = X_train[:,:,[1]].reshape(lat_x_train_shape[0]*lat_x_train_shape[1],1)
        # lat_y_train = y_train[:,[0]].reshape(lat_y_train_shape[0]*lat_y_train_shape[1])
        #
        # lat_X_test = X_test[:, :, [1]].reshape(lat_x_test_shape[0]*lat_x_test_shape[1],1)
        # lat_y_test = y_test[:, [0]].reshape(lat_y_test_shape[0]*lat_y_test_shape[1])
        scatter_plot_arr = []
        lat_X_train, lat_y_train, lat_X_test, lat_y_test = self.get_latitude(X_train, y_train, X_test, y_test)

        fig = go.Figure([
            go.Scatter(x=lat_X_train.squeeze(), y=lat_y_train,
                       name='train', mode='markers'),
            go.Scatter(x=lat_X_test.squeeze(), y=lat_y_test,
                       name='test', mode='markers'),
            # go.Scatter(x=x_range, y=y_range,
            #            name='prediction')
        ])
        lat_div = opy.plot(fig, auto_open=False, output_type='div')
        scatter_plot_arr.append({"col_name": "latitude", "plot": lat_div})

        log_X_train, log_y_train, log_X_test, log_y_test = self.get_logitude(X_train, y_train, X_test, y_test)

        fig = go.Figure([
            go.Scatter(x=log_X_train.squeeze(), y=log_y_train,
                       name='train', mode='markers'),
            go.Scatter(x=log_X_test.squeeze(), y=log_y_test,
                       name='test', mode='markers'),
            # go.Scatter(x=x_range, y=y_range,
            #            name='prediction')
        ])
        log_div = opy.plot(fig, auto_open=False, output_type='div')
        scatter_plot_arr.append({"col_name": "longitude", "plot": log_div})

        return scatter_plot_arr

    def get_latitude(self, X_train, y_train, X_test, y_test):
        lat_x_train_shape = X_train[:, :, [1]].shape
        lat_y_train_shape = y_train[:, [0]].shape

        lat_x_test_shape = X_test[:, :, [1]].shape
        lat_y_test_shape = y_test[:, [0]].shape

        lat_X_train = X_train[:, :, [1]].reshape(lat_x_train_shape[0] * lat_x_train_shape[1], 1)
        lat_y_train = y_train[:, [0]].reshape(lat_y_train_shape[0] * lat_y_train_shape[1])

        lat_X_test = X_test[:, :, [1]].reshape(lat_x_test_shape[0] * lat_x_test_shape[1], 1)
        lat_y_test = y_test[:, [0]].reshape(lat_y_test_shape[0] * lat_y_test_shape[1])

        return lat_X_train, lat_y_train, lat_X_test, lat_y_test

    def get_logitude(self, X_train, y_train, X_test, y_test):
        log_x_train_shape = X_train[:, :, [1]].shape
        log_y_train_shape = y_train[:, [0]].shape

        log_x_test_shape = X_test[:, :, [1]].shape
        log_y_test_shape = y_test[:, [0]].shape

        log_X_train = X_train[:, :, [2]].reshape(log_x_train_shape[0] * log_x_train_shape[1], 1)
        log_y_train = y_train[:, [1]].reshape(log_y_train_shape[0] * log_y_train_shape[1])

        log_X_test = X_test[:, :, [2]].reshape(log_x_test_shape[0] * log_x_test_shape[1], 1)
        log_y_test = y_test[:, [1]].reshape(log_y_test_shape[0] * log_y_test_shape[1])

        return log_X_train, log_y_train, log_X_test, log_y_test



