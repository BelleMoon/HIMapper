from textwrap import dedent

from plyer import filechooser

from kivy.app import App
from kivy.lang import Builder
from kivy.properties import ListProperty
from kivy.properties import StringProperty
from kivy.uix.button import Button
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.textinput import TextInput
import copy
from globals import global_vars
from kivy.core.window import Window
from kivy.uix.popup import Popup
from kivy.uix.label import Label
import threading




import os


seg_dir = ''
global_vars.K_VALUE = 3

class Segmentation():
    def load_image(self, image_path):
        print(image_path)
        # Carrega a imagem e converte de BGR para RGB
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image_rgb

    def resize_image(self, image, width=600):
        # Redimensiona a imagem mantendo a proporção
        height = int(image.shape[0] * (width / image.shape[1]))
        return cv2.resize(image, (width, height))
    
    def apply_color_mask(self, image, lower_bound, upper_bound):
        # Converte a imagem de RGB para HSV
        hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        # Aplica a máscara de cor
        mask = cv2.inRange(hsv_image, lower_bound, upper_bound)
        return mask

    def color_pixels(self, image):
        # Definindo os intervalos de cor em HSV
        # Cinza (áreas construídas) -> vermelho na saída
        lower_gray = np.array([0, 0, 50])
        upper_gray = np.array([180, 50, 200])
        # Azul, Verde, Laranja (áreas com menor tendência a ilhas de calor) -> azul na saída
        lower_blue = np.array([90, 50, 50])
        upper_blue = np.array([140, 255, 255])
        lower_green = np.array([35, 50, 50])
        upper_green = np.array([85, 255, 255])
        lower_orange = np.array([10, 50, 50])
        upper_orange = np.array([25, 255, 255])

        # Aplicando as máscaras de cor
        gray_mask = self.apply_color_mask(image, lower_gray, upper_gray)
        blue_mask = self.apply_color_mask(image, lower_blue, upper_blue)
        green_mask = self.apply_color_mask(image, lower_green, upper_green)
        orange_mask = self.apply_color_mask(image, lower_orange, upper_orange)

        # Unindo as máscaras de cor para áreas com menor tendência
        non_heat_island_mask = cv2.bitwise_or(blue_mask, green_mask)
        non_heat_island_mask = cv2.bitwise_or(non_heat_island_mask, orange_mask)

        # Inicializando a imagem de saída com uma cor de fundo padrão (amarelo claro)
        output_image = np.full_like(image, [255, 255, 0])

        # Classificando pixels
        output_image[gray_mask > 0] = [255, 0, 0]     # Vermelho para áreas construídas
        output_image[non_heat_island_mask > 0] = [0, 0, 255]  # Azul para áreas com menor tendência

        return output_image

    def classify_pixels(self, image, k):
        # Redimensiona a imagem para um array 2D de pixels e 3D de features (Valores de cores, colors)
        reshaped_image = image.reshape((-1, 3))

        # Converte os dados para np.float32
        reshaped_image = np.float32(reshaped_image)

        # Define os critérios de parada (precisão e número máximo de iterações)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 80, 0.2)

        # Aplica o algoritmo k-means
        _, labels, centers = cv2.kmeans(reshaped_image, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

        # Converte os centros dos clusters para uint8
        centers = np.uint8(centers)

        # Reatribui os pixels na imagem original com base nos rótulos obtidos do k-means
        segmented_image = centers[labels.flatten()]

        # Redimensiona a imagem de volta para o formato original
        segmented_image = segmented_image.reshape(image.shape)

        return segmented_image
    

class MyBoxLayout(TextInput):
    def __init__(self, **kwargs):
        super(MyBoxLayout, self).__init__(**kwargs)

    def on_enter(self, instance):
        global_vars.K_VALUE = int(instance.text)
        
        instance.text = ''  # Limpa o texto após pressionar Enter
        instance.hint_text = f'K atual mudado para {global_vars.K_VALUE}'  

class FileChoose(Button):
    '''
    Button that triggers 'filechooser.open_file()' and processes
    the data response from filechooser Activity.
    '''

    selection = ListProperty([])

    def choose(self):
        '''
        Call plyer filechooser API to run a filechooser Activity.
        '''
        filechooser.open_file(on_selection=self.handle_selection)

    def handle_selection(self, selection):
        '''
        Callback function for handling the selection response from Activity.
        '''
        self.selection = selection

    def on_selection(self, *a, **k):
        '''
        Update TextInput.text after FileChoose.selection is changed
        via FileChoose.handle_selection.
        '''
        global seg_dir
        print(self.selection)
        App.get_running_app().root.ids.imageView.source = str(self.selection[0])
        seg_dir = str(self.selection[0])

class MyCustomButton(Button):
    def do_transformation(self):
        global seg_dir

        y = copy.deepcopy(global_vars.K_VALUE)
        print(y)

        segmentation = Segmentation()

        image = segmentation.load_image(seg_dir)
        classified_image = segmentation.classify_pixels(image, int(y))
        color_image = segmentation.color_pixels(classified_image)

        output_path = f'output.jpg'

        # Assuming cv2 is imported properly and used correctly
        cv2.imwrite(output_path, cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR))
        App.get_running_app().root.ids.imageView.source = str(output_path)

        # Criando o conteúdo do Popup
        content = Label(text='A imagem foi segmentada e salva como output.jpg!')

        # Criando o Popup
        popup = Popup(title='Sucesso', content=content,
                      size_hint=(None, None), size=(400, 200))

        # Exibindo o Popup
        popup.open()

class ChooserApp(App):
    def build(self):
        self.title = 'Heat Island Mapper'
        self.icon = 'icon.png'
        Window.size = (1920, 1080)

        return Builder.load_string(dedent('''
            <FileChoose>:
            BoxLayout:
                orientation: 'vertical'

                Image:
                    id: imageView
                    source: 'null3.png'
                    allow_stretch: True

                MyBoxLayout:
                    id: text_input
                    multiline: False
                    size_hint_y: None
                    height: '48dp'
                    font_size: '24sp'
                    hint_text: 'Digite um valor para K e pressione Enter'
                    on_text_validate: self.on_enter(self)

                BoxLayout:
                    orientation: 'horizontal'
                    size_hint_y: 0.1

                    FileChoose:
                        on_release: self.choose()
                        text: 'Selecionar Arquivo'

                    MyCustomButton:
                        text: 'Fazer Transformação'
                        on_release: self.do_transformation()
        '''))

if __name__ == '__main__':
    ChooserApp().run()
