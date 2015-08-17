from os.path import dirname, realpath, join

from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.checkbox import CheckBox
from kivy.properties import BooleanProperty
from kivy.app import App
from kivy.lang import Builder
from kivy.properties import NumericProperty, StringProperty, BooleanProperty,ListProperty
from kivy.uix.screenmanager import Screen
from kivy.uix.settings import SettingsWithNoMenu

import isaac as sc
import json

from isaac.autotuning.tune import tune

__version__ = '1.0'

class IsaacScreen(Screen):
    fullscreen = BooleanProperty(False)

    def add_widget(self, *args):
        if 'content' in self.ids:
            return self.ids.content.add_widget(*args)
        return super(IsaacScreen, self).add_widget(*args)

class IsaacHandler:
    
    def __init__(self):
        platforms = sc.driver.get_platforms()
        self.devices = [d for platform in platforms for d in platform.get_devices()]

     
class IsaacApp(App):
    
    screen_names = ListProperty([])


    def build(self):        
        self.tuner = 'ISAAC Kernels Tuner'

        #Settings
        self.settings_cls = SettingsWithNoMenu
        self.use_kivy_settings = False

        #Screen Manager
        self.screen_names = ['Tune']
        self.screens = {}
        current_directory = dirname(realpath(__file__))
        for name in self.screen_names:
            path = join(current_directory, 'screens',  '{}.kv'.format(name.lower()))
            self.screens[name] = Builder.load_file(path)
            
        #Default view
        self.show_tune()
    
    def start_tuning(self):
        #FIXME: will be buggy if two devices from two different platforms have the same name
        device = next(x for x in self.isaac_handler.devices if x.name==self.config.get('hardware', 'device'))
        operation = sc.templates.axpy
        json_path = ''
        #FIXME: Move profiling logics into tuning
        sc.driver.default.queue_properties = sc.driver.PROFILING_ENABLE
        tune(device, operation, json_path)
        
    def show_benchmark(self):
        pass
        
    def show_tune(self):
        if self.root.ids.sm.current != 'Tune':
            self.root.ids.sm.switch_to(self.screens['Tune'], direction='left')
        
    
    def display_settings(self, settings):
        if 'Settings' not in self.screens:
            self.screens['Settings'] = Screen(name='Settings')
            self.screens['Settings'].add_widget(settings)
        self.root.ids.sm.switch_to(self.screens['Settings'], direction='left')

    def build_config(self, config):
        self.isaac_handler = IsaacHandler()
        config.setdefaults('hardware', {'device':  self.isaac_handler.devices[0].name})
        config.setdefaults('autotuning', {'blas1': 'Intermediate', 'blas2': 'Intermediate', 'blas3': 'Intermediate'})
        
    def build_settings(self, settings):
        
        layout = [{'type': 'title',
                  'title': 'Hardware'},
                  {'type': 'options',
                  'title': 'Device',
                  'section': 'hardware',
                  'key': 'device',
                  'options': [device.name for device in self.isaac_handler.devices]},
                  {'type': 'title',
                  'title': 'Auto-tuning'}]
        
        for operation in ['BLAS1', 'BLAS2', 'BLAS3']:
            layout +=   [{'type': 'options',
                        'desc': 'Desired level of auto-tuning for ' + operation,
                        'title': operation,
                        'section': 'autotuning',
                        'key': operation.lower(),
                        'options': ['Simple', 'Intermediate', 'Full']}]
        
        settings.add_json_panel('Settings',
                                self.config,
                                data=json.dumps(layout))
                                
    def close_settings(self, *args):
        pass

        
            
if __name__ == '__main__':
    IsaacApp().run()
