#System
from os.path import dirname, realpath, join
import isaac as sc
import json
import thread
from time import sleep

#Tuner
from tune.tune import Tuner
from tune.tools import metric_name_of

#Kivy
from kivy.logger import Logger
from kivy.uix.scrollview import ScrollView
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.checkbox import CheckBox
from kivy.properties import BooleanProperty
from kivy.app import App
from kivy.lang import Builder
from kivy.properties import NumericProperty, StringProperty, BooleanProperty,ListProperty
from kivy.uix.screenmanager import Screen
from kivy.uix.settings import SettingsWithNoMenu


#Check if on Android
import imp
try:
    imp.find_module('android')
    on_android = True
except ImportError:
    on_android = False


if on_android:
    from android.runnable import run_on_ui_thread


__version__ = '1.0'

class ScrollableLabel(ScrollView):
    text = StringProperty('')
    font_name = StringProperty('')

class LabelLogger:
    def __init__(self, label):
        self.label = label;
        
    def info(self, msg):
        self.label.text += msg + '\n'
        
class LabelProgressBar:
    
    def __init__(self, length, label, metric_name):
        self.prefix = ''
        self.text = ''
        self.label = label
        self.metric_name = metric_name
        self.length = length
        self.old_percent = 0
    
    def set_prefix(self, prefix):
        self.prefix = prefix
        self.text_start = self.label.text
        self.label.text = self.text_start + "{0}: [{1}] {2: >3}%".format(prefix.ljust(17), ' '*self.length, 0)
    
    def set_finished(self):
        self.old_percent = 0
        self.label.text += '\n'
        
    def update(self, i, total, x, y, complete=False):
        percent = float(i) / total
        hashes = '#' * int(round(percent * self.length))
        spaces = ' ' * (self.length - len(hashes))
        #Format of structures to print
        xformat = ','.join(map(str,map(int, x)))
        yformat = int(y)
        percent = int(round(percent * 100))
        msg = ("{0}: [{1}] {2: >3}% [{3} {4}]").format(self.prefix.ljust(17), hashes + spaces, percent, yformat, self.metric_name)
        if percent > self.old_percent:
            sleep(.01)
            self.label.text = self.text_start + msg
            self.old_percent = percent
        
        
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
        
        #Logger
        self.logger = LabelLogger(self.screens['Tune'].ids.out)
    
    if on_android:
        @run_on_ui_thread
        def lock_screen(self):
            from jnius import autoclass
            PythonActivity = autoclass('org.renpy.android.PythonActivity')
            Params = autoclass('android.view.WindowManager$LayoutParams')
            PythonActivity.mActivity.getWindow().addFlags(Params.FLAG_KEEP_SCREEN_ON)
    
        @run_on_ui_thread
        def unlock_screen(self):
            PythonActivity = autoclass('org.renpy.android.PythonActivity')
            Params = autoclass('android.view.WindowManager$LayoutParams')
            PythonActivity.mActivity.getWindow().clearFlags(Params.FLAG_KEEP_SCREEN_ON)
        
    def start_tuning(self):
        button = self.screens['Tune'].ids.action_button
        if button.text == 'Run':
            #FIXME: will be buggy if two devices from two different platforms have the same name
            device = next(x for x in self.isaac_handler.devices if x.name==self.config.get('hardware', 'device')) 
            #FIXME: Move profiling logics into tuning
            sc.driver.default.queue_properties = sc.driver.PROFILING_ENABLE 
            self.logger.info('Using ' + device.name)
            self.logger.info('')
            
            def run():
                if on_android:
                    self.lock_screen()
                operations = [('blas1', (sc.templates.axpy,)),
                              ('blas2', (sc.templates.gemv_n, sc.templates.gemv_t)),
                              ('blas3', (sc.templates.gemm_nn, sc.templates.gemm_tn, sc.templates.gemm_nt, sc.templates.gemm_tt))]
                for opclass, optype in operations:
                    for op in optype:
                        progress_bar = LabelProgressBar(10, self.logger.label, metric_name_of(op))
                        tuner = Tuner(self.logger, device, op, json_path='', progress_bar=progress_bar)
                        tuner.run(self.config.get('autotuning', opclass).lower())
            
            tid = thread.start_new_thread(run, ())
        else:
            pass
        button.text = 'Running...' if button.text == 'Run' else button.text
        
    def show_benchmark(self):
        pass
    
    def on_pause(self):
        print 'pause'
        return True
    
    def on_resume(self):
        print 'resume'
        return True
        
    def show_tune(self):
        if self.root.ids.sm.current != 'Tune':
            self.root.ids.sm.switch_to(self.screens['Tune'], direction='left')
        
    
    def display_settings(self, settings):
        if 'Settings' not in self.screens:
            self.screens['Settings'] = Screen(name='Settings')
            self.screens['Settings'].add_widget(settings)
        if self.root.ids.sm.current != 'Settings':
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
        
        settings.add_json_panel('ISAAC',
                                self.config,
                                data=json.dumps(layout))
                                                        
    def close_settings(self, *args):
        pass

        
            
if __name__ == '__main__':
    IsaacApp().run()
