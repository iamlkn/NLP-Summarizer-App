import kivy
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.uix.button import Button
from kivy.uix.slider import Slider
from kivy.uix.dropdown import DropDown
from kivy.uix.spinner import Spinner
from kivy.core.window import Window
from kivy.clock import Clock
from kivy.core.text import LabelBase
import predict

kivy.require('2.0.0')

class SummarizerLayout(BoxLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Placeholder for the summarization function
        self.summarize_text = lambda text, length, method: "This is a placeholder summary."

    def on_summarize(self):
        input_text = self.ids.input_text.text
        summary_length = int(self.ids.length_slider.value)
        extraction_method = self.ids.model_spinner.text

        if input_text:
            # Simulate the "Summarizing..." spinner effect
            self.ids.output_label.text = "Summarizing..."
            Clock.schedule_once(lambda dt: self.display_summary(input_text, summary_length, extraction_method), 2)  # Simulate 2 seconds of processing
        else:
            self.ids.output_label.text = "Please enter some text to summarize."

    def display_summary(self, input_text, summary_length, extraction_method):
        try:
            pred = predict.Predict()
            summary = pred.predict(input_text.split(), beam_search=True)
            self.ids.output_label.text = summary
        except Exception as e:
            self.ids.output_label.text = f"Error: {e}"


# Register the fonts
LabelBase.register(name='Roboto',
                   fn_regular='fonts/Roboto-Regular.ttf',
                   fn_bold='fonts/Roboto-Bold.ttf')
LabelBase.register(name='Montserrat',
                   fn_regular='fonts/Montserrat-Regular.ttf',
                   fn_bold='fonts/Montserrat-Bold.ttf')

class SummarizerApp(App):
    def build(self):
        self.load_kv('style.kv')
        Window.clearcolor = (0.95, 0.95, 0.95, 1)
        return SummarizerLayout()

if __name__ == '__main__':
    SummarizerApp().run()