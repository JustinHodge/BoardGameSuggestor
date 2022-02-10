import tkinter as tk
from tkinter.ttk import *
class Dashboard():
    def __init__(self, controller):
        self.controller = controller
        self.window = tk.Tk()
        self.window.title('Board Game Suggestor')
        self.game_options = []
        self.list_of_choices = []
        # write your code here

        self.result_window = Frame(self.window, border=1)
        self.result_window.grid(row=4, column= 1)

        self.get_results_button = Button(self.window, text='Get My Results', command=self.getResults)
        self.get_results_button.grid(row=5, column=1, pady=15, padx=15)

        self.exit_button = Button(self.window, text='EXIT', command=self.exit)
        self.exit_button.grid(row=5, column=2, pady=15, padx=15)

    def populate_dropdowns(self, name_index_dict):
        values = []
        for k in name_index_dict.keys():
            values += k
        return values

    def getResults(self):
        self.list_of_choices = [self.game_choice1.get(), self.game_choice2.get(), self.game_choice3.get()]
        self.controller.getPrediction(self.list_of_choices)

    def exit(self):
        self.window.destroy()
    
    def set_result(self, title):
        for widget in self.result_window.winfo_children():
            widget.destroy()
        result_label = Label(self.result_window, text='You should play ' + title)
        result_label.pack()

    def create_dropdowns(self, name_index_dict):
        values = list(name_index_dict.keys())
        self.game_options = values

        self.choice_label1 = Label(self.window, text='Game #1')
        self.choice_label1.grid(row=1,column=1)
        self.game_choice1 = Combobox(self.window, values=values)
        self.game_choice1.grid(row=1,column=2, pady=15, padx=10)

        self.choice_label2 = Label(self.window, text='Game #1')
        self.choice_label2.grid(row=2,column=1)
        self.game_choice2 = Combobox(self.window, values=values)
        self.game_choice2.grid(row=2,column=2, pady=15, padx=10)

        self.choice_label3 = Label(self.window, text='Game #1')
        self.choice_label3.grid(row=3,column=1)
        self.game_choice3 = Combobox(self.window, values=values)
        self.game_choice3.grid(row=3,column=2, pady=15, padx=10)
    
    def check_input(self, event):
        # NO WORKY
        pass
        # value = event.widget.get()
        # widget = event.widget

        # if value == '':
        #     widget['values'] = self.game_options
        # else:
        #     data = []
        #     for item in self.game_options:
        #         if value.lower() in item.lower():
        #             data.append(item)

        #     widget['values'] = data
        # self.window.update()
