import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score, ShuffleSplit, LeaveOneOut, TimeSeriesSplit
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.utils import resample
from sklearn.metrics import accuracy_score
from sklearn.exceptions import ConvergenceWarning
import warnings
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')  # Use the TkAgg backend
from PIL import Image, ImageTk
from itertools import cycle, count
import fitz  # PyMuPDF
import os
class ScrollingText(tk.Frame):
    def __init__(self, master=None, **kwargs):
        super().__init__(master, **kwargs)
        self.scroll_text = "ENHANCING DATA ANALYSIS THROUGH [CROSS-VALIDATION_&_BOOTSTRAP_EVALUATION_OF_ML_ALGORITHIMS]\t\t\t\t\t"
        self.scroll_label = tk.Label(self, text=self.scroll_text,anchor='w', font=("times new roman ", 13, "bold"), bg="#191970", fg="#7FFF00")
        self.scroll_label.pack(pady=(0,0))
        self.scroll()

    def scroll(self):
        self.scroll_text = self.scroll_text[-1] + self.scroll_text[:-1]
        self.scroll_label.config(text=self.scroll_text,width=150)
        self.after(500, self.scroll)  # Adjust the delay (in ms) to control the scrolling speed
        
class ImageLabel(tk.Label):
    def load(self, im):
        if isinstance(im, str):
            im = Image.open(im)
        frames = []

        try:
            for i in count(5):
                frames.append(ImageTk.PhotoImage(im.copy()))
                im.seek(i)
        except EOFError:
            pass
        self.frames = cycle(frames)

        try:
            self.delay = im.info['duration']
        except:
            self.delay = 100

        if len(frames) == 1:
            self.config(image=next(self.frames))
        else:
            self.next_frame()

    def unload(self):
        self.config(image=None)
        self.frames = None

    def next_frame(self):
        if self.frames:
            self.config(image=next(self.frames))
            self.after(self.delay, self.next_frame)

class DatasetAnalyzerApp:
    def __init__(self, root):
        self.gif_label = None  # Initialize the GIF label
        self.root = root
        self.root.title("Dataset Analyzer")
        self.datasets = {
            'Lung Cancer': 'datasets\\survey_lung_cancer.csv',
            'Brain Stroke': 'datasets\\brain_stroke.csv',
            'Diabetes': 'datasets\\diabetes.csv',
            'Heart Disease': 'datasets\\heart.csv',
            'Hypothyroid': 'datasets\\hypothyroid.csv'
        }
        self.algorithms = {
            'Logistic Regression': LogisticRegression(max_iter=2000),
            'Random Forest': RandomForestClassifier(),
            'Naive Bayes': GaussianNB(),
            'KNN': KNeighborsClassifier(),
            'Decision Tree': DecisionTreeClassifier(),
            'SVM': SVC(),
            'ANN': MLPClassifier(max_iter=2000)
        }
        self.cv_techniques = {
            'KFold': KFold(n_splits=10, shuffle=True, random_state=42),
            'StratifiedKFold': StratifiedKFold(n_splits=10, shuffle=True, random_state=42),
            'Holdout': ShuffleSplit(n_splits=10, test_size=0.2, random_state=42),
            'LeaveOneOut': LeaveOneOut(),
            'MonteCarlo': ShuffleSplit(n_splits=100, test_size=0.2, random_state=42),
            'TimeSeries': TimeSeriesSplit(n_splits=10)
        }
        self.dataset_var = tk.StringVar()
        self.dataset_var.set(list(self.datasets.keys())[0])
        self.algorithm_vars = [tk.IntVar() for _ in range(len(self.algorithms))]
        self.cv_technique_vars = [tk.IntVar() for _ in range(len(self.cv_techniques))]
        self.results_text = tk.Text(root, wrap='word', width=500, height=20, state='disabled')  # Set the initial state to 'disabled'
        self.create_widgets()
   
    def get_target_column(self, dataset_name):
        if dataset_name == 'Lung Cancer':
            return 'LUNG_CANCER'
        elif dataset_name == 'Brain Stroke':
            return 'stroke'
        elif dataset_name == 'Diabetes':
            return 'Outcome'
        elif dataset_name == 'Heart Disease':
            return 'HeartDisease'
        elif dataset_name == 'Hypothyroid':
            return 'binaryClass'
        else:
            return None  # Handle the case for unknown dataset names

    def prepare_data_for_analysis(self, selected_dataset, algorithms, cv_techniques):
        dataset_path = self.datasets[selected_dataset]
        df = pd.read_csv(dataset_path)

        target_column = self.get_target_column(selected_dataset)
        X = df.drop(target_column, axis=1)
        y = df[target_column]
        X = pd.get_dummies(X)

        results = self.perform_analysis(X, y, algorithms, cv_techniques, selected_dataset)

        # Create a DataFrame for results
        df_results = pd.DataFrame(results, columns=['Dataset', 'Algorithm', 'CV_Technique', 'CV_Accuracy (%)', 'NO_OF_Iterations', 'BSTP_Accuracy (%)'])
        
        return X, y, df_results
    
    def create_widgets(self):
        canvas = tk.Canvas(self.root, bg="gray")
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        scrollbar = tk.Scrollbar(self.root, command=canvas.yview)
        scrollbar.pack(side=tk.LEFT, fill='y')

        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.bind('<Configure>', lambda e: canvas.configure(scrollregion=canvas.bbox('all')))

        frame = tk.Frame(canvas, bg="Teal")
        canvas.create_window((0, 0), window=frame, anchor='nw')

        # Create the results output box with black background and white text color
        self.results_text = tk.Text(root, wrap='word',width=300, height=20 ,state='disabled', bg="dark Gray", fg='black', bd=2, highlightthickness=2)
        self.results_text.pack(expand=True, fill='both')
        
        # Load the GIF
        self.gif_label = ImageLabel(self.results_text)
        self.gif_label.pack()
        self.gif_label.load('gif2.gif')

        # File Load Button
        load_file_button = tk.Button(frame, text="DataSet Infromation Manual", font=("modern no. 20", 11, "bold"), command=self.load_and_display_file, bg="pink", fg="black", borderwidth=2)
        load_file_button.pack(anchor='w',pady=(0,5))
        
        # Dataset Selection
        dataset_label = tk.Label(frame, text="Select Dataset:", font=("Times New Roman", 12, "bold"))
        dataset_label.pack(anchor='center')
        for dataset_name in self.datasets.keys():
            dataset_radio = tk.Radiobutton(frame, text=dataset_name, variable=self.dataset_var, value=dataset_name, bg="teal")
            dataset_radio.pack(anchor=tk.W)

        # Algorithm Instructions Button
        algo_instructions_button = tk.Button(frame, text="Algorithms Infromation Manual", font=("modern no. 20", 11, "bold"),command=self.show_algorithm_instructions, bg="pink", fg="black",  borderwidth=2)
        algo_instructions_button.pack(anchor='w',pady=(0,5))

        # Algorithm Instructions Button
        algorithm_label = tk.Label(frame, text="Select Algorithms:",  font=("Times New Roman", 12, "bold"))
        algorithm_label.pack(anchor='center')
        
    # Algorithm Toggle All Button
        algorithm_toggle_all_button = tk.Button(frame, text="Select All",font=("modern no. 20", 11), command=lambda: self.toggle_all(self.algorithm_vars), bg="pink", fg="black")
        algorithm_toggle_all_button.pack(anchor='w')

        for i, algo_name in enumerate(self.algorithms.keys()):
            algo_checkbox = tk.Checkbutton(frame, text=algo_name, variable=self.algorithm_vars[i],bg="teal",)
            algo_checkbox.pack(anchor=tk.W)

        # Cross-Validation Technique Selection
        cv_instructions_button = tk.Button(frame, text="CrossValidation Infromation Manual",font=("modern no. 20", 11, "bold"),command=self.show_cv_instructions, bg="pink", fg="black", borderwidth=2)
        cv_instructions_button.pack(anchor='w',pady=(0,5))

        cv_label = tk.Label(frame, text="Select Crossvalidation techniques:", font=("Times New Roman", 12, "bold"))
        cv_label.pack(anchor="center",padx=(71,71),pady=(0,5))

    # Cross-Validation Technique Toggle All Button
        cv_toggle_all_button = tk.Button(frame, text="Select All",font=("modern no. 20", 11),command=lambda: self.toggle_all(self.cv_technique_vars), bg="pink", fg="black")
        cv_toggle_all_button.pack(anchor='w')

        for i, cv_name in enumerate(self.cv_techniques.keys()):
            cv_checkbox = tk.Checkbutton(frame, text=cv_name, variable=self.cv_technique_vars[i],bg="teal",)
            cv_checkbox.pack(anchor=tk.W)

        # Calculate Button
        calculate_button = tk.Button(frame, text="Calculate Results", command=self.calculate_results, bg="pink",font=("Times New Roman", 12, "bold"),fg="black",borderwidth=2)
        calculate_button.pack(pady=10)
         
        # Results Table
        self.results_text.pack(expand=True, fill='both')

    def load_and_display_file(self):
        pdf_file_path = 'datasets\\datasetsinfo.pdf'  # Change the file path here
        try:
            if not os.path.isfile(pdf_file_path):
                raise FileNotFoundError(f"The file '{pdf_file_path}' was not found.")

            pdf_document = fitz.open(pdf_file_path)
            pdf_text = ""
            for page_number in range(pdf_document.page_count):
                page = pdf_document.load_page(page_number)
                pdf_text += page.get_text()

            self.show_info_popup("PDF Instructions", pdf_text)
        except FileNotFoundError as e:
            self.show_info_popup("File Error", str(e))
        except Exception as e:
            self.show_info_popup("Error", str(e))
        finally:
            pdf_document.close()  # Close the PDF document when done

    def show_algorithm_instructions(self):
        pdf_file_path = 'datasets\\algorithms info.pdf'  # Change the file path here
        try:
            if not os.path.isfile(pdf_file_path):
                raise FileNotFoundError(f"The file '{pdf_file_path}' was not found.")

            pdf_document = fitz.open(pdf_file_path)
            pdf_text = ""
            for page_number in range(pdf_document.page_count):
                page = pdf_document.load_page(page_number)
                pdf_text += page.get_text()

            self.show_info_popup("PDF Instructions", pdf_text)
        except FileNotFoundError as e:
            self.show_info_popup("File Error", str(e))
        except Exception as e:
            self.show_info_popup("Error", str(e))
        finally:
            pdf_document.close()  # Close the PDF document when done

    

    def show_cv_instructions(self):
        pdf_file_path = 'datasets\\cvinfo.pdf'  # Change the PDF file path here

        try:
            if not os.path.isfile(pdf_file_path):
                raise FileNotFoundError(f"The file '{pdf_file_path}' was not found.")

            pdf_document = fitz.open(pdf_file_path)
            pdf_text = ""
            for page_number in range(pdf_document.page_count):
                page = pdf_document.load_page(page_number)
                pdf_text += page.get_text()

            self.show_info_popup("PDF Instructions", pdf_text)
        except FileNotFoundError as e:
            self.show_info_popup("File Error", str(e))
        except Exception as e:
            self.show_info_popup("Error", str(e))
        finally:
            pdf_document.close()  # Close the PDF document when done

    def show_info_popup(self, title, content, width=180, height=50, font=None):
        popup = tk.Toplevel()
        popup.title(title)

        text_widget = tk.Text(popup, wrap='word',width=width, height=height , font=font ,padx=100,pady=50 )
        text_widget.pack()

        # Set the content temporarily to 'normal' state to insert the content
        text_widget.config(state='normal')
        text_widget.insert(tk.END, content)
        text_widget.config(state='disabled')  # Set the state back to 'disabled'

    def calculate_cross_validation_accuracy(self, algorithm, X, y, cv):
        scores = cross_val_score(algorithm, X, y, cv=cv, scoring='accuracy')
        return scores.mean() * 100

    def calculate_results(self):
        selected_dataset = self.dataset_var.get()
        dataset_path = self.datasets[selected_dataset]
        df = pd.read_csv(dataset_path)

        target_column = None
        if selected_dataset == 'Lung Cancer':
            target_column = 'LUNG_CANCER'
        elif selected_dataset == 'Brain Stroke':
            target_column = 'stroke'
        elif selected_dataset == 'Diabetes':
            target_column = 'Outcome'
        elif selected_dataset == 'Heart Disease':
            target_column = 'HeartDisease'
        elif selected_dataset == 'Hypothyroid':
            target_column = 'binaryClass'

        X = df.drop(target_column, axis=1)
        y = df[target_column]
        X = pd.get_dummies(X)

        algorithms_to_use = [algo_name for i, algo_name in enumerate(self.algorithms.keys()) if self.algorithm_vars[i].get() == 1]
        cv_techniques_to_use = [cv_name for i, cv_name in enumerate(self.cv_techniques.keys()) if self.cv_technique_vars[i].get() == 1]

        # Show the processing message
        messagebox.showinfo("Data Processing", "Please wait. The data is being processed...")
        

        results = self.perform_analysis(X, y, algorithms_to_use, cv_techniques_to_use, selected_dataset)

        # After the calculations are complete, hide or remove the GIF label
        if self.gif_label:
            self.gif_label.unload()  # Unload the GIF frames
            self.gif_label.destroy()  # Destroy the GIF label widget
            self.gif_label = None  # Reset the GIF label variable
            
        self.display_results(results)
    
    def toggle_all(self, variables):
        toggle_state = 1  # Default to select all

    # Check if any variable is not selected, then toggle to select all
        if all(var.get() == 1 for var in variables):
            toggle_state = 0  # Deselect all

    # Toggle the state of all variables
        for var in variables:
            var.set(toggle_state)

    def calculate_bootstrap_accuracy(self, algorithm, X, y):
        best_accuracy = 0
        num_bootstrap = 500  # Number of bootstrap iterations
        for _ in range(num_bootstrap):
            X_boot, y_boot = resample(X, y, random_state=42)
            model = algorithm.fit(X_boot, y_boot)
            y_pred = model.predict(X)
            accuracy = accuracy_score(y, y_pred)
            if accuracy > best_accuracy:
                best_accuracy = accuracy

        return (best_accuracy * 100, num_bootstrap)

    def perform_analysis(self, X, y, algorithms, cv_techniques, dataset_name):
        results = []

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)

            for algo_name, algo in self.algorithms.items():
                if algo_name in algorithms:
                    for cv_name, cv in self.cv_techniques.items():
                        if cv_name in cv_techniques:
                            cv_results = self.calculate_cross_validation_accuracy(algo, X, y, cv)
                            bootstrap_accuracy, num_bootstrap = self.calculate_bootstrap_accuracy(algo, X, y)
                            results.append((dataset_name, algo_name, cv_name, f"{cv_results:.2f}%", f"{num_bootstrap} Iterations", f"{bootstrap_accuracy:.2f}%"))

        return results

    def display_results(self, results):
        self.results_text.config(state='normal')
        self.results_text.delete(1.0, tk.END)

        # Create a DataFrame for results
        df_results = pd.DataFrame(results, columns=['Dataset', 'Algorithm', 'CV_Technique', 'CV_Accuracy (%)', 'NO_OF_Iterations', 'BSTP_Accuracy (%)'])

        # Convert DataFrame to a formatted string with tab spacing and extended column width
        formatted_results = df_results.to_string(index=False, justify='left', col_space=17, float_format='%.2f')
        self.results_text.insert(tk.END, formatted_results)

        # Find the best accuracy row in the DataFrame (both CV and Bootstrap)
        best_cv_accuracy_row = df_results.loc[df_results['CV_Accuracy (%)'].str.rstrip('%').astype(float).idxmax()]
        best_bootstrap_accuracy_row = df_results.loc[df_results['BSTP_Accuracy (%)'].str.rstrip('%').astype(float).idxmax()]

        # Highlight the best CV accuracy value in the text widget
        self.results_text.tag_configure("highlight", foreground="red")
        best_cv_accuracy = best_cv_accuracy_row['CV_Accuracy (%)']
        best_cv_accuracy_index = " ".join([best_cv_accuracy_row['Algorithm'], "-", best_cv_accuracy_row['CV_Technique']])
        self.results_text.insert(tk.END, f"\n\nBest CV Accuracy: {best_cv_accuracy} (Algorithm - CV Technique: {best_cv_accuracy_index})", "highlight")

        # Highlight the best Bootstrap accuracy value in the text widget
        self.results_text.tag_configure("highlight_bs", foreground="blue")
        best_bootstrap_accuracy = best_bootstrap_accuracy_row['BSTP_Accuracy (%)']
        best_bootstrap_iterations = best_bootstrap_accuracy_row['NO_OF_Iterations']
        best_bootstrap_accuracy_index = " ".join([best_bootstrap_accuracy_row['Algorithm'], "-", best_bootstrap_iterations])
        self.results_text.insert(tk.END, f"\nBest Bootstrap Accuracy: {best_bootstrap_accuracy} (Algorithm - No. of Iterations: {best_bootstrap_accuracy_index})", "highlight_bs")

        ##### Show the popup message for both best CV and best Bootstrap accuracy
        popup_message = f"Best CV Accuracy: {best_cv_accuracy} (Algorithm - CV Technique: {best_cv_accuracy_index})\n\n\n" \
                        f"Best Bootstrap Accuracy: {best_bootstrap_accuracy} (Algorithm - No. of Iterations: {best_bootstrap_accuracy_index})"
        self.show_info_popup("Best Accuracy", popup_message, width=80, height=6,font=("Helvetica",12, "bold"))
        
        self.create_bar_graph(df_results)
        self.results_text.config(state='disabled')  # Disable the text widget after updating the results

    def create_bar_graph(self, df_results):
        plt.figure(figsize=(12, 6), dpi=50)  # Adjust the figsize and dpi as needed for window size
        algo_names = df_results['Algorithm']
        cv_techniques = df_results['CV_Technique']
        cv_accuracies = df_results['CV_Accuracy (%)'].str.rstrip('%').astype(float)
        bootstrap_accuracies = df_results['BSTP_Accuracy (%)'].str.split(' ').str[-1].str.rstrip('%').astype(float)

        bar_width = 0.35
        index = range(len(df_results))

        plt.bar(index, cv_accuracies, bar_width, label='CV Accuracy')
        plt.bar([i + bar_width for i in index], bootstrap_accuracies, bar_width, label='Bootstrap Accuracy')

        plt.xlabel('Algorithm - CV Technique', fontsize=16)  # Set the fontsize here
        plt.ylabel('Accuracy (%)', fontsize=16)  # Set the fontsize here
        plt.title('Cross-Validation and Bootstrap Accuracies for Each Algorithm and CV Technique', fontsize=18)  # Set the fontsize here
        plt.xticks([i + bar_width / 2 for i in index], [f'{algo} - {cv}' for algo, cv in zip(algo_names, cv_techniques)], rotation=45, ha='right', fontsize=14)  # Set the fontsize here
        plt.legend(fontsize=14)  # Set the fontsize for legend

        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    root = tk.Tk()
    root.title("project title")
    frame = ScrollingText(root)
    # Create the ScrollingText widget and pack it at the top (north) without expanding it
    frame = ScrollingText(root)
    frame.pack(fill=tk.BOTH, expand=False, anchor='n')
    app = DatasetAnalyzerApp(root)
    root.mainloop()
