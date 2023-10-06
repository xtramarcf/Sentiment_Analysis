import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import backend.evaluate_news_sentiment as ens


# Function for validate input values. Only accept positive values.
def validate_positive_number(input_text):
    try:
        value = int(input_text)
        if value >= 0:
            return True
        else:
            return False
    except ValueError:
        return False


# Creates the root window with all its components.
def start_application():
    # Method will start, when the evaluate button is clicked.
    # The method shows a message, that the user has to wait until everything is loaded.
    # The method get_all_sentiment is used for scraping all the articles and analyzing
    # them regarding the sentiment.
    def evaluate_button_clicked():
        result_label.config(text=f"")

        # Checks, if input fields are filled.
        if input_max_items.get() == "" or input_keyword.get() == "":
            result_label.config(text=f"Please enter input values!")
        else:
            # Shows that data is loading
            label_progress = ttk.Label(frame, text="\nPlease wait...")
            label_progress.grid(row=6, column=0, columnspan=2, padx=10, pady=10)
            root.update()
            max_items = int(input_max_items.get())
            keyword = input_keyword.get()
            archive_search = checkbox_archive_search.get()
            result = ens.get_all_sentiment(keyword, archive_search, max_items=max_items)
            if result == 1:
                result_label.config(text=f"No articles found.")
            else:
                article_found = str(result[0])
                event_registry_api_sentiment = result[1]
                twitter_sentiment = result[3]
                news_sentiment = result[2]

                result_label.config(text=f"Result: \n"
                                         f"Number of found articles: {article_found}\n"
                                         f"API Sentiment Analysis:    {event_registry_api_sentiment}\n"
                                         f"Model Sentiment Analysis:    {twitter_sentiment}\n")
                                         # f"Model 2 Sentiment Analysis:    {news_sentiment}\n")
            label_progress.grid_forget()
        root.update()

    # Creates the root window with a defined size.
    root = tk.Tk()
    root.title("Sentiment Analysis")
    root.geometry("470x450")
    root.resizable(width=False, height=False)

    # Create a frame for better organization
    frame = ttk.Frame(root)
    frame.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")

    # Show the sentiment image.
    image = Image.open('frontend/sentiment.webp').resize((300, 150))
    photo = ImageTk.PhotoImage(image)
    sentiment_image = ttk.Label(frame, image=photo)
    sentiment_image.grid(row=1, column=0, columnspan=2, padx=10, pady=10)

    # Input keyword/company
    label_keyword = ttk.Label(frame, text="Company / Keyword:")
    label_keyword.grid(row=2, column=0, padx=10, pady=5, sticky="w")
    input_keyword = ttk.Entry(frame)
    input_keyword.grid(row=2, column=1, padx=10, pady=5, sticky="w")

    # Checkbox for selecting 31 days search or archive search
    label_checkbox = ttk.Label(frame, text="True = Archive search, False = Last 31 days search:")
    label_checkbox.grid(row=3, column=0, padx=10, pady=5, sticky="w")
    checkbox_archive_search = tk.BooleanVar()
    checkbox = ttk.Checkbutton(frame, variable=checkbox_archive_search)
    checkbox.grid(row=3, column=1, padx=10, pady=5, sticky="w")

    # Input for setting the max amount of articles to evaluate.
    # Accepts only positive values
    validate_positive_number_func = root.register(validate_positive_number)
    label_max_items = ttk.Label(frame, text="Maximum amount of articles:")
    label_max_items.grid(row=4, column=0, padx=10, pady=5, sticky="w")
    input_max_items = ttk.Entry(frame, validate="key", validatecommand=(validate_positive_number_func, "%P"))
    input_max_items.grid(row=4, column=1, padx=10, pady=5, sticky="w")

    # Create label for displaying the results.
    result_label = ttk.Label(frame, text="", wraplength=400)
    result_label.grid(row=6, column=0, columnspan=2, padx=10, pady=10)

    # Button for getting and evaluating the news by a given keyword.
    button = ttk.Button(frame, text="Evaluate company sentiment",
                        command=evaluate_button_clicked)
    button.grid(row=5, column=0, columnspan=2, padx=10, pady=10)

    # Adjust column and row weights for resizing
    frame.grid_rowconfigure(6, weight=1)
    frame.grid_columnconfigure(1, weight=1)

    # Starts the main root with runs to frontend.
    root.mainloop()
