from datasets import load_dataset

def fetch_dataset():
    """
    Fetches the 'alvanlii/reddit-comments-uwaterloo' dataset from Hugging Face.
    """
    dataset = load_dataset("alvanlii/reddit-comments-uwaterloo", 'year_2024')
    return dataset

def preprocess_data(dataset):
    """
    Extracts and preprocesses the text data from the dataset.
    """
    texts = dataset['train']['content']
    return texts
