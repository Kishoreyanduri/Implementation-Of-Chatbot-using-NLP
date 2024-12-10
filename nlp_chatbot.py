import nltk
import numpy as np
import tensorflow as tf
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('wordnet')

class NLPChatbot:
    def __init__(self):
        # Initialize lemmatizer
        self.lemmatizer = WordNetLemmatizer()
        
        # Predefined intents and responses
        self.intents = {
            'greeting': [
                'Hello!', 
                'Hi there!', 
                'Greetings!',
                'Nice to meet you!'
            ],
            'goodbye': [
                'Goodbye!', 
                'See you later!', 
                'Bye!',
                'Take care!'
            ],
            'help': [
                'How can I assist you today?',
                'I\'m here to help. What do you need?',
                'What can I do for you?'
            ],
            'unknown': [
                'I\'m not sure I understand.',
                'Could you rephrase that?',
                'I don\'t quite get what you mean.'
            ]
        }
        
        # Training data preparation
        self.training_sentences = []
        self.training_labels = []
        self.prepare_training_data()
        
        # Model parameters
        self.max_words = 1000
        self.max_len = 20
        
        # Prepare model
        self.tokenizer = Tokenizer(num_words=self.max_words, oov_token='<OOV>')
        self.tokenizer.fit_on_texts(self.training_sentences)
        
        # Encode labels
        self.label_encoder = LabelEncoder()
        self.encoded_labels = self.label_encoder.fit_transform(self.training_labels)
        
        # Prepare sequences
        self.sequences = self.tokenizer.texts_to_sequences(self.training_sentences)
        self.padded_sequences = pad_sequences(self.sequences, maxlen=self.max_len, padding='post', truncating='post')
        
        # Build neural network model
        self.model = self.build_model()
        
    def prepare_training_data(self):
        """Prepare training data for the chatbot"""
        training_data = [
            ("hello", "greeting"),
            ("hi", "greeting"),
            ("hey", "greeting"),
            ("goodbye", "goodbye"),
            ("bye", "goodbye"),
            ("see you", "goodbye"),
            ("help me", "help"),
            ("what can you do", "help"),
            ("I need assistance", "help")
        ]
        
        for pattern, intent in training_data:
            # Tokenize and lemmatize
            words = word_tokenize(pattern.lower())
            lemmatized_words = [self.lemmatizer.lemmatize(word) for word in words]
            
            self.training_sentences.append(' '.join(lemmatized_words))
            self.training_labels.append(intent)
    
    def build_model(self):
        """Build neural network model for intent classification"""
        model = Sequential([
            Embedding(self.max_words, 16, input_length=self.max_len),
            LSTM(32),
            Dense(16, activation='relu'),
            Dropout(0.5),
            Dense(len(set(self.training_labels)), activation='softmax')
        ])
        
        model.compile(
            optimizer='adam', 
            loss='sparse_categorical_crossentropy', 
            metrics=['accuracy']
        )
        
        # Train the model
        model.fit(
            self.padded_sequences, 
            self.encoded_labels, 
            epochs=50, 
            verbose=0
        )
        
        return model
    
    def preprocess_text(self, text):
        """Preprocess input text"""
        # Tokenize and lemmatize
        words = word_tokenize(text.lower())
        lemmatized_words = [self.lemmatizer.lemmatize(word) for word in words]
        return ' '.join(lemmatized_words)
    
    def predict_intent(self, text):
        """Predict the intent of the input text"""
        # Preprocess text
        processed_text = self.preprocess_text(text)
        
        # Convert to sequence
        sequence = self.tokenizer.texts_to_sequences([processed_text])
        padded_sequence = pad_sequences(sequence, maxlen=self.max_len, padding='post', truncating='post')
        
        # Predict intent
        prediction = self.model.predict(padded_sequence)
        intent_index = np.argmax(prediction)
        predicted_intent = self.label_encoder.inverse_transform([intent_index])[0]
        
        return predicted_intent
    
    def get_response(self, intent):
        """Get a response based on the predicted intent"""
        return np.random.choice(self.intents.get(intent, self.intents['unknown']))
    
    def chat(self):
        """Interactive chat loop"""
        print("Chatbot: Hello! Type 'quit' to exit.")
        
        while True:
            user_input = input("You: ")
            
            if user_input.lower() == 'quit':
                print("Chatbot: Goodbye!")
                break
            
            # Predict intent
            intent = self.predict_intent(user_input)
            
            # Get and print response
            response = self.get_response(intent)
            print(f"Chatbot: {response}")

# Example usage
if __name__ == "__main__":
    chatbot = NLPChatbot()
    chatbot.chat()