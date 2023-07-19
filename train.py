import torch
import torch.optim as optim

from utils import read_data_from_file, make_word_dictionary, make_label_dictionary, make_label_vector, get_index_vector
from model import FastTextClassifier

torch.manual_seed(1)

# hyperparams
unk_threshold = 5
hidden_size = 30
number_of_epochs = 5
max_ngrams = 2

# Get data - use only a part of it to speed up training/testing
training_data = read_data_from_file('data/train.csv', 10000)
test_data = read_data_from_file('data/test.csv', 1000)

# Get dictionaries
word_dictionary = make_word_dictionary(training_data, unk_threshold=unk_threshold, max_ngrams=max_ngrams)
label_dictionary = make_label_dictionary(training_data)


# Get FastText classifier
model = FastTextClassifier(vocab_size=len(word_dictionary),
                           hidden_size=hidden_size,
                           num_labels=len(label_dictionary)
                           )

# define the loss and optimizer
loss_function = torch.nn.NLLLoss()
optimizer = optim.Adam(model.parameters())

# vectorize the data once
training_data = [
    (make_label_vector(label, label_dictionary), (get_index_vector(instance, word_dictionary, max_ngrams=max_ngrams)))
                           for instance, label in training_data]
test_data = [
    (make_label_vector(label, label_dictionary), (get_index_vector(instance, word_dictionary, max_ngrams=max_ngrams)))
                           for instance, label in test_data] # test data is vectorized once as well, because evaluation is done every epoch

# Training
def evaluate_model(model, test_data, word_dictionary, label_dictionary) -> float:
    # evaluate the model
    t = 0

    with torch.no_grad():

        for label, words in test_data:

            log_probs = model(words)

            if torch.argmax(log_probs).item() == label:
                t += 1

        return t / len(test_data)

# Go over the training dataset
for epoch in range(number_of_epochs):

    # go through each training data point
    for target, words in training_data:
        model.zero_grad()

        log_probabilities_for_each_class = model.forward(words)
        loss = loss_function(log_probabilities_for_each_class, target)

        loss.backward()
        optimizer.step()

    accuracy = evaluate_model(model, test_data, word_dictionary, label_dictionary)
    print(f"epoch: {epoch+1} acc: {accuracy}")