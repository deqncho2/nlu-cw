#### QUESTION 2 B #####

from rnn1 import *
import numpy as np

data_folder = "../data"
np.random.seed(2018)

train_size = 25000
dev_size = 1000
vocab_size = 2000
epoch_number = 10

### CHANGE THESE ACCORDING TO 2(a)!!! ####
### ALSO, CHANGE COMPUTE_LOSS TO COMPUTE_MEAN_LOSS IN TRAIN#####
hdim = 25
lookback = 0
lr = 0.5
##########################################

# get the data set vocabulary
vocab = pd.read_table(data_folder + "/vocab.wiki.txt", header=None, sep="\s+", index_col=0, names=['count', 'freq'], )
num_to_word = dict(enumerate(vocab.index[:vocab_size]))
word_to_num = invert_dict(num_to_word)

# calculate loss vocabulary words due to vocab_size
fraction_lost = fraq_loss(vocab, word_to_num, vocab_size)
print("Retained %d words from %d (%.02f%% of all tokens)\n" % (vocab_size, len(vocab), 100*(1-fraction_lost)))

docs = load_lm_dataset(data_folder + '/wiki-train.txt')
S_train = docs_to_indices(docs, word_to_num, 1, 1)
X_train, D_train = seqs_to_lmXY(S_train)

# Load the dev set (for tuning hyperparameters)
docs = load_lm_dataset(data_folder + '/wiki-dev.txt')
S_dev = docs_to_indices(docs, word_to_num, 1, 1)
X_dev, D_dev = seqs_to_lmXY(S_dev)

X_train = X_train[:train_size]
D_train = D_train[:train_size]
X_dev = X_dev[:dev_size]
D_dev = D_dev[:dev_size]

# q = best unigram frequency from omitted vocab
# this is the best expected loss out of that set
q = vocab.freq[vocab_size] / sum(vocab.freq[vocab_size:])

##########################
r = RNN(vocab_size, hdim, vocab_size)
run_loss = r.train(X_train, D_train, X_dev, D_dev, epochs=epoch_number, learning_rate=lr, back_steps=lookback)

##########################

# run_loss = -1
adjusted_loss = adjust_loss(run_loss, fraction_lost, q)

print("Unadjusted: %.03f" % np.exp(run_loss))
print("Adjusted for missing vocab: %.03f" % np.exp(adjusted_loss))

##write results
file_q2a = open("q2b.txt", "w")
file_q2a.write("hidden dims: " + str(hdim) + " lookback: " + str(lookback) + " learning-rate: " + str(lr) + " loss: " + str(np.exp(run_loss)) + " adjusted loss: " + str(np.exp(adjusted_loss)) + "\n")
file_q2a.close()

np.save("rnn.W", r.W)
np.save("rnn.U", r.U)
np.save("rnn.V", r.V)