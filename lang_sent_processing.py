from utils import *

loss = 'sparse_categorical_crossentropy'

class Languager():

    def __init__(self, data_path, embedding_dimensions, word_lookback):
        '''
        :param data_path: path to raw data
        :param clean_function: function to produce formatted data
        :param stateful: whether or not stateful model
        :param n_hidden: number of hidden units in recurrent layers
        :param n_layers: number of hidden rnn layers
        :param embedding_dimensions: dimension of embeedings
        :param batch_size: size of batches
        :param word_lookback: amount of previous words used in predictions
        :return:
        '''
        self.data_path = data_path
        self.e_d = embedding_dimensions
        self.w_l = word_lookback

    def data_fetch(self):
        '''

        :param data_path: path to raw data
        :param clean_function: function to clean data into corpus and list of tweet forms
        :return:
            1. list of full text (by word)
            2. list of comments, which are lists of words

        '''

        corpus, tweets, labels = tweet_cleaning(self.data_path)

        self.corpus = corpus
        self.tweets = tweets
        self.sft_labels = to_categorical(labels, num_classes=2)
        self.sig_labels = labels

    def tokens(self, vocab_size=100000, test_amt=1.0):
        '''

        :param corpus: list of text. each item = word
        :return:
            1. input list in numerical form
            2. trained tokenizer

        '''

        def get_words(joined_text):
            return sorted(list(set(joined_text)))

        def tokens_sequences(corpus, vocab_size):
            tokenizer = Tokenizer(num_words=vocab_size)
            tokenizer.fit_on_texts(corpus[:int(len(corpus) * test_amt)])
            seqs = tokenizer.texts_to_sequences(corpus[:int(len(corpus) * test_amt)])
            joined = [y for x in seqs for y in x]
            unique_tokens = get_words(joined)
            vocab_size = len(unique_tokens) + 1
            print('Fitted!')
            return seqs, joined, vocab_size, tokenizer, unique_tokens

        c_seqs, c_joined, c_vocab_size, c_tokenizer, corpus_words = tokens_sequences(self.corpus, vocab_size)
        t_seqs, t_joined, t_vocab_size, t_tokenizer, tweet_words = tokens_sequences(self.tweets, vocab_size)

        self.corpus_seqs = c_seqs
        self.corpus_joined = c_joined
        self.corpus_vocab_size = c_vocab_size
        self.corpus_tokenizer = c_tokenizer
        self.corpus_words = corpus_words

        self.tweets_seqs = t_seqs
        self.tweets_padded = pad_sequences(t_seqs, maxlen=self.w_l)
        self.tweets_joined = t_joined
        self.tweets_vocab_size = t_vocab_size
        self.tweets_tokenizer = t_tokenizer
        self.tweets_words = tweet_words

    def create_sequences(self):
        '''

        :param text: list of words
        :param word_lookback: number of words to consider in predictions
        :return: n lists of sequences for word

        '''
        inputs = [[self.corpus_joined[i + n] for i in range(0, len(self.corpus_joined) - 1 - self.w_l, self.w_l)] for n
                  in range(self.w_l)]

        outputs = [[self.corpus_joined[i + n] for i in range(1, len(self.corpus_joined) - self.w_l, self.w_l)] for n in
                   range(self.w_l)]

        output = [self.corpus_joined[i + self.w_l] for i in range(0, len(self.corpus_joined) - 1 - self.w_l, self.w_l)]

        self.inputs = inputs
        self.outputs = outputs
        self.output = output

    def inputs_outputs(self):
        '''

        :param inputs: list of input sequences
        :param outputs: list of output sequences
        :param output: output sequence for single item prediction
        :return:
            1. xs
            2. ys
            3. y
            4. xs_rnn
            5. ys_rnn
        '''

        xs = [np.stack(c[:-2]) for c in self.inputs]
        print(f'Input variable "xs" created.')
        print(f'List of type {type(xs[0])} and shape {xs[0].shape}')
        print(f'-----------------------------------')

        ys = [np.stack(c[:-2]) for c in self.outputs]
        print(f'Input variable "ys" created.')
        print(f'List of type {type(ys[0])} and shape {ys[0].shape}')
        print(f'-----------------------------------')

        y = np.stack(self.output[:-2])
        print(f'Input variable "y" created.')
        print(f'Type {type(y)} and shape {y.shape}')
        print(f'-----------------------------------')

        x_rnn = np.stack(np.squeeze(xs), axis=1)
        print(f'Input variable "x_rnn" created.')
        print(f'Type {type(x_rnn)} and shape {x_rnn.shape}')
        print(f'-----------------------------------')

        y_rnn = np.atleast_3d(np.stack(ys, axis=1))
        print(f'Input variable "y_rnn" created.')
        print(f'Type {type(y_rnn)} and shape {y_rnn.shape}')
        print(f'-----------------------------------')

        self.xs = xs
        self.ys = ys
        self.y = y
        self.x_rnn = x_rnn
        self.y_rnn = y_rnn

    def language_train(self, epochs, batch_size, lr, layers=[], dropouts=[], model='new', valid_amt=0, stateful=True,
                       compile=True, shuffle=False, save=False):
        mx = len(self.x_rnn) // batch_size * batch_size
        trn_gen, val_gen = trn_val_gens(self.x_rnn[:mx], self.y_rnn[:mx], batch_size, valid_amt)

        vocab_size = self.corpus_vocab_size
        cyclic_lr = CyclicLR(base_lr=lr, max_lr=.01, step_size=len(self.x_rnn) / batch_size * 2, gamma=0.99)
        checkpoints = ModelCheckpoint('lang_model.h5', monitor='loss', verbose=1, save_best_only=True,
                                      save_weights_only=False, mode='auto', period=1)

        optimizer = Adam(lr=lr, decay=1e-6)

        callbacks = [cyclic_lr]
        if save == True: callbacks.append(checkpoints)

        if model == 'new':
            input = Input(shape=(self.w_l), batch_shape=(batch_size, self.w_l))
            emb = Embedding(vocab_size, self.e_d)(input)
            x = SpatialDropout1D(dropouts[0])(emb)
            x = BatchNormalization()(x)

            for i in range(len(layers)):
                if stateful == True:
                    x = LSTM(layers[i], return_sequences=True, stateful=True, dropout=dropouts[i],
                             recurrent_dropout=dropouts[i])(x)
                else:
                    x = LSTM(layers[i], return_sequences=True, stateful=False, dropout=dropouts[i],
                             recurrent_dropout=dropouts[i])(x)
            output = TimeDistributed(Dense(vocab_size, activation='softmax'))(x)
        if compile:
            model = Model(inputs=input, outputs=output)
            model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer)
        model.summary()

        print('Feeding in generators...')
        print(f'Length trn_gen: {len(trn_gen)}')
        print(f'Length val_gen: {len(val_gen)}')

        history = model.fit_generator(generator=trn_gen,
                                      steps_per_epoch=len(trn_gen),
                                      epochs=epochs,
                                      validation_data=val_gen,
                                      validation_steps=len(val_gen),
                                      max_queue_size=8,
                                      workers=multiprocessing.cpu_count(),
                                      callbacks=callbacks,
                                      verbose=1,
                                      shuffle=shuffle)

        self.lang_model = model
        self.lang_history = history

    def sentiment_train(self, epochs, batch_size, lr, layers=[], dropouts=[], model='new', valid_amt=0, compile=True,
                        language_model=None, shuffle=False, save=False):
        mx = len(self.tweets_padded) // batch_size * batch_size
        trn_gen, val_gen = trn_val_gens(self.tweets_padded[:mx], self.sft_labels[:len(self.tweets_padded)][:mx],
                                        batch_size, valid_amt)

        cyclic_lr = CyclicLR(base_lr=lr, max_lr=100 * lr, step_size=len(trn_gen) / batch_size * 2, gamma=0.99)
        checkpoints = ModelCheckpoint('sent_model.h5', monitor='loss', verbose=1, save_best_only=True,
                                      save_weights_only=False, mode='auto', period=1)

        callbacks = [cyclic_lr]
        if save == True: callbacks.append(checkpoints)

        optimizer = Adam(lr=lr, decay=1e-6)

        vocab_size = self.corpus_vocab_size

        if model == 'rnn':
            model = Sequential()
            if language_model is not None:
                lang_embedding = language_model.layers[1]
                emb_weights = lang_embedding.get_weights()
                model.add(
                    Embedding(vocab_size, self.e_d, input_length=self.w_l, batch_input_shape=(batch_size, self.w_l),
                              weights=emb_weights, mask_zero=True))
            else:
                model.add(
                    Embedding(vocab_size, self.e_d, input_length=self.w_l, batch_input_shape=(batch_size, self.w_l)))
            model.add(SpatialDropout1D(dropouts[0]))

            model.add(BatchNormalization())
            for i in range(len(layers) - 3):
                model.add(Bidirectional(
                    LSTM(layers[i], return_sequences=True, dropout=dropouts[i], recurrent_dropout=dropouts[i])))

            model.add(Bidirectional(
                LSTM(layers[-3], return_sequences=False, dropout=dropouts[-3], recurrent_dropout=dropouts[-3])))

            model.add(Dense(layers[-2], activation='elu'))
            model.add(Dropout(dropouts[-2]))
            model.add(BatchNormalization())
            model.add(Dense(layers[-1], activation='elu'))
            model.add(Dropout(dropouts[-1]))
            model.add(BatchNormalization())
            model.add(Dense(2, activation='softmax'))

        if model == 'cnn':
            model = Sequential()
            if language_model is not None:
                lang_embedding = language_model.layers[1]
                emb_weights = lang_embedding.get_weights()
                model.add(
                    Embedding(vocab_size, self.e_d, input_length=self.w_l, batch_input_shape=(batch_size, self.w_l),
                              weights=emb_weights, mask_zero=True))
            else:
                model.add(
                    Embedding(vocab_size, self.e_d, input_length=self.w_l, batch_input_shape=(batch_size, self.w_l)))
            model.add(SpatialDropout1D(dropouts[0]))
            model.add(BatchNormalization())
            model.add(NonMasking())
            for i in range(len(layers) - 3):
                model.add(Conv1D(layers[i], kernel_size=3 - i, activation='elu', padding='same'))
                model.add(Conv1D(layers[i], kernel_size=3 - i, activation='elu', padding='same'))
                model.add(Conv1D(layers[i], kernel_size=3 - i, activation='elu', padding='same'))
                model.add(Conv1D(layers[i], kernel_size=3 - i, activation='elu', padding='same'))
                model.add(Dropout(dropouts[i]))

            model.add(Flatten())
            model.add(Dense(layers[-2], activation='elu'))
            model.add(Dropout(dropouts[-2]))
            model.add(BatchNormalization())
            model.add(Dense(layers[-1], activation='elu'))
            model.add(Dropout(dropouts[-1]))
            model.add(BatchNormalization())
            model.add(Dense(2, activation='softmax'))

        if compile:
            model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        model.summary()

        print('Feeding in generators...')
        print(f'Length trn_gen: {len(trn_gen)}')
        print(f'Length val_gen: {len(val_gen)}')

        history = model.fit_generator(generator=trn_gen,
                                      epochs=epochs,
                                      validation_data=val_gen,
                                      max_queue_size=8,
                                      workers=multiprocessing.cpu_count(),
                                      verbose=1,
                                      shuffle=shuffle)
        self.sent_model = model
        self.sent_history = history

    def predict_text(self, model, text):
        tokenizer = self.corpus_tokenizer
        idxs = [tokenizer.word_index[c] for c in text]
        arr = np.array(idxs)[np.newaxis, :]
        p = model.predict(arr)[0]
        print(list(text))
        return [self.corpus_words[np.argmax(o)] for o in p]