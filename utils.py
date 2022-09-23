# -*- coding: utf-8 -*-
"""
Created on Fri Sep 23 15:52:45 2022

@author: Heini
"""

#%% imports
import tensorflow as tf
import tensorflow_text as tf_text

#%% Text Vectorization

def standardize_da(text):
    # Split accecented characters.
    text = tf_text.normalize_utf8(text, 'NFKD')
    text = tf.strings.lower(text)
    # Keep space, a to z, and select punctuation.
    text = tf.strings.regex_replace(text, '[^ a-zæøå.?!,¿]', '')
    # Add spaces around punctuation.
    text = tf.strings.regex_replace(text, '[.?!,¿]', r' \0 ')
    # Strip whitespace.
    text = tf.strings.strip(text)

    text = tf.strings.join(['[START]', text, '[END]'], separator=' ')
    return text

def standardize_fo(text):
    # Split accecented characters.
    text = tf_text.normalize_utf8(text, 'NFKD')
    text = tf.strings.lower(text)
    # Keep space, a to z, and select punctuation.
    text = tf.strings.regex_replace(text, '[^ a-záðíóúýæø.?!,¿]', '')
    # Add spaces around punctuation.
    text = tf.strings.regex_replace(text, '[.?!,¿]', r' \0 ')
    # Strip whitespace.
    text = tf.strings.strip(text)

    text = tf.strings.join(['[START]', text, '[END]'], separator=' ')
    return text

#%% Load model
with tf.keras.utils.CustomObjectScope({'standardize_da': standardize_da, "standardize_fo": standardize_fo}):

    model = tf.keras.models.load_model(r"C:\Users\Heini\Desktop\Data Science\Fun Projects\DA-FO Translator\DA-FO Seq2Seq Models\vers_1.1(100epochs)")
#%% Translator
def translate_noatt(da_text, model = model, max_seq = 100):
    da_tokens = model.da_text_processor([da_text]) # Shape: (1, Ts)
    da_vectors = model.da_embedding(da_tokens, training = False) # Shape: (1, Ts, embedding_dim)
    da_rnn_out, fhstate, fcstate, bhstate, bcstate = model.da_rnn(da_vectors, training = False) # Shape: (batch, rnn_output_dim)
    da_hstate = tf.concat([fhstate, bhstate], -1)
    da_cstate = tf.concat([fcstate, bcstate], -1)
    state = [da_hstate, da_cstate]
    #print(da_rnn_out.shape)
    
    index_from_string = tf.keras.layers.StringLookup(
        vocabulary = model.fo_text_processor.get_vocabulary(),
        mask_token = "")
    trans = ["[START]"]
    vectors = []
    
    for i in range(max_seq):
        token = index_from_string([[trans[i]]]) # Shape: (1, 1)
        vector = model.fo_embedding(token, training = False) # Shape: (1, 1, embedding_dim)
        vectors.append(vector)
        query = tf.concat(vectors, axis = 1)
        context = model.attention(inputs = [query, da_rnn_out], training = False)
        trans_vector, hstate, cstate = model.fo_rnn(context[:,-1:,:], initial_state = state, training = False) # Shape: (1, 1, rnn_output_dim), (1, rnn_output_dim), (1, rnn_output_dim)
        state = [hstate, cstate]
        out = model.out(trans_vector) # Shape: (1, 1, fo_vocab_size)
        out = tf.squeeze(out) # Shape: (fo_vocab_size,)
        word_index = tf.math.argmax(out)
        word = model.fo_text_processor.get_vocabulary()[word_index]
        if word == "[UNK]":
            word = list(da_text.split(" "))[i]
        trans.append(word)
        if word == '[END]':
            trans = trans[:-1]
            break

    return ' '.join(trans[1:])