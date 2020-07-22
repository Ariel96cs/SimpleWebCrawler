import os
import csv
import subprocess
import re
import random
import numpy as np
from tqdm import tqdm

def get_row_vector(matrix, row_id):
  return matrix[row_id, :]

def get_column_vector(matrix, col_id):
  return matrix[:, col_id]

def create_term_document_matrix(line_tuples, document_names, vocab):
    '''Returns a numpy array containing the term document matrix for the input lines.

    Inputs:
      line_tuples: A list of tuples, containing the name of the document and 
      a tokenized line from that document.
      document_names: A list of the document names
      vocab: A list of the tokens in the vocabulary

    # NOTE: THIS DOCSTRING WAS UPDATED ON JAN 24, 12:39 PM.

    Let m = len(vocab) and n = len(document_names).

    Returns:
      td_matrix: A mxn numpy array where the number of rows is the number of words
          and each column corresponds to a document. A_ij contains the
          frequency with which word i occurs in document j.
    '''

    vocab_to_id = dict(zip(vocab, range(0, len(vocab))))
    docname_to_id = dict(zip(document_names, range(0, len(document_names))))

    # YOUR CODE HERE

    # matrix = np.array([[0 for _ in document_names] for _ in vocab])
    matrix = np.zeros((len(vocab),len(document_names)))

    _fill_term_document_matrix(matrix, line_tuples, docname_to_id, vocab_to_id)

    return matrix


def _fill_term_document_matrix(matrix,line_tuples,dict_doc_pos,dict_vocab_pos):
    for docName, tokenized in tqdm(line_tuples):
        doc_pos = dict_doc_pos[docName]
        for token in tokenized:
            try:
                docIndex, tokenIndex = doc_pos, dict_vocab_pos[token]
                matrix[tokenIndex][docIndex] += 1
            except KeyError:
                continue
    return matrix

def create_term_context_matrix(line_tuples, vocab, context_window_size=1):
  '''Returns a numpy array containing the term context matrix for the input lines.

  Inputs:
    line_tuples: A list of tuples, containing the name of the document and 
    a tokenized line from that document.
    vocab: A list of the tokens in the vocabulary

  # NOTE: THIS DOCSTRING WAS UPDATED ON JAN 24, 12:39 PM.

  Let n = len(vocab).

  Returns:
    tc_matrix: A nxn numpy array where A_ij contains the frequency with which
        word j was found within context_window_size to the left or right of
        word i in any sentence in the tuples.
  '''

  vocab_to_id = dict(zip(vocab, range(0, len(vocab))))

  # YOUR CODE HERE
  # esta matriz es simetrica
  print("Inicializando la matriz")
  # matrix = np.array([[0 for _ in vocab] for _ in tqdm(vocab)])
  matrix = np.zeros((len(vocab),len(vocab)))
  print("Inicializada a matriz de terminos todo en 0")
  for doc,line in tqdm(line_tuples):
      for i,word in enumerate(line):
          for j in range(len(line)):
              if j-i > context_window_size:
                  break
              elif i-j > 0 and i-j > context_window_size:
                  continue
              word2_id = vocab_to_id[line[j]]
              word1_id = vocab_to_id[word]
              matrix[word1_id,word2_id] += 1
              matrix[word2_id,word1_id] += 1

  return matrix

def create_PPMI_matrix(term_context_matrix):
  '''Given a term context matrix, output a PPMI matrix.
  
  See section 15.1 in the textbook.
  
  Hint: Use numpy matrix and vector operations to speed up implementation.
  
  Input:
    term_context_matrix: A nxn numpy array, where n is
        the numer of tokens in the vocab.
  
  Returns: A nxn numpy matrix, where A_ij is equal to the
     point-wise mutual information between the ith word
     and the jth word in the term_context_matrix.
  '''       
  
  # YOUR CODE HERE
  print(" Inicializando la matriz")
  ppmi_matrix = np.copy(term_context_matrix)
  total = np.sum(term_context_matrix)
  print(" Inicializada")

  pi = np.zeros(term_context_matrix.shape[0])
  pj = np.zeros(term_context_matrix.shape[1])

  for i in range(len(pi)):
      pi[i] = np.sum(term_context_matrix[i, :])
      pj[i] = np.sum(term_context_matrix[:, i])

  ppmi_matrix = np.multiply(ppmi_matrix,total)
  vectors_sum = np.multiply(pi,pj)
  ppmi_matrix = np.divide(ppmi_matrix,vectors_sum)

  return ppmi_matrix

def create_tf_idf_matrix(term_document_matrix):
  '''Given the term document matrix, output a tf-idf weighted version.

  See section 15.2.1 in the textbook.
  
  Hint: Use numpy matrix and vector operations to speed up implementation.

  Input:
    term_document_matrix: Numpy array where each column represents a document 
    and each row, the frequency of a word in that document.

  Returns:
    A numpy array with the same dimension as term_document_matrix, where
    A_ij is weighted by the inverse document frequency of document h.
  '''

  # YOUR CODE HERE
  tftd = np.log10(term_document_matrix+1)
  N = term_document_matrix.shape[1]
  dft = np.zeros(term_document_matrix.shape[0])
  for t in range(term_document_matrix.shape[1]):
    for j in range(N):
      if term_document_matrix[t,j] == 0: continue
      dft[t] += 1
  idft = np.log10(np.divide(N,dft))

  # return np.multiply(tftd,idft)
  return np.transpose(np.multiply(np.transpose(tftd),idft))

def compute_cosine_similarity(vector1, vector2):
  '''Computes the cosine similarity of the two input vectors.

  Inputs:
    vector1: A nx1 numpy array
    vector2: A nx1 numpy array

  Returns:
    A scalar similarity value.
  '''

  c = np.array([2 for _ in range(len(vector1))])
  numerator = np.dot(vector1, vector2)
  denominator = np.sqrt(np.sum(np.power(vector1, c))*np.sum(np.power(vector2, c)))
  return numerator/denominator

def compute_jaccard_similarity(vector1, vector2):
  '''Computes the cosine similarity of the two input vectors.

  Inputs:
    vector1: A nx1 numpy array
    vector2: A nx1 numpy array

  Returns:
    A scalar similarity value.
  '''
  
  # YOUR CODE HERE
  c = np.array([2 for _ in range(len(vector1))])
  numerator = np.dot(vector1, vector2)
  denominator = np.sum(np.power(vector1, c)) + np.sum(np.power(vector2, c)) - numerator
  return numerator/denominator

def compute_dice_similarity(vector1, vector2):
  '''Computes the cosine similarity of the two input vectors.

  Inputs:
    vector1: A nx1 numpy array
    vector2: A nx1 numpy array

  Returns:
    A scalar similarity value.
  '''

  # YOUR CODE HERE
  numerator = 2*np.dot(vector1, vector2)
  c = np.array([2 for _ in range(len(vector1))])
  denominator = np.sum(np.power(vector1, c)) + np.sum(np.power(vector2, c))
  return numerator/denominator

def rank_plays(target_play_index, term_document_matrix, similarity_fn):
  ''' Ranks the similarity of all of the plays to the target play.

  # NOTE: THIS DOCSTRING WAS UPDATED ON JAN 24, 12:51 PM.

  Inputs:
    target_play_index: The integer index of the play we want to compare all others against.
    term_document_matrix: The term-document matrix as a mxn numpy array.
    similarity_fn: Function that should be used to compared vectors for two
      documents. Either compute_dice_similarity, compute_jaccard_similarity, or
      compute_cosine_similarity.

  Returns:
    A length-n list of integer indices corresponding to play names,
    ordered by decreasing similarity to the play indexed by target_play_index
  '''
  
  # YOUR CODE HERE
  result = [(similarity_fn(term_document_matrix[:,target_play_index],term_document_matrix[:,i]),i) for i in tqdm(range(term_document_matrix.shape[1])) if i != target_play_index]
  result.sort(reverse=True)
  return [i for _,i in result]

def rank_words(target_word_index, matrix, similarity_fn):
  ''' Ranks the similarity of all of the words to the target word.

  # NOTE: THIS DOCSTRING WAS UPDATED ON JAN 24, 12:51 PM.

  Inputs:
    target_word_index: The index of the word we want to compare all others against.
    matrix: Numpy matrix where the ith row represents a vector embedding of the ith word.
    similarity_fn: Function that should be used to compared vectors for two word
      embeddings. Either compute_dice_similarity, compute_jaccard_similarity, or
      compute_cosine_similarity.

  Returns:
    A length-n list of integer word indices, ordered by decreasing similarity to the 
    target word indexed by word_index
  '''

  # YOUR CODE HERE
  result = [(similarity_fn(matrix[:,target_word_index], matrix[:,i]),i) for i in
            tqdm(range(matrix.shape[0])) if i != target_word_index]
  result.sort(reverse=True)
  return [i for _,i in result]

similarity_fns = [compute_cosine_similarity, compute_jaccard_similarity, compute_dice_similarity]