/*
    usage/main.hh
    Q@khaa.pk
 */

#include <iostream>

#include "../lib/argsv-cpp/lib/parser/parser.hh"
#include "../lib/sundry/cooked_read_new.hh"
#include "../lib/sundry/cooked_write_new.hh"
#include "../lib/read_write_weights/header.hh"

#ifndef WORD_EMBEDDING_ALGORITHMS_CBOW_USAGE_MAIN_HH
#define WORD_EMBEDDING_ALGORITHMS_CBOW_USAGE_MAIN_HH

#define CBOW_DEFAULT_CORPUS_FILE ".\\data\\corpus.txt"

#ifdef SKIP_GRAM_CONTEXT_WINDOW_SIZE
#undef SKIP_GRAM_CONTEXT_WINDOW_SIZE
#endif
#define SKIP_GRAM_CONTEXT_WINDOW_SIZE 4

#ifdef SKIP_GRAM_WINDOW_SIZE
#undef SKIP_GRAM_WINDOW_SIZE
#endif
#define SKIP_GRAM_WINDOW_SIZE 4

#ifdef GRAMMAR_END_OF_TOKEN_MARKER
#undef GRAMMAR_END_OF_TOKEN_MARKER
#endif
#ifdef GRAMMAR_END_OF_LINE_MARKER
#undef GRAMMAR_END_OF_LINE_MARKER
#endif
#ifdef GRAMMAR_START_OF_COMMENT_MARKER
#undef GRAMMAR_START_OF_COMMENT_MARKER	
#endif

#define GRAMMAR_END_OF_TOKEN_MARKER ' '
#define GRAMMAR_END_OF_LINE_MARKER '\n'
#define GRAMMAR_START_OF_COMMENT_MARKER '('

#include "../lib/WordEmbedding-Algorithms/Word2Vec/CBOW/header.hh"

#define COMMAND "h -h help --help ? /? (Displays the help screen, listing available commands and their descriptions.)\n\
v -v version --version /v (Shows the current version of the software.)\n\
e epoch --epoch /e (Sets the epoch count, determining the number of iterations for the training loop.)\n\
corpus --corpus (Path to the file which has the training data.)\n\
verbose --verbose (Enables detailed output for each operation during execution.)\n\
lr --lr learningrate (Defines the learning rate parameter to control the rate of convergence.)\n\
w1 --w1 (Specifies the file containing the trained input weights.)\n\
w2 --w2 (Specifies the file containing the trained output weights.)\n\
input --input (Specifies the filenames to retrieve the partially input trained weights during training.)\n\
output --output (Specifies the filenames to store the output trained weights after completion of training.)\n\
rs --rs (Sets the regularization strength, used to prevent overfitting.)\n\
vc --vc --validation_corpus (Path to the file which has the data the model has never seen during training.)\n"

#endif