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

#define COMMAND "h -h help --help ? /? (Displays help screen with available commands and descriptions)\n\
v -v version --version /v (Shows software version information)\n\
e epoch --epoch /e (Sets epoch count for training iterations)\n\
corpus --corpus (Path to training data file)\n\
verbose --verbose (Enables detailed operation output during execution)\n\
lr --lr learningrate (Sets learning rate parameter for convergence control)\n\
w1 --w1 (File containing trained input weights)\n\
w2 --w2 (File containing trained output weights)\n\
input --input (Input filenames for partially trained weights during training)\n\
output --output (Output filenames for storing trained weights after training)\n\
rs --rs (Sets regularization strength to prevent overfitting; set to 0 if not using this option)\n\
--w2-t --w2-transpose --w2-swap-axes --w2-axis-flip --reshape-w2 --w2_row_to_col --w2-flip (Transposes W2 weight matrix)\n\
--ns ns --negative-samples (Number of negative samples from corpus/vocabulary; set to 0 if not using this option)\n"

#endif