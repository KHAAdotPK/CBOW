/*
    usage/main.hh
    Q@khaa.pk
 */

// https://youtube.com/clip/UgkxOY6Uv_SIhahJIA0_WAjG5s-Uq9dPXoam?si=PPBjZOmndTI5I7sF

#include <iostream>

#ifndef WORD_EMBEDDING_ALGORITHMS_CBOW_USAGE_MAIN_HH
#define WORD_EMBEDDING_ALGORITHMS_CBOW_USAGE_MAIN_HH

#define CBOW_DEFAULT_CORPUS_FILE ".\\data\\corpus.txt"

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

#include "../lib/argsv-cpp/lib/parser/parser.hh"
#include "../lib/sundry/cooked_read_new.hh"
#include "../lib/sundry/cooked_write_new.hh"

#include "../lib/WordEmbedding-Algorithms/Word2Vec/CBOW/header.hh"

#define COMMAND "h -h help --help ? /? (Displays help screen)\nv -v version --version /v (Displays version number)\ne epoch --epoch /e (Sets epoch or number of times the training loop would run)\ncorpus --corpus (Path to the file which has the training data)\nverbose --verbose (Display of output, verbosly)\nlr --lr learningrate (Learning rate)\nrs --rs (Regularization strength)"

#endif