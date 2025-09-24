:: -------------------------------------------------------------------------
:: This script parses command-line arguments for an encoder-decoder
:: program. It supports the following options:
::  - "verbose": Enables verbose output
::  - "e [number]": Sets the number of epochs (default is 1)
::  - "w1 [filename]": Specifies the path to the weights file
::  - "w2 [filename]": Specifies the path to the weights file
::  - "lr [number]": Sets the learning rate (default is 0.09)
::  - "rs [number]": Sets the regularization strength (default is 0.000000)
::  - "corpus [filename]": Specifies the path to the corpus file
::  - "input": Specifies that the program should read from weight files
::  - "output": Specifies that the program should write to weight files
::
:: The script loops through arguments using SHIFT and handles each 
:: option accordingly.
:: Delayed expansion is enabled to support dynamic variable updates
:: if needed in future modifications.
:: ------------------------------------------------------------------------
:: set corpus_filename_option="./data/NEW-INPUT.txt"

@echo off
setlocal enabledelayedexpansion

set verbose_option=
set w1_filename_option="./data/weights/w1p.dat"
set w2_filename_option="./data/weights/w2p.dat"
set output_w1_filename_option=
set output_w2_filename_option=
set corpus_filename_option="./data/adult_abdominal_pain_input.txt"
set epochs_option=1
set learning_rate_option=0.025
set regularization_strength_option=0.000000
set negative_samples_option="--negative_samples 0"
set input_option=
set output_option=
set cbow_debug_forward_pair="CbowDebugForwardPair=no"
set error_message_text=
set help_option=
set validation_corpus_option=
set w2_transpose_option=

:start_parsing_args

if "%1"=="verbose" (
    set verbose_option=verbose
    shift
    goto :start_parsing_args
) else if "%1"=="--w2-t" (
   set w2_transpose_option="--w2_row_to_col"
   shift
   goto :start_parsing_args         
) else if "%1"=="e" (
    if "%2" neq "" (    
        set epochs_option="%2"        
        shift
    ) 
    shift
    goto :start_parsing_args
) else if "%1"=="w1" (
    if "%2" neq "" (
        set w1_filename_option="%2"
        shift
    )
    shift
    goto :start_parsing_args
) else if "%1"=="lr" (
    if "%2" neq "" (
        set learning_rate_option="%2"
        shift
    )
    shift
    goto :start_parsing_args
) else if "%1"=="rs" (
    if "%2" neq "" (
        set regularization_strength_option="%2"
        shift
    )
    shift
    goto :start_parsing_args
) else if "%1"=="w1" (
    if "%2" neq "" (
        set w1_filename_option="%2"
        shift
    )
    shift
    goto :start_parsing_args
) else if "%1"=="w2" (
    if "%2" neq "" (
        set w2_filename_option="%2"
        shift
    )
    shift
    goto :start_parsing_args
) else if "%1"=="corpus" (
    if "%2" neq "" (
        set corpus_filename_option="%2"
        shift
    )
    shift
    goto :start_parsing_args
) else if "%1"=="input" (    
    set input_option="%1"
    shift    
    goto :start_parsing_args
) else if "%1"=="output" (    
    set output_option="%1"
    set error_message_text="`output` option requires 2 additional parameters"
    if "%2"=="" (
        goto :error_message        
    )
    if "%3"=="" (
        goto :error_message
    )
    set output_w1_filename_option="%2"
    set output_w2_filename_option="%3"
    :: Shift three times to remove "output" and the next two arguments
    shift
    shift
    shift        
    goto :start_parsing_args    
) else if "%1"=="build" (
    if "%2" neq "" ( 
        if "%2"=="debug_forward_pair" (            
            set cbow_debug_forward_pair="CBOW_DEBUG_PAIR"            
            shift
        )
    )
    shift
    goto :build
)  else if "%1"=="help" ( 
    set help_option="%1"
    shift
    goto :start_parsing_args
)  else if "%1"=="vc" (
   set error_message_text="`vc` option requires 1 additional parameters"  
   if "%2" == "" (
      goto :error_message
   )
   set validation_corpus_option="%1" "%2"
   shift
   shift
   goto :start_parsing_args
) else if "%1"=="ns" (
   set error_message_text="`nc` option requires 1 additional parameters"  
   if "%2" == "" (
      goto :error_message
   )
   set negative_samples_option="%1" "%2"
   shift
   shift
   goto :start_parsing_args
)

@ .\cow.exe corpus %corpus_filename_option% lr %learning_rate_option% epoch %epochs_option% rs %regularization_strength_option% %verbose_option% %input_option% %output_option% %output_w1_filename_option% %output_w2_filename_option% w1 %w1_filename_option% w2 %w2_filename_option% %help_option% %validation_corpus_option% %w2_transpose_option% %negative_samples_option%
goto :eof

:build
@  cl main.cpp /EHsc /Fecow.exe /D %cbow_debug_forward_pair%
goto :eof

:error_message
echo ERROR: %error_message_text%

:eof


 
 