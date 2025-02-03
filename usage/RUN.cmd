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

@echo off
setlocal enabledelayedexpansion

set verbose_option=
set w1_filename_option="./data/weights/w1p.dat"
set w2_filename_option="./data/weights/w2p.dat"
set corpus_filename_option="./data/NEW-INPUT.txt"
set epochs_option=1
set learning_rate_option=0.09
set regularization_strength_option=0.000000
set input_option=
set output_option=

:start_parsing_args

if "%1"=="verbose" (
    set verbose_option=verbose
    shift
    goto :start_parsing_args
) else if "%1"=="e" (
    if "%2" neq "" (    
        set epochs_option=%2        
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
        set learning_rate_option=%2
        shift
    )
    shift
    goto :start_parsing_args
) else if "%1"=="rs" (
    if "%2" neq "" (
        set regularization_strength_option=%2
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
    shift    
    goto :start_parsing_args    
) else if "%1"=="build" (
    goto :build
)  

@ .\cow.exe corpus %corpus_filename_option% lr %learning_rate_option% epoch %epochs_option% rs %regularization_strength_option% %verbose_option% %input_option% %output_option% w1 %w1_filename_option% w2 %w2_filename_option%

goto :eof

:build
@  cl main.cpp /EHsc /Fecow.exe

:eof


 
 