# aaltoasr-rec / aaltoasr-align User's Guide

**TODO: testing phase instructions.**

## Introduction

`aaltoasr-rec` and `aaltoasr-align` are two command-line scripts for
using the [AaltoASR][aaltoasr] tools for simple speech recognition and
segmentation (forced-alignment) tasks.

**(TODO: not done yet.)**
To use the tools, load them in your path with `module load aaltoasr`.

[aaltoasr]: https://github.com/aalto-speech/AaltoASR "AaltoASR github page"

## Notes and disclaimers

The segmentation (and recognition) work best for input files of
moderate length; most preferrably, a single utterance.  It is possible
to use longer files, but the results may vary.  Unfortunately, there's
no support for automatically splitting a longer audio file, as there
would be no way to do the corresponding splitting on the input
transcript.

The acoustic models use cross-word triphone models (i.e., each phoneme
can have different models depending on the surrounding context),
trained using the conventional maximum-likelihood scheme.  As the
training does not (deliberately) focus on segmentation ability,
depending on the context, the phoneme boundaries could be very far
indeed from their "linguistically correct" positions.

> Possible future add-on project: discriminative training of a
monophone model with a segmentation-related criterion.

Similarly, the statistical morphemes (generated with the
[Morfessor][morfessor] method) are inspired by the MDL principle, and
make no pretense of being any sort of linguistic construct.

[morfessor]: https://github.com/aalto-speech/morfessor "morfessor-2.0 github page"

## Usage

The `aaltoasr-rec` and `aaltoasr-align` tools, for the most part,
share the same command line arguments.  Differences have been noted in
the descriptions of individual arguments.

The overall command lines have the form:

    aaltoasr-rec [options] input
    aaltoasr-align [options] -t transcript input

The input file can be in any format accepted by the `sox` utility;
type `sox -h | grep "FILE FORMATS"` (after loading the `aaltoasr`
module) for a list.

The following options are available:

* **-t *file*, --trans *file*** (required for `aaltoasr-align`)  
Specifies that *file* contains a transcript of the contents of the
input audio file.  For `aaltoasr-rec`, it is used to compute the
recognition error rates (**TODO**: missing feature).  For
`aaltoasr-align`, the sequence of phonemes expected in the input file
is taken directly from the transcript.

* **--noexp**  
Disable the automatic expansion of the transcript file.

* **-o *file*, --output *file***  
Output the recognition/segmentation results to *file*.  If not
provided, results are written to the standard output.

* **-m *mode*, --mode *mode***  
The *mode* parameter is a comma-separated list of terms corresponding
to what kind of outputs to generate.  The default settings are `trans`
and `segword` for `aaltoasr-rec` and `aaltoasr-align`, respectively.
The following terms are known:
    * `trans` (`aaltoasr-rec` only): transcript from the recognizer.
    * `segword`: segmentation (list of start and end times) at word level.
    * `segmorph` (`aaltoasr-rec` only): segmentation at the statistical
      morpheme (units of the recognizer's language model) level.
    * `segphone`: segmentation at the phoneme level.

* **-T *file*, --tg *file***  
In addition to the plaintext outputs, write all segmentation levels to
*file* in the Praat TextGrid format.

* **-f, --format** (`aaltoasr-rec` only)  
Postprocess the recognizer transcript to a slightly more
human-readable format, by removing morph breaks and changing word
break tags to spaces.

* **-M *model*, --model *model***  
Select *model* as the acoustic model.  Use `-M '?'` for a list.

* **-L *L*, --lmscale *L*** (`aaltoasr-rec` only)  
Use *L* as the scale factor for language model probabilities.
Tweaking this parameter can yield better recognition results of
e.g. noisy files.

* **--tempdir *dir***  
Use *dir* as the directory for temporary files.

* **-v, --verbose**  
Print output also from the recognition/alignment tools.

* **-q, --quiet**  
Do not print any status messages, only the final results.

## Technical details

Foo.