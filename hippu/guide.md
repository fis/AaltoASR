# aaltoasr-rec / aaltoasr-align User's Guide

<!---

**Testing the beta version:**

To set things up, type one of the following, depending on the
environment.  (If your shell is not bash, try typing `bash` first.)

**hippu.csc.fi (hippu3, hippu4):**

    source /v/users/htkallas/aaltoasr/init.sh

**Aalto/SPA systems:**

    source /akulabra/home/t40511/htkallas/aaltoasr/init.sh

**Aalto/ICS workstations:**

    source /share/puhe/htkallas/aaltoasr/init.sh

-->

## Introduction

`aaltoasr-rec` and `aaltoasr-align` are two command-line scripts for
using the [AaltoASR][aaltoasr] tools for simple speech recognition and
segmentation (forced-alignment) tasks.  `aaltoasr-adapt` provides
additional rudimentary speaker adaptation support.

To use the tools, load them in your path with `module load aaltoasr`.

[aaltoasr]: https://github.com/aalto-speech/AaltoASR "AaltoASR github page"

## Notes and disclaimers

The segmentation (and recognition) work best for input files of
moderate length; most preferrably, a single utterance.  It is possible
to use longer files, but the results may vary.  Unfortunately, there's
no support for automatically splitting a longer audio file with
`aaltoasr-align`, as there would be no way to do the corresponding
splitting on the input transcript.  For `aaltoasr-rec`, the `-s`
argument can be used to automatically split the file to segments of
desired size.

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

The speed/accuracy tradeoff of recognition can be controlled by
various parameters.  The default values are set to favour accuracy
over speed, so (depending on the input signal) recognition can easily
take up to 20 times as much time as the length of the input audio.

## Usage examples

Recognize the speech in the audio file `speech.wav`:

    aaltoasr-rec speech.wav

Recognize the speech of a long audio file `speech.wav`; speed up the
process by splitting the file into approximately 5-minute segments and
running the recognition on four cores in parallel:

    aaltoasr-rec -s 300 -n 4 speech.wav

Recognize the speech in `speech.wav`, but also generate all supported
levels of segmentation; write output in `speech.txt` and the
segmentations additionally in TextGrid format to `speech.textgrid`:

    aaltoasr-rec -m trans,segword,segmorph,segphone \
      -o speech.txt -T speech.textgrid `speech.wav`

Given the speech recording `speech.wav` and a corresponding plaintext
transcription in `speech.txt`, produce a TextGrid alignment in
`speech.textgrid` (along with a word-level alignment to standard
output):

    aaltoasr-align -T `speech.textgrid` -t `speech.txt` `speech.wav`

Recognize or align multiple files at once, on (up to) 4 cores:

    aaltoasr-rec -n 4 speech1.wav speech2.wav
    aaltoasr-align -n 4 -t speech1.txt -t speech2.txt speech1.wav speech2.wav

See the section "Adaptation" for examples of `aaltoasr-adapt` use.

## Option reference

The `aaltoasr-rec` and `aaltoasr-align` tools, for the most part,
share the same command line arguments.  Differences have been noted in
the descriptions of individual arguments.

The overall command lines have the form:

    aaltoasr-rec [options] input [input ...]
    aaltoasr-align [options] -t transcript [-t transcript ...] input [input ...]

The input file can be in any format accepted by the `sox` utility;
type `sox -h | grep "FILE FORMATS"` (after loading the `aaltoasr`
module) for a list.

The following options are available:

* **-t *file*, --trans *file*** (required for `aaltoasr-align`)  
Specifies that *file* contains a transcript of the contents of the
input audio file.  For `aaltoasr-rec`, it is used to compute
recognition error rates.  For `aaltoasr-align`, the sequence of
phonemes expected in the input file is taken directly from the
transcript.  If multiple input files are provided, the `-t` option (if
present) must also be repeated the corresponding number of times.

* **-a *file*, --adapt *file***  
Provide a speaker adaptation file (generated with `aaltoasr-adapt`).
For details, see the section titled "Adaptation", below.

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

* **-r, --raw** (`aaltoasr-rec` only)  
Normally, the recognize transcript is postprocessed to a more
human-readable format, by removing morph breaks and converting word
break tags to spaces.  Pass this flag to instead get the raw output of
the recognizer as-is.

* **-s *[S]*, --split *[S]*** (`aaltoasr-rec` only)  
Split the input audio to segments of approximately *S* (by default,
60) seconds.  The splitting is done using a heuristic that attempts to
select a silent period of the input file, but this is not guaranteed
to work.

* **-n *N*, --cores *N***  
Run the recognition process in parallel on up to *N* cores.  This
parallelization only works when there are multiple independent files
to process, and therefore only has an effect when used with multiple
input files or in conjunction with the `-s`/`--split` option.

* **-v, --verbose**  
Print output also from the recognition/alignment tools.

* **-q, --quiet**  
Do not print any status messages, only the final results.

* **--tempdir *dir***  
Use *dir* as the directory for temporary files.

* **-M *model*, --model *model***  
Select *model* as the acoustic model.  The default is `16k`.  The
following models are known:
    * `16k`: 16 kHz (slightly) multicondition SPEECON model.
    * `8k`: 8 kHz SPEECHDAT model.

* **-L *L*, --lmscale *L*** (`aaltoasr-rec` only)  
Use *L* as the scale factor for language model probabilities.
Tweaking this parameter can yield better recognition results of
e.g. noisy files.  The default value is 30.

* **--align-window *W*** (`aaltoasr-align` only)  
Set the length of the Viterbi alignment window, in frames.  The
default value is 1000.  For challenging material, a larger window may
be necessary to avoid discontinuities in the alignment.

* **--align-beam *B*** (`aaltoasr-align` only)
* **--align-sbeam *S*** (`aaltoasr-align` only)  
These two parameters control the log-probability and state beam width
in the Viterbi search, respectively.  The default value is 100 for
both.  If the beam size is too low to find a solution, both will be
automatically doubled and the search retried, but specifying a
suitable initial value is faster.

* **--noexp**  
Disable the automatic expansion of the transcript file.  By default,
the transcript will be preprocessed to expand numbers and some
abbreviations, as well as to transform some non-native Finnish letters
to a more phonetic form.

## Adaptation

The `aaltoasr` scripts support a limited form of acoustic model
adaptation, which can be helpful if the speaker (or recording
environment) differ much from what's expected.  To use the adaptation,
train a profile with `aaltoasr-adapt`, and then use that with
`aaltoasr-rec` or `aaltoasr-align` as follows:

    aaltoasr-adapt -o speaker.conf -t training.txt training.wav
    aaltoasr-rec -a speaker.conf test.wav

For a single test file with unknown contents, it is also possible to
do a two-pass recognition process:

    aaltoasr-adapt -o speaker.conf test.wav
    aaltoasr-rec -a speaker.conf test.wav

Similarly, it is possible to do a two-pass alignment as follows:

    aaltoasr-adapt -o speaker.conf -t test.txt test.wav
    aaltoasr-align -a speaker.conf -t test.txt test.wav

Multiple adaptation input files can be used in a manner consistent
with `aaltoasr-rec`/`aaltoasr-align`.  If a transcript is provided
with the `-t` parameter to `aaltoasr-adapt`, supervised adaptation
will be done; if the `-t` parameter is not used, the adaptation is
unsupervised.  Internally, for supervised adaptation the transcript
and audio will be aligned with `aaltoasr-align`, while for
unsupervised adaptation the audio will be processed with
`aaltoasr-rec` first.  Whether unsupervised adaptation is beneficial
at all depends somewhat on the quality of this initial recognition
step.

The `aaltoasr-adapt` script knows of the following options:

* **-o *file*, --output *file*** (required)  
Write the generated adaptation data to this file.

* **-t *file*, --trans *file***  
Provide a transcript of the audio file for supervised adaptation.  If
multiple inputs are specified, the `-t` option must be correspondingly
repeated.

* **-v, --verbose**  
Print output also from the recognition/alignment/adaptation tools.

* **-a *A B ...*, --args *A B ...***  
Pass the arguments *A B ...* as extra arguments to the underlying
invocation of `aaltoasr-align` (supervised) or `aaltoasr-rec`
(unsupervised).  The `-a` option must be the last thing on the command
line.  This option can be used to tune the alignment/recognition
parameters for the adaptation process.

## Technical details

The individual executables of the AaltoASR tools support a number of
features not covered by this guide.  The tools can be found in the
`bin` directory sibling to the `scripts` directory; type `which
aaltoasr-rec` to locate them.  The [Github page][aaltoasr] will
hopefully at some point contain detailed documentation of them.

The form of adaptation supported by the scripts is restricted to a
single feature domain CMLLR transformation.  The underlying toolset
also supports model-space MLLR (optionally with a regression tree for
multiple transformations per speaker), but training that is beyond the
scope of this manual.  However, any speaker configuration can (in
theory) be used with the scripts.  The speaker identifier used by the
scripts is `UNK`.

To use custom acoustic models, make a local copy of the (very short)
`aaltoasr-rec` (or `aaltoasr-align`) script, and have it modify the
`aaltoasr.models` list before initializing the `AaltoASR` object.
Absolute paths can be used.
