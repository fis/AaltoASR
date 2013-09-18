# -*- coding: utf-8 -*-

import argparse
import os
import re
import shutil
import sys
import tempfile
import textwrap
from itertools import groupby
from os.path import basename, join
from subprocess import call, check_output, CalledProcessError, STDOUT

"""Aalto ASR tools for CSC Hippu environment.

This module contains the required glue code for using the Aalto ASR
tools for simple speech recognition and forced alignment tasks, in the
CSC Hippu environment.
"""

# Overall settings and paths to data files

rootdir = '/akulabra/home/t40511/htkallas/aaltoasr'
models = {
    '16k': { 'path': 'speecon_mfcc_gain3500_occ225_1.11.2007_20', 'srate': 16000, 'fstep': 128, 'default': 1 },
    '8k': { 'path': 'speechdat_mfcc_gain4000_occ350_13.2.2008_20', 'srate': 8000, 'fstep': 128 }
    }

def bin(prog):
    return join(rootdir, 'bin', prog)

default_lmscale = 30

# Command-line help

help = {
    'rec': {
        'desc': 'Recognize speech from an audio file.',
        'usage': '%(prog)s [options] input',
        'modes': set(('trans', 'segword', 'segmorph', 'segphone')),
        'defmode': 'trans',
        'extra': ('The MODE parameter specifies which results to include in the generated output.  '
                  'It has the form of a comma-separated list of the terms "trans", "segword", '
                  '"segmorph" and "segphone", denoting a transcript of the recognized text, '
                  'a word-level, statistical-morpheme-level or phoneme-level segmentation, '
                  'respectively. The listed items will be included in the plaintext output. '
                  'The default is "trans". For more details, see the User\'s Guide at: '
                  'XXX.')
        },
    'align': {
        'desc': 'Align a transcription to a speech audio file.',
        'usage': '%(prog)s [options] -t transcript input',
        'modes': set(('segword', 'segphone')),
        'defmode': 'segword',
        'extra': ('The MODE parameter specifies which results to include in the generated output. '
                  'It has the form of a comma-separated list of the terms "segword" and "segphone", '
                  'denoting a word-level or phoneme-level segmentation, respectively. The listed '
                  'items will be included in the plaintext output. The default is "segword".  '
                  'For more details, see the User\'s Guide at: '
                  'XXX.')
        }
    }

# Class for implementing the rec/align tools

class AaltoASR(object):
    def __init__(self, tool):
        """Initialize and parse command line attributes."""

        self.tool = tool

        # Ask argparse to grok the command line arguments

        thelp = help[tool]

        parser = argparse.ArgumentParser(description=thelp['desc'], usage=thelp['usage'], epilog=thelp['extra'])

        parser.add_argument('input', help='input audio file')
        parser.add_argument('-t', '--trans', help='provide an input transcript file', metavar='file',
                            required=True if tool == 'align' else False,
                            type=argparse.FileType('r'), default=None)
        parser.add_argument('-o', '--output', help='output results to file [default stdout]', metavar='file',
                            type=argparse.FileType('w'), default=sys.stdout)
        parser.add_argument('-m', '--mode', help='which kind of results to output (see below)', metavar='MODE',
                            default=thelp['defmode'])
        parser.add_argument('-T', '--tg', help='output also a Praat TextGrid segmentation to file', metavar='file',
                            type=argparse.FileType('w'), default=None)
        if tool == 'rec':
            parser.add_argument('-r', '--raw', help='produce raw recognizer output (with morph breaks)',
                                action='store_true')
        parser.add_argument('-M', '--model', help='acoustic model to use; "-M ?" for list', metavar='M')
        parser.add_argument('--noexp', help='disable input transcript expansion', action='store_true')
        if tool == 'rec':
            parser.add_argument('-L', '--lmscale', help='language model scale factor [default %d]' % default_lmscale, metavar='L',
                                type=int, default=default_lmscale)
        parser.add_argument('--tempdir', help='directory for temporary files', metavar='D')
        parser.add_argument('-v', '--verbose', help='print output also from invoked commands', action='store_true')
        parser.add_argument('-q', '--quiet', help='do not print status messages', action='store_true')

        self.args = parser.parse_args()

        # Check applicable arguments for validity

        if not os.access(self.args.input, os.R_OK):
            err('input file not readable: %s' % self.args.input, exit=2)

        self.mode = set()
        for word in self.args.mode.split(','):
            if word not in thelp['modes']:
                err('invalid output mode: %s' % word, exit=2)
            self.mode.add(word)

        self.tg = self.args.tg is not None

        if self.args.model is None:
            self.model = [m for m in models.values() if 'default' in m][0]
        elif self.args.model in models:
            self.model = models[self.args.model]
        else:
            err('unknown acoustic model: %s' % self.args.model)
            sys.stderr.write('supported models:\n')
            for m in sorted(models.keys()):
                sys.stderr.write('  %s [sample rate: %d Hz]\n' % (m, models[m]['srate']))
            sys.exit(2)
        self.mpath = join(rootdir, 'model', self.model['path'])


    def __enter__(self):
        """Make a working directory for a single execution."""

        self.workdir = tempfile.mkdtemp(prefix='aaltoasr', dir=self.args.tempdir)

        return self

    def __exit__(self, type, value, traceback):
        """Clean up the working directory and any temporary files."""

        if self.workdir.find('aaltoasr') >= 0: # sanity check
            shutil.rmtree(self.workdir)


    def convert_input(self):
        """Convert input audio to something suitable for the model."""

        self.log('converting input audio file to %d Hz mono' % self.model['srate'])

        audiofile = join(self.workdir, 'input.wav')

        if call([bin('sox'), self.args.input,
                 '-t', 'wav', '-r', str(self.model['srate']), '-b', '16', '-e', 'signed-integer', '-c', '1',
                 audiofile]) != 0:
            err("input conversion of '%s' with sox failed" % self.args.input, exit=1)

        self.audiofile = audiofile


    def align(self):
        """Do segmentation with the Viterbi alignment tool."""

        self.convert_input()

        self.log('computing Viterbi alignment')

        recipe = join(self.workdir, 'align.recipe')
        alignfile = join(self.workdir, 'align.phn')
        outfile = join(self.workdir, 'align.out')

        # Convert the transcription to a phoneme list

        phones = text2phn(self.args.trans, self.workdir, expand=not self.args.noexp)
        self.phones = phones

        # Write out the cross-word triphone transcript

        with open(alignfile, 'w', encoding='iso-8859-1') as f:
            for pnum, para in enumerate(phones):
                phns = para['phns']
                f.write('__\n')
                for phnum, ph in enumerate(phns):
                    if ph == '_':
                        f.write('_\n')
                    else:
                        prevph, nextph = '_', '_'
                        for prev in range(phnum-1, -1, -1):
                            if phns[prev] != '_': prevph = phns[prev]; break
                        for next in range(phnum+1, len(phns)):
                            if phns[next] != '_': nextph = phns[next]; break
                        f.write('%s-%s+%s\n' % (prevph, ph, nextph))
            f.write('__\n')

        # Make a recipe for the alignment

        with open(recipe, 'w') as f:
            f.write('audio=%s transcript=%s alignment=%s' % (self.audiofile, alignfile, outfile))

        # Run the Viterbi alignment

        cmd_out = sys.stderr if self.args.verbose else open('/dev/null', 'w')

        if call([bin('align'),
                 '-b', self.mpath, '-c', self.mpath+'.cfg',
                 #'--swins', '2000',
                 '-i', '1',
                 '-r', recipe], stdout=cmd_out, stderr=cmd_out) != 0:
            err('align failed', exit=1)

        self.alignment = outfile


    def phone_probs(self):
        """Create a LNA file for the provided audio."""

        self.convert_input()

        self.log('computing acoustic model likelihoods')

        recipe = join(self.workdir, 'input.recipe')
        lnafile = join(self.workdir, 'input.lna')

        # Construct an input recipe

        with open(recipe, 'w') as f:
            f.write('audio=%s lna=%s\n' % (self.audiofile, lnafile))

        # Run phone_probs on the file

        cmd_out = sys.stderr if self.args.verbose else open('/dev/null', 'w')

        if call([bin('phone_probs'),
                 '-b', self.mpath, '-C', self.mpath+'.gcl', '-c', self.mpath+'.cfg',
                 '--eval-ming', '0.2', '-i', '1',
                 '-r', recipe], stdout=cmd_out) != 0:
            err('phone_probs failed', exit=1)

        self.lna = lnafile


    def rec(self):
        """Run the recognizer for the generated LNA file."""

        self.log('recognizing speech')

        # Call rec.py on the lna file

        recipe = join(self.workdir, 'rec.recipe')
        with open(recipe, 'w') as f:
            f.write('lna=%s\n' % basename(self.lna))

        try:
            cmd = [bin('rec.py'),
                   rootdir, self.mpath, recipe, self.workdir, str(self.args.lmscale),
                   '1' if 'segphone' in self.mode or self.tg else '0']
            if 'segmorph' in self.mode or 'segword' in self.mode or self.tg:
                cmd.append(join(self.workdir, 'wordhist'))
            cmd_out = sys.stderr if self.args.verbose else open('/dev/null', 'w')
            rec_out = check_output(cmd, stderr=cmd_out)
            rec_out = rec_out.decode('iso-8859-1')
        except CalledProcessError as e:
            sys.stderr.write(e.output)
            err('rec.py failed %s %s' % (e.cmd, e.returncode), exit=1)

        # Parse the recognizer output to extract recognition result and state segmentation

        rec_trans = None
        rec_seg = []

        re_trans = re.compile(r'^REC: (.*)$')
        re_seg = re.compile(r'^(\d+) (\d+) (\d+)$')

        for line in rec_out.splitlines():
            m = re_trans.match(line)
            if m is not None:
                rec_trans = m.group(1)
            m = re_seg.match(line)
            if m is not None:
                rec_seg.append(tuple(int(val) for val in m.group(1, 2, 3)))

        if rec_trans is None:
            sys.stderr.write(rec_out)
            err('unable to find recognition transcript in output', exit=1)

        self.rec = rec_trans.strip()

        fstep = self.model['fstep']

        # If necessary, find labels for states and write an alignment file

        if 'segphone' in self.mode or self.tg:
            labels = get_labels(self.mpath + '.ph')

            alignment = join(self.workdir, 'rec.align')
            with open(alignment, 'w', encoding='iso-8859-1') as f:
                for start, end, state in rec_seg:
                    f.write('%d %d %s\n' % (start*fstep, end*fstep, labels[state]))

            self.alignment = alignment

        # If necessary, parse the generated word history file

        if 'segmorph' in self.mode or 'segword' in self.mode or self.tg:
            re_line = re.compile(r'^(\S+)\s+(\d+)')

            with open(join(self.workdir, 'wordhist'), 'r', encoding='iso-8859-1') as f:
                morphseg = []
                prev_end = 0

                for line in f:
                    m = re_line.match(line)
                    if m is None: continue # skip malformed
                    morph, end = m.group(1), int(m.group(2))
                    morphseg.append((prev_end*fstep, end*fstep, morph))
                    prev_end = end

            while len(morphseg) > 0 and (morphseg[0][2] == '<s>' or morphseg[0][2] == '<w>'):
                morphseg.pop(0)
            while len(morphseg) > 0 and (morphseg[-1][2] == '</s>' or morphseg[-1][2] == '<w>'):
                morphseg.pop()

            self.morphseg = morphseg


    def gen_output(self):
        """Construct the requested outputs."""

        # Start by parsing the state alignment into a phoneme segmentation

        if 'segphone' in self.mode or ('segword' in self.mode and self.tool == 'align') or self.tg:
            # Parse the raw state-level alignment

            rawseg = []

            re_line = re.compile(r'^(\d+) (\d+) ([^\.]+)\.(\d+)')
            with open(self.alignment, 'r', encoding='iso-8859-1') as f:
                for line in f:
                    m = re_line.match(line)
                    if m is None:
                        err('invalid alignment line: %s' % line, exit=1)
                    rawseg.append((int(m.group(1)), int(m.group(2)), m.group(3), int(m.group(4))))

            # Recover the phoneme level segments from the state level alignment file

            phseg = []

            cur_ph, cur_state = None, 0

            for start, end, rawph, state in rawseg:
                ph = trip2ph(rawph)

                if ph == cur_ph and state == cur_state + 1:
                    # Hack: try to fix cases where the first state of a phoneme after a long silence
                    # actually includes (most of) the silence too.  Ditto for last states.
                    if rawph.startswith('_-') and state == 1 and start - phseg[-1][0] > 2*(end-start):
                        phseg[-1][0] = start
                    if rawph.endswith('+_') and state == 2 and end - start > 2*(start - phseg[-1][0]):
                        continue # don't update end
                    phseg[-1][1] = end
                else:
                    phseg.append([start, end, ph])

                cur_ph, cur_state = ph, state

            # Split into list of utterances

            uttseg = []

            for issep, group in groupby(phseg, lambda item: item[2] == '__'):
                if not issep:
                    uttseg.append(list(group))

        # Merge phoneme segmentation to words in transcript if aligning

        if ('segword' in self.mode or self.tg) and self.tool == 'align':
            if len(self.phones) != len(uttseg):
                err('segmenter confused: utterance counts differ (%d != %d)' %
                    (len(self.phones['words']), len(uttseg)), exit=1)

            wordseg = []

            for uttidx, seg in enumerate(uttseg):
                wseg = []
                phstack = list(seg)
                for w in intersperse(self.phones[uttidx]['words'], '_'):
                    start, end = 0, 0
                    for phnum, ph in enumerate(w.replace('-', '_')):
                        if ph != phstack[0][2]:
                            err('segmenter confused: trans/align mismatch in utt %d word "%s"' % (uttidx+1, w), exit=1)
                        if phnum == 0: start = phstack[0][0]
                        end = phstack[0][1]
                        phstack.pop(0)
                    if w != '_': wseg.append((start, end, w))
                wordseg.append(wseg)

        # Merge morpheme segmentation into words if required

        if ('segword' in self.mode or self.tg) and self.tool == 'rec':
            wordseg = []

            for issep, group in groupby(self.morphseg, lambda item: item[2] == '<s>' or item[2] == '</s>'):
                if issep: continue

                utt = []

                for issep, wordgroup in groupby(group, lambda item: item[2] == '<w>'):
                    if issep: continue
                    word = list(wordgroup)
                    utt.append((word[0][0], word[-1][1], ''.join(m[2] for m in word)))

                wordseg.append(utt)

        # Generate requested plaintext outputs

        out = self.args.output
        srate = float(self.model['srate'])

        out_started = [False]
        def hdr(text):
            if out_started[0]: out.write('\n\n\n')
            out.write('### %s\n\n' % text)
            out_started[0] = True

        if 'trans' in self.mode:
            hdr('Recognizer transcript:')
            if not self.args.raw:
                for utt in self.rec.split('<s>'):
                    utt = utt.replace('</s>', '')
                    utt = utt.replace(' ', '')
                    utt = utt.replace('<w>', ' ').strip()
                    if len(utt) > 0:
                        out.write('%s\n' % utt)
            else:
                out.write('%s\n' % self.rec)

        if 'segword' in self.mode:
            hdr('Word-level segmentation:')
            for utt in intersperse(wordseg, ()):
                if len(utt) == 0: out.write('\n')
                else:
                    for start, end, w in utt:
                        out.write('%.3f %.3f %s\n' % (start/srate, end/srate, w))

        if 'segmorph' in self.mode:
            hdr('Morpheme[*]-level segmentation:   ([*] statistical)')
            for start, end, morph in self.morphseg:
                if morph == '<s>' or morph == '</s>': continue
                elif morph == '<w>': out.write('\n')
                else: out.write('%.3f %.3f %s\n' % (start/srate, end/srate, morph))

        if 'segphone' in self.mode:
            hdr('Phoneme-level segmentation:')
            for utt in intersperse(uttseg, ()):
                if len(utt) == 0: out.write('\n\n')
                else:
                    for start, end, ph in utt:
                        if ph == '_': out.write('\n')
                        else: out.write('%.3f %.3f %s\n' % (start/srate, end/srate, ph))

        # Generate Praat TextGrid output

        if self.tg:
            tgfile = self.args.tg

            tgfile.write(('''
File type = "ooTextFile"
Object class = "TextGrid"

xmin = %.3f
xmax = %.3f
tiers? <exists>
size = %d
item []:
''' % (min(uttseg[0][0][0], wordseg[0][0][0]) / srate,
       max(uttseg[-1][-1][1], wordseg[-1][-1][1]) / srate,
       4 if self.tool == 'rec' else 3)).lstrip())

            tierno = 1

            # Utterance tier

            tgfile.write('    item[%d]:\n' % tierno); tierno += 1
            tgfile.write('        class = "IntervalTier"\n')
            tgfile.write('        name = "utterance"\n')
            tgfile.write('        xmin = %.3f\n' % (wordseg[0][0][0] / srate))
            tgfile.write('        xmax = %.3f\n' % (wordseg[-1][-1][1] / srate))
            tgfile.write('        intervals: size = %d\n' % len(wordseg))

            for uttnum, utt in enumerate(wordseg):
                tgfile.write('        intervals [%d]:\n' % (uttnum+1))
                tgfile.write('            xmin = %.3f\n' % (utt[0][0] / srate))
                tgfile.write('            xmax = %.3f\n' % (utt[-1][1] / srate))
                tgfile.write('            text = "%s"\n' % ' '.join(w[2] for w in utt))

            # Word tier

            tgfile.write('    item[%d]:\n' % tierno); tierno += 1
            tgfile.write('        class = "IntervalTier"\n')
            tgfile.write('        name = "word"\n')
            tgfile.write('        xmin = %.3f\n' % (wordseg[0][0][0] / srate))
            tgfile.write('        xmax = %.3f\n' % (wordseg[-1][-1][1] / srate))
            tgfile.write('        intervals: size = %d\n' % sum(len(utt) for utt in wordseg))

            for wnum, word in enumerate(w for utt in wordseg for w in utt):
                tgfile.write('        intervals [%d]:\n' % (wnum+1))
                tgfile.write('            xmin = %.3f\n' % (word[0] / srate))
                tgfile.write('            xmax = %.3f\n' % (word[1] / srate))
                tgfile.write('            text = "%s"\n' % word[2])

            # Morph tier

            if self.tool == 'rec':
                fmorphseg = [morph for morph in self.morphseg if morph[2][0] != '<']

                tgfile.write('    item[%d]:\n' % tierno); tierno += 1
                tgfile.write('        class = "IntervalTier"\n')
                tgfile.write('        name = "morph"\n')
                tgfile.write('        xmin = %.3f\n' % (fmorphseg[0][0] / srate))
                tgfile.write('        xmax = %.3f\n' % (fmorphseg[-1][1] / srate))
                tgfile.write('        intervals: size = %d\n' % len(fmorphseg))

                for morphnum, morph in enumerate(fmorphseg):
                    tgfile.write('        intervals [%d]:\n' % (morphnum+1))
                    tgfile.write('            xmin = %.3f\n' % (morph[0] / srate))
                    tgfile.write('            xmax = %.3f\n' % (morph[1] / srate))
                    tgfile.write('            text = "%s"\n' % morph[2])

            # Phoneme tier

            fphseg = [ph for ph in phseg if ph[2][0] != '_']

            tgfile.write('    item[%d]:\n' % tierno); tierno += 1
            tgfile.write('        class = "IntervalTier"\n')
            tgfile.write('        name = "phone"\n')
            tgfile.write('        xmin = %.3f\n' % (fphseg[0][0] / srate))
            tgfile.write('        xmax = %.3f\n' % (fphseg[-1][1] / srate))
            tgfile.write('        intervals: size = %d\n' % len(fphseg))

            for phnum, ph in enumerate(fphseg):
                tgfile.write('        intervals [%d]:\n' % (phnum+1))
                tgfile.write('            xmin = %.3f\n' % (ph[0] / srate))
                tgfile.write('            xmax = %.3f\n' % (ph[1] / srate))
                tgfile.write('            text = "%s"\n' % ph[2])


    def log(self, msg):
        if not self.args.quiet:
            sys.stderr.write('%s: %s\n' % (basename(sys.argv[0]), msg))

# "Free-format" transcript to phoneme list conversion

def text2phn(input, workdir, expand=True):
    """Convert a transcription to a list of phonemes."""

    # Read in, split to trimmed paragraphs

    try: input = input.read()
    except: pass

    if not type(input) is str:
        try: input = input.decode('utf-8')
        except UnicodeError:
            input = input.decode('iso-8859-1')

    input = filter(None, (re.sub(r'\s+', ' ', para.strip()) for para in input.split('\n\n')))

    # Attempt to expand abbreviations etc., if requested

    if expand:
        expander = join(rootdir, 'lavennin', 'bin', 'lavennin')
        exp_in = join(workdir, 'expander_in.txt')
        exp_out = join(workdir, 'expander_out.txt')

        with open(exp_in, 'w', encoding='iso-8859-1') as f:
            f.write('\n'.join(input) + '\n')

        if call([expander, workdir, exp_in, exp_out]) != 0:
            err('transcript expansion script failed', exit=1)

        with open(exp_out, 'r', encoding='iso-8859-1') as f:
            input = filter(None, (para.strip() for para in f.readlines()))

    # Go from utterances to list of words

    input = [re.sub(r'\s+', ' ',
                    re.sub('[^a-zåäö -]', '', para.lower())
                    ).strip().split()
             for para in input]

    # Add phoneme lists

    return [{ 'words': para, 'phns': list('_'.join(para).replace('-', '_')) } for para in input]

# Miscellaneous helpers

def err(msg, exit=-1):
    sys.stderr.write('%s: error: %s\n' % (basename(sys.argv[0]), msg))
    if exit >= 0: sys.exit(exit)

def intersperse(iterable, delim):
    """Haskell intersperse: return elements of iterable interspersed with delim."""

    it = iter(iterable)
    yield next(it)
    for x in it:
        yield delim
        yield x

def trip2ph(triphone):
    if len(triphone) == 5: return triphone[2]
    elif triphone == '__': return '__'
    elif triphone[0] == '_': return '_'

    err('unknown triphone: %s' % triphone, exit=1)

def get_labels(phfile):
    re_index = re.compile(r'^\d+ (\d+) (.*)')

    labels = {}

    with open(phfile, 'r', encoding='iso-8859-1') as ph:
        if ph.readline() != 'PHONE\n': err('bad phoneme file: wrong header', exit=1)

        phcount = int(ph.readline())

        while True:
            line = ph.readline().rstrip()
            if not line: break
            line = line

            m = re_index.match(line)
            if m is None:
                err('bad phoneme file: wrong index line: %s' % line, exit=1)

            phnstates = int(m.group(1))
            phname = m.group(2)

            line = ph.readline().strip()
            phstates = [int(s) for s in line.split()]
            if len(phstates) != phnstates:
                err('bad phoneme file: wrong number of states', exit=1)

            s = 0
            for idx in phstates:
                ph.readline() # skip the transition probs
                if idx < 0: continue # dummy states don't need labels
                labels[idx] = '%s.%d' % (phname, s)
                s += 1

    return labels
