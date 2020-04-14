import os
import sys
import pprint as pp # debug
import time # timestamp
import multiprocessing as mp # To lookup info online faster
import argparse

# Lookup word information
from vocabulary.vocabulary import Vocabulary # online dictionary
import nltk # local dictionary
import wikipediaapi # for more advance definitions
import urllib # google search
from bs4 import BeautifulSoup # google search
import requests # google search
import webbrowser # google search

# Capture source image
import cv2 # for webcam usage
from Foundation import * # For osascript crap (applescript)

# Google Vision OCR
from google.cloud import vision
from google.cloud.vision import types
import io

# sample questions
import json

os.environ['NO_PROXY'] = '*'
VERSION = "2018.05.24"

# text colors
class bcolors:
	HEADER = '\033[95m'
	OKBLUE = '\033[94m'
	OKGREEN = '\033[92m'
	WARNING = '\033[93m'
	FAIL = '\033[91m'
	ENDC = '\033[0m'
	BOLD = '\033[1m'
	UNDERLINE = '\033[4m'


# sample questions from previous games
sample_questions = {}

# load sample questions
def load_json():
	global sample_questions
	sample_questions = json.loads(open("Data/questions.json").read())

#Class
class HQTrivia():
    #initialization
    def __init__(self):
        # Use input image or quicktime(live)
        self.use_quicktime = False
        self.use_input = False

        # the filename of the image (no extension = capturing image)
        self.picture = 'source'
        # location of where to work on self.picture
        self.location = os.getcwd()

        # Replace with your own auth file name
        self.google_auth_json = 'HQproject-a1a4e25e4b45.json'

        # wikipedia setting (english)
        self.wiki = wikipediaapi.Wikipedia('en')
        self.vb = Vocabulary()

        # The OCR text (directly converted from image)
        self.raw = ''
        # processed texts
        self.question = ''
        self.question_nouns = ''
        self.answers = {}
        self.lookup_info = {}

        # For debugging
        self.times = {}
        self.verbose = False

    def debug(self, msg):
    # in multiprocessing environments, following line helps
        sys.stdout.flush()
        #print(os.path.basename(__file__)+ ": "+ str(msg))
        print(str(msg))

    def capture(self, ftype='png'):
    # Are you using input or QuickTime
        pre = "capture() | "
        if self.verbose:
            self.debug(pre + "choosing how to capture...")
            if self.use_input:
                self.debug(pre + "using user input image")
            return

        # add extension name 'png' (default)
        #self.picture += '.' + ftype
        if self.use_quicktime:
            if self.verbose:
                self.debug(pre + "using quicktime")
            self.scan_quicktime(ftype)

    def scan_quicktime(self, ftype='png'):
    # function to take screenshot via AppleScript (wire connection to computer)
    # To do: 1. open QuickTime player and do a movie recording
    #        2. Select drop down arrow next to record button, select device
    # Steps: 1. Get Window ID of QuickTime Player
    #        2. Run shell script to screen-capture the window ID
        pre = "scan_quicktime() | "
        if self.verbose:
            self.debug(pre + "starting QuickTime")
            start = time.time()

        # save captured QuickTime image to this location
        full_path = os.path.join(self.location, self.picture)
        script = """tell application "QuickTime Player"
set winID to id of window 1
end tell
do shell script "screencapture -x -t png -l " & winID &"""
        script += ' " ' + full_path + '"'
        # replace 'png' with ftype
        script = script.replace('png', ftype)

        # Take screenshot
        s = NSAppleScript.alloc().initWithSource_(script)
        s.executeAndReturnError_(None)

        if self.verbose:
            diff = time.time() - start
            self.debug(pre + "Quicktime - elapsed {!s}".format(diff))
            self.times['scan_quicktime()'] = diff

    def ocr_vision(self, queue):
        # Use Google Cloud Vision API to process OCR
        pre = "ocr_vision() | "
        if self.verbose:
            start = time.time()
            self.debug(pre + "submitting job")

        # Authenticate using Google Cloud Vision API json file
        try:
            file_path = os.path.join(self.location, self.google_auth_json)
            if not os.path.isfile(file_path):
                if self.verbose:
                    self.debug(pre + "no auth file found!")
                queue.put("END")
                return
        except:
            if self.verbose:
                self.debug(pre + "no auth file found!")
            queue.put("END")
            return

        # Google Cloud Vision API credential
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = file_path

        # Instantiates a client
        client = vision.ImageAnnotatorClient()

        # get the image file (full path)
        if not os.path.isfile(self.picture):
            full_path = os.path.join(self.location, self.picture)
        else:
            full_path = self.picture

        # loads the image into memory
        with io.open(full_path, 'rb') as image_file:
            content = image_file.read()

        image = types.Image(content=content)

        # text detection on the image
        response = client.text_detection(image=image)
        text = response.text_annotations

        for t in text:
            self.raw = t.description
            break

        # making raw to array (split by newline)
        self.raw = self.raw.split('\n')
        if self.verbose:
            self.debug(pre + "raw texts - " + str(self.raw))

        # clean up the text
        i = 0
        pop_i_list = []
        cleantext = []
        detect_count = False
        while i < len(self.raw):
            value = self.raw[i].lower()
            if value == '' or detect_count == False:
                if(value in ['0','1','2','3','4','5','6','7','8','9','10']):
                    detect_count = True
                #self.debug('[delete] - '+str(i)+ " "+ value)
                pop_i_list.append(i)
            else:
                cleantext.append(value)
            i+=1

        # delete "swipe left comment" and return data to parent process
        self.raw = cleantext
        #self.raw.pop(-1)
        queue.put(self.raw)

        if self.verbose:
            self.debug(pre + "cleaned texts - " + str(self.raw))
            diff = time.time() - start
            self.debug(pre + "elapsed {!s}".format(diff))
        queue.put("END")

    def parse(self):
    # Parse the raw OCR text to find Q&A
        pre = "parse() | "
        if self.verbose:
            self.debug(pre + "starting")
            start = time.time()

        # Save it to question and answer variable
        check_q = True
        count_answer = 1
        for line in self.raw:
            # check for question mark in the question
            if check_q:
                if len(line) > 2:
                    if '?' not in line:
                        self.question += line + ' '
                    else:
                        self.question += line
                        check_q = False
            else:
                if 'Swipe left' not in line:
                    if len(line) > 0 and line != '-':
                        ans = line
                        self.answers[ans] = {
                            "answer": ans,
                            "index": str(count_answer),
                            "keywords": [],
                            "score": 0
                        }
                        self.lookup_info[ans] = []
                        count_answer += 1
                else:
                    break

        # checking parsed results
        if '?' not in self.question:
            self.debug(pre + "Could not find question!")
            raise
        if len(self.answers) < 1:
            self.debug(pre + "Could not find answers!")
            raise
        elif len(self.answers) > 3:
            self.debug(pre + "Found more than three answers!")
            raise

        # Use local dictionary (nltk) to find nouns
        for q in nltk.pos_tag(nltk.word_tokenize(self.question)):
            if q[1] == 'NN' or q[1] == 'NNP' or q[1]=='NNS' or q[1]=='JJ' or q[1]=='JJS':
                self.question_nouns += " " + q[0]
            #else:
            #    print(q[0]+ q[1])
        self.question_nouns = self.question_nouns.strip().split(' ')

        if self.verbose:
            self.debug(pre + "question = " + str(self.question))
            self.debug(
                pre + "nouns in question - {!s}".format(self.question_nouns))
            self.debug(pre + "answer = " + str(self.answers))
            diff = time.time() - start
            self.debug(pre + "elapsed {!s}".format(diff))
            self.times["parse"] = diff

    def keywords(self, words):
    # Function to find words in a string that are also in question
    # and return keywords found
        keywords = []
        for w in words:
            if len(w) > 2:
                if w in self.question_nouns:
                    if w not in keywords:
                        keywords.append(w)

        return keywords

    def lookup_wiki(self, queue):
    # Get wiki info about answer
    # Needs to return results to parent (for multi-processing)
        pre = "lookup_wiki() | "

        if self.verbose:
            self.debug(pre + "starting")
            start = time.time()

        # search in wikipedia for each answer
        for index, ans in self.answers.items():
            l_info = self.lookup_info[ans['answer']]

            try:
                page = self.wiki.page(ans['answer'])
                if page.exists():
                    try:
                        words = []
                        for i in page.sections:
                            words += i.text.split(' ')
                    except:
                        self.debug(
                            pre + "issue with wikipedia for {!s}"
                            .format(ans['answer']))
                    else:
                        l_info.append("[Wikipedia]: " + page.summary)
                        queue.put([ans['answer'], self.keywords(words), l_info, 0])

                else:
                    a = ans['answer'].split(' ')
                    if len(a) < 2:

                        # Could not find page, so throw exception and move on
                        self.debug(
                            pre + "no results for {!s} in wikipedia... ".
                            format(ans['asnwer']))
                        raise

                    else:
                        # Try searching each word in answer as last resort
                        for w in a:
                            if len(w) > 3:
                                page = self.wiki.page(w)
                                if page.exists():
                                    try:
                                        words = []
                                        for i in page.sections:
                                            words += i.text.split(' ')
                                    except:
                                        self.debug(
                                            pre +
                                            "issue with wikipedia for {!s}"
                                            .format(ans['answer']))
                                    else:
                                        l_info.append(
                                            "[Wikipedia {!s}]: ".format(w) +
                                            page.summary)
                                        queue.put([
                                            ans['answer'],
                                            self.keywords(words),
                                            l_info,
                                            0])


            except:
                self.debug(
                    pre + "issue with wikipedia for {!s}... "
                    .format(ans['answer']))
                self.debug(sys.exc_info())

        queue.put("END")
        if self.verbose:
            self.debug(pre + "elapsed " + str(time.time() - start))

    def lookup_dict_and_syn(self, queue):
    # Use nltk to look up word info(synonym). Use online dictionary if fails.
        pre = "lookup_dict_and_syn() | "
        if self.verbose:
            self.debug(pre + "starting")
            start = time.time()

        # Get dictionary/synonyms
        for index, ans in self.answers.items():
            l_info = self.lookup_info[ans['answer']]
            a = ans['answer'].split(' ') # incase of multi word answers

            for w in a:
                # don't waste time on looking for smaller words
                if len(w) > 3:
                    # definition
                    define = nltk.corpus.wordnet.synsets(w)
                    synset_found = False
                    if len(define) < 1:
                        # local dictionary didn't find anything so search online
                        if self.verbose:
                            self.debug(
                                pre + "nltk none for {!s}, using vocabulary"
                                .format(w))
                        try:
                            define = self.vb.meaning(w, format='list')
                            if define != False:
                                # Multiple definitions possible
                                for d in define:
                                    l_info.append(
                                        "[Meaning {!s}]: ".format(w) + d)
                                    queue.put([
                                        ans['answer'],
                                        self.keywords(d),
                                        l_info,
                                        0])
                        except:
                            self.debug(
                                pre + "issue with vocabulary for {!s}... "
                                .format(w))
                            self.debug(sys.exc_info())
                    else:
                        synset_found = True
                        l_info.append(
                            "[Meaning {!s}]: ".format(w) +
                            define[0].definition())
                        queue.put([
                            ans['answer'],
                            self.keywords(define[0].definition()),
                            l_info, 0])

                    # Synonyms
                    if synset_found:
                        synonyms = [l.name() for s in define for l in s.lemmas()]

                        # Remove duplicates nltk adds
                        s = []
                        i = 0
                        while i < len(synonyms):
                            if synonyms[i] in s:
                                synonyms.pop(i)
                            else:
                                s.append(synonyms[i])
                                i += 1
                        syn = ', '.join(s)
                        l_info.append("[Synonyms {!s}]: ".format(w) + syn)
                        queue.put([ans['answer'], self.keywords(syn), l_info, 0])
                    else:
                        # Local dictionary didn't find anything so search online
                        self.debug(
                            pre + "nltk has nothing for {!s}, using vocabulary"
                            .format(w))
                        try:
                            synonyms = self.vb.synonym(w, format='list')
                            if synonyms != False:
                                l_info.append(
                                    "[Synonyms {!s}]: ".format(w) +
                                    str(synonyms))
                                queue.put([
                                    ans['answer'],
                                    self.keywords(str(synonyms)),
                                    l_info, 0])
                        except:
                            self.debug(
                                pre + "issue with vocabulary for {!s}... "
                                .format(w))
                            self.debug(sys.exc_info())

        queue.put("END")
        if self.verbose:
            self.debug(
                pre + "elapsed " + str(time.time() - start))


    def lookup_google_search(self, queue):
    # Do google search for each answer
    # Find if words in results are found in the question
        pre = "lookup_google_search() | "

        if self.verbose:
            self.debug(pre + "starting")
            start = time.time()

        # Google Search
        for index, ans in self.answers.items():
            l_info = self.lookup_info[ans['answer']]
            try:
                search = '"'+ans['answer']+'" + '+ self.question
                r = requests.get("https://www.google.com/search", params={'q':search})

                soup = BeautifulSoup(r.text, "lxml")
                res = soup.find("div", {"id": "resultStats"})
                count = int(res.text.replace(",", "").split()[1])

                #find_all() - mnethod to look through a tag's descendent (class in CSS)
                #for g in soup.find_all(class_='st'):
                #    results += " " + g.text
                #remove new line
                cleaned_results = search
                l_info.append("[Google]: finished searching")
                queue.put([
                    ans['answer'],
                    self.keywords(''),
                    l_info,
                    count])
            except:
                self.debug(
                    "issue with google search for {!s}... "
                    .format(ans['answer']))
                self.debug(sys.exc_info())

        queue.put("END")
        if self.verbose:
            self.debug(
                pre + "google search elapsed " + str(time.time() - start))

    def display(self):
        # Clear the screen
        os.system('cls' if os.name == 'nt' else 'clear')

        # Text to output to screen
        output = []

        # Question
        output.append(bcolors.HEADER + '\n\nQuestion - ' + self.question + '\n' + bcolors.ENDC)

        # Answers & Lookup Info
        # choice to track answer with the highest score
        choice = {'index': [], 'score': 0, 'l_info': []}
        # a is the key and ans is the value; items() is for dict datastructure
        i=1
        for a, ans in self.answers.items():
            if ans['score'] == choice['score']:
                choice['index'].append(a)
            if 'NOT' in self.question:
                if ans['score'] < choice['score']:
                    choice['index'] = [a]
                    choice['score'] = ans['score']
            else:
                if ans['score'] > choice['score']:
                    choice['index'] = [a]
                    choice['score'] = ans['score']

            temp = bcolors.OKBLUE + "Choice " + str(i) + ": "+ ans['answer'] +' - Score ' + str(ans['score']) + bcolors.ENDC
            output.append(temp)

            for l_info in self.lookup_info[ans['answer']]:
                for l in l_info:
                    l_index = l.split(':')[0]
                    if l_index not in choice['l_info']:
                        choice['l_info'].append(l_index)
                        if len(l) > 140:
                            output.append(l[0:140])
                        else:
                            output.append(l)
            output.append("[Keywords]: " + str(ans['keywords']))
            output.append("")

            i+=1

        # Highest scoring answer
        if len(choice['index']) > 0:
            choose = []
            for i in choice['index']:
                choose.append(self.answers[i]['answer'])
            msg = "Best Answer - " + bcolors.UNDERLINE + ', '.join(choose)

            # If negative word, choose the lowest score
            if 'NOT' in self.question:
                msg += (" - NOT keyword so lowest score is " +
                        str(choice['score'])) + bcolors.ENDC
            else:
                msg += (" - highest score is " + str(choice['score'])) + bcolors.ENDC
            output.append(msg)
        else:
            output.append("Answer - Unknown")
        output.append("")
        output.insert(1, '\n')

        # print it all
        for line in output:
            print(line)

if __name__ == '__main__':
    start = time.time()

    # Setup command line options
    parser = argparse.ArgumentParser(
        description='Automate searching for answers in HQ Trivia')
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        '-q', '--quicktime',
        action='store_true', default=False,
        help="Use quicktime to capture source image"
    )
    group.add_argument(
        '-i', '--input',
        action='store',
        help="Use image provided instead of capturing"
    )
    group.add_argument(
        '-e', '--example',
        action='store_true', default=False,
        help="Use example questions"
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true', default=False,
        help="Print debug information"
    )
    parser.add_argument(
        '-V', '--version',
        action='store_true', default=False,
        help="Version of script"
    )
    options = parser.parse_args()

    # Configure class with command option
    hq = HQTrivia()
    hq.verbose = options.verbose
    if options.verbose:
        hq.verbose = options.verbose
    if options.version:
        hq.debug("version - " + VERSION)

    # Select Quicktime, Input Image, or Sample Question Option
    if options.quicktime:
        hq.use_quicktime = options.quicktime
        hq.use_input = False
    elif options.input:
        if len(options.input) > 0:
            hq.use_quicktime = False
            hq.use_input = True
            hq.picture = options.input
        else:
            exit()
    elif options.example:
        load_json()
        for key in sample_questions:
            hq.question = key
            i=1
            for ans in sample_questions[key]:
                hq.answers[ans] = {
                    "answer": ans,
                    "index": i,
                    "keywords": [],
                    "score": 0
                }
                hq.lookup_info[ans] = []
                i+=1
            break
    else:
        exit()

    if options.quicktime or options.input:
        # Capture image first
        hq.capture()

        # Read the picture (use multiprocessing for multiple OCR readers)
        updated = {'vision': 0}
        vision_raw = ''
        q_vision = mp.Queue()
        p_vision = mp.Process(target=hq.ocr_vision, args=(q_vision,))
        p_vision.daemon = True
        start_ocr = time.time()
        p_vision.start()
        while True:
            if not q_vision.empty():
                data = q_vision.get()
                if data != "END":
                    vision_raw = data
                    updated['vision'] = 1
                    hq.times['ocr_vision'] = time.time() - start_ocr
                else:
                    hq.times['ocr_vision'] = time.time() - start_ocr
                    updated['vision'] = 2

            # Make sure it doesn't take too long
            if ((int(time.time() - start_ocr) > 10) or
                (updated['vision'] > 0) or (updated['vision'] == 1)):
                break

        # Choose vision text over tesseract since it's better
        #print(vision_raw)
        if len(vision_raw) > 0:
            hq.raw = vision_raw
            hq.debug("Using Google Vision OCR")
        else:
            hq.debug("COULD NOT FIND TEXT!")
            exit()
        diff = time.time() - start_ocr
        hq.debug("OCR | elapsed {!s}".format(diff))
        hq.times["ocr"] = diff

        # Parse the picture text
        hq.parse()
        #exit()
    # Get information about answers (time consuming so do multiprocessing)
    q_wiki = mp.Queue()
    q_dict = mp.Queue()
    q_gsearch = mp.Queue()
    # Queue returns list of [answer_text, keyword_list, lookup_info]
    p_wiki = mp.Process(target=hq.lookup_wiki, args=(q_wiki,))
    p_dict = mp.Process(target=hq.lookup_dict_and_syn, args=(q_dict,))
    p_gsearch = mp.Process(target=hq.lookup_google_search, args=(q_gsearch,))
    p_wiki.daemon = True
    p_dict.daemon = True
    p_gsearch.deamon = True
    start_lookup = time.time()
    p_wiki.start()
    p_dict.start()
    p_gsearch.start()

    while True:
        # Thread counts finished
        count_cur = 0
        count_max = 3

        def update_display(data):
            ans = data[0]
            keys = data[1]
            l_info = data[2]
            score = data[3]

            for k in keys:
                if k not in hq.answers[ans]['keywords']:
                    hq.answers[ans]['keywords'].append(k)
            hq.answers[ans]['score'] += score
            hq.lookup_info[ans].append(l_info)
            hq.display()

        if not q_wiki.empty():
            data = q_wiki.get()
            if data != "END":
                update_display(data)
            else:
                hq.times['lookup_wiki'] = time.time() - start_lookup
                count_cur += 1

        if not q_gsearch.empty():
            data = q_gsearch.get()
            if data != "END":
                update_display(data)
            else:
                hq.times['lookup_google_search'] = time.time() - start_lookup
                count_cur += 1

        if not q_dict.empty():
            data = q_dict.get()
            if data != "END":
                update_display(data)
            else:
                hq.times['lookup_dict_and_syn'] = time.time() - start_lookup
                count_cur += 1

        # Make sure it doesn't take too long
        if int(time.time() - start_lookup) > 10 or count_cur == count_max:
            break

    # Display the final results!
    diff = time.time() - start_lookup
    hq.debug("Lookups elapsed {!s}".format(diff))
    hq.times['lookups'] = diff
    hq.display()

    # Show total times for everything
    diff = time.time() - start
    hq.times['total'] = diff
    if hq.verbose:
        hq.debug("Time")
        pp.pprint(hq.times)
