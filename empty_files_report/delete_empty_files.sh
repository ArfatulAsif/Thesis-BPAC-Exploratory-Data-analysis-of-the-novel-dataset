#!/usr/bin/env bash
# REVIEW BEFORE RUNNING. Removes empty/dead files that were in the scored set.
# (Non-conforming empties are listed in empty_files.csv but not removed here.)
set -e
rm -f "data\BPAC\Faridpur\speaker_female_03\sound_161.wav"
rm -f "data\BPAC\mymensingh\speaker_female_03\sound_154.wav"
rm -f "data\BPAC\mymensingh\speaker_female_03\sound_186.wav"
