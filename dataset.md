# BPAC dataset layout

The layout every Part's `config.yml` expects on disk. Audio is **not** committed
to this repository — see [Getting the data](README.md#getting-the-data).

> **Anonymity.** Speaker directories carry no personal information. Participants
> consented to release of their **region, gender and age** only; names and
> contact details are withheld from every public artifact. See
> [Privacy and anonymization](README.md#privacy-and-anonymization).

## Directory structure

```
data/
├── BPAC/
│   ├── chattogram/
│   ├── dhaka/
│   ├── faridpur/
│   ├── khulna/
│   ├── mymensingh/
│   ├── rajshahi/
│   ├── rangpur/
│   └── sylhet/
└── best_diverse_sentences.csv
```

Each region directory holds one directory per speaker, and each speaker
directory holds that speaker's 250 recordings:

```
data/
├── BPAC/
│   ├── <region>/
│   │   ├── speaker_<gender>_<NN>/
│   │   │   ├── sound_001.wav
│   │   │   ├── sound_002.wav
│   │   │   ├── ...
│   │   │   └── sound_250.wav
│   │   ├── speaker_<gender>_<NN>/
│   │   └── ...
│   └── ...
└── best_diverse_sentences.csv
```

Concretely, for Chattogram:

```
data/BPAC/chattogram/
├── speaker_female_01/
│   ├── sound_001.wav
│   ├── sound_002.wav
│   └── ...
├── speaker_female_02/
├── speaker_female_03/
├── speaker_male_01/
├── speaker_male_02/
└── speaker_male_03/
```

## Speaker identity

`<gender>` is `male` or `female`; `<NN>` is a zero-padded sequential index.

**`<gender>_<NN>` is unique only *within* a region.** `speaker_female_01` exists
under `chattogram/` *and* under `dhaka/` — they are different people. The region
directory supplies the rest of the identity, so a globally unique speaker key is:

```
<region>_<gender>_<NN>        e.g.  sylhet_male_04
```

This is exactly the `speaker_uid` column in
[`verified_manifest.csv`](verified_manifest.csv), and the prefix of every
`utterance_id`:

```
<region>_<gender>_<NN>_<sentence_id>      e.g.  sylhet_male_04_137
```

## The parallel property

`sound_NNN.wav` is always that speaker's reading of **sentence `NNN`** of the
shared script. Because all 51 speakers read the same 250 sentences, any two
speakers' — or any two regions' — renditions of the same sentence can be
retrieved and compared directly. This is the property every analysis in this
repository depends on.

Sentence numbering is verified, not assumed: `Part_B/alignment_verification/`
CER-scores every recording against all 250 candidate sentences and corrects any
mismatch in `verified_manifest.csv`. Use that manifest as the source of truth
rather than re-deriving sentence IDs from filenames.

## The script

`best_diverse_sentences.csv` holds the 250-sentence script as `id,sentence`:

```csv
id,sentence
1,গত শুক্রবার তাঁরা প্রায় এক ঘণ্টা ধরে হোয়াইট হাউসে বৈঠক করেছেন।
2,ভারতের হয়ে একাই চার উইকেট শিকার করেছেন কুলদীপ যাদব।
3,শুরুতে শুধু ফোনে ভিডিও করতেন।
...
9,"টানা চার মাস, আগস্ট থেকে নভেম্বর পর্যন্ত চলে এ কর্মযজ্ঞ।"
10,ডান দিকে ঝাঁপিয়ে তার স্পটকিক রুখে দেন মেনিয়।
```

Sentences containing commas are quoted. Text is UTF-8, NFC-normalized before
any counting or scoring.
